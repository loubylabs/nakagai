"""Event-driven bar-replay backtester. Share-only, long AND short, cash-settlement aware.

Pillar 4 (Replay) of the platform's docs/internal/PILLARS.md. Its contract: given
the same bars and the same spec this produces the same trades, and those trades
are what a real desk would plausibly have got. Every number in the platform's
evidence store is downstream of this file, so a change to fill arithmetic
invalidates stored evidence and calls for a re-prove, not just a green suite.

Bar loop order, which is load-bearing: exits before entries, gap before intrabar,
context built at bar close, entries filled at the NEXT bar's open. Execution
costs live in costs.py. The platform's tests/waterfall/test_stage4_replay.py is
the end-to-end statement of this pillar's contract; tests/test_engine_fills.py
is the unit-level one.
"""

import math
from dataclasses import dataclass

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import PreloadedBars, build_context, visible_counts
from nakagai.engine.costs import FeeModel, SlippageModel
from nakagai.engine.portfolio import SettledLedger
from nakagai.strategies.base import Direction, PositionAction, Signal, Strategy


@dataclass
class Trade:
    symbol: str
    direction: Direction
    qty: int
    entry_ts: pd.Timestamp
    entry: float
    exit_ts: pd.Timestamp
    exit: float
    stop: float
    target: float
    pnl: float
    r_multiple: float
    setup_tags: tuple[str, ...]
    exit_reason: str
    # Round-trip commissions and fees, already deducted from `pnl`. Zero for
    # the broker this platform trades through today; carried per trade anyway
    # so a fee change is re-provable rather than archaeological.
    fees: float = 0.0


@dataclass
class _Position:
    signal: Signal
    qty: int
    entry_ts: pd.Timestamp
    entry: float
    reserved: float
    stop: float = 0.0     # live levels; strategies may ratchet via manage()
    target: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    rejected_unsettled: int
    starting_equity: float


class Engine:
    def __init__(self, strategy: Strategy, cache: BarCache, symbol: str,
                 start: pd.Timestamp, end: pd.Timestamp,
                 equity0: float = 10_000.0, risk_pct: float = 0.01,
                 slippage: SlippageModel | None = None,
                 fees: FeeModel | None = None,
                 tfs: TimeframeSet = DEFAULT_TIMEFRAMES):
        self.strategy, self.cache, self.symbol = strategy, cache, symbol
        self.start, self.end = start, end
        self.equity0, self.risk_pct = equity0, risk_pct
        self.slippage = slippage if slippage is not None else SlippageModel()
        self.fees = fees if fees is not None else FeeModel()
        self.tfs = tfs

    def slippage_for(self, price: float) -> float:
        """Per-share slippage at this price. Exposed so callers and tests can
        ask the engine what it will charge without reaching into the model."""
        return self.slippage.per_share(price)

    def run(self) -> BacktestResult:
        view = PreloadedBars(self.cache, self.symbol, self.tfs)
        bars = view.load(self.symbol, self.tfs.driving)
        bars = bars[(bars.index >= self.start) & (bars.index < self.end)]
        # End-anchored primitives (fvg_nearest, order_block) are evaluated row
        # by row, so bound EVERY timeframe to the rows this replay actually
        # visits rather than the whole frame. The driving timeframe is not a
        # special case: a spec on 1h reads 1h rows through the 1h span, and
        # leaving that one unbounded would walk the entire 1h history per
        # replay. visible_counts maps the window onto each frame's own index,
        # so the span is exactly the rows the cursor can land on.
        if len(bars):
            closes = bars.index + self.tfs.step
            for tf in self.tfs.all:
                counts = visible_counts(view.load(self.symbol, tf).index,
                                        closes, tf, self.tfs)
                view.fe.set_span(tf, int(counts.min()) - 1, int(counts.max()))
        ledger = SettledLedger(self.equity0)
        trades: list[Trade] = []
        rejected = 0
        position: _Position | None = None
        pending: Signal | None = None
        curve: dict[pd.Timestamp, float] = {}

        for bar in bars.itertuples():   # itertuples: iterrows builds a Series per bar
            ts = bar.Index
            now = ts + self.tfs.step  # bar close time

            # 1) exits on this bar (stop first when both are in range)
            if position is not None:
                exit_price, reason = self._check_exit(position, bar)
                if exit_price is not None:
                    trades.append(self._close(position, now, exit_price, reason, ledger))
                    position = None

            # 2) pending entry fills at this bar's open
            if pending is not None:
                if position is None:
                    position, ok = self._try_fill(pending, ts, float(bar.open), ledger, now)
                    if not ok:
                        rejected += 1
                pending = None

            # 3) manage + 4) new signals (point-in-time context as of bar close)
            ctx = build_context(view, self.symbol, now, tfs=self.tfs)
            if position is not None:
                if self.strategy.manage(position, ctx) == PositionAction.EXIT:
                    trades.append(self._close(position, now, float(bar.close), "manage", ledger))
                    position = None
            if position is None and pending is None:
                signals = self.strategy.on_bar(ctx)
                if signals:
                    pending = signals[0]

            curve[now] = self._mark(ledger, position, float(bar.close), now)

        if position is not None and len(bars):
            last = bars.iloc[-1]
            now = bars.index[-1] + self.tfs.step
            trades.append(self._close(position, now, float(last["close"]), "eod_window", ledger))
            curve[now] = self._mark(ledger, None, float(last["close"]), now)

        return BacktestResult(trades, pd.Series(curve, dtype="float64"), rejected, self.equity0)

    def _try_fill(self, sig: Signal, ts, open_price: float, ledger: SettledLedger, now) -> tuple["_Position | None", bool]:
        fill = (self.slippage.buy(open_price) if sig.direction == Direction.LONG
                else self.slippage.sell(open_price))
        risk_per_share = abs(fill - sig.stop)
        if risk_per_share <= 0:
            return None, True  # degenerate signal: skip silently, not a settlement rejection
        equity = self._mark(ledger, None, open_price, now)
        qty = math.floor((self.risk_pct * equity) / risk_per_share)
        if qty <= 0:
            return None, True
        cost = qty * fill
        if not ledger.reserve(cost, now):
            return None, False
        return _Position(sig, qty, ts, fill, cost, stop=sig.stop, target=sig.target), True

    def _check_exit(self, pos: _Position, bar) -> tuple[float | None, str]:
        """Exit price for this bar, or (None, "") if neither level was reached.

        Two rules, in this order, and the order is the whole point.

        1. GAP. If the bar's open is already beyond a level, that level was
           never available: the market's first print of the bar is the first
           price at which this position could actually have traded, so the open
           is the fill. Filling at the level instead books a loss the desk never
           had access to (on a stop) or a win it never got (on a target), which
           models overnight and open-gap risk as exactly zero. That is the
           single largest source of real loss for a 15m swing book, and it was
           the state of this function before H1.

        2. INTRABAR. Otherwise the bar traded through the level from the inside,
           and OHLC cannot say which of stop and target came first. Assume the
           stop. Pessimism is the only defensible reading of an ambiguous bar,
           and rule 1 must not be allowed to relax it: a bar that opens between
           the levels and touches both is an intrabar case, not a gap.
        """
        s = pos.signal
        open_price = float(bar.open)
        if s.direction == Direction.LONG:
            if open_price <= pos.stop:
                return self.slippage.sell(open_price), "stop"
            if open_price >= pos.target:
                return self.slippage.sell(open_price), "target"
            if bar.low <= pos.stop:
                return self.slippage.sell(pos.stop), "stop"
            if bar.high >= pos.target:
                return self.slippage.sell(pos.target), "target"
        else:
            if open_price >= pos.stop:
                return self.slippage.buy(open_price), "stop"
            if open_price <= pos.target:
                return self.slippage.buy(open_price), "target"
            if bar.high >= pos.stop:
                return self.slippage.buy(pos.stop), "stop"
            if bar.low <= pos.target:
                return self.slippage.buy(pos.target), "target"
        return None, ""

    def _close(self, pos: _Position, now, exit_price: float, reason: str, ledger: SettledLedger) -> Trade:
        s = pos.signal
        fees = self.fees.charge(pos.qty)
        if s.direction == Direction.LONG:
            pnl = (exit_price - pos.entry) * pos.qty - fees
            ledger.credit(pos.qty * exit_price - fees, now)
        else:
            pnl = (pos.entry - exit_price) * pos.qty - fees
            ledger.credit(pos.reserved + pnl, now)
        risk = abs(pos.entry - s.stop) * pos.qty
        return Trade(self.symbol, s.direction, pos.qty, pos.entry_ts, pos.entry, now, exit_price,
                     pos.stop, pos.target, pnl, pnl / risk if risk else 0.0, s.setup_tags, reason,
                     fees)

    def _mark(self, ledger: SettledLedger, pos: "_Position | None", price: float, now) -> float:
        # settled() first: it sweeps anything that has matured out of pending,
        # so the two calls must stay in this order or matured cash is counted twice.
        equity = ledger.settled(now) + ledger.pending_total()
        if pos is not None:
            if pos.signal.direction == Direction.LONG:
                equity += pos.qty * price
            else:
                equity += pos.reserved + (pos.entry - price) * pos.qty
        return equity
