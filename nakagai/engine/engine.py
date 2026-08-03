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
from nakagai.strategies.rules.vocabulary import core_vocabulary


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
    # Maximum adverse and favorable excursion, in R against the signal's
    # ORIGINAL stop (the same denominator r_multiple uses, so a ratcheted stop
    # cannot make two fields on one trade disagree about what 1R meant).
    #
    # R rather than dollars, so the figures are comparable across symbols and
    # price levels: "no trade in this catalog ever went more than 0.6R against
    # us" is a sentence about stop placement, where the same claim in dollars
    # is a sentence about share price. The dollar excursion is recoverable as
    # mae * abs(entry - stop).
    #
    # Both are magnitudes and never negative. A trade that only ever went in
    # our favor has mae 0.0, which is a measurement, not a missing value.
    mae: float = 0.0
    mfe: float = 0.0


@dataclass
class _Position:
    signal: Signal
    qty: int
    entry_ts: pd.Timestamp
    entry: float
    reserved: float
    stop: float = 0.0     # live levels; strategies may ratchet via manage()
    target: float = 0.0
    # Running excursion extremes, in price. Seeded to the entry in __post_init__
    # so a position that closes on its own entry bar reports 0.0 for both,
    # rather than an excursion measured from an unset sentinel.
    worst: float = 0.0
    best: float = 0.0

    def __post_init__(self):
        self.worst = self.best = self.entry

    def observe(self, high: float, low: float) -> None:
        """Fold one bar's extremes into the running excursion.

        Reads only. This is the whole of P0's contact with the bar loop, and
        it deliberately touches nothing the fill path reads, which is what
        lets the excursion ship without a re-prove (tests/test_engine_excursion
        holds the golden that proves it)."""
        if self.signal.direction == Direction.LONG:
            self.worst = min(self.worst, low)
            self.best = max(self.best, high)
        else:
            self.worst = max(self.worst, high)
            self.best = min(self.best, low)


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
        vocabulary = getattr(self.strategy, "vocabulary", core_vocabulary())
        view = PreloadedBars(self.cache, self.symbol, self.tfs, vocabulary)
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
                    # The exit price only, never this bar's full range: the
                    # position was closed partway through the bar and OHLC
                    # cannot say what happened before that. Folding the whole
                    # range in would credit the trade with an excursion it was
                    # no longer open for, which is the one way a favorable
                    # excursion could flatter a trade that had already exited.
                    position.observe(exit_price, exit_price)
                    trades.append(self._close(position, now, exit_price, reason, ledger))
                    position = None
                else:
                    position.observe(float(bar.high), float(bar.low))

            # 2) pending entry fills at this bar's open
            if pending is not None:
                if position is None:
                    position, ok = self._try_fill(pending, ts, float(bar.open), ledger, now)
                    if not ok:
                        rejected += 1
                    elif position is not None:
                        # The entry bar counts. The fill is at this bar's open
                        # and step 1 already ran, so the position is live for
                        # the whole of this bar by the engine's own model.
                        position.observe(float(bar.high), float(bar.low))
                pending = None

            # 3) manage + 4) new signals (point-in-time context as of bar close)
            ctx = build_context(view, self.symbol, now, tfs=self.tfs,
                                vocabulary=vocabulary)
            if position is not None:
                if self.strategy.manage(position, ctx) == PositionAction.EXIT:
                    position.observe(float(bar.close), float(bar.close))
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
            position.observe(float(last["close"]), float(last["close"]))
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
        # Per-share risk against the SIGNAL's stop, the same denominator
        # r_multiple uses above. pos.stop may have been ratcheted by manage();
        # dividing by that instead would make a trade's mae and its r_multiple
        # disagree about the size of 1R.
        risk_ps = abs(pos.entry - s.stop)
        if s.direction == Direction.LONG:
            adverse, favorable = pos.entry - pos.worst, pos.best - pos.entry
        else:
            adverse, favorable = pos.worst - pos.entry, pos.entry - pos.best
        return Trade(self.symbol, s.direction, pos.qty, pos.entry_ts, pos.entry, now, exit_price,
                     pos.stop, pos.target, pnl, pnl / risk if risk else 0.0, s.setup_tags, reason,
                     fees,
                     mae=adverse / risk_ps if risk_ps else 0.0,
                     mfe=favorable / risk_ps if risk_ps else 0.0)

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
