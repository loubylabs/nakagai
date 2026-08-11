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

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, EXCHANGE_TZ, TimeframeSet
from nakagai.engine.context import PreloadedBars, build_context, visible_counts
from nakagai.engine.costs import FeeModel, SlippageModel
from nakagai.engine.portfolio_types import (
    PositionView, Signal, brackets_protectively,
)
from nakagai.engine.provenance import ARITHMETIC_VERSION, FILL_MODE
from nakagai.strategies.base import Direction, Strategy, call_manage, call_on_bar
from nakagai.strategies.rules.vocabulary import core_vocabulary


def _next_weekday(d: dt.date) -> dt.date:
    d += dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


class _SettledLedger:
    """T+1 settled-funds ledger for cash-account realism, skipping holidays.

    Private to this engine, and retired with it. The portfolio replay settles
    through `nakagai.engine.portfolio`, whose ledger takes its settlement
    calendar from the schedule instead of assuming every weekday is a session.
    """

    def __init__(self, cash: float):
        self._settled = float(cash)
        self._pending: list[tuple[dt.date, float]] = []  # (settle_date_NY, amount)

    def settled(self, now: pd.Timestamp) -> float:
        today = now.tz_convert(EXCHANGE_TZ).date()
        still = []
        for settle_date, amount in self._pending:
            if settle_date <= today:
                self._settled += amount
            else:
                still.append((settle_date, amount))
        self._pending = still
        return self._settled

    def reserve(self, amount: float, now: pd.Timestamp) -> bool:
        if self.settled(now) < amount:
            return False
        self._settled -= amount
        return True

    def pending_total(self) -> float:
        """Cash credited but not yet settled.

        Equity marking needs this, and reaching into _pending to get it meant a
        change to settlement bookkeeping broke the engine from a distance. Call
        settled() first: it sweeps matured entries out of _pending, so asking
        in the other order double-counts anything that has just settled.
        """
        return sum(amount for _, amount in self._pending)

    def credit(self, amount: float, now: pd.Timestamp):
        self._pending.append((_next_weekday(now.tz_convert(EXCHANGE_TZ).date()), float(amount)))


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
    # What the opening fill cost. The fee model prices one fill, so the entry
    # charge is taken at the entry and carried here rather than reconstructed
    # at the close from a doubled exit charge.
    entry_fee: float
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


def _view_of(pos: _Position) -> PositionView:
    """The immutable position a strategy is allowed to see.

    Initial levels come from the signal and never move; live levels are the
    engine's, ratcheted only by a decision it applied itself.
    """
    return PositionView(
        direction=pos.signal.direction, qty=pos.qty, entry_ts=pos.entry_ts,
        entry=pos.entry, initial_stop=pos.signal.stop,
        initial_target=pos.signal.target, live_stop=pos.stop,
        live_target=pos.target,
    )


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: pd.Series
    rejected_unsettled: int
    starting_equity: float


class Engine:
    arithmetic_version = ARITHMETIC_VERSION
    fill_mode = FILL_MODE

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

    def run(self) -> BacktestResult:
        vocabulary = getattr(self.strategy, "vocabulary", core_vocabulary())
        view = PreloadedBars(self.cache, self.symbol, self.tfs,
                             vocabulary=vocabulary)
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
        ledger = _SettledLedger(self.equity0)
        trades: list[Trade] = []
        rejected = 0
        position: _Position | None = None
        # Every signal the last close produced, in the order it produced them.
        # This engine holds one position, so the ones it cannot take expire
        # unfilled at the next open rather than being dropped at the close.
        pending: tuple[Signal, ...] = ()
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

            # 2) pending entries fill at this bar's open, in signal order
            for sig in pending:
                if position is not None:
                    break
                position, ok = self._try_fill(sig, ts, float(bar.open), ledger, now)
                if not ok:
                    rejected += 1
                elif position is not None:
                    # The entry bar counts. The fill is at this bar's open
                    # and step 1 already ran, so the position is live for
                    # the whole of this bar by the engine's own model.
                    position.observe(float(bar.high), float(bar.low))
            pending = ()

            # 3) manage + 4) new signals (point-in-time context as of bar close)
            ctx = build_context(view, self.symbol, now, tfs=self.tfs,
                                vocabulary=vocabulary)
            close = float(bar.close)
            if position is not None:
                decision = call_manage(self.strategy, _view_of(position), ctx,
                                       deciding_close=close)
                # Replacements land before the exit, so a decision that
                # ratchets and closes on the same bar records the tighter
                # level it actually exited under.
                if decision.stop is not None:
                    position.stop = decision.stop
                if decision.target is not None:
                    position.target = decision.target
                if decision.action == "exit":
                    position.observe(close, close)
                    trades.append(self._close(position, now, close, "manage", ledger))
                    position = None
            # Nothing is pending here: step 2 either filled the intent or let
            # it expire, so being flat is the whole condition.
            if position is None:
                pending = call_on_bar(self.strategy, ctx, deciding_close=close)

            curve[now] = self._mark(ledger, position, close, now)

        if position is not None and len(bars):
            last = bars.iloc[-1]
            now = bars.index[-1] + self.tfs.step
            position.observe(float(last["close"]), float(last["close"]))
            trades.append(self._close(position, now, float(last["close"]), "eod_window", ledger))
            curve[now] = self._mark(ledger, None, float(last["close"]), now)

        return BacktestResult(trades, pd.Series(curve, dtype="float64"), rejected, self.equity0)

    def _try_fill(self, sig: Signal, ts, open_price: float, ledger: _SettledLedger, now) -> tuple["_Position | None", bool]:
        fill = (self.slippage.buy(open_price) if sig.direction == Direction.LONG
                else self.slippage.sell(open_price))
        if not brackets_protectively(sig.direction, fill, sig.stop, sig.target):
            # The market gapped past a protective level between the deciding
            # close and this open. Entering here would open a position that
            # its own stop or target already sits inside, which is not a
            # position: the print that broke the level happened before the
            # entry existed, so it could not have triggered anything.
            return None, True
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
        return _Position(sig, qty, ts, fill, cost, self.fees.charge(qty),
                         stop=sig.stop, target=sig.target), True

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

    def _close(self, pos: _Position, now, exit_price: float, reason: str, ledger: _SettledLedger) -> Trade:
        s = pos.signal
        # One charge per fill. The entry paid its own at the fill; this is the
        # exit's, and the trade carries the pair.
        fees = pos.entry_fee + self.fees.charge(pos.qty)
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

    def _mark(self, ledger: _SettledLedger, pos: "_Position | None", price: float, now) -> float:
        # settled() first: it sweeps anything that has matured out of pending,
        # so the two calls must stay in this order or matured cash is counted twice.
        equity = ledger.settled(now) + ledger.pending_total()
        if pos is not None:
            if pos.signal.direction == Direction.LONG:
                equity += pos.qty * price
            else:
                equity += pos.reserved + (pos.entry - price) * pos.qty
        return equity
