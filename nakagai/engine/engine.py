"""Event-driven bar-replay backtester. Share-only, long AND short, cash-settlement aware."""

import math
from dataclasses import dataclass

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import PreloadedBars, build_context
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
                 slippage: float = 0.01, tfs: TimeframeSet = DEFAULT_TIMEFRAMES):
        self.strategy, self.cache, self.symbol = strategy, cache, symbol
        self.start, self.end = start, end
        self.equity0, self.risk_pct, self.slippage = equity0, risk_pct, slippage
        self.tfs = tfs

    def run(self) -> BacktestResult:
        view = PreloadedBars(self.cache, self.symbol, self.tfs)
        bars = view.load(self.symbol, self.tfs.driving)
        bars = bars[(bars.index >= self.start) & (bars.index < self.end)]
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
        fill = open_price + self.slippage if sig.direction == Direction.LONG else open_price - self.slippage
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
        s = pos.signal
        if s.direction == Direction.LONG:
            if bar.low <= pos.stop:
                return pos.stop - self.slippage, "stop"
            if bar.high >= pos.target:
                return pos.target - self.slippage, "target"
        else:
            if bar.high >= pos.stop:
                return pos.stop + self.slippage, "stop"
            if bar.low <= pos.target:
                return pos.target + self.slippage, "target"
        return None, ""

    def _close(self, pos: _Position, now, exit_price: float, reason: str, ledger: SettledLedger) -> Trade:
        s = pos.signal
        if s.direction == Direction.LONG:
            pnl = (exit_price - pos.entry) * pos.qty
            ledger.credit(pos.qty * exit_price, now)
        else:
            pnl = (pos.entry - exit_price) * pos.qty
            ledger.credit(pos.reserved + pnl, now)
        risk = abs(pos.entry - s.stop) * pos.qty
        return Trade(self.symbol, s.direction, pos.qty, pos.entry_ts, pos.entry, now, exit_price,
                     pos.stop, pos.target, pnl, pnl / risk if risk else 0.0, s.setup_tags, reason)

    def _mark(self, ledger: SettledLedger, pos: "_Position | None", price: float, now) -> float:
        equity = ledger.settled(now) + sum(a for _, a in ledger._pending)
        if pos is not None:
            if pos.signal.direction == Direction.LONG:
                equity += pos.qty * price
            else:
                equity += pos.reserved + (pos.entry - price) * pos.qty
        return equity
