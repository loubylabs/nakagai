"""RuleStrategy: interprets a RuleSpec against MarketContext.

Registered as "rules" in the strategy registry, so saved configs holding a spec run
through the existing backtest/scan machinery unchanged: params = {"spec": {...}}.
With no spec it is inert; the scanner can instantiate it harmlessly.
"""

import pandas as pd

from nakagai.strategies import indicators as ind
from nakagai.strategies.base import Direction, MarketContext, PositionAction, Signal, Strategy
from nakagai.strategies.risk import stop_target
from nakagai.strategies.rules.exprs import eval_group
from nakagai.strategies.rules.spec import validate_spec
from nakagai.strategies.util import fresh_bar, first_bar_of_session, rr_signal


class RuleStrategy(Strategy):
    name = "rules"
    title = "Custom rules"
    description = ("User-defined strategy: entry conditions, stop, and target "
                   "described as declarative rules: imported, hand-built, or "
                   "compiled from a natural-language description.")
    category = "custom"
    tags = ("custom", "rules", "imported")
    DEFAULT_PARAMS = {}

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.spec = self.params.get("spec") or {}
        # A bad spec must fail loudly at construction (backtest submission),
        # not silently emit nothing per bar. Empty spec = intentionally inert.
        if self.spec and validate_spec(self.spec):
            raise ValueError("; ".join(validate_spec(self.spec)))

    def _bars_for(self, ctx: MarketContext) -> pd.DataFrame:
        return ctx.bars[self.spec.get("timeframe", "1h")]

    def _fresh(self, ctx: MarketContext) -> bool:
        tf = self.spec.get("timeframe", "1h")
        if tf == ctx.tfs.driving:
            return True
        if tf in ctx.tfs.session_aligned:
            return first_bar_of_session(ctx)
        return fresh_bar(ctx, tf)

    def _stop_target(self, ctx: MarketContext, bars: pd.DataFrame,
                     direction: Direction) -> tuple[float, float | None, float]:
        """-> (stop, target, rr); target None means rr-derived in rr_signal."""
        return stop_target(self.spec.get("risk", {}), ctx, bars, direction)

    def _group_at(self, ctx: MarketContext, group: dict) -> bool:
        """One all/any tree at `now`.

        The tree is evaluated on the SPEC's timeframe (so `crosses_above`
        compares consecutive spec-timeframe bars, as the per-bar path did) and
        the resulting boolean is lifted onto the driving index, where the
        cursor reads it. A point-in-time context has no replay, so it falls
        back to evaluating the tree directly.
        """
        tf = self.spec.get("timeframe", "1h")
        if ctx.fe is None:
            return eval_group(group, ctx, self._bars_for(ctx), {})
        i = ctx.cursor.get(ctx.tfs.driving, -1)
        if i < 0:
            return False
        return bool(ctx.fe.driving_group(group, tf).iloc[i])

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        if not self.spec or ctx.driving_bars.empty or not self._fresh(ctx):
            return []
        bars = self._bars_for(ctx)
        if len(bars) < 2:
            return []
        name = str(self.spec.get("name", "rules"))
        for side, direction in (("long", Direction.LONG), ("short", Direction.SHORT)):
            if side in self.spec and self._group_at(ctx, self.spec[side]):
                stop, target, rr = self._stop_target(ctx, bars, direction)
                sig = rr_signal(ctx, direction, stop, rr, ("rules", name),
                                f"{name}: {side} rules matched on "
                                f"{self.spec.get('timeframe', '1h')}",
                                confidence=0.5, target=target)
                if sig:
                    return [sig]
        return []

    def manage(self, position, ctx: MarketContext) -> PositionAction:
        exits = self.spec.get("exits") if self.spec else None
        if not exits:
            return PositionAction.HOLD
        bars = self._bars_for(ctx)
        if bars.empty or ctx.driving_bars.empty:
            return PositionAction.HOLD
        direction = position.signal.direction
        long = direction == Direction.LONG
        ref = float(ctx.driving_bars["close"].iloc[-1])

        if "exit" in exits and self._group_at(ctx, exits["exit"]):
            return PositionAction.EXIT
        if "time_stop" in exits:
            # held starts at 1 (not 0) on the fill bar: manage() runs in the
            # same loop pass as the fill, one driving bar after entry_ts.
            held = (ctx.now - position.entry_ts) / ctx.tfs.step
            if held >= exits["time_stop"]["bars"]:
                return PositionAction.EXIT

        def ratchet(candidate: float) -> None:
            if pd.isna(candidate):
                return
            position.stop = max(position.stop, candidate) if long \
                else min(position.stop, candidate)

        if "breakeven_at" in exits:
            risk = abs(position.entry - position.signal.stop)
            if risk > 0:
                r_now = (ref - position.entry) / risk if long else (position.entry - ref) / risk
                if r_now >= float(exits["breakeven_at"]["rr"]):
                    ratchet(position.entry)
        if "trailing" in exits:
            t = exits["trailing"]
            if t["kind"] == "atr":
                a = ind.atr(bars, int(t.get("n", 14))).iloc[-1]
                dist = float(t.get("mult", 2.0)) * a if not pd.isna(a) else float("nan")
            else:
                dist = ref * float(t.get("pct", 2.0)) / 100
            ratchet(ref - dist if long else ref + dist)
        return PositionAction.HOLD
