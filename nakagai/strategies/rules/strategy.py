"""RuleStrategy: interprets a RuleSpec against MarketContext.

Registered as "rules" in the strategy registry, so saved configs holding a spec run
through the existing backtest/scan machinery unchanged: params = {"spec": {...}}.
With no spec it is inert; the scanner can instantiate it harmlessly.
"""

import pandas as pd
from typing import ClassVar

from nakagai.engine.portfolio_types import ManagementDecision, PositionView, Signal
from nakagai.strategies import indicators as ind
from nakagai.strategies.base import HOLD, Direction, MarketContext, Strategy
from nakagai.strategies.risk import stop_target
from nakagai.strategies.rules.spec import (
    TRAILING_ATR_MULT_DEFAULT, TRAILING_ATR_N_DEFAULT, TRAILING_PCT_DEFAULT,
    validate_spec,
)
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, VocabularyFactory, core_vocabulary, resolve_vocabulary,
)
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
    VOCABULARY_FACTORY: ClassVar[VocabularyFactory] = core_vocabulary

    @classmethod
    def bound(cls, vocabulary_factory: VocabularyFactory) -> type["RuleStrategy"]:
        return type("BoundRuleStrategy", (cls,),
                    {"VOCABULARY_FACTORY": vocabulary_factory})

    def __init__(self, params: dict | None = None,
                 vocabulary: Vocabulary | None = None):
        super().__init__(params)
        self.vocabulary = resolve_vocabulary(
            vocabulary if vocabulary is not None
            else type(self).VOCABULARY_FACTORY())
        self.spec = self.params.get("spec") or {}
        # A bad spec must fail loudly at construction (backtest submission),
        # not silently emit nothing per bar. Empty spec = intentionally inert.
        errors = validate_spec(self.spec, self.vocabulary) if self.spec else []
        if errors:
            raise ValueError("; ".join(errors))

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
        cursor reads it.
        """
        tf = self.spec.get("timeframe", "1h")
        i = ctx.cursor.get(ctx.tfs.driving, -1)
        if i < 0:
            return False
        return bool(ctx.fe.driving_group(group, tf).iloc[i])

    def on_bar(self, ctx: MarketContext) -> tuple[Signal, ...]:
        if not self.spec or ctx.driving_bars.empty or not self._fresh(ctx):
            return ()
        bars = self._bars_for(ctx)
        if len(bars) < 2:
            return ()
        name = str(self.spec.get("name", "rules"))
        for side, direction in (("long", Direction.LONG), ("short", Direction.SHORT)):
            if side in self.spec and self._group_at(ctx, self.spec[side]):
                stop, target, rr = self._stop_target(ctx, bars, direction)
                sig = rr_signal(ctx, direction, stop, rr, ("rules", name),
                                f"{name}: {side} rules matched on "
                                f"{self.spec.get('timeframe', '1h')}",
                                confidence=0.5, target=target)
                if sig:
                    return (sig,)
        return ()

    def manage(self, position: PositionView,
               ctx: MarketContext) -> ManagementDecision:
        """Exits and ratchets, as one returned decision.

        The ratchets compute a local stop and hand it back. They never assign
        into `position`: the view is immutable, and the engine owns the live
        levels. `max`/`min` against the live stop is what makes a ratchet a
        ratchet, so the returned stop can only ever tighten.
        """
        exits = self.spec.get("exits") if self.spec else None
        if not exits:
            return HOLD
        bars = self._bars_for(ctx)
        if bars.empty or ctx.driving_bars.empty:
            return HOLD
        long = position.direction == Direction.LONG
        ref = float(ctx.driving_bars["close"].iloc[-1])

        if "exit" in exits and self._group_at(ctx, exits["exit"]):
            return ManagementDecision(action="exit", stop=None, target=None)
        if "time_stop" in exits:
            # held starts at 1 (not 0) on the fill bar: manage() runs in the
            # same loop pass as the fill, one driving bar after entry_ts.
            held = (ctx.now - position.entry_ts) / ctx.tfs.step
            if held >= exits["time_stop"]["bars"]:
                return ManagementDecision(action="exit", stop=None, target=None)

        stop = position.live_stop

        def ratchet(candidate: float) -> float:
            if pd.isna(candidate):
                return stop
            return max(stop, candidate) if long else min(stop, candidate)

        if "breakeven_at" in exits:
            risk = abs(position.entry - position.initial_stop)
            if risk > 0:
                r_now = (ref - position.entry) / risk if long else (position.entry - ref) / risk
                if r_now >= float(exits["breakeven_at"]["rr"]):
                    stop = ratchet(position.entry)
        if "trailing" in exits:
            t = exits["trailing"]
            if t["kind"] == "atr":
                a = ind.atr(bars, int(t.get("n", TRAILING_ATR_N_DEFAULT))).iloc[-1]
                dist = (float(t.get("mult", TRAILING_ATR_MULT_DEFAULT)) * a
                        if not pd.isna(a) else float("nan"))
            else:
                dist = ref * float(t.get("pct", TRAILING_PCT_DEFAULT)) / 100
            stop = ratchet(ref - dist if long else ref + dist)
        if stop == position.live_stop:
            return HOLD
        # A ratchet that lands ON the deciding close is not a stop, it is the
        # price. ATR is exactly zero over a window of zero-range bars (a halt,
        # an illiquid tape, a flat fixture), which makes `dist` zero and the
        # candidate the close itself. Hold the existing stop rather than hand
        # the boundary a level it is required to refuse.
        if not (stop < ref if long else stop > ref):
            return HOLD
        return ManagementDecision(action="hold", stop=stop, target=None)
