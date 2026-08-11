"""RuleStrategy: interprets a RuleSpec against MarketContext.

Registered as "rules" in the strategy registry, so saved configs holding a spec run
through the existing backtest/scan machinery unchanged: params = {"spec": {...}}.
With no spec it is inert; the scanner can instantiate it harmlessly.
"""

from collections.abc import Mapping
from typing import ClassVar

import pandas as pd

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
from nakagai.strategies.util import rr_signal

# The frame a spec is evaluated on when it names none. One spelling, because
# the dependency walk and the evaluator have to agree about which frames a
# spec reads: a walk that guessed a different default would declare a frame
# nobody hydrates, or omit one the evaluator then asks for.
SPEC_TIMEFRAME_DEFAULT = "1h"


def spec_timeframes(spec: Mapping) -> tuple:
    """Every timeframe a RuleSpec reads, unvalidated and possibly repeated.

    Two sources, and both are frames the replay has to prepare: the spec's own
    `timeframe`, which its conditions are evaluated on, and any node's `tf`,
    which moves that node's children onto another frame. An empty spec reads
    neither, because an inert strategy returns before it touches a frame.

    Values travel out exactly as the spec spelled them, so a caller building
    `StrategyDependencies` refuses an unsupported one through the same door
    every other timeframe goes through. Keep this in step with `_bars_for` and
    `_group_at`: a frame this misses is a frame nobody hydrates.
    """
    if not isinstance(spec, Mapping) or not spec:
        return ()
    found = [spec.get("timeframe", SPEC_TIMEFRAME_DEFAULT)]
    stack: list = [spec]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            if "tf" in node:
                found.append(node["tf"])
            stack.extend(node.values())
        elif isinstance(node, (tuple, list)):
            stack.extend(node)
    return tuple(found)


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
                 vocabulary: Vocabulary | None = None, *,
                 name: str | None = None):
        super().__init__(params)
        if name is not None:
            # A registry definition names its own runtime. The assignment
            # shadows the class attribute on this instance alone, so one
            # immutable definition can build a play called `sma_cross` without
            # minting a subclass per catalog entry, which is the mutable
            # class binding the canonical path does not use.
            self.name = name
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
        return ctx.bars[self.spec.get("timeframe", SPEC_TIMEFRAME_DEFAULT)]

    def _fresh(self, ctx: MarketContext) -> bool:
        """May this play decide at `ctx.now`, given the frame it is decided on.

        Asked of the context, never reconstructed. A play decided on the frame
        the engine replays is fresh on every step of it; every other one is
        fresh where its context says so, which for a portfolio replay is the
        schedule's own `fresh_context_at` and for a caller with no schedule is
        the label rule in `strategies/util.label_freshness`.

        A missing key is a spec naming a frame nobody hydrated, and it raises
        here rather than reading False. Going quiet would make the play emit
        nothing for the whole replay and report it as a market with no setups.
        """
        tf = self.spec.get("timeframe", SPEC_TIMEFRAME_DEFAULT)
        return tf == ctx.tfs.driving or ctx.fresh[tf]

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
        tf = self.spec.get("timeframe", SPEC_TIMEFRAME_DEFAULT)
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
                                f"{self.spec.get('timeframe', SPEC_TIMEFRAME_DEFAULT)}",
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
