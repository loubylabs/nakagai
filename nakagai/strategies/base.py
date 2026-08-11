"""Axis 1 contract: strategies are pure functions MarketContext -> (Signal, ...).

A strategy proposes and the engine decides. Everything a strategy hands back
is an immutable value, checked at the return, and a strategy can no longer
reach into engine-owned state to ratchet a stop or close a position.

The types themselves live in `nakagai.engine.portfolio_types`, which owns the
whole canonical contract and imports nothing from this package. This module
adds the boundary: the three functions the replay calls instead of calling a
strategy directly, and the closed error taxonomy they raise.

- `validate_signal_sequence` and `validate_management_decision` check a
  returned value against the contract and raise `StrategyOutputError`.
- `strategy_operation` wraps a call INTO strategy code: an arbitrary exception
  becomes `StrategyRuntimeError`, a mutation attempt becomes
  `StrategyOutputError`.

Neither error is ever converted into an empty signal list. A strategy that
refused and a strategy that saw nothing are different observations, and a
portfolio replay that cannot tell them apart reports contention it never had.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass, field
from enum import StrEnum
from typing import ClassVar

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.portfolio_types import (
    DIRECTIONS,
    JSONValue,
    ManagementDecision,
    PositionView,
    ReplayInputError,
    Signal,
    StrategyOutputError,
    StrategyRuntimeError,
    brackets_protectively,
    _require_binary64,
    _require_choice,
    _require_instance,
    _require_name,
    _require_positive,
    _require_symbol,
    _require_tags,
)

# One shared answer for "nothing to do with this position". Frozen, so a
# single instance is safe to hand back from every quiet management call.
HOLD = ManagementDecision(action="hold", stop=None, target=None)


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


@dataclass
class MarketContext:
    symbol: str
    now: pd.Timestamp
    bars: dict[str, pd.DataFrame]
    tfs: TimeframeSet = DEFAULT_TIMEFRAMES
    # Whole-frame node evaluation, the one walker over the rule grammar.
    # build_context always supplies one: a replay's covers the untruncated
    # frames, a point-in-time caller's covers the frames already cut at `now`.
    # It defaults to None only so a hand-built context stays constructible for
    # the strategies that never touch the grammar. `cursor[tf]` is the row index
    # of the bar closing at `now`, or -1 when that timeframe has nothing visible.
    fe: object | None = None
    cursor: dict[str, int] = field(default_factory=dict)

    @property
    def driving_bars(self) -> pd.DataFrame:
        return self.bars[self.tfs.driving]


@dataclass(frozen=True)
class ParamSpec:
    """One tunable knob: default value + soft bounds + UI copy.

    lo/hi are the slider range the UI offers and the fallback validation
    bounds when config/guardrails.yaml has no param_bounds entry. `advanced`
    hides the knob behind progressive disclosure in the tuning UI.
    """

    default: float | int
    lo: float
    hi: float
    step: float = 1.0
    label: str = ""
    help: str = ""
    advanced: bool = False

    @property
    def kind(self) -> str:
        return "int" if isinstance(self.default, int) else "float"

    def to_dict(self) -> dict:
        return {
            "default": self.default, "lo": self.lo, "hi": self.hi,
            "step": self.step, "label": self.label, "help": self.help,
            "advanced": self.advanced, "kind": self.kind,
        }


class Strategy(ABC):
    name: str
    # Catalog metadata: what the Strategy Lab renders on template cards.
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = "other"  # trend|mean-reversion|breakout|momentum|price-action|other
    tags: ClassVar[tuple[str, ...]] = ()
    timeframe: ClassVar[str] = "15m"  # driving timeframe for entries
    PARAMS: ClassVar[dict[str, ParamSpec]] = {}
    DEFAULT_PARAMS: ClassVar[dict] = {}
    # Cross-param ordering rules: each (a, b) means effective a must stay
    # strictly below effective b (fast_n < slow_n). Validated at save/backtest
    # and surfaced to the UI; on_bar guards remain the backstop.
    LT_CONSTRAINTS: ClassVar[tuple[tuple[str, str], ...]] = ()

    def __init_subclass__(cls, **kwargs):
        # PARAMS is the one source of truth; DEFAULT_PARAMS stays derived so
        # every pre-ParamSpec call site (tuning gate, CLI, tests) keeps working.
        super().__init_subclass__(**kwargs)
        if "PARAMS" in cls.__dict__ and "DEFAULT_PARAMS" not in cls.__dict__:
            cls.DEFAULT_PARAMS = {k: s.default for k, s in cls.PARAMS.items()}

    @classmethod
    def spec_bounds(cls) -> dict[str, tuple[float, float]]:
        return {k: (s.lo, s.hi) for k, s in cls.PARAMS.items()}

    @classmethod
    def constraint_errors(cls, params: dict) -> list[str]:
        effective = {**cls.DEFAULT_PARAMS, **(params or {})}
        return [
            f"{a} ({effective[a]}) must be less than {b} ({effective[b]})"
            for a, b in cls.LT_CONSTRAINTS
            if effective.get(a) is not None and effective.get(b) is not None
            and effective[a] >= effective[b]
        ]

    @classmethod
    def meta(cls) -> dict:
        return {
            "title": cls.title or cls.name,
            "description": cls.description,
            "category": cls.category,
            "tags": list(cls.tags),
            "timeframe": cls.timeframe,
        }

    def __init__(self, params: dict | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    @abstractmethod
    def on_bar(self, ctx: MarketContext) -> Sequence[Signal]:
        """Every signal this strategy proposes at this close, in order.

        The order is semantic: it carries into replay-wide signal ordinals,
        so returning two signals means two proposals, not a preference list.
        """
        raise NotImplementedError

    def manage(self, position: PositionView, ctx: MarketContext) -> ManagementDecision:
        """What to do with one open position, as a value.

        `position` is an immutable view. Replacement stops and targets travel
        back in the decision; assigning to the view raises.
        """
        return HOLD


# ------------------------------------------------------------- the boundary


def _output_error(code: str, message: str, **details: JSONValue) -> StrategyOutputError:
    return StrategyOutputError(code, message, details)


@contextmanager
def _as_output_error() -> Iterator[None]:
    """Field checks are shared with the request contract, where the same shape
    violation is a caller error. Coming out of a strategy return it is a
    strategy output error, so translate the class and keep the code."""
    try:
        yield
    except ReplayInputError as exc:
        raise StrategyOutputError(exc.code, exc.message, exc.details) from exc


@contextmanager
def strategy_operation(operation: str, **details: JSONValue) -> Iterator[None]:
    """The one door into strategy code: construction, `on_bar`, `manage`,
    a composite member, a dependency function, or a helper they call.

    Everything that escapes lands in the closed taxonomy, and nothing is
    swallowed. `operation` and the identifying details are stable strings a
    caller can branch on; the original exception stays attached as the cause
    for a human, and its traceback text is never serialized into the details.
    """
    try:
        yield
    except (StrategyOutputError, StrategyRuntimeError):
        raise
    except FrozenInstanceError as exc:
        raise StrategyOutputError(
            "strategy_mutated_engine_state",
            f"{operation} tried to assign to an engine-owned value",
            {"operation": operation, **details},
        ) from exc
    except ReplayInputError as exc:
        # The strategy built a value core refuses. That is an output error
        # wherever it was caught, and it must not read as a runtime fault.
        raise StrategyOutputError(
            exc.code, exc.message,
            {**exc.details, "operation": operation, **details},
        ) from exc
    except Exception as exc:
        raise StrategyRuntimeError(
            "strategy_raised", f"{operation} raised {type(exc).__name__}",
            {"operation": operation, "error": type(exc).__name__, **details},
        ) from exc


def call_on_bar(strategy: Strategy, ctx: MarketContext, *, deciding_close: float,
                **details: JSONValue) -> tuple[Signal, ...]:
    """Evaluate one strategy at one close and return every valid signal.

    `details` name this runtime for an operator: a composite passes its block
    id, the replay passes its play. They travel into a runtime error and
    identify which of several instances of one strategy class failed.
    """
    with strategy_operation("on_bar", strategy=_strategy_name(strategy),
                            symbol=ctx.symbol, **details):
        returned = strategy.on_bar(ctx)
    return validate_signal_sequence(returned, symbol=ctx.symbol,
                                    deciding_close=deciding_close)


def call_manage(strategy: Strategy, position: PositionView, ctx: MarketContext,
                *, deciding_close: float, **details: JSONValue) -> ManagementDecision:
    """Manage one open position at one close and return its decision."""
    with strategy_operation("manage", strategy=_strategy_name(strategy),
                            symbol=ctx.symbol, **details):
        returned = strategy.manage(position, ctx)
    return validate_management_decision(returned, position=position,
                                        deciding_close=deciding_close)


def validate_signal_sequence(value: object, *, symbol: str,
                             deciding_close: float) -> tuple[Signal, ...]:
    """Every signal `on_bar` returned, in the order it returned them.

    A string, mapping, generator, or any other non-sequence is refused rather
    than iterated: a generator would be consumed once and a string would
    decompose into characters, and both read downstream as a strategy that
    signalled something it never meant.
    """
    # The symbol and the close are the ENGINE's, so a bad one is a replay
    # input error. Only what the strategy handed back is its output.
    expected = _require_symbol(symbol, "symbol")
    close = _require_positive(deciding_close, "deciding_close")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _output_error(
            "invalid_type", "on_bar must return an ordered sequence of signals",
            field="on_bar", seen=type(value).__name__,
        )
    return tuple(
        _validated_signal(item, index, expected, close)
        for index, item in enumerate(value)
    )


def _validated_signal(item: object, index: int, symbol: str,
                      deciding_close: float) -> Signal:
    where = f"signals[{index}]"
    if not isinstance(item, Signal):
        raise _output_error(
            "invalid_type", "on_bar returned a value that is not a signal",
            field=where, seen=type(item).__name__,
        )
    with _as_output_error():
        _require_symbol(item.symbol, f"{where}.symbol")
        direction = _require_choice(item.direction, f"{where}.direction", DIRECTIONS)
        entry_ref = _require_positive(item.entry_ref, f"{where}.entry_ref")
        stop = _require_positive(item.stop, f"{where}.stop")
        target = _require_positive(item.target, f"{where}.target")
        confidence = _require_binary64(item.confidence, f"{where}.confidence")
        tags = _require_tags(item.setup_tags, f"{where}.setup_tags")
        _require_name(item.rationale, f"{where}.rationale")
    if item.symbol != symbol:
        raise _output_error(
            "invalid_value", "the signal names another symbol",
            field=f"{where}.symbol", symbol=symbol,
        )
    if not 0.0 < confidence <= 1.0:
        raise _output_error(
            "invalid_value", "confidence must fall in (0, 1]",
            field=f"{where}.confidence",
        )
    if not tags:
        raise _output_error(
            "invalid_value", "a signal must carry at least one setup tag",
            field=f"{where}.setup_tags",
        )
    if entry_ref != deciding_close:
        raise _output_error(
            "invalid_value", "the entry reference is not the deciding raw close",
            field=f"{where}.entry_ref",
        )
    if not brackets_protectively(direction, entry_ref, stop, target):
        raise _output_error(
            "invalid_value", "protective levels do not bracket the entry reference",
            field=f"{where}.stop", direction=direction,
        )
    return item


def validate_management_decision(value: object, *, position: PositionView,
                                 deciding_close: float) -> ManagementDecision:
    """The decision `manage` returned, checked against the live position.

    Only what the decision REPLACES is judged, under two separate rules:

    - each replacement sits on its own protective side of the deciding close;
    - the pair the decision lands in still opens outward, replacement or live.

    A null stop or target keeps the live level, and a live level is the
    engine's own state: where the close sits relative to it is a fact about
    the market and the engine's exit ordering, never a claim the strategy
    made. Re-checking that would abort the replay over an ordinary losing
    trade whose bar closed beyond its stop, and would blame the strategy for
    it. A decision that replaces nothing is therefore checked for nothing.
    """
    # Engine-supplied, so its own failure is a replay input error.
    _require_instance(position, "position", PositionView)
    close = _require_positive(deciding_close, "deciding_close")
    if not isinstance(value, ManagementDecision):
        raise _output_error(
            "invalid_type", "manage must return a management decision",
            field="manage", seen=type(value).__name__,
        )
    long = position.direction == Direction.LONG
    if value.stop is not None:
        loosened = (value.stop < position.live_stop if long
                    else value.stop > position.live_stop)
        if loosened:
            raise _output_error(
                "invalid_value", "a replacement stop cannot loosen the live stop",
                field="stop", direction=position.direction,
            )
        # A stop protects from below on a long and from above on a short.
        if not (value.stop < close if long else value.stop > close):
            raise _output_error(
                "unprotective_replacement",
                "a replacement stop does not protect the deciding close",
                field="stop", direction=position.direction,
            )
    if value.target is not None:
        if not (value.target > close if long else value.target < close):
            raise _output_error(
                "unprotective_replacement",
                "a replacement target is already behind the deciding close",
                field="target", direction=position.direction,
            )
    if value.stop is not None or value.target is not None:
        # A SECOND invariant, and not a restatement of the one above. That one
        # asks where a replacement sits against the close; this one asks
        # whether the pair it lands in still opens outward. A replacement is
        # checked against the close alone, so it can pass that and still cross
        # an untouched counterpart level, which leaves a position whose stop
        # and target cover the whole real line and which the engine then exits
        # at the next open whatever the price does. Do NOT collapse this back
        # into a `stop < close < target` pivot: demanding that the close sit
        # between them judges engine-owned state and refuses an ordinary
        # losing trade.
        stop = position.live_stop if value.stop is None else value.stop
        target = position.live_target if value.target is None else value.target
        if not (stop < target if long else target < stop):
            raise _output_error(
                "crossed_protective_levels",
                "the decided stop and target cross each other",
                field="stop" if value.stop is not None else "target",
                direction=position.direction,
            )
    return value


def _strategy_name(strategy: Strategy) -> str:
    """A name for the error details even when the object has none to give."""
    name = getattr(strategy, "name", None)
    return name if isinstance(name, str) else type(strategy).__name__
