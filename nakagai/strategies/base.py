"""Axis 1 contract: strategies are pure functions MarketContext -> [Signal]."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"


class PositionAction(StrEnum):
    HOLD = "hold"
    EXIT = "exit"


@dataclass(frozen=True)
class Signal:
    symbol: str
    direction: Direction
    entry: float | None  # None = market at the next driving-bar open
    stop: float
    target: float
    confidence: float
    setup_tags: tuple[str, ...]
    rationale: str


@dataclass
class MarketContext:
    symbol: str
    now: pd.Timestamp
    bars: dict[str, pd.DataFrame]
    tfs: TimeframeSet = DEFAULT_TIMEFRAMES

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
    def on_bar(self, ctx: MarketContext) -> list[Signal]: ...

    def manage(self, position, ctx: MarketContext) -> PositionAction:
        return PositionAction.HOLD
