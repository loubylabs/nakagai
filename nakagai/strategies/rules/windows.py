"""Immutable time-window rows carried by a strategy vocabulary."""

from dataclasses import dataclass
from datetime import time
from typing import Literal, TypeAlias
from zoneinfo import ZoneInfo


WindowRecurrence: TypeAlias = Literal[
    "weekday",
    "xnys_session",
    "prior_session",
    "prior_iso_week",
    "prior_calendar_month",
]
WindowConfidence: TypeAlias = Literal["standard", "low_iex"]

RECURRENCES = (
    "weekday",
    "xnys_session",
    "prior_session",
    "prior_iso_week",
    "prior_calendar_month",
)
CONFIDENCE_LEVELS = ("standard", "low_iex")


@dataclass(frozen=True)
class WindowSpec:
    """One permanent named time scope in the strategy grammar."""

    name: str
    tz: str
    start: time
    end: time
    recurrence: WindowRecurrence
    confidence: WindowConfidence

    def __post_init__(self) -> None:
        ZoneInfo(self.tz)
        if self.start == self.end:
            raise ValueError(f"window {self.name!r} needs distinct start and end")
        if self.recurrence not in RECURRENCES:
            raise ValueError(
                f"window {self.name!r} has unknown recurrence "
                f"{self.recurrence!r} (valid: {RECURRENCES})")
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"window {self.name!r} has unknown confidence "
                f"{self.confidence!r} (valid: {CONFIDENCE_LEVELS})")


PRIOR_DAY = WindowSpec(
    "prior_day",
    "America/New_York",
    time(9, 30),
    time(16),
    "prior_session",
    "standard",
)
