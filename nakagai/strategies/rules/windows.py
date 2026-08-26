"""Immutable time-window rows carried by a strategy vocabulary."""

from dataclasses import dataclass
from datetime import time
from typing import Literal, TypeAlias
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nakagai.data.schema import SESSION_CLOSE, SESSION_OPEN


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
        for label, boundary in (("start", self.start), ("end", self.end)):
            if (boundary.tzinfo is not None or boundary.second != 0
                    or boundary.microsecond != 0):
                raise ValueError(
                    f"window {self.name!r} {label} must be a naive "
                    "minute-resolution time")
        if self.start == self.end:
            raise ValueError(f"window {self.name!r} needs distinct start and end")
        if self.recurrence not in RECURRENCES:
            raise ValueError(
                f"window {self.name!r} has unknown recurrence "
                f"{self.recurrence!r} (valid: {RECURRENCES})")
        if self.recurrence == "xnys_session":
            if self.tz != "America/New_York":
                raise ValueError(
                    f"window {self.name!r} with xnys_session recurrence must "
                    "use America/New_York")
            if self.start < time(*SESSION_OPEN):
                raise ValueError(
                    f"window {self.name!r} with xnys_session recurrence must "
                    "start at or after 09:30 America/New_York")
            if self.end < self.start:
                raise ValueError(
                    f"window {self.name!r} with xnys_session recurrence must "
                    "end after its start on the same America/New_York date")
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


def window_duration(window: WindowSpec) -> pd.Timedelta:
    """Return the row's nominal wall-clock span."""
    start = window.start.hour * 60 + window.start.minute
    end = window.end.hour * 60 + window.end.minute
    if end < start:
        end += 24 * 60
    return pd.Timedelta(minutes=end - start)


def _local_naive(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    return index.tz_convert(tz).tz_localize(None)


def _wall_clock_mask(index: pd.DatetimeIndex, window: WindowSpec) -> np.ndarray:
    local = index.tz_convert(window.tz)
    minutes = local.hour * 60 + local.minute
    start = window.start.hour * 60 + window.start.minute
    end = window.end.hour * 60 + window.end.minute
    if start < end:
        inside = (minutes >= start) & (minutes < end)
    else:
        inside = (minutes >= start) | (minutes < end)
    return np.asarray(inside & (local.weekday < 5))


def _occurrence_bounds(
    days: pd.DatetimeIndex,
    window: WindowSpec,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    start = days + pd.Timedelta(
        hours=window.start.hour,
        minutes=window.start.minute,
    )
    end = days + pd.Timedelta(
        hours=window.end.hour,
        minutes=window.end.minute,
    )
    if window.end < window.start:
        end += pd.Timedelta(days=1)
    return (
        start.tz_localize(window.tz).tz_convert("UTC"),
        end.tz_localize(window.tz).tz_convert("UTC"),
    )


def _current_occurrences(
    index: pd.DatetimeIndex,
    window: WindowSpec,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    local = _local_naive(index, window.tz)
    if window.recurrence == "weekday":
        first = local.min().normalize() - pd.Timedelta(days=1)
        last = local.max().normalize()
        days = pd.date_range(first, last, freq="D")
        days = days[days.weekday < 5]
    else:
        exchange = index.tz_convert("America/New_York")
        minutes = exchange.hour * 60 + exchange.minute
        regular_open = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
        regular_close = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
        is_regular = (
            (exchange.weekday < 5)
            & (minutes >= regular_open)
            & (minutes < regular_close)
        )
        days = _local_naive(index[is_regular], window.tz).normalize().unique().sort_values()
    return _occurrence_bounds(pd.DatetimeIndex(days), window)


def _current_occurrence_state(
    index: pd.DatetimeIndex,
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map rows to their latest open and its causal lifecycle state."""
    timestamps = index.asi8
    occurrence = np.searchsorted(starts.asi8, timestamps, side="right") - 1
    has_occurrence = occurrence >= 0
    safe_occurrence = occurrence.clip(min=0)
    inside = has_occurrence & (timestamps < ends.asi8[safe_occurrence])
    closed = has_occurrence & (timestamps >= ends.asi8[safe_occurrence])
    return occurrence, inside, closed


def _aggregate_current(
    source: pd.Series,
    window: WindowSpec,
    reducer: Literal["max", "min", "first", "last"],
) -> pd.Series:
    starts, ends = _current_occurrences(source.index, window)
    if not len(starts):
        return pd.Series(np.nan, index=source.index, dtype="float64")

    occurrence, inside, closed = _current_occurrence_state(
        source.index,
        starts,
        ends,
    )

    keys = pd.Series(occurrence, index=source.index)
    grouped = source.astype("float64").where(inside).groupby(keys).agg(reducer)
    aggregates = np.full(len(starts), np.nan, dtype="float64")
    grouped = grouped[grouped.index >= 0]
    aggregates[grouped.index.to_numpy(dtype="int64")] = grouped.to_numpy(dtype="float64")

    result = np.full(len(source), np.nan, dtype="float64")
    result[closed] = aggregates[occurrence[closed]]
    return pd.Series(result, index=source.index, dtype="float64")


def _period_keys(
    index: pd.DatetimeIndex,
    window: WindowSpec,
    session_aligned: bool,
) -> np.ndarray:
    if session_aligned:
        local = index.tz_convert("UTC").tz_localize(None)
    else:
        local = _local_naive(index, window.tz)
    if window.recurrence == "prior_session":
        periods = local.normalize()
    elif window.recurrence == "prior_iso_week":
        periods = local.to_period("W-SUN").start_time
    else:
        periods = local.to_period("M").start_time
    return periods.asi8


def _aggregate_prior(
    source: pd.Series,
    window: WindowSpec,
    reducer: Literal["max", "min", "first", "last"],
    session_aligned: bool,
) -> pd.Series:
    keys = _period_keys(source.index, window, session_aligned)
    usable = np.ones(len(source), dtype=bool)
    if not session_aligned:
        usable = _wall_clock_mask(source.index, window)
    inside = source.astype("float64").where(usable)
    aggregates = inside.groupby(keys).agg(reducer).dropna().sort_index()
    if aggregates.empty:
        return pd.Series(np.nan, index=source.index, dtype="float64")

    aggregate_keys = aggregates.index.to_numpy(dtype="int64")
    prior = np.searchsorted(aggregate_keys, keys, side="left") - 1
    has_prior = prior >= 0
    result = np.full(len(source), np.nan, dtype="float64")
    values = aggregates.to_numpy(dtype="float64")
    result[has_prior] = values[prior[has_prior]]
    return pd.Series(result, index=source.index, dtype="float64")


def aggregate_window(
    source: pd.Series,
    window: WindowSpec,
    reducer: Literal["max", "min", "first", "last"],
    *,
    session_aligned: bool = False,
) -> pd.Series:
    """Aggregate one recurring window without exposing an active occurrence."""
    if source.empty:
        return pd.Series(np.nan, index=source.index, dtype="float64")
    if window.recurrence in ("weekday", "xnys_session"):
        return _aggregate_current(source, window, reducer)
    return _aggregate_prior(source, window, reducer, session_aligned)
