"""Canonical OHLCV bar schema shared by all providers, the cache, and the engine."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

# The exchange's own conventions, in one place because three modules were each
# carrying their own copy of the timezone and the platform carried a fourth of
# the opening bell. Everything session-shaped in the engine is expressed
# against these: which UTC date a daily bar's label belongs to
# (engine/context.closed_before) and when a new regular session begins
# (strategies/util.first_bar_of_session).
EXCHANGE_TZ = "America/New_York"
# The regular session's open, as (hour, minute) of EXCHANGE_TZ. Deliberately
# the REGULAR open and not the first bar of the calendar date: caches are not
# RTH-only, so the first bar of a date is whatever pre-market print the
# provider happened to return, which is not a fact about the session.
SESSION_OPEN = (9, 30)


def empty_bars() -> pd.DataFrame:
    """The canonical "no bars" frame: right columns, right dtypes, UTC index.

    Lives here beside BAR_COLUMNS because every producer of an empty result owes
    the same shape, and a provider returning a bare DataFrame() instead sends a
    frame with no columns downstream, where it fails far from its cause.
    """
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


@dataclass(frozen=True)
class TimeframeSet:
    """The engine's time vocabulary: one driving timeframe (the replay
    cadence) plus higher context timeframes. A session-aligned timeframe
    (daily bars labeled at midnight UTC of their session date) has no fixed
    delta; visibility is decided by session date, not bar arithmetic."""

    driving: str
    higher: tuple[str, ...] = ()
    deltas: Mapping[str, pd.Timedelta] = field(default_factory=dict)
    session_aligned: frozenset[str] = frozenset()

    def __post_init__(self):
        if self.driving in self.session_aligned:
            raise ValueError("the driving timeframe cannot be session-aligned")
        if self.driving not in self.deltas:
            raise ValueError(f"driving timeframe {self.driving!r} needs a delta")
        for tf in self.higher:
            if tf not in self.deltas and tf not in self.session_aligned:
                raise ValueError(
                    f"timeframe {tf!r} needs a delta or a session_aligned entry")

    @property
    def all(self) -> tuple[str, ...]:
        return (self.driving, *self.higher)

    @property
    def step(self) -> pd.Timedelta:
        return self.deltas[self.driving]


# 4h is DERIVED, not fetched: it is resampled from the cached 1h bars against
# the Eastern wall clock (nakagai/data/resample.py). It is deliberately NOT
# session-aligned. An ET-anchored bucket labeled 12:00 closes at 16:00, four
# hours later, so plain label + delta is the exact visibility rule and
# engine/context.closed_before needs no special case for it. Only the daily
# bar, whose label carries a date rather than a close time, does.
DEFAULT_TIMEFRAMES = TimeframeSet(
    driving="15m",
    higher=("1h", "4h", "1d"),
    deltas={"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4)},
    session_aligned=frozenset({"1d"}),
)
TIMEFRAMES = DEFAULT_TIMEFRAMES.all


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized copy (sorted, deduped, float64) or raise ValueError."""
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("bars index must be tz-aware UTC DatetimeIndex")
    out = df[BAR_COLUMNS].astype("float64").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "ts"
    return out
