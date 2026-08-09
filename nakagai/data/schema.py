"""Canonical OHLCV bar schema shared by all providers, the cache, and the engine."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

# The exchange's own conventions, in one place because three modules were each
# carrying their own copy of the timezone and the platform carried a fourth of
# the opening bell. Everything session-shaped in the engine is expressed
# against these: which UTC date a daily bar's label belongs to
# (engine/context.closed_before), when a new regular session begins
# (strategies/util.first_bar_of_session), and which bars belong to a session at
# all (rth_mask below).
EXCHANGE_TZ = "America/New_York"
# The regular session's open, as (hour, minute) of EXCHANGE_TZ. Deliberately
# the REGULAR open and not the first bar of the calendar date: caches are not
# RTH-only, so the first bar of a date is whatever pre-market print the
# provider happened to return, which is not a fact about the session.
SESSION_OPEN = (9, 30)
# The regular session's close, same convention, and the far end of the same
# fact: the last bar of a session is the one that OPENS before the bell, so
# everything here treats the window as half-open, [SESSION_OPEN, SESSION_CLOSE).
# A bar labeled 16:00 is the first post-market print rather than the closing
# one, for the same reason the 08:00 print is not the open.
SESSION_CLOSE = (16, 0)


def empty_bars() -> pd.DataFrame:
    """The canonical "no bars" frame: right columns, right dtypes, UTC index.

    Lives here beside BAR_COLUMNS because every producer of an empty result owes
    the same shape, and a provider returning a bare DataFrame() instead sends a
    frame with no columns downstream, where it fails far from its cause.
    """
    idx = pd.DatetimeIndex([], tz="UTC", name="ts")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=idx)


def session_opens(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The regular open of each bar's OWN New York session, in UTC, one per row.

    The canonical form of the question. Everything session-anchored reads the
    open through here, the scalar `session_open` below included, so there is
    exactly one construction and no two callers can drift apart.

    Built by dropping to naive exchange-local time, normalizing to that
    session's local midnight, adding the SESSION_OPEN offset there, and only
    then re-localizing. The shortcut a reader reaches for first, adding a 9h30m
    Timedelta to a tz-aware midnight, is wrong: pandas does tz-aware arithmetic
    on the underlying UTC instants, so on the two Sundays a year the New York
    offset moves it lands at 08:30 or 10:30 local instead of 09:30. Going
    through naive local time puts the offset on the wall clock, which is where
    the fact lives: the bell rings at 09:30 in New York on every session,
    whatever the UTC instant happens to be that week.

    No `ambiguous` or `nonexistent` argument is needed on the re-localize, and
    none should be added. The US transitions happen at 02:00 local, so 09:30 is
    never inside the spring-forward gap and never repeated in the fall-back
    fold; passing a policy there would only hide a real error if one ever did
    appear.

    Pure index arithmetic with no Python loop, because the scalar face of this
    is called once per replayed bar per session-anchored play.
    """
    ny = index.tz_convert(EXCHANGE_TZ)
    naive = ny.tz_localize(None).normalize() + pd.Timedelta(
        hours=SESSION_OPEN[0], minutes=SESSION_OPEN[1])
    return naive.tz_localize(EXCHANGE_TZ).tz_convert("UTC")


@lru_cache(maxsize=512)
def _open_of_session(day: date) -> pd.Timestamp:
    """`session_opens` for one session, memoized on the session it answers for.

    Every bar of a session shares one open, so the answer depends on the
    session and not on the bar. Keying the memo that way is what lets a replay
    pay for the construction once a day rather than once a bar; keying it on
    the bar's own timestamp would hold a growing table at a zero hit rate,
    because a replay never visits the same bar twice.

    Safe to hold indefinitely: a pure function of a calendar date and the
    exchange's fixed conventions, neither of which anything in a running
    process can change. 512 entries is two years of sessions, and a replay
    walks time forward, so the LRU never thrashes.
    """
    return session_opens(pd.DatetimeIndex([pd.Timestamp(day, tz=EXCHANGE_TZ)]))[0]


def session_open(ts: pd.Timestamp) -> pd.Timestamp:
    """The regular open of the NY session `ts` falls in, in UTC.

    The scalar face of `session_opens`, and deliberately not a second
    construction: the DST reasoning that makes this correct is written once,
    over there, and routing through it is what stops a scalar form and a vector
    form disagreeing on the two Sundays a year where the answer moves.

    All this adds is WHICH session to ask about, the bar's exchange-local
    calendar date, which is the same session every session-scoped primitive
    groups on (strategies/rules/primitives._session_keys).

    Going through the memo rather than handing `session_opens` a one-row index
    directly, and that is not a micro-optimization. This is a hot path:
    `strategies/util.first_bar_of_session` calls it once per replayed bar, and
    on a 1d spec it is the gate every other bar of the session stops at, so it
    guards the expensive work instead of doing any. Six pandas index operations
    on a single row are nearly all fixed overhead, ~66us against ~6us for the
    scalar construction this replaced, and a measured 1d replay ran half again
    as long for it. Keyed on the date it is ~4us, cheaper than either.
    """
    return _open_of_session(ts.tz_convert(EXCHANGE_TZ).date())


def _is_session_frame(idx: pd.DatetimeIndex) -> bool:
    """True when the frame's rows are SESSION bars: at most one per UTC date.

    The property, not a label pattern. Both daily conventions the engine meets
    satisfy it and neither is recognisable from a single timestamp: legacy
    providers may label daily rows at 00:00Z, while BarCache.upsert
    canonicalizes cached daily rows to midnight America/New_York, expressed as
    04:00Z or 05:00Z. What they share is one row per session, which is also
    exactly what makes the UTC calendar
    date of the label the session date, the fact engine/context.closed_before
    already rests on.

    A frame of fewer than two rows carries no evidence of its own cadence, so
    it reads as intraday: that is the answer for the driving frames the grammar
    admits everywhere except a 1d spec, and a 1d spec reaching one row has one
    session of history.
    """
    return len(idx) >= 2 and idx.normalize().is_unique


def rth_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Which bars sit inside the regular session, as a bool Series on `index`.

    True for [09:30, 16:00) of each bar's own New York session. Half-open on
    the right because a bar is labeled by its OPEN: the 15:45 bar is the last
    one the session contains, and a bar labeled 16:00 has already crossed into
    the post-market.

    Read off the exchange wall clock rather than by comparing against
    `session_opens`. Converting to EXCHANGE_TZ has already put the DST shift
    where it belongs, so the hour and minute fields answer the question
    directly and correctly on both sides of a transition.

    A frame that already holds one row per session comes back ALL TRUE, and
    that branch is load-bearing rather than a convenience. A session bar's
    label is not a time inside its session at all: legacy providers may label it
    at 00:00Z, while BarCache.upsert canonicalizes cached daily rows to midnight
    America/New_York, expressed as 04:00Z or 05:00Z. All of those labels sit
    outside [09:30, 16:00).
    Without the branch the mask would come back all False and silently blank
    every session-scoped reading on every daily spec. That is a live path and
    not a hypothetical: prev_session_*, gap_pct and vwap all legitimately run
    on 1d frames, and turnaround_tuesday is a shipped 1d catalog play.
    """
    if _is_session_frame(index):
        return pd.Series(True, index=index, dtype=bool)
    ny = index.tz_convert(EXCHANGE_TZ)
    minute_of_day = ny.hour * 60 + ny.minute
    open_at = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_at = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    return pd.Series((minute_of_day >= open_at) & (minute_of_day < close_at),
                     index=index, dtype=bool)


@dataclass(frozen=True)
class TimeframeSet:
    """The engine's time vocabulary: one driving timeframe (the replay
    cadence) plus higher context timeframes. A session-aligned timeframe
    (daily bars keyed by their session date and labeled at midnight
    America/New_York, expressed as 04:00Z or 05:00Z) has no fixed delta;
    visibility is decided by session date, not bar arithmetic."""

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
