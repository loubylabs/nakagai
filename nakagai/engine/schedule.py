"""The embedded schedule, validated once, and every lookup built on it.

The schedule is the replay's clock. Early closes, holidays, and daylight
saving enter core as data on it, and core answers questions about time by
reading it rather than by asking an installed exchange calendar or by doing
`TimeframeSet` arithmetic. That is the whole point of carrying it: a worker's
locally installed calendar package cannot change an accepted request.

`ReplaySchedule` itself only checks the types and nonemptiness of its own
fields, because a value type cannot see the request it belongs to. Everything
that needs two rows compared, or a row compared against the window, lands
here:

- the identity the request agreed to, and the digest recomputed over the body;
- base intervals: strictly increasing, nonoverlapping, dated to their own
  exchange session, with contiguous per-session ordinals;
- context bars: grouped in the fixed timeframe order, strictly increasing
  inside each group, and each one carrying the frozen label semantics for its
  timeframe;
- window boundaries that land exactly on scheduled interval edges.

A schedule that passes becomes a `ValidatedSchedule`, which is the only value
the rest of the replay asks about sessions, availability, and freshness.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

import pandas as pd
from pandas import Timestamp

from nakagai.engine.canonical import schedule_digest
from nakagai.engine.portfolio_types import (
    CONTEXT_TIMEFRAMES,
    PortfolioReplayRequest,
    ReplayInputError,
    ReplaySchedule,
    ScheduledBaseInterval,
    ScheduledContextBar,
    _require_choice,
    _require_instance,
    _require_timestamp,
)

# The three timeframes whose bars are cached at a UTC left edge, derived into
# Eastern wall-clock buckets, or aligned to a session. One entry per supported
# context timeframe, so a new timeframe cannot be added without deciding what
# its label means.
_CONTEXT_SOURCES = {
    "1h": "fetched_left_edge",
    "4h": "derived_1h_et_midnight",
    "1d": "session_aligned",
}


def _refuse(message: str, field: str, **details) -> ReplayInputError:
    return ReplayInputError("invalid_schedule", message, {"field": field, **details})


@dataclass(frozen=True)
class ValidatedSchedule:
    """A schedule that has been checked against one request.

    Built only by `validate_schedule`. Every derived collection is computed
    once, there, so a lookup here is a comparison or a binary search rather
    than a rule being re-decided.
    """

    request: PortfolioReplayRequest
    schedule: ReplaySchedule
    sessions: tuple[date, ...]
    test_intervals: tuple[ScheduledBaseInterval, ...]
    context_index: Mapping[str, tuple[ScheduledContextBar, ...]]

    @property
    def identity(self):
        return self.schedule.identity

    @property
    def base_intervals(self) -> tuple[ScheduledBaseInterval, ...]:
        return self.schedule.base_intervals

    def next_session(self, session_date: date) -> date:
        """The next exchange session after `session_date`, for T+1 settlement.

        The schedule is the calendar, so the session after the last one it
        carries is not a fact core has. Asking for it is a refusal rather than
        a guess: weekends and holidays are exactly what makes guessing wrong.
        """
        position = bisect_right(self.sessions, session_date)
        if position == len(self.sessions):
            raise _refuse(
                "the schedule carries no session after this one", "session_date",
                session_date=session_date.isoformat(),
            )
        return self.sessions[position]

    def context_bars(self, timeframe: str) -> tuple[ScheduledContextBar, ...]:
        return self.context_index.get(
            _require_choice(timeframe, "timeframe", CONTEXT_TIMEFRAMES), (),
        )

    def context_bar(self, timeframe: str, label_ts: Timestamp) -> ScheduledContextBar:
        """The one bar of `timeframe` carrying `label_ts`, or `KeyError`."""
        stamp = _require_timestamp(label_ts, "label_ts")
        for row in self.context_bars(timeframe):
            if row.label_ts == stamp:
                return row
        raise KeyError((timeframe, stamp))

    def closed_base_count(self, at: Timestamp) -> int:
        """How many base intervals have fully closed at `at`.

        Closes strictly increase, so this is the length of the visible prefix
        of any frame whose labels are the scheduled opens.
        """
        return bisect_right(
            self.base_intervals, _require_timestamp(at, "at"),
            key=lambda row: row.close_ts,
        )

    def available_context_count(self, timeframe: str, at: Timestamp) -> int:
        """How many bars of `timeframe` are available at `at`.

        `available_at` rises with `label_ts` inside a timeframe (an hourly and
        a four-hour bar become available at their own period ends, a daily bar
        at the next session's open), so availability is a prefix too.
        """
        return bisect_right(
            self.context_bars(timeframe), _require_timestamp(at, "at"),
            key=lambda row: row.available_at,
        )

    def available_context(
        self, timeframe: str, at: Timestamp,
    ) -> tuple[ScheduledContextBar, ...]:
        return self.context_bars(timeframe)[:self.available_context_count(timeframe, at)]


def validate_schedule(
    request: PortfolioReplayRequest, schedule: ReplaySchedule,
) -> ValidatedSchedule:
    """Check one schedule against one request, or refuse the replay."""
    _require_instance(request, "request", PortfolioReplayRequest)
    _require_instance(schedule, "schedule", ReplaySchedule)
    identity = schedule.identity
    if identity != request.schedule_identity:
        raise _refuse("the schedule carries another identity", "identity")
    recomputed = schedule_digest(schedule)
    if recomputed != identity.schedule_digest:
        raise ReplayInputError(
            "schedule_digest_mismatch",
            "the schedule body does not hash to its own declared digest",
            {"expected": identity.schedule_digest, "actual": recomputed},
        )
    timezone = identity.timezone
    base_step = pd.Timedelta(identity.base_timeframe)
    blocks = _validate_base_intervals(schedule.base_intervals, timezone)
    index = _validate_context_bars(schedule, blocks, timezone, base_step)
    return ValidatedSchedule(
        request=request,
        schedule=schedule,
        sessions=tuple(sorted(blocks)),
        test_intervals=_validate_window(request, schedule),
        context_index=MappingProxyType(index),
    )


# ------------------------------------------------------------ base intervals


def _validate_base_intervals(
    intervals: tuple[ScheduledBaseInterval, ...], timezone: str,
) -> dict[date, list[ScheduledBaseInterval]]:
    """The clock: one strictly increasing, nonoverlapping run of intervals.

    Two spec rules need no check of their own here and are deliberately
    absent rather than forgotten. Unique `(open_ts, close_ts)` pairs follow
    from strictly increasing opens. One contiguous block per session follows
    from that same ordering plus each interval being dated to its own exchange
    session, because converting increasing instants to a timezone cannot make
    a date come back.
    """
    blocks: dict[date, list[ScheduledBaseInterval]] = {}
    previous: ScheduledBaseInterval | None = None
    for row in intervals:
        if previous is not None:
            if row.open_ts <= previous.open_ts:
                raise _refuse(
                    "base intervals open in strictly increasing order", "open_ts",
                    open_ts=row.open_ts.isoformat(),
                )
            if row.close_ts <= previous.close_ts:
                raise _refuse(
                    "base intervals close in strictly increasing order", "close_ts",
                    close_ts=row.close_ts.isoformat(),
                )
            if row.open_ts < previous.close_ts:
                raise _refuse(
                    "base intervals cannot overlap", "interval_overlap",
                    open_ts=row.open_ts.isoformat(),
                )
        if row.session_date != row.open_ts.tz_convert(timezone).date():
            raise _refuse(
                "an interval belongs to the exchange session it opens in",
                "session_date", open_ts=row.open_ts.isoformat(),
                session_date=row.session_date.isoformat(),
            )
        block = blocks.setdefault(row.session_date, [])
        if row.interval_ordinal != len(block):
            raise _refuse(
                "interval ordinals start at zero and are contiguous inside a session",
                "interval_ordinal", open_ts=row.open_ts.isoformat(),
                interval_ordinal=row.interval_ordinal,
            )
        block.append(row)
        previous = row
    return blocks


# -------------------------------------------------------------- context bars


def _validate_context_bars(
    schedule: ReplaySchedule, blocks: dict[date, list[ScheduledBaseInterval]],
    timezone: str, base_step: pd.Timedelta,
) -> dict[str, tuple[ScheduledContextBar, ...]]:
    """Grouping and ordering first, then each timeframe's frozen semantics.

    The two passes are separate on purpose. Ordering is what makes every
    lookup a binary search, so it is established over the whole tuple before
    any single row is asked what its label means.
    """
    index = _group_context_bars(schedule.context_bars)
    sessions = sorted(blocks)
    for rows in index.values():
        for row in rows:
            _require_session_coverage(row, schedule.base_intervals)
            _require_label_semantics(row, blocks, sessions, timezone, base_step,
                                     schedule.base_intervals)
    return index


def _group_context_bars(
    bars: tuple[ScheduledContextBar, ...],
) -> dict[str, tuple[ScheduledContextBar, ...]]:
    grouped: dict[str, list[ScheduledContextBar]] = {}
    order = -1
    for row in bars:
        position = CONTEXT_TIMEFRAMES.index(row.timeframe)
        if position < order:
            raise _refuse(
                "context bars group by timeframe in the fixed order "
                f"{', '.join(CONTEXT_TIMEFRAMES)}", "timeframe",
                timeframe=row.timeframe,
            )
        order = position
        rows = grouped.setdefault(row.timeframe, [])
        if rows:
            _require_context_order(rows[-1], row)
        rows.append(row)
    return {timeframe: tuple(rows) for timeframe, rows in grouped.items()}


def _require_context_order(
    previous: ScheduledContextBar, row: ScheduledContextBar,
) -> None:
    """Labels, period starts, and period ends all rise inside one timeframe.

    A unique `(period_start, period_end)` pair needs no check of its own: two
    bars sharing a period would already have failed the period-start rule.
    """
    for field, earlier, later in (
        ("label_ts", previous.label_ts, row.label_ts),
        ("period_start", previous.period_start, row.period_start),
        ("period_end", previous.period_end, row.period_end),
    ):
        if later <= earlier:
            raise _refuse(
                f"{field} rises strictly inside one timeframe", field,
                timeframe=row.timeframe, label_ts=row.label_ts.isoformat(),
            )


def _require_session_coverage(
    row: ScheduledContextBar, intervals: tuple[ScheduledBaseInterval, ...],
) -> None:
    """Every scheduled interval the bar's period covers is its own session."""
    covered = False
    start = bisect_right(intervals, row.period_start, key=lambda item: item.close_ts)
    for interval in intervals[start:]:
        if interval.open_ts >= row.period_end:
            break
        covered = True
        if interval.session_date != row.session_date:
            raise _refuse(
                "a context bar covers only the session it names", "session_date",
                timeframe=row.timeframe, label_ts=row.label_ts.isoformat(),
                session_date=row.session_date.isoformat(),
            )
    if not covered:
        raise _refuse(
            "a context bar covers at least one scheduled base interval",
            "session_coverage", timeframe=row.timeframe,
            label_ts=row.label_ts.isoformat(),
        )


def _require_label_semantics(
    row: ScheduledContextBar, blocks: dict[date, list[ScheduledBaseInterval]],
    sessions: list[date], timezone: str, base_step: pd.Timedelta,
    intervals: tuple[ScheduledBaseInterval, ...],
) -> None:
    """The frozen label contract, one branch per supported timeframe."""
    if row.source != _CONTEXT_SOURCES[row.timeframe]:
        raise _refuse(
            "a context bar names the source its timeframe comes from", "source",
            timeframe=row.timeframe, source=row.source,
        )
    if row.timeframe == "1d":
        _require_daily_semantics(row, blocks, sessions, timezone)
        return
    if row.timeframe == "1h":
        # The cached label is a UTC left edge, so the check is a UTC one.
        aligned = (row.label_ts.minute, row.label_ts.second,
                   row.label_ts.microsecond) == (0, 0, 0)
    else:
        # A four-hour bucket is anchored at Eastern local midnight, so 12:00
        # in New York is one bucket edge in January and in July even though
        # the two are different UTC hours.
        local = row.label_ts.tz_convert(timezone)
        aligned = local.hour % 4 == 0 and (
            local.minute, local.second, local.microsecond) == (0, 0, 0)
    if not aligned:
        raise _refuse(
            "a context label sits on its timeframe's own anchor", "label_anchor",
            timeframe=row.timeframe, label_ts=row.label_ts.isoformat(),
        )
    if row.period_start != row.label_ts:
        raise _refuse(
            "an intraday context bar is labeled at its own left edge", "period_start",
            timeframe=row.timeframe, label_ts=row.label_ts.isoformat(),
        )
    _require_period_length(row, timezone)
    if row.available_at != row.period_end:
        raise _refuse(
            "an intraday context bar becomes available at its period end",
            "available_at", timeframe=row.timeframe,
            label_ts=row.label_ts.isoformat(),
        )
    _require_emission_freshness(row, intervals, base_step)


def _require_period_length(row: ScheduledContextBar, timezone: str) -> None:
    """One hour of absolute time, or four hours of Eastern wall clock.

    The four-hour rule reads the wall clock through local time rather than
    through the instants, which is the same reason the derived bars are
    resampled that way: a bucket that spans a transition is three or five
    absolute hours and is still four hours on the exchange's clock.
    """
    if row.timeframe == "1h":
        actual = row.period_end - row.period_start
    else:
        actual = (row.period_end.tz_convert(timezone).tz_localize(None)
                  - row.period_start.tz_convert(timezone).tz_localize(None))
    if actual != pd.Timedelta(row.timeframe):
        raise _refuse(
            "a context period is exactly one bar of its own timeframe",
            "period_length", timeframe=row.timeframe,
            label_ts=row.label_ts.isoformat(),
        )


def _require_emission_freshness(
    row: ScheduledContextBar, intervals: tuple[ScheduledBaseInterval, ...],
    base_step: pd.Timedelta,
) -> None:
    """Fresh at the scheduled base close inside `[period_end, +one base bar)`.

    A bar whose period ends after the last scheduled close of its session,
    which is what an early close does to the noon four-hour bucket, has no
    such interval and therefore no freshness at all.
    """
    limit = row.period_end + base_step
    position = bisect_left(intervals, row.period_end, key=lambda item: item.close_ts)
    expected = None
    if position < len(intervals) and intervals[position].close_ts < limit:
        expected = intervals[position].close_ts
    if row.fresh_context_at != expected:
        raise _refuse(
            "a context bar is fresh at the base close that follows its period",
            "fresh_context_at", timeframe=row.timeframe,
            label_ts=row.label_ts.isoformat(),
        )


def _require_daily_semantics(
    row: ScheduledContextBar, blocks: dict[date, list[ScheduledBaseInterval]],
    sessions: list[date], timezone: str,
) -> None:
    """Labeled at Eastern midnight, spanning its session, released next open."""
    local = row.label_ts.tz_convert(timezone)
    if local.date() != row.session_date or (
        local.hour, local.minute, local.second, local.microsecond
    ) != (0, 0, 0, 0):
        raise _refuse(
            "a daily bar is labeled at midnight in the exchange's timezone",
            "label_anchor", label_ts=row.label_ts.isoformat(),
            session_date=row.session_date.isoformat(),
        )
    block = blocks[row.session_date]
    if (row.period_start, row.period_end) != (block[0].open_ts, block[-1].close_ts):
        raise _refuse(
            "a daily period runs from its session's first open to its last close",
            "period_span", session_date=row.session_date.isoformat(),
        )
    position = bisect_right(sessions, row.session_date)
    if position == len(sessions):
        raise _refuse(
            "a daily bar becomes available at the next scheduled session open, "
            "and this schedule has no next session", "available_at",
            session_date=row.session_date.isoformat(),
        )
    following = blocks[sessions[position]]
    if row.available_at != following[0].open_ts:
        raise _refuse(
            "a daily bar becomes available at the next scheduled session open",
            "available_at", session_date=row.session_date.isoformat(),
        )
    if row.fresh_context_at != following[0].close_ts:
        raise _refuse(
            "a daily bar is fresh at the next session's first base close",
            "fresh_context_at", session_date=row.session_date.isoformat(),
        )


# --------------------------------------------------------- window boundaries


def _validate_window(
    request: PortfolioReplayRequest, schedule: ReplaySchedule,
) -> tuple[ScheduledBaseInterval, ...]:
    """Both window edges and the IC tail land on scheduled interval edges.

    The schedule is materialized from `train_start` through `ic_tail_end`, so
    those two are the schedule's own first open and last close rather than
    merely some open and some close inside it.

    "At least one test interval" needs no check: `test_start` is an interval
    open and `test_start < test_end`, so that interval is always one.
    """
    window = request.window
    intervals = schedule.base_intervals
    if intervals[0].open_ts != window.train_start:
        raise _refuse(
            "the schedule begins at the window's train start", "train_start",
            expected=window.train_start.isoformat(),
            actual=intervals[0].open_ts.isoformat(),
        )
    if intervals[-1].close_ts != request.ic_tail_end:
        raise _refuse(
            "the schedule ends at the request's IC tail", "ic_tail_end",
            expected=request.ic_tail_end.isoformat(),
            actual=intervals[-1].close_ts.isoformat(),
        )
    if not any(row.open_ts == window.test_start for row in intervals):
        raise _refuse(
            "the test range begins at a scheduled interval open", "test_start",
            test_start=window.test_start.isoformat(),
        )
    if not any(row.close_ts == window.test_end for row in intervals):
        raise _refuse(
            "the test range ends at a scheduled interval close", "test_end",
            test_end=window.test_end.isoformat(),
        )
    return tuple(row for row in intervals
                 if window.test_start <= row.open_ts < window.test_end)
