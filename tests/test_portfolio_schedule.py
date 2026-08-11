"""The embedded schedule: goldens for its labels and every refusal.

The schedule is how early closes, holidays, and daylight saving enter core as
data. Core never asks an installed calendar what a boundary is, so these tests
are the place the frozen label semantics are pinned: literal UTC instants for
one full session, one early close, one absent holiday, and both 2026 daylight
saving transitions.

Every refusal fixture rebuilds the schedule through `schedule_with`, which
recomputes the digest. Editing a built schedule in place would leave the old
digest behind, the digest check would fire first, and each test would pass
without ever reaching the rule it names.
"""

import dataclasses
from datetime import date

import pandas as pd
import pytest

from nakagai.engine.portfolio_types import (
    ReplayInputError,
    ReplaySchedule,
    ReplayWindow,
    ScheduledBaseInterval,
    ScheduledContextBar,
)
from nakagai.engine.canonical import schedule_digest
from nakagai.engine.schedule import ValidatedSchedule, validate_schedule
from tests.portfolio_fixtures import (
    FALL_SESSION_ONE,
    FALL_SESSION_TWO,
    FULL_SESSION_INTERVALS,
    SESSION_ONE,
    SESSION_ONE_INTERVALS,
    SESSION_TWO,
    SESSION_TWO_INTERVALS,
    SPRING_SESSION_ONE,
    SPRING_SESSION_TWO,
    base_context_bars,
    base_identity,
    base_intervals,
    base_request,
    base_schedule,
    fall_request,
    fall_schedule,
    schedule_with,
    spring_request,
    spring_schedule,
    ts,
)

EXCHANGE_TZ = "America/New_York"
HOLIDAY = date(2026, 11, 26)


def validated() -> ValidatedSchedule:
    return validate_schedule(base_request(), base_schedule())


def refuse(schedule: ReplaySchedule, **overrides) -> ReplayInputError:
    """Validate a mutated schedule against a request that accepts its identity."""
    request = base_request(schedule_identity=schedule.identity, **overrides)
    with pytest.raises(ReplayInputError) as raised:
        validate_schedule(request, schedule)
    return raised.value


def intervals_with(index: int, **changes) -> tuple[ScheduledBaseInterval, ...]:
    rows = list(base_intervals())
    rows[index] = dataclasses.replace(rows[index], **changes)
    return tuple(rows)


def context_with(index: int, **changes) -> tuple[ScheduledContextBar, ...]:
    rows = list(base_context_bars())
    rows[index] = dataclasses.replace(rows[index], **changes)
    return tuple(rows)


def local(stamp: pd.Timestamp) -> pd.Timestamp:
    return stamp.tz_convert(EXCHANGE_TZ)


# ------------------------------------------------------- sessions and clock


def test_the_base_schedule_holds_one_full_session_and_one_early_close():
    schedule = validated()
    assert schedule.sessions == (SESSION_ONE, SESSION_TWO)
    assert len(schedule.base_intervals) == SESSION_ONE_INTERVALS + SESSION_TWO_INTERVALS
    assert schedule.base_intervals[0].open_ts == ts("2026-11-25T14:30:00Z")
    assert schedule.base_intervals[SESSION_ONE_INTERVALS - 1].close_ts == ts(
        "2026-11-25T21:00:00Z")


def test_the_early_close_stops_three_and_a_half_hours_after_the_bell():
    schedule = validated()
    half_day = [row for row in schedule.base_intervals
                if row.session_date == SESSION_TWO]
    assert len(half_day) == SESSION_TWO_INTERVALS
    assert half_day[0].open_ts == ts("2026-11-27T14:30:00Z")
    assert half_day[-1].close_ts == ts("2026-11-27T18:00:00Z")
    # 13:00 in New York, which no UTC hour comparison would have found on its
    # own: the same wall clock is 17:00Z in summer.
    assert local(half_day[-1].close_ts).strftime("%H:%M") == "13:00"


def test_the_holiday_is_absent_and_the_next_session_skips_it():
    schedule = validated()
    assert HOLIDAY not in schedule.sessions
    assert schedule.next_session(SESSION_ONE) == SESSION_TWO


def test_the_last_session_has_no_next_session():
    with pytest.raises(ReplayInputError) as raised:
        validated().next_session(SESSION_TWO)
    assert raised.value.code == "invalid_schedule"
    assert raised.value.details["field"] == "session_date"


def test_test_intervals_are_exactly_the_opens_inside_the_test_range():
    schedule = validated()
    window = schedule.request.window
    assert len(schedule.test_intervals) == SESSION_TWO_INTERVALS
    assert schedule.test_intervals[0].open_ts == window.test_start
    assert schedule.test_intervals[-1].close_ts == window.test_end
    assert all(window.test_start <= row.open_ts < window.test_end
               for row in schedule.test_intervals)


def test_closed_base_count_counts_only_intervals_already_closed():
    schedule = validated()
    assert schedule.closed_base_count(ts("2026-11-25T14:44:59Z")) == 0
    assert schedule.closed_base_count(ts("2026-11-25T14:45:00Z")) == 1
    # The whole warmup session has closed before the first test open, and the
    # absent holiday adds nothing between the two.
    assert schedule.closed_base_count(ts("2026-11-27T14:30:00Z")) == 26
    assert schedule.closed_base_count(ts("2026-11-27T18:00:00Z")) == 40


# ----------------------------------------------------- frozen label goldens


def test_the_hourly_bar_keeps_its_cached_utc_left_edge_label():
    bar = validated().context_bar("1h", ts("2026-11-27T14:00:00Z"))
    assert bar.source == "fetched_left_edge"
    assert bar.period_start == ts("2026-11-27T14:00:00Z")
    assert bar.period_end == ts("2026-11-27T15:00:00Z")
    assert bar.available_at == ts("2026-11-27T15:00:00Z")
    # The base close that lands on the period end, not the period end itself.
    assert bar.fresh_context_at == ts("2026-11-27T15:00:00Z")


def test_the_four_hour_bucket_is_anchored_at_new_york_midnight():
    bar = validated().context_bar("4h", ts("2026-11-27T17:00:00Z"))
    assert bar.source == "derived_1h_et_midnight"
    assert local(bar.label_ts).strftime("%H:%M") == "12:00"
    assert bar.period_end == ts("2026-11-27T21:00:00Z")
    assert local(bar.period_end).strftime("%H:%M") == "16:00"
    # Its period ends three hours after the half day does, so no scheduled
    # base close falls in [period_end, period_end + 15m) and it is never fresh.
    assert bar.fresh_context_at is None


def test_the_daily_bar_is_labeled_at_new_york_midnight_of_its_session():
    bar = validated().context_bar("1d", ts("2026-11-25T05:00:00Z"))
    assert bar.source == "session_aligned"
    assert bar.session_date == SESSION_ONE
    assert local(bar.label_ts) == pd.Timestamp("2026-11-25 00:00", tz=EXCHANGE_TZ)
    assert (bar.period_start, bar.period_end) == (
        ts("2026-11-25T14:30:00Z"), ts("2026-11-25T21:00:00Z"))
    # Available at the NEXT scheduled session open, which is two calendar days
    # later because the holiday sits between them.
    assert bar.available_at == ts("2026-11-27T14:30:00Z")
    assert bar.fresh_context_at == ts("2026-11-27T14:45:00Z")


def test_the_final_session_carries_no_daily_bar():
    assert [row.session_date for row in validated().context_bars("1d")] == [SESSION_ONE]


# ------------------------------------------------------------ availability


@pytest.mark.parametrize(
    ("timeframe", "at", "expected"),
    [
        ("1h", "2026-11-25T14:59:59Z", 0),
        ("1h", "2026-11-25T15:00:00Z", 1),
        ("1h", "2026-11-27T15:00:00Z", 2),
        # The four-hour bucket only becomes available after the schedule ends.
        ("4h", "2026-11-27T18:00:00Z", 0),
        ("4h", "2026-11-27T21:00:00Z", 1),
        ("1d", "2026-11-27T14:29:59Z", 0),
        ("1d", "2026-11-27T14:30:00Z", 1),
    ],
)
def test_context_availability_is_an_exact_lookup(timeframe, at, expected):
    schedule = validated()
    assert schedule.available_context_count(timeframe, ts(at)) == expected
    assert len(schedule.available_context(timeframe, ts(at))) == expected


def test_an_unknown_context_label_has_no_bar():
    with pytest.raises(KeyError):
        validated().context_bar("1h", ts("2026-11-27T16:00:00Z"))


# --------------------------------------------------- daylight saving goldens


def test_spring_forward_moves_every_boundary_an_hour_earlier_in_utc():
    schedule = validate_schedule(spring_request(), spring_schedule())
    assert schedule.sessions == (SPRING_SESSION_ONE, SPRING_SESSION_TWO)
    before, after = schedule.base_intervals[0], schedule.base_intervals[
        FULL_SESSION_INTERVALS]
    assert (before.open_ts, after.open_ts) == (
        ts("2026-03-06T14:30:00Z"), ts("2026-03-09T13:30:00Z"))
    # One bell, two UTC hours. This is the whole point of carrying the
    # schedule as data.
    assert local(before.open_ts).strftime("%H:%M") == "09:30"
    assert local(after.open_ts).strftime("%H:%M") == "09:30"


def test_spring_forward_moves_the_four_hour_bucket_and_the_daily_label():
    schedule = validate_schedule(spring_request(), spring_schedule())
    buckets = schedule.context_bars("4h")
    assert [row.label_ts for row in buckets] == [
        ts("2026-03-06T17:00:00Z"), ts("2026-03-09T16:00:00Z")]
    assert [local(row.label_ts).strftime("%H:%M") for row in buckets] == [
        "12:00", "12:00"]
    assert [row.period_end for row in buckets] == [
        ts("2026-03-06T21:00:00Z"), ts("2026-03-09T20:00:00Z")]
    daily = schedule.context_bars("1d")[0]
    assert daily.label_ts == ts("2026-03-06T05:00:00Z")
    # An Eastern-standard daily bar becoming available at a daylight-time open.
    assert daily.available_at == ts("2026-03-09T13:30:00Z")
    assert daily.fresh_context_at == ts("2026-03-09T13:45:00Z")


def test_fall_back_moves_every_boundary_an_hour_later_in_utc():
    schedule = validate_schedule(fall_request(), fall_schedule())
    assert schedule.sessions == (FALL_SESSION_ONE, FALL_SESSION_TWO)
    before, after = schedule.base_intervals[0], schedule.base_intervals[
        FULL_SESSION_INTERVALS]
    assert (before.open_ts, after.open_ts) == (
        ts("2026-10-30T13:30:00Z"), ts("2026-11-02T14:30:00Z"))
    assert local(before.open_ts).strftime("%H:%M") == "09:30"
    assert local(after.open_ts).strftime("%H:%M") == "09:30"


def test_fall_back_moves_the_four_hour_bucket_and_the_daily_label():
    schedule = validate_schedule(fall_request(), fall_schedule())
    buckets = schedule.context_bars("4h")
    assert [row.label_ts for row in buckets] == [
        ts("2026-10-30T16:00:00Z"), ts("2026-11-02T17:00:00Z")]
    assert [local(row.label_ts).strftime("%H:%M") for row in buckets] == [
        "12:00", "12:00"]
    daily = schedule.context_bars("1d")[0]
    assert daily.label_ts == ts("2026-10-30T04:00:00Z")
    assert local(daily.label_ts) == pd.Timestamp("2026-10-30 00:00", tz=EXCHANGE_TZ)
    assert daily.available_at == ts("2026-11-02T14:30:00Z")


# ------------------------------------------------------- identity and digest


def test_a_schedule_digest_that_does_not_match_its_own_body_refuses():
    honest = base_schedule()
    tampered = ReplaySchedule(
        identity=base_identity("9f" * 32),
        base_intervals=honest.base_intervals,
        context_bars=honest.context_bars,
    )
    request = base_request(schedule_identity=tampered.identity)
    with pytest.raises(ReplayInputError) as raised:
        validate_schedule(request, tampered)
    assert raised.value.code == "schedule_digest_mismatch"


def test_a_schedule_carrying_another_identity_refuses():
    schedule = base_schedule()
    other = dataclasses.replace(
        schedule.identity, calendar_version="exchange_calendars:9.9.9:nakagai-rth-v1")
    request = base_request(schedule_identity=other)
    with pytest.raises(ReplayInputError) as raised:
        validate_schedule(request, schedule)
    assert (raised.value.code, raised.value.details["field"]) == (
        "invalid_schedule", "identity")


@pytest.mark.parametrize("value", [None, "schedule", base_intervals()])
def test_validate_schedule_requires_the_declared_types(value):
    with pytest.raises(ReplayInputError) as raised:
        validate_schedule(base_request(), value)
    assert raised.value.code == "invalid_type"


# ---------------------------------------------------- base interval refusals


def test_base_intervals_out_of_order_refuse():
    # The sixth interval opens back inside the fifth one's hour.
    error = refuse(schedule_with(
        intervals=intervals_with(5, open_ts=ts("2026-11-25T15:00:00Z"))))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "open_ts")


def test_a_repeated_base_interval_refuses():
    rows = list(base_intervals())
    rows[1] = dataclasses.replace(rows[0], interval_ordinal=1)
    error = refuse(schedule_with(intervals=tuple(rows)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "open_ts")


def test_base_intervals_that_close_out_of_order_refuse():
    error = refuse(schedule_with(
        intervals=intervals_with(0, close_ts=ts("2026-11-25T16:00:00Z"))))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "close_ts")


def test_overlapping_base_intervals_refuse():
    error = refuse(schedule_with(
        intervals=intervals_with(1, open_ts=ts("2026-11-25T14:40:00Z"))))
    assert (error.code, error.details["field"]) == (
        "invalid_schedule", "interval_overlap")


def test_an_interval_dated_to_another_session_refuses():
    error = refuse(schedule_with(intervals=intervals_with(0, session_date=HOLIDAY)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "session_date")


@pytest.mark.parametrize("ordinal", [1, 5])
def test_interval_ordinals_that_do_not_start_at_zero_refuse(ordinal):
    error = refuse(schedule_with(intervals=intervals_with(0, interval_ordinal=ordinal)))
    assert (error.code, error.details["field"]) == (
        "invalid_schedule", "interval_ordinal")


def test_a_gap_in_the_interval_ordinals_refuses():
    error = refuse(schedule_with(intervals=intervals_with(3, interval_ordinal=4)))
    assert (error.code, error.details["field"]) == (
        "invalid_schedule", "interval_ordinal")


# ------------------------------------------------------ context bar refusals


def test_context_bars_out_of_timeframe_order_refuse():
    rows = list(base_context_bars())
    rows[2], rows[3] = rows[3], rows[2]
    error = refuse(schedule_with(context_bars=tuple(rows)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "timeframe")


def test_a_timeframe_split_across_two_groups_refuses():
    rows = list(base_context_bars())
    rows.append(dataclasses.replace(rows[0], label_ts=ts("2026-11-25T16:00:00Z"),
                                    period_start=ts("2026-11-25T16:00:00Z"),
                                    period_end=ts("2026-11-25T17:00:00Z"),
                                    available_at=ts("2026-11-25T17:00:00Z"),
                                    fresh_context_at=ts("2026-11-25T17:00:00Z")))
    error = refuse(schedule_with(context_bars=tuple(rows)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "timeframe")


def test_a_repeated_context_label_refuses():
    rows = list(base_context_bars())
    rows.insert(1, rows[1])
    error = refuse(schedule_with(context_bars=tuple(rows)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "label_ts")


def test_a_repeated_context_period_refuses():
    error = refuse(schedule_with(context_bars=context_with(
        1,
        label_ts=ts("2026-11-25T16:00:00Z"),
        period_start=ts("2026-11-25T14:00:00Z"),
        period_end=ts("2026-11-25T15:00:00Z"),
    )))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "period_start")


def test_context_periods_that_end_out_of_order_refuse():
    error = refuse(schedule_with(context_bars=context_with(
        0,
        period_end=ts("2026-11-27T20:00:00Z"),
        available_at=ts("2026-11-27T20:00:00Z"),
        fresh_context_at=ts("2026-11-27T20:00:00Z"),
    )))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "period_end")


def test_valid_cross_timeframe_label_and_boundary_overlap_passes():
    # One hourly bar and one four-hour bucket that share a label and a period
    # start. Different timeframes may collide exactly like this.
    rows = list(base_context_bars())
    rows.insert(2, ScheduledContextBar(
        timeframe="1h", session_date=SESSION_TWO,
        label_ts=ts("2026-11-27T17:00:00Z"),
        period_start=ts("2026-11-27T17:00:00Z"),
        period_end=ts("2026-11-27T18:00:00Z"),
        available_at=ts("2026-11-27T18:00:00Z"),
        fresh_context_at=ts("2026-11-27T18:00:00Z"),
        source="fetched_left_edge",
    ))
    schedule = schedule_with(context_bars=tuple(rows))
    validated_schedule = validate_schedule(
        base_request(schedule_identity=schedule.identity), schedule)
    shared = [row.label_ts for row in validated_schedule.schedule.context_bars
              if row.label_ts == ts("2026-11-27T17:00:00Z")]
    assert len(shared) == 2


@pytest.mark.parametrize(
    ("index", "changes", "field"),
    [
        # An hourly label that is not a UTC left edge.
        (1, {"label_ts": "2026-11-27T14:30:00Z", "period_start": "2026-11-27T14:30:00Z",
             "period_end": "2026-11-27T15:30:00Z", "available_at": "2026-11-27T15:30:00Z",
             "fresh_context_at": "2026-11-27T15:30:00Z"}, "label_anchor"),
        # An hourly period that is not one hour long.
        (1, {"period_end": "2026-11-27T16:00:00Z", "available_at": "2026-11-27T16:00:00Z",
             "fresh_context_at": "2026-11-27T16:00:00Z"}, "period_length"),
        # An hourly bar whose label and period start disagree.
        (1, {"label_ts": "2026-11-27T13:00:00Z"}, "period_start"),
        # Availability later than the period end.
        (1, {"available_at": "2026-11-27T15:15:00Z",
             "fresh_context_at": "2026-11-27T15:15:00Z"}, "available_at"),
        # Freshness that is not the base close inside the emission window.
        (1, {"fresh_context_at": "2026-11-27T15:15:00Z"}, "fresh_context_at"),
        # A four-hour bucket that is not anchored at Eastern midnight: 16:00Z
        # is 11:00 in New York, one hour off every bucket edge.
        (2, {"label_ts": "2026-11-27T16:00:00Z", "period_start": "2026-11-27T16:00:00Z",
             "period_end": "2026-11-27T20:00:00Z",
             "available_at": "2026-11-27T20:00:00Z"}, "label_anchor"),
        # A four-hour bucket whose period is not four wall-clock hours.
        (2, {"period_end": "2026-11-27T22:00:00Z",
             "available_at": "2026-11-27T22:00:00Z"}, "period_length"),
        # A bucket that no scheduled base close follows cannot be fresh at all.
        (2, {"fresh_context_at": "2026-11-27T21:00:00Z"}, "fresh_context_at"),
        # A daily bar labeled somewhere other than Eastern midnight.
        (3, {"label_ts": "2026-11-25T00:00:00Z"}, "label_anchor"),
        # A daily period that is not the session's own first open to last close.
        (3, {"period_start": "2026-11-25T14:45:00Z"}, "period_span"),
        (3, {"period_end": "2026-11-25T20:45:00Z"}, "period_span"),
        # A daily bar available at something other than the next session open.
        (3, {"available_at": "2026-11-27T14:45:00Z"}, "available_at"),
        # A daily bar fresh at something other than that session's first close.
        (3, {"fresh_context_at": "2026-11-27T15:00:00Z"}, "fresh_context_at"),
    ],
)
def test_context_label_semantics_refuse(index, changes, field):
    replacement = {name: (None if value is None else ts(value))
                   for name, value in changes.items()}
    error = refuse(schedule_with(context_bars=context_with(index, **replacement)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", field)


@pytest.mark.parametrize(
    ("index", "source"),
    [(1, "session_aligned"), (2, "fetched_left_edge"), (3, "derived_1h_et_midnight")],
)
def test_a_context_bar_carrying_the_wrong_source_refuses(index, source):
    error = refuse(schedule_with(context_bars=context_with(index, source=source)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "source")


def test_a_context_bar_naming_another_session_refuses():
    error = refuse(schedule_with(context_bars=context_with(1, session_date=SESSION_ONE)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "session_date")


def test_a_context_bar_covering_no_scheduled_interval_refuses():
    rows = list(base_context_bars())
    rows[0] = dataclasses.replace(
        rows[0],
        label_ts=ts("2026-11-25T09:00:00Z"),
        period_start=ts("2026-11-25T09:00:00Z"),
        period_end=ts("2026-11-25T10:00:00Z"),
        available_at=ts("2026-11-25T10:00:00Z"),
        fresh_context_at=None,
    )
    error = refuse(schedule_with(context_bars=tuple(rows)))
    assert (error.code, error.details["field"]) == (
        "invalid_schedule", "session_coverage")


def test_a_daily_bar_with_no_following_session_refuses():
    rows = list(base_context_bars())
    rows.append(ScheduledContextBar(
        timeframe="1d", session_date=SESSION_TWO,
        label_ts=ts("2026-11-27T05:00:00Z"),
        period_start=ts("2026-11-27T14:30:00Z"),
        period_end=ts("2026-11-27T18:00:00Z"),
        available_at=ts("2026-11-27T18:00:00Z"),
        fresh_context_at=None,
        source="session_aligned",
    ))
    error = refuse(schedule_with(context_bars=tuple(rows)))
    assert (error.code, error.details["field"]) == ("invalid_schedule", "available_at")


def transition_schedule(period_end: str) -> ReplaySchedule:
    """A session inside the four-hour bucket that spans the spring transition.

    Contrived on purpose: no exchange trades at 01:00 on the Sunday the clocks
    move, so this is the only way to put the wall-clock rule under a test that
    can fail. The 00:00 Eastern bucket of 2026-03-08 runs to 04:00 Eastern,
    which is 05:00Z to 08:00Z, three hours of absolute time and four hours of
    the clock on the wall.
    """
    session = date(2026, 3, 8)
    intervals = tuple(
        ScheduledBaseInterval(
            session_date=session, interval_ordinal=ordinal,
            open_ts=ts("2026-03-08T06:00:00Z") + pd.Timedelta(minutes=15 * ordinal),
            close_ts=ts("2026-03-08T06:15:00Z") + pd.Timedelta(minutes=15 * ordinal),
        )
        for ordinal in range(2)
    )
    draft = ReplaySchedule(
        identity=base_identity(),
        base_intervals=intervals,
        context_bars=(ScheduledContextBar(
            timeframe="4h", session_date=session,
            label_ts=ts("2026-03-08T05:00:00Z"),
            period_start=ts("2026-03-08T05:00:00Z"),
            period_end=ts(period_end),
            available_at=ts(period_end),
            fresh_context_at=None,
            source="derived_1h_et_midnight",
        ),),
    )
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def transition_request(schedule: ReplaySchedule):
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-03-08T06:00:00Z"),
            train_end=ts("2026-03-08T06:15:00Z"),
            test_start=ts("2026-03-08T06:15:00Z"),
            test_end=ts("2026-03-08T06:30:00Z"),
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=ts("2026-03-08T06:30:00Z"),
    )


def test_a_four_hour_period_is_four_hours_of_wall_clock_not_of_elapsed_time():
    schedule = transition_schedule("2026-03-08T08:00:00Z")
    validated_schedule = validate_schedule(transition_request(schedule), schedule)
    bucket = validated_schedule.context_bars("4h")[0]
    assert bucket.period_end - bucket.period_start == pd.Timedelta(hours=3)


def test_a_four_hour_period_of_four_elapsed_hours_across_the_transition_refuses():
    schedule = transition_schedule("2026-03-08T09:00:00Z")
    with pytest.raises(ReplayInputError) as raised:
        validate_schedule(transition_request(schedule), schedule)
    assert (raised.value.code, raised.value.details["field"]) == (
        "invalid_schedule", "period_length")


def test_a_schedule_with_no_context_bars_at_all_is_valid():
    schedule = schedule_with(context_bars=())
    request = base_request(schedule_identity=schedule.identity)
    assert validate_schedule(request, schedule).context_bars("1h") == ()


# ----------------------------------------------------- window boundary rules


@pytest.mark.parametrize(
    ("window", "ic_tail_end", "field"),
    [
        (("2026-11-25T14:45:00Z", "2026-11-27T14:30:00Z", "2026-11-27T14:30:00Z",
          "2026-11-27T18:00:00Z"), "2026-11-27T18:00:00Z", "train_start"),
        (("2026-11-25T14:30:00Z", "2026-11-27T14:35:00Z", "2026-11-27T14:35:00Z",
          "2026-11-27T18:00:00Z"), "2026-11-27T18:00:00Z", "test_start"),
        (("2026-11-25T14:30:00Z", "2026-11-27T14:30:00Z", "2026-11-27T14:30:00Z",
          "2026-11-27T17:55:00Z"), "2026-11-27T18:00:00Z", "test_end"),
        (("2026-11-25T14:30:00Z", "2026-11-27T14:30:00Z", "2026-11-27T14:30:00Z",
          "2026-11-27T17:00:00Z"), "2026-11-27T17:00:00Z", "ic_tail_end"),
    ],
)
def test_a_window_boundary_off_the_schedule_refuses(window, ic_tail_end, field):
    train_start, train_end, test_start, test_end = (ts(value) for value in window)
    error = refuse(
        base_schedule(),
        window=ReplayWindow(train_start=train_start, train_end=train_end,
                            test_start=test_start, test_end=test_end),
        ic_tail_end=ts(ic_tail_end),
    )
    assert (error.code, error.details["field"]) == ("invalid_schedule", field)
