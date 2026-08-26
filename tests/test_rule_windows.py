import math
from datetime import time

import pandas as pd
import pytest

from nakagai.strategies.rules.windows import (
    PRIOR_DAY,
    WindowSpec,
    aggregate_window,
    window_duration,
)


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def series(rows: list[tuple[str, float]]) -> pd.Series:
    return pd.Series(
        [value for _, value in rows],
        index=pd.DatetimeIndex([utc(stamp) for stamp, _ in rows], name="ts"),
    )


NY_OPEN_30 = WindowSpec(
    "ny_open_30",
    "America/New_York",
    time(9, 30),
    time(10),
    "xnys_session",
    "standard",
)
WEEKDAY_OPEN_30 = WindowSpec(
    "weekday_open_30",
    "America/New_York",
    time(9, 30),
    time(10),
    "weekday",
    "standard",
)
NY_PM = WindowSpec(
    "ny_pm",
    "America/New_York",
    time(12),
    time(16),
    "xnys_session",
    "standard",
)
LONDON = WindowSpec(
    "london",
    "Europe/London",
    time(8),
    time(16, 30),
    "weekday",
    "low_iex",
)
ASIA = WindowSpec(
    "asia",
    "America/New_York",
    time(20),
    time(4),
    "weekday",
    "low_iex",
)
PRIOR_WEEK = WindowSpec(
    "prior_week",
    "America/New_York",
    time(9, 30),
    time(16),
    "prior_iso_week",
    "standard",
)
PRIOR_MONTH = WindowSpec(
    "prior_month",
    "America/New_York",
    time(9, 30),
    time(16),
    "prior_calendar_month",
    "standard",
)


def test_window_duration_uses_wall_clock_time_and_wraps_overnight():
    assert window_duration(NY_OPEN_30) == pd.Timedelta(minutes=30)
    assert window_duration(ASIA) == pd.Timedelta(hours=8)


@pytest.mark.parametrize(
    ("reducer", "want"),
    [("max", 103.0), ("min", 101.0), ("first", 102.0), ("last", 103.0)],
)
def test_current_window_reducers_use_only_half_open_occurrence_rows(reducer, want):
    source = series([
        ("2026-01-06 14:30", 102.0),
        ("2026-01-06 14:45", 101.0),
        ("2026-01-06 14:50", 103.0),
        ("2026-01-06 15:00", 999.0),
    ])

    got = aggregate_window(source, NY_OPEN_30, reducer)

    assert got.loc[utc("2026-01-06 15:00")] == want


def test_current_window_is_nan_until_the_close_bar():
    highs = series([
        ("2026-01-06 14:30", 100.0),
        ("2026-01-06 14:45", 103.0),
        ("2026-01-06 15:00", 101.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert math.isnan(got.loc[utc("2026-01-06 14:45")])
    assert got.loc[utc("2026-01-06 15:00")] == 103.0


def test_next_open_clears_the_carried_value():
    highs = series([
        ("2026-01-06 14:30", 100.0),
        ("2026-01-06 14:45", 103.0),
        ("2026-01-06 15:00", 101.0),
        ("2026-01-07 14:15", 99.0),
        ("2026-01-07 14:30", 98.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert got.loc[utc("2026-01-07 14:15")] == 103.0
    assert math.isnan(got.loc[utc("2026-01-07 14:30")])


def test_a_gap_crossing_the_next_open_clears_without_an_open_bar():
    highs = series([
        ("2026-01-06 14:30", 100.0),
        ("2026-01-06 15:00", 99.0),
        ("2026-01-07 14:45", 101.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert got.loc[utc("2026-01-06 15:00")] == 100.0
    assert math.isnan(got.loc[utc("2026-01-07 14:45")])


def test_a_gap_crossing_the_close_reveals_the_completed_value():
    highs = series([
        ("2026-01-06 14:30", 100.0),
        ("2026-01-06 14:45", 104.0),
        ("2026-01-06 15:15", 99.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert got.loc[utc("2026-01-06 15:15")] == 104.0


def test_an_occurrence_with_no_usable_value_replaces_the_older_value_with_nan():
    highs = series([
        ("2026-01-06 14:30", 103.0),
        ("2026-01-06 15:00", 99.0),
        ("2026-01-07 14:30", math.nan),
        ("2026-01-07 15:00", 98.0),
        ("2026-01-07 16:00", 97.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert got.loc[utc("2026-01-06 15:00")] == 103.0
    assert math.isnan(got.loc[utc("2026-01-07 15:00")])
    assert math.isnan(got.loc[utc("2026-01-07 16:00")])


def test_weekday_value_carries_over_the_weekend_until_monday_open():
    highs = series([
        ("2026-01-09 14:30", 105.0),
        ("2026-01-09 15:00", 99.0),
        ("2026-01-10 17:00", 1.0),
        ("2026-01-11 17:00", 2.0),
        ("2026-01-12 14:15", 3.0),
        ("2026-01-12 14:30", 4.0),
    ])

    got = aggregate_window(highs, WEEKDAY_OPEN_30, "max")

    assert got.loc[utc("2026-01-10 17:00")] == 105.0
    assert got.loc[utc("2026-01-11 17:00")] == 105.0
    assert got.loc[utc("2026-01-12 14:15")] == 105.0
    assert math.isnan(got.loc[utc("2026-01-12 14:30")])


def test_xnys_value_carries_through_saturday_regular_clock_rows():
    highs = series([
        ("2026-01-09 14:30", 105.0),
        ("2026-01-09 15:00", 99.0),
        ("2026-01-10 14:15", 1.0),
        ("2026-01-10 14:30", 2.0),
        ("2026-01-10 15:00", 3.0),
        ("2026-01-10 17:00", 4.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert list(got.loc[utc("2026-01-10 14:15"):]) == [105.0] * 4


def test_an_empty_weekday_occurrence_clears_an_older_value():
    highs = series([
        ("2026-01-05 14:30", 105.0),
        ("2026-01-05 15:00", 99.0),
        ("2026-01-07 12:00", 1.0),
    ])

    got = aggregate_window(highs, WEEKDAY_OPEN_30, "max")

    assert math.isnan(got.loc[utc("2026-01-07 12:00")])


def test_xnys_does_not_create_an_occurrence_for_an_absent_holiday():
    highs = series([
        ("2026-07-02 13:30", 105.0),
        ("2026-07-02 14:00", 99.0),
        ("2026-07-03 13:15", 1.0),
        ("2026-07-06 13:15", 2.0),
        ("2026-07-06 13:30", 3.0),
    ])

    got = aggregate_window(highs, NY_OPEN_30, "max")

    assert got.loc[utc("2026-07-03 13:15")] == 105.0
    assert got.loc[utc("2026-07-06 13:15")] == 105.0
    assert math.isnan(got.loc[utc("2026-07-06 13:30")])


def test_an_early_close_partial_is_not_revealed_after_the_next_open():
    highs = series([
        ("2026-11-27 17:00", 101.0),
        ("2026-11-27 17:30", 105.0),
        ("2026-11-30 17:15", 99.0),
    ])

    got = aggregate_window(highs, NY_PM, "max")

    assert math.isnan(got.loc[utc("2026-11-30 17:15")])


def test_london_boundaries_follow_uk_wall_clock_across_the_dst_mismatch():
    highs = series([
        ("2026-03-27 08:00", 101.0),
        ("2026-03-27 16:30", 1.0),
        ("2026-04-01 07:00", 102.0),
        ("2026-04-01 15:30", 2.0),
    ])

    got = aggregate_window(highs, LONDON, "max")

    assert math.isnan(got.loc[utc("2026-03-27 08:00")])
    assert got.loc[utc("2026-03-27 16:30")] == 101.0
    assert math.isnan(got.loc[utc("2026-04-01 07:00")])
    assert got.loc[utc("2026-04-01 15:30")] == 102.0


def test_overnight_occurrence_belongs_to_the_date_its_start_falls_on():
    highs = series([
        ("2026-01-10 01:00", 101.0),
        ("2026-01-10 08:45", 106.0),
        ("2026-01-10 09:00", 1.0),
        ("2026-01-11 17:00", 2.0),
        ("2026-01-13 00:45", 3.0),
        ("2026-01-13 01:00", 4.0),
    ])

    got = aggregate_window(highs, ASIA, "max")

    assert math.isnan(got.loc[utc("2026-01-10 08:45")])
    assert got.loc[utc("2026-01-10 09:00")] == 106.0
    assert got.loc[utc("2026-01-11 17:00")] == 106.0
    assert got.loc[utc("2026-01-13 00:45")] == 106.0
    assert math.isnan(got.loc[utc("2026-01-13 01:00")])


def test_prior_session_skips_weekend_and_excludes_extended_hours():
    highs = series([
        ("2026-01-09 13:00", 500.0),
        ("2026-01-09 14:30", 101.0),
        ("2026-01-09 20:45", 105.0),
        ("2026-01-09 21:00", 900.0),
        ("2026-01-10 15:00", 999.0),
        ("2026-01-12 13:00", 1.0),
    ])

    got = aggregate_window(highs, PRIOR_DAY, "max")

    assert got.loc[utc("2026-01-12 13:00")] == 105.0


def test_prior_iso_week_skips_an_entirely_empty_week():
    highs = series([
        ("2026-01-09 14:30", 101.0),
        ("2026-01-09 20:45", 108.0),
        ("2026-01-19 13:00", 1.0),
    ])

    got = aggregate_window(highs, PRIOR_WEEK, "max")

    assert got.loc[utc("2026-01-19 13:00")] == 108.0


def test_prior_calendar_month_skips_an_entirely_empty_month():
    lows = series([
        ("2026-01-30 14:30", 97.0),
        ("2026-01-30 20:45", 94.0),
        ("2026-03-02 13:00", 1.0),
    ])

    got = aggregate_window(lows, PRIOR_MONTH, "min")

    assert got.loc[utc("2026-03-02 13:00")] == 94.0


def test_session_aligned_input_treats_each_daily_row_as_one_regular_session():
    closes = series([
        ("2026-01-09 00:00", 100),
        ("2026-01-12 00:00", 103),
        ("2026-01-13 00:00", 107),
    ])

    got = aggregate_window(
        closes,
        PRIOR_DAY,
        "last",
        session_aligned=True,
    )

    assert math.isnan(got.iloc[0])
    assert list(got.iloc[1:]) == [100.0, 103.0]
    assert got.index.equals(closes.index)
    assert got.dtype == "float64"


def test_empty_input_preserves_its_index_and_returns_float64():
    source = pd.Series(
        [],
        index=pd.DatetimeIndex([], tz="UTC", name="ts"),
        dtype="int64",
    )

    got = aggregate_window(source, NY_OPEN_30, "max")

    assert got.index.equals(source.index)
    assert got.dtype == "float64"
