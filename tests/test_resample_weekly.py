"""1w bars are DERIVED from cached 1d bars: one bar per exchange week.

A week is Monday to Friday on the New York calendar, labeled with its first
SESSION's date at New York midnight (the key BarCache gives a daily bar), so a
holiday Monday moves the label to Tuesday. A week is published only once it
is known complete: the source holds its Friday bar, or a bar from a later
week, or the caller's clock (`now`) has reached a later New York week.
"""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.data.resample import DERIVED, resample_bars
from nakagai.data.schema import BAR_COLUMNS, EXCHANGE_TZ
from nakagai.data.sync import derive_incremental

ET = EXCHANGE_TZ


def _labels(days, *, canonical=True):
    """Session dates as the cache keys them: New York midnight (canonical) or
    the legacy 00:00Z a provider may hand over."""
    naive = pd.DatetimeIndex(list(days))
    tz = ET if canonical else "UTC"
    return naive.tz_localize(tz).tz_convert("UTC").rename("ts")


def _daily(days, *, canonical=True, start=100.0):
    idx = _labels(days, canonical=canonical)
    close = start + np.arange(len(idx), dtype="float64")
    return pd.DataFrame({"open": close - 0.5, "high": close + 1.0,
                         "low": close - 1.0, "close": close,
                         "volume": 1_000.0}, index=idx)


def _weekdays(first, last, *, skip=()):
    days = pd.bdate_range(first, last)
    return [d.strftime("%Y-%m-%d") for d in days
            if d.strftime("%Y-%m-%d") not in skip]


def _label(day):
    return pd.Timestamp(day, tz=ET).tz_convert("UTC")


def _utc(stamp):
    return pd.Timestamp(stamp, tz="UTC")


def test_1w_is_derived_from_1d():
    assert DERIVED["1w"] == "1d"


def test_a_week_is_labeled_by_its_first_session_and_aggregates_ohlcv():
    idx = _labels(_weekdays("2026-06-01", "2026-06-05"))
    src = pd.DataFrame({"open": [10.0, 11.0, 12.0, 13.0, 14.0],
                        "high": [15.0, 19.0, 16.0, 17.0, 18.0],
                        "low": [9.0, 8.5, 8.0, 9.5, 9.0],
                        "close": [11.0, 12.0, 13.0, 14.0, 15.0],
                        "volume": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=idx)
    out = resample_bars(src, "1w")
    assert list(out.columns) == BAR_COLUMNS
    assert list(out.index) == [_label("2026-06-01")]
    row = out.iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"], row["volume"]) == (
        10.0, 19.0, 8.0, 15.0, 1500.0)


def test_a_forming_week_is_withheld():
    src = _daily(_weekdays("2026-06-01", "2026-06-03"))          # Mon..Wed
    assert resample_bars(src, "1w").empty
    # The clock inside the same week is no evidence the week is over.
    assert resample_bars(src, "1w", now=_utc("2026-06-03 20:30")).empty


def test_a_holiday_friday_week_waits_for_a_bar_from_the_next_week():
    # Good Friday, 2026-04-03: the week of 30 March has four sessions.
    short_week = _weekdays("2026-03-30", "2026-04-02")
    assert resample_bars(_daily(short_week), "1w").empty
    out = resample_bars(_daily(short_week + ["2026-04-06"]), "1w")
    assert list(out.index) == [_label("2026-03-30")]
    assert out["close"].iloc[0] == 103.0                         # Thursday's close
    assert out["volume"].iloc[0] == 4_000.0
    # and the week of 6 April, one session old, is still forming
    assert _label("2026-04-06") not in out.index


@pytest.mark.parametrize("now, published", [
    ("2026-04-03 20:30", False),   # Good Friday itself: still the same week
    ("2026-04-06 03:30", False),   # Sunday 23:30 in New York; UTC is Monday
    ("2026-04-06 04:00", True),    # Monday 00:00 in New York
    ("2026-04-06 13:45", True),    # the first scan of the next week
])
def test_the_clock_closes_a_holiday_friday_week_before_the_next_session_lands(
        now, published):
    src = _daily(_weekdays("2026-03-30", "2026-04-02"))
    out = resample_bars(src, "1w", now=_utc(now))
    assert (list(out.index) == [_label("2026-03-30")]) is published


def test_a_holiday_monday_moves_the_label_to_tuesday():
    # Labor Day, 2026-09-07.
    out = resample_bars(_daily(_weekdays("2026-09-08", "2026-09-11")), "1w")
    assert list(out.index) == [_label("2026-09-08")]


@pytest.mark.parametrize("first, last, labels, offsets", [
    # spring forward on Sunday 2026-03-08
    ("2026-03-02", "2026-03-13", ["2026-03-02", "2026-03-09"], ["-0500", "-0400"]),
    # fall back on Sunday 2026-11-01
    ("2026-10-26", "2026-11-06", ["2026-10-26", "2026-11-02"], ["-0400", "-0500"]),
])
def test_week_labels_sit_on_new_york_midnight_across_daylight_saving(
        first, last, labels, offsets):
    out = resample_bars(_daily(_weekdays(first, last)), "1w")
    assert list(out.index) == [_label(d) for d in labels]
    local = out.index.tz_convert(ET)
    assert [ts.hour for ts in local] == [0, 0]
    assert [ts.strftime("%z") for ts in local] == offsets


def test_legacy_utc_midnight_daily_labels_make_the_same_weeks():
    days = _weekdays("2026-06-01", "2026-06-12")
    pd.testing.assert_frame_equal(
        resample_bars(_daily(days, canonical=False), "1w"),
        resample_bars(_daily(days), "1w"))


def test_derive_incremental_rewrites_a_week_built_on_a_forming_friday(tmp_path):
    """The scan fetches 1d intraday, so Friday's daily bar can be partial when
    the week is first published. The next derive resumes from that week's own
    label and rewrites it whole."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1d", _daily(_weekdays("2026-06-01", "2026-06-04")))
    forming = _daily(["2026-06-05"], start=50.0)
    cache.upsert("SPY", "1d", forming)
    derive_incremental(cache, "SPY", "1w")
    assert cache.load("SPY", "1w")["close"].iloc[-1] == 50.0

    cache.upsert("SPY", "1d", _daily(["2026-06-05"], start=55.0))     # final
    cache.upsert("SPY", "1d", _daily(["2026-06-08"], start=60.0))
    derive_incremental(cache, "SPY", "1w")
    weekly = cache.load("SPY", "1w")
    assert list(weekly.index) == [_label("2026-06-01")]
    assert weekly["close"].iloc[-1] == 55.0


def test_derive_incremental_passes_the_clock_through(tmp_path):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1d", _daily(_weekdays("2026-03-30", "2026-04-02")))
    assert derive_incremental(cache, "SPY", "1w") == 0
    assert derive_incremental(cache, "SPY", "1w", now=_utc("2026-04-06 13:45")) == 1
    assert list(cache.load("SPY", "1w").index) == [_label("2026-03-30")]


def test_deriving_weeks_in_steps_matches_deriving_in_one_pass(tmp_path):
    days = _weekdays("2026-05-04", "2026-06-26", skip=("2026-05-25", "2026-06-19"))
    full = _daily(days)
    stepwise = BarCache(tmp_path / "stepwise")
    for i in range(0, len(full), 3):
        stepwise.upsert("SPY", "1d", full.iloc[i:i + 3])
        derive_incremental(stepwise, "SPY", "1w")
    one_pass = BarCache(tmp_path / "one_pass")
    one_pass.upsert("SPY", "1d", full)
    derive_incremental(one_pass, "SPY", "1w")
    pd.testing.assert_frame_equal(stepwise.load("SPY", "1w"),
                                  one_pass.load("SPY", "1w"))


def test_a_week_relabeled_by_a_late_first_session_keeps_one_row(tmp_path):
    """A week first derived without its Monday is labeled Tuesday. When the
    Monday bar arrives and a full derive relabels the week, the cache must
    hold the week once, under the newer label, not twice."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1w", _daily(["2026-06-02"], start=10.0))
    cache.upsert("SPY", "1w", _daily(["2026-06-01"], start=20.0))
    weekly = cache.load("SPY", "1w")
    assert list(weekly.index) == [_label("2026-06-01")]
    assert weekly["close"].iloc[0] == 20.0
