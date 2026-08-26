"""Generated Pine window state agrees with FrameEval on hard calendars."""

from datetime import time

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.rules import lower_pine
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import core_vocabulary
from nakagai.strategies.rules.windows import WindowSpec
from tests.pine_interpreter import as_series, run_program


def _bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    base = 100.0 + np.arange(len(index), dtype="float64")
    return pd.DataFrame({
        "open": base - 0.25,
        "high": base + np.resize(np.array([1.0, 3.0, 2.0]), len(index)),
        "low": base - np.resize(np.array([2.0, 1.0, 4.0]), len(index)),
        "close": base + np.resize(np.array([0.5, -0.5]), len(index)),
        "volume": np.full(len(index), 1_000.0),
    }, index=index)


def _local_spans(tz: str, spans) -> pd.DataFrame:
    index = pd.DatetimeIndex([
        stamp
        for day, start, end in spans
        for stamp in pd.date_range(
            f"{day} {start}", f"{day} {end}", freq="15min", tz=tz)
    ]).tz_convert("UTC")
    return _bars(index)


def _pine(node: dict, bars: pd.DataFrame, vocabulary) -> pd.Series:
    spec = {
        "version": 2,
        "name": "window_parity",
        "timeframe": "15m",
        "long": {"all": [{"lhs": node, "op": ">", "rhs": -1_000_000}]},
    }
    program = lower_pine(spec, vocabulary=vocabulary)
    target = f"nk_{node['ind']}_1"
    lines = []
    for calculation in program.calculations:
        lines.extend(calculation.splitlines())
        if any(line.startswith(f"{target} =") for line in calculation.splitlines()):
            break
    sources = {helper.id: helper.source for helper in program.helpers}
    rows = run_program(sources, lines, bars, pd.Timedelta(minutes=15))
    return as_series([row[target] for row in rows], bars)


def _assert_parity(node: dict, bars: pd.DataFrame, window: WindowSpec) -> None:
    vocabulary = core_vocabulary().with_windows(window)
    engine = FrameEval(
        {"15m": bars}, vocabulary=vocabulary).series(node, "15m")
    pine = _pine(node, bars, vocabulary)
    np.testing.assert_allclose(
        pine.to_numpy(), engine.to_numpy(dtype="float64"),
        rtol=0.0, atol=0.0, equal_nan=True)
    assert engine.notna().sum() >= 4


NY_AM = WindowSpec(
    "ny_am", "America/New_York", time(9, 30), time(12),
    "xnys_session", "standard")
NY_PM = WindowSpec(
    "ny_pm", "America/New_York", time(12), time(16),
    "xnys_session", "standard")


@pytest.mark.parametrize("term, source", [
    ("highest", "high"),
    ("lowest", "low"),
    ("first", "open"),
    ("last", "close"),
])
def test_all_reducers_match_on_ordinary_current_sessions(term, source):
    bars = _local_spans("America/New_York", [
        ("2026-02-02", "08:00", "17:00"),
        ("2026-02-03", "08:00", "17:00"),
        ("2026-02-04", "08:00", "17:00"),
    ])
    _assert_parity(
        {"ind": term, "of": {"src": source}, "window": "ny_am"},
        bars, NY_AM)


@pytest.mark.parametrize("first_close_minute", [0, 15])
def test_ny_pm_becomes_visible_on_the_first_bar_at_or_after_its_close(
        first_close_minute):
    bars = _local_spans("America/New_York", [
        ("2026-02-02", "08:00", "17:00"),
        ("2026-02-03", "08:00", "17:00"),
    ])
    if first_close_minute:
        local = bars.index.tz_convert("America/New_York")
        bars = bars[~((local.hour == 16) & (local.minute == 0))]
    vocabulary = core_vocabulary().with_windows(NY_PM)
    node = {"ind": "highest", "of": {"src": "high"}, "window": "ny_pm"}
    engine = FrameEval(
        {"15m": bars}, vocabulary=vocabulary).series(node, "15m")
    pine = _pine(node, bars, vocabulary)
    local = bars.index.tz_convert("America/New_York")
    first_close = np.flatnonzero(
        (local.hour == 16) & (local.minute == first_close_minute))[0]
    assert not np.isnan(engine.iloc[first_close])
    assert pine.iloc[first_close] == engine.iloc[first_close]
    np.testing.assert_allclose(
        pine.to_numpy(), engine.to_numpy(dtype="float64"),
        rtol=0.0, atol=0.0, equal_nan=True)


def test_an_early_close_partial_does_not_leak_after_the_next_ny_pm_opens():
    bars = _local_spans("America/New_York", [
        ("2026-11-27", "08:00", "12:45"),
        ("2026-11-30", "08:00", "17:00"),
        ("2026-12-01", "08:00", "17:00"),
    ])
    vocabulary = core_vocabulary().with_windows(NY_PM)
    node = {"ind": "highest", "of": {"src": "high"}, "window": "ny_pm"}
    engine = FrameEval(
        {"15m": bars}, vocabulary=vocabulary).series(node, "15m")
    pine = _pine(node, bars, vocabulary)
    np.testing.assert_allclose(
        pine.to_numpy(), engine.to_numpy(dtype="float64"),
        rtol=0.0, atol=0.0, equal_nan=True)
    local = bars.index.tz_convert("America/New_York")
    next_active = (
        (local.date == pd.Timestamp("2026-11-30").date())
        & (local.hour >= 12) & (local.hour < 16))
    assert engine[next_active].isna().all()
    assert pine[next_active].isna().all()


def test_an_overnight_occurrence_belongs_to_the_date_it_opens():
    asia = WindowSpec(
        "asia", "America/New_York", time(20), time(4),
        "weekday", "low_iex")
    bars = _local_spans("America/New_York", [
        ("2026-02-02", "19:00", "23:45"),
        ("2026-02-03", "00:00", "05:00"),
        ("2026-02-03", "19:00", "23:45"),
        ("2026-02-04", "00:00", "05:00"),
    ])
    _assert_parity(
        {"ind": "highest", "of": {"src": "high"}, "window": "asia"},
        bars, asia)


def test_an_overnight_window_does_not_reopen_friday_on_monday_morning():
    asia = WindowSpec(
        "asia", "America/New_York", time(20), time(4),
        "weekday", "low_iex")
    bars = _local_spans("America/New_York", [
        ("2026-02-06", "19:00", "23:45"),
        ("2026-02-07", "00:00", "05:00"),
        ("2026-02-09", "00:00", "05:00"),
        ("2026-02-09", "19:00", "23:45"),
        ("2026-02-10", "00:00", "05:00"),
    ])
    _assert_parity(
        {"ind": "last", "of": {"src": "close"}, "window": "asia"},
        bars, asia)


def test_prior_day_skips_a_holiday_and_keeps_an_early_close_session():
    prior_day = WindowSpec(
        "prior_day", "America/New_York", time(9, 30), time(16),
        "prior_session", "standard")
    bars = _local_spans("America/New_York", [
        ("2026-07-02", "08:00", "12:45"),
        ("2026-07-06", "08:00", "17:00"),
        ("2026-07-07", "08:00", "17:00"),
    ])
    _assert_parity(
        {"ind": "last", "of": {"src": "close"}, "window": "prior_day"},
        bars, prior_day)


@pytest.mark.parametrize("row", [
    WindowSpec(
        "prior_week", "America/New_York", time(9, 30), time(16),
        "prior_iso_week", "standard"),
    WindowSpec(
        "prior_month", "America/New_York", time(9, 30), time(16),
        "prior_calendar_month", "standard"),
])
def test_completed_prior_periods_shift_only_at_their_calendar_boundary(row):
    spans = (
        [(f"2025-12-{day:02d}", "09:30", "16:00") for day in range(22, 27)]
        + [(f"2025-12-{day:02d}", "09:30", "16:00") for day in range(29, 32)]
        + [(f"2026-01-{day:02d}", "09:30", "16:00") for day in range(2, 10)]
        + [("2026-02-02", "09:30", "16:00")]
    )
    bars = _local_spans("America/New_York", spans)
    _assert_parity(
        {"ind": "lowest", "of": {"src": "low"}, "window": row.name},
        bars, row)


def test_london_wall_clock_survives_the_us_uk_dst_mismatch():
    london = WindowSpec(
        "london", "Europe/London", time(8), time(16, 30),
        "weekday", "low_iex")
    bars = _local_spans("UTC", [
        ("2026-03-23", "07:00", "18:00"),
        ("2026-03-24", "07:00", "18:00"),
        ("2026-03-30", "06:00", "17:00"),
        ("2026-03-31", "06:00", "17:00"),
    ])
    _assert_parity(
        {"ind": "first", "of": {"src": "open"}, "window": "london"},
        bars, london)


def test_a_selected_frame_runs_the_window_state_inside_request_security():
    london = WindowSpec(
        "london", "Europe/London", time(8), time(16, 30),
        "weekday", "low_iex")
    vocabulary = core_vocabulary().with_windows(london)
    chart = _local_spans("UTC", [
        ("2026-03-23", "07:00", "18:00"),
        ("2026-03-24", "07:00", "18:00"),
        ("2026-03-30", "06:00", "17:00"),
        ("2026-03-31", "06:00", "17:00"),
    ])
    aggregate = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    hourly = chart.resample("1h").agg(aggregate).dropna()
    node = {
        "ind": "highest", "of": {"src": "high"}, "tf": "1h",
        "window": "london",
    }
    spec = {
        "version": 2,
        "name": "selected_window_parity",
        "timeframe": "15m",
        "long": {"all": [{"lhs": node, "op": ">", "rhs": -1_000_000}]},
    }
    program = lower_pine(spec, vocabulary=vocabulary)
    request_index = next(
        index for index, calculation in enumerate(program.calculations)
        if "request.security" in calculation)
    request = program.calculations[request_index]
    requested_function = program.calculations[request_index - 1]
    target = request.split("_raw = request.security", 1)[0]
    lines: list[str] = []
    for calculation in program.calculations[:request_index + 3]:
        lines.extend(calculation.splitlines())
    lines.append(f"nk_selected_probe = {target}")
    rows = run_program(
        {helper.id: helper.source for helper in program.helpers},
        lines, chart, pd.Timedelta(minutes=15),
        {"60": (hourly, pd.Timedelta(hours=1))})
    pine = as_series([row["nk_selected_probe"] for row in rows], chart)
    engine = FrameEval(
        {"15m": chart, "1h": hourly}, vocabulary=vocabulary,
    ).series(node, "15m")
    np.testing.assert_allclose(
        pine.to_numpy(), engine.to_numpy(dtype="float64"),
        rtol=0.0, atol=0.0, equal_nan=True)
    assert engine.notna().sum() >= 4
    assert "window_clock" in requested_function
    assert all(
        "window_clock" not in calculation
        for calculation in program.calculations
        if calculation is not requested_function)
