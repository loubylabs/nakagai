import numpy as np
import pandas as pd

from nakagai.strategies.base import MarketContext


def _session_bars():
    """Two NY sessions of 15m bars, deterministic prices."""
    idx = []
    for day in ("2026-01-05", "2026-01-06"):
        idx.extend(pd.date_range(f"{day} 14:30", f"{day} 20:45", freq="15min", tz="UTC"))
    idx = pd.DatetimeIndex(idx)
    close = np.linspace(100, 110, len(idx))
    return pd.DataFrame({"open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=idx)


def _ctx(bars):
    return MarketContext(symbol="SPY", now=bars.index[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def test_opening_range_high_is_per_session_and_constant_after_window():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    orh = PRIMITIVES["opening_range_high"]["fn"](_ctx(bars), bars, minutes=30)
    day2 = orh[bars.index.tz_convert("America/New_York").date == pd.Timestamp("2026-01-06").date()]
    first_two = bars.loc[day2.index[:2], "high"].max()
    assert (day2.dropna() == first_two).all()          # constant within the session
    assert pd.isna(orh.iloc[0])                        # NaN until the window completes


def _reference_opening_range(bars, minutes, col, how):
    """The obvious per-session loop, kept as the definition of correct.

    The shipped primitive is vectorized for speed; this stays as the thing it
    must agree with, so the rewrite is checked against something readable
    rather than against itself.
    """
    from nakagai.data.schema import EXCHANGE_TZ
    out = pd.Series(np.nan, index=bars.index)
    days = np.asarray(bars.index.tz_convert(EXCHANGE_TZ).date)
    for _, day in bars.groupby(days):
        start = day.index[0]
        window = day[day.index < start + pd.Timedelta(minutes=minutes)]
        done = day.index >= start + pd.Timedelta(minutes=minutes)
        if done.any():
            out.loc[day.index[done]] = getattr(window[col], how)()
    return out


def _ragged_sessions():
    """Sessions the tidy fixture does not cover: a half day that closes before
    the opening range completes, a late open, and a normal day."""
    spans = [("2026-01-05 14:30", "2026-01-05 20:45"),   # full session
             ("2026-01-06 14:30", "2026-01-06 14:45"),   # closes inside the 30m window
             ("2026-01-07 15:30", "2026-01-07 18:00"),   # late open, short session
             ("2026-01-08 14:30", "2026-01-08 21:00")]
    idx = pd.DatetimeIndex([t for a, b in spans
                            for t in pd.date_range(a, b, freq="15min", tz="UTC")])
    close = np.linspace(100, 130, len(idx))
    return pd.DataFrame({"open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=idx)


def test_opening_range_matches_the_reference_loop_on_ragged_sessions():
    """A session that ends before its opening range completes must stay NaN,
    and a late open must measure from its own first bar, not from the clock."""
    from nakagai.strategies.rules.primitives import _opening_range
    bars = _ragged_sessions()
    for col, how in (("high", "max"), ("low", "min")):
        got = _opening_range(bars, 30, col, how)
        want = _reference_opening_range(bars, 30, col, how)
        pd.testing.assert_series_equal(got, want, check_names=False)


def test_opening_range_of_an_empty_frame_is_empty():
    from nakagai.strategies.rules.primitives import _opening_range
    empty = pd.DataFrame({"high": [], "low": []},
                         index=pd.DatetimeIndex([], tz="UTC"))
    assert _opening_range(empty, 30, "high", "max").empty


def test_opening_range_cost_does_not_grow_with_history():
    """The regression guard.

    This primitive is called once per replayed bar, so any per-session Python
    loop inside it makes a window replay O(sessions x bars) and gets heavier
    every month as history accumulates. That is not hypothetical: the loop this
    replaced is why the proving farm's permutation step burned its 90-minute
    budget every week without finishing a single permutation, for every play
    that reads an opening range.

    Measured on three years of 15m bars (750 sessions): the loop 90ms per call,
    vectorized 2.6ms. The 25ms bound sits between them with room on both sides,
    so a slow or loaded machine does not flake and the loop cannot come back.
    """
    import time
    from nakagai.strategies.rules.primitives import _opening_range
    idx = pd.DatetimeIndex([t for d in pd.bdate_range("2023-07-03", periods=750)
                            for t in pd.date_range(f"{d.date()} 14:30", f"{d.date()} 20:45",
                                                   freq="15min", tz="UTC")])
    close = np.linspace(100, 400, len(idx))
    bars = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=idx)
    _opening_range(bars, 30, "high", "max")        # warm pandas' caches
    t0 = time.perf_counter()
    _opening_range(bars, 30, "high", "max")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.025, (
        f"opening_range took {elapsed*1000:.0f}ms over {len(idx)} bars; it is called "
        "once per replayed bar, so this is a per-bar cost, not a per-run one")


def test_prev_session_close_and_gap_pct():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    ctx = _ctx(bars)
    day1_close = bars[bars.index.date == pd.Timestamp("2026-01-05").date()]["close"].iloc[-1]
    psc = PRIMITIVES["prev_session_close"]["fn"](ctx, bars)
    assert psc.iloc[-1] == day1_close
    day2_open = bars[bars.index.date == pd.Timestamp("2026-01-06").date()]["open"].iloc[0]
    gap = PRIMITIVES["gap_pct"]["fn"](ctx, bars)
    assert abs(gap.iloc[-1] - 100 * (day2_open - day1_close) / day1_close) < 1e-9


def test_swing_high_returns_last_confirmed_swing_level():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    bars.loc[bars.index[20], "high"] = 200.0           # inject an obvious swing
    sh = PRIMITIVES["swing_high"]["fn"](_ctx(bars), bars, k=3)
    assert sh.iloc[-1] == 200.0


def test_time_primitives():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    ctx = _ctx(bars)
    dow = PRIMITIVES["day_of_week"]["fn"](ctx, bars)
    assert dow.iloc[0] == 0 and dow.iloc[-1] == 1      # Mon, Tue
    mins = PRIMITIVES["minutes_into_session"]["fn"](ctx, bars)
    assert mins.iloc[0] == 0.0 and mins.iloc[2] == 30.0


def test_bars_since_counts_bars_since_condition_true():
    from nakagai.strategies.rules.primitives import bars_since
    bars = _session_bars()
    mask = pd.Series(False, index=bars.index)
    mask.iloc[-4] = True
    out = bars_since(_ctx(bars), bars, cond={"placeholder": True},
                     eval_fn=lambda cond, bars_: mask)
    assert out.iloc[-1] == 3.0
    assert pd.isna(out.iloc[0])                        # never true yet -> NaN


def test_fvg_nearest_returns_float_or_nan():
    from nakagai.strategies.rules.primitives import PRIMITIVES
    bars = _session_bars()
    v = PRIMITIVES["fvg_nearest"]["fn"](_ctx(bars), bars, direction="long", field="top")
    assert isinstance(v, float)                        # NaN when no unfilled FVG exists


def test_day_of_week_reads_the_utc_calendar_day_on_daily_frames():
    # Session-aligned daily bars are labeled midnight UTC of their session date;
    # converting midnight UTC to NY lands on the prior evening, which used to
    # shift every weekday back by one (a Monday bar read as Sunday).
    from nakagai.strategies.rules.primitives import PRIMITIVES
    idx = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")  # Mon..Fri
    bars = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=idx)
    dow = PRIMITIVES["day_of_week"]["fn"](None, bars)
    assert list(dow) == [0.0, 1.0, 2.0, 3.0, 4.0]


def _ny(day: str, clock: str) -> pd.Timestamp:
    """A bar's label written on the exchange's clock, returned as the UTC label
    the bar actually carries."""
    from nakagai.data.schema import EXCHANGE_TZ
    return pd.Timestamp(f"{day} {clock}", tz=EXCHANGE_TZ).tz_convert("UTC")


def _clock_bars(volumes, start="2026-01-05"):
    """One bar per (NY session, clock time), with every volume stated outright.

    `volumes` is one dict per session, clock time -> volume. A clock time a
    session does not list simply has no bar there, which is what a half day or
    a late open looks like on the tape. The fixture is written on the exchange
    clock and converted to UTC because that is the direction the fact runs: the
    09:30 bar is labeled 14:30 UTC in winter and 13:30 UTC in summer.
    """
    days = pd.bdate_range(start, periods=len(volumes))
    rows = [(_ny(str(day.date()), clock), float(v))
            for day, per_clock in zip(days, volumes)
            for clock, v in sorted(per_clock.items())]
    close = np.linspace(100, 110, len(rows))
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": [v for _, v in rows]},
                        index=pd.DatetimeIndex([ts for ts, _ in rows]))


def _rvol(bars, sessions):
    from nakagai.strategies.rules.primitives import PRIMITIVES
    return PRIMITIVES["rvol"]["fn"](_ctx(bars), bars, sessions=sessions)


def test_rvol_is_this_bars_volume_over_the_same_clock_times_trailing_median():
    """Five ordinary 09:30 bars, one of them a 10x print, then a 3x session.

    A mean baseline reads 280 across that window and calls the last bar 1.07x,
    which is the failure the primitive exists to remove: one halted or
    news-driven morning in the window hides the next genuine surge behind it.
    The median reads 100 and calls the bar what it is. The 09:45 bucket is
    deliberately wild and must not move the 09:30 answer by anything at all.
    """
    nine30 = [100, 100, 100, 100, 1000, 300]
    nine45 = [7, 5000, 12, 3, 90000, 4]
    bars = _clock_bars([{"09:30": a, "09:45": b} for a, b in zip(nine30, nine45)])
    rvol = _rvol(bars, sessions=5)
    assert rvol.loc[_ny("2026-01-12", "09:30")] == 3.0
    # answered from its own bucket, whose trailing median is 12
    assert rvol.loc[_ny("2026-01-12", "09:45")] == 4 / 12


def test_rvol_baseline_excludes_the_current_session():
    """The easiest thing here to get subtly wrong.

    The five sessions before the last one are [10, 10, 100, 1000, 1000] at
    09:30, median 100, so a 1000-volume bar is 10x. Let that bar into its own
    window and the five become [10, 100, 1000, 1000, 1000], median 1000, and
    the same bar reads 1.0: a completely unremarkable morning. Same bars,
    opposite verdict, which is why the baseline is shifted.
    """
    bars = _clock_bars([{"09:30": v} for v in (10, 10, 100, 1000, 1000, 1000)])
    assert _rvol(bars, sessions=5).loc[_ny("2026-01-12", "09:30")] == 10.0


def test_rvol_is_nan_until_the_bucket_holds_a_full_window():
    """Half a window is a different measurement, not a rough one."""
    bars = _clock_bars([{"09:30": 100}] * 6)
    rvol = _rvol(bars, sessions=5)
    assert rvol.iloc[:5].isna().all()          # four earlier sessions is not five
    assert rvol.iloc[5] == 1.0


def test_rvol_buckets_on_the_exchange_clock_and_not_on_the_utc_label():
    """One bucket has to survive the March clock change.

    The 09:30 bar is labeled 14:30 UTC in winter and 13:30 UTC from the second
    Sunday in March. Bucketing on the UTC label splits that single bucket in
    two twice a year, and every bar reads NaN for `sessions` days afterwards
    for want of history it demonstrably has.
    """
    bars = _clock_bars([{"09:30": 100}] * 20 + [{"09:30": 300}], start="2026-02-16")
    assert bars.index[0].hour == 14 and bars.index[-1].hour == 13   # it straddles
    assert _rvol(bars, sessions=20).iloc[-1] == 3.0


def test_rvol_steps_back_by_an_occurrence_of_the_clock_time_not_by_a_session():
    """A session with no bar at this clock time is simply not in the bucket.

    Half days and late opens are ordinary. Stepping the baseline back by a
    calendar session rather than by an occurrence drags that hole through the
    window and blanks the clock time for the next `sessions` days.
    """
    volumes = [{"09:30": 100, "10:00": 100} for _ in range(7)]
    del volumes[3]["10:00"]                    # a session that closed before 10:00
    volumes[-1]["10:00"] = 300
    bars = _clock_bars(volumes)
    assert _rvol(bars, sessions=5).loc[_ny("2026-01-13", "10:00")] == 3.0


def test_rvol_of_an_untraded_clock_time_is_nan_and_not_infinite():
    """A zero baseline is an absence of anything to compare against.

    Illiquid names print zero-volume bars at the same clock time for days on
    end. Dividing by that median gives inf, which passes every `>` threshold a
    spec can write; NaN reads False, which is the honest answer.
    """
    bars = _clock_bars([{"09:30": 0}] * 5 + [{"09:30": 400}])
    assert pd.isna(_rvol(bars, sessions=5).iloc[-1])


def test_rvol_is_registered_with_its_bounds_and_its_default():
    from nakagai.strategies.rules.primitives import (ARG_DEFAULTS, PRIMITIVES,
                                                     SESSION_SCOPED_PRIMS)
    assert PRIMITIVES["rvol"]["args"] == {"sessions": (5, 60)}
    assert ARG_DEFAULTS["rvol"] == {"sessions": 20}
    assert "rvol" in SESSION_SCOPED_PRIMS


def test_the_two_timeframe_sets_are_not_the_same_set():
    """Two rules, two sets. A foreign `tf` is refused for every session-scoped
    primitive; a session-aligned DRIVING frame is refused for the four that a
    one-bar session cannot answer. day_of_week sits in the first set and not
    the second, because a daily bar is one session and its weekday is exactly
    what the primitive promises; turnaround_tuesday, a shipped 1d play, is
    built on that reading."""
    from nakagai.strategies.rules.primitives import (
        DRIVING_FRAME_INTRADAY_PRIMS, SESSION_SCOPED_PRIMS)
    assert DRIVING_FRAME_INTRADAY_PRIMS < SESSION_SCOPED_PRIMS
    assert DRIVING_FRAME_INTRADAY_PRIMS == {
        "opening_range_high", "opening_range_low", "minutes_into_session", "rvol"}
    assert "day_of_week" not in DRIVING_FRAME_INTRADAY_PRIMS
