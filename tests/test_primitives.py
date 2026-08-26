import numpy as np
import pandas as pd
import pytest

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


# The extended-hours session as a cache holds it, with every band flat and
# every number below named in an assertion, so nothing can pass by coincidence.
#
#                   high     low    close
#     08:00-09:15    120      80      118      pre-market
#     09:30-15:45    110      90      101      the regular session
#     16:00-19:45    130      70      125      post-market
#
# The off-hours bands are WIDER than the regular one on both sides, which is
# what a thin off-hours tape does and is the whole of chrvsd/nakagai#276: the
# calendar-date aggregate answers 130 and 70 where the session answers 110 and
# 90, and calls the 19:45 print the session's close. Two bars carry their own
# numbers because two primitives read exactly them: the 09:30 bar's OPEN is
# what a gap is measured from, and the 15:45 bar's CLOSE is what every
# "yesterday's close" in the catalog means.
PRE_MARKET_HIGH, PRE_MARKET_LOW, PRE_MARKET_CLOSE = 120.0, 80.0, 118.0
SESSION_HIGH, SESSION_LOW = 110.0, 90.0
POST_MARKET_HIGH, POST_MARKET_LOW, POST_MARKET_CLOSE = 130.0, 70.0, 125.0


def _extended_session(day: str, bell_open: float, last_close: float,
                      start: str = "08:00", end: str = "19:45"):
    """One session on the exchange clock, from `start` to `end` inclusive."""
    from nakagai.data.schema import EXCHANGE_TZ
    idx = pd.date_range(f"{day} {start}", f"{day} {end}", freq="15min",
                        tz=EXCHANGE_TZ).tz_convert("UTC")
    ny = idx.tz_convert(EXCHANGE_TZ)
    clock = ny.hour * 60 + ny.minute
    band = np.select([clock < 570, clock < 960], [0, 1], default=2)
    close = np.choose(band, [PRE_MARKET_CLOSE, 101.0, POST_MARKET_CLOSE])
    bars = pd.DataFrame(
        {"open": close,
         "high": np.choose(band, [PRE_MARKET_HIGH, SESSION_HIGH, POST_MARKET_HIGH]),
         "low": np.choose(band, [PRE_MARKET_LOW, SESSION_LOW, POST_MARKET_LOW]),
         "close": close, "volume": 1000.0}, index=idx)
    bars.loc[idx[clock == 570], "open"] = bell_open
    bars.loc[idx[clock == 945], "close"] = last_close
    return bars


# The bell open and the 15:45 close of each session, so the gap between them is
# a number written here rather than derived from the fixture.
FIRST_BELL, FIRST_CLOSE = 100.0, 105.0
SECOND_BELL, SECOND_CLOSE = 107.0, 106.0


def _two_extended_sessions(first="2026-01-05", second="2026-01-06"):
    return pd.concat([_extended_session(first, FIRST_BELL, FIRST_CLOSE),
                      _extended_session(second, SECOND_BELL, SECOND_CLOSE)])


def _extended_hours_session(day="2026-01-05"):
    return _extended_session(day, FIRST_BELL, FIRST_CLOSE)


def _clock(bars) -> np.ndarray:
    ny = bars.index.tz_convert("America/New_York")
    return np.asarray(ny.hour * 60 + ny.minute)


def _second_session(bars, day="2026-01-06") -> np.ndarray:
    return np.asarray(bars.index.tz_convert("America/New_York").date
                      == pd.Timestamp(day).date())


def test_the_gap_is_the_bell_s_open_against_the_previous_regular_close():
    """The same regression on the gap, where both sides were wrong at once.

    The open was the first bar of the New York date, an 08:00 pre-market print
    at 118, and the close was the previous date's last post-market print at
    125, so the "overnight gap" was measured across a span that already
    contained the move it was meant to name. Both wrong readings are present in
    this fixture and each would change the answer on its own, so agreeing with
    the right number cannot happen by halves.
    """
    from nakagai.strategies.rules.primitives import gap_pct
    bars = _two_extended_sessions()
    today = _second_session(bars)
    gap = gap_pct(None, bars)
    want = 100 * (SECOND_BELL - FIRST_CLOSE) / FIRST_CLOSE
    at_or_after_the_bell = today & (_clock(bars) >= 570)
    assert np.allclose(gap[at_or_after_the_bell], want)
    # The three readings each wrong half produces, named so that agreeing with
    # `want` cannot be an arithmetic coincidence: both sides wrong, the close
    # alone wrong, the open alone wrong.
    for wrong in (100 * (PRE_MARKET_CLOSE - POST_MARKET_CLOSE) / POST_MARKET_CLOSE,
                  100 * (SECOND_BELL - POST_MARKET_CLOSE) / POST_MARKET_CLOSE,
                  100 * (PRE_MARKET_CLOSE - FIRST_CLOSE) / FIRST_CLOSE):
        assert not np.isclose(want, wrong)
        assert not np.isclose(gap.dropna(), wrong).any()


def test_the_gap_is_nan_before_the_session_it_measures_has_opened():
    """No lookahead, and it is the pre-market bars that need saying so.

    pandas broadcasts a group's first regular open over the WHOLE group,
    backwards included, so without the `started` mask an 08:00 bar would read
    the 09:30 open an hour and a half before it printed. That is lookahead on
    exactly the bars an overnight-gap play is most tempted to trade: it would
    fire in replay and never live. A condition over NaN reads False.
    """
    from nakagai.strategies.rules.primitives import gap_pct
    bars = _two_extended_sessions()
    gap = gap_pct(None, bars)
    before_the_bell = _second_session(bars) & (_clock(bars) < 570)
    assert before_the_bell.sum() == 6, "08:00 through 09:15 is six 15m bars"
    assert gap[before_the_bell].isna().all()
    # Past 16:00 it stays readable, the same way the other session-scoped
    # terms treat the post-market: the gap is a fact about the session and it
    # does not stop being true when the bell rings.
    assert gap[_second_session(bars) & (_clock(bars) >= 960)].notna().all()


def test_the_gap_keeps_the_existing_zero_denominator_behavior():
    from nakagai.strategies.rules.primitives import gap_pct
    idx = pd.date_range("2026-01-05", periods=2, freq="B", tz="UTC")
    bars = pd.DataFrame(
        {"open": [0.0, 10.0], "high": [0.0, 10.0], "low": [0.0, 10.0],
         "close": [0.0, 10.0], "volume": 1000.0},
        index=idx,
    )

    assert np.isposinf(gap_pct(None, bars).iloc[1])


@pytest.mark.parametrize("first, second, label",
                         [("2026-03-06", "2026-03-09", "EST into EDT"),
                          ("2026-10-30", "2026-11-02", "EDT into EST")])
def test_the_session_bounds_hold_across_a_clock_change(first, second, label):
    """09:30 and 16:00 New York, whatever UTC instants those are that week.

    Both transitions, because they move the offset in opposite directions and
    a sign error passes one of them. The fixture is written on the exchange
    clock and converted, which is the direction the fact runs: the 09:30 bar is
    14:30 UTC on the Friday and 13:30 UTC on the Monday, so a mask built on the
    UTC label would take the 08:30 pre-market bar into one session's aggregate
    and drop the 15:30 bar out of the other's.
    """
    from nakagai.strategies.rules.primitives import gap_pct
    bars = _two_extended_sessions(first, second)
    today = _second_session(bars, second)
    at_the_bell = today & (_clock(bars) == 570)
    assert at_the_bell.sum() == 1, label
    assert gap_pct(None, bars)[at_the_bell].iloc[0] == pytest.approx(
        100 * (SECOND_BELL - FIRST_CLOSE) / FIRST_CLOSE), label


def test_the_gap_skips_a_session_that_never_opens():
    """The prior-session row means the previous observed nonempty session."""
    from nakagai.strategies.rules.primitives import gap_pct
    bars = pd.concat([_extended_session("2026-01-05", FIRST_BELL, FIRST_CLOSE),
                      _extended_session("2026-01-06", 0.0, 0.0,
                                        start="08:00", end="09:15"),
                      _extended_session("2026-01-07", SECOND_BELL, SECOND_CLOSE)])
    dark = _second_session(bars, "2026-01-06")
    assert dark.sum() == 6, "the middle session must still be pre-market only"
    after = _second_session(bars, "2026-01-07")
    want = 100 * (SECOND_BELL - FIRST_CLOSE) / FIRST_CLOSE
    assert np.allclose(gap_pct(None, bars)[after & (_clock(bars) >= 570)], want)
    # The dark session itself has no open of its own, so it has no gap.
    assert gap_pct(None, bars)[dark].isna().all()


# Both daily conventions the engine meets. Neither label sits inside
# [09:30, 16:00), so a session-scoped mask without data/schema.rth_mask's
# session-frame branch blanks every one of these rather than failing anywhere a
# reader would look.
DAILY_BARS = {"open": [100.0, 110.0, 120.0, 130.0],
              "high": [105.0, 115.0, 125.0, 135.0],
              "low": [95.0, 105.0, 115.0, 125.0],
              "close": [102.0, 112.0, 122.0, 132.0], "volume": 1000.0}


@pytest.mark.parametrize("labels, convention", [
    (["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"], "midnight UTC"),
    (["2026-01-05 05:00", "2026-01-06 05:00", "2026-01-07 05:00",
      "2026-01-08 05:00"], "midnight Eastern")])
def test_the_gap_reads_a_daily_frame_as_one_session_per_row(
        labels, convention):
    """The trap this whole design turns on.

    A daily bar is labeled at midnight under both conventions the engine meets,
    the cache's own UTC resample and Alpaca's Eastern stamp, and both sit
    outside the regular window. A wall-clock mask alone would answer all False
    on either and silently blank every session-scoped reading of every daily
    spec: no exception, no empty frame, just NaN where a level used to be.
    The gap primitive must still treat each row as one regular session.
    """
    from nakagai.strategies.rules.primitives import gap_pct
    bars = pd.DataFrame(DAILY_BARS, index=pd.DatetimeIndex(labels, tz="UTC"))
    gap = gap_pct(None, bars)
    assert pd.isna(gap.iloc[0]), convention
    assert gap.iloc[1] == pytest.approx(100 * (110.0 - 102.0) / 102.0), convention


def test_swing_high_returns_last_confirmed_swing_level():
    from nakagai.strategies.rules.primitives import swing_high
    bars = _session_bars()
    bars.loc[bars.index[20], "high"] = 200.0           # inject an obvious swing
    sh = swing_high(_ctx(bars), bars, k=3)
    assert sh.iloc[-1] == 200.0


def test_time_primitives():
    from nakagai.strategies.rules.primitives import day_of_week, minutes_into_session
    bars = _session_bars()
    ctx = _ctx(bars)
    dow = day_of_week(ctx, bars)
    assert dow.iloc[0] == 0 and dow.iloc[-1] == 1      # Mon, Tue
    mins = minutes_into_session(ctx, bars)
    assert mins.iloc[0] == 0.0 and mins.iloc[2] == 30.0


def _minutes_by_clock(bars):
    """minutes_into_session keyed by New York wall clock, for legible asserts."""
    from nakagai.strategies.rules.primitives import minutes_into_session
    mins = minutes_into_session(_ctx(bars), bars)
    ny = bars.index.tz_convert("America/New_York")
    return dict(zip([f"{t.hour:02d}:{t.minute:02d}" for t in ny], mins))


def test_minutes_into_session_is_nan_before_the_bell_and_zero_at_it():
    """The pre-market NaN, which is deliberate and load-bearing.

    On the right anchor a pre-market bar is NEGATIVE, and a negative passes
    every `< N` gate the catalog writes: first_hour_reversal gates on
    `minutes_into_session < 240` and would start firing at 08:00, an hour and a
    half before any of its premises exist. A condition over NaN reads False, so
    NaN fails closed. Fixing it here costs one `.where()`; fixing it in the
    specs would cost a lower bound in thirteen files and be forgotten in the
    fourteenth.
    """
    mins = _minutes_by_clock(_extended_hours_session())
    assert np.isnan(mins["08:00"]) and np.isnan(mins["09:15"])
    assert mins["09:30"] == 0.0
    assert mins["10:00"] == 30.0
    assert mins["15:45"] == 375.0


def test_minutes_into_session_keeps_counting_after_the_close():
    """Past 16:00 it goes on counting rather than going NaN.

    That is the reading the catalog is written against, not an oversight to
    tidy later. Every affected play exits on `minutes_into_session >= 375`,
    which is satisfied at 15:45 INSIDE the session, so blanking the
    post-market would change no exit; and every entry gate is a `< N` that a
    post-close value already fails.
    """
    mins = _minutes_by_clock(_extended_hours_session())
    assert mins["16:00"] == 390.0
    assert mins["19:45"] == 615.0
    assert mins["16:00"] >= 375.0        # the exit every session play carries


def test_minutes_into_session_on_a_session_frame_is_zero_on_every_bar():
    # One row IS the whole session there, so it opens at its own open and no
    # other number is available. vocabulary.py documents that reading and
    # spec.py refuses a 1d driving frame over it, which is what keeps a play
    # from being written against a constant.
    from nakagai.strategies.rules.primitives import minutes_into_session
    idx = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    assert (minutes_into_session(None, _flat(idx)) == 0.0).all()


@pytest.mark.parametrize("day, offset", [("2026-03-06", "EST, before"),
                                         ("2026-03-09", "EDT, after"),
                                         ("2026-10-30", "EDT, before"),
                                         ("2026-11-02", "EST, after")])
def test_the_bell_is_the_same_bell_on_both_sides_of_a_clock_change(day, offset):
    """09:30 New York, whatever UTC instant that is that week.

    The trap the anchor has to survive: the 09:30 bar is 14:30 UTC in winter
    and 13:30 UTC in summer, so anything that reaches the bell by adding a
    fixed offset to a UTC midnight lands an hour out for half the year, and a
    session play would gate on 10:30 or 08:30 without saying so. Both March and
    November are here because the two transitions move the offset in opposite
    directions and a sign error passes one of them.
    """
    bars = _extended_hours_session(day)
    mins = _minutes_by_clock(bars)
    assert np.isnan(mins["09:15"]), offset
    assert mins["09:30"] == 0.0 and mins["15:45"] == 375.0, offset


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
    from nakagai.strategies.rules.primitives import fvg_nearest
    bars = _session_bars()
    v = fvg_nearest(_ctx(bars), bars, direction="long", field="top")
    assert isinstance(v, float)                        # NaN when no unfilled FVG exists


def _flat(idx):
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=idx)


def test_day_of_week_reads_the_utc_calendar_day_on_daily_frames():
    # Session-aligned daily bars are labeled midnight UTC of their session date;
    # converting midnight UTC to NY lands on the prior evening, which would
    # shift every weekday back by one (a Monday bar read as Sunday).
    from nakagai.strategies.rules.primitives import day_of_week
    idx = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")  # Mon..Fri
    dow = day_of_week(None, _flat(idx))
    assert list(dow) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_day_of_week_reads_a_daily_frame_stamped_at_the_eastern_midnight():
    # The other daily convention: Alpaca stamps a 1Day bar at midnight Eastern,
    # which is 04:00 UTC under EDT and 05:00 under EST. Same session date, a
    # different clock on the label, so a rule reading the label's clock would
    # have to know both. Reading the FRAME knows neither.
    from nakagai.strategies.rules.primitives import day_of_week
    idx = pd.DatetimeIndex([f"2026-01-{d:02d} 05:00" for d in (5, 6, 7, 8, 9)],
                           tz="UTC")
    assert list(day_of_week(None, _flat(idx))) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_an_after_hours_bar_keeps_its_own_session_s_weekday():
    # The trap the midnight-UTC rule fell into. The caches are not RTH-only, so
    # a 19:00 EST post-market bar carries exactly the 00:00 UTC label a
    # resampled daily bar carries. Reading the label's clock answered Tuesday
    # for a Monday evening, on one bar of the session and no other. An intraday
    # frame is intraday on every bar of it.
    from nakagai.strategies.rules.primitives import day_of_week
    idx = pd.DatetimeIndex(["2026-01-05 23:00",   # NY Mon 18:00
                            "2026-01-06 00:00",   # NY Mon 19:00
                            "2026-01-06 01:00",   # NY Mon 20:00
                            "2026-01-06 14:30"],  # NY Tue 09:30
                           tz="UTC")
    assert list(day_of_week(None, _flat(idx))) == [0.0, 0.0, 0.0, 1.0]


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
    from nakagai.strategies.rules.primitives import rvol
    return rvol(_ctx(bars), bars, sessions=sessions)


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
    from nakagai.strategies.rules.vocabulary import core_vocabulary
    term = core_vocabulary().primitives["rvol"]
    assert term.args == {"sessions": (5, 60)}
    assert term.defaults == {"sessions": 20}
    assert term.session_scoped


def test_the_two_timeframe_sets_are_not_the_same_set():
    """Two rules, two sets. A foreign `tf` is refused for every session-scoped
    primitive; a session-aligned DRIVING frame is refused for the two that a
    one-bar session cannot answer. day_of_week sits in the first set and not
    the second, because a daily bar is one session and its weekday is exactly
    what the primitive promises; turnaround_tuesday, a shipped 1d play, is
    built on that reading."""
    from nakagai.strategies.rules.vocabulary import core_vocabulary
    vocabulary = core_vocabulary()
    intraday = {name for name, term in vocabulary.primitives.items()
                if term.driving_frame_intraday}
    session = {name for name, term in vocabulary.primitives.items()
               if term.session_scoped}
    assert intraday < session
    assert intraday == {"minutes_into_session", "rvol"}
    assert "day_of_week" not in intraday


def test_every_refused_primitive_carries_its_own_reason():
    """Every core intraday-only primitive has a focused refusal reason."""
    from nakagai.strategies.rules.spec import _ONE_BAR_SESSION
    from nakagai.strategies.rules.vocabulary import core_vocabulary
    intraday = {name for name, term in core_vocabulary().primitives.items()
                if term.driving_frame_intraday}
    assert set(_ONE_BAR_SESSION) == intraday
