import pandas as pd
import pytest

from nakagai.data.schema import (
    BAR_COLUMNS,
    DEFAULT_TIMEFRAMES,
    EXCHANGE_TZ,
    TIMEFRAMES,
    TimeframeSet,
    _is_session_frame,
    rth_mask,
    session_open,
    session_opens,
    validate_bars,
)


def test_validate_passes_and_sorts(make_bars):
    df = make_bars(5)
    shuffled = df.sample(frac=1, random_state=1)
    out = validate_bars(shuffled)
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "ts"


def test_validate_rejects_naive_index(make_bars):
    df = make_bars(3)
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        validate_bars(df)


def test_validate_rejects_missing_columns(make_bars):
    df = make_bars(3).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        validate_bars(df)


def test_constants():
    assert TIMEFRAMES == ("15m", "1h", "4h", "1d")


def test_default_timeframes_match_legacy_axis():
    assert DEFAULT_TIMEFRAMES.all == ("15m", "1h", "4h", "1d")
    assert DEFAULT_TIMEFRAMES.step == pd.Timedelta(minutes=15)
    assert DEFAULT_TIMEFRAMES.deltas["1h"] == pd.Timedelta(hours=1)
    assert DEFAULT_TIMEFRAMES.deltas["4h"] == pd.Timedelta(hours=4)
    assert DEFAULT_TIMEFRAMES.session_aligned == frozenset({"1d"})
    assert TIMEFRAMES == DEFAULT_TIMEFRAMES.all


def test_four_hour_is_not_session_aligned():
    """4h buckets are anchored on the Eastern wall clock (nakagai/data/resample),
    so a bucket labeled 12:00 closes at 16:00: label + delta is the whole
    visibility rule and closed_before needs no special case for it."""
    assert "4h" not in DEFAULT_TIMEFRAMES.session_aligned
    assert DEFAULT_TIMEFRAMES.deltas["4h"] == pd.Timedelta(hours=4)


def test_timeframe_set_validation():
    with pytest.raises(ValueError):
        TimeframeSet(driving="1h")  # driving needs a delta
    with pytest.raises(ValueError):
        TimeframeSet(driving="1d", session_aligned=frozenset({"1d"}))  # driving cannot be session-aligned
    with pytest.raises(ValueError):
        TimeframeSet(driving="15m", higher=("2h",),
                     deltas={"15m": pd.Timedelta(minutes=15)})  # 2h has no delta and is not session-aligned


def test_timeframe_set_custom_axis():
    tfs = TimeframeSet(driving="1h", higher=("1d",),
                       deltas={"1h": pd.Timedelta(hours=1)},
                       session_aligned=frozenset({"1d"}))
    assert tfs.all == ("1h", "1d")
    assert tfs.step == pd.Timedelta(hours=1)


def _utc(when: str) -> pd.Timestamp:
    return pd.Timestamp(when, tz="UTC")


# The two Sundays a year the New York offset moves, which is where every naive
# session-open construction goes wrong and nowhere else. In 2026 the second
# Sunday in March is the 8th (02:00 EST becomes 03:00 EDT) and the first Sunday
# in November is the 1st (02:00 EDT becomes 01:00 EST).
SPRING_FORWARD = "2026-03-08"
FALL_BACK = "2026-11-01"


def test_session_opens_is_the_bell_of_each_bars_own_session():
    """Every bar of a session answers the same instant, whether it sits before
    the bell, on it, or hours after. 14:30 UTC is 09:30 New York in winter."""
    idx = pd.DatetimeIndex(["2026-01-06 13:00",    # 08:00 NY, a pre-market print
                            "2026-01-06 14:30",    # 09:30 NY, the open itself
                            "2026-01-06 20:45",    # 15:45 NY, the session's last
                            "2026-01-07 14:30"],   # the next session's open
                           tz="UTC")
    assert list(session_opens(idx)) == [_utc("2026-01-06 14:30")] * 3 + [
        _utc("2026-01-07 14:30")]


def test_session_opens_holds_the_wall_clock_across_the_spring_forward():
    """The bell is a wall-clock fact, so it stays 09:30 in New York through a
    DST change and the UTC instant is what moves.

    This is the case that fails if the construction ever goes back to adding a
    9h30m Timedelta to a tz-aware midnight: that arithmetic runs on the
    underlying UTC instants, so local midnight EST plus 9h30m lands at 10:30
    EDT here, an hour after the open, and the wrong side of every session
    anchor built on it.
    """
    got = session_opens(pd.DatetimeIndex([f"{SPRING_FORWARD} 18:00"], tz="UTC"))[0]
    assert got == _utc(f"{SPRING_FORWARD} 13:30")
    assert got.tz_convert(EXCHANGE_TZ).strftime("%H:%M") == "09:30"


def test_session_opens_holds_the_wall_clock_across_the_fall_back():
    """The mirror case, and it fails the other way under the same mistake:
    local midnight EDT plus 9h30m lands at 08:30 EST, an hour BEFORE the open,
    which quietly pulls pre-market prints inside the session."""
    got = session_opens(pd.DatetimeIndex([f"{FALL_BACK} 18:00"], tz="UTC"))[0]
    assert got == _utc(f"{FALL_BACK} 14:30")
    assert got.tz_convert(EXCHANGE_TZ).strftime("%H:%M") == "09:30"


def test_the_two_dst_sundays_put_the_same_bell_an_hour_apart_in_utc():
    """Same wall time on both, one hour apart as UTC instants. Stated as its
    own assertion because "always 09:30 local" and "always the same UTC
    instant" are both plausible-looking readings and only the first is true."""
    spring = session_opens(pd.DatetimeIndex([f"{SPRING_FORWARD} 18:00"], tz="UTC"))[0]
    fall = session_opens(pd.DatetimeIndex([f"{FALL_BACK} 18:00"], tz="UTC"))[0]
    assert (fall.hour * 60 + fall.minute) - (spring.hour * 60 + spring.minute) == 60


def test_session_open_is_the_scalar_face_of_session_opens():
    """The scalar form is a one-row call into the vector form, so a
    disagreement between them is impossible rather than merely unlikely."""
    idx = pd.DatetimeIndex(["2026-01-06 13:00", "2026-01-06 20:45",
                            f"{SPRING_FORWARD} 18:00", f"{FALL_BACK} 18:00"],
                           tz="UTC")
    assert [session_open(ts) for ts in idx] == list(session_opens(idx))


def test_session_open_is_keyed_on_the_session_and_not_on_the_bar():
    """The scalar form memoizes, so what it keys on is behavior worth pinning:
    two bars of one session share an answer, and the next session gets its own.
    A bar before the bell belongs to the session it is waiting for, not to the
    one that closed the evening before."""
    premarket = session_open(_utc("2026-01-06 13:00"))    # 08:00 NY
    afternoon = session_open(_utc("2026-01-06 20:45"))    # 15:45 NY, same session
    evening = session_open(_utc("2026-01-06 23:00"))      # 18:00 NY, still the 6th
    assert premarket == afternoon == evening == _utc("2026-01-06 14:30")
    assert session_open(_utc("2026-01-07 13:00")) == _utc("2026-01-07 14:30")


def test_session_open_answers_what_it_did_before_it_moved():
    """Behavior of the scalar form, pinned across its move down from
    strategies/util so the refactor stays a move and not a change."""
    assert session_open(_utc("2026-01-06 20:45")) == _utc("2026-01-06 14:30")
    assert session_open(_utc(f"{SPRING_FORWARD} 18:00")) == _utc(f"{SPRING_FORWARD} 13:30")
    assert session_open(_utc(f"{FALL_BACK} 18:00")) == _utc(f"{FALL_BACK} 14:30")


def test_rth_mask_is_half_open_on_the_session_bounds():
    """A bar is labeled by its OPEN, so 15:45 is inside the session and 16:00
    is already the post-market. The pre-market print is out for the same
    reason the caches carry it at all: they are not RTH-only."""
    idx = pd.DatetimeIndex(["2026-01-06 13:00",    # 08:00 NY, pre-market
                            "2026-01-06 14:30",    # 09:30 NY, the open
                            "2026-01-06 20:45",    # 15:45 NY, the session's last
                            "2026-01-06 21:00",    # 16:00 NY, the bell
                            "2026-01-07 00:00"],   # 19:00 NY, post-market
                           tz="UTC")
    assert list(rth_mask(idx)) == [False, True, True, False, False]


def test_rth_mask_is_all_true_on_a_session_frame():
    """A session bar's label is not a time inside its session, so asking where
    it falls in [09:30, 16:00) is the wrong question and the honest answer is
    that the whole row is the session.

    Both daily conventions are here because both fail the clock test and they
    fail it at different hours: the cache's own resample stamps midnight UTC
    and Alpaca's 1Day bars stamp midnight Eastern. Without the session-frame
    branch this mask is all False on either, which does not raise anywhere; it
    silently blanks every session-scoped reading on every daily spec. gap_pct
    and vwap both run on 1d frames today. Generic prior-session window
    aggregates preserve the same one-row-is-one-session rule through their
    `session_aligned` branch.
    """
    midnight_utc = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    eastern_midnight = pd.DatetimeIndex(
        [f"2026-01-{d:02d} 05:00" for d in (5, 6, 7, 8, 9)], tz="UTC")
    for idx in (midnight_utc, eastern_midnight):
        mask = rth_mask(idx)
        assert mask.dtype == bool
        assert mask.index.equals(idx)
        assert mask.all()


def test_a_frame_is_read_as_daily_only_when_it_holds_one_bar_per_session():
    daily = pd.date_range("2026-01-05", periods=5, freq="B", tz="UTC")
    assert _is_session_frame(daily)
    intraday = pd.date_range("2026-01-05 14:30", periods=5, freq="15min",
                             tz="UTC")
    assert not _is_session_frame(intraday)
    # An intraday frame whose rows happen to sit on different dates is still
    # one bar per date, and that is the honest reading: nothing in the frame
    # says otherwise.
    assert _is_session_frame(pd.DatetimeIndex(
        ["2026-01-05 14:30", "2026-01-06 15:30"], tz="UTC"))
    # Too short to carry evidence of its own cadence.
    assert not _is_session_frame(daily[:1])
