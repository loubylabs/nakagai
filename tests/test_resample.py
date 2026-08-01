"""4h bars are DERIVED from cached 1h bars, anchored on the Eastern wall clock.

The whole design rests on one property: a 4h bucket must close at the same NY
wall-clock times all year (00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00), so
the last regular-session bucket closes exactly on the 16:00 bell in winter and
in summer alike. Alpaca's own 4Hour bars are anchored on the UTC clock, which
puts that close at 16:00 ET under EDT and 15:00 ET under EST, so a backtest
spanning a DST change measures two different instruments. The parametrized
boundary test below is the one that justifies deriving these bars locally
instead of fetching them; read it first.
"""

import pandas as pd
import pytest

from nakagai.data import sync
from nakagai.data.cache import BarCache
from nakagai.data.resample import DERIVED, resample_bars
from nakagai.data.schema import BAR_COLUMNS, EXCHANGE_TZ
from nakagai.data.sync import derive_incremental
from nakagai.engine.context import build_context

ET = EXCHANGE_TZ


def _hourly_et(start_et: str, n: int, base: float = 100.0) -> pd.DataFrame:
    """n 1h bars beginning at `start_et` NY wall clock, labeled in UTC.

    Bars advance in ABSOLUTE time (one real hour apart), which is what a
    provider returns and what makes the DST cases in this file honest.
    """
    idx = (pd.date_range(pd.Timestamp(start_et, tz=ET), periods=n, freq="1h")
           .tz_convert("UTC").rename("ts"))
    close = base + pd.Series(range(n), index=idx, dtype="float64")
    return pd.DataFrame({"open": close - 0.5, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": 1_000.0},
                        index=idx)


def _et_hours(df: pd.DataFrame) -> list[int]:
    return [ts.hour for ts in df.index.tz_convert(ET)]


# -- the boundary property, the reason this module exists ---------------------


@pytest.mark.parametrize("day,offset", [("2026-01-06", "-05:00"),   # EST
                                        ("2026-06-02", "-04:00")])  # EDT
def test_buckets_close_on_the_same_et_wall_clock_in_est_and_edt(day, offset):
    """Boundaries land on 00/04/08/12/16/20 ET on a winter date and a summer
    date, so the last regular-session bucket closes ON the bell in both."""
    src = _hourly_et(f"{day} 00:00", 30)      # the whole day, plus the next morning
    out = resample_bars(src, "4h")

    et = out.index.tz_convert(ET)
    same_day = et[et.normalize() == pd.Timestamp(day, tz=ET)]
    assert [ts.hour for ts in same_day] == [0, 4, 8, 12, 16, 20]
    # The UTC offset of the labels is the day's real offset, i.e. the labels are
    # wall-clock anchored rather than shifted by an hour half the year.
    assert {ts.strftime("%z") for ts in same_day} == {offset.replace(":", "")}
    # The 12:00 bucket is the last one inside the regular session, and it ends
    # exactly at the 16:00 close. This is the fact an absolute-time resample
    # gets wrong for half the year.
    noon = same_day[same_day.hour == 12][0]
    assert noon + pd.Timedelta(hours=4) == pd.Timestamp(f"{day} 16:00", tz=ET)


def test_boundaries_hold_across_a_spring_forward_and_a_fall_back():
    """Wall-clock anchoring is the point, so the grid must survive both 2026 US
    transitions: 2026-03-08 (spring forward) and 2026-11-01 (fall back)."""
    for start in ("2026-03-07 00:00", "2026-10-31 00:00"):
        out = resample_bars(_hourly_et(start, 24 * 3), "4h")
        assert set(_et_hours(out)) == {0, 4, 8, 12, 16, 20}
        assert out.index.is_monotonic_increasing
        assert not out.index.has_duplicates


# -- the still-forming bucket -------------------------------------------------


def test_a_still_forming_bucket_is_withheld():
    """1h bars through the 13:00 label cover only 13:00-14:00, so the
    12:00-16:00 bucket is incomplete. Publishing it would put a partial bar in
    the cache, where every reader downstream takes it for a closed one."""
    out = resample_bars(_hourly_et("2026-06-02 08:00", 6), "4h")   # labels 08..13
    assert _et_hours(out) == [8]


def test_the_withheld_bucket_is_published_once_its_final_hour_arrives():
    out = resample_bars(_hourly_et("2026-06-02 08:00", 8), "4h")   # labels 08..15
    assert _et_hours(out) == [8, 12]


def test_a_bucket_held_back_by_a_gap_is_written_complete_on_a_later_derive(tmp_path):
    """Self-healing, end to end: the cache never holds the partial version."""
    cache = BarCache(tmp_path)
    src = _hourly_et("2026-06-02 08:00", 8)                         # labels 08..15
    cache.upsert("SPY", "1h", src.iloc[:6])                         # through 13:00
    derive_incremental(cache, "SPY", "4h")
    assert _et_hours(cache.load("SPY", "4h")) == [8]

    cache.upsert("SPY", "1h", src.iloc[6:])                         # 14:00, 15:00
    derive_incremental(cache, "SPY", "4h")
    derived = cache.load("SPY", "4h")
    assert _et_hours(derived) == [8, 12]
    # and the newly complete bucket is the whole 12:00-16:00 window, not the
    # half of it that existed when the bucket was first held back
    noon = src.loc[src.index >= pd.Timestamp("2026-06-02 12:00", tz=ET)]
    assert derived["volume"].iloc[-1] == noon["volume"].sum()
    assert derived["high"].iloc[-1] == noon["high"].max()


# -- empty buckets ------------------------------------------------------------


def test_overnight_buckets_with_no_prints_produce_no_rows():
    """resample().agg() emits a row per period whether or not anything traded,
    and validate_bars would accept those NaN rows as real bars."""
    src = pd.concat([_hourly_et("2026-06-02 09:00", 7),      # 09:00-15:00
                     _hourly_et("2026-06-03 09:00", 7)])
    out = resample_bars(src, "4h")
    assert not out.isna().to_numpy().any()
    assert _et_hours(out) == [8, 12, 8, 12]                  # no 16/20/00/04 rows
    assert (out["volume"] > 0).all()


# -- aggregation --------------------------------------------------------------


def test_ohlcv_aggregates_first_max_min_last_sum():
    idx = (pd.date_range(pd.Timestamp("2026-06-02 08:00", tz=ET), periods=4, freq="1h")
           .tz_convert("UTC").rename("ts"))
    src = pd.DataFrame({"open": [10.0, 11.0, 12.0, 13.0],
                        "high": [15.0, 19.0, 16.0, 17.0],
                        "low": [9.0, 8.5, 8.0, 9.5],
                        "close": [11.0, 12.0, 13.0, 14.0],
                        "volume": [100.0, 200.0, 300.0, 400.0]}, index=idx)
    out = resample_bars(src, "4h")
    assert len(out) == 1
    assert list(out.columns) == BAR_COLUMNS
    row = out.iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"], row["volume"]) == (
        10.0, 19.0, 8.0, 14.0, 1000.0)


def test_an_empty_source_derives_nothing():
    out = resample_bars(_hourly_et("2026-06-02 08:00", 0), "4h")
    assert out.empty
    assert list(out.columns) == BAR_COLUMNS
    assert str(out.index.tz) == "UTC"


# -- the derived-timeframe declaration ---------------------------------------


def test_derived_declares_where_4h_comes_from():
    """One place says which timeframes are computed rather than fetched, so a
    caller asks instead of hardcoding the pair."""
    assert DERIVED["4h"] == "1h"
    assert "1h" not in DERIVED and "15m" not in DERIVED and "1d" not in DERIVED


def test_resampling_a_timeframe_nobody_derives_is_refused():
    with pytest.raises(ValueError, match="2h"):
        resample_bars(_hourly_et("2026-06-02 08:00", 8), "2h")


# -- derive_incremental -------------------------------------------------------


def test_derive_incremental_is_idempotent(tmp_path):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1h", _hourly_et("2026-06-01 08:00", 60))
    derive_incremental(cache, "SPY", "4h")
    once = cache.load("SPY", "4h")
    derive_incremental(cache, "SPY", "4h")
    pd.testing.assert_frame_equal(once, cache.load("SPY", "4h"))


def test_derive_incremental_covers_the_whole_history_on_a_cold_cache(tmp_path):
    cache = BarCache(tmp_path)
    src = _hourly_et("2026-05-01 00:00", 24 * 20)
    cache.upsert("SPY", "1h", src)
    derive_incremental(cache, "SPY", "4h")
    pd.testing.assert_frame_equal(cache.load("SPY", "4h"), resample_bars(src, "4h"),
                                  check_freq=False)   # parquet drops the index freq


def test_derive_incremental_rederives_only_the_tail(tmp_path, monkeypatch):
    """Three years of 1h bars must not be re-derived every cycle. The resume
    point is the last derived bucket's own label, which is a bucket boundary,
    so the trailing window can never start mid-bucket and write a partial bar
    over a complete one."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1h", _hourly_et("2026-05-01 00:00", 24 * 20))
    derive_incremental(cache, "SPY", "4h")

    seen: list[int] = []
    real = sync.resample_bars

    def spy(src, timeframe):
        seen.append(len(src))
        return real(src, timeframe)

    monkeypatch.setattr(sync, "resample_bars", spy)
    cache.upsert("SPY", "1h", _hourly_et("2026-05-21 00:00", 3))
    derive_incremental(cache, "SPY", "4h")
    assert seen == [7], "the last derived bucket's 4 hours, plus the 3 new ones"


def test_deriving_in_steps_matches_deriving_in_one_pass(tmp_path):
    """No seam: an incrementally built 4h cache is byte-identical to one
    derived from the full 1h history at once. This is what makes the derived
    parquet the backtester reads the same object the live path wrote."""
    full = _hourly_et("2026-06-01 00:00", 24 * 6)
    stepwise = BarCache(tmp_path / "stepwise")
    for i in range(0, len(full), 5):
        stepwise.upsert("SPY", "1h", full.iloc[i:i + 5])
        derive_incremental(stepwise, "SPY", "4h")
    one_pass = BarCache(tmp_path / "one_pass")
    one_pass.upsert("SPY", "1h", full)
    derive_incremental(one_pass, "SPY", "4h")
    pd.testing.assert_frame_equal(stepwise.load("SPY", "4h"),
                                  one_pass.load("SPY", "4h"))


def test_derive_incremental_writes_nothing_without_source_bars(tmp_path):
    cache = BarCache(tmp_path)
    assert derive_incremental(cache, "SPY", "4h") == 0
    assert not cache.path("SPY", "4h").exists()


def test_derive_incremental_writes_nothing_when_every_bucket_is_still_forming(tmp_path):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "1h", _hourly_et("2026-06-02 08:00", 2))   # 08:00, 09:00
    assert derive_incremental(cache, "SPY", "4h") == 0
    assert not cache.path("SPY", "4h").exists()


# -- the engine reads a derived bar like any other ---------------------------


def test_the_engine_gates_a_4h_bar_at_label_plus_four_hours(tmp_path, make_bars):
    """4h is NOT session-aligned: an ET-anchored bucket labeled 12:00 closes at
    16:00, so plain label + delta is the right visibility rule and closed_before
    needs no special case for it."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(40, "15m", start="2026-06-02 13:30"))
    cache.upsert("SPY", "1h", _hourly_et("2026-06-02 04:00", 12))
    cache.upsert("SPY", "1d", make_bars(5, "1d", start="2026-05-26 00:00"))
    derive_incremental(cache, "SPY", "4h")

    noon = pd.Timestamp("2026-06-02 12:00", tz=ET)
    assert noon in cache.load("SPY", "4h").index                    # premise

    before = build_context(cache, "SPY", pd.Timestamp("2026-06-02 15:45", tz=ET))
    assert noon not in before.bars["4h"].index
    on_the_bell = build_context(cache, "SPY", pd.Timestamp("2026-06-02 16:00", tz=ET))
    assert on_the_bell.bars["4h"].index[-1] == noon
