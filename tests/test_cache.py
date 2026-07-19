import pandas as pd

from nakagai.data.cache import BarCache, MemoryBars


def test_upsert_then_load_roundtrip(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    df = make_bars(10)
    assert cache.upsert("SPY", "15m", df) == 10
    loaded = cache.load("SPY", "15m")
    pd.testing.assert_frame_equal(loaded, df)


def test_upsert_merges_and_overwrites_overlap(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    first = make_bars(10)
    cache.upsert("SPY", "15m", first)
    # 5 overlapping bars with revised closes + 5 new
    second = make_bars(10, start="2026-06-01 14:45")
    second["close"] += 99.0
    cache.upsert("SPY", "15m", second)
    loaded = cache.load("SPY", "15m")
    assert len(loaded) == 15  # 5 old + 10 second (5 overlapped, revised wins)
    assert loaded.loc[pd.Timestamp("2026-06-01 14:45", tz="UTC"), "close"] == second["close"].iloc[0]


def test_interpolated_rows_dropped(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    df = make_bars(6)
    df["interpolated"] = [False, False, True, True, False, False]
    assert cache.upsert("SPY", "15m", df) == 4
    assert len(cache.load("SPY", "15m")) == 4


def test_small_hourly_cache_roundtrip(tmp_path, make_bars):
    # Regression: load() on a 1-2 row cache must not crash (infer_freq needs >= 3
    # timestamps), and freq restoration must work for non-15m timeframes.
    cache = BarCache(tmp_path)
    first = make_bars(2, timeframe="1h")
    assert cache.upsert("SPY", "1h", first) == 2
    # second upsert exercises load() against the 2-row cache
    second = make_bars(2, timeframe="1h", start="2026-06-01 14:30")
    second["close"] += 99.0
    assert cache.upsert("SPY", "1h", second) == 2
    loaded = cache.load("SPY", "1h")
    expected = pd.concat([first, second])
    expected = expected[~expected.index.duplicated(keep="last")].sort_index()
    expected.index.freq = "1h"
    pd.testing.assert_frame_equal(loaded, expected)


def test_memory_bars_mirrors_the_barcache_contract(make_bars):
    df = make_bars(4)
    mem = MemoryBars({("SPY", "15m"): df})
    pd.testing.assert_frame_equal(mem.load("SPY", "15m"), df)
    missing = mem.load("SPY", "1d")   # same shape BarCache.load gives for a missing file
    assert missing.empty and list(missing.columns) == ["open", "high", "low", "close", "volume"]
    assert str(missing.index.tz) == "UTC"


def test_coverage_reports_cached_span(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    df = make_bars(10)
    cache.upsert("SPY", "15m", df)
    first, last = cache.coverage("SPY", "15m")
    assert first == df.index[0]
    assert last == df.index[-1]


def test_coverage_none_when_missing(tmp_path):
    assert BarCache(tmp_path).coverage("NOPE", "15m") is None


def test_load_missing_returns_empty_schema(tmp_path):
    cache = BarCache(tmp_path)
    df = cache.load("NOPE", "1d")
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
