import pandas as pd
import pytest

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


def test_upsert_holds_the_pair_lock_for_the_read_modify_write(tmp_path, make_bars, monkeypatch):
    """upsert is a read-concat-write, which is exactly what nakagai/filelock.py
    was written for. Asserted directly rather than only through a race, because
    a race that happens to not collide is a green test that proves nothing."""
    import contextlib
    from pathlib import Path

    import nakagai.data.cache as cache_mod

    held = []
    real_lock = cache_mod.file_lock

    @contextlib.contextmanager
    def recording_lock(target, *a, **kw):
        held.append(Path(target))
        with real_lock(target, *a, **kw):
            yield

    monkeypatch.setattr(cache_mod, "file_lock", recording_lock)
    BarCache(tmp_path).upsert("SPY", "15m", make_bars(3))
    assert held == [tmp_path / "SPY_15m.parquet"]


def test_upsert_is_atomic_so_a_failed_write_leaves_the_prior_parquet_intact(
        tmp_path, make_bars, monkeypatch):
    """A crash mid-write must leave the previous parquet readable. Writing in
    place truncates it, and every later read of that pair raises."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(3))

    real_to_parquet = pd.DataFrame.to_parquet

    def explode(self, path, *a, **kw):
        if str(path).endswith(".tmp"):
            raise OSError("disk full")
        return real_to_parquet(self, path, *a, **kw)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError):
        cache.upsert("SPY", "15m", make_bars(3, start="2026-06-01 14:15"))
    monkeypatch.undo()

    # The three bars from the successful first write survived the failed second.
    assert len(cache.load("SPY", "15m")) == 3


def test_concurrent_upserts_do_not_lose_bars(tmp_path):
    """Four processes upserting the same pair must all persist. An unlocked
    read-concat-write keeps only whichever finished last.

    The writers spin to a shared wall-clock deadline before touching the cache,
    so they enter the critical section together. Without that barrier each one
    spends its first few hundred ms importing pandas and they serialize by
    accident, which would make this test pass against the very bug it exists to
    catch.
    """
    import subprocess
    import sys
    import time

    script = (
        "import sys, time\n"
        "import pandas as pd\n"
        "from nakagai.data.cache import BarCache\n"
        "ts, root, deadline = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
        "df = pd.DataFrame({'open': [1.0], 'high': [1.5], 'low': [0.5],\n"
        "                   'close': [1.2], 'volume': [100.0]},\n"
        "                  index=pd.DatetimeIndex([ts], tz='UTC', name='ts'))\n"
        "while time.time() < deadline:\n"
        "    time.sleep(0.002)\n"
        "BarCache(root).upsert('SPY', '15m', df)\n"
    )
    stamps = [f"2026-06-01T13:{m:02d}:00Z" for m in (30, 45, 0, 15)]
    deadline = time.time() + 5.0        # generous: covers interpreter + pandas import
    procs = [subprocess.Popen([sys.executable, "-c", script, ts, str(tmp_path), str(deadline)])
             for ts in stamps]
    for p in procs:
        assert p.wait(timeout=60) == 0

    assert len(BarCache(tmp_path).load("SPY", "15m")) == len(stamps)
