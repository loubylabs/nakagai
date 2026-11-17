import pandas as pd
import pytest

from nakagai.data.cache import BarCache, MemoryBars


def _bars_at(stamps: list[str], closes: list[float], volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(stamps, name="ts"),
    )


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


def test_daily_upsert_collapses_summer_labels_to_new_york_midnight(tmp_path):
    cache = BarCache(tmp_path)
    mixed = _bars_at(
        ["2026-07-14T00:00:00Z", "2026-07-14T04:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    assert cache.upsert("SPY", "1d", mixed) == 2

    loaded = cache.load("SPY", "1d")
    expected = _bars_at(["2026-07-14T04:00:00Z"], [200.0], [2_000.0])
    pd.testing.assert_frame_equal(loaded, expected)


def test_daily_upsert_collapses_winter_labels_to_new_york_midnight(tmp_path):
    cache = BarCache(tmp_path)
    mixed = _bars_at(
        ["2026-01-14T00:00:00Z", "2026-01-14T05:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    cache.upsert("SPY", "1d", mixed)

    expected = _bars_at(["2026-01-14T05:00:00Z"], [200.0], [2_000.0])
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)


def test_daily_upsert_keeps_incoming_row_for_same_session(tmp_path):
    cache = BarCache(tmp_path)
    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-14T04:00:00Z"], [100.0], [1_000.0]),
    )

    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-14T00:00:00Z"], [101.0], [3_000.0]),
    )

    expected = _bars_at(["2026-07-14T04:00:00Z"], [101.0], [3_000.0])
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)
    assert cache.rebased == set()


def test_daily_upsert_repairs_mixed_sessions_already_on_disk(tmp_path):
    cache = BarCache(tmp_path)
    cache.path("SPY", "1d").parent.mkdir(parents=True, exist_ok=True)
    _bars_at(
        [
            "2026-07-14T00:00:00Z",
            "2026-07-14T04:00:00Z",
            "2026-07-15T04:00:00Z",
        ],
        [100.0, 200.0, 300.0],
        [1_000.0, 2_000.0, 3_000.0],
    ).to_parquet(cache.path("SPY", "1d"))

    cache.upsert(
        "SPY",
        "1d",
        _bars_at(["2026-07-16T04:00:00Z"], [400.0], [4_000.0]),
    )

    expected = _bars_at(
        [
            "2026-07-14T04:00:00Z",
            "2026-07-15T04:00:00Z",
            "2026-07-16T04:00:00Z",
        ],
        [200.0, 300.0, 400.0],
        [2_000.0, 3_000.0, 4_000.0],
    )
    expected.index.freq = "D"
    pd.testing.assert_frame_equal(cache.load("SPY", "1d"), expected)


def test_intraday_upsert_preserves_exact_timestamps(tmp_path):
    cache = BarCache(tmp_path)
    bars = _bars_at(
        ["2026-07-14T00:00:00Z", "2026-07-14T04:00:00Z"],
        [100.0, 200.0],
        [1_000.0, 2_000.0],
    )

    cache.upsert("SPY", "15m", bars)

    pd.testing.assert_frame_equal(cache.load("SPY", "15m"), bars)


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


# -- a moved price basis ------------------------------------------------------
# The vendor split-adjusts at fetch time. After a split, a re-fetched overlap bar
# comes back with its open moved by the split ratio, and every cached row is on
# a basis the vendor no longer serves. upsert then keeps only the incoming frame.


def test_a_split_in_the_overlap_bar_truncates_to_the_incoming_frame(
        tmp_path, make_bars, caplog):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10))
    assert cache.rebased == set()
    tail = make_bars(10)[-1:] / 2                       # the last bar, halved
    tail = pd.concat([tail, make_bars(2, start="2026-06-01 16:00", base=50.5)])

    with caplog.at_level("WARNING", logger="nakagai.data.cache"):
        assert cache.upsert("SPY", "15m", tail) == 3

    pd.testing.assert_frame_equal(cache.load("SPY", "15m"), tail,
                                  check_freq=False)
    assert cache.rebased == {("SPY", "15m")}
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "SPY" in warnings[0].getMessage() and "15m" in warnings[0].getMessage()
    assert "0.5" in warnings[0].getMessage()


def test_a_full_range_over_a_mixed_file_replaces_it_whole(tmp_path, make_bars):
    """An old split left the file half on each basis. A full-range fetch
    disagrees on the old rows, so none of them survive."""
    cache = BarCache(tmp_path)
    truth = make_bars(10)
    mixed = pd.concat([truth.iloc[:5], truth.iloc[5:] / 2])
    mixed.to_parquet(cache.path("SPY", "15m"))

    full = truth / 2
    cache.upsert("SPY", "15m", full)

    pd.testing.assert_frame_equal(cache.load("SPY", "15m"), full)
    assert cache.rebased == {("SPY", "15m")}


def test_an_ordinary_overlap_with_a_revised_close_merges(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10))
    final = make_bars(10)[-1:].copy()
    final["close"] += 0.5                              # the forming bar finished
    cache.upsert("SPY", "15m", pd.concat(
        [final, make_bars(1, start="2026-06-01 16:00", base=101.0)]))

    loaded = cache.load("SPY", "15m")
    assert len(loaded) == 11
    assert loaded["close"].iloc[9] == final["close"].iloc[0]
    assert cache.rebased == set()


@pytest.mark.parametrize("factor", [1.14, 1 / 1.14])
def test_an_open_moved_inside_the_band_merges(tmp_path, make_bars, factor):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10))
    moved = make_bars(10)[-1:].copy()
    moved["open"] *= factor
    cache.upsert("SPY", "15m", moved)
    assert len(cache.load("SPY", "15m")) == 10
    assert cache.rebased == set()


@pytest.mark.parametrize("old_open, new_open", [
    (0.0, 50.0), (100.0, 0.0), (float("nan"), 50.0), (100.0, float("nan")),
    (-1.0, 50.0), (100.0, -1.0),
])
def test_unusable_opens_never_trigger(tmp_path, make_bars, old_open, new_open):
    cache = BarCache(tmp_path)
    seeded = make_bars(10)
    seeded.iloc[-1, seeded.columns.get_loc("open")] = old_open
    cache.upsert("SPY", "15m", seeded)
    incoming = make_bars(10)[-1:].copy()
    incoming["open"] = new_open
    cache.upsert("SPY", "15m", incoming)
    assert len(cache.load("SPY", "15m")) == 10
    assert cache.rebased == set()


def test_weekly_overlap_is_compared_by_week_not_label(tmp_path):
    """A week first labeled Tuesday and re-derived under Monday is the same
    bar, so its opens are compared."""
    tue = pd.Timestamp("2026-06-02", tz="America/New_York").tz_convert("UTC")
    mon = pd.Timestamp("2026-06-01", tz="America/New_York").tz_convert("UTC")
    prior = pd.Timestamp("2026-05-26", tz="America/New_York").tz_convert("UTC")

    def week(stamps, opens):
        return pd.DataFrame({"open": opens, "high": [o + 1 for o in opens],
                             "low": [o - 1 for o in opens], "close": opens,
                             "volume": 1_000.0},
                            index=pd.DatetimeIndex(stamps, name="ts"))

    same = BarCache(tmp_path / "same")
    same.upsert("SPY", "1w", week([prior, tue], [99.0, 100.0]))
    same.upsert("SPY", "1w", week([mon], [101.0]))
    assert list(same.load("SPY", "1w").index) == [prior, mon]
    assert same.rebased == set()

    split = BarCache(tmp_path / "split")
    split.upsert("SPY", "1w", week([prior, tue], [99.0, 100.0]))
    split.upsert("SPY", "1w", week([mon], [50.0]))
    assert list(split.load("SPY", "1w").index) == [mon]
    assert split.rebased == {("SPY", "1w")}


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
        # Skip interpreter teardown. Arrow's thread pool can abort there
        # ("terminate called without an active exception", exit -6) after
        # the write is already on disk, apache/arrow#34314 (#1027 on the
        # platform tracker). A failed upsert still raises above and exits 1.
        "sys.stdout.flush()\n"
        "import os; os._exit(0)\n"
    )
    stamps = [f"2026-06-01T13:{m:02d}:00Z" for m in (30, 45, 0, 15)]
    deadline = time.time() + 5.0        # generous: covers interpreter + pandas import
    procs = [subprocess.Popen([sys.executable, "-c", script, ts, str(tmp_path), str(deadline)])
             for ts in stamps]
    for p in procs:
        assert p.wait(timeout=60) == 0

    assert len(BarCache(tmp_path).load("SPY", "15m")) == len(stamps)
