"""fetch_incremental: syncs resume from the last cached bar instead of
re-pulling the whole configured range on every run (Alpaca free tier is
200 req/min per account, shared by every machine using the key)."""

import pandas as pd

from nakagai.data.base import DataProvider
from nakagai.data.cache import BarCache
from nakagai.data.sync import fetch_incremental


class CapturingProvider(DataProvider):
    name = "capturing"

    def __init__(self, make_bars):
        self._make = make_bars
        self.calls: list[tuple] = []

    def fetch_bars(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))
        return self._make(4, start=start.strftime("%Y-%m-%d %H:%M"))


START = pd.Timestamp("2026-06-01 13:30", tz="UTC")
END = pd.Timestamp("2026-06-02", tz="UTC")


def test_empty_cache_fetches_the_full_range(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END)
    assert p.calls == [("SPY", "15m", START, END)]


def test_resumes_from_the_last_cached_bar(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    seeded = make_bars(10)
    cache.upsert("SPY", "15m", seeded)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END)
    # the last bar is re-fetched: it may have been cached while still forming
    assert p.calls[0][2] == seeded.index[-1]
    assert p.calls[0][3] == END


def test_stale_cache_resumes_from_last_bar_not_requested_start(tmp_path, make_bars):
    """A start after the cache's last bar must not win, or the cache gets a
    hole that later incremental syncs would never fill."""
    cache = BarCache(tmp_path)
    seeded = make_bars(4, start="2026-05-01 13:30")
    cache.upsert("SPY", "15m", seeded)
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", pd.Timestamp("2026-05-20", tz="UTC"), END)
    assert p.calls[0][2] == seeded.index[-1]


def test_start_before_cached_history_forces_a_full_fetch(tmp_path, make_bars):
    """Widening the configured start must backfill the older range."""
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(4))
    p = CapturingProvider(make_bars)
    earlier = pd.Timestamp("2026-05-01", tz="UTC")
    fetch_incremental(cache, p, "SPY", "15m", earlier, END)
    assert p.calls[0][2] == earlier


def test_full_ignores_the_cache(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10))
    p = CapturingProvider(make_bars)
    fetch_incremental(cache, p, "SPY", "15m", START, END, full=True)
    assert p.calls == [("SPY", "15m", START, END)]
