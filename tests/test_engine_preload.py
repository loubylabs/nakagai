import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.engine.context import PreloadedBars, build_context
from nakagai.engine.engine import Engine
from nakagai.strategies.base import Strategy


class CountingCache:
    """BarCache wrapper that counts load() calls."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def load(self, symbol, timeframe):
        self.calls += 1
        return self.inner.load(symbol, timeframe)


class Quiet(Strategy):
    name = "quiet"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        return []


def test_preloaded_bars_matches_cache(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(10, "15m"))
    view = PreloadedBars(cache, "SPY")
    pd.testing.assert_frame_equal(view.load("SPY", "15m"), cache.load("SPY", "15m"))
    assert view.load("SPY", "1h").empty  # missing timeframe -> empty schema frame


def test_preloaded_context_equals_cache_context(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    cache.upsert("SPY", "15m", make_bars(20, "15m", start="2026-06-01 13:30"))
    cache.upsert("SPY", "1h", make_bars(8, "1h", start="2026-06-01 13:00"))
    cache.upsert("SPY", "1d", make_bars(5, "1d", start="2026-05-26 00:00"))
    now = pd.Timestamp("2026-06-01 15:00", tz="UTC")
    a = build_context(cache, "SPY", now)
    b = build_context(PreloadedBars(cache, "SPY"), "SPY", now)
    pd.testing.assert_frame_equal(a.bars["15m"], b.bars["15m"])
    pd.testing.assert_frame_equal(a.bars["1h"], b.bars["1h"])
    pd.testing.assert_frame_equal(a.bars["1d"], b.bars["1d"])


def test_engine_loads_each_timeframe_once(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    df = make_bars(20, "15m", start="2026-06-01 13:30")
    cache.upsert("SPY", "15m", df)
    counting = CountingCache(cache)
    eng = Engine(Quiet(), counting, "SPY", df.index[0], df.index[-1] + pd.Timedelta(minutes=15))
    eng.run()
    # one load per timeframe on the axis, independent of bar count
    assert counting.calls == len(DEFAULT_TIMEFRAMES.all)
