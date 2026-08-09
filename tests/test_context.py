import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.engine.context import build_context


def _fill(cache, make_bars):
    cache.upsert("SPY", "15m", make_bars(20, "15m", start="2026-06-01 13:30"))
    cache.upsert("SPY", "1h", make_bars(8, "1h", start="2026-06-01 13:00"))
    cache.upsert("SPY", "1d", make_bars(5, "1d", start="2026-05-26 00:00"))


def test_no_future_bars(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    now = pd.Timestamp("2026-06-01 15:00", tz="UTC")  # 15m bar 14:45 just closed
    ctx = build_context(cache, "SPY", now)
    assert ctx.bars["15m"].index.max() == pd.Timestamp("2026-06-01 14:45", tz="UTC")
    assert ctx.bars["1h"].index.max() == pd.Timestamp("2026-06-01 14:00", tz="UTC")  # 14:00 bar closed at 15:00
    # daily: NY date of now is 2026-06-01 -> only bars strictly before that date
    assert ctx.bars["1d"].index.max() == pd.Timestamp("2026-05-30 04:00", tz="UTC")


def test_partial_hour_excluded(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    now = pd.Timestamp("2026-06-01 14:45", tz="UTC")
    ctx = build_context(cache, "SPY", now)
    # the 14:00 1h bar closes at 15:00; it must NOT be visible at 14:45
    assert ctx.bars["1h"].index.max() == pd.Timestamp("2026-06-01 13:00", tz="UTC")


def test_same_day_daily_bar_excluded(tmp_path, make_bars):
    cache = BarCache(tmp_path)
    _fill(cache, make_bars)
    # Daily bars are labeled at New York midnight in UTC (see closed_before's
    # "1d" branch). Add a daily bar stamped 2026-06-01 04:00 UTC: it is today's
    # bar for the 2026-06-01 New York session.
    cache.upsert("SPY", "1d", make_bars(1, "1d", start="2026-06-01 00:00"))
    now = pd.Timestamp("2026-06-01 15:00", tz="UTC")  # NY date 2026-06-01
    ctx = build_context(cache, "SPY", now)
    # today's bar is look-ahead: it must NOT be visible (rule is strict <)
    assert pd.Timestamp("2026-06-01 04:00", tz="UTC") not in ctx.bars["1d"].index
    assert ctx.bars["1d"].index.max() == pd.Timestamp("2026-05-30 04:00", tz="UTC")
