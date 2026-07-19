"""run_screen: deterministic evaluation over a synthetic bar cache."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.screen.runner import run_screen
from nakagai.screen.universe import DAILY, FULL

NOW = pd.Timestamp("2026-07-17 20:05", tz="UTC")


def _daily_bars(closes):
    idx = pd.date_range(end="2026-07-16", periods=len(closes), freq="B", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": 1_000_000.0}, index=idx)


def _hourly_bars(closes):
    idx = pd.date_range(end="2026-07-16 20:00", periods=len(closes), freq="1h", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1,
                         "close": c, "volume": 1_000_000.0}, index=idx)


@pytest.fixture
def cache(tmp_path):
    cache = BarCache(tmp_path / "cache")
    cache.upsert("UP", "1d", _daily_bars(np.linspace(50, 100, 60)))    # rising
    cache.upsert("DOWN", "1d", _daily_bars(np.linspace(100, 50, 60)))  # falling
    cache.upsert("DOWN", "1h", _hourly_bars(np.linspace(100, 50, 60)))  # falling
    return cache


ABOVE_SMA20 = {"version": 1, "tf": "1d", "conditions": {"all": [
    {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}]}}


def test_run_screen_matches_and_sorts_matched_first(cache):
    result = run_screen(ABOVE_SMA20, {"DOWN": DAILY, "UP": DAILY}, cache, now=NOW)
    assert [r["symbol"] for r in result["rows"]] == ["UP", "DOWN"]
    up, down = result["rows"]
    assert up["matched"] is True and down["matched"] is False
    assert up["last_close"] == pytest.approx(100.0)
    assert up["bar_time"].startswith("2026-07-16")
    assert result["universe"] == {"full": 0, "daily": 2, "skipped": 0}
    assert result["errors"] == []


def test_run_screen_skips_daily_tier_on_an_intraday_spec(cache):
    spec = {**ABOVE_SMA20, "tf": "1h"}
    result = run_screen(spec, {"UP": DAILY, "DOWN": FULL}, cache, now=NOW)
    by_sym = {r["symbol"]: r for r in result["rows"]}
    assert by_sym["UP"]["matched"] is None
    assert "intraday screen" in by_sym["UP"]["note"]
    assert result["universe"]["skipped"] == 1


def test_run_screen_notes_missing_bars(cache):
    result = run_screen(ABOVE_SMA20, {"GHOST": FULL}, cache, now=NOW)
    row = result["rows"][0]
    assert row["matched"] is None and "no 1d bars cached" in row["note"]


def test_run_screen_notes_short_history(cache):
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}}]}}
    result = run_screen(spec, {"UP": DAILY}, cache, now=NOW)
    row = result["rows"][0]
    assert row["matched"] is None and "60 bars cached" in row["note"]


def test_run_screen_isolates_a_bad_symbol(cache, monkeypatch):
    import nakagai.screen.runner as runner_mod

    real = runner_mod.build_context

    def boom(cache_, sym, now):
        if sym == "UP":
            raise RuntimeError("corrupt parquet")
        return real(cache_, sym, now)

    monkeypatch.setattr(runner_mod, "build_context", boom)
    result = run_screen(ABOVE_SMA20, {"UP": DAILY, "DOWN": DAILY}, cache, now=NOW)
    by_sym = {r["symbol"]: r for r in result["rows"]}
    assert by_sym["DOWN"]["matched"] is False
    assert by_sym["UP"]["matched"] is None and "error" in by_sym["UP"]["note"]
    assert any("UP" in e for e in result["errors"])


def test_run_screen_syncs_full_tier_only(cache):
    calls = []

    class _Provider:
        def fetch_bars(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe))
            return _daily_bars([1.0])

    run_screen(ABOVE_SMA20, {"UP": FULL, "DOWN": DAILY}, cache, now=NOW,
               providers={"1d": _Provider()})
    assert ("UP", "1d") in calls
    assert all(sym != "DOWN" for sym, _ in calls)


def test_run_screen_widens_the_1d_sync_window_for_a_long_lookback(cache):
    starts = []

    class _RecordingProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            starts.append(start)
            return _daily_bars([1.0])

    sma200 = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}}]}}
    run_screen(sma200, {"UP": FULL}, cache, now=NOW,
               providers={"1d": _RecordingProvider()})
    assert starts and (NOW - starts[0]) >= pd.Timedelta(days=399)


def test_run_screen_evaluates_cached_bars_when_sync_fails(cache):
    class _FlakyProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            raise RuntimeError("429 rate limited")

    result = run_screen(ABOVE_SMA20, {"UP": FULL}, cache, now=NOW,
                        providers={"1d": _FlakyProvider()})
    row = result["rows"][0]
    assert row["symbol"] == "UP"
    assert row["matched"] is True  # cached bars still evaluate
    assert "sync failed" in row["note"]
    assert any("sync failed" in e for e in result["errors"])
