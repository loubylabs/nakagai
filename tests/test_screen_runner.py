"""run_screen: deterministic evaluation over a synthetic bar cache."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import BarCache
from nakagai.screen.runner import run_screen
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary

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


FOUR_HOUR_ABOVE_SMA20 = {"version": 1, "tf": "4h", "conditions": {"all": [
    {"lhs": {"src": "close"}, "op": ">",
     "rhs": {"ind": "sma", "n": 20}}]}}


DOUBLE_CLOSE = {"version": 1, "tf": "1d", "conditions": {"all": [
    {"lhs": {"ind": "double_close"}, "op": ">", "rhs": {"src": "close"}}]}}


def _injected():
    return core_vocabulary().with_terms(
        Term("double_close", "series", {}, {}, lambda s, _a: s * 2))


def test_run_screen_evaluates_an_injected_term(cache):
    """The evaluation half of the screen surface reads the same vocabulary.

    Threading the validator alone would pass a screen that then errors per
    symbol, which reads on the rows as a screen that simply matched nothing.
    """
    result = run_screen(DOUBLE_CLOSE, ["UP"], cache, now=NOW,
                        vocabulary=_injected())
    assert result["errors"] == []
    assert result["rows"][0]["matched"] is True

    fallback = run_screen(DOUBLE_CLOSE, ["UP"], cache, now=NOW)
    assert fallback["rows"][0]["matched"] is None
    assert fallback["errors"] and "double_close" in fallback["errors"][0]


def test_run_screen_matches_and_sorts_matched_first(cache):
    result = run_screen(ABOVE_SMA20, ["DOWN", "UP"], cache, now=NOW)
    assert [r["symbol"] for r in result["rows"]] == ["UP", "DOWN"]
    up, down = result["rows"]
    assert up["matched"] is True and down["matched"] is False
    assert up["last_close"] == pytest.approx(100.0)
    assert up["bar_time"].startswith("2026-07-16")
    assert result["universe"] == {"screened": 2, "skipped": 0}
    assert result["errors"] == []


def test_run_screen_rows_carry_no_tier(cache):
    result = run_screen(ABOVE_SMA20, ["UP"], cache, now=NOW)
    assert result["rows"], "fixture produced no rows"
    assert all("tier" not in row for row in result["rows"])


def test_run_screen_universe_counts_screened_and_skipped(cache):
    result = run_screen(ABOVE_SMA20, ["UP"], cache, now=NOW)
    assert set(result["universe"]) == {"screened", "skipped"}
    assert result["universe"]["screened"] == 1


def test_run_screen_notes_missing_bars(cache):
    result = run_screen(ABOVE_SMA20, ["GHOST"], cache, now=NOW)
    row = result["rows"][0]
    assert row["matched"] is None and "no 1d bars cached" in row["note"]


def test_run_screen_notes_short_history(cache):
    spec = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}}]}}
    result = run_screen(spec, ["UP"], cache, now=NOW)
    row = result["rows"][0]
    assert row["matched"] is None and "60 bars cached" in row["note"]


def test_run_screen_isolates_a_bad_symbol(cache, monkeypatch):
    import nakagai.screen.runner as runner_mod

    real = runner_mod.build_context

    def boom(cache_, sym, now, **kwargs):
        if sym == "UP":
            raise RuntimeError("corrupt parquet")
        return real(cache_, sym, now, **kwargs)

    monkeypatch.setattr(runner_mod, "build_context", boom)
    result = run_screen(ABOVE_SMA20, ["UP", "DOWN"], cache, now=NOW)
    by_sym = {r["symbol"]: r for r in result["rows"]}
    assert by_sym["DOWN"]["matched"] is False
    assert by_sym["UP"]["matched"] is None and "error" in by_sym["UP"]["note"]
    assert any("UP" in e for e in result["errors"])


def test_run_screen_syncs_every_symbol_when_providers_given(cache):
    # There is no tier left to exempt: on-demand fetch means every symbol in
    # the request gets the same sync treatment.
    calls = []

    class _Provider:
        def fetch_bars(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe))
            return _daily_bars([1.0])

    run_screen(ABOVE_SMA20, ["UP", "DOWN"], cache, now=NOW,
               providers={"1d": _Provider()})
    assert ("UP", "1d") in calls
    assert ("DOWN", "1d") in calls


def test_run_screen_widens_the_1d_sync_window_for_a_long_lookback(cache):
    starts = []

    class _RecordingProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            starts.append(start)
            return _daily_bars([1.0])

    sma200 = {"version": 1, "tf": "1d", "conditions": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}}]}}
    run_screen(sma200, ["UP"], cache, now=NOW,
               providers={"1d": _RecordingProvider()})
    assert starts and (NOW - starts[0]) >= pd.Timedelta(days=399)


def test_run_screen_evaluates_cached_bars_when_sync_fails(cache):
    class _FlakyProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            raise RuntimeError("429 rate limited")

    result = run_screen(ABOVE_SMA20, ["UP"], cache, now=NOW,
                        providers={"1d": _FlakyProvider()})
    row = result["rows"][0]
    assert row["symbol"] == "UP"
    assert row["matched"] is True  # cached bars still evaluate
    assert "sync failed" in row["note"]
    assert any("sync failed" in e for e in result["errors"])


def test_run_screen_fetches_the_source_and_derives_a_referenced_timeframe(tmp_path):
    cache = BarCache(tmp_path / "derived-cache")
    calls = []

    class _TrappingDerivedProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            raise AssertionError("derived timeframe must not be fetched")

    class _HourlyProvider:
        def fetch_bars(self, symbol, timeframe, start, end):
            calls.append((symbol, timeframe))
            return _hourly_bars(np.linspace(1, 96, 96))

    result = run_screen(
        FOUR_HOUR_ABOVE_SMA20, ["ONLY"], cache, now=NOW,
        providers={"4h": _TrappingDerivedProvider(), "1h": _HourlyProvider()})

    assert calls == [("ONLY", "1h")]
    assert len(cache.load("ONLY", "4h")) >= 20
    assert result["errors"] == []
    assert result["rows"][0]["matched"] is True


def test_run_screen_derives_from_cached_source_without_providers(tmp_path):
    cache = BarCache(tmp_path / "cached-source")
    cache.upsert("CACHED", "1h", _hourly_bars(np.linspace(1, 96, 96)))

    result = run_screen(FOUR_HOUR_ABOVE_SMA20, ["CACHED"], cache, now=NOW)

    assert len(cache.load("CACHED", "4h")) >= 20
    assert result["errors"] == []
    assert result["rows"][0]["matched"] is True


def test_run_screen_uses_cached_derived_bars_when_derivation_fails(
        tmp_path, monkeypatch):
    import nakagai.screen.runner as runner_mod

    cache = BarCache(tmp_path / "derive-failure")
    idx = pd.date_range(end="2026-07-16 20:00", periods=60,
                        freq="4h", tz="UTC")
    closes = pd.Series(np.linspace(1, 60, 60), index=idx, dtype=float)
    bars = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1,
         "close": closes, "volume": 1_000_000.0}, index=idx)
    cache.upsert("CACHED", "4h", bars)

    def fail_derive(*args, **kwargs):
        raise RuntimeError("derive failed")

    monkeypatch.setattr(runner_mod, "derive_incremental", fail_derive)
    result = run_screen(FOUR_HOUR_ABOVE_SMA20, ["CACHED"], cache, now=NOW)

    assert result["rows"][0]["matched"] is True
    assert "sync failed: derive failed" in result["rows"][0]["note"]
    assert any("derive failed" in error for error in result["errors"])


def test_run_screen_isolates_a_derived_failure_by_symbol(tmp_path, monkeypatch):
    import nakagai.screen.runner as runner_mod

    cache = BarCache(tmp_path / "selective-derive-failure")
    idx = pd.date_range(end="2026-07-16 20:00", periods=60,
                        freq="4h", tz="UTC")
    closes = pd.Series(np.linspace(1, 60, 60), index=idx, dtype=float)
    cached_derived = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1,
         "close": closes, "volume": 1_000_000.0}, index=idx)
    for symbol in ("BAD", "GOOD"):
        cache.upsert(symbol, "1h", _hourly_bars(np.linspace(1, 96, 96)))
        cache.upsert(symbol, "4h", cached_derived)

    real_derive = runner_mod.derive_incremental
    calls = []

    def selective_derive(cache_, symbol, timeframe):
        calls.append(symbol)
        if symbol == "BAD":
            raise RuntimeError("derive failed")
        return real_derive(cache_, symbol, timeframe)

    monkeypatch.setattr(runner_mod, "derive_incremental", selective_derive)
    result = run_screen(FOUR_HOUR_ABOVE_SMA20, ["BAD", "GOOD"], cache, now=NOW)

    by_symbol = {row["symbol"]: row for row in result["rows"]}
    assert calls == ["BAD", "GOOD"]
    assert by_symbol["BAD"]["matched"] is True
    assert "sync failed: derive failed" in by_symbol["BAD"]["note"]
    assert by_symbol["GOOD"]["matched"] is True
    assert by_symbol["GOOD"]["note"] == ""
    assert result["errors"] == ["BAD: sync failed: derive failed"]
