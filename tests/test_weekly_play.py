"""A weekly play end to end: 1d bars in, derived weeks, one decision a week."""

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import EXCHANGE_TZ
from nakagai.data.sync import derive_incremental
from nakagai.engine.context import build_context
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.spec import validate_spec

RISK = {"stop": {"kind": "percent", "pct": 2.0},
        "target": {"kind": "rr", "rr": 2.0}}
WEEKLY = {"version": 2, "name": "weekly-close", "timeframe": "1w",
          "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 120}]},
          "risk": RISK}


def _bars(index, close):
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame({"open": close - 0.5, "high": close + 1.0,
                         "low": close - 1.0, "close": close, "volume": 1000.0},
                        index=index)


def _cache(tmp_path):
    """Daily bars 4 May to 5 June 2026 (Memorial Day 25 May skipped), closes
    100..123, so the weekly closes are 104, 109, 114, 118, 123; and 15m
    driving bars for Thu 4, Fri 5, Mon 8 and Tue 9 June."""
    cache = BarCache(tmp_path)
    days = pd.bdate_range("2026-05-04", "2026-06-05").drop(pd.Timestamp("2026-05-25"))
    daily = days.tz_localize(EXCHANGE_TZ).tz_convert("UTC").rename("ts")
    cache.upsert("SPY", "1d", _bars(daily, 100.0 + np.arange(len(daily))))
    ranges = [pd.date_range(f"{day} 13:30", f"{day} 19:45", freq="15min", tz="UTC")
              for day in ("2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09")]
    driving = ranges[0].append(ranges[1:]).rename("ts")
    cache.upsert("SPY", "15m", _bars(driving, np.full(len(driving), 150.0)))
    derive_incremental(cache, "SPY", "1w", now=pd.Timestamp("2026-06-08 13:45", tz="UTC"))
    return cache


def test_a_weekly_rule_validates():
    assert validate_spec(WEEKLY) == []


def test_a_weekly_rule_cannot_mix_in_a_daily_reference():
    spec = {**WEEKLY, "long": {"all": [
        {"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": 0}]}}
    assert any("session-aligned" in err for err in validate_spec(spec))


def test_the_cache_holds_five_complete_weeks(tmp_path):
    weekly = _cache(tmp_path).load("SPY", "1w")
    assert list(weekly.index.tz_convert(EXCHANGE_TZ).strftime("%Y-%m-%d")) == [
        "2026-05-04", "2026-05-11", "2026-05-18", "2026-05-26", "2026-06-01"]
    assert list(weekly["close"]) == [104.0, 109.0, 114.0, 118.0, 123.0]


def test_a_weekly_play_decides_once_on_the_first_session_after_its_week(tmp_path):
    cache = _cache(tmp_path)
    strategy = RuleStrategy({"spec": WEEKLY})
    closes = cache.load("SPY", "15m").index + pd.Timedelta(minutes=15)
    fired = [now for now in closes
             if strategy.on_bar(build_context(cache, "SPY", now, reference_pairs=()))]
    # Thursday 13:45 also opens the gate (first bar of the history), but the
    # week it reads closed at 118, under the rule's 120.
    assert fired == [pd.Timestamp("2026-06-08 13:45", tz="UTC")]


def test_friday_after_the_bell_still_reads_last_week(tmp_path):
    ctx = build_context(_cache(tmp_path), "SPY",
                        pd.Timestamp("2026-06-05 20:00", tz="UTC"), reference_pairs=())
    assert ctx.bars["1w"].index[-1] == pd.Timestamp("2026-05-26", tz=EXCHANGE_TZ)
    assert ctx.bars["1w"]["close"].iloc[-1] == 118.0


def test_an_intraday_play_may_read_the_weekly_close(tmp_path):
    spec = {"version": 2, "name": "above-last-week", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": {"src": "close", "tf": "1w"}}]},
            "risk": RISK}
    assert validate_spec(spec) == []
    cache = _cache(tmp_path)
    strategy = RuleStrategy({"spec": spec})
    ctx = build_context(cache, "SPY", pd.Timestamp("2026-06-08 13:45", tz="UTC"),
                        reference_pairs=())
    assert strategy.on_bar(ctx)          # 150 > 123, last week's close
