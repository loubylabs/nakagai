import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.engine.engine import Engine
from nakagai.strategies.base import Direction
from nakagai.strategies.rules import RuleStrategy

RISK = {"stop": {"kind": "percent", "pct": 5.0}, "target": {"kind": "rr", "rr": 20.0}}


def _cache(root, closes):
    cache = BarCache(root)
    idx = pd.date_range("2026-01-05 14:30", periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2,
                       "close": c, "volume": 1000.0})
    cache.upsert("SPY", "15m", df)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    cache.upsert("SPY", "1h", df.resample("1h").agg(agg).dropna())
    cache.upsert("SPY", "1d", df.resample("1D").agg(agg).dropna())
    return cache


def _spec(exits):
    return {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 100.0}]},
            "exits": exits, "risk": RISK}


def _run(tmp_path, closes, exits):
    cache = _cache(tmp_path, closes)
    engine = Engine(RuleStrategy({"spec": _spec(exits)}), cache, "SPY",
                    pd.Timestamp("2026-01-05", tz="UTC"),
                    pd.Timestamp("2026-01-09", tz="UTC"))
    return engine.run()


def test_time_stop_closes_after_n_bars(tmp_path):
    closes = [99.0, 101.0] + [101.0] * 40      # entry, then drift sideways forever
    res = _run(tmp_path, closes, {"time_stop": {"bars": 4}})
    assert len(res.trades) >= 1
    t = res.trades[0]
    assert t.exit_reason == "manage"
    bars_held = (t.exit_ts - t.entry_ts) / pd.Timedelta(minutes=15)
    assert bars_held <= 6                       # entered next open + 4 bars + close-at-bar-end


def test_rule_exit_fires(tmp_path):
    closes = [99.0, 101.0] + list(np.linspace(101, 120, 30))
    exits = {"exit": {"any": [{"lhs": {"src": "close"}, "op": ">", "rhs": 110.0}]}}
    res = _run(tmp_path, closes, exits)
    assert res.trades and res.trades[0].exit_reason == "manage"
    assert res.trades[0].exit < 120             # left before the top


def test_trailing_stop_ratchets_and_stops_out(tmp_path):
    up = list(np.linspace(99, 115, 20))
    down = list(np.linspace(115, 104, 12))
    res = _run(tmp_path, up + down, {"trailing": {"kind": "percent", "pct": 3.0}})
    assert res.trades
    t = res.trades[0]
    assert t.exit_reason == "stop"
    assert t.stop > t.entry                     # the recorded stop ratcheted above entry
    assert t.pnl > 0


def test_breakeven_moves_stop_to_entry(tmp_path):
    up = list(np.linspace(99, 112, 16))         # > 1R in profit at 5% risk
    down = list(np.linspace(112, 90, 16))
    res = _run(tmp_path, up + down, {"breakeven_at": {"rr": 1.0}})
    assert res.trades
    t = res.trades[0]
    assert t.exit_reason == "stop"
    assert abs(t.exit - t.entry) < 1.0          # stopped near breakeven, not at -5%


def test_no_exits_block_behaves_like_before(tmp_path):
    closes = [99.0, 101.0] + list(np.linspace(101, 90, 20))
    res = _run(tmp_path, closes, None or {})
    assert res.trades and res.trades[0].exit_reason in ("stop", "eod_window")


def _short_spec(exits):
    return {"version": 2, "name": "ts", "timeframe": "15m",
            "short": {"all": [{"lhs": {"src": "close"}, "op": "crosses_below", "rhs": 100.0}]},
            "exits": exits, "risk": RISK}


def _run_short(tmp_path, closes, exits):
    cache = _cache(tmp_path, closes)
    engine = Engine(RuleStrategy({"spec": _short_spec(exits)}), cache, "SPY",
                    pd.Timestamp("2026-01-05", tz="UTC"),
                    pd.Timestamp("2026-01-09", tz="UTC"))
    return engine.run()


def test_short_atr_trailing_stop_ratchets_down_and_stops_out(tmp_path):
    # Flat first to build 14-bar ATR history without crossing the entry
    # level. A steady fall crosses below 100 (short entry) and keeps
    # falling, so the ATR trailing stop ratchets DOWN (min(), since the
    # position is SHORT) below the entry price. A hard rally then pushes a
    # bar's high back above that ratcheted (lower-than-entry) stop.
    flat = [105.0] * 20
    fall_to_entry = list(np.linspace(105, 90, 15))
    fall_further = list(np.linspace(90, 40, 30))
    rally = list(np.linspace(40, 90, 10))
    closes = flat + fall_to_entry + fall_further + rally
    res = _run_short(tmp_path, closes, {"trailing": {"kind": "atr", "n": 14, "mult": 2.0}})
    assert res.trades
    t = res.trades[0]
    assert t.direction == Direction.SHORT
    assert t.exit_reason == "stop"
    assert t.stop < t.entry            # ratcheted below entry, never above
    assert t.pnl > 0
