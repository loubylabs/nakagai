import numpy as np
import pandas as pd

import pytest

from nakagai.data.cache import BarCache
from nakagai.engine.engine import Engine
from nakagai.engine.portfolio_types import ManagementDecision, PositionView
from nakagai.strategies.base import Direction, MarketContext
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.vocabulary import core_vocabulary

RISK = {"stop": {"kind": "percent", "pct": 5.0}, "target": {"kind": "rr", "rr": 20.0}}
# A short's rr target walks DOWN from the reference, so rr 20 against a 5%
# stop lands it on zero, which is not a price any tape reaches. rr 15 keeps
# the target far enough below the fall (~25 against a low of 40) that these
# exits still fire on the stop, and keeps it a real level.
SHORT_RISK = {"stop": {"kind": "percent", "pct": 5.0},
              "target": {"kind": "rr", "rr": 15.0}}


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
            "exits": exits, "risk": SHORT_RISK}


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


# ------------------------------- management is a value, not a mutation

def _view(**overrides):
    fields = {"direction": "long", "qty": 10,
              "entry_ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"),
              "entry": 100.0, "initial_stop": 95.0, "initial_target": 200.0,
              "live_stop": 95.0, "live_target": 200.0}
    return PositionView(**{**fields, **overrides})


def _manage_ctx(closes, now_offset=1):
    idx = pd.date_range("2026-01-05 14:30", periods=len(closes), freq="15min",
                        tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    bars = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2,
                         "close": c, "volume": 1000.0}, index=idx)
    return MarketContext(symbol="SPY", now=idx[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def test_a_ratchet_comes_back_as_a_decision_and_leaves_the_view_alone():
    strategy = RuleStrategy({"spec": _spec({"trailing": {"kind": "percent",
                                                         "pct": 3.0}})})
    position = _view()
    ctx = _manage_ctx([110.0] * 20)
    decision = strategy.manage(position, ctx)
    assert decision.action == "hold"
    assert decision.stop == pytest.approx(110.0 * 0.97)
    assert position.live_stop == 95.0     # the view never moved


def test_a_ratchet_that_would_loosen_the_live_stop_is_not_returned():
    """3% below 110 is 106.7, which is BELOW a live stop already at 108. A
    trailing stop that gave that back would widen the risk it exists to cut."""
    strategy = RuleStrategy({"spec": _spec({"trailing": {"kind": "percent",
                                                         "pct": 3.0}})})
    decision = strategy.manage(_view(live_stop=108.0), _manage_ctx([110.0] * 20))
    assert decision.stop is None


def test_a_spec_with_no_exits_block_holds():
    strategy = RuleStrategy({"spec": _spec({})})
    assert strategy.manage(_view(), _manage_ctx([110.0] * 20)) == ManagementDecision(
        action="hold", stop=None, target=None)


def test_a_rule_exit_comes_back_as_an_exit_decision():
    exits = {"exit": {"any": [{"lhs": {"src": "close"}, "op": ">", "rhs": 105.0}]}}
    strategy = RuleStrategy({"spec": _spec(exits)})
    ctx = _manage_ctx([110.0] * 20)
    ctx.fe = FrameEval(ctx.bars, vocabulary=core_vocabulary())
    ctx.cursor = {"15m": len(ctx.bars["15m"]) - 1}
    assert strategy.manage(_view(), ctx).action == "exit"


def test_the_engine_applies_a_returned_stop_to_the_live_position(tmp_path):
    """End to end: the ratchet only reaches the recorded trade because the
    engine applied the returned decision, since nothing else can move it."""
    up = list(np.linspace(99, 115, 20))
    down = list(np.linspace(115, 104, 12))
    res = _run(tmp_path, up + down, {"trailing": {"kind": "percent", "pct": 3.0}})
    t = res.trades[0]
    assert t.stop > t.entry * 1.0        # the live stop moved above the entry
    assert t.stop != t.entry * 0.95      # and is no longer the initial 5% stop
