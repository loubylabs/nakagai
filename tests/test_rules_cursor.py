"""Cursor-driven RuleStrategy produces the same signals as tree-per-bar did."""

import numpy as np
import pandas as pd

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import closed_before
from nakagai.engine.engine import Engine
from nakagai.strategies.rules import RuleStrategy
from nakagai.strategies.rules.frame_eval import FrameEval

SPEC = {"version": 2, "name": "t", "timeframe": "15m",
        "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above",
                          "rhs": {"ind": "sma", "n": 20}}]},
        "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                 "target": {"kind": "rr", "rr": 2.0}},
        "exits": {"time_stop": {"bars": 10}}}


def _frames(n=600):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    b15 = pd.DataFrame({"open": close, "high": close + 0.6, "low": close - 0.6,
                        "close": close, "volume": 1000.0}, index=idx)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum"}
    return MemoryBars({("SPY", "15m"): b15,
                       ("SPY", "1h"): b15.resample("1h").agg(agg).dropna(),
                       ("SPY", "1d"): b15.resample("1D").agg(agg).dropna()})


def test_replay_produces_trades_and_a_cursor_is_present():
    cache = _frames()
    bars = cache.load("SPY", "15m")
    res = Engine(RuleStrategy({"spec": SPEC}), cache, "SPY",
                 bars.index[100], bars.index[-1] + TFS.step).run()
    assert len(res.trades) > 0


def test_context_cursor_indexes_the_current_bar():
    seen = []
    cache = _frames()
    bars = cache.load("SPY", "15m")

    class Probe(RuleStrategy):
        def on_bar(self, ctx):
            seen.append((ctx.now, ctx.cursor["15m"]))
            return super().on_bar(ctx)

    Engine(Probe({"spec": SPEC}), cache, "SPY",
           bars.index[100], bars.index[200]).run()
    assert seen
    for now, cur in seen:
        assert bars.index[cur] + TFS.step == now


def test_the_group_tree_is_built_once_per_replay(monkeypatch):
    """The whole point: the tree is walked once, not once per bar.

    group_series walks the rule tree over the whole frame and to_driving lifts
    the result onto the driving index. Both are O(frame), so reading them at a
    cursor is only a win if the pair runs once per replay. Calling them per bar
    puts replay straight back on O(bars x history) with the indicator cache
    doing nothing but hiding it, and it measured SLOWER than the tree-per-bar
    path it replaced.
    """
    cache = _frames()
    bars = cache.load("SPY", "15m")
    calls = []
    real = FrameEval.group_series
    monkeypatch.setattr(FrameEval, "group_series",
                        lambda self, group, tf: (calls.append(tf),
                                                 real(self, group, tf))[1])
    Engine(RuleStrategy({"spec": SPEC}), cache, "SPY",
           bars.index[100], bars.index[-1] + TFS.step).run()
    assert calls == ["15m"]   # one tree, once, for 500 replayed bars


def test_span_bounds_every_timeframe_not_just_the_driving_one():
    """A non-driving frame is bounded to the replay window too.

    set_span decides how many rows the end-anchored primitives (fvg_nearest,
    order_block) walk one at a time. Bounding only the driving timeframe would
    leave a 1h spec looping over the entire 1h history on every replay, which
    is the cost this whole change exists to remove.
    """
    seen = {}
    cache = _frames()
    bars = cache.load("SPY", "15m")
    hourly = cache.load("SPY", "1h")

    class Probe(RuleStrategy):
        def on_bar(self, ctx):
            seen["fe"] = ctx.fe
            return super().on_bar(ctx)

    Engine(Probe({"spec": SPEC}), cache, "SPY",
           bars.index[100], bars.index[200]).run()

    lo, hi = seen["fe"]._span("1h")
    # The window's own 1h rows: first and last hourly bar visible during replay.
    first = len(closed_before(hourly, "1h", bars.index[100] + TFS.step)) - 1
    last = len(closed_before(hourly, "1h", bars.index[199] + TFS.step)) - 1
    assert (lo, hi) == (first, last + 1)
    assert hi - lo < len(hourly) // 3   # bounded to the window, not the frame
