"""FrameEval: whole-frame node values that agree with prefix evaluation."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import closed_before
from nakagai.strategies.rules.frame_eval import FrameEval
from tests.whole_frame_oracle import prefix_value

NODES = [
    {"src": "close"},
    {"ind": "sma", "n": 20},
    {"ind": "ema", "n": 50},
    {"ind": "rsi", "n": 14},
    {"ind": "atr", "n": 14},
    {"ind": "vwap"},
    {"ind": "bb", "n": 20, "k": 2.0, "field": "upper"},
    {"ind": "macd", "fast": 12, "slow": 26, "signal": 9, "field": "hist"},
    {"prim": "opening_range_high", "minutes": 30},
    {"prim": "minutes_into_session"},
    {"prim": "prev_session_high"},
    {"prim": "swing_high", "k": 3},
    {"op": "-", "args": [{"src": "close"}, {"ind": "sma", "n": 10}]},
]


def _frames(n=400):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    b15 = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                        "close": close, "volume": 1000.0}, index=idx)
    b1h = b15.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    b1d = b15.resample("1D").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    return {"15m": b15, "1h": b1h, "1d": b1d}


@pytest.mark.parametrize("node", NODES, ids=lambda n: str(n))
def test_whole_frame_row_equals_prefix_last_row(node):
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    whole = fe.series(node, "15m")
    driving = frames["15m"]
    for i in (120, 200, 311, len(driving) - 1):
        now = driving.index[i] + TFS.step
        want = prefix_value(node, frames, "15m", now, TFS)
        got = float(whole.iloc[i]) if isinstance(whole, pd.Series) else float(whole)
        assert (pd.isna(got) and pd.isna(want)) or got == want, f"row {i}"


def test_cross_timeframe_node_never_sees_an_unclosed_bar():
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    node = {"src": "close", "tf": "1h"}
    got = fe.series(node, "15m")
    driving, hourly = frames["15m"], frames["1h"]
    for i in range(80, len(driving)):
        now = driving.index[i] + TFS.step
        visible = hourly[hourly.index + TFS.deltas["1h"] <= now]
        want = visible["close"].iloc[-1] if len(visible) else float("nan")
        assert (pd.isna(got.iloc[i]) and pd.isna(want)) or got.iloc[i] == want


def test_nothing_visible_yet_is_nan_not_the_last_bar():
    """The rows before the first higher bar has closed.

    The visibility map counts closed bars, so an empty history is 0, and the
    position it converts to is -1. Read as an index that is the LAST row, which
    would hand the opening rows of a replay a value from the end of history.
    That is the worst-shaped lookahead bug there is: silent, and profitable.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    got = fe.series({"src": "close", "tf": "1d"}, "15m")
    driving = frames["15m"]
    blind = [i for i in range(len(driving))
             if not len(closed_before(frames["1d"], "1d",
                                      driving.index[i] + TFS.step, TFS))]
    assert blind, "fixture must contain rows with no closed daily bar yet"
    assert got.iloc[blind].isna().all()
    assert not (got.iloc[blind] == frames["1d"]["close"].iloc[-1]).any()
    seeing = [i for i in range(len(driving)) if i not in set(blind)]
    assert got.iloc[seeing].notna().all(), "the visible rows must still carry a value"


@pytest.mark.parametrize("node", [
    {"prim": "fvg_nearest", "direction": "long", "field": "top"},
    {"prim": "order_block", "direction": "long", "field": "top"},
], ids=lambda n: n["prim"])
def test_end_anchored_primitive_matches_prefix_over_its_span(node):
    """End-anchored primitives read the tail of the frame they are handed, so
    they are the one node kind a whole-frame pass may not broadcast. Inside the
    span they must equal the prefix answer row for row; outside it they are NaN
    rather than a value carried from somewhere else in history."""
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    fe.set_span("15m", 300, 340)
    got = fe.series(node, "15m")
    for i in range(300, 340):
        now = frames["15m"].index[i] + TFS.step
        want = prefix_value(node, frames, "15m", now, TFS)
        assert (pd.isna(got.iloc[i]) and pd.isna(want)) or got.iloc[i] == want, f"row {i}"
    assert got.iloc[340:].isna().all()


def test_series_is_memoized_per_node():
    fe = FrameEval(_frames(), TFS, "SPY")
    a = fe.series({"ind": "sma", "n": 20}, "15m")
    b = fe.series({"ind": "sma", "n": 20}, "15m")
    assert a is b
