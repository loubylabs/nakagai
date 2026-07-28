"""FrameEval: whole-frame node values that agree with prefix evaluation."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import closed_before
from nakagai.strategies.indicators import crossed_above
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


def test_memo_key_separates_evaluation_timeframes():
    """The same node at two timeframes is two different series.

    _eval evaluates `of` sub-nodes at src_tf, so one spec can ask for
    {'src': 'close'} on 15m and again on 1h within a single replay. A memo key
    that keyed on the node alone would hand the second caller the first
    caller's series, indexed on the wrong timeframe, and every later read would
    be off by whatever the two indexes disagree about.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    node = {"src": "close"}
    at15, at1h = fe.series(node, "15m"), fe.series(node, "1h")
    assert at15 is not at1h
    assert not at15.index.equals(at1h.index)
    assert at15.index.equals(frames["15m"].index)
    assert at1h.index.equals(frames["1h"].index)


def _closes(values) -> dict:
    idx = pd.date_range("2026-01-05 14:30", periods=len(values), freq="15min",
                        tz="UTC")
    c = pd.Series(values, index=idx, dtype="float64")
    return {"15m": pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                                 "close": c, "volume": 1000.0}, index=idx)}


def test_cross_reads_the_bar_before_never_the_bar_after():
    """crosses_above compares row i against row i-1, and row i-1 only.

    The shift direction is the whole causality of the operator. A .shift(-1)
    here still produces plausible-looking crosses, just ones that consult the
    NEXT bar, which is lookahead that no metric would flag.
    """
    frames = _closes([1.0, 2.0, 4.0, 5.0, 2.0, 6.0])
    fe = FrameEval(frames, TFS, "SPY")
    cond = {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 3.0}
    got = fe.condition_series(cond, "15m")
    assert got.dtype == bool
    assert list(got) == [False, False, True, False, False, True]

    close = frames["15m"]["close"]
    for i in range(len(close)):
        assert bool(got.iloc[i]) == crossed_above(close.iloc[: i + 1], 3.0), \
            f"row {i} disagrees with the prefix-and-iloc[-2] semantics"


def test_cross_below_reads_the_bar_before():
    frames = _closes([6.0, 5.0, 2.0, 1.0, 4.0, 1.0])
    fe = FrameEval(frames, TFS, "SPY")
    cond = {"lhs": {"src": "close"}, "op": "crosses_below", "rhs": 3.0}
    got = fe.condition_series(cond, "15m")
    assert list(got) == [False, False, True, False, False, True]


def test_cross_against_an_end_anchored_level_keeps_the_scalar_broadcast():
    """Both bars of the cross see the level known at the CURRENT bar.

    fvg_nearest returned one float per replayed bar, and crossed_above's scalar
    branch compared close[i-1] AND close[i] against that single level. Shifting
    the level series instead tests the earlier bar against the level as it stood
    a bar ago, which fires and suppresses different trades. This is the fixture
    that says so: `shifted` is what a plain .shift(1) on both operands would
    give, and it must not match.
    """
    frames = _frames()
    fe = FrameEval(frames, TFS, "SPY")
    lo, hi = 200, 400
    fe.set_span("15m", lo, hi)
    node = {"prim": "fvg_nearest", "direction": "long", "field": "top"}
    got = fe.condition_series(
        {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": node}, "15m")

    driving, level = frames["15m"], fe.series(node, "15m")
    close = driving["close"]
    want, shifted = [], []
    for i in range(lo, hi):
        now = driving.index[i] + TFS.step
        prefix = closed_before(driving, "15m", now, TFS)
        want.append(bool(crossed_above(
            prefix["close"], prefix_value(node, frames, "15m", now, TFS))))
        shifted.append(bool(close.iloc[i - 1] <= level.iloc[i - 1]
                            and close.iloc[i] > level.iloc[i]))
    assert [bool(v) for v in got.iloc[lo:hi]] == want
    assert any(want), "fixture must contain at least one firing cross"
    assert shifted != want, \
        "fixture must contain a row where shifting the level changes the answer"


def test_a_higher_timeframe_condition_is_false_until_its_bar_has_closed():
    """The lift onto the driving index keeps booleans boolean.

    strategy.py reads this series as bool(series.iloc[i]). Lifting through the
    float branch would leave the not-yet-visible rows NaN, and bool(nan) is
    True, so a condition on a bar that has not closed yet would read SATISFIED
    for every row of the opening blind window.
    """
    frames = _frames()
    frames["1h"] = frames["1h"].iloc[8:]
    fe = FrameEval(frames, TFS, "SPY")
    group = {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 0.0},
                     {"lhs": {"src": "high"}, "op": ">=", "rhs": {"src": "low"}}]}
    out = fe.driving_group(group, "1h")

    driving = frames["15m"]
    blind = [i for i in range(len(driving))
             if not len(closed_before(frames["1h"], "1h",
                                      driving.index[i] + TFS.step, TFS))]
    assert len(blind) > 8, "fixture must open with a real blind window"
    assert out.dtype == bool
    assert not out.iloc[blind].any()
    assert not any(bool(out.iloc[i]) for i in blind)
    seeing = sorted(set(range(len(driving))) - set(blind))
    assert out.iloc[seeing].all(), "the visible rows must still read satisfied"


def _hand_bars(closes, start, freq):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0}, index=idx)


def test_a_daily_bar_is_invisible_within_its_own_session():
    """Jan 6 rows see Jan 5's daily close, never Jan 6's own.

    A daily bar is labeled at midnight UTC of the session it summarizes, so a
    plain label-ordered ffill hands every row of Jan 6 the close of the session
    it is still living through. That is the sharpest lookahead the grammar can
    express, and the one a session-aligned visibility rule exists to refuse.
    """
    frames = {"15m": _hand_bars([100.0] * 26, "2026-01-06 14:30", "15min"),
              "1h": _hand_bars([100.0] * 8, "2026-01-06 14:00", "1h"),
              "1d": _hand_bars([95.0, 96.0], "2026-01-05 00:00", "1D")}
    fe = FrameEval(frames, TFS, "SPY")
    assert (fe.series({"src": "close", "tf": "1d"}, "15m") == 95.0).all()


def test_a_tf_qualified_primitive_evaluates_on_its_native_frame():
    """prev_session_high[1h] reads the 1h frame, then lands by visibility.

    The primitive has to see hourly bars to have an hourly answer; evaluating
    it against the driving frame and relabeling would give a 15m answer wearing
    an hourly name. The Jan 5 session's hourly high is 202.0, and every Jan 6
    driving row carries it once the hourly bars that establish it have closed.
    """
    b1h = pd.concat([_hand_bars([199.5, 201.5], "2026-01-05 13:00", "1h"),
                     _hand_bars([190.0, 191.0], "2026-01-06 13:00", "1h")])
    frames = {"15m": _hand_bars([100.0] * 8, "2026-01-06 14:30", "15min"),
              "1h": b1h,
              "1d": _hand_bars([95.0, 96.0], "2026-01-05 00:00", "1D")}
    fe = FrameEval(frames, TFS, "SPY")
    out = fe.series({"prim": "prev_session_high", "tf": "1h"}, "15m")
    assert out.index.equals(frames["15m"].index)
    assert (out == 202.0).all()


def test_session_aligned_destination_timeframe_is_refused():
    """A daily bar's label carries no close time, so its visibility cutoff
    would have to be guessed. _positions refuses rather than guess."""
    fe = FrameEval(_frames(), TFS, "SPY")
    with pytest.raises(ValueError, match="session-aligned"):
        fe.series({"src": "close", "tf": "15m"}, "1d")
