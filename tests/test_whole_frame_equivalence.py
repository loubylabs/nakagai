"""The equivalence this whole design rests on, as a permanent gate.

For every node type in the grammar: the value FrameEval computes over the whole
frame at row i equals what the per-bar path computed over the prefix ending at
i. If this fails, stored evidence is no longer reproducible.

EVERY_NODE is meant to be exhaustive over node KINDS, and a test asserts that it
is: every indicator name, every primitive name, and every math op has to appear.
Adding a name to the grammar without adding it here fails that test rather than
quietly shrinking the gate.

The oracle lives in tests/whole_frame_oracle.py and shares only the function
lookup tables with FrameEval, never the walk, so a bug in one cannot hide by
being mirrored in the other.
"""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import closed_before, visible_counts
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.spec import INDICATORS, MATH_OPS
from tests.whole_frame_oracle import prefix_value

EVERY_NODE = [
    {"src": "open"}, {"src": "close"}, {"src": "volume"},
    {"ind": "sma", "n": 20}, {"ind": "ema", "n": 20}, {"ind": "rsi", "n": 14},
    {"ind": "roc", "n": 10}, {"ind": "zscore", "n": 20},
    {"ind": "highest", "n": 20}, {"ind": "lowest", "n": 20},
    {"ind": "stdev", "n": 20},
    {"ind": "macd", "fast": 12, "slow": 26, "signal": 9, "field": "macd"},
    {"ind": "macd", "fast": 12, "slow": 26, "signal": 9, "field": "hist"},
    {"ind": "bb", "n": 20, "k": 2.0, "field": "lower"},
    {"ind": "atr", "n": 14}, {"ind": "donchian", "n": 20, "field": "upper"},
    {"ind": "supertrend", "n": 10, "mult": 3.0, "field": "line"},
    {"ind": "supertrend", "n": 10, "mult": 3.0, "field": "direction"},
    {"ind": "vwap"}, {"ind": "stoch", "n": 14, "d": 3, "field": "k"},
    {"ind": "adx", "n": 14}, {"ind": "obv"},
    {"ind": "ichimoku", "tenkan_n": 9, "kijun_n": 26, "senkou_n": 52,
     "disp": 26, "field": "tenkan"},
    # senkou_a is the one displaced field: it reads a value computed `disp`
    # bars back. Displacement is causal, so it belongs in the gate.
    {"ind": "ichimoku", "tenkan_n": 9, "kijun_n": 26, "senkou_n": 52,
     "disp": 26, "field": "senkou_a"},
    {"ind": "keltner", "n": 20, "mult": 2.0, "field": "upper"},
    {"ind": "cci", "n": 20}, {"ind": "mfi", "n": 14}, {"ind": "wpr", "n": 14},
    # `of` routes a series indicator through another node instead of close.
    {"ind": "rsi", "n": 14, "of": {"src": "high"}},
    {"prim": "opening_range_high", "minutes": 30},
    {"prim": "opening_range_low", "minutes": 30},
    {"prim": "prev_session_high"}, {"prim": "prev_session_low"},
    {"prim": "prev_session_close"}, {"prim": "gap_pct"},
    {"prim": "swing_high", "k": 3}, {"prim": "swing_low", "k": 3},
    {"prim": "day_of_week"}, {"prim": "minutes_into_session"},
    {"prim": "leg_retrace", "direction": "long", "k": 3},
    # The end-anchored pair. FrameEval cannot broadcast these, so it replays
    # them row by row over its span; the oracle calls the same scalar function
    # on the prefix. What is under test is the span, the offset and the
    # placement back onto the frame's index, not the primitive's arithmetic.
    {"prim": "fvg_nearest", "direction": "long", "field": "top", "state": "open"},
    {"prim": "fvg_nearest", "direction": "long", "field": "mid", "state": "inverted"},
    {"prim": "order_block", "direction": "long", "field": "bottom", "body_atr": 0.8},
    # bars_since is the only primitive that needs the evaluator handed back to
    # it, so this row is as much a test of that injection as of the counting.
    # The condition has to flip: close against a moving average does, close
    # against its own bar's open on a synthetic frame may not.
    {"prim": "bars_since",
     "cond": {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}},
    {"op": "-", "args": [{"src": "high"}, {"src": "low"}]},
    {"op": "+", "args": [{"src": "high"}, {"src": "low"}]},
    {"op": "*", "args": [{"src": "close"}, {"ind": "sma", "n": 10}]},
    # The denominator is a price, never 0. _math maps a zero denominator to NaN
    # and the oracle does not, so a node that can divide by zero would be
    # testing that difference rather than the equivalence.
    {"op": "/", "args": [{"src": "close"}, {"ind": "sma", "n": 10}]},
    {"op": "min", "args": [{"src": "high"}, {"src": "low"}]},
    {"op": "max", "args": [{"src": "high"}, {"src": "low"}]},
    {"op": "abs", "args": [{"op": "-", "args": [{"src": "close"},
                                                {"ind": "sma", "n": 10}]}]},
    {"src": "close", "tf": "1h"}, {"ind": "ema", "n": 20, "tf": "1h"},
    {"src": "close", "tf": "1d"}, {"ind": "sma", "n": 5, "tf": "1d"},
    {"prim": "prev_session_high", "tf": "1h"},
]


@pytest.fixture(scope="module")
def frames():
    """Multi-session RTH-shaped bars: 26 bars a day, weekends absent.

    Each bar opens at the previous close, so candle bodies take both signs. A
    frame where every body is the same small positive number reads as valid
    OHLCV and silently empties several nodes of content: order_block finds no
    displacement candle and no opposing one, so it is NaN at every row, and any
    close-against-open condition is constant. A node that is NaN or constant
    everywhere passes this gate no matter what the evaluator does.
    """
    rng = np.random.default_rng(19)
    days = pd.bdate_range("2026-01-05", periods=40, tz="UTC")
    stamps = [d + pd.Timedelta(hours=14, minutes=30) + i * pd.Timedelta(minutes=15)
              for d in days for i in range(26)]
    idx = pd.DatetimeIndex(stamps, name="ts")
    n = len(idx)
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = np.concatenate([[close[0] - 0.05], close[:-1]])
    b15 = pd.DataFrame({"open": open_,
                        "high": np.maximum(open_, close) + np.abs(rng.normal(0, 0.15, n)),
                        "low": np.minimum(open_, close) - np.abs(rng.normal(0, 0.15, n)),
                        "close": close, "volume": 1000.0 + rng.integers(0, 500, n)},
                       index=idx)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return {"15m": b15, "1h": b15.resample("1h").agg(agg).dropna(),
            "1d": b15.resample("1D").agg(agg).dropna()}


def test_every_node_kind_in_the_grammar_is_covered():
    """The gate is only a gate while the list is exhaustive.

    A new indicator, primitive or math op added to the grammar and not added
    here would leave a hole that reads as coverage, so name the omission out
    loud instead.
    """
    def named(kind):
        return {n[kind] for n in EVERY_NODE if kind in n} | {
            a[kind] for n in EVERY_NODE if "op" in n for a in n["args"]
            if isinstance(a, dict) and kind in a}

    assert named("ind") == set(INDICATORS)
    assert named("prim") == set(PRIMITIVES)
    assert {n["op"] for n in EVERY_NODE if "op" in n} == set(MATH_OPS)


@pytest.mark.parametrize("node", EVERY_NODE, ids=lambda n: str(n))
def test_whole_frame_equals_prefix_at_every_probe_row(node, frames):
    fe = FrameEval(frames, TFS, "SPY")
    whole = fe.series(node, "15m")
    driving = frames["15m"]
    rows = list(range(300, len(driving), 37))
    assert rows, "need probe rows"
    for i in rows:
        now = driving.index[i] + TFS.step
        want = prefix_value(node, frames, "15m", now, TFS)
        got = float(whole.iloc[i]) if isinstance(whole, pd.Series) else float(whole)
        assert (pd.isna(got) and pd.isna(want)) or got == want, (
            f"{node} row {i}: whole-frame {got!r} != prefix {want!r}")


def test_session_visibility_does_not_depend_on_a_label_plus_one_day_shortcut(frames):
    """The daily map must use the NY session rule, not label + 1 day.

    A daily bar is visible only once its session date has arrived in NY. If the
    map ever regresses to a fixed one-day offset this fails on the first bar
    after a weekend, where label + 1 day lands on a Saturday.
    """
    daily = frames["1d"]
    closes = frames["15m"].index + TFS.step
    got = visible_counts(daily.index, closes, "1d", TFS)
    want = [len(closed_before(daily, "1d", now, TFS)) for now in closes]
    assert list(got) == want
