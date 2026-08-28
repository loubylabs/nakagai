"""tf on primitive nodes: HTF structure with LTF entries, lookahead-safe."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.primitives import _swing
from nakagai.strategies.rules.spec import _expr_text, validate_spec


def _frame(vals, freq, start="2026-01-05 14:30"):
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.DataFrame({"open": vals, "high": [v + 1 for v in vals],
                         "low": [v - 1 for v in vals], "close": vals,
                         "volume": 1000.0}, index=idx)


def _fe(f15, f1h):
    return FrameEval(
        "SPY", {("SPY", "15m"): f15, ("SPY", "1h"): f1h}, TFS)


H1 = [100, 104, 108, 104, 100, 98, 96, 98, 100, 102, 104, 106]


def test_prim_tf_lands_on_the_driving_index_and_not_by_naive_ffill():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")  # 48 x 15m spanning the same 12h
    out = _fe(f15, f1h).series({"prim": "swing_high", "k": 2, "tf": "1h"}, "15m")
    sw = _swing(f1h, "high", 2, find_max=True)
    naive = sw.reindex(f15.index.union(f1h.index)).ffill().reindex(f15.index)
    assert out.index.equals(f15.index)
    # A label-ordered ffill hands the 15m bar labeled 14:30 the hourly bar
    # labeled 14:30, which does not close until 15:30. The visibility rule is
    # what separates the two, so they must not agree.
    assert not out.equals(naive)


def test_prim_tf_value_is_from_the_last_closed_htf_bar_only():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")
    out = _fe(f15, f1h).series({"prim": "swing_high", "k": 2, "tf": "1h"}, "15m")
    sw = _swing(f1h, "high", 2, find_max=True)
    for ts in f15.index:
        now = ts + TFS.step                     # the 15m bar's own close time
        visible = sw[sw.index + TFS.deltas["1h"] <= now]
        want = visible.iloc[-1] if len(visible) else np.nan
        got = out.loc[ts]
        assert (np.isnan(want) and np.isnan(got)) or got == want


def test_bars_since_tf_counts_htf_bars():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")
    node = {"prim": "bars_since", "tf": "1h",
            "cond": {"lhs": {"src": "close"}, "op": ">", "rhs": 107}}
    out = _fe(f15, f1h).series(node, "15m")
    # close > 107 only at 1h bar index 2; the last 15m bar sits 9 hourly bars later
    assert out.iloc[-1] == pytest.approx(9.0)


def test_memo_keeps_tf_variants_apart():
    f1h = _frame(H1, "1h")
    f15 = _frame([100] * 20 + [110, 130, 110] + [100] * 25, "15min")
    fe = _fe(f15, f1h)
    with_tf = fe.series({"prim": "swing_high", "k": 2, "tf": "1h"}, "15m")
    without = fe.series({"prim": "swing_high", "k": 2}, "15m")
    assert not with_tf.equals(without)


def _spec(cond):
    return {"version": 2, "name": "t", "timeframe": "15m", "long": {"all": [cond]}}


def test_validate_accepts_tf_on_prims():
    assert validate_spec(_spec({"lhs": {"prim": "swing_high", "k": 3, "tf": "1h"},
                                "op": "<", "rhs": {"src": "close"}})) == []
    assert validate_spec(_spec({"lhs": {"prim": "bars_since", "tf": "1h",
                                        "cond": {"lhs": {"src": "close"}, "op": ">", "rhs": 1}},
                                "op": "<=", "rhs": 6})) == []


def test_validate_rejects_unknown_tf_on_prims():
    errs = validate_spec(_spec({"lhs": {"prim": "swing_high", "tf": "2h"},
                                "op": "<", "rhs": {"src": "close"}}))
    assert any("tf must be one of" in e for e in errs)


def test_describe_renders_tf_suffix():
    from nakagai.strategies.rules.vocabulary import core_vocabulary
    assert _expr_text({"prim": "swing_high", "k": 3, "tf": "1h"},
                      core_vocabulary()) == "swing_high(3)[1h]"


def test_validate_rejects_tf_on_rvol():
    """rvol's baseline is the bar's own place in ITS session's volume shape, so
    a foreign frame answers a different question under the same name."""
    errs = validate_spec(_spec({"lhs": {"prim": "rvol", "tf": "1d"},
                                "op": ">", "rhs": 2}))
    assert any("rvol is session-scoped and takes no tf" in e for e in errs)
    # and the bare form stays valid, or the rejection above proves nothing
    assert validate_spec(_spec({"lhs": {"prim": "rvol", "sessions": 20},
                                "op": ">", "rhs": 2})) == []


def test_validate_rejects_tf_on_day_of_week():
    errs = validate_spec(_spec({"lhs": {"prim": "day_of_week", "tf": "1h"},
                                "op": "<", "rhs": {"src": "close"}}))
    assert any("day_of_week is session-scoped and takes no tf" in e for e in errs)


def test_validate_still_accepts_tf_on_structural_prim():
    assert validate_spec(_spec({"lhs": {"prim": "swing_high", "k": 3, "tf": "1h"},
                                "op": "<", "rhs": {"src": "close"}})) == []


def test_validate_rejects_session_prim_inside_tf_bars_since():
    errs = validate_spec(_spec({"lhs": {"prim": "bars_since", "tf": "1d",
                                        "cond": {"lhs": {"prim": "day_of_week"},
                                                 "op": "<", "rhs": 1}},
                                "op": "<=", "rhs": 1}))
    assert any("day_of_week is session-scoped and cannot sit inside "
               "bars_since.cond with tf" in e for e in errs), errs


def test_validate_rejects_session_prim_nested_deep_in_tf_bars_since():
    inner = {"prim": "bars_since",
             "cond": {"lhs": {"prim": "minutes_into_session"}, "op": ">", "rhs": 30}}
    errs = validate_spec(_spec({"lhs": {"prim": "bars_since", "tf": "1h",
                                        "cond": {"lhs": inner, "op": "<=", "rhs": 2}},
                                "op": "<=", "rhs": 6}))
    assert any("minutes_into_session is session-scoped" in e for e in errs)


def test_validate_accepts_structural_prims_inside_tf_bars_since():
    assert validate_spec(_spec({"lhs": {"prim": "bars_since", "tf": "1h",
                                        "cond": {"lhs": {"src": "close"}, "op": ">",
                                                 "rhs": {"prim": "swing_high", "k": 3}}},
                                "op": "<=", "rhs": 6})) == []


def test_validate_accepts_session_prim_in_untf_bars_since():
    assert validate_spec(_spec({"lhs": {"prim": "bars_since",
                                        "cond": {"lhs": {"prim": "day_of_week"},
                                                 "op": "<", "rhs": 1}},
                                "op": "<=", "rhs": 1})) == []


def test_validate_survives_adversarially_deep_bars_since_cond():
    deep = {"lhs": {"src": "close"}, "op": ">", "rhs": 1}
    for _ in range(3000):
        deep = {"lhs": deep, "op": ">", "rhs": 1}
    errs = validate_spec(_spec({"lhs": {"prim": "bars_since", "tf": "1h", "cond": deep},
                                "op": "<=", "rhs": 1}))
    assert errs  # rejected with validation errors, never a crash
