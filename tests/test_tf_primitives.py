"""tf on primitive nodes: HTF structure with LTF entries, lookahead-safe."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules.exprs import eval_expr
from nakagai.strategies.rules.primitives import _swing
from nakagai.strategies.rules.spec import _expr_text, validate_spec


def _frame(vals, freq, start="2026-01-05 14:30"):
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.DataFrame({"open": vals, "high": [v + 1 for v in vals],
                         "low": [v - 1 for v in vals], "close": vals,
                         "volume": 1000.0}, index=idx)


def _ctx(f15, f1h):
    return MarketContext(symbol="SPY", now=f15.index[-1],
                         bars={"15m": f15, "1h": f1h}, tfs=DEFAULT_TIMEFRAMES)


H1 = [100, 104, 108, 104, 100, 98, 96, 98, 100, 102, 104, 106]


def test_prim_tf_aligns_htf_series_to_driving_index():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")  # 48 x 15m spanning the same 12h
    out = eval_expr({"prim": "swing_high", "k": 2, "tf": "1h"}, _ctx(f15, f1h), f15, {})
    expected = (_swing(f1h, "high", 2, find_max=True)
                .reindex(f15.index.union(f1h.index)).ffill().reindex(f15.index))
    pd.testing.assert_series_equal(out, expected)


def test_prim_tf_value_is_from_the_last_closed_htf_bar_only():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")
    ctx = _ctx(f15, f1h)
    out = eval_expr({"prim": "swing_high", "k": 2, "tf": "1h"}, ctx, f15, {})
    sw = _swing(f1h, "high", 2, find_max=True)
    for ts in f15.index:
        visible = sw[sw.index <= ts]
        want = visible.iloc[-1] if len(visible) else np.nan
        got = out.loc[ts]
        assert (np.isnan(want) and np.isnan(got)) or got == want


def test_bars_since_tf_counts_htf_bars():
    f1h = _frame(H1, "1h")
    f15 = _frame(list(range(100, 148)), "15min")
    node = {"prim": "bars_since", "tf": "1h",
            "cond": {"lhs": {"src": "close"}, "op": ">", "rhs": 107}}
    out = eval_expr(node, _ctx(f15, f1h), f15, {})
    # close > 107 only at 1h bar index 2; the last 15m bar sits 9 hourly bars later
    assert out.iloc[-1] == pytest.approx(9.0)


def test_memo_keeps_tf_variants_apart():
    f1h = _frame(H1, "1h")
    f15 = _frame([100] * 20 + [110, 130, 110] + [100] * 25, "15min")
    ctx, memo = _ctx(f15, f1h), {}
    with_tf = eval_expr({"prim": "swing_high", "k": 2, "tf": "1h"}, ctx, f15, memo)
    without = eval_expr({"prim": "swing_high", "k": 2}, ctx, f15, memo)
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
    errs = validate_spec(_spec({"lhs": {"prim": "swing_high", "tf": "4h"},
                                "op": "<", "rhs": {"src": "close"}}))
    assert any("tf must be one of" in e for e in errs)


def test_describe_renders_tf_suffix():
    assert _expr_text({"prim": "swing_high", "k": 3, "tf": "1h"}) == "swing_high(3)[1h]"
