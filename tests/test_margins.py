"""Margin evaluation: the ranking layer over FrameEval's node values.

The walk itself is FrameEval's and is covered in test_frame_eval.py; what is
left here is only what margins.py still owns, namely signed distances and the
rank-then-combine step.
"""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.strategies.indicators import rsi as _rsi
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.margins import (condition_margin, group_margin,
                                              spec_margin)


def _bars(closes, start="2026-01-05 14:30", freq="15min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


def _fe(b15, b1h=None, b1d=None):
    return FrameEval({"15m": b15, "1h": b1h if b1h is not None else b15,
                      "1d": b1d if b1d is not None else b15}, TFS, "SPY")


def test_icir_no_longer_abstains_on_end_anchored_specs():
    # The abstention existed because the margin walker broadcast one
    # end-of-frame float across every row. The primitives are row-wise now, so
    # the guard is not merely unused, it is wrong: it would hide a real lens.
    import nakagai.icir as icir
    assert not hasattr(icir, "_uses_end_anchored")
    assert not hasattr(icir, "END_ANCHORED_PRIMS")


def test_comparison_margin_is_signed_distance():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    gt = condition_margin({"lhs": {"src": "close"}, "op": ">",
                           "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    lt = condition_margin({"lhs": {"src": "close"}, "op": "<",
                           "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    sma5 = b["close"].rolling(5).mean()
    assert gt.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    assert lt.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])
    assert (gt.dropna() + lt.dropna()).abs().max() < 1e-12


def test_cross_margin_is_the_current_gap():
    b = _bars([100.0] * 30 + [99.0, 103.0])
    fe = _fe(b)
    m = condition_margin({"lhs": {"src": "close"}, "op": "crosses_above",
                          "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    sma5 = b["close"].rolling(5).mean()
    assert m.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    below = condition_margin({"lhs": {"src": "close"}, "op": "crosses_below",
                              "rhs": {"ind": "sma", "n": 5}}, fe, "15m")
    assert below.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])


def test_scalar_rhs_broadcasts():
    b = _bars(np.linspace(10, 50, 30))
    m = condition_margin({"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30},
                         _fe(b), "15m")
    assert isinstance(m, pd.Series) and len(m) == len(b)
    expected_rsi = _rsi(b["close"], 14)
    assert m.iloc[-1] == pytest.approx(30 - expected_rsi.iloc[-1])


def test_primitive_margin_is_a_series():
    b = _bars(np.linspace(100, 110, 30))
    m = condition_margin({"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"prim": "prev_session_high"}}, _fe(b), "15m")
    assert isinstance(m, pd.Series) and len(m) == len(b)


def test_group_margin_ranks_then_combines():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    c1 = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c2 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    m1 = condition_margin(c1, fe, "15m").loc[idx].rank(pct=True)
    m2 = condition_margin(c2, fe, "15m").loc[idx].rank(pct=True)
    g_all = group_margin({"all": [c1, c2]}, fe, "15m", idx)
    g_any = group_margin({"any": [c1, c2]}, fe, "15m", idx)
    both = pd.concat([m1, m2], axis=1)
    pd.testing.assert_series_equal(g_all, both.min(axis=1, skipna=False),
                                   check_names=False)
    pd.testing.assert_series_equal(g_any, both.max(axis=1), check_names=False)
    assert g_all.dropna().between(0, 1).all()


def test_group_margin_handles_nested_groups():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    c1 = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c2 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    c_flat = {"lhs": {"src": "close"}, "op": ">", "rhs": 0}
    group = {"all": [{"any": [c1, c2]}, c_flat]}
    g = group_margin(group, fe, "15m", idx)

    m1 = condition_margin(c1, fe, "15m").loc[idx].rank(pct=True)
    m2 = condition_margin(c2, fe, "15m").loc[idx].rank(pct=True)
    inner_any = pd.concat([m1, m2], axis=1).max(axis=1)
    m_flat = condition_margin(c_flat, fe, "15m").loc[idx].rank(pct=True)
    expected = pd.concat([inner_any.rank(pct=True), m_flat], axis=1).min(axis=1, skipna=False)
    pd.testing.assert_series_equal(g, expected, check_names=False)


def test_all_group_propagates_warmup_nan_any_survives_it():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index          # includes sma warmup rows
    c_warm = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}
    c_ok = {"lhs": {"src": "close"}, "op": ">", "rhs": 0}
    g_all = group_margin({"all": [c_warm, c_ok]}, fe, "15m", idx)
    g_any = group_margin({"any": [c_warm, c_ok]}, fe, "15m", idx)
    assert g_all.iloc[:19].isna().all() and g_all.iloc[19:].notna().all()
    assert g_any.notna().all()


def test_spec_margin_long_only_and_sides():
    b = _bars(np.linspace(100, 120, 40))
    fe = _fe(b)
    idx = b.index[10:]
    cond = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    long_only = {"version": 2, "name": "t", "timeframe": "15m",
                 "long": {"all": [cond]}}
    short_only = {"version": 2, "name": "t", "timeframe": "15m",
                  "short": {"all": [cond]}}
    ml = spec_margin(long_only, fe, idx)
    ms = spec_margin(short_only, fe, idx)
    pd.testing.assert_series_equal(ml, -ms, check_names=False)
    assert ml.dropna().between(0, 1).all()


def test_spec_margin_empty_spec_is_empty():
    b = _bars(np.linspace(100, 120, 40))
    out = spec_margin({"version": 2, "name": "t", "timeframe": "15m"},
                      _fe(b), b.index)
    assert isinstance(out, pd.Series) and out.empty


def test_bars_since_shares_the_one_walker_and_its_alignment():
    # There used to be two walkers keying their memos identically, so a
    # cross-timeframe node reachable both inside a bars_since cond and as a
    # direct condition poisoned whichever cache filled second. One walker means
    # one memo and one alignment: the daily close stays the previous session's
    # on both routes rather than leaking Jan 6's into Jan 6's own rows.
    b15 = _bars([100.0] * 26, start="2026-01-06 14:30")
    b1d = _bars([95.0, 96.0], start="2026-01-05 00:00", freq="1D")
    fe = _fe(b15, b1d=b1d)
    ref = {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "close", "tf": "1d"}}
    bars_since_cond = {"lhs": {"prim": "bars_since", "cond": ref}, "op": ">", "rhs": 0}
    group = {"all": [bars_since_cond, ref]}
    group_margin(group, fe, "15m", b15.index)
    assert (fe.series({"src": "close", "tf": "1d"}, "15m") == 95.0).all()
