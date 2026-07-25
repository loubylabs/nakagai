"""Margin evaluation: graded signal strength with visibility-safe alignment."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.base import MarketContext
from nakagai.strategies.indicators import rsi as _rsi
from nakagai.strategies.rules.margins import condition_margin, margin_expr, group_margin, spec_margin


def _bars(closes, start="2026-01-05 14:30", freq="15min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


def _ctx(b15, b1h=None, b1d=None):
    return MarketContext("SPY", b15.index[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": b15, "1h": b1h if b1h is not None else b15,
                               "1d": b1d if b1d is not None else b15})


def test_comparison_margin_is_signed_distance():
    b = _bars(np.linspace(100, 120, 40))
    ctx = _ctx(b)
    gt = condition_margin({"lhs": {"src": "close"}, "op": ">",
                           "rhs": {"ind": "sma", "n": 5}}, ctx, b, "15m", {})
    lt = condition_margin({"lhs": {"src": "close"}, "op": "<",
                           "rhs": {"ind": "sma", "n": 5}}, ctx, b, "15m", {})
    sma5 = b["close"].rolling(5).mean()
    assert gt.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    assert lt.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])
    assert (gt.dropna() + lt.dropna()).abs().max() < 1e-12


def test_cross_margin_is_the_current_gap():
    b = _bars([100.0] * 30 + [99.0, 103.0])
    ctx = _ctx(b)
    m = condition_margin({"lhs": {"src": "close"}, "op": "crosses_above",
                          "rhs": {"ind": "sma", "n": 5}}, ctx, b, "15m", {})
    sma5 = b["close"].rolling(5).mean()
    assert m.iloc[-1] == pytest.approx(b["close"].iloc[-1] - sma5.iloc[-1])
    below = condition_margin({"lhs": {"src": "close"}, "op": "crosses_below",
                              "rhs": {"ind": "sma", "n": 5}}, ctx, b, "15m", {})
    assert below.iloc[-1] == pytest.approx(sma5.iloc[-1] - b["close"].iloc[-1])


def test_scalar_rhs_broadcasts():
    b = _bars(np.linspace(10, 50, 30))
    m = condition_margin({"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30},
                         _ctx(b), b, "15m", {})
    assert isinstance(m, pd.Series) and len(m) == len(b)
    expected_rsi = _rsi(b["close"], 14)
    assert m.iloc[-1] == pytest.approx(30 - expected_rsi.iloc[-1])


def test_daily_reference_not_visible_within_its_own_session():
    # Jan 6 15m rows must see the Jan 5 daily close (95), never Jan 6's (96):
    # the daily bar only closes at the end of its session. exprs._align would
    # leak 96 here; the margin walker must not.
    b15 = _bars([100.0] * 26, start="2026-01-06 14:30")
    b1d = _bars([95.0, 96.0], start="2026-01-05 00:00", freq="1D")
    out = margin_expr({"src": "close", "tf": "1d"}, _ctx(b15, b1d=b1d),
                      b15, "15m", {})
    assert (out == 95.0).all()


def test_hourly_reference_visible_only_after_its_close():
    # 1h bar labeled 14:00 closes at 15:00; the first 15m row that may use it
    # is 14:45 (whose own close is 15:00).
    b15 = _bars([100.0] * 8, start="2026-01-05 14:30")   # 14:30 .. 16:15
    b1h = _bars([90.0, 95.0], start="2026-01-05 13:00", freq="1h")
    out = margin_expr({"src": "close", "tf": "1h"}, _ctx(b15, b1h=b1h),
                      b15, "15m", {})
    assert out.loc["2026-01-05 14:30+00:00"] == 90.0
    assert out.loc["2026-01-05 14:45+00:00"] == 95.0
    assert (out.loc["2026-01-05 15:00+00:00":] == 95.0).all()


def test_primitive_margin_is_a_series():
    b = _bars(np.linspace(100, 110, 30))
    m = condition_margin({"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"prim": "prev_session_high"}},
                         _ctx(b), b, "15m", {})
    assert isinstance(m, pd.Series) and len(m) == len(b)


def test_tf_qualified_primitive_is_visibility_shifted():
    # prev_session_high on the 1h tf must evaluate on the 1h frame (not the
    # 15m driving frame) and land on the driving index shifted by the 1h/15m
    # close-delta gap: 1h closes 45 minutes later than a 15m bar labeled the
    # same time, so the prior session's 1h high only becomes visible 45
    # minutes after the 15m bar it would naively align to.
    b1h = pd.concat([
        _bars([199.5, 201.5], start="2026-01-05 13:00", freq="1h"),
        _bars([190.0, 191.0], start="2026-01-06 13:00", freq="1h"),
    ])
    b15 = _bars([100.0] * 8, start="2026-01-06 14:30")
    ctx = _ctx(b15, b1h=b1h)
    m = margin_expr({"prim": "prev_session_high", "tf": "1h"}, ctx, b15, "15m", {})
    assert isinstance(m, pd.Series)
    assert m.index.equals(b15.index)
    # Jan 5's session high (from closes 199.5/201.5 -> highs 200.0/202.0) is
    # the value every Jan 6 row should carry, ffilled in from 45 minutes
    # after the 1h bars close.
    assert (m == 202.0).all()


def test_group_margin_ranks_then_combines():
    b = _bars(np.linspace(100, 120, 40))
    ctx = _ctx(b)
    idx = b.index[10:]
    c1 = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    c2 = {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 70}
    m1 = condition_margin(c1, ctx, b, "15m", {}).loc[idx].rank(pct=True)
    m2 = condition_margin(c2, ctx, b, "15m", {}).loc[idx].rank(pct=True)
    g_all = group_margin({"all": [c1, c2]}, ctx, b, "15m", {}, idx)
    g_any = group_margin({"any": [c1, c2]}, ctx, b, "15m", {}, idx)
    both = pd.concat([m1, m2], axis=1)
    pd.testing.assert_series_equal(g_all, both.min(axis=1, skipna=False),
                                   check_names=False)
    pd.testing.assert_series_equal(g_any, both.max(axis=1), check_names=False)
    assert g_all.dropna().between(0, 1).all()


def test_all_group_propagates_warmup_nan_any_survives_it():
    b = _bars(np.linspace(100, 120, 40))
    ctx = _ctx(b)
    idx = b.index          # includes sma warmup rows
    c_warm = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 20}}
    c_ok = {"lhs": {"src": "close"}, "op": ">", "rhs": 0}
    g_all = group_margin({"all": [c_warm, c_ok]}, ctx, b, "15m", {}, idx)
    g_any = group_margin({"any": [c_warm, c_ok]}, ctx, b, "15m", {}, idx)
    assert g_all.iloc[:19].isna().all() and g_all.iloc[19:].notna().all()
    assert g_any.notna().all()


def test_spec_margin_long_only_and_sides():
    b = _bars(np.linspace(100, 120, 40))
    ctx = _ctx(b)
    idx = b.index[10:]
    cond = {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 5}}
    long_only = {"version": 2, "name": "t", "timeframe": "15m",
                 "long": {"all": [cond]}}
    short_only = {"version": 2, "name": "t", "timeframe": "15m",
                  "short": {"all": [cond]}}
    ml = spec_margin(long_only, ctx, idx)
    ms = spec_margin(short_only, ctx, idx)
    pd.testing.assert_series_equal(ml, -ms, check_names=False)
    assert ml.dropna().between(0, 1).all()


def test_spec_margin_empty_spec_is_empty():
    b = _bars(np.linspace(100, 120, 40))
    out = spec_margin({"version": 2, "name": "t", "timeframe": "15m"},
                      _ctx(b), b.index)
    assert isinstance(out, pd.Series) and out.empty
