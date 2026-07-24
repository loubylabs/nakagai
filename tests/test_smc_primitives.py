"""leg_retrace and order_block: SMC primitives over confirmed structure."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.rules.primitives import (
    ARG_DEFAULTS, PRIMITIVES, leg_retrace, order_block,
)


def _bars(highs, lows, closes=None, opens=None):
    n = len(highs)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else closes
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1000.0}, index=idx)


# swing high 20 at i=2 (confirmed i=4), swing low 10 at i=6 (confirmed i=8)
HI = [15, 18, 20, 18, 15, 14, 12, 14, 15, 16]
LO = [13, 16, 18, 16, 13, 12, 10, 12, 13, 14]


def test_leg_retrace_long_is_position_below_the_high():
    bars = _bars(HI, LO, closes=[14, 17, 19, 17, 14, 13, 11, 13, 14, 15])
    out = leg_retrace(None, bars, direction="long", k=2)
    # both swings known only from i=8 (low confirms last): H=20, L=10
    assert out.iloc[:8].isna().all()
    assert out.iloc[8] == pytest.approx((20 - 14) / (20 - 10))
    assert out.iloc[9] == pytest.approx((20 - 15) / (20 - 10))


def test_leg_retrace_short_mirrors():
    bars = _bars(HI, LO, closes=[14, 17, 19, 17, 14, 13, 11, 13, 14, 15])
    out = leg_retrace(None, bars, direction="short", k=2)
    assert out.iloc[8] == pytest.approx((14 - 10) / (20 - 10))


def test_leg_retrace_nan_on_degenerate_range():
    # After the rally, the pullback swing low (30, confirmed at i=9) sits ABOVE
    # the stale swing high (20): the range is degenerate until a new high confirms.
    # Highs rise monotonically after i=4, so no new swing high ever does.
    hi = [15, 18, 20, 18, 15, 33, 34, 35, 36, 37, 38, 39]
    lo = [13, 16, 18, 16, 13, 32, 31, 30, 31, 32, 33, 34]
    bars = _bars(hi, lo)
    out = leg_retrace(None, bars, direction="long", k=2)
    # i=6..8: H=20, L=13 (the i=4 low confirmed at i=6): a real range
    assert out.iloc[6:9].notna().all()
    # i=9 on: the swing low updates to 30 > H=20, so the range reads NaN
    assert out.iloc[9:].isna().all()


def test_leg_retrace_never_repaints():
    bars = _bars(HI, LO, closes=[14, 17, 19, 17, 14, 13, 11, 13, 14, 15])
    full = leg_retrace(None, bars, direction="long", k=2)
    trunc = leg_retrace(None, bars.iloc[:9], direction="long", k=2)
    pd.testing.assert_series_equal(full.iloc[:9], trunc)


def test_leg_retrace_registered():
    assert PRIMITIVES["leg_retrace"]["args"] == {
        "direction": ("long", "short"), "k": (1, 10)}
    assert ARG_DEFAULTS["leg_retrace"] == {"direction": "long", "k": 3}


# 20 quiet bars keep ATR ~1; then one red candle (the order block), then a
# bullish displacement candle with a 4-point body.
OB_QUIET = [[100, 100.5, 99.5, 100]] * 20
OB_SET = OB_QUIET + [[101, 101.6, 99.8, 100.0],    # red candle: the OB
                     [100.0, 104.2, 99.9, 104.0]]  # displacement up


def _obars(rows):
    idx = pd.date_range("2026-01-05 14:30", periods=len(rows), freq="15min", tz="UTC")
    df = pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = 1000.0
    return df


def test_order_block_long_finds_last_red_before_displacement():
    bars = _obars(OB_SET)
    assert order_block(None, bars, direction="long", field="top") == pytest.approx(101.6)
    assert order_block(None, bars, direction="long", field="bottom") == pytest.approx(99.8)
    assert order_block(None, bars, direction="long", field="mid") == pytest.approx((101.6 + 99.8) / 2)


def test_order_block_nan_without_displacement():
    assert np.isnan(order_block(None, _obars(OB_QUIET), direction="long"))


def test_order_block_nan_without_opposing_candle():
    rows = [[100 + i, 101 + i, 99.5 + i, 100.8 + i] for i in range(20)]  # all green
    rows.append([120.0, 124.4, 119.9, 124.2])  # displacement, but no red before it
    assert np.isnan(order_block(None, _obars(rows), direction="long"))


def test_order_block_body_atr_threshold():
    bars = _obars(OB_SET)
    assert np.isnan(order_block(None, bars, direction="long", body_atr=5.0))


def test_order_block_short_mirrors():
    rows = OB_QUIET + [[100.0, 101.7, 99.9, 101.5],   # green candle: the OB
                       [101.5, 101.6, 97.2, 97.4]]    # displacement down
    bars = _obars(rows)
    assert order_block(None, bars, direction="short", field="bottom") == pytest.approx(99.9)


def test_order_block_registered():
    assert PRIMITIVES["order_block"]["args"] == {
        "direction": ("long", "short"), "field": ("top", "bottom", "mid"),
        "body_atr": (0.5, 5.0), "lookback": (10, 200)}
    assert ARG_DEFAULTS["order_block"] == {
        "direction": "long", "field": "top", "body_atr": 1.5, "lookback": 40}
