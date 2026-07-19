"""Golden + no-lookahead tests for the four Playbook indicators."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies import indicators as ind


def _bars(closes, spread=1.0, vol=1000.0):
    idx = pd.date_range("2026-01-05", periods=len(closes), freq="D", tz="UTC")
    c = pd.Series(np.asarray(closes, dtype=float), index=idx)
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread,
                         "close": c, "volume": vol}, index=idx)


def test_cci_golden_value_on_linear_ramp():
    # tp == close (spread folded out by using high=low=close), tp = 1..30.
    # Window of 20: mean deviation is 5.0, last deviation 9.5,
    # cci = 9.5 / (0.015 * 5.0) = 126.666...
    bars = _bars(np.arange(1.0, 31.0), spread=0.0)
    assert round(float(ind.cci(bars, 20).iloc[-1]), 2) == 126.67


def test_mfi_is_100_when_every_bar_is_up():
    bars = _bars(np.arange(1.0, 31.0))
    out = ind.mfi(bars, 14)
    assert float(out.iloc[-1]) == 100.0


def test_wpr_golden_value_on_linear_ramp():
    # n=14 window: high = c+1, low = (c-13)-1, range 15, close 1 below the high:
    # wpr = -100 * 1/15 = -6.67
    bars = _bars(np.arange(1.0, 31.0))
    assert round(float(ind.wpr(bars, 14).iloc[-1]), 2) == -6.67


def test_wpr_stays_in_bounds():
    rng = np.random.default_rng(7)
    bars = _bars(100 + rng.normal(0, 2, 80).cumsum())
    out = ind.wpr(bars, 14).dropna()
    assert ((out <= 0) & (out >= -100)).all()


def test_keltner_mid_is_ema_and_bands_bracket_it():
    bars = _bars(100 + np.sin(np.arange(60) / 5.0))
    kc = ind.keltner(bars, 20, 2.0)
    pd.testing.assert_series_equal(kc["mid"], ind.ema(bars["close"], 20),
                                   check_names=False)
    tail = kc.dropna()
    assert (tail["upper"] > tail["mid"]).all()
    assert (tail["lower"] < tail["mid"]).all()


@pytest.mark.parametrize("fn", [
    lambda b: ind.cci(b, 20),
    lambda b: ind.mfi(b, 14),
    lambda b: ind.wpr(b, 14),
    lambda b: ind.keltner(b, 20, 2.0)["upper"],
])
def test_no_lookahead(fn):
    rng = np.random.default_rng(11)
    bars = _bars(100 + rng.normal(0, 2, 40).cumsum())
    full = fn(bars)
    prefix = fn(bars.iloc[:30])
    assert float(full.iloc[29]) == pytest.approx(float(prefix.iloc[-1]))
