"""Indicator library: correctness on hand-checkable series, NaN discipline."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies import indicators as ind


def _bars(closes, start="2026-01-05 14:30", freq="15min", volume=1000.0):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    c = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c,
                         "volume": volume}, index=idx)


def test_sma_ema_basic():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ind.sma(s, 3).iloc[-1] == 4.0
    assert np.isnan(ind.sma(s, 3).iloc[1])
    e = ind.ema(s, 3)
    assert np.isnan(e.iloc[1]) and e.iloc[-1] == pytest.approx(4.0625)


def test_rsi_extremes():
    up = pd.Series(np.arange(30, dtype="float64"))
    down = pd.Series(np.arange(30, 0, -1, dtype="float64"))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    flat = pd.Series(np.full(30, 5.0))
    r = ind.rsi(flat, 14)
    assert np.isnan(r.iloc[-1]) or 0 <= r.iloc[-1] <= 100


def test_macd_shape_and_sign():
    s = pd.Series(np.linspace(100, 130, 80))
    m = ind.macd(s)
    assert set(m.columns) == {"macd", "signal", "hist"}
    assert m["macd"].iloc[-1] > 0  # rising series -> positive momentum


def test_bollinger_orders():
    s = pd.Series(np.random.default_rng(0).normal(100, 2, 100))
    b = ind.bollinger(s, 20, 2.0)
    tail = b.dropna()
    assert (tail["upper"] >= tail["mid"]).all() and (tail["mid"] >= tail["lower"]).all()


def test_atr_positive_and_nan_leadin():
    bars = _bars(np.linspace(100, 110, 40))
    a = ind.atr(bars, 14)
    assert np.isnan(a.iloc[5])
    assert a.iloc[-1] > 0


def test_donchian_excludes_current_bar():
    bars = _bars([10, 11, 12, 13, 14, 100])
    d = ind.donchian(bars, 5)
    # channel at the last bar covers the PRIOR 5 bars -> upper = 14 + 1(high pad)
    assert d["upper"].iloc[-1] == 15.0
    assert bars["close"].iloc[-1] > d["upper"].iloc[-1]  # breakout detectable


def test_roc_percent():
    s = pd.Series([100.0, 100.0, 110.0])
    assert ind.roc(s, 2).iloc[-1] == pytest.approx(10.0)


def test_session_vwap_resets_daily():
    day1 = _bars([100.0] * 4, start="2026-01-05 14:30")
    day2 = _bars([200.0] * 4, start="2026-01-06 14:30")
    bars = pd.concat([day1, day2])
    v = ind.session_vwap(bars)
    assert v.iloc[3] == pytest.approx(100.0)
    assert v.iloc[-1] == pytest.approx(200.0)  # no bleed from day 1


def test_ichimoku_shape_and_displacement():
    closes = np.linspace(100, 130, 80)
    ich = ind.ichimoku(_bars(closes), 5, 10, 20, 5)
    assert set(ich.columns) == {"tenkan", "kijun", "senkou_a", "senkou_b"}
    tail = ich.dropna()
    assert (tail["tenkan"] >= tail["kijun"]).all()  # rising series: fast >= slow
    # displaced cloud lags a rising price
    assert tail["senkou_a"].iloc[-1] < ich["tenkan"].iloc[-1]


def test_supertrend_direction_flips():
    closes = np.concatenate([np.linspace(100, 130, 40), np.linspace(130, 90, 40)])
    st = ind.supertrend(_bars(closes), 10, 3.0)
    assert st["direction"].iloc[35] == 1
    assert st["direction"].iloc[-1] == -1


def test_crossed_above_below():
    a = pd.Series([1.0, 3.0])
    b = pd.Series([2.0, 2.0])
    assert ind.crossed_above(a, b) and ind.crossed_above(a, 2.0)
    assert not ind.crossed_below(a, b)
    assert ind.crossed_below(pd.Series([3.0, 1.0]), 2.0)
    # NaNs and short series never cross
    assert not ind.crossed_above(pd.Series([np.nan, 3.0]), 2.0)
    assert not ind.crossed_above(pd.Series([3.0]), 2.0)


def _ohlc(n=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=idx)


def test_highest_lowest_stdev_roll_and_warmup():
    from nakagai.strategies.indicators import highest, lowest, stdev
    s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    assert highest(s, 3).iloc[-1] == 5.0
    assert lowest(s, 3).iloc[-1] == 2.0
    assert pd.isna(highest(s, 3).iloc[1])          # warmup is NaN
    assert stdev(s, 3).iloc[-1] == s.iloc[-3:].std(ddof=0)


def test_stoch_bounded_0_100():
    from nakagai.strategies.indicators import stoch
    st = stoch(_ohlc(), 14, 3)
    valid = st.dropna()
    assert set(st.columns) == {"k", "d"}
    assert ((valid >= 0) & (valid <= 100)).all().all()


def test_adx_positive_and_warmup_nan():
    from nakagai.strategies.indicators import adx
    a = adx(_ohlc(), 14)
    assert pd.isna(a.iloc[5])
    assert (a.dropna() >= 0).all()


def test_obv_accumulates_signed_volume():
    from nakagai.strategies.indicators import obv
    bars = _ohlc(5)
    bars["close"] = [100, 101, 100, 100, 102]   # up, down, flat, up
    bars["volume"] = 10.0
    o = obv(bars)
    assert o.iloc[-1] == 10.0 - 10.0 + 0.0 + 10.0
