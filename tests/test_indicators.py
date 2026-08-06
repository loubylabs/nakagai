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


def _extended_hours_bars(day="2026-01-05", pre=50.0, regular=100.0):
    """One session, 08:00 through 10:00 New York, flat inside each band.

    high, low and close all carry the band's price, so the typical price IS the
    price and every VWAP below is a number written here rather than derived.
    Volume is equal on every bar, which is generous to the defect: a real
    pre-market bar carries a fraction of the regular volume and still moved the
    level, because for the first bars after the bell it is most of what has
    accumulated.
    """
    from nakagai.data.schema import EXCHANGE_TZ
    idx = pd.date_range(f"{day} 08:00", f"{day} 10:00", freq="15min",
                        tz=EXCHANGE_TZ).tz_convert("UTC")
    ny = idx.tz_convert(EXCHANGE_TZ)
    price = np.where(ny.hour * 60 + ny.minute < 570, pre, regular)
    return pd.DataFrame({"open": price, "high": price, "low": price,
                         "close": price, "volume": 1000.0}, index=idx)


def test_session_vwap_leaves_the_pre_market_out_of_the_accumulator():
    """chrvsd/nakagai#276, on the volume side.

    The caches are not RTH-only, so cumulating every bar of the New York date
    let a thin, wide-spread pre-market tape decide where VWAP sat for the first
    hour of trading: with six 08:00-to-09:15 bars at 50 in the accumulator, the
    09:30 bar read 57.14 rather than 100, and `close > vwap` answered a
    question about the pre-market rather than about the session. Every regular
    bar here trades at exactly 100, so the session's VWAP is 100 on every one
    of them and nothing else is a rounding difference.
    """
    bars = _extended_hours_bars()
    ny = bars.index.tz_convert("America/New_York")
    regular = np.asarray(ny.hour * 60 + ny.minute) >= 570
    vwap = ind.session_vwap(bars)
    assert regular.sum() == 3 and (~regular).sum() == 6
    assert np.allclose(vwap[regular], 100.0)
    # The exact wrong answer, named: six pre-market bars at 50 plus one regular
    # bar at 100, all on equal volume.
    assert vwap[regular].iloc[0] != pytest.approx((6 * 50.0 + 100.0) / 7)


def test_session_vwap_is_nan_before_the_bell_rather_than_the_pre_market_s_own():
    """A pre-market bar reads NaN, and it falls out of the accumulator being
    empty rather than being blanked separately: no volume has entered the
    session yet, so the zero-denominator guard answers NaN. A condition over
    NaN reads False, which is what keeps a play off a session VWAP that has
    not begun. Answering the pre-market's own VWAP instead would be a
    different measurement wearing the session's name."""
    bars = _extended_hours_bars()
    ny = bars.index.tz_convert("America/New_York")
    vwap = ind.session_vwap(bars)
    assert vwap[np.asarray(ny.hour * 60 + ny.minute) < 570].isna().all()


@pytest.mark.parametrize("labels, convention", [
    (["2026-01-05", "2026-01-06"], "midnight UTC"),
    (["2026-01-05 05:00", "2026-01-06 05:00"], "midnight Eastern")])
def test_session_vwap_reads_a_daily_frame_as_one_session_per_row(labels,
                                                                 convention):
    """A daily bar is labeled at midnight under both conventions the engine
    meets, and both sit outside [09:30, 16:00). Without rth_mask's
    session-frame branch every bar of a daily frame would be masked out and the
    whole series would come back NaN, silently, on a spec that reads VWAP off
    its own daily bars. Here one row is its own session, so VWAP is that row's
    typical price."""
    idx = pd.DatetimeIndex(labels, tz="UTC")
    bars = pd.DataFrame({"open": [100.0, 200.0], "high": [102.0, 203.0],
                         "low": [98.0, 196.0], "close": [100.0, 201.0],
                         "volume": 1000.0}, index=idx)
    vwap = ind.session_vwap(bars)
    assert vwap.iloc[0] == pytest.approx(100.0), convention
    assert vwap.iloc[1] == pytest.approx(200.0), convention


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
