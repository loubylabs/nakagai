"""Shared indicator library for template strategies and the rule engine.

Every function takes OHLCV DataFrames (or close Series) and returns a Series
aligned to the input index. Pure pandas/numpy, with no state and no look-ahead: each
value uses only rows at or before its own timestamp.
"""

import numpy as np
import pandas as pd

from nakagai.data.schema import EXCHANGE_TZ


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(int(n)).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (smoothed with alpha=1/n)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    n = int(n)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_down != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(int(n)).std(ddof=0)
    return pd.DataFrame({"upper": mid + k * sd, "mid": mid, "lower": mid - k * sd})


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder-smoothed average true range."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"],
         (bars["high"] - prev_close).abs(),
         (bars["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    n = int(n)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def donchian(bars: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Channel of the prior n bars (shifted: excludes the current bar, so a
    close above `upper` is a genuine breakout of past highs)."""
    n = int(n)
    upper = bars["high"].rolling(n).max().shift(1)
    lower = bars["low"].rolling(n).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower, "mid": (upper + lower) / 2})


def roc(close: pd.Series, n: int) -> pd.Series:
    """Rate of change over n bars, in percent."""
    return close.pct_change(int(n)) * 100


def zscore(close: pd.Series, n: int = 20) -> pd.Series:
    m = close.rolling(int(n)).mean()
    sd = close.rolling(int(n)).std(ddof=0)
    return (close - m) / sd.replace(0.0, np.nan)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, reset each NY session date."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical * bars["volume"]
    dates = bars.index.tz_convert(EXCHANGE_TZ).date
    grouped_pv = pd.Series(pv.values, index=bars.index).groupby(dates).cumsum()
    grouped_v = bars["volume"].groupby(dates).cumsum()
    return grouped_pv / grouped_v.replace(0, np.nan)


def ichimoku(bars: pd.DataFrame, tenkan_n: int = 9, kijun_n: int = 26,
             senkou_n: int = 52, disp: int = 26) -> pd.DataFrame:
    """Ichimoku lines. senkou_a/senkou_b are displaced forward, so at any row
    they hold the cloud that applies to THAT bar (no look-ahead)."""
    def midline(n: int) -> pd.Series:
        n = int(n)
        return (bars["high"].rolling(n).max() + bars["low"].rolling(n).min()) / 2

    tenkan, kijun = midline(tenkan_n), midline(kijun_n)
    disp = int(disp)
    return pd.DataFrame({
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(disp),
        "senkou_b": midline(senkou_n).shift(disp),
    })


def supertrend(bars: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """Supertrend line + direction (+1 up-trend, -1 down-trend).

    Iterative by definition (bands ratchet against the trend), so this is a
    python loop, fine at the bar counts strategies see per on_bar call.
    """
    a = atr(bars, n)
    hl2 = (bars["high"] + bars["low"]) / 2
    upper_basic = hl2 + mult * a
    lower_basic = hl2 - mult * a
    close = bars["close"]

    line = np.full(len(bars), np.nan)
    direction = np.zeros(len(bars))
    up, lo = np.nan, np.nan
    d = 1
    for i in range(len(bars)):
        ub, lb, c = upper_basic.iloc[i], lower_basic.iloc[i], close.iloc[i]
        if np.isnan(ub) or np.isnan(lb):
            continue
        # ratchet: bands only tighten while the trend holds
        up = ub if np.isnan(up) or ub < up or close.iloc[i - 1] > up else up
        lo = lb if np.isnan(lo) or lb > lo or close.iloc[i - 1] < lo else lo
        if d == 1 and c < lo:
            d, up = -1, ub
        elif d == -1 and c > up:
            d, lo = 1, lb
        direction[i] = d
        line[i] = lo if d == 1 else up
    return pd.DataFrame({"line": line, "direction": direction}, index=bars.index)


def crossed_above(a: pd.Series, b) -> bool:
    """True when a crossed above b between the last two bars. b may be a
    Series or a scalar."""
    if len(a) < 2:
        return False
    b_prev, b_now = (b.iloc[-2], b.iloc[-1]) if isinstance(b, pd.Series) else (b, b)
    vals = a.iloc[-2], a.iloc[-1], b_prev, b_now
    if any(pd.isna(v) for v in vals):
        return False
    return a.iloc[-2] <= b_prev and a.iloc[-1] > b_now


def crossed_below(a: pd.Series, b) -> bool:
    if len(a) < 2:
        return False
    b_prev, b_now = (b.iloc[-2], b.iloc[-1]) if isinstance(b, pd.Series) else (b, b)
    vals = a.iloc[-2], a.iloc[-1], b_prev, b_now
    if any(pd.isna(v) for v in vals):
        return False
    return a.iloc[-2] >= b_prev and a.iloc[-1] < b_now


def highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).max()


def lowest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).min()


def stdev(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n)).std(ddof=0)


def stoch(bars: pd.DataFrame, n: int = 14, d: int = 3) -> pd.DataFrame:
    """Stochastic oscillator %K (fast) and %D (SMA of %K)."""
    n, d = int(n), int(d)
    lo = bars["low"].rolling(n).min()
    hi = bars["high"].rolling(n).max()
    k = 100 * (bars["close"] - lo) / (hi - lo).replace(0.0, np.nan)
    return pd.DataFrame({"k": k, "d": k.rolling(d).mean()})


def adx(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's average directional index."""
    n = int(n)
    up = bars["high"].diff()
    down = -bars["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    a = atr(bars, n).replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def obv(bars: pd.DataFrame) -> pd.Series:
    """On-balance volume: cumulative volume signed by the close-to-close move."""
    sign = np.sign(bars["close"].diff()).fillna(0.0)
    return (sign * bars["volume"]).cumsum()


def keltner(bars: pd.DataFrame, n: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Keltner channels: EMA midline with ATR bands."""
    mid = ema(bars["close"], n)
    band = float(mult) * atr(bars, n)
    return pd.DataFrame({"upper": mid + band, "mid": mid, "lower": mid - band})


def cci(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Commodity channel index over the typical price (Lambert's 0.015 scale)."""
    n = int(n)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    m = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * mad.replace(0.0, np.nan))


def mfi(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Money flow index: volume-weighted RSI of the typical price."""
    n = int(n)
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    flow = tp * bars["volume"]
    delta = tp.diff()
    pos = flow.where(delta > 0, 0.0).rolling(n).sum()
    neg = flow.where(delta < 0, 0.0).rolling(n).sum()
    ratio = pos / neg.replace(0.0, np.nan)
    out = 100 - 100 / (1 + ratio)
    return out.where(neg != 0, 100.0)


def wpr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Williams %R: 0 at the n-bar high down to -100 at the n-bar low."""
    n = int(n)
    hi = bars["high"].rolling(n).max()
    lo = bars["low"].rolling(n).min()
    return -100 * (hi - bars["close"]) / (hi - lo).replace(0.0, np.nan)
