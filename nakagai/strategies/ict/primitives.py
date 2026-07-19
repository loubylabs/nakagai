"""Shared ICT market-structure primitives: swings and ATR."""

import numpy as np
import pandas as pd


def _strict_extrema(values: np.ndarray, k: int, find_max: bool) -> np.ndarray:
    """Boolean mask: strictly greater (or less) than every neighbor within k bars.

    Vectorized equivalent of the per-bar window scan. The engine calls this on
    the full visible history every replay bar, so it must not be a Python loop.
    """
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    if n < 2 * k + 1:
        return mask
    s = pd.Series(values)
    if find_max:
        left = s.rolling(k).max().shift(1).to_numpy()   # max of the k bars before i
        right = s[::-1].rolling(k).max().shift(1).to_numpy()[::-1]  # ...and after i
        mask = (values > left) & (values > right)       # NaN edges compare False
    else:
        left = s.rolling(k).min().shift(1).to_numpy()
        right = s[::-1].rolling(k).min().shift(1).to_numpy()[::-1]
        mask = (values < left) & (values < right)
    return mask


def swing_highs(df: pd.DataFrame, k: int = 2) -> pd.Series:
    h = df["high"]
    mask = _strict_extrema(h.to_numpy(dtype="float64"), k, find_max=True)
    return pd.Series(h.to_numpy(dtype="float64")[mask], index=h.index[mask], dtype="float64")


def swing_lows(df: pd.DataFrame, k: int = 2) -> pd.Series:
    l = df["low"]
    mask = _strict_extrema(l.to_numpy(dtype="float64"), k, find_max=False)
    return pd.Series(l.to_numpy(dtype="float64")[mask], index=l.index[mask], dtype="float64")


def atr(df: pd.DataFrame, n: int = 14) -> float:
    if len(df) < 2:
        return float("nan")
    prev_close = df["close"].shift(1)
    tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    return float(tr.dropna().tail(n).mean())
