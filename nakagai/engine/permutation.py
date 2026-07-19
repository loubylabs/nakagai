"""Bar permutation: null price series for Monte Carlo permutation tests.

Masters-style decomposition: each bar after the anchor splits into a gap
(log open vs previous close) and an intrabar shape (log high/low/close
relative to the bar's own open, kept tupled with that bar's volume). A
permuted series shuffles the gap sequence and the shape sequence
independently, then rebuilds OHLC from the anchor bar. The marginal return
distribution and bar geometry survive; the temporal ordering, which is the
only thing a real edge can exploit, does not.
"""

import hashlib

import numpy as np
import pandas as pd

from nakagai.data.schema import validate_bars


def permutation_seed(symbol: str, timeframe: str, epoch: str, i: int) -> int:
    """Stable across processes and sessions: same epoch, identical nulls."""
    key = f"{symbol}|{timeframe}|{epoch}|{i}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def permute_bars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    bars = validate_bars(df)
    n = len(bars)
    if n < 3:
        return bars.copy()
    o, h, l, c = (np.log(bars[k].to_numpy()) for k in ("open", "high", "low", "close"))
    v = bars["volume"].to_numpy()

    gaps = rng.permutation(o[1:] - c[:-1])
    order = rng.permutation(n - 1)
    dh, dl, dc = (h - o)[1:][order], (l - o)[1:][order], (c - o)[1:][order]
    vol = v[1:][order]

    # close-to-close log step of permuted bar i is gap + close-shape; the
    # cumulative sum anchors every bar to the original starting price.
    log_close = np.concatenate([[c[0]], c[0] + np.cumsum(gaps + dc)])
    log_open = np.concatenate([[o[0]], log_close[:-1] + gaps])
    log_high = np.concatenate([[h[0]], log_open[1:] + dh])
    log_low = np.concatenate([[l[0]], log_open[1:] + dl])
    return pd.DataFrame(
        {"open": np.exp(log_open), "high": np.exp(log_high),
         "low": np.exp(log_low), "close": np.exp(log_close),
         "volume": np.concatenate([[v[0]], vol])},
        index=bars.index)
