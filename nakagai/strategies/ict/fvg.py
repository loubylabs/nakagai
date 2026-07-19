"""Fair value gaps: 3-candle imbalances left by displacement; entries on retrace into them."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nakagai.strategies.base import Direction
from nakagai.strategies.ict.primitives import atr


@dataclass(frozen=True)
class FVG:
    ts: pd.Timestamp
    direction: Direction
    top: float
    bottom: float


def find_unfilled_fvgs(df: pd.DataFrame, min_size_atr: float = 0.25, lookback: int = 40) -> list[FVG]:
    df = df.tail(lookback)
    a = atr(df)
    if len(df) < 3 or not a or np.isnan(a):
        return []
    out: list[FVG] = []
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    for i in range(2, len(df)):
        later_l = lows[i + 1 :]
        later_h = highs[i + 1 :]
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= min_size_atr * a:
            bottom, top = highs[i - 2], lows[i]
            filled = bool(later_l.size) and later_l.min() <= bottom
            if not filled:
                out.append(FVG(df.index[i], Direction.LONG, float(top), float(bottom)))
        elif highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= min_size_atr * a:
            bottom, top = highs[i], lows[i - 2]
            filled = bool(later_h.size) and later_h.max() >= top
            if not filled:
                out.append(FVG(df.index[i], Direction.SHORT, float(top), float(bottom)))
    return out
