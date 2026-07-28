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


def find_fvgs(df: pd.DataFrame, min_size_atr: float = 0.25,
              lookback: int = 40) -> list[tuple[FVG, str]]:
    """Every qualifying 3-candle imbalance in the window with its lifecycle
    state: "open" (never wick-filled), "filled" (a later wick reached the far
    boundary), or "inverted" (a later bar CLOSED through the far boundary,
    flipping the zone's polarity)."""
    df = df.tail(lookback)
    a = atr(df)
    if len(df) < 3 or not a or np.isnan(a):
        return []
    out: list[tuple[FVG, str]] = []
    highs, lows = df["high"].to_numpy(), df["low"].to_numpy()
    closes = df["close"].to_numpy()
    for i in range(2, len(df)):
        later_l, later_h, later_c = lows[i + 1:], highs[i + 1:], closes[i + 1:]
        if lows[i] > highs[i - 2] and (lows[i] - highs[i - 2]) >= min_size_atr * a:
            bottom, top = highs[i - 2], lows[i]
            if later_c.size and later_c.min() < bottom:
                state = "inverted"
            elif later_l.size and later_l.min() <= bottom:
                state = "filled"
            else:
                state = "open"
            out.append((FVG(df.index[i], Direction.LONG, float(top), float(bottom)), state))
        elif highs[i] < lows[i - 2] and (lows[i - 2] - highs[i]) >= min_size_atr * a:
            bottom, top = highs[i], lows[i - 2]
            if later_c.size and later_c.max() > top:
                state = "inverted"
            elif later_h.size and later_h.max() >= top:
                state = "filled"
            else:
                state = "open"
            out.append((FVG(df.index[i], Direction.SHORT, float(top), float(bottom)), state))
    return out
