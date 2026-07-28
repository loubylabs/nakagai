"""RuleSpec v2 indicator dispatch tables and scalar/series math helpers.

Not a walker. The one walk over the grammar lives in frame_eval.FrameEval;
this module holds only what that walk dispatches into: the three indicator
tables (series-of-a-series, frame-returning, whole-bar) plus the arithmetic
helpers. Keeping them here rather than in frame_eval.py lets the test oracle
share the function lookup without sharing the walk, so a bug in the walker
cannot hide by being mirrored in its own oracle.
"""

import pandas as pd

from nakagai.strategies import indicators as ind

_SERIES_FNS = {"sma": lambda s, n: ind.sma(s, n), "ema": lambda s, n: ind.ema(s, n),
               "rsi": lambda s, n: ind.rsi(s, n), "roc": lambda s, n: ind.roc(s, n),
               "zscore": lambda s, n: ind.zscore(s, n), "highest": lambda s, n: ind.highest(s, n),
               "lowest": lambda s, n: ind.lowest(s, n), "stdev": lambda s, n: ind.stdev(s, n)}
_FRAME_FNS = {"macd": lambda s, a: ind.macd(s, a["fast"], a["slow"], a["signal"]),
              "bb": lambda s, a: ind.bollinger(s, a["n"], a["k"])}
_BAR_FNS = {"atr": lambda b, a: ind.atr(b, a["n"]),
            "donchian": lambda b, a: ind.donchian(b, a["n"]),
            "supertrend": lambda b, a: ind.supertrend(b, a["n"], a["mult"]),
            "vwap": lambda b, a: ind.session_vwap(b),
            "stoch": lambda b, a: ind.stoch(b, a["n"], a["d"]),
            "adx": lambda b, a: ind.adx(b, a["n"]),
            "obv": lambda b, a: ind.obv(b),
            "ichimoku": lambda b, a: ind.ichimoku(b, a["tenkan_n"], a["kijun_n"],
                                                  a["senkou_n"], a["disp"]),
            "keltner": lambda b, a: ind.keltner(b, a["n"], a["mult"]),
            "cci": lambda b, a: ind.cci(b, a["n"]),
            "mfi": lambda b, a: ind.mfi(b, a["n"]),
            "wpr": lambda b, a: ind.wpr(b, a["n"])}


def _math(op: str, args: list):
    if op == "abs":
        return args[0].abs() if isinstance(args[0], pd.Series) else abs(args[0])
    out = args[0]
    for a in args[1:]:
        if op == "+":
            out = out + a
        elif op == "-":
            out = out - a
        elif op == "*":
            out = out * a
        elif op == "/":
            denom = a.replace(0.0, float("nan")) if isinstance(a, pd.Series) else \
                (float("nan") if a == 0 else a)
            out = out / denom
        elif op in ("min", "max"):
            both = pd.concat([_as_series(out, a), _as_series(a, out)], axis=1)
            out = both.min(axis=1) if op == "min" else both.max(axis=1)
    return out


def _as_series(v, like):
    if isinstance(v, pd.Series):
        return v
    idx = like.index if isinstance(like, pd.Series) else None
    return pd.Series(v, index=idx)
