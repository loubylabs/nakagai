"""RuleSpec v2 scalar and series math helpers."""

import pandas as pd

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
