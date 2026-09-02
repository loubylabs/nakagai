"""One arithmetic contract for rule and screen expression evaluators."""

import pandas as pd


def _as_series(value, like):
    if isinstance(value, pd.Series):
        return value
    index = like.index if isinstance(like, pd.Series) else None
    return pd.Series(value, index=index)


def apply_math(op: str, args: list):
    """Apply one grammar math op to scalars, series, or a mixture.

    Infinity remains a number. Division by zero becomes NaN. Min and max skip
    NaN when another operand is available, matching pandas row reduction.
    """
    if op == "abs":
        return args[0].abs() if isinstance(args[0], pd.Series) else abs(args[0])
    out = args[0]
    for arg in args[1:]:
        if op == "+":
            out = out + arg
        elif op == "-":
            out = out - arg
        elif op == "*":
            out = out * arg
        elif op == "/":
            denominator = (
                arg.replace(0.0, float("nan"))
                if isinstance(arg, pd.Series)
                else float("nan") if arg == 0 else arg
            )
            out = out / denominator
        elif op in ("min", "max"):
            if isinstance(out, pd.Series) or isinstance(arg, pd.Series):
                values = pd.concat([
                    _as_series(out, arg),
                    _as_series(arg, out),
                ], axis=1)
                out = values.min(axis=1) if op == "min" else values.max(axis=1)
            else:
                available = [value for value in (out, arg)
                             if not pd.isna(value)]
                out = ((min(available) if op == "min" else max(available))
                       if available else float("nan"))
    return out
