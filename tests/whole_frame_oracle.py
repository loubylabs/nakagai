"""The OLD evaluation semantics, stated once, for tests to check against.

Cut every frame at `now` with closed_before, call the node's function on the
prefix, take the last value. That is exactly what the per-bar path did. This
shares only the function lookup tables with the implementation, never the walk,
so a bug in FrameEval cannot hide by being mirrored here.

Test-only. Never imported by library code.
"""

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import closed_before
from nakagai.strategies.rules.exprs import _BAR_FNS, _FRAME_FNS, _SERIES_FNS
from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.spec import ARG_DEFAULTS, BAR_INDICATORS

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b}


def prefix_value(node, frames: dict, eval_tf: str, now: pd.Timestamp,
                 tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> float:
    """The node's value at `now` under the old prefix-and-iloc[-1] semantics."""
    if isinstance(node, (int, float)):
        return float(node)
    cut = {tf: closed_before(f, tf, now, tfs) for tf, f in frames.items()}
    src_tf = node.get("tf", eval_tf)
    bars = cut[src_tf]
    if not len(bars):
        return float("nan")
    if "src" in node:
        return float(bars[node["src"]].iloc[-1])
    if "op" in node:
        vals = [prefix_value(a, frames, eval_tf, now, tfs) for a in node["args"]]
        if node["op"] == "abs":
            return abs(vals[0])
        if node["op"] in ("min", "max"):
            return (min if node["op"] == "min" else max)(vals)
        out = vals[0]
        for v in vals[1:]:
            out = _OPS[node["op"]](out, v)
        return float(out)
    if "ind" in node:
        name = node["ind"]
        a = {**ARG_DEFAULTS.get(name, {}),
             **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        if name in BAR_INDICATORS:
            out = _BAR_FNS[name](bars, a)
        else:
            of = node.get("of", {"src": "close"})
            series = bars[of["src"]] if "src" in of else None
            if series is None:
                raise ValueError("oracle supports only {'src': ...} in `of`")
            fn = _SERIES_FNS.get(name)
            out = fn(series, a["n"]) if fn else _FRAME_FNS[name](series, a)
        if isinstance(out, pd.DataFrame):
            out = out[a["field"]]
        return float(out.iloc[-1])
    name = node["prim"]
    a = {**PRIM_DEFAULTS.get(name, {}),
         **{k: v for k, v in node.items() if k not in ("prim", "tf")}}
    out = PRIMITIVES[name]["fn"](None, bars, **a)
    return float(out.iloc[-1]) if isinstance(out, pd.Series) else float(out)
