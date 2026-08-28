"""The OLD evaluation semantics, stated once, for tests to check against.

Cut every frame at `now` with closed_before, call the node's function on the
prefix, take the last value. That is exactly what the per-bar path did. This
shares only the function lookup tables with the implementation, never the walk,
so a bug in FrameEval cannot hide by being mirrored here.

Test-only. Never imported by library code.
"""

import operator

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import closed_before
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "/": lambda a, b: a / b}
_CMP = {">": operator.gt, "<": operator.lt,
        ">=": operator.ge, "<=": operator.le}


def _condition_mask(cond: dict, bars: pd.DataFrame, frames: dict,
                    symbol: str, tf: str,
                    tfs: TimeframeSet, vocabulary: Vocabulary) -> pd.Series:
    """The condition row by row, every row under the same prefix rule.

    bars_since is the one primitive that wants a whole boolean series rather
    than a single value, and the old path built it by evaluating the condition
    as one series over the prefix frame. Every node in the grammar is causal,
    which is the property the equivalence test pins for the operands
    themselves, so row j of that series is the operand's own prefix value at
    row j. Building it that way keeps this file to a single rule, cut and take
    the last value, instead of growing a second series walker that could mirror
    FrameEval's mistakes. Comparisons against NaN are False here exactly as
    they are in pandas.
    """
    if tf in tfs.session_aligned:
        raise ValueError("the oracle evaluates bars_since on intraday frames only")
    delta = tfs.deltas[tf]
    cmp = _CMP[cond["op"]]
    vals = [(prefix_value(
                cond["lhs"], frames, symbol, tf, ts + delta, tfs, vocabulary),
             prefix_value(
                cond["rhs"], frames, symbol, tf, ts + delta, tfs, vocabulary))
            for ts in bars.index]
    return pd.Series([bool(cmp(a, b)) for a, b in vals], index=bars.index)


def prefix_value(node, frames: dict, eval_symbol: str, eval_tf: str,
                 now: pd.Timestamp,
                 tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
                 vocabulary: Vocabulary | None = None) -> float:
    """One pair-keyed node value under old prefix-and-iloc[-1] semantics."""
    vocabulary = resolve_vocabulary(vocabulary)
    if isinstance(node, (int, float)):
        return float(node)
    src_symbol = node.get("sym", eval_symbol)
    src_tf = node.get("tf", eval_tf)
    frame = frames[(src_symbol, src_tf)]
    bars = closed_before(frame, src_tf, now, tfs)
    if not len(bars):
        return float("nan")
    if "src" in node:
        return float(bars[node["src"]].iloc[-1])
    if "op" in node:
        vals = [prefix_value(
                    a, frames, src_symbol, src_tf, now, tfs, vocabulary)
                for a in node["args"]]
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
        term = vocabulary.indicators[name]
        a = {**term.defaults,
             **{k: v for k, v in node.items()
                if k not in ("ind", "of", "tf", "sym")}}
        if term.kind == "bar":
            out = term.fn(bars, a)
        else:
            of = node.get("of", {"src": "close"})
            series = bars[of["src"]] if "src" in of else None
            if series is None:
                raise ValueError("oracle supports only {'src': ...} in `of`")
            out = term.fn(series, a)
        if isinstance(out, pd.DataFrame):
            out = out[a["field"]]
        return float(out.iloc[-1])
    name = node["prim"]
    term = vocabulary.primitives[name]
    a = {**term.defaults,
         **{k: v for k, v in node.items()
            if k not in ("prim", "tf", "sym")}}
    if name == "bars_since":
        a["eval_fn"] = lambda cond, b: _condition_mask(
            cond, b, frames, src_symbol, src_tf, tfs, vocabulary)
    out = term.fn(None, bars, **a)
    return float(out.iloc[-1]) if isinstance(out, pd.Series) else float(out)
