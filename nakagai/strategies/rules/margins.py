"""Margin evaluation: a spec's rule tree as a graded signal-strength series.

Evaluation-only (the ICIR lens); nothing here feeds the signal path. The
walker mirrors exprs._eval with ONE difference: cross-timeframe alignment is
visibility-shifted so a full-frame vectorized pass stays lookahead-safe. Row
t may use a bar from another timeframe only if that bar's CLOSE time is at or
before row t's own close time (the same guarantee closed_before gives the
engine per bar). Known shared wart: bars_since conditions align via
eval_condition_series, same as the signal path.
"""

import json

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules.exprs import (_BAR_FNS, _FRAME_FNS, _SERIES_FNS,
                                            _math, eval_condition_series)
from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.spec import ARG_DEFAULTS, BAR_INDICATORS


def _close_delta(tf: str, tfs: TimeframeSet) -> pd.Timedelta:
    """Label-to-close offset: session bars are labeled at midnight UTC of
    their session date and treated as closing one day later."""
    return pd.Timedelta(days=1) if tf in tfs.session_aligned else tfs.deltas[tf]


def _align_visible(v, bars: pd.DataFrame, src_tf: str, dst_tf: str,
                   tfs: TimeframeSet):
    """ffill-align v onto bars.index, shifted so row t only sees src bars
    already closed by t's own close time."""
    if not isinstance(v, pd.Series) or v.index.equals(bars.index):
        return v
    shift = _close_delta(src_tf, tfs) - _close_delta(dst_tf, tfs)
    out = v.copy()
    out.index = out.index + shift
    return out.reindex(bars.index.union(out.index)).ffill().reindex(bars.index)


def margin_expr(node, ctx: MarketContext, bars: pd.DataFrame, bars_tf: str,
                memo: dict):
    if isinstance(node, (int, float)):
        return float(node)
    key = (id(bars), json.dumps(node, sort_keys=True))
    if key in memo:
        return memo[key]
    memo[key] = out = _margin_eval(node, ctx, bars, bars_tf, memo)
    return out


def _margin_eval(node: dict, ctx: MarketContext, bars: pd.DataFrame,
                 bars_tf: str, memo: dict):
    tf = node.get("tf", bars_tf)
    tf_bars = ctx.bars[node["tf"]] if "tf" in node else bars
    if "src" in node:
        return _align_visible(tf_bars[node["src"]], bars, tf, bars_tf, ctx.tfs)
    if "op" in node:
        args = [margin_expr(a, ctx, bars, bars_tf, memo) for a in node["args"]]
        return _math(node["op"], args)
    if "ind" in node:
        name = node["ind"]
        a = {**ARG_DEFAULTS.get(name, {}),
             **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        if name in BAR_INDICATORS:
            out = _BAR_FNS[name](tf_bars, a)
        else:
            of = node.get("of", {"src": "close"})
            series = margin_expr(of, ctx, tf_bars, tf, memo)
            if isinstance(series, float):
                series = pd.Series(series, index=tf_bars.index)
            fn = _SERIES_FNS.get(name)
            out = fn(series, a["n"]) if fn else _FRAME_FNS[name](series, a)
        if isinstance(out, pd.DataFrame):
            out = out[a["field"]]
        return _align_visible(out, bars, tf, bars_tf, ctx.tfs)
    name = node["prim"]
    a = {**PRIM_DEFAULTS.get(name, {}),
         **{k: v for k, v in node.items() if k != "prim"}}
    if name == "bars_since":
        a["eval_fn"] = lambda cond, b: eval_condition_series(cond, ctx, b, memo)
    return PRIMITIVES[name]["fn"](ctx, bars, **{k: v for k, v in a.items()
                                                if k != "prim"})


def condition_margin(cond: dict, ctx: MarketContext, bars: pd.DataFrame,
                     bars_tf: str, memo: dict) -> pd.Series:
    """Signed distance of a condition: positive = holds, magnitude = how
    strongly. Crosses grade the current gap; the cross event itself stays a
    signal-path concept."""
    lhs = margin_expr(cond["lhs"], ctx, bars, bars_tf, memo)
    rhs = margin_expr(cond["rhs"], ctx, bars, bars_tf, memo)
    if not isinstance(lhs, pd.Series):
        lhs = pd.Series(lhs, index=bars.index)
    if not isinstance(rhs, pd.Series):
        rhs = pd.Series(rhs, index=bars.index)
    if cond["op"] in (">", ">=", "crosses_above"):
        return lhs - rhs
    return rhs - lhs
