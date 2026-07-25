"""Margin evaluation: a spec's rule tree as a graded signal-strength series.

Evaluation-only (the ICIR lens); nothing here feeds the signal path. The
walker mirrors exprs._eval with ONE difference: cross-timeframe alignment is
visibility-shifted so a full-frame vectorized pass stays lookahead-safe. Row
t may use a bar from another timeframe only if that bar's CLOSE time is at or
before row t's own close time (the same guarantee closed_before gives the
engine per bar).

Caveats, not a blanket "stays lookahead-safe" claim:
- End-anchored scalar primitives (fvg_nearest, order_block) compute one float
  from the tail of the evaluation frame; broadcasting that across every row
  is lookahead. They are NOT safe here, and the icir lens abstains for specs
  that use them (see icir.py's END_ANCHORED_PRIMS).
- bars_since conditions evaluate through the signal path's plain (unshifted)
  alignment, via a private memo, not the walker's visibility-shifted one.
- Group members are ranked within the full evaluation window, so the
  combined margin is a diagnostic over that window, not a tradable
  point-in-time series.
"""

import json

import pandas as pd

from nakagai.data.schema import TimeframeSet
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules.exprs import (_BAR_FNS, _FRAME_FNS, _SERIES_FNS,
                                            _math, eval_condition_series)
from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.spec import ARG_DEFAULTS, BAR_INDICATORS


def _close_delta(tf: str, tfs: TimeframeSet) -> pd.Timedelta:
    """Label-to-close offset: session bars are labeled at midnight UTC of
    their session date and treated as closing one day later.
    This label+1day rule matches closed_before's NY-midnight rule only
    because cached bars are RTH-only; extended-hours data would need the
    NY rule here instead."""
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
         **{k: v for k, v in node.items() if k not in ("prim", "tf")}}
    if name == "bars_since":
        # A private memo, never the walker's: exprs' plain alignment and the
        # walker's visibility-shifted alignment key their memo entries the
        # same way ((id(bars), json.dumps(node))), so a cross-timeframe node
        # appearing both inside this cond and as a direct condition would
        # poison the walker's cache in either direction if they shared one.
        a["eval_fn"] = lambda cond, b: eval_condition_series(cond, ctx, b, {})
    return _align_visible(PRIMITIVES[name]["fn"](ctx, tf_bars, **a), bars, tf,
                          bars_tf, ctx.tfs)


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


def group_margin(group: dict, ctx: MarketContext, bars: pd.DataFrame,
                 bars_tf: str, memo: dict, index: pd.DatetimeIndex) -> pd.Series:
    """One all/any group as a percentile margin on `index` rows. Members are
    rank-transformed within `index` (the walk-forward window) so different
    native scales combine fairly; `all` takes the min (an unknown member
    keeps the row unknown), `any` the max (one known member suffices)."""
    key, items = next(iter(group.items()))
    members = [
        (group_margin(i, ctx, bars, bars_tf, memo, index)
         if ("all" in i or "any" in i)
         else condition_margin(i, ctx, bars, bars_tf, memo).loc[index])
        .rank(pct=True)
        for i in items]
    both = pd.concat(members, axis=1)
    return both.min(axis=1, skipna=False) if key == "all" else both.max(axis=1)


def spec_margin(spec: dict, ctx: MarketContext,
                index: pd.DatetimeIndex) -> pd.Series:
    """The spec as one graded factor on `index` rows: rank(long) minus
    rank(short), missing side = 0. Positive IC downstream always means the
    signal points the right way, for shorts too."""
    tf = spec.get("timeframe", "1h")
    bars = ctx.bars[tf]
    memo: dict = {}
    sides = {side: group_margin(spec[side], ctx, bars, tf, memo, index)
             for side in ("long", "short") if side in spec}
    if not sides:
        return pd.Series(dtype=float)
    zero = pd.Series(0.0, index=index)
    return sides.get("long", zero) - sides.get("short", zero)
