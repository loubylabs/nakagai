"""RuleSpec v2 expression evaluator: vectorized pandas over MarketContext."""

import json

import pandas as pd

from nakagai.strategies import indicators as ind
from nakagai.strategies.base import MarketContext
from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.spec import ARG_DEFAULTS, BAR_INDICATORS

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


def _align(v, bars: pd.DataFrame):
    """Bring a series computed on another timeframe onto the driving index.
    Frames only contain CLOSED bars (build_context), so ffill is lookahead-safe."""
    if isinstance(v, pd.Series) and not v.index.equals(bars.index):
        return v.reindex(bars.index.union(v.index)).ffill().reindex(bars.index)
    return v


def eval_expr(node, ctx: MarketContext, bars: pd.DataFrame, memo: dict):
    if isinstance(node, (int, float)):
        return float(node)
    # Frame-aware key: the same sub-node evaluated against different driving
    # frames (e.g. inside a tf indicator's `of`) must not share a memo slot.
    # id(bars) is stable for the memo's lifetime (one on_bar call).
    key = (id(bars), json.dumps(node, sort_keys=True))
    if key in memo:
        return memo[key]
    memo[key] = out = _eval(node, ctx, bars, memo)
    return out


def _eval(node: dict, ctx: MarketContext, bars: pd.DataFrame, memo: dict):
    tf_bars = ctx.bars[node["tf"]] if "tf" in node else bars
    if "src" in node:
        return _align(tf_bars[node["src"]], bars)
    if "op" in node:
        args = [eval_expr(a, ctx, bars, memo) for a in node["args"]]
        return _math(node["op"], args)
    if "ind" in node:
        name = node["ind"]
        a = {**ARG_DEFAULTS.get(name, {}),
             **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        if name in BAR_INDICATORS:
            out = _BAR_FNS[name](tf_bars, a)
        else:
            # Evaluate `of` with tf_bars as the driving frame so the indicator
            # computes over native tf bars; only the final result is aligned
            # back to the driving index (mirrors the bar-indicator branch).
            of = node.get("of", {"src": "close"})
            series = eval_expr(of, ctx, tf_bars, memo)
            if isinstance(series, float):
                series = pd.Series(series, index=tf_bars.index)
            fn = _SERIES_FNS.get(name)
            out = fn(series, a["n"]) if fn else _FRAME_FNS[name](series, a)
        if isinstance(out, pd.DataFrame):
            out = out[a["field"]]
        return _align(out, bars)
    name = node["prim"]
    a = {**PRIM_DEFAULTS.get(name, {}),
         **{k: v for k, v in node.items() if k != "prim"}}
    if name == "bars_since":
        a["eval_fn"] = lambda cond, b: eval_condition_series(cond, ctx, b, memo)
    return PRIMITIVES[name]["fn"](ctx, bars, **{k: v for k, v in a.items() if k != "prim"})


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


def _last(v) -> float:
    return float(v.iloc[-1]) if isinstance(v, pd.Series) else float(v)


def eval_condition_series(cond: dict, ctx, bars, memo: dict) -> pd.Series:
    """Elementwise boolean series (comparison ops only; used by bars_since)."""
    lhs = eval_expr(cond["lhs"], ctx, bars, memo)
    rhs = eval_expr(cond["rhs"], ctx, bars, memo)
    if not isinstance(lhs, pd.Series):
        lhs = pd.Series(lhs, index=bars.index)
    op = cond["op"]
    out = {">": lhs > rhs, "<": lhs < rhs, ">=": lhs >= rhs, "<=": lhs <= rhs}[op]
    return out.fillna(False)


def eval_condition(cond: dict, ctx, bars, memo: dict) -> bool:
    lhs = eval_expr(cond["lhs"], ctx, bars, memo)
    rhs = eval_expr(cond["rhs"], ctx, bars, memo)
    op = cond["op"]
    if op == "crosses_above":
        if not isinstance(lhs, pd.Series):
            return False
        return ind.crossed_above(lhs, rhs)
    if op == "crosses_below":
        if not isinstance(lhs, pd.Series):
            return False
        return ind.crossed_below(lhs, rhs)
    a, b = _last(lhs), _last(rhs)
    if pd.isna(a) or pd.isna(b):
        return False
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[op]


def eval_group(group: dict, ctx, bars, memo: dict) -> bool:
    key, items = next(iter(group.items()))
    results = (eval_group(i, ctx, bars, memo) if ("all" in i or "any" in i)
               else eval_condition(i, ctx, bars, memo) for i in items)
    return all(results) if key == "all" else any(results)
