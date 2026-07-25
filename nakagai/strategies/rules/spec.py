"""RuleSpec v2: the declarative strategy IR.

A spec is plain JSON, version 2 only. Operands are expression trees over
series sources, an indicator registry, math ops, and stateful primitives.
Entries are nested all/any condition groups per side; exits and risk are
first-class blocks. validate_spec reports precise per-path errors (the NL
compiler's retry loop feeds on them); describe_spec renders the trust-step
readback; canon.py owns identity hashing.
"""

from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.primitives import PRIMITIVES
from nakagai.strategies.rules.primitives import SESSION_SCOPED_PRIMS

VERSION = 2
SOURCES = ("open", "high", "low", "close", "volume")
TIMEFRAMES = ("15m", "1h", "1d")
OPS = (">", "<", ">=", "<=", "crosses_above", "crosses_below")
CROSS_OPS = ("crosses_above", "crosses_below")
MATH_OPS: dict[str, tuple[int, int]] = {   # op -> (min arity, max arity)
    "+": (2, 8), "-": (2, 2), "*": (2, 8), "/": (2, 2),
    "abs": (1, 1), "min": (2, 8), "max": (2, 8),
}
# Series indicators take `of` (any expression, default close-of-timeframe).
SERIES_INDICATORS: dict[str, dict] = {
    "sma": {"n": (2, 500)}, "ema": {"n": (2, 500)}, "rsi": {"n": (2, 100)},
    "roc": {"n": (1, 500)}, "zscore": {"n": (2, 500)},
    "highest": {"n": (2, 500)}, "lowest": {"n": (2, 500)}, "stdev": {"n": (2, 500)},
    "macd": {"fast": (2, 100), "slow": (3, 200), "signal": (2, 100),
             "field": ("macd", "signal", "hist")},
    "bb": {"n": (2, 200), "k": (0.5, 5.0), "field": ("upper", "mid", "lower")},
}
# Bar indicators need full OHLCV; no `of`.
BAR_INDICATORS: dict[str, dict] = {
    "atr": {"n": (2, 100)},
    "donchian": {"n": (2, 300), "field": ("upper", "lower", "mid")},
    "supertrend": {"n": (2, 100), "mult": (0.5, 10.0), "field": ("line", "direction")},
    "vwap": {},
    "stoch": {"n": (2, 100), "d": (1, 50), "field": ("k", "d")},
    "adx": {"n": (2, 100)},
    "obv": {},
    "ichimoku": {"tenkan_n": (2, 100), "kijun_n": (2, 200), "senkou_n": (2, 300),
                 "disp": (1, 100), "field": ("tenkan", "kijun", "senkou_a", "senkou_b")},
    "keltner": {"n": (2, 200), "mult": (0.5, 10.0), "field": ("upper", "mid", "lower")},
    "cci": {"n": (2, 200)},
    "mfi": {"n": (2, 100)},
    "wpr": {"n": (2, 100)},
}
INDICATORS = {**SERIES_INDICATORS, **BAR_INDICATORS}
ARG_DEFAULTS: dict[str, dict] = {
    "sma": {"n": 20}, "ema": {"n": 20}, "rsi": {"n": 14}, "roc": {"n": 20},
    "zscore": {"n": 20}, "highest": {"n": 20}, "lowest": {"n": 20}, "stdev": {"n": 20},
    "macd": {"fast": 12, "slow": 26, "signal": 9, "field": "macd"},
    "bb": {"n": 20, "k": 2.0, "field": "mid"},
    "atr": {"n": 14}, "donchian": {"n": 20, "field": "upper"},
    "supertrend": {"n": 10, "mult": 3.0, "field": "line"},
    "stoch": {"n": 14, "d": 3, "field": "k"}, "adx": {"n": 14},
    "ichimoku": {"tenkan_n": 9, "kijun_n": 26, "senkou_n": 52, "disp": 26,
                 "field": "tenkan"},
    "keltner": {"n": 20, "mult": 2.0, "field": "mid"},
    "cci": {"n": 20}, "mfi": {"n": 14}, "wpr": {"n": 14},
}
STOP_KINDS = ("atr", "percent")
TARGET_KINDS = ("rr", "percent")
MAX_DEPTH = 8
MAX_CONDITIONS = 30
MAX_NODES = 40           # indicator + primitive nodes per spec
TIME_STOP_BOUNDS = (1, 500)
BREAKEVEN_RR_BOUNDS = (0.1, 10.0)

DEFAULT_RISK = {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                "target": {"kind": "rr", "rr": 2.0}}


class _Budget:
    def __init__(self):
        self.conditions = 0
        self.nodes = 0


def _check_args(name: str, given: dict, schema: dict, path: str, errs: list[str],
                skip: tuple[str, ...] = ()) -> None:
    for arg, v in given.items():
        if arg in skip:
            continue
        if arg not in schema:
            errs.append(f"{path}: {name} takes no arg {arg!r} (valid: {sorted(schema)})")
            continue
        rule = schema[arg]
        if isinstance(rule, tuple) and all(isinstance(r, str) for r in rule):
            if v not in rule:
                errs.append(f"{path}: {name}.{arg} must be one of {rule}, got {v!r}")
        else:
            lo, hi = rule
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                errs.append(f"{path}: {name}.{arg} must be a number in [{lo}, {hi}], got {v!r}")


def _check_tf(node: dict, path: str, errs: list[str],
              allowed_extra: tuple[str, ...] = ()) -> None:
    """Rejects a `tf` not in TIMEFRAMES, and (for src leaves) any extra key
    beyond src/tf."""
    if "tf" in node and node["tf"] not in TIMEFRAMES:
        errs.append(f"{path}: tf must be one of {TIMEFRAMES}, got {node['tf']!r}")
    if allowed_extra:
        unknown = set(node) - set(allowed_extra) - {"tf"}
        if unknown:
            errs.append(f"{path}: unknown keys {sorted(unknown)}")


def _check_expr(node, path: str, errs: list[str], budget: _Budget,
                depth: int = 0, series_required: bool = False) -> None:
    if depth > MAX_DEPTH:
        errs.append(f"{path}: expression depth exceeds {MAX_DEPTH}")
        return
    if isinstance(node, bool):
        errs.append(f"{path}: booleans are not operands")
        return
    if isinstance(node, (int, float)):
        if series_required:
            errs.append(f"{path}: the left side of a cross must be a series, not a number")
        return
    if not isinstance(node, dict):
        errs.append(f"{path}: operand must be a number or an expression object")
        return
    if "src" in node:
        if node["src"] not in SOURCES:
            errs.append(f"{path}: unknown source {node['src']!r} (valid: {SOURCES})")
        _check_tf(node, path, errs, allowed_extra=("src",))
        return
    if "op" in node:
        op = node["op"]
        if op not in MATH_OPS:
            errs.append(f"{path}: unknown math op {op!r} (valid: {sorted(MATH_OPS)})")
            return
        args = node.get("args")
        lo, hi = MATH_OPS[op]
        if not isinstance(args, list) or not lo <= len(args) <= hi:
            errs.append(f"{path}: {op!r} takes {lo}-{hi} args")
            return
        for i, a in enumerate(args):
            _check_expr(a, f"{path}.args[{i}]", errs, budget, depth + 1)
        if set(node) - {"op", "args"}:
            errs.append(f"{path}: math nodes take only op/args")
        return
    if "ind" in node:
        budget.nodes += 1
        name = node["ind"]
        if name not in INDICATORS:
            errs.append(f"{path}: unknown indicator {name!r} (valid: {sorted(INDICATORS)})")
            return
        if "of" in node:
            if name in BAR_INDICATORS:
                errs.append(f"{path}: {name} works on full bars and takes no `of`")
            else:
                _check_expr(node["of"], f"{path}.of", errs, budget, depth + 1)
        _check_args(name, node, INDICATORS[name], path, errs, skip=("ind", "of", "tf"))
        _check_tf(node, path, errs)
        return
    if "prim" in node:
        budget.nodes += 1
        name = node["prim"]
        if name not in PRIMITIVES:
            errs.append(f"{path}: unknown primitive {name!r} (valid: {sorted(PRIMITIVES)})")
            return
        schema = PRIMITIVES[name]["args"]
        if name == "bars_since":
            cond = node.get("cond")
            if not isinstance(cond, dict) or not {"lhs", "op", "rhs"} <= set(cond):
                errs.append(f"{path}: bars_since needs cond = {{lhs, op, rhs}}")
            elif cond.get("op") in CROSS_OPS:
                errs.append(f"{path}: bars_since conditions use comparison ops only")
            else:
                _check_condition(cond, f"{path}.cond", errs, budget, depth + 1)
            _check_args(name, node, {}, path, errs, skip=("prim", "cond", "tf"))
        else:
            _check_args(name, node, schema, path, errs, skip=("prim", "tf"))
        if "tf" in node and name in SESSION_SCOPED_PRIMS:
            errs.append(f"{path}: {name} is session-scoped and takes no tf")
        _check_tf(node, path, errs)
        return
    errs.append(f"{path}: expression object needs one of src/ind/op/prim")


def _check_condition(cond, path: str, errs: list[str], budget: _Budget,
                     depth: int = 0) -> None:
    budget.conditions += 1
    if budget.conditions > MAX_CONDITIONS:
        errs.append(f"{path}: more than {MAX_CONDITIONS} conditions")
        return
    if not isinstance(cond, dict) or not {"lhs", "op", "rhs"} <= set(cond):
        errs.append(f"{path}: condition needs lhs, op, rhs")
        return
    op = cond["op"]
    if op not in OPS:
        errs.append(f"{path}: unknown op {op!r} (valid: {OPS})")
    _check_expr(cond["lhs"], f"{path}.lhs", errs, budget, depth,
                series_required=op in CROSS_OPS)
    _check_expr(cond["rhs"], f"{path}.rhs", errs, budget, depth)
    if budget.nodes > MAX_NODES:
        errs.append(f"{path}: more than {MAX_NODES} indicator/primitive nodes")


def _check_group(group, path: str, errs: list[str], budget: _Budget,
                 depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        errs.append(f"{path}: group depth exceeds {MAX_DEPTH}")
        return
    if not isinstance(group, dict) or len(group) != 1 or next(iter(group)) not in ("all", "any"):
        errs.append(f"{path}: expected {{\"all\": [...]}} or {{\"any\": [...]}}")
        return
    key, items = next(iter(group.items()))
    if not isinstance(items, list) or not items:
        errs.append(f"{path}.{key}: must be a non-empty list")
        return
    for i, item in enumerate(items):
        p = f"{path}.{key}[{i}]"
        if isinstance(item, dict) and ("all" in item or "any" in item):
            _check_group(item, p, errs, budget, depth + 1)
            continue
        _check_condition(item, p, errs, budget)


def _not_num(v, lo, hi) -> bool:
    """True when v is not a plain int/float in [lo, hi]; bools are excluded.
    This never raises, so it is safe to call on any user-supplied value
    (a string, a list, None) before it ever reaches float()/int()."""
    return isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi


def _check_exits(exits, errs: list[str], budget: _Budget) -> None:
    if not isinstance(exits, dict):
        errs.append("exits must be an object")
        return
    unknown = set(exits) - {"exit", "trailing", "time_stop", "breakeven_at"}
    if unknown:
        errs.append(f"exits: unknown keys {sorted(unknown)}")
    if "exit" in exits:
        _check_group(exits["exit"], "exits.exit", errs, budget)
    if "trailing" in exits:
        t = exits["trailing"]
        if not isinstance(t, dict) or t.get("kind") not in ("atr", "percent"):
            errs.append("exits.trailing.kind must be 'atr' or 'percent'")
        elif t["kind"] == "atr":
            if _not_num(t.get("mult", 2.0), 0.5, 10):
                errs.append("exits.trailing.mult must be in [0.5, 10]")
            if _not_num(t.get("n", 14), 2, 100):
                errs.append("exits.trailing.n must be in [2, 100]")
            unknown_t = set(t) - {"kind", "mult", "n"}
            if unknown_t:
                errs.append(f"exits.trailing: unknown keys {sorted(unknown_t)}")
        else:
            if _not_num(t.get("pct", 2.0), 0.1, 50):
                errs.append("exits.trailing.pct must be in [0.1, 50]")
            unknown_t = set(t) - {"kind", "pct"}
            if unknown_t:
                errs.append(f"exits.trailing: unknown keys {sorted(unknown_t)}")
    if "time_stop" in exits:
        ts = exits["time_stop"]
        b = ts.get("bars") if isinstance(ts, dict) else None
        lo, hi = TIME_STOP_BOUNDS
        if isinstance(b, bool) or not isinstance(b, int) or not lo <= b <= hi:
            errs.append(f"exits.time_stop.bars must be an integer in [{lo}, {hi}]")
    if "breakeven_at" in exits:
        ba = exits["breakeven_at"]
        r = ba.get("rr") if isinstance(ba, dict) else None
        lo, hi = BREAKEVEN_RR_BOUNDS
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not lo <= r <= hi:
            errs.append(f"exits.breakeven_at.rr must be a number in [{lo}, {hi}]")


def validate_risk(risk) -> list[str]:
    """Risk-block validation shared by rule specs and composite specs."""
    errs: list[str] = []
    if not isinstance(risk, dict):
        return ["risk must be an object"]
    stop = risk.get("stop", DEFAULT_RISK["stop"])
    if not isinstance(stop, dict):
        errs.append("risk.stop must be an object")
    elif stop.get("kind") not in STOP_KINDS:
        errs.append(f"risk.stop.kind must be one of {STOP_KINDS}")
    elif stop["kind"] == "atr":
        if _not_num(stop.get("mult", 2.0), 0.1, 10):
            errs.append("risk.stop.mult must be in [0.1, 10]")
        if _not_num(stop.get("n", 14), 2, 100):
            errs.append("risk.stop.n must be in [2, 100]")
    elif _not_num(stop.get("pct", 2.0), 0.05, 50):
        errs.append("risk.stop.pct must be in [0.05, 50]")
    target = risk.get("target", DEFAULT_RISK["target"])
    if not isinstance(target, dict):
        errs.append("risk.target must be an object")
    elif target.get("kind") not in TARGET_KINDS:
        errs.append(f"risk.target.kind must be one of {TARGET_KINDS}")
    elif target["kind"] == "rr":
        if _not_num(target.get("rr", 2.0), 0.1, 20):
            errs.append("risk.target.rr must be in [0.1, 20]")
    elif _not_num(target.get("pct", 4.0), 0.05, 100):
        errs.append("risk.target.pct must be in [0.05, 100]")
    return errs


def risk_text(risk: dict) -> str:
    """The 'Stop: … Target: …' sentence shared by both describe functions."""
    stop = risk.get("stop", DEFAULT_RISK["stop"])
    target = risk.get("target", DEFAULT_RISK["target"])
    stop_text = (f"{stop.get('mult', 2.0):g}x ATR({int(stop.get('n', 14))}) from entry"
                 if stop.get("kind") == "atr" else f"{stop.get('pct', 2.0):g}% from entry")
    target_text = (f"{target.get('rr', 2.0):g}x the risked distance"
                   if target.get("kind") == "rr" else f"{target.get('pct', 4.0):g}% from entry")
    return f"Stop: {stop_text}. Target: {target_text}."


def validate_spec(spec) -> list[str]:
    """Structural + semantic validation. Empty list = usable spec."""
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    errs: list[str] = []
    if spec.get("version") != VERSION:
        errs.append(f"spec version must be {VERSION} (got {spec.get('version')!r})")
    if not str(spec.get("name", "")).strip():
        errs.append("spec needs a name")
    if spec.get("timeframe", "1h") not in TIMEFRAMES:
        errs.append(f"timeframe must be one of {TIMEFRAMES}")
    sides = [s for s in ("long", "short") if s in spec]
    if not sides:
        errs.append("spec needs at least one of long/short entry groups")
    budget = _Budget()
    for side in sides:
        _check_group(spec[side], side, errs, budget)
    if "exits" in spec:
        _check_exits(spec["exits"], errs, budget)
    errs.extend(validate_risk(spec.get("risk", {})))
    return errs


def validate_condition_group(group, path: str = "conditions") -> list[str]:
    """Standalone validation of one all/any condition group with a fresh
    budget. The screener's whole schema is one such group; validate_spec's
    per-side entry groups go through the same walker."""
    errs: list[str] = []
    _check_group(group, path, errs, _Budget())
    return errs


def group_text(group: dict) -> str:
    """Public name for the group renderer; the screener readback reuses it."""
    return _group_text(group)


def _expr_text(node) -> str:
    if isinstance(node, (int, float)):
        return f"{node:g}"
    if "src" in node:
        return node["src"] if "tf" not in node else f"{node['src']}[{node['tf']}]"
    if "op" in node:
        op, args = node["op"], node["args"]
        if op == "abs":
            return f"abs({_expr_text(args[0])})"
        return "(" + f" {op} ".join(_expr_text(a) for a in args) + ")"
    if "ind" in node:
        name = node["ind"]
        args = {**ARG_DEFAULTS.get(name, {}),
                **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        field = args.pop("field", None)
        parts = [f"{v}" for v in args.values()]
        if name in SERIES_INDICATORS:
            of = node.get("of", {"src": "close"})
            if of != {"src": "close"}:
                parts.append(f"of={_expr_text(of)}")
        inner = ", ".join(parts)
        text = f"{name}({inner})" if inner else name
        if "tf" in node:
            text += f"[{node['tf']}]"
        return f"{text}.{field}" if field else text
    name = node["prim"]
    if name == "bars_since":
        text = f"bars_since({_condition_text(node['cond'])})"
    else:
        args = {**PRIM_DEFAULTS.get(name, {}),
                **{k: v for k, v in node.items() if k not in ("prim", "cond", "tf")}}
        if "minutes" in args:
            minutes = args.pop("minutes")
            inner = ", ".join([f"{minutes}m"] + [f"{v}" for v in args.values()])
        else:
            inner = ", ".join(f"{v}" for v in args.values())
        text = f"{name}({inner})" if inner else name
    if "tf" in node:
        text += f"[{node['tf']}]"
    return text


_OP_TEXT = {">": "is above", "<": "is below", ">=": "is at or above",
            "<=": "is at or below", "crosses_above": "crosses above",
            "crosses_below": "crosses below"}


def _condition_text(cond: dict) -> str:
    return f"{_expr_text(cond['lhs'])} {_OP_TEXT[cond['op']]} {_expr_text(cond['rhs'])}"


def _group_text(group, indent: str = "  ") -> str:
    key, items = next(iter(group.items()))
    joiner = "ALL of:" if key == "all" else "ANY of:"
    lines = [joiner]
    for item in items:
        if "all" in item or "any" in item:
            lines.append(indent + _group_text(item, indent + "  ").replace("\n", "\n" + indent))
        else:
            lines.append(f"{indent}- {_condition_text(item)}")
    return "\n".join(lines)


def _exits_text(exits: dict) -> list[str]:
    lines = []
    if "exit" in exits:
        lines.append("Exit early when " + _group_text(exits["exit"]))
    if "trailing" in exits:
        t = exits["trailing"]
        if t["kind"] == "atr":
            lines.append(f"Trailing stop: {t.get('mult', 2.0):g}x ATR({int(t.get('n', 14))}).")
        else:
            lines.append(f"Trailing stop: {t.get('pct', 2.0):g}% from the high water mark.")
    if "time_stop" in exits:
        lines.append(f"Time stop: {exits['time_stop']['bars']} 15-minute bars.")
    if "breakeven_at" in exits:
        lines.append(f"Move stop to breakeven at {exits['breakeven_at']['rr']:g}R.")
    return lines


def describe_spec(spec: dict) -> str:
    """Plain-English restatement of a validated spec: the trust step shown to
    the user before they save or backtest an imported/NL-built strategy."""
    lines = [f"Strategy \"{spec.get('name', 'unnamed')}\" on {spec.get('timeframe', '1h')} bars."]
    for side in ("long", "short"):
        if side in spec:
            lines.append(f"Enter {side} when " + _group_text(spec[side]))
    if "exits" in spec:
        lines.extend(_exits_text(spec["exits"]))
    risk = spec.get("risk", DEFAULT_RISK)
    lines.append(risk_text(risk))
    return "\n".join(lines)
