"""Canonical form + content hash: the identity of a spec's LOGIC.

Display-only fields are stripped and every optional arg/default is
materialized, so two specs that trade identically hash identically.
"""

import hashlib
import json

from nakagai.strategies.rules.primitives import ARG_DEFAULTS as PRIM_DEFAULTS
from nakagai.strategies.rules.spec import (
    ARG_DEFAULTS, DEFAULT_RISK, SERIES_INDICATORS, VERSION,
)

_TRAILING_DEFAULTS = {"atr": {"n": 14, "mult": 2.0}, "percent": {"pct": 2.0}}


def _num(v):
    """Numeric scalars normalize to float so 20 and 20.0 hash identically;
    strings (field/direction/kind/tf) and nested objects pass through."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _canon_expr(node):
    if isinstance(node, (int, float)):
        return float(node)
    if "src" in node:
        return {"src": node["src"], **({"tf": node["tf"]} if "tf" in node else {})}
    if "op" in node:
        return {"op": node["op"], "args": [_canon_expr(a) for a in node["args"]]}
    if "ind" in node:
        name = node["ind"]
        args = {**ARG_DEFAULTS.get(name, {}),
                **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        out = {"ind": name, **{k: _num(v) for k, v in args.items()}}
        if name in SERIES_INDICATORS:
            out["of"] = _canon_expr(node.get("of", {"src": "close"}))
        if "tf" in node:
            out["tf"] = node["tf"]
        return out
    name = node["prim"]
    args = {**PRIM_DEFAULTS.get(name, {}),
            **{k: v for k, v in node.items() if k not in ("prim", "cond")}}
    out = {"prim": name, **{k: _num(v) for k, v in args.items()}}
    if "cond" in node:
        out["cond"] = _canon_cond(node["cond"])
    return out


def _canon_cond(c):
    return {"lhs": _canon_expr(c["lhs"]), "op": c["op"], "rhs": _canon_expr(c["rhs"])}


def _canon_group(g):
    key = next(iter(g))
    return {key: [_canon_group(i) if isinstance(i, dict) and ("all" in i or "any" in i)
                  else _canon_cond(i) for i in g[key]]}


def _canon_exits(exits: dict) -> dict:
    out = {}
    if "exit" in exits:
        out["exit"] = _canon_group(exits["exit"])
    if "trailing" in exits:
        t = exits["trailing"]
        merged = {"kind": t["kind"], **_TRAILING_DEFAULTS[t["kind"]],
                  **{k: v for k, v in t.items() if k != "kind"}}
        out["trailing"] = {k: _num(v) for k, v in merged.items()}
    if "time_stop" in exits:
        out["time_stop"] = {"bars": int(exits["time_stop"]["bars"])}
    if "breakeven_at" in exits:
        out["breakeven_at"] = {"rr": float(exits["breakeven_at"]["rr"])}
    return out


def canonical_spec(spec: dict) -> dict:
    out = {"version": VERSION, "timeframe": spec.get("timeframe", "1h")}
    for side in ("long", "short"):
        if side in spec:
            out[side] = _canon_group(spec[side])
    if "exits" in spec:
        out["exits"] = _canon_exits(spec["exits"])
    risk = spec.get("risk", {})
    out["risk"] = {"stop": {**DEFAULT_RISK["stop"], **risk.get("stop", {})},
                   "target": {**DEFAULT_RISK["target"], **risk.get("target", {})}}
    return out


def spec_hash(spec: dict) -> str:
    blob = json.dumps(canonical_spec(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
