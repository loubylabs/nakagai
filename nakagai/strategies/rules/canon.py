"""Canonical form + content hash: the identity of a spec's LOGIC.

Display-only fields are stripped and every optional arg/default is
materialized, so two specs that trade identically hash identically.
"""

import hashlib
import json

from nakagai.strategies.rules.spec import DEFAULT_RISK, VERSION
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

_TRAILING_DEFAULTS = {"atr": {"n": 14, "mult": 2.0}, "percent": {"pct": 2.0}}


def _num(v):
    """Numeric scalars normalize to float so 20 and 20.0 hash identically;
    strings (field/direction/kind/tf) and nested objects pass through."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _canon_expr(node, vocabulary: Vocabulary):
    if isinstance(node, (int, float)):
        return float(node)
    if "src" in node:
        return {"src": node["src"], **({"tf": node["tf"]} if "tf" in node else {})}
    if "op" in node:
        return {"op": node["op"],
                "args": [_canon_expr(a, vocabulary) for a in node["args"]]}
    if "ind" in node:
        name = node["ind"]
        term = vocabulary.indicators[name]
        args = {**term.defaults,
                **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        out = {"ind": name, **{k: _num(v) for k, v in args.items()}}
        if term.kind != "bar":
            out["of"] = _canon_expr(node.get("of", {"src": "close"}), vocabulary)
        if "tf" in node:
            out["tf"] = node["tf"]
        return out
    name = node["prim"]
    args = {**vocabulary.primitives[name].defaults,
            **{k: v for k, v in node.items() if k not in ("prim", "cond")}}
    out = {"prim": name, **{k: _num(v) for k, v in args.items()}}
    if "cond" in node:
        out["cond"] = _canon_cond(node["cond"], vocabulary)
    return out


def _canon_cond(c, vocabulary: Vocabulary):
    return {"lhs": _canon_expr(c["lhs"], vocabulary), "op": c["op"],
            "rhs": _canon_expr(c["rhs"], vocabulary)}


def _canon_group(g, vocabulary: Vocabulary):
    key = next(iter(g))
    return {key: [_canon_group(i, vocabulary)
                  if isinstance(i, dict) and ("all" in i or "any" in i)
                  else _canon_cond(i, vocabulary) for i in g[key]]}


def _canon_exits(exits: dict, vocabulary: Vocabulary) -> dict:
    out = {}
    if "exit" in exits:
        out["exit"] = _canon_group(exits["exit"], vocabulary)
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


def canonical_spec(spec: dict, vocabulary: Vocabulary | None = None) -> dict:
    vocabulary = resolve_vocabulary(vocabulary)
    out = {"version": VERSION, "timeframe": spec.get("timeframe", "1h")}
    for side in ("long", "short"):
        if side in spec:
            out[side] = _canon_group(spec[side], vocabulary)
    if "exits" in spec:
        out["exits"] = _canon_exits(spec["exits"], vocabulary)
    risk = spec.get("risk", {})
    out["risk"] = {"stop": {**DEFAULT_RISK["stop"], **risk.get("stop", {})},
                   "target": {**DEFAULT_RISK["target"], **risk.get("target", {})}}
    return out


def spec_hash(spec: dict, vocabulary: Vocabulary | None = None) -> str:
    blob = json.dumps(canonical_spec(spec, vocabulary), sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
