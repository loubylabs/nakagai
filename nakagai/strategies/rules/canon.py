"""Canonical form + content hash: the identity of a spec's LOGIC.

Display-only fields are stripped and every optional arg/default is
materialized, so two specs that trade identically hash identically.
"""

import hashlib
import json

from nakagai.strategies.rules.spec import DEFAULT_RISK, VERSION, is_group_node
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_condition_rule, resolve_vocabulary,
)

_TRAILING_DEFAULTS = {"atr": {"n": 14, "mult": 2.0}, "percent": {"pct": 2.0}}


def _num(v):
    """Numeric scalars normalize to float so 20 and 20.0 hash identically;
    strings (field/direction/kind/tf) and nested objects pass through."""
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def canonical_expr(node, vocabulary: Vocabulary):
    """One expression node in canonical form: defaults materialized, numeric
    scalars normalized to float. Public because the Pine compiler keys its
    node memo and its shared inputs on exactly this form, and a second
    canonicalizer beside it would be a second definition of "the same node"."""
    if isinstance(node, (int, float)):
        return float(node)
    if "src" in node:
        return {"src": node["src"], **({"tf": node["tf"]} if "tf" in node else {})}
    if "op" in node:
        return {"op": node["op"],
                "args": [canonical_expr(a, vocabulary) for a in node["args"]]}
    if "ind" in node:
        name = node["ind"]
        term = vocabulary.indicators[name]
        args = {**term.defaults,
                **{k: v for k, v in node.items() if k not in ("ind", "of", "tf")}}
        out = {"ind": name, **{k: _num(v) for k, v in args.items()}}
        if term.kind != "bar":
            out["of"] = canonical_expr(node.get("of", {"src": "close"}), vocabulary)
        if "tf" in node:
            out["tf"] = node["tf"]
        return out
    name = node["prim"]
    term = vocabulary.primitives[name]
    # Re-keyed onto the term's condition-typed arg NAMES, generic over which
    # primitive and which key it happens to declare, rather than the literal
    # key "cond". Keying on the literal key already worked for a second
    # primitive that also happened to call its arg "cond" and silently did the
    # wrong thing for one that called it something else: the condition fell
    # into the generic args merge, where _num passes a dict straight through,
    # so nothing inside it was canonicalized and two specs whose conditions
    # differ only in a materialized default or an int-versus-float literal
    # hashed apart.
    condition_args = {a for a, rule in term.args.items() if is_condition_rule(rule)}
    args = {**term.defaults,
            **{k: v for k, v in node.items() if k not in ("prim", *condition_args)}}
    out = {"prim": name, **{k: _num(v) for k, v in args.items()}}
    for a in sorted(condition_args):
        if a in node:
            out[a] = _canon_cond(node[a], vocabulary)
    return out


def _canon_cond(c, vocabulary: Vocabulary):
    return {"lhs": canonical_expr(c["lhs"], vocabulary), "op": c["op"],
            "rhs": canonical_expr(c["rhs"], vocabulary)}


def _canon_group(g, vocabulary: Vocabulary):
    key = next(iter(g))
    if key == "not":
        # `not`'s value is a single nested group, not a list of items, so it
        # is canonicalized on its own rather than through the comprehension
        # below: `for i in g[key]` over a dict would silently iterate its KEY
        # STRINGS instead of raising, corrupting the hash rather than
        # crashing. Structural, per N3-D7: {"not": {"not": G}} canonicalizes
        # as a double negation and is not simplified away, because
        # canonicalization is a structural transform and simplifying logic
        # would change what a spec_hash identifies.
        return {key: _canon_group(g[key], vocabulary)}
    return {key: [_canon_group(i, vocabulary) if is_group_node(i)
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
