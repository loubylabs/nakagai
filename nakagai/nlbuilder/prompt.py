"""System prompt for the NL->spec compiler, rendered from the live grammar
registries so it can never drift from the validator."""

import json

from nakagai.strategies.composite.spec import (
    DEFAULT_WINDOW_BARS, MAX_BLOCKS, WINDOW_BARS_BOUNDS)
from nakagai.strategies.rules import primitives as prims
from nakagai.strategies.rules import spec as g

_EXAMPLES = """\
Description: "buy the dip when rsi 14 recovers above 30"
{"spec": {"version": 2, "name": "rsi-dip-buy", "timeframe": "1h",
"long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30}]},
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}},
"clarifications": ["defaulted timeframe to 1h", "used the default ATR stop and 2R target since none were given"]}

Description: "opening range breakout, 30 minute range, confirmed by volume 50% above its 20 bar average, only when the daily close is above the 50 day sma"
{"spec": {"version": 2, "name": "orb-volume-confirm", "timeframe": "15m",
"long": {"all": [
{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": {"prim": "opening_range_high", "minutes": 30}},
{"lhs": {"src": "volume"}, "op": ">", "rhs": {"op": "*", "args": [1.5, {"ind": "sma", "n": 20, "of": {"src": "volume"}}]}},
{"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": {"ind": "sma", "n": 50}}
]},
"exits": {"time_stop": {"bars": 16}, "trailing": {"kind": "atr", "n": 14, "mult": 2.5}},
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}},
"clarifications": ["used a 16 bar time stop and a 2.5x ATR trail since neither was specified"]}

Description: "trade based on my broker's news sentiment feed"
{"not_expressible": "the grammar has no source for external news sentiment data, only price/volume series, indicators, and primitives"}\
"""

_COMPOSITE_EXAMPLE = """\

Description: "combine the donchian breakout with the adx pullback, both have to fire, plus my own leg that only takes it when rsi 14 is under 40"
{"kind": "composite", "spec": {"version": 1, "name": "donch-adx-rsi-confluence",
"blocks": {"a": {"strategy": "donchian_breakout"}, "b": {"strategy": "adx_pullback"},
"c": {"strategy": "rules", "params": {"spec": {"version": 2, "name": "rsi-under-40", "timeframe": "1d",
"long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 40}]},
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}}}}},
"long": {"all": ["a", "b", "c"]}, "window_bars": 4,
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}},
"clarifications": ["used a 4 bar vote window and the default ATR stop and 2R target"]}\
"""


def _bounds(schema: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in schema.items()) or "no args"


def _composite_section(members: dict) -> str:
    plays = "\n".join(
        f"- {name} [{(cls.DEFAULT_PARAMS.get('spec') or {}).get('timeframe', '?')}]"
        f" {cls.title or name}: {(cls.description or '').strip()}"
        for name, cls in sorted(members.items()) if name != "rules")
    return f"""

# Composites (one strategy built from several)
When the user asks to COMBINE, confirm, or gate several strategies, return
{{"kind": "composite", "spec": {{...}}, "clarifications": [...]}} instead of a
rules spec. A composite spec:
{{"version": 1, "name": str, "blocks": {{<id>: <block>, ...}},
"long"/"short": <vote tree>, "window_bars": {WINDOW_BARS_BOUNDS[0]}-{WINDOW_BARS_BOUNDS[1]}
(default {DEFAULT_WINDOW_BARS}), "risk": {{...}}}}
A block is either a catalog play referenced by name, carrying NO params:
{{"strategy": "<catalog name>"}}
or a bespoke or tuned leg written with the RuleSpec grammar above:
{{"strategy": "rules", "params": {{"spec": {{<RuleSpec v2>}}}}}}
A vote tree is {{"all": [...]}} or {{"any": [...]}} over block ids, nestable.
At most {MAX_BLOCKS} blocks; a composite cannot contain another composite.
The composite owns its own stop and target, so member risk is ignored; a rules
leg still needs a valid "risk" block to validate, it is simply unused.
Prefer legs that share one timeframe.

# Catalog plays (usable as bare blocks; they take no param overrides)
{plays}"""


def render_system_prompt(members: dict | None = None) -> str:
    ind_lines = "\n".join(
        f"- {name}({_bounds(sch)})" + (" [takes of=<expr>]" if name in g.SERIES_INDICATORS else "")
        for name, sch in sorted(g.INDICATORS.items()))
    prim_lines = "\n".join(f"- {name}({_bounds(p['args'])})"
                           for name, p in sorted(prims.PRIMITIVES.items()))
    composite = _composite_section(members) if members else ""
    example = _COMPOSITE_EXAMPLE if members else ""
    return f"""You compile plain-English trading strategy descriptions into
nakagai strategy JSON. Reply with EXACTLY ONE JSON object and nothing else
(no prose, no code fences): either
{{"kind": "rules", "spec": {{...}}, "clarifications": [...]}}
or {{"not_expressible": "<one-sentence reason>"}}. Always include "kind".

# Grammar (version 2, required)
A spec: {{"version": 2, "name": str, "timeframe": one of {g.TIMEFRAMES},
"long"/"short": condition groups, "exits"?: {{...}}, "risk": {{...}}}}.
Conditions: {{"lhs": <expr>, "op": one of {g.OPS}, "rhs": <expr>}} inside
nested {{"all": [...]}} / {{"any": [...]}} groups. The lhs of a cross must be
a series expression, never a number.

Expressions are numbers or objects:
- series leaf: {{"src": one of {g.SOURCES}, "tf"?: one of {g.TIMEFRAMES}}}
- indicator: {{"ind": <name>, <args>, "of"?: <expr>, "tf"?: <tf>}}
- math: {{"op": one of {sorted(g.MATH_OPS)}, "args": [<expr>, ...]}}
- primitive: {{"prim": <name>, <args>, "tf"?: <tf>}}

# Indicators (name(arg=bounds or choices))
{ind_lines}

# Primitives (session/state aware; bars_since takes cond={{lhs,op,rhs}} with comparison ops only)
{prim_lines}

# Exits (all optional)
{{"exit": <condition group>, "trailing": {{"kind": "atr"|"percent", "n"?, "mult"? | "pct"?}},
"time_stop": {{"bars": {g.TIME_STOP_BOUNDS[0]}-{g.TIME_STOP_BOUNDS[1]}}} (bars are always
15-minute bars, regardless of the spec's own timeframe),
"breakeven_at": {{"rr": {g.BREAKEVEN_RR_BOUNDS[0]}-{g.BREAKEVEN_RR_BOUNDS[1]}}}}}

# Risk (required)
{{"stop": {{"kind": "atr", "n": 2-100, "mult": 0.1-10}} or {{"kind": "percent", "pct": 0.05-50}},
"target": {{"kind": "rr", "rr": 0.1-20}} or {{"kind": "percent", "pct": 0.05-100}}}}
Default when unspecified: {json.dumps(g.DEFAULT_RISK)}.

# Limits
Max expression depth {g.MAX_DEPTH}; max {g.MAX_CONDITIONS} conditions; max
{g.MAX_NODES} indicator+primitive nodes.

# Rules of engagement
- If the user leaves something unspecified, choose a sensible default and add
  a short entry to "clarifications" saying what you assumed.
- If the request needs anything the grammar cannot express, return
  "not_expressible" with the reason instead of guessing a wrong spec.
- Keep "name" short and lowercase-hyphenated.
- highest/lowest are unshifted rolling extrema that include the current bar,
  so a breakout cross like "close crosses_above highest(close, n)" can never
  fire; use donchian(n).upper/.lower for breakout crosses instead.
{composite}
# Examples
{_EXAMPLES}{example}"""
