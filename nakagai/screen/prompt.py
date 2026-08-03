"""System prompt for the NL->ScreenSpec compiler, rendered from the caller's
own vocabulary so it can never drift from the validator that will judge the
reply."""

from nakagai.strategies.rules import spec as g
from nakagai.strategies.rules.vocabulary import Vocabulary, resolve_vocabulary

_EXAMPLES = """\
Description: "oversold names: rsi 14 under 30 on the daily"
{"spec": {"version": 1, "tf": "1d",
"conditions": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}]}}}

Description: "price above the 200 day average on at least double average volume"
{"spec": {"version": 1, "tf": "1d",
"conditions": {"all": [
{"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 200}},
{"lhs": {"src": "volume"}, "op": ">", "rhs": {"op": "*", "args": [2, {"ind": "sma", "n": 20, "of": {"src": "volume"}}]}}
]}},
"clarifications": ["read 'average volume' as its 20 bar simple average"]}

Description: "stocks with strong earnings surprises"
{"not_expressible": "the grammar has no fundamentals or earnings data, only price/volume series, indicators, and primitives"}\
"""


def _bounds(schema: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in schema.items()) or "no args"


def render_screen_prompt(vocabulary: Vocabulary | None = None) -> str:
    vocabulary = resolve_vocabulary(vocabulary)
    ind_lines = "\n".join(
        f"- {name}({_bounds(term.args)})"
        + (" [takes of=<expr>]" if term.kind != "bar" else "")
        for name, term in sorted(vocabulary.indicators.items()))
    prim_lines = "\n".join(f"- {name}({_bounds(term.args)})"
                           for name, term in sorted(vocabulary.primitives.items()))
    return f"""You compile plain-English market screens into nakagai ScreenSpec v1
JSON. A screen is a filter: it answers "which symbols match this condition
right now." Reply with EXACTLY ONE JSON object and nothing else (no prose, no
code fences): either {{"spec": {{...}}, "clarifications": [...]}} or
{{"not_expressible": "<one-sentence reason>"}}.

# Schema (version 1, required)
A spec: {{"version": 1, "tf": one of {g.TIMEFRAMES} (the base bars the verdict
is read on; default "1d"), "conditions": one condition group}}. Nothing else:
no name, no entries or sides, no exits, no stops or targets. If the user asks
for those they are describing a strategy, not a screen; still compile the
FILTER part and add a clarification that entries/exits belong to the strategy
builder.
Conditions: {{"lhs": <expr>, "op": one of {g.OPS}, "rhs": <expr>}} inside
nested {{"all": [...]}} / {{"any": [...]}} groups. The lhs of a cross must be
a series expression, never a number. Cross ops fire on the latest completed
bar transition.

Expressions are numbers or objects:
- series leaf: {{"src": one of {g.SOURCES}, "tf"?: one of {g.TIMEFRAMES}}}
- indicator: {{"ind": <name>, <args>, "of"?: <expr>, "tf"?: <tf>}}
- math: {{"op": one of {sorted(g.MATH_OPS)}, "args": [<expr>, ...]}}
- primitive: {{"prim": <name>, <args>}}

# Indicators (name(arg=bounds or choices))
{ind_lines}

# Primitives (session/state aware; bars_since takes cond={{lhs,op,rhs}} with comparison ops only)
{prim_lines}

# Limits
Max expression depth {g.MAX_DEPTH}; max {g.MAX_CONDITIONS} conditions; max
{g.MAX_NODES} indicator+primitive nodes.

# Rules of engagement
- Prefer tf "1d" unless the user clearly means intraday.
- Vague thresholds ("oversold", "high volume"): pick the conventional value
  (rsi 30/70; volume 2x its 20 bar average) and say so in "clarifications".
- If the request needs anything the grammar cannot express, return
  "not_expressible" with the reason instead of guessing a wrong spec.
- highest/lowest are unshifted rolling extrema that include the current bar,
  so a breakout cross like "close crosses_above highest(close, n)" can never
  fire; use donchian(n).upper/.lower for breakout crosses instead.

# Examples
{_EXAMPLES}"""
