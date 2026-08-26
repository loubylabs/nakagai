"""System prompt for the NL->ScreenSpec compiler, rendered from the caller's
own vocabulary so it can never drift from the validator that will judge the
reply."""

from nakagai.strategies.rules import spec as g
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_condition_rule, resolve_vocabulary)

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
    # A condition-typed arg renders its real shape, {lhs,op,rhs}, rather than
    # the schema's bare string "condition": the model needs to know it is
    # building a nested condition there, not passing a scalar. Read off the
    # arg's declared type, so a term added to the vocabulary is described
    # without anyone editing this prompt.
    return ", ".join(
        f"{k}={{lhs,op,rhs}}" if is_condition_rule(v) else f"{k}={v}"
        for k, v in schema.items()) or "no args"


def render_screen_prompt(vocabulary: Vocabulary | None = None) -> str:
    vocabulary = resolve_vocabulary(vocabulary)
    ind_lines = "\n".join(
        f"- {name}({_bounds(term.args)})"
        + (" [takes of=<expr>]" if term.kind != "bar" else "")
        + (f" [window {'required; ' if term.window_required else ''}"
           f"reducer={term.window_reduce}]" if term.window_reduce else "")
        for name, term in sorted(vocabulary.indicators.items()))
    prim_lines = "\n".join(f"- {name}({_bounds(term.args)})"
                           for name, term in sorted(vocabulary.primitives.items()))
    window_lines = g.window_prompt_text(vocabulary)
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
nested {{"all": [...]}} / {{"any": [...]}} groups. A group may be negated with
{{"not": <group>}}: it takes a group, never a bare condition, so "RSI is not
above 70" is {{"not": {{"all": [{{"lhs": {{"ind": "rsi", "n": 14}}, "op": ">",
"rhs": 70}}]}}}}. The lhs of a cross must be a series expression, never a
number. Cross ops fire on the latest completed bar transition.

Expressions are numbers or objects:
- series leaf: {{"src": one of {g.SOURCES}, "tf"?: one of {g.TIMEFRAMES}}}
- indicator: {{"ind": <name>, <args>, "of"?: <expr>, "tf"?: <tf>,
  "window"?: <registered window>}}
- math: {{"op": one of {sorted(g.MATH_OPS)}, "args": [<expr>, ...]}}
- primitive: {{"prim": <name>, <args>}}

# Indicators (name(arg=bounds or choices))
{ind_lines}

# Windows (named scopes for window-capable indicators)
{window_lines}

# Primitives (session/state aware; an arg shown as {{lhs,op,rhs}} is a nested
condition and takes comparison ops only, never a cross)
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
