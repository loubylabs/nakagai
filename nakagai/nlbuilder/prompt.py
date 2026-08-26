"""System prompt for the NL->spec compiler, rendered from the caller's own
vocabulary so it can never drift from the validator that will judge the
reply."""

import json
from collections.abc import Mapping

from nakagai.strategies.composite.spec import (
    BESPOKE_LEG, DEFAULT_WINDOW_BARS, MAX_BLOCKS, MAX_TREE_DEPTH,
    WINDOW_BARS_BOUNDS)
from nakagai.strategies.rules import spec as g
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_condition_rule, resolve_vocabulary)

_BASE_EXAMPLE = """\
Description: "buy the dip when rsi 14 recovers above 30"
{"spec": {"version": 2, "name": "rsi-dip-buy", "timeframe": "1h",
"long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30}]},
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}},
"clarifications": ["defaulted timeframe to 1h", "used the default ATR stop and 2R target since none were given"]}\
"""

_OPENING_RANGE_EXAMPLE = """\
Description: "opening range breakout, 30 minute range, confirmed by volume 50% above its 20 bar average, only when the daily close is above the 50 day sma"
{"spec": {"version": 2, "name": "orb-volume-confirm", "timeframe": "15m",
"long": {"all": [
{"lhs": {"src": "close"}, "op": "crosses_above", "rhs": {"ind": "highest", "of": {"src": "high"}, "window": "ny_open_30"}},
{"lhs": {"src": "volume"}, "op": ">", "rhs": {"op": "*", "args": [1.5, {"ind": "sma", "n": 20, "of": {"src": "volume"}}]}},
{"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": {"ind": "sma", "n": 50}}
]},
"exits": {"time_stop": {"bars": 16}, "trailing": {"kind": "atr", "n": 14, "mult": 2.5}},
"risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0}, "target": {"kind": "rr", "rr": 2.0}}},
"clarifications": ["used a 16 bar time stop and a 2.5x ATR trail since neither was specified"]}\
"""

_NOT_EXPRESSIBLE_EXAMPLE = """\
Description: "trade based on my broker's news sentiment feed"
{"not_expressible": "the grammar has no source for external news sentiment data, only price/volume series, indicators, and primitives"}\
"""


def _examples(vocabulary: Vocabulary) -> str:
    examples = [_BASE_EXAMPLE]
    if "ny_open_30" in vocabulary.windows:
        examples.append(_OPENING_RANGE_EXAMPLE)
    examples.append(_NOT_EXPRESSIBLE_EXAMPLE)
    return "\n\n".join(examples)


# The inline legs the composite example writes when the caller declared the
# bespoke leg. The second is only reached when the caller declared NO catalog
# play, where legs are the only block kind available and one block would not
# demonstrate a composite at all.
_EXAMPLE_LEG = {
    "version": 2, "name": "rsi-under-40", "timeframe": "1d",
    "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 40}]},
}
_EXAMPLE_LEG_2 = {
    "version": 2, "name": "above-the-50", "timeframe": "1d",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"ind": "sma", "n": 50}}]},
}


def _bounds(schema: dict) -> str:
    # A condition-typed arg renders its real shape, {lhs,op,rhs}, rather than
    # the schema's bare string "condition": the model needs to know it is
    # building a nested condition there, not passing a scalar. Read off the
    # arg's declared type, so a term added to the vocabulary is described
    # without anyone editing this prompt.
    return ", ".join(
        f"{k}={{lhs,op,rhs}}" if is_condition_rule(v) else f"{k}={v}"
        for k, v in schema.items()) or "no args"


def _worked_example(plays: Mapping[str, Mapping]) -> str:
    """One composite, written over plays this caller actually declared.

    Every strategy name is READ from `plays` rather than written here. The
    example used to name two plays hard-coded into this module, and core's own
    shipped catalog contains neither, so a caller on it got a prompt teaching
    `donchian_breakout` and `adx_pullback` beside a catalog listing three other
    plays, and a validator answering `unknown strategy` to both. An example is
    the part of a prompt a model copies most literally, so it is the last place
    a name should be invented.

    The inline leg appears on the same condition the grammar states it, because
    an example reaching for a block kind the prompt just called unavailable
    teaches the model to spend retries on it.

    Empty only when the caller declared nothing a block could name. A caller
    with the bespoke leg and no catalog at all still gets one, written from two
    inline legs, because legs voting against each other is a composite the
    grammar allows and one block would not demonstrate a composite at all.
    """
    names = [name for name in sorted(plays) if name != BESPOKE_LEG]
    has_leg = BESPOKE_LEG in plays
    if not names and not has_leg:
        return ""
    blocks: dict = {chr(ord("a") + i): {"strategy": name}
                    for i, name in enumerate(names[:2])}
    described = " with ".join(names[:2])
    if has_leg:
        legs = [_EXAMPLE_LEG] if names else [_EXAMPLE_LEG, _EXAMPLE_LEG_2]
        for leg in legs:
            blocks[chr(ord("a") + len(blocks))] = {
                "strategy": BESPOKE_LEG,
                "params": {"spec": {**leg, "risk": g.DEFAULT_RISK}}}
        described += (", plus my own leg that only takes it when rsi 14 is "
                      "under 40" if names else
                      "two legs of my own: one that only takes it when rsi 14 "
                      "is under 40, and one that needs the close above its 50 "
                      "bar average")
    reply = {
        "kind": "composite",
        "spec": {"version": 1, "name": "confluence", "blocks": blocks,
                 "long": {"all": sorted(blocks)},
                 "window_bars": DEFAULT_WINDOW_BARS, "risk": g.DEFAULT_RISK},
        "clarifications": [f"used a {DEFAULT_WINDOW_BARS} bar vote window and "
                           "the default stop and target"],
    }
    return (f'\n\nDescription: "combine {described}, all of them have to fire"'
            f"\n{json.dumps(reply)}")


def _composite_section(plays: Mapping[str, Mapping]) -> str:
    """The catalog the model may name as bare blocks, one line per play.

    `plays` is CARD metadata, keyed by the name a block references: the title,
    the description, and the bound spec a timeframe is read off. That is what
    `strategies.catalog.load_entries` returns, and it is deliberately not a
    `StrategyDefinition`, which carries a name, a digest, the functions a replay
    builds and grades with, and its member tree: nothing a reader could be told
    about. This read those three fields off each
    member as class attributes until 0.5.0 stopped minting the subclasses that
    carried them (chrvsd/nakagai#417).

    A card missing a field is described by what it does have rather than
    refused: the catalog is content, and a play with no description is worth
    less to the model than a full card but more than a prompt that would not
    render at all.

    `BESPOKE_LEG` is taught only when the caller declared it. Teaching it
    unconditionally is what let the compiler return a composite the caller
    could not build, because core's own `catalog_definitions` registers no
    such member and the block raised `unknown strategy 'rules'` at
    construction. It is also skipped in the listing below whether or not it is
    declared: it has no card, and a spec file that happened to be named
    `rules` would otherwise be advertised as a bare block that then fails
    every retry for want of `params.spec`.
    """
    lines = "\n".join(
        f"- {name} [{(entry.get('spec') or {}).get('timeframe', '?')}]"
        f" {entry.get('title') or name}: {(entry.get('description') or '').strip()}"
        for name, entry in sorted(plays.items()) if name != BESPOKE_LEG)
    bespoke = (f"""
or a bespoke or tuned leg written with the RuleSpec grammar above:
{{"strategy": "{BESPOKE_LEG}", "params": {{"spec": {{<RuleSpec v2>}}}}}}"""
               if BESPOKE_LEG in plays else """
Every block names a catalog play; there is no way to write a leg inline here,
so express what the catalog cannot as a single rules spec instead.""")
    risk = ("The composite owns its own stop and target, so member risk is "
            f"ignored; a {BESPOKE_LEG}\nleg still needs a valid \"risk\" block "
            "to validate, it is simply unused."
            if BESPOKE_LEG in plays else
            "The composite owns its own stop and target, so member risk is "
            "ignored.")
    return f"""

# Composites (one strategy built from several)
When the user asks to COMBINE, confirm, or gate several strategies, return
{{"kind": "composite", "spec": {{...}}, "clarifications": [...]}} instead of a
rules spec. A composite spec:
{{"version": 1, "name": str, "blocks": {{<id>: <block>, ...}},
"long"/"short": <vote tree>, "window_bars": {WINDOW_BARS_BOUNDS[0]}-{WINDOW_BARS_BOUNDS[1]}
(default {DEFAULT_WINDOW_BARS}), "risk": {{...}}}}
A block is either a catalog play referenced by name, carrying NO params:
{{"strategy": "<catalog name>"}}{bespoke}
A vote tree is {{"all": [...]}} or {{"any": [...]}} over block ids, nestable.
At most {MAX_BLOCKS} blocks, and vote trees nest at most {MAX_TREE_DEPTH} deep;
a composite cannot contain another composite.
{risk}
Prefer legs that share one timeframe.

# Catalog plays (usable as bare blocks; they take no param overrides)
{lines}"""


def render_system_prompt(plays: Mapping[str, Mapping] | None = None, *,
                         vocabulary: Vocabulary | None = None) -> str:
    vocabulary = resolve_vocabulary(vocabulary)
    ind_lines = "\n".join(
        f"- {name}({_bounds(term.args)})"
        + (" [takes of=<expr>]" if term.kind != "bar" else "")
        + (f" [window {'required; ' if term.window_required else ''}"
           f"reducer={term.window_reduce}]" if term.window_reduce else "")
        for name, term in sorted(vocabulary.indicators.items()))
    prim_lines = "\n".join(
        f"- {name}({_bounds(term.args)})"
        + (" [session-scoped, no tf]" if term.session_scoped else "")
        # Refused outright on a 1d spec, so say it here rather than spending a
        # retry on the refusal. day_of_week carries only the first marker: it
        # takes no tf, but on daily bars it reads exactly right.
        + (" [needs an intraday spec timeframe]"
           if term.driving_frame_intraday else "")
        for name, term in sorted(vocabulary.primitives.items()))
    composite = _composite_section(plays) if plays else ""
    example = _worked_example(plays) if plays else ""
    window_lines = g.window_prompt_text(vocabulary)
    examples = _examples(vocabulary)
    return f"""You compile plain-English trading strategy descriptions into
nakagai strategy JSON. Reply with EXACTLY ONE JSON object and nothing else
(no prose, no code fences): either
{{"kind": "rules", "spec": {{...}}, "clarifications": [...]}}
or {{"not_expressible": "<one-sentence reason>"}}. Always include "kind".

# Grammar (version 2, required)
A spec: {{"version": 2, "name": str, "timeframe": one of {g.TIMEFRAMES},
"long"/"short": condition groups, "exits"?: {{...}}, "risk": {{...}}}}.
Conditions: {{"lhs": <expr>, "op": one of {g.OPS}, "rhs": <expr>}} inside
nested {{"all": [...]}} / {{"any": [...]}} groups. A group may be negated with
{{"not": <group>}}: it takes a group, never a bare condition, so "RSI is not
above 70" is {{"not": {{"all": [{{"lhs": {{"ind": "rsi", "n": 14}}, "op": ">",
"rhs": 70}}]}}}}. The lhs of a cross must be a series expression, never a
number.

Expressions are numbers or objects:
- series leaf: {{"src": one of {g.SOURCES}, "tf"?: one of {g.TIMEFRAMES}}}
- indicator: {{"ind": <name>, <args>, "of"?: <expr>, "tf"?: <tf>,
  "window"?: <registered window>}}
- math: {{"op": one of {sorted(g.MATH_OPS)}, "args": [<expr>, ...]}}
- primitive: {{"prim": <name>, <args>, "tf"?: <tf>}}

# Indicators (name(arg=bounds or choices))
{ind_lines}

# Windows (named scopes for window-capable indicators)
{window_lines}

# Primitives (session/state aware; an arg shown as {{lhs,op,rhs}} is a nested
condition and takes comparison ops only, never a cross)
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
{examples}{example}"""
