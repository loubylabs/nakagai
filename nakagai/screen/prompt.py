"""System prompt for the NL->ScreenSpec compiler, rendered from the caller's
own vocabulary so it can never drift from the validator that will judge the
reply."""

from nakagai.strategies.rules import spec as g
from nakagai.screen.facts import DISCOVERY_FACTS, FACT_LABELS
from nakagai.screen.sectors import SECTOR_NAMES
from nakagai.strategies.rules.vocabulary import (
    Vocabulary, is_condition_rule, resolve_vocabulary)

_BASE_UNIVERSE = "All eligible US common stocks"
_FACT_GROUPS = {
    "fundamentals": (
        "shares_outstanding", "market_cap", "price_to_ncav",
    ),
    "market_activity": (
        "price", "change_pct", "gap_pct", "session_volume",
    ),
    "dilution": (
        "shelf_filings_365d", "offering_filings_365d", "atm_offerings_365d",
    ),
    "short_interest": (
        "short_pct_outstanding", "days_to_cover", "short_interest_change_pct",
    ),
    "insiders": ("insider_buyers_30d", "insider_buy_usd_30d"),
    "congress": ("congress_buyers_30d",),
    "funds": ("fund_holders", "fund_holders_added"),
    "earnings": ("days_to_earnings",),
    "news": ("news_new_stories",),
}
_CAPABILITY_EXAMPLES = (
    "Market cap under $300 million with price under $10",
    "RSI under 30 with volume above twice its 20-day average.",
)

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

Description: "oversold chip stocks"
{"spec": {"version": 1, "tf": "1d", "universe": {"industries": ["3674"]},
"conditions": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}]}},
"clarifications": ["read 'chip stocks' as SIC 3674, semiconductors"]}

Description: "Any low float bangers?"
{"not_expressible": "Float is not supported."}\
"""


def screen_capabilities() -> dict:
    """JSON-ready capability projection shared with product surfaces."""
    return {
        "base_universe": _BASE_UNIVERSE,
        "fact_groups": {
            group: list(names) for group, names in _FACT_GROUPS.items()
        },
        "fact_labels": dict(FACT_LABELS),
        "technical_scope": "daily",
        "logic": ["all", "any", "not"],
        "examples": list(_CAPABILITY_EXAMPLES),
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
    fact_lines = "\n".join(
        f"- {name}: {FACT_LABELS[name]}" for name in DISCOVERY_FACTS)
    window_lines = g.window_prompt_text(vocabulary)
    sector_names = ", ".join(SECTOR_NAMES)
    return f"""You compile plain-English market screens into nakagai ScreenSpec v1
JSON. A screen is a filter: it answers "which symbols match this condition
right now." Reply with EXACTLY ONE JSON object and nothing else (no prose, no
code fences): either {{"spec": {{...}}, "clarifications": [...]}} or
{{"not_expressible": "<one-sentence reason>"}}.

# Schema (version 1, required)
A spec: {{"version": 1, "tf": one of {g.TIMEFRAMES} (the base bars the verdict
is read on; default "1d"), "universe"?: {{...}} (see Universe),
"conditions": one condition group}}. Nothing else:
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
SYMBOL is an uppercase market symbol matching [A-Z][A-Z0-9.-]{{0,9}}.
- current fact: {{"fact": <name>}} using one name listed below. A fact takes
  no sym, tf, window, or provider key and cannot be the lhs of a cross.
- series leaf: {{"src": one of {g.SOURCES}, "sym"?: <SYMBOL>,
  "tf"?: one of {g.TIMEFRAMES}}}
- indicator: {{"ind": <name>, <args>, "of"?: <expr>, "tf"?: <tf>,
  "window"?: <registered window>, "sym"?: <SYMBOL>}}
- math: {{"op": one of {sorted(g.MATH_OPS)}, "args": [<expr>, ...]}}
- primitive: {{"prim": <name>, <args>, "sym"?: <SYMBOL>}}

# Current discovery facts
{fact_lines}

# Universe
Optional: {{"sectors": [...], "industries": [...]}} narrows the screen to a
slice of the market. Sectors are sorted by SEC industry code into these 11
names: {sector_names}. Industries are four-digit SEC SIC codes as strings, e.g.
"3674" for semiconductors. A symbol is in when its sector or its code is
listed. Use it only when the user names a sector or an industry, never as a
condition, and omit a list rather than send it empty. SIC cannot separate
internet and media platforms from Technology (Alphabet and Meta file under
computer services), so when the user means internet, media or communication
platforms, list both "Communication Services" and "Technology" and add a
clarification saying so.

# Indicators (name(arg=bounds or choices))
{ind_lines}

# Windows (named scopes for window-capable indicators)
{window_lines}

# Primitives (session/state aware; an arg shown as {{lhs,op,rhs}} is a nested
condition and takes comparison ops only, never a cross)
{prim_lines}

# Limits
Max expression depth {g.MAX_DEPTH}; max {g.MAX_CONDITIONS} conditions; max
{g.MAX_NODES} indicator+primitive nodes; max {g.MAX_EXPR_NODES} expression nodes
(every number, source, fact, math op, indicator and primitive counts).

# Rules of engagement
- Prefer tf "1d" unless the user clearly means intraday.
- price_to_ncav below 1 means the market values the company under its net
  current assets (current assets less all liabilities); it is absent for
  companies without a positive NCAV.
- insider facts count open-market purchases (Form 4 code P) traded in the last
  30 days; grants, option exercises and sales never count.
- congress_buyers_30d counts buy disclosures FILED in the last 30 days;
  members disclose up to 45 days after the trade.
- fund facts count the named 13F funds we track, from each fund's newest filed
  quarter; they are not all funds and a filing lags its quarter end.
  fund_holders_added counts tracked funds with no position in their previous
  quarter.
- days_to_earnings is whole days from today (US Eastern) to the next scheduled
  report; 0 is today.
- news_new_stories counts distinct stories about the company over the last 7
  days (US Eastern dates, today included) that were judged a new development;
  reprints and recaps do not count, and a story not yet judged makes the value
  unavailable.
- Vague thresholds ("oversold", "high volume"): pick the conventional value
  (rsi 30/70; volume 2x its 20 bar average) and say so in "clarifications".
- If the request needs anything the grammar cannot express, return
  "not_expressible" with the reason instead of guessing a wrong spec.
- Float (float shares, low float) is not supported: return
  "not_expressible" saying so, and never substitute shares outstanding.
- highest/lowest are unshifted rolling extrema that include the current bar,
  so a breakout cross like "close crosses_above highest(close, n)" can never
  fire; use donchian(n).upper/.lower for breakout crosses instead.

# Examples
{_EXAMPLES}"""
