import json
import re
from datetime import time
from pathlib import Path

import pandas as pd
import pytest

from nakagai.model import ModelReply, _reply
from nakagai.nlbuilder import compiler
from nakagai.nlbuilder.compiler import MODEL, PROVIDER, _check, compile_strategy
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.strategies.catalog import catalog_definitions, load_entries
from nakagai.strategies.rules import core_vocabulary, validate_spec
from nakagai.strategies.rules.vocabulary import Term
from nakagai.strategies.rules.windows import WindowSpec

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"

GOOD_SPEC = {"version": 2, "name": "dip", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}

RELATIVE_SCOPE_SPEC = {
    "version": 2,
    "name": "relative_scope",
    "timeframe": "15m",
    "long": {"all": [
        {
            "lhs": {"src": "close"},
            "op": ">",
            "rhs": {"src": "close", "sym": "SPY"},
        },
        {
            "lhs": {
                "ind": "sma", "n": 20, "of": {"src": "close"},
                "sym": "SPY", "tf": "15m",
            },
            "op": ">",
            "rhs": {
                "ind": "sma", "n": 20,
                "of": {"src": "close", "sym": "QQQ", "tf": "1d"},
                "sym": "SPY", "tf": "15m",
            },
        },
    ]},
    "risk": {
        "stop": {"kind": "atr", "n": 14, "mult": 2.0},
        "target": {"kind": "rr", "rr": 2.0},
    },
}

RELATIVE_SCOPE_READBACK = (
    'Strategy "relative_scope" on 15m bars.\n'
    "Enter long when ALL of:\n"
    "  - close is above SPY:close\n"
    "  - SPY:sma(20)[15m] is above "
    "SPY:sma(20, of=QQQ:close[1d])[15m]\n"
    "Stop: 2x ATR(14) from entry. Target: 2x the risked distance."
)

LOW_IEX_DISCLOSURE = "US-equity extended-hours IEX data can be sparse."
PROMPT_VOCABULARY = core_vocabulary().with_windows(
    WindowSpec("london", "Europe/London", time(8), time(16, 30),
               "weekday", "low_iex"),
    WindowSpec("ny_am", "America/New_York", time(9, 30), time(12),
               "xnys_session", "standard"),
    WindowSpec("ny_open_30", "America/New_York", time(9, 30), time(10),
               "xnys_session", "standard"),
)


TOKENS = (100, 50, 10, 5)


class FakeModel:
    """A `Complete` over queued replies; records every call for assertions.

    Keyword-only like the protocol it stands in for, so a positional mistake
    in the compiler is a TypeError here rather than a system prompt quietly
    delivered as a message.
    """

    def __init__(self, replies, tokens=TOKENS, error="", cost_numerator=0,
                 cost_from_provider=False, rate_table_complete=True,
                 spend_unknown=False):
        self._replies = list(replies)
        self._tokens = tokens
        self._error = error
        self._cost_numerator = cost_numerator
        self._cost_from_provider = cost_from_provider
        self._rate_table_complete = rate_table_complete
        self._spend_unknown = spend_unknown
        self.requests = []

    def __call__(self, *, system, messages, max_tokens):
        self.requests.append({"system": system, "messages": messages,
                              "max_tokens": max_tokens})
        return ModelReply(self._replies.pop(0), *self._tokens,
                          self._cost_numerator, self._cost_from_provider,
                          self._rate_table_complete, self._spend_unknown,
                          self._error)


class _MeteredReplies:
    """A complete callable that returns the arrived replies it was given."""

    def __init__(self, replies):
        self._replies = list(replies)

    def __call__(self, *, system, messages, max_tokens):
        return self._replies.pop(0)


def test_prompt_renders_registries_and_is_deterministic():
    p1, p2 = render_system_prompt(), render_system_prompt()
    assert p1 == p2
    for needle in ("crosses_above", "gap_pct", "bars_since", "supertrend",
                   "time_stop", "not_expressible", '"version": 2'):
        assert needle in p1, needle
    assert '"sym"?: <SYMBOL>' in p1
    assert "[A-Z][A-Z0-9.-]{0,9}" in p1


def test_rule_prompt_renders_window_rows_from_the_supplied_vocabulary():
    prompt = render_system_prompt(vocabulary=PROMPT_VOCABULARY)
    lines = prompt.splitlines()
    london = next(line for line in lines if line.startswith("- london:"))
    assert london == (
        "- london: timezone=Europe/London; span=[08:00, 16:30); "
        "recurrence=weekday; confidence=low_iex. " + LOW_IEX_DISCLOSURE)
    ny_am = next(line for line in lines if line.startswith("- ny_am:"))
    assert ny_am == (
        "- ny_am: timezone=America/New_York; span=[09:30, 12:00); "
        "recurrence=xnys_session; confidence=standard")
    assert LOW_IEX_DISCLOSURE not in ny_am
    assert '"window"?: <registered window>' in prompt
    assert "- first(no args) [takes of=<expr>] [window required; reducer=first]" in lines


def test_rule_prompt_teaches_opening_range_through_the_window_axis():
    examples = render_system_prompt(
        vocabulary=PROMPT_VOCABULARY).split("# Examples", 1)[1]
    assert ('"rhs": {"ind": "highest", "of": {"src": "high"}, '
            '"window": "ny_open_30"}' in examples)


def _advertised_replies(prompt: str) -> list[dict]:
    """Decode each complete JSON reply in the advertised examples section."""
    examples = prompt.split("# Examples", 1)[1]
    decoder = json.JSONDecoder()
    replies = []
    cursor = 0
    while True:
        start = examples.find("\n{", cursor)
        if start < 0:
            return replies
        reply, consumed = decoder.raw_decode(examples, start + 1)
        replies.append(reply)
        cursor = start + 1 + consumed


@pytest.mark.parametrize("vocabulary,has_opening_range", [
    (core_vocabulary(), False),
    (PROMPT_VOCABULARY, True),
], ids=["core", "ny-open-30"])
def test_every_advertised_rule_example_validates_with_its_supplied_vocabulary(
        vocabulary, has_opening_range):
    prompt = render_system_prompt(vocabulary=vocabulary)
    replies = _advertised_replies(prompt)
    specs = [reply["spec"] for reply in replies if "spec" in reply]

    assert specs
    assert all(validate_spec(spec, vocabulary) == [] for spec in specs)
    assert any("ny_open_30" in json.dumps(spec) for spec in specs) is has_opening_range
    assert ("ny_open_30" in prompt) is has_opening_range


_CAT_SPEC = {"version": 2, "name": "donch", "timeframe": "1d",
             "long": {"all": [{"lhs": {"src": "close"}, "op": "crosses_above",
                               "rhs": {"ind": "sma", "n": 50}}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}
# Catalog CARD metadata, the shape `load_entries` hands back and the only shape
# that carries what this prompt needs: a frozen `StrategyDefinition` has no
# title, no description, and no readable spec to take a timeframe off. `rules`
# is deliberately absent here, so this is a caller that registered no bespoke
# leg, which is what core's own `catalog_definitions` produces.
_CARDS = {"donchian_breakout": {
    "title": "Donchian channel breakout",
    "description": "Buys a break of the upper Donchian channel.",
    "spec": _CAT_SPEC}}
# The world a caller declares: catalog cards plus, when the caller registers
# one, the bespoke leg under its own name. It has no card of its own, and the
# value is never read; the key is the declaration.
_PLAYS = {**_CARDS, "rules": {}}


def test_prompt_flags_the_primitives_a_daily_spec_cannot_use():
    """validate_spec refuses these on a 1d driving frame, so the prompt says it
    up front rather than spending a retry on the refusal. day_of_week is not
    flagged: it takes no tf, but on daily bars its reading is exactly right."""
    lines = render_system_prompt().splitlines()
    for name in ("minutes_into_session", "rvol"):
        line = next(ln for ln in lines if ln.startswith(f"- {name}("))
        assert "intraday spec timeframe" in line, line
    dow = next(ln for ln in lines if ln.startswith("- day_of_week("))
    assert "intraday spec timeframe" not in dow, dow


def test_prompt_without_plays_has_no_composite_section():
    p = render_system_prompt()
    assert "composite" not in p.lower()
    assert '"kind"' in p          # the discriminator is always taught


def test_prompt_with_plays_renders_composite_contract_and_catalog():
    p = render_system_prompt(_PLAYS)
    for needle in ("# Composites", '"kind": "composite"', "window_bars",
                   "donchian_breakout", "Donchian channel breakout", "[1d]",
                   "take no param overrides"):
        assert needle in p, needle


def test_the_bespoke_leg_is_never_listed_as_a_catalog_play():
    """`rules` is declared like any member and described by the grammar, so
    listing it under "Catalog plays (usable as bare blocks)" would advertise a
    bare block that fails every retry for want of `params.spec`.

    The fixture carries a full card under that name on purpose. Asserting
    absence against a `plays` that has no `rules` key at all is a guard that
    cannot fail, which is what this file shipped before: deleting the filter
    left the whole suite green."""
    dressed = {**_PLAYS, "rules": {
        "title": "House rules play", "description": "d",
        "spec": {**_CAT_SPEC, "timeframe": "1h"}}}
    p = render_system_prompt(dressed)
    assert "- rules [" not in p
    assert "House rules play" not in p
    # and the play beside it still is listed, so this is not passing by
    # rendering nothing at all
    assert "- donchian_breakout [1d]" in p


def test_the_worked_example_names_only_plays_the_caller_declared():
    """An example is the part of a prompt a model copies most literally, so a
    name invented here is a block the validator refuses on every retry. The
    example used to hard-code two plays core's own shipped catalog does not
    contain, which is the state this asserts against."""
    plays = load_entries(SPECS, core_vocabulary)
    p = render_system_prompt(plays)
    example = p.split("# Examples", 1)[1]
    named = set(re.findall(r'"strategy": "([^"]+)"', example))
    assert named, "the composite example has to name something"
    assert named <= set(plays) | {"rules"}, named
    # and it is genuinely built from this caller's catalog, not a fixed pair
    assert named & set(plays)


def test_the_worked_example_is_validated_by_the_checker_it_teaches():
    """The strongest form of the same guard: run the example through the
    validator the reply will meet."""
    plays = load_entries(SPECS, core_vocabulary)
    example = render_system_prompt(plays).split("# Examples", 1)[1]
    reply = json.loads(example[example.rindex("\n{"):].strip())
    errors, _ = _check("composite", reply["spec"], plays, core_vocabulary())
    assert errors == []


def test_the_bespoke_leg_is_taught_only_when_the_caller_declares_it():
    """Teaching a block kind the caller cannot build is the defect that made
    the compiler return unbuildable composites: core registered `rules` for
    itself, so validation passed and `CompositeStrategy` then raised
    `unknown strategy 'rules'`."""
    with_leg = render_system_prompt(_PLAYS)
    assert '{"strategy": "rules", "params": {"spec": {<RuleSpec v2>}}}' in with_leg
    assert '"strategy": "rules"' in with_leg          # and the worked example

    # The worked example moves with the grammar, asserted on text only the
    # three-block form carries. Asserting '"strategy": "rules"' alone cannot
    # fail here: the grammar line above already contains that substring, so it
    # would pass with the example collapsed to the catalog-only form.
    assert '"strategy": "rules", "params"' in with_leg.split("# Examples", 1)[1]

    without = render_system_prompt(_CARDS)
    assert '{"strategy": "rules"' not in without
    assert "rules" not in without.split("# Examples", 1)[1]
    assert "there is no way to write a leg inline here" in without
    assert "# Composites" in without                  # composites still offered
    # The risk sentence is the third thing that moves, and it was the one with
    # no guard at all: leaving it on the bespoke-leg wording tells the model a
    # rules leg exists three lines after saying one does not.
    assert "a rules" not in without
    assert "a rules" in with_leg


def test_the_prompt_and_the_validator_spell_the_bespoke_leg_once():
    """One protocol name, one declaration. Renaming it in the prompt alone
    would teach a grammar the validator refuses, and the refusal would advise
    the model to use a word the prompt no longer contains."""
    from nakagai.nlbuilder import prompt as prompt_module
    from nakagai.strategies.composite import spec as spec_module

    assert prompt_module.BESPOKE_LEG is spec_module.BESPOKE_LEG


def test_the_bespoke_leg_declaration_is_a_key_and_never_a_value():
    """What a caller must supply to declare the leg, pinned.

    Core ships no producer that emits a `plays` mapping containing `rules`:
    `load_entries` returns one entry per spec file. The platform assembles it
    (`nakagai_platform.registry.builder_plays`), so core's own fixtures are
    hand-written, which is the fixture-shape hazard that let #417 hide.

    What makes that safe is that the VALUE is never read: the listing filters
    the key out and both validators read membership alone. So an empty dict, a
    full card, and anything else all render one prompt, and a platform that
    supplies a different shape than these fixtures cannot break."""
    shapes = [{}, {"title": "t", "description": "d", "spec": _CAT_SPEC}, None]
    rendered = {render_system_prompt({**_CARDS, "rules": shape})
                for shape in shapes}
    assert len(rendered) == 1
    # and it is genuinely the declared-leg prompt, not the undeclared one
    assert "there is no way to write a leg inline here" not in rendered.pop()


def test_a_caller_with_only_the_bespoke_leg_still_gets_a_worked_example():
    """Legs voting against each other is a composite the grammar allows, so a
    caller who declared the leg and no catalog is not taught the contract and
    then shown nothing. The example used to be empty here, which made the
    release note describing it false."""
    p = render_system_prompt({"rules": {}})
    assert "# Composites" in p
    example = p.split("# Examples", 1)[1]
    reply = json.loads(example[example.rindex("\n{"):].strip())
    kinds = [b["strategy"] for b in reply["spec"]["blocks"].values()]
    assert kinds == ["rules", "rules"], kinds
    errors, _ = _check("composite", reply["spec"], {"rules": {}}, core_vocabulary())
    assert errors == []


def test_nothing_declared_at_all_renders_no_composite_section():
    """The one case that really has nothing to combine."""
    assert render_system_prompt({}) == render_system_prompt()


def test_a_card_missing_fields_still_renders_a_line():
    """The catalog is content, and a half-filled card is worth less to the
    model than a full one but more than a prompt that will not render. Strict
    subscripting here would raise inside `render_system_prompt`, which is the
    same class of break as chrvsd/nakagai#417."""
    p = render_system_prompt({"bare": {}, "no_desc": {"title": "T", "spec": _CAT_SPEC}})
    assert "- bare [?] bare: " in p
    assert "- no_desc [1d] T: " in p


def test_prompt_with_plays_is_deterministic():
    assert render_system_prompt(_PLAYS) == render_system_prompt(_PLAYS)


def test_the_prompt_takes_the_catalog_the_way_core_ships_it():
    """The regression every fixture in this file used to hide.

    `_PLAYS` above is hand-written, and the shape it replaced was a minted
    `RuleStrategy` subclass carrying `title`, `description` and
    `DEFAULT_PARAMS`, which core 0.5.0 stopped producing. So the suite stayed
    green over a prompt no shipped caller could render: the platform hands
    `load_entries`/`catalog_definitions` output and got AttributeError
    (chrvsd/nakagai#417).

    This drives core's own two producers instead, and asserts the pair agrees:
    one card per definition, and every card reaching the prompt.
    """
    plays = load_entries(SPECS, core_vocabulary)
    assert plays, "the shipped catalog is what makes this test meaningful"
    p = render_system_prompt(plays)
    for name, entry in plays.items():
        assert f"- {name} [{entry['spec']['timeframe']}]" in p
        assert entry["title"] in p
        assert entry["description"].strip() in p
    assert {d.name for d in catalog_definitions(SPECS, core_vocabulary)} == set(plays)


def test_happy_path_returns_spec_and_readback():
    client = FakeModel([json.dumps({"spec": GOOD_SPEC, "clarifications": ["defaulted timeframe to 1h"]})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert "dip" in res.readback
    assert res.clarifications == ["defaulted timeframe to 1h"]
    assert res.attempts == 1
    # One plain string, not a block list: `cache_control` was Anthropic's, and
    # this compiler no longer speaks any provider's dialect.
    assert client.requests[0]["system"] == render_system_prompt()
    assert client.requests[0]["max_tokens"] == 8000


def test_blank_prompt_policy_keeps_the_strategy_prompt_byte_identical():
    expected = render_system_prompt().encode("utf-8")
    for policy in ("", " \n\t "):
        client = FakeModel([json.dumps({"spec": GOOD_SPEC})])
        compile_strategy("buy rsi dips", client=client, prompt_policy=policy)
        assert client.requests[0]["system"].encode("utf-8") == expected


def test_prompt_policy_is_appended_once_before_the_first_strategy_call():
    policy = "# House policy\n- one rule"
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    client = FakeModel([
        json.dumps({"spec": bad}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_strategy(
        "buy rsi dips", client=client, prompt_policy=f"  {policy}\n")
    expected = render_system_prompt() + "\n\n" + policy
    assert result.attempts == 2
    assert [call["system"] for call in client.requests] == [
        expected, expected,
    ]
    assert expected.count(policy) == 1


def test_plain_string_candidate_validator_is_one_complete_error():
    seen = 0

    def retry_validator(kind, spec):
        nonlocal seen
        seen += 1
        assert kind == "rules"
        assert spec == GOOD_SPEC
        return "house policy failed" if seen == 1 else []

    client = FakeModel([
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    result = compile_strategy(
        "buy rsi dips", client=client, max_retries=1,
        candidate_validator=retry_validator)
    assert result.spec == GOOD_SPEC
    assert client.requests[1]["messages"][-1]["content"] == (
        "The spec failed validation. Fix exactly these errors and resend the "
        "full JSON object:\n- house policy failed"
    )

    terminal = compile_strategy(
        "buy rsi dips",
        client=FakeModel([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: "house policy failed",
    )
    assert terminal.not_expressible == (
        "could not produce a valid spec; last errors: house policy failed"
    )
    assert "h; o; u; s; e" not in terminal.not_expressible

    ordered = compile_strategy(
        "buy rsi dips",
        client=FakeModel([json.dumps({"spec": GOOD_SPEC})]),
        max_retries=0,
        candidate_validator=lambda kind, spec: ("first error", "second error"),
    )
    assert ordered.not_expressible == (
        "could not produce a valid spec; last errors: first error; second error"
    )


def test_strategy_normalizer_is_revalidated_before_the_caller_validator():
    validator_calls = []
    invalid = {"version": 2, "name": "normalized-invalid", "timeframe": "2h"}
    client = FakeModel([
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ])
    calls = 0

    def normalize(kind, spec):
        nonlocal calls
        calls += 1
        assert kind == "rules"
        return invalid if calls == 1 else GOOD_SPEC

    result = compile_strategy(
        "buy rsi dips", client=client, max_retries=1,
        candidate_normalizer=normalize,
        candidate_validator=lambda kind, spec: validator_calls.append((kind, spec)) or [],
    )
    assert result.spec == GOOD_SPEC
    assert result.attempts == 2
    assert "timeframe" in client.requests[1]["messages"][-1]["content"]
    assert validator_calls == [("rules", GOOD_SPEC)]


def test_strategy_caller_channels_share_the_default_three_attempt_stream():
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    replies = [
        json.dumps({"spec": bad}),
        json.dumps({"spec": GOOD_SPEC}),
        json.dumps({"spec": GOOD_SPEC}),
    ]
    normalized = []

    def normalize(kind, spec):
        normalized.append((kind, spec))
        return spec

    result = compile_strategy(
        "buy rsi dips", client=FakeModel(replies),
        candidate_normalizer=normalize,
        candidate_validator=lambda kind, spec: (
            "integer policy failed", "symbol policy failed"),
    )
    assert result.attempts == 3
    assert result.usage == {
        "input_tokens": 300,
        "output_tokens": 150,
        "cache_read_tokens": 30,
        "cache_write_tokens": 15,
    }
    assert normalized == [("rules", GOOD_SPEC), ("rules", GOOD_SPEC)]
    assert result.not_expressible == (
        "could not produce a valid spec; last errors: "
        "integer policy failed; symbol policy failed"
    )


def test_strategy_readback_uses_the_native_valid_normalized_candidate():
    seen = []
    client = FakeModel([json.dumps({"spec": GOOD_SPEC})])
    result = compile_strategy(
        "compare the market", client=client,
        candidate_normalizer=lambda kind, spec: RELATIVE_SCOPE_SPEC,
        candidate_validator=lambda kind, spec: seen.append((kind, spec)) or [],
    )
    assert seen == [("rules", RELATIVE_SCOPE_SPEC)]
    assert result.spec == RELATIVE_SCOPE_SPEC
    assert result.readback == RELATIVE_SCOPE_READBACK


def test_validation_errors_trigger_retry_with_error_feedback():
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    client = FakeModel([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry_text = json.dumps(client.requests[1]["messages"])
    assert "timeframe" in retry_text            # the validator errors were fed back


def test_gives_up_after_max_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    client = FakeModel([bad, bad, bad])
    res = compile_strategy("x", client=client, max_retries=2)
    assert res.spec is None and res.attempts == 3
    assert "timeframe" in res.not_expressible


def test_not_expressible_passthrough():
    client = FakeModel([json.dumps({"not_expressible": "renko bars are not supported"})])
    res = compile_strategy("renko magic", client=client)
    assert res.spec is None and "renko" in res.not_expressible


def test_json_fences_are_tolerated():
    client = FakeModel(["```json\n" + json.dumps({"spec": GOOD_SPEC}) + "\n```"])
    assert compile_strategy("x", client=client).spec == GOOD_SPEC


def test_revision_includes_current_spec_in_user_turn():
    client = FakeModel([json.dumps({"spec": GOOD_SPEC})])
    compile_strategy("tighten the stop", current_spec=GOOD_SPEC, client=client)
    assert "current spec" in json.dumps(client.requests[0]["messages"]).lower()


def test_an_empty_reply_retries_instead_of_crashing():
    """A model that answers with nothing at all. `ModelReply.text` is always a
    string, so this is the retry the empty case still has to become."""
    client = FakeModel(["", json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "not parseable JSON" in retry["content"]


def test_usage_is_summed_across_retries_with_cache_split():
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    client = FakeModel([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.attempts == 2
    assert res.usage == {"input_tokens": 200, "output_tokens": 100,
                         "cache_read_tokens": 20, "cache_write_tokens": 10}
    assert res.model == "deepseek/deepseek-v4-flash-0731"


def test_normalized_cache_subsets_are_summed_across_arrived_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    first = _reply(_ArrivedResponse({
        "choices": [{"message": {"content": bad}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 7,
            "prompt_tokens_details": {
                "cached_tokens": 25,
                "cache_write_tokens": 15,
            },
        },
    }))
    second = _reply(_ArrivedResponse({
        "choices": [{"message": {
            "content": json.dumps({"spec": GOOD_SPEC}),
        }}],
        "usage": {
            "prompt_tokens": 80,
            "completion_tokens": 9,
            "prompt_tokens_details": {
                "cached_tokens": 10,
                "cache_write_tokens": 20,
            },
        },
    }))

    result = compile_strategy("buy rsi dips", client=_MeteredReplies([
        first,
        second,
    ]))

    assert result.usage == {
        "input_tokens": 110,
        "output_tokens": 16,
        "cache_read_tokens": 35,
        "cache_write_tokens": 35,
    }
    assert result.spend_unknown is False
    assert result.retries_taken == 1


def test_usage_present_on_not_expressible():
    client = FakeModel([json.dumps({"not_expressible": "no renko"})])
    res = compile_strategy("renko", client=client)
    assert res.not_expressible and res.usage["input_tokens"] == 100


def test_a_fresh_compile_result_has_zero_unknown_spend_and_retries():
    result = compiler.CompileResult()

    assert result.spend_unknown is False
    assert result.retries_taken == 0


# Production break caught: retry compilation could drop an arrived exact bill.
def test_two_exact_bills_sum_when_validation_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    exact_two_attempt_result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 1_111_111, True, False, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   2_222_222, True, False, False, ""),
    ]))

    assert exact_two_attempt_result.cost_numerator == 3_333_333
    assert exact_two_attempt_result.cost_from_provider is True
    assert exact_two_attempt_result.spend_unknown is False
    assert exact_two_attempt_result.retries_taken == 1


# Production break caught: alternating complete settlement bases could be
# flattened into one certain aggregate that neither basis could settle.
def test_mixed_exact_only_and_counter_only_retries_are_unknown():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 1_111_111, True, False, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   0, False, True, False, ""),
    ]))

    assert result.spec == GOOD_SPEC
    assert result.usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
    }
    assert result.cost_numerator == 1_111_111
    assert result.cost_from_provider is False
    assert result.spend_unknown is True
    assert result.retries_taken == 1


def test_mixed_counter_only_and_exact_only_retries_are_unknown():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 0, False, True, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   1_111_111, True, False, False, ""),
    ]))

    assert result.spec == GOOD_SPEC
    assert result.usage == {
        "input_tokens": 200,
        "output_tokens": 100,
        "cache_read_tokens": 20,
        "cache_write_tokens": 10,
    }
    assert result.cost_numerator == 1_111_111
    assert result.cost_from_provider is False
    assert result.spend_unknown is True
    assert result.retries_taken == 1


def test_all_counter_complete_retries_remain_table_priceable():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 0, False, True, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   0, False, True, False, ""),
    ]))

    assert result.cost_numerator == 0
    assert result.cost_from_provider is False
    assert result.spend_unknown is False


def test_exact_and_counter_complete_can_join_a_counter_only_retry():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 1_111_111, True, True, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   0, False, True, False, ""),
    ]))

    assert result.cost_numerator == 1_111_111
    assert result.cost_from_provider is False
    assert result.spend_unknown is False


# Production break caught: an unknown earlier attempt could be erased by a retry.
def test_unknown_spend_is_ored_across_arrived_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 1_111_111, True, True, True, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   2_222_222, True, True, False, ""),
    ]))

    assert result.cost_numerator == 3_333_333
    assert result.cost_from_provider is False
    assert result.spend_unknown is True
    assert result.retries_taken == 1


# Production break caught: an unbilled retry could still claim full invoice evidence.
def test_a_synthetic_retry_revokes_exact_bill_provenance():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    mixed_two_attempt_result = compile_strategy("x", client=_MeteredReplies([
        ModelReply(bad, *TOKENS, 1_111_111, True, True, False, ""),
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   0, False, True, False, ""),
    ]))

    assert mixed_two_attempt_result.cost_numerator == 1_111_111
    assert mixed_two_attempt_result.cost_from_provider is False


BAD_SPEC_REPLY = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})


class _ArrivedResponse:
    status_code = 200

    def __init__(self, doc):
        self._doc = doc

    def json(self):
        return self._doc


# Production break caught: malformed evidence on a retry could be forgotten.
def test_malformed_arrived_evidence_is_sticky_and_keeps_known_lower_bounds():
    incomplete = _reply(_ArrivedResponse({
        "choices": [{"message": {"content": BAD_SPEC_REPLY}}],
        "usage": {"prompt_tokens": 17,
                  "prompt_tokens_details": {"cached_tokens": 4}},
    }))
    result = compile_strategy("x", client=_MeteredReplies([
        incomplete,
        ModelReply(json.dumps({"spec": GOOD_SPEC}), *TOKENS,
                   2_222_222, True, True, False, ""),
    ]))

    assert result.usage == {
        "input_tokens": 113,
        "output_tokens": 50,
        "cache_read_tokens": 14,
        "cache_write_tokens": 5,
    }
    assert result.cost_numerator == 2_222_222
    assert result.cost_from_provider is False
    assert result.spend_unknown is True
    assert result.retries_taken == 1


class _BilledFailure:
    """First call answers with an invalid spec (so the loop retries), second
    comes back FAILED and non-free: the seam for the property `ModelReply`
    exists to carry."""

    def __init__(self, error, tokens):
        self.calls = 0
        self._error = error
        self._tokens = tokens

    def __call__(self, *, system, messages, max_tokens):
        self.calls += 1
        if self.calls == 1:
            return ModelReply(
                BAD_SPEC_REPLY, *TOKENS, 0, False, True, False, "")
        return ModelReply(
            "", *self._tokens, 0, False, True, False, self._error)


def test_a_billed_failure_records_its_tokens_before_the_error():
    """The whole point of `ModelReply.error`. A response that ARRIVED and
    failed was charged for, so its counts land in `usage` before the error ends
    the loop. Billing after the error return instead records that call as free,
    and a caller settling a reserve against `usage` under-reports by exactly
    the failing attempt."""
    res = compile_strategy("x", client=_BilledFailure(
        "model call failed: HTTP 500: upstream fell over", (41, 7, 3, 2)))
    assert res.error == "model call failed: HTTP 500: upstream fell over"
    assert res.spec is None
    assert res.attempts == 2                     # the failing attempt counts
    assert res.usage == {"input_tokens": 141, "output_tokens": 57,
                         "cache_read_tokens": 13, "cache_write_tokens": 7}


def test_a_transport_failure_reports_zero_observed_usage_and_unknown_spend():
    """No response supplied usage, but the attempted delivery may be billed."""
    client = FakeModel([""], tokens=(0, 0, 0, 0),
                       error="model call failed: connection reset",
                       rate_table_complete=False,
                       spend_unknown=True)
    res = compile_strategy("x", client=client)
    assert res.error == "model call failed: connection reset"
    assert res.spec is None
    assert res.attempts == 1
    assert res.usage == {"input_tokens": 0, "output_tokens": 0,
                         "cache_read_tokens": 0, "cache_write_tokens": 0}
    assert res.spend_unknown is True
    assert res.retries_taken == 0


class _RaisingModel:
    """First call succeeds (invalid spec -> retry), second RAISES.

    `Complete` says a failure comes back as `ModelReply.error`, but the
    callable belongs to the caller, so the loop cannot assume it obeys. A raise
    escaping here would take every count already accumulated with it."""

    def __init__(self, first_reply, cost_numerator=0, cost_from_provider=False):
        self._sent = False
        self._first = first_reply
        self._cost_numerator = cost_numerator
        self._cost_from_provider = cost_from_provider

    def __call__(self, *, system, messages, max_tokens):
        if self._sent:
            raise RuntimeError("api down")
        self._sent = True
        return ModelReply(self._first, *TOKENS, self._cost_numerator,
                          self._cost_from_provider, True, False, "")


def test_a_callable_that_raises_returns_an_error_result_with_partial_usage():
    res = compile_strategy("x", client=_RaisingModel(BAD_SPEC_REPLY))
    assert "api down" in res.error
    assert res.spec is None
    assert res.attempts == 2                       # the raising attempt counts
    assert res.usage["input_tokens"] == 100        # attempt 1's usage survives


# Production break caught: a callable exception could erase prior cost evidence.
def test_a_raised_retry_revokes_exact_bill_provenance():
    raised_after_exact_result = compile_strategy(
        "x", client=_RaisingModel(BAD_SPEC_REPLY, 1_111_111, True))

    assert raised_after_exact_result.cost_numerator == 1_111_111
    assert raised_after_exact_result.cost_from_provider is False
    assert raised_after_exact_result.spend_unknown is True
    assert raised_after_exact_result.retries_taken == 1


def test_no_client_means_openrouter_on_the_pinned_endpoint(monkeypatch):
    """What a caller who supplies nothing gets. Porting the call shape while
    leaving the old model id would send a Claude id to OpenRouter and fail at
    request time, so the id is asserted literally, and so is the provider pin:
    one id is served by many providers at different quantizations."""
    built = []

    def fake_openrouter_complete(**kwargs):
        built.append(kwargs)
        return FakeModel([json.dumps({"spec": GOOD_SPEC})])

    monkeypatch.setattr(compiler, "openrouter_complete", fake_openrouter_complete)

    assert compile_strategy("buy rsi dips").spec == GOOD_SPEC
    assert built == [{"model": "deepseek/deepseek-v4-flash-0731",
                      "provider": {"require_parameters": True,
                                   "order": ["alibaba"],
                                   "allow_fallbacks": False}}]
    assert (MODEL, PROVIDER) == (built[0]["model"], built[0]["provider"])

    # And a caller naming a model still selects it, which is what `result.model`
    # claims it did.
    res = compile_strategy("buy rsi dips", model="qwen/qwen3-max")
    assert built[1]["model"] == "qwen/qwen3-max"
    assert res.model == "qwen/qwen3-max"


def test_missing_default_key_is_a_known_zero_attempt_compile(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = compile_strategy("buy rsi dips")

    assert "OPENROUTER_API_KEY" in result.error
    assert result.attempts == 0
    assert result.retries_taken == 0
    assert result.usage == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    assert result.cost_numerator == 0
    assert result.cost_from_provider is False
    assert result.spend_unknown is False


GOOD_COMPOSITE = {
    "version": 1, "name": "confluence",
    "blocks": {"a": {"strategy": "donchian_breakout"},
               "b": {"strategy": "rules", "params": {"spec": GOOD_SPEC}}},
    "long": {"all": ["a", "b"]}, "window_bars": 4,
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}}}


def test_composite_kind_validates_and_describes_as_a_composite():
    client = FakeModel([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine them", client=client, plays=_PLAYS)
    assert res.kind == "composite"
    assert res.spec == GOOD_COMPOSITE
    assert "Composite" in res.readback and "2 blocks" in res.readback


def test_missing_kind_defaults_to_rules():
    client = FakeModel([json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client, plays=_PLAYS)
    assert res.kind == "rules" and res.spec == GOOD_SPEC


def test_unrecognized_kind_falls_back_to_rules():
    client = FakeModel([json.dumps({"kind": "Composite", "spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client, plays=_PLAYS)
    assert res.kind == "rules" and res.spec == GOOD_SPEC


def test_bad_vote_tree_retries_with_composite_errors():
    bad = {**GOOD_COMPOSITE, "long": {"all": ["a", "zz"]}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": bad}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    assert "zz" in json.dumps(client.requests[1]["messages"])


def test_bad_rules_leg_retries_with_block_prefixed_errors():
    bad = {**GOOD_COMPOSITE,
           "blocks": {"a": {"strategy": "donchian_breakout"},
                      "b": {"strategy": "rules",
                            "params": {"spec": {**GOOD_SPEC, "timeframe": "2h"}}}}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": bad}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    assert "blocks.b" in json.dumps(client.requests[1]["messages"])


def test_composite_without_plays_is_rejected_not_crashed():
    client = FakeModel([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})] * 3)
    res = compile_strategy("combine", client=client, max_retries=2)
    assert res.spec is None
    assert "not available" in res.not_expressible


def test_plays_reach_the_system_prompt():
    client = FakeModel([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    compile_strategy("combine", client=client, plays=_PLAYS)
    assert "donchian_breakout" in client.requests[0]["system"]


def test_a_bespoke_leg_the_caller_never_declared_is_sent_back():
    """The regression the member view used to hide. Core added `rules` to the
    membership itself, so a composite naming it validated clean and then raised
    `unknown strategy 'rules'` at `CompositeStrategy` construction, which is
    past every retry the model could have acted on.

    `_CARDS` is the catalog without the bespoke leg, exactly what core's own
    `catalog_definitions` registers."""
    assert "rules" not in _CARDS
    client = FakeModel([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})] * 3)
    res = compile_strategy("combine", client=client, plays=_CARDS, max_retries=2)
    assert res.spec is None
    assert "unknown strategy 'rules'" in res.not_expressible


def test_a_malformed_strategy_name_becomes_a_retry_not_a_crash():
    """End to end, over the loop that actually depends on it. The model can emit
    anything, which is why the retry loop exists at all; a reply whose
    `strategy` is a list used to raise TypeError out of validation, and the
    platform turned that into a 503 rather than sending the model back."""
    malformed = {**GOOD_COMPOSITE,
                 "blocks": {**GOOD_COMPOSITE["blocks"], "a": {"strategy": []}}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": malformed}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "unknown strategy" in retry["content"]


def test_an_unknown_play_is_sent_back_by_name():
    """The permissive direction of the membership check. Accepting every name
    would let the model invent a play, report success, and leave the failure to
    surface at construction rather than as a retry it could act on."""
    invented = {**GOOD_COMPOSITE,
                "blocks": {**GOOD_COMPOSITE["blocks"],
                           "a": {"strategy": "golden_cross"}}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": invented}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    # The USER turn, which carries the validator's errors. Asserting over the
    # whole message list cannot fail: it also holds the assistant echo of the
    # model's own reply, which contains the invented name by construction.
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "unknown strategy 'golden_cross'" in retry["content"]


def test_a_catalog_play_carrying_param_overrides_is_sent_back():
    """The other half of the block rule, end to end. A card's spec is bound, so
    a block naming it has no param surface for an override to land on, and an
    override that was silently ignored would run something other than what the
    author asked for."""
    tuned = {**GOOD_COMPOSITE,
             "blocks": {"a": {"strategy": "donchian_breakout",
                              "params": {"n": 55}},
                        "b": {"strategy": "rules", "params": {"spec": GOOD_SPEC}}}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": tuned}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    assert "takes no param overrides" in json.dumps(client.requests[1]["messages"])


def test_composite_check_does_not_drop_the_caller_vocabulary():
    """_check holds a vocabulary and hands it to validate_spec on the rules
    branch, but dropped it on the composite branch: a composite whose rules
    leg uses an injected term was refused as an unknown indicator here while
    CompositeStrategy.__init__ validates the identical spec clean. The one
    surface that composed the strategy must not be the one that calls it
    invalid."""
    house = core_vocabulary().with_terms(
        Term("always_one", "series", {}, {},
             lambda s, a: pd.Series(1.0, index=s.index)))
    inner = {"version": 2, "name": "leg", "timeframe": "15m",
             "long": {"all": [{"lhs": {"ind": "always_one"}, "op": ">", "rhs": 0.0}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}
    spec = {"name": "c", "blocks": {"a": {"strategy": "rules",
                                          "params": {"spec": inner}}},
            "long": {"all": ["a"]}}

    errors, _ = _check("composite", spec, _PLAYS, house)

    assert errors == []


def test_a_reply_that_is_valid_json_but_not_an_object_is_a_retry():
    """`json.loads` returns a list for `[]`, and the loop read `.get` off it.
    Valid JSON is not the contract; the contract is exactly one JSON object."""
    client = FakeModel(["[]", json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "not parseable JSON" in retry["content"]


def test_a_hostile_rules_leg_inside_a_composite_is_a_retry():
    """The composite path reaches the rule grammar through `validate_spec`, so
    a leg carrying an unhashable primitive name used to raise past the loop."""
    hostile = {**GOOD_COMPOSITE,
               "blocks": {**GOOD_COMPOSITE["blocks"],
                          "b": {"strategy": "rules", "params": {"spec": {
                              "version": 2, "name": "leg", "timeframe": "15m",
                              "long": {"all": [{"lhs": {"prim": []},
                                                "op": ">", "rhs": 1}]},
                              "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                                       "target": {"kind": "rr", "rr": 2.0}}}}}}}
    client = FakeModel([json.dumps({"kind": "composite", "spec": hostile}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "unknown primitive" in retry["content"]


def test_a_reply_json_cannot_even_parse_is_a_retry():
    """`json.loads` raises a BARE ValueError for an integer past the
    interpreter's digit limit, not the JSONDecodeError the retry boundary named,
    so a reply the model could have corrected escaped the loop instead."""
    huge = '{"spec": ' + "9" * 5000 + "}"
    client = FakeModel([huge, json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "not parseable JSON" in retry["content"]


def _deep(levels: int) -> str:
    return "[" * levels + "]" * levels


def test_a_reply_nested_past_the_decoder_is_a_retry():
    """`json.loads` raises RecursionError inside the decoder itself, which is
    not a ValueError, so a reply nested thousands deep escaped the loop."""
    client = FakeModel(['{"spec": ' + _deep(10_000) + "}",
                         json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2


def test_an_absurdly_nested_vote_tree_is_a_retry():
    """`_check_tree` recursed to the interpreter's limit on a tree that is
    absurd rather than merely wrong, and a RecursionError out of a validator is
    a dead request rather than an error the model can act on."""
    tree = {"all": ["a"]}
    for _ in range(1000):
        tree = {"all": [tree]}
    deep = {**GOOD_COMPOSITE, "long": tree}
    # and the bound is TAUGHT, so the model is not refused for following the
    # grammar it was given: a validator stricter than its own prompt burns
    # retries on a rule nobody stated.
    assert "nest at most" in render_system_prompt(_PLAYS)
    client = FakeModel([json.dumps({"kind": "composite", "spec": deep}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "nest at most" in retry["content"]


def test_a_number_the_canon_cannot_hold_is_a_retry():
    """It validated clean and then raised OverflowError, and 0.6.1 moved that
    raise from the readback into the platform's save path by fixing the
    renderer and dropping the check.

    The check is right, and the reason is `canon.canonical_expr`: it returns
    `float(node)` for every numeric scalar, which is what makes 20 and 20.0 one
    spec. A number outside the float range therefore has no canonical form, so
    no `spec_hash`, so it can be neither stored nor identified. Refusing it here
    is the one place that can say why."""
    huge = {**GOOD_SPEC,
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": 10 ** 1000}]}}
    client = FakeModel([json.dumps({"spec": huge}),
                         json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "out of range" in retry["content"]


def test_the_readback_still_renders_a_number_the_validator_would_refuse():
    """Defense in depth, and not redundant with the check above. A describer is
    read by surfaces that must not raise whatever reaches them, and it is
    reachable with a spec this validator never saw."""
    from nakagai.strategies.rules import describe_spec
    huge = {**GOOD_SPEC,
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": 10 ** 1000}]}}
    text = describe_spec(huge, core_vocabulary())
    assert "1" + "0" * 1000 in text


def test_an_injected_intraday_primitive_is_a_retry_not_a_keyerror():
    """`_ONE_BAR_SESSION` explains CORE's primitives, and the flag it explains
    is settable by any caller injecting a vocabulary. The validator runs with
    the caller's terms in it, so a subscript there raised KeyError on a term the
    prompt had just taught the model. The platform injects house terms on every
    compile, which is exactly this shape."""
    vocab = core_vocabulary().with_terms(
        Term("custom_intraday", "primitive", {}, {},
             lambda s, a: pd.Series(1.0, index=s.index),
             driving_frame_intraday=True))
    daily = {**GOOD_SPEC, "timeframe": "1d",
             "long": {"all": [{"lhs": {"prim": "custom_intraday"},
                               "op": ">", "rhs": 0}]}}
    client = FakeModel([json.dumps({"spec": daily}),
                         json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("x", client=client, vocabulary=vocab)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "needs intraday bars" in retry["content"]
