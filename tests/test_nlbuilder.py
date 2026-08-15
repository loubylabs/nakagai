import json
import re
from pathlib import Path

import pandas as pd

from nakagai.nlbuilder.compiler import _check, compile_strategy
from nakagai.nlbuilder.prompt import render_system_prompt
from nakagai.strategies.catalog import catalog_definitions, load_entries
from nakagai.strategies.rules import core_vocabulary
from nakagai.strategies.rules.vocabulary import Term

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"

GOOD_SPEC = {"version": 2, "name": "dip", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 30}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}


class _Block:
    type = "text"
    def __init__(self, text): self.text = text


class _Usage:
    def __init__(self, input_tokens=100, output_tokens=50,
                 cache_read_input_tokens=10, cache_creation_input_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class _Resp:
    def __init__(self, text, usage=None):
        self.content = [_Block(text)]
        self.usage = usage or _Usage()


class FakeClient:
    """Yields queued replies; records every request for prompt assertions."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.messages = self
    def create(self, **kwargs):
        self.requests.append(kwargs)
        return _Resp(self._replies.pop(0))


def test_prompt_renders_registries_and_is_deterministic():
    p1, p2 = render_system_prompt(), render_system_prompt()
    assert p1 == p2
    for needle in ("crosses_above", "opening_range_high", "bars_since", "supertrend",
                   "time_stop", "not_expressible", '"version": 2'):
        assert needle in p1, needle


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
    for name in ("opening_range_high", "opening_range_low",
                 "minutes_into_session", "rvol"):
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
    client = FakeClient([json.dumps({"spec": GOOD_SPEC, "clarifications": ["defaulted timeframe to 1h"]})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert "dip" in res.readback
    assert res.clarifications == ["defaulted timeframe to 1h"]
    assert res.attempts == 1
    assert client.requests[0]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_validation_errors_trigger_retry_with_error_feedback():
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    client = FakeClient([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry_text = json.dumps(client.requests[1]["messages"])
    assert "timeframe" in retry_text            # the validator errors were fed back


def test_gives_up_after_max_retries():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    client = FakeClient([bad, bad, bad])
    res = compile_strategy("x", client=client, max_retries=2)
    assert res.spec is None and res.attempts == 3
    assert "timeframe" in res.not_expressible


def test_not_expressible_passthrough():
    client = FakeClient([json.dumps({"not_expressible": "renko bars are not supported"})])
    res = compile_strategy("renko magic", client=client)
    assert res.spec is None and "renko" in res.not_expressible


def test_json_fences_are_tolerated():
    client = FakeClient(["```json\n" + json.dumps({"spec": GOOD_SPEC}) + "\n```"])
    assert compile_strategy("x", client=client).spec == GOOD_SPEC


def test_revision_includes_current_spec_in_user_turn():
    client = FakeClient([json.dumps({"spec": GOOD_SPEC})])
    compile_strategy("tighten the stop", current_spec=GOOD_SPEC, client=client)
    assert "current spec" in json.dumps(client.requests[0]["messages"]).lower()


class _NoTextResp:
    """A response with no text content block at all: a thinking-only turn.
    `_text` raises StopIteration on this; the except clause names
    StopIteration precisely to turn it into a retry, not a crash."""
    def __init__(self):
        self.content = []


class _RawClient:
    """Like FakeClient, but hands back exactly what's queued instead of
    wrapping every reply in a text _Resp: the seam for injecting a response
    object that has no text block at all."""
    def __init__(self, replies):
        self._replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self._replies.pop(0)


def test_textless_reply_retries_instead_of_crashing():
    client = _RawClient([_NoTextResp(), _Resp(json.dumps({"spec": GOOD_SPEC}))])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC
    assert res.attempts == 2


def test_usage_is_summed_across_retries_with_cache_split():
    bad = {**GOOD_SPEC, "timeframe": "2h"}
    client = FakeClient([json.dumps({"spec": bad}), json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.attempts == 2
    assert res.usage == {"input_tokens": 200, "output_tokens": 100,
                         "cache_read_tokens": 20, "cache_write_tokens": 10}
    assert res.model == "claude-opus-4-8"


def test_usage_present_on_not_expressible():
    client = FakeClient([json.dumps({"not_expressible": "no renko"})])
    res = compile_strategy("renko", client=client)
    assert res.not_expressible and res.usage["input_tokens"] == 100


class _RaisingClient:
    """First create() succeeds (invalid spec -> retry), second raises: the
    seam for proving partial-loop usage survives an API failure."""
    def __init__(self, first_reply):
        self._sent = False
        self._first = first_reply
        self.messages = self

    def create(self, **kwargs):
        if self._sent:
            raise RuntimeError("api down")
        self._sent = True
        return _Resp(self._first)


def test_client_exception_returns_error_result_with_partial_usage():
    bad = json.dumps({"spec": {**GOOD_SPEC, "timeframe": "2h"}})
    res = compile_strategy("x", client=_RaisingClient(bad))
    assert "api down" in res.error
    assert res.spec is None
    assert res.attempts == 2                       # the raising attempt counts
    assert res.usage["input_tokens"] == 100        # attempt 1's usage survives


def test_usageless_response_is_tolerated():
    # _NoTextResp has no usage attribute at all; must not crash the summing.
    client = _RawClient([_NoTextResp(), _Resp(json.dumps({"spec": GOOD_SPEC}))])
    res = compile_strategy("buy rsi dips", client=client)
    assert res.spec == GOOD_SPEC and res.usage["input_tokens"] == 100


GOOD_COMPOSITE = {
    "version": 1, "name": "confluence",
    "blocks": {"a": {"strategy": "donchian_breakout"},
               "b": {"strategy": "rules", "params": {"spec": GOOD_SPEC}}},
    "long": {"all": ["a", "b"]}, "window_bars": 4,
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}}}


def test_composite_kind_validates_and_describes_as_a_composite():
    client = FakeClient([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine them", client=client, plays=_PLAYS)
    assert res.kind == "composite"
    assert res.spec == GOOD_COMPOSITE
    assert "Composite" in res.readback and "2 blocks" in res.readback


def test_missing_kind_defaults_to_rules():
    client = FakeClient([json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client, plays=_PLAYS)
    assert res.kind == "rules" and res.spec == GOOD_SPEC


def test_unrecognized_kind_falls_back_to_rules():
    client = FakeClient([json.dumps({"kind": "Composite", "spec": GOOD_SPEC})])
    res = compile_strategy("buy rsi dips", client=client, plays=_PLAYS)
    assert res.kind == "rules" and res.spec == GOOD_SPEC


def test_bad_vote_tree_retries_with_composite_errors():
    bad = {**GOOD_COMPOSITE, "long": {"all": ["a", "zz"]}}
    client = FakeClient([json.dumps({"kind": "composite", "spec": bad}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    assert "zz" in json.dumps(client.requests[1]["messages"])


def test_bad_rules_leg_retries_with_block_prefixed_errors():
    bad = {**GOOD_COMPOSITE,
           "blocks": {"a": {"strategy": "donchian_breakout"},
                      "b": {"strategy": "rules",
                            "params": {"spec": {**GOOD_SPEC, "timeframe": "2h"}}}}}
    client = FakeClient([json.dumps({"kind": "composite", "spec": bad}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    assert "blocks.b" in json.dumps(client.requests[1]["messages"])


def test_composite_without_plays_is_rejected_not_crashed():
    client = FakeClient([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})] * 3)
    res = compile_strategy("combine", client=client, max_retries=2)
    assert res.spec is None
    assert "not available" in res.not_expressible


def test_plays_reach_the_system_prompt():
    client = FakeClient([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    compile_strategy("combine", client=client, plays=_PLAYS)
    assert "donchian_breakout" in client.requests[0]["system"][0]["text"]


def test_a_bespoke_leg_the_caller_never_declared_is_sent_back():
    """The regression the member view used to hide. Core added `rules` to the
    membership itself, so a composite naming it validated clean and then raised
    `unknown strategy 'rules'` at `CompositeStrategy` construction, which is
    past every retry the model could have acted on.

    `_CARDS` is the catalog without the bespoke leg, exactly what core's own
    `catalog_definitions` registers."""
    assert "rules" not in _CARDS
    client = FakeClient([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})] * 3)
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
    client = FakeClient([json.dumps({"kind": "composite", "spec": malformed}),
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
    client = FakeClient([json.dumps({"kind": "composite", "spec": invented}),
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
    client = FakeClient([json.dumps({"kind": "composite", "spec": tuned}),
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
    client = FakeClient(["[]", json.dumps({"spec": GOOD_SPEC})])
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
    client = FakeClient([json.dumps({"kind": "composite", "spec": hostile}),
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
    client = FakeClient([huge, json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "not parseable JSON" in retry["content"]


def _deep(levels: int) -> str:
    return "[" * levels + "]" * levels


def test_a_reply_nested_past_the_decoder_is_a_retry():
    """`json.loads` raises RecursionError inside the decoder itself, which is
    not a ValueError, so a reply nested thousands deep escaped the loop."""
    client = FakeClient(['{"spec": ' + _deep(10_000) + "}",
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
    client = FakeClient([json.dumps({"kind": "composite", "spec": deep}),
                         json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.spec == GOOD_COMPOSITE and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "nest at most" in retry["content"]


def test_a_number_too_large_for_the_readback_is_rendered_not_refused():
    """It validated clean and then raised OverflowError in the DESCRIBER, one
    step after the retry loop stopped watching: the spec was good and the
    request died anyway.

    Refusing it at validation was the first repair, and a lens was right that
    it changed the verdict on a spec the grammar accepts. The renderer is what
    could not cope, so the renderer is what was fixed: this spec compiles on
    attempt ONE, and its readback carries the number exactly."""
    huge = {**GOOD_SPEC,
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": 10 ** 1000}]}}
    client = FakeClient([json.dumps({"spec": huge})])
    res = compile_strategy("buy dips", client=client)
    assert res.spec == huge and res.attempts == 1
    assert "1" + "0" * 1000 in res.readback


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
    client = FakeClient([json.dumps({"spec": daily}),
                         json.dumps({"spec": GOOD_SPEC})])
    res = compile_strategy("x", client=client, vocabulary=vocab)
    assert res.spec == GOOD_SPEC and res.attempts == 2
    retry = [m for m in client.requests[1]["messages"] if m["role"] == "user"][-1]
    assert "needs intraday bars" in retry["content"]
