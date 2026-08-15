import json
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
# is deliberately absent. The bespoke-leg escape hatch is core's own and is
# taught unconditionally, so it is not a card a caller supplies.
_PLAYS = {"donchian_breakout": {
    "title": "Donchian channel breakout",
    "description": "Buys a break of the upper Donchian channel.",
    "spec": _CAT_SPEC}}


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
    # "rules" is the bespoke-leg escape hatch, not a catalog play entry
    assert "- rules [" not in p


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


def test_a_bespoke_rules_leg_needs_no_card_to_be_accepted():
    """`rules` is core's own escape hatch, taught unconditionally by the
    prompt, so the validator has to accept a block naming it whatever cards the
    caller supplied. `_PLAYS` holds no `rules` key and GOOD_COMPOSITE's second
    block is one, so a member view built from the cards alone refuses this."""
    assert "rules" not in _PLAYS
    client = FakeClient([json.dumps({"kind": "composite", "spec": GOOD_COMPOSITE})])
    res = compile_strategy("combine", client=client, plays=_PLAYS)
    assert res.kind == "composite" and res.spec == GOOD_COMPOSITE


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
