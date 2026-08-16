"""Composite over the example catalog: member FACTORIES are the only door.

Membership arrives one way, as a mapping from strategy name to a callable
that builds a fresh member, passed to `__init__`. There is no class-bound
alternative, so a composite handed no membership only ever accepts an empty
spec and every populated one names its members explicitly.
"""

import json
from pathlib import Path

import pytest

import pandas as pd

from nakagai.engine.portfolio_types import (
    ReplayInputError, Signal, StrategyOutputError, StrategyRuntimeError,
)
from nakagai.engine import rules_definition, spec_base_digest
from nakagai.strategies.base import MarketContext, Strategy
from nakagai.strategies.catalog import catalog_definitions
from nakagai.strategies.composite import (CompositeStrategy, describe_composite_spec,
                                          resolve_config_refs,
                                          validate_composite_spec)
from nakagai.strategies.composite.strategy import member_blocks
from nakagai.strategies.rules import RuleStrategy, core_vocabulary
from nakagai.strategies.rules.vocabulary import Term

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


def _catalog_members():
    """Name to factory, which is the one shape membership arrives in.

    Built from the frozen catalog definitions rather than from minted classes:
    a definition's `factory` IS the member builder, so a composite over the
    shipped specs is assembled exactly the way a registry bundle assembles one.
    """
    return {item.name: item.factory
            for item in catalog_definitions(SPECS, core_vocabulary)}


def _two_member_spec():
    return {
        "name": "two-member-composite",
        "blocks": {
            "a": {"strategy": "sma_cross", "params": {}},
            "b": {"strategy": "rsi_reversion", "params": {}},
        },
        "long": {"any": ["a", "b"]},
    }


def test_validate_against_example_members():
    members = _catalog_members()
    assert validate_composite_spec(_two_member_spec(), members, allow_refs=False) == []


def test_validate_rejects_unknown_member():
    errs = validate_composite_spec(_two_member_spec(), {}, allow_refs=False)
    assert errs


def test_supplied_membership_instantiates_members():
    strat = CompositeStrategy({"spec": _two_member_spec()},
                              members=_catalog_members())
    assert set(strat._members) == {"a", "b"}


def test_a_composite_handed_no_membership_rejects_a_populated_spec():
    """No class binding to fall back on, so every block names a strategy
    nothing can build and the spec is refused by name."""
    with pytest.raises(ValueError):
        CompositeStrategy({"spec": _two_member_spec()})


from nakagai.strategies.composite import validate_composite_blocks

_LEG_SPEC = {"version": 2, "name": "rsi-leg", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 40}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}

# The value model, which is the only shape core's own producers hand this
# function: `catalog_definitions` returns definitions and `composite_definition`
# takes a mapping of them. The retired shape was a minted RuleStrategy subclass
# carrying a `PARAMS` class attribute, and pinning THAT here is what let this
# validator keep reading an attribute no caller on 0.5.0 could supply while the
# suite stayed green (chrvsd/nakagai#417).
_MEMBERS = {
    "rules": rules_definition("rules",
                              spec_base_digest({"$unbound_adapter": "rules"})),
    "bare_play": rules_definition("bare_play", spec_base_digest(_LEG_SPEC),
                                  spec=_LEG_SPEC),
}


def test_a_definition_member_is_judged_rather_than_raising():
    """The break itself. `members` is read for MEMBERSHIP and nothing else, so
    a frozen definition answers here exactly as a class used to. Before the
    fix this raised AttributeError: 'StrategyDefinition' object has no
    attribute 'PARAMS', which reached the platform as a 500 rather than as a
    validation answer."""
    spec = {"blocks": {"a": {"strategy": "bare_play"}}}
    assert validate_composite_blocks(spec, _MEMBERS) == []


def test_what_a_bound_member_actually_does_with_an_override():
    """The two failure modes the block rule exists to get ahead of, measured
    rather than described. Both are worse than the refusal, in different ways,
    and a docstring claiming only one of them sent a reader looking for the
    wrong thing (the release-note correction in this branch).

    `params.spec` is refused at the factory, by the contract's own
    `ReplayInputError` and not merely by something with that message, so a block
    supplying one dies mid-replay rather than at validation.

    Any other key is carried into `params` and does not reach the spec the
    strategy decides from. That is the provable half of "silently ignored": the
    factory does not merge it, so the play a tuned-looking block runs is the
    untuned one. Whether a future `RuleStrategy` starts consulting such a key is
    that class's business and chrvsd/nakagai#460's.
    """
    bound = catalog_definitions(SPECS, core_vocabulary)[0]
    with pytest.raises(ReplayInputError, match="already binds its rule spec"):
        bound.factory({"spec": {"version": 2}})

    plain = bound.factory({})
    # EVERY field the bound spec actually has, read off the spec rather than
    # listed here, plus one name it does not. A merge is likeliest to be written
    # for a field the spec already carries, and a hand-written list of probes is
    # outrun by the next field added to the grammar; deriving them keeps this
    # exhaustive over whatever a catalog spec holds.
    probes = {key: f"probe-{key}" for key in plain.spec}
    probes["made_up"] = 99
    assert len(probes) > 3, probes           # the catalog spec is not degenerate
    for key, value in sorted(probes.items()):
        built = bound.factory({key: value})
        # the VALUE, not merely the key: a factory that kept the key and
        # replaced what it held would satisfy a presence check.
        assert built.params[key] == value, key
        assert built.spec == plain.spec, key            # and the spec is untouched
    assert "made_up" not in json.dumps(plain.spec)


def test_an_unbound_member_under_another_name_is_refused():
    """A KNOWN limitation, pinned so it is found by reading rather than by
    debugging. The bespoke leg is recognized by the literal name `rules`, so an
    unbound definition registered as anything else is refused for carrying
    params even though its spec legitimately travels there and its own factory
    builds it.

    Not a regression: the class model refused it identically, because
    `Strategy.PARAMS` is empty on an unbound adapter too. Closing it needs the
    caller to say which members are unbound, which this signature does not
    carry."""
    unbound = rules_definition("private_rules",
                               spec_base_digest({"$unbound_adapter": "private"}))
    spec = {"blocks": {"a": {"strategy": "private_rules",
                             "params": {"spec": _LEG_SPEC}}}}
    errs = validate_composite_blocks(spec, {"private_rules": unbound})
    assert errs == ["blocks.a: private_rules is a built-in spec and takes no "
                    "param overrides; use a rules block for a tuned leg"]
    # The half that makes it a limitation rather than a rule: the definition it
    # just refused builds that exact block without complaint.
    assert unbound.factory({"spec": _LEG_SPEC}) is not None


def test_an_unhashable_strategy_name_is_an_error_and_not_a_crash():
    """A validator that raises is not a validator.

    `strategy` is whatever arrived in the JSON. A list or an object is
    unhashable, so a membership test on it raised TypeError out of both of these
    functions, and the NL builder's retry loop is the caller that could not
    survive it: `metered_compile` turned the raise into a 503, so a malformed
    model reply ended the request instead of becoming the retry the model could
    have acted on."""
    for bad in ([], {"a": 1}, {"nested": {}}, 7.5, None):
        spec = {"name": "c", "blocks": {"a": {"strategy": bad}},
                "long": {"all": ["a"]},
                "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                         "target": {"kind": "rr", "rr": 2.0}}}
        errs = validate_composite_spec(spec, _MEMBERS, allow_refs=False)
        assert any("unknown strategy" in e for e in errs), (bad, errs)
        # and the per-block layer leaves it to the sibling rather than raising
        assert validate_composite_blocks(spec, _MEMBERS) == [], bad


def test_an_unhashable_config_ref_is_an_error_and_not_a_crash():
    """`resolve_config_refs` had the same shape one function down: it looked a
    ref up in the caller's saved configs without checking it was a name."""
    spec = {"blocks": {"a": {"config": ["not", "a", "name"]}}}
    resolved, errs = resolve_config_refs(spec, {"real-one": {}})
    assert errs and "no saved strategy config" in errs[0]
    assert resolved["blocks"]["a"] == spec["blocks"]["a"]


def test_membership_is_all_that_is_read_of_a_member():
    """A bare name set validates identically to a mapping of definitions.

    Stated as a test because it is the contract the NL builder now relies on:
    `_check` hands its caller's catalog CARDS straight to both validators, and
    never holds a definition at all."""
    spec = {"blocks": {"a": {"strategy": "bare_play", "params": {"rsi_n": 10}},
                       "b": {"strategy": "rules", "params": {"spec": _LEG_SPEC}}}}
    assert (validate_composite_blocks(spec, frozenset(_MEMBERS))
            == validate_composite_blocks(spec, _MEMBERS))


def test_valid_blocks_produce_no_errors():
    spec = {"blocks": {"a": {"strategy": "bare_play"},
                       "b": {"strategy": "rules", "params": {"spec": _LEG_SPEC}}}}
    assert validate_composite_blocks(spec, _MEMBERS) == []


def test_rules_block_nested_spec_is_validated():
    bad = {**_LEG_SPEC, "timeframe": "2h"}
    spec = {"blocks": {"a": {"strategy": "rules", "params": {"spec": bad}}}}
    errs = validate_composite_blocks(spec, _MEMBERS)
    assert any(e.startswith("blocks.a: ") and "timeframe" in e for e in errs)


def test_rules_block_needs_params_spec():
    spec = {"blocks": {"a": {"strategy": "rules", "params": {}}}}
    errs = validate_composite_blocks(spec, _MEMBERS)
    assert errs == ["blocks.a: rules blocks need params.spec (the rule JSON object)"]


def test_catalog_block_rejects_param_overrides():
    spec = {"blocks": {"a": {"strategy": "bare_play", "params": {"rsi_n": 10}}}}
    errs = validate_composite_blocks(spec, _MEMBERS)
    assert len(errs) == 1
    assert errs[0].startswith("blocks.a: bare_play is a built-in spec")


def test_unknown_member_is_left_to_the_structural_validator():
    spec = {"blocks": {"a": {"strategy": "nope", "params": {"x": 1}}}}
    assert validate_composite_blocks(spec, _MEMBERS) == []


def test_config_ref_blocks_are_skipped():
    spec = {"blocks": {"a": {"config": "saved-thing"}}}
    assert validate_composite_blocks(spec, _MEMBERS) == []


def test_default_vocabulary_rejects_a_term_the_caller_never_injected():
    """No vocabulary passed means core_vocabulary(), so a term that only
    exists in some other caller's house vocabulary is unknown here."""
    inner = {"version": 2, "name": "leg", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "always_one"}, "op": ">", "rhs": 0.0}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}
    spec = {"blocks": {"a": {"strategy": "rules", "params": {"spec": inner}}}}
    errs = validate_composite_blocks(spec, _MEMBERS)
    assert any("always_one" in e for e in errs)


def test_a_passed_vocabulary_reaches_the_nested_rules_block():
    """The caller's vocabulary must not be dropped on the way into the
    rules block's own spec validation, or a term the caller injected reads
    as unknown even though it is the one the caller means to use."""
    house = core_vocabulary().with_terms(
        Term("always_one", "series", {}, {},
             lambda s, a: pd.Series(1.0, index=s.index)))
    inner = {"version": 2, "name": "leg", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "always_one"}, "op": ">", "rhs": 0.0}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}
    spec = {"blocks": {"a": {"strategy": "rules", "params": {"spec": inner}}}}
    assert validate_composite_blocks(spec, _MEMBERS, house) == []


def test_a_composite_carries_the_vocabulary_its_members_were_built_with():
    """The live scanner resolves `getattr(strategy, "vocabulary", ...)`.
    Without this attribute a composite silently evaluates on core's terms
    while its own members evaluate on the house's, which fails as a wrong
    number rather than as an error."""
    house = core_vocabulary().with_terms(
        Term("always_one", "series", {}, {},
             lambda s, a: pd.Series(1.0, index=s.index)))
    spec = {"name": "c",
            "blocks": {"a": {"strategy": "rules", "params": {"spec": _LEG_SPEC}}},
            "long": {"all": ["a"]}}
    strategy = CompositeStrategy(
        {"spec": spec},
        members={"rules": lambda params: RuleStrategy(params, vocabulary=house)})

    assert strategy.vocabulary is house


# ------------------------------------------------- members vote, one by one

def _ctx(close: float = 100.0) -> MarketContext:
    idx = pd.date_range("2026-01-05 14:30", periods=4, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 1_000.0}, index=idx)
    return MarketContext(symbol="SPY", now=idx[-1] + pd.Timedelta(minutes=15),
                         bars={"15m": bars, "1h": bars, "1d": bars})


def _member_signal(ctx, direction, tag):
    ref = float(ctx.driving_bars["close"].iloc[-1])
    stop, target = ((ref - 1.0, ref + 2.0) if direction == "long"
                    else (ref + 1.0, ref - 2.0))
    return Signal(ctx.symbol, direction, ref, stop, target, 0.5, (tag,),
                  f"member {direction}")


class _BothWays(Strategy):
    """One member, two signals in one call: a long AND a short."""

    name = "both_ways"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        return (_member_signal(ctx, "long", "up"),
                _member_signal(ctx, "short", "down"))


class _Boom(Strategy):
    name = "boom"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        raise ValueError("member is broken")


class _Bad(Strategy):
    name = "bad"
    DEFAULT_PARAMS = {}

    def on_bar(self, ctx):
        ref = float(ctx.driving_bars["close"].iloc[-1])
        # Stop on the wrong side of the reference for a long.
        return (Signal(ctx.symbol, "long", ref, ref + 1.0, ref + 2.0, 0.5,
                       ("bad",), "wrong side"),)


def _both_sides_spec(member: str):
    return {"version": 1, "name": "c", "window_bars": 1,
            "blocks": {"a": {"strategy": member, "params": {}}},
            "long": {"all": ["a"]}, "short": {"all": ["a"]},
            "risk": {"stop": {"kind": "percent", "pct": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}


def _composite(member_cls):
    return CompositeStrategy({"spec": _both_sides_spec(member_cls.name)},
                             members={member_cls.name: member_cls})


def test_a_member_returning_two_signals_casts_both_votes():
    """Reading only the member's first signal loses the short vote outright,
    so the composite would fire one side where its own logic says two."""
    signals = _composite(_BothWays).on_bar(_ctx())
    assert [signal.direction for signal in signals] == ["long", "short"]


def test_composite_output_is_an_ordered_tuple():
    signals = _composite(_BothWays).on_bar(_ctx())
    assert isinstance(signals, tuple)
    assert [tags[0] for tags in (s.setup_tags for s in signals)] == ["composite",
                                                                    "composite"]


def test_a_raising_member_aborts_the_run_and_names_its_block():
    with pytest.raises(StrategyRuntimeError) as caught:
        _composite(_Boom).on_bar(_ctx())
    assert caught.value.code == "strategy_raised"
    assert caught.value.details["block"] == "a"
    assert caught.value.details["strategy"] == "boom"
    assert caught.value.details["error"] == "ValueError"


def test_a_member_signal_outside_the_contract_aborts_the_run():
    with pytest.raises(StrategyOutputError) as caught:
        _composite(_Bad).on_bar(_ctx())
    assert caught.value.code == "invalid_value"


def test_an_inert_composite_returns_an_empty_tuple():
    assert CompositeStrategy({}).on_bar(_ctx()) == ()


# --------------------------------------------------- membership as factories


def test_member_blocks_reads_declared_order_and_skips_unresolved_refs():
    spec = {"blocks": {"z": {"strategy": "bare_play"},
                       "a": {"strategy": "rules", "params": {"spec": _LEG_SPEC}},
                       "r": {"config": "saved-elsewhere"}}}
    assert [block for block, _, _ in member_blocks(spec)] == ["z", "a"]
    assert [strategy for _, strategy, _ in member_blocks(spec)] == ["bare_play",
                                                                   "rules"]
    assert member_blocks({}) == ()


def test_a_composite_builds_members_from_supplied_factories():
    """The canonical membership door: params in, a fresh member out.

    Nothing about this composite's members lives on a class, so two
    composites over one spec share no member instance, and the name the
    registry definition gave it travels with the runtime.
    """
    calls = []

    def build_bare(params):
        calls.append(params)
        return _MEMBERS["bare_play"].factory(params)

    spec = _two_member_spec()
    spec["blocks"] = {"a": {"strategy": "bare_play", "params": {}}}
    spec["long"] = {"all": ["a"]}
    first = CompositeStrategy({"spec": spec}, members={"bare_play": build_bare},
                              name="combo")
    second = CompositeStrategy({"spec": spec}, members={"bare_play": build_bare},
                               name="combo")
    assert first.name == second.name == "combo"
    assert first._members["a"] is not second._members["a"]
    assert len(calls) == 2


def test_only_the_supplied_membership_decides_what_a_block_may_name():
    spec = {"version": 1, "name": "c", "blocks": {"a": {"strategy": "rules",
                                                       "params": {"spec": _LEG_SPEC}}},
            "long": {"all": ["a"]}}
    # A membership that knows only bare_play cannot build this block, and there
    # is no second place a name could resolve from.
    with pytest.raises(ValueError):
        CompositeStrategy({"spec": spec},
                          members={"bare_play": _MEMBERS["bare_play"].factory})
    built = CompositeStrategy({"spec": spec},
                              members={"rules": _MEMBERS["rules"].factory})
    assert isinstance(built._members["a"], RuleStrategy)


# --------------------------------------------------------------------------
# A validator must RETURN errors, never raise. The NL builder's retry loop is
# the caller that depends on it: its whole purpose is to hand the model its own
# validator's errors and ask again, and a model can emit anything.


_HOSTILE = ([], {}, {"a": 1}, 7, 7.5, None, True, "", [[]], {"k": []})


def _hostile_specs():
    """One spec per place a caller value reaches a lookup, times every awkward
    JSON value. Generated rather than listed, because the defect was never one
    site: it was a habit of testing membership on whatever arrived."""
    risk = {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
            "target": {"kind": "rr", "rr": 2.0}}
    for bad in _HOSTILE:
        yield {"version": 2, "name": "x", "timeframe": bad,
               "long": {"all": []}, "risk": risk}
        for leaf in ({"prim": bad}, {"ind": bad}, {"src": bad},
                     {"op": bad, "args": [1, 2]},
                     # NESTED, and under a crossing op. The end-anchored and
                     # session-scoped walks reach their lookup by a different
                     # route than the leaf checker, and a crossing is what
                     # triggers them, so a crossing-free crossing missed a
                     # whole site.
                     {"op": "+", "args": [{"src": "close"}, {"prim": bad}]}):
            for op in (">", "crosses_above"):
                yield {"version": 2, "name": "x", "timeframe": "15m",
                       "long": {"all": [{"lhs": leaf, "op": op, "rhs": 1}]},
                       "risk": risk}
        yield {"version": 2, "name": "x", "timeframe": "15m",
               "long": {"all": [{"lhs": {"src": "close"}, "op": bad, "rhs": 1}]},
               "risk": risk}
        yield {"version": 2, "name": "x", "timeframe": "15m",
               "long": {"all": [{"lhs": {"src": "close", "tf": bad},
                                 "op": ">", "rhs": 1}]}, "risk": risk}


def test_no_hostile_json_makes_the_rule_validator_raise():
    from nakagai.strategies.rules import validate_spec
    vocab = core_vocabulary()
    for spec in _hostile_specs():
        errs = validate_spec(spec, vocab)
        assert isinstance(errs, list), spec
        assert errs, spec        # and it says something, rather than accepting


def test_no_hostile_json_makes_the_composite_validators_raise():
    for bad in _HOSTILE:
        for spec in (bad,
                     {"blocks": bad},
                     {"name": "c", "blocks": {"a": bad}, "long": {"all": ["a"]}},
                     {"name": "c", "blocks": {"a": {"strategy": bad}},
                      "long": {"all": ["a"]}},
                     {"name": "c", "blocks": {"a": {"strategy": "rules",
                                                    "params": bad}},
                      "long": {"all": ["a"]}},
                     {"name": "c", "blocks": {"a": {"config": bad}},
                      "long": {"all": ["a"]}}):
            assert isinstance(
                validate_composite_spec(spec, _MEMBERS, allow_refs=False), list), spec
            assert isinstance(validate_composite_blocks(spec, _MEMBERS), list), spec
            resolved, errs = resolve_config_refs(spec, {"saved": bad})
            assert isinstance(errs, list), spec

    # And the case the loop above cannot reach: a ref that IS a name, pointing
    # at a saved config whose VALUE is hostile. This is the one that gets
    # inlined into a spec the engine runs, so its shape is checked before it is
    # indexed rather than after it fails.
    named = {"name": "c", "blocks": {"a": {"config": "saved"}},
             "long": {"all": ["a"]}}
    for bad in _HOSTILE:
        resolved, errs = resolve_config_refs(named, {"saved": bad})
        assert isinstance(errs, list), bad
        assert errs and "no saved strategy config" in errs[0], (bad, errs)
        # refused rather than half-inlined
        assert resolved["blocks"]["a"] == {"config": "saved"}, bad
    # a well-formed one still inlines
    good = {"strategy": "bare_play", "params": {"n": 1}}
    resolved, errs = resolve_config_refs(named, {"saved": good})
    assert errs == [] and resolved["blocks"]["a"] == good
    # and `configs` itself is a caller value: it was read with `.get` on the
    # strength of the REF being a name, which says nothing about the mapping.
    for bad in _HOSTILE:
        resolved, errs = resolve_config_refs(named, bad)
        assert isinstance(errs, list), bad


def test_a_describer_answers_rather_than_raising_on_a_non_spec():
    """A describer's contract is a spec that VALIDATED, and that is not a
    weakening: every shipped path validates first and describes only on the
    clean branch (`api/builder_routes.py::describe_endpoint` raises 422 before
    describing, `describe_composite_endpoint` answers `readback: ""` on the
    error branch, and `nlbuilder._check` returns the describer to a caller that
    applies it only when the error list is empty).

    So the guard here is the TOP-level shape, which is the one thing a caller
    can get wrong without the validator having seen it at all, plus a name the
    grammar does not define, which a valid-looking spec can still carry. Do not
    widen this to arbitrary interior shapes: that would assert a totality the
    describers do not claim, and a test asserting more than its subject
    promises is the overclaim this branch has already corrected twice."""
    from nakagai.strategies.rules import describe_spec
    vocab = core_vocabulary()
    for bad in _HOSTILE:
        assert isinstance(describe_spec(bad, vocab), str), bad
        assert isinstance(describe_composite_spec(bad), str), bad
        assert isinstance(describe_composite_spec({"blocks": bad}), str), bad
    # and every spec that VALIDATES describes, which is the real contract
    good = {"version": 2, "name": "d", "timeframe": "1h",
            "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": ">",
                              "rhs": 30}]},
            "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                     "target": {"kind": "rr", "rr": 2.0}}}
    from nakagai.strategies.rules import validate_spec
    assert validate_spec(good, vocab) == []
    assert "RSI" in describe_spec(good, vocab).upper()
