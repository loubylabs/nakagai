"""The frozen registry: one closed bundle, pure dependencies, fresh runtimes.

Three properties carry the whole module. The bundle is a value, so the order
it was supplied in cannot reach its digest. Dependency resolution is pure, so
asking what a play reads never builds the thing that reads it. And a factory
is a constructor rather than a cache, so two candidates over one definition
share no object and therefore no state.
"""

import dataclasses

import pytest

from nakagai.engine.portfolio_types import ReplayInputError
from nakagai.engine.registry import (
    REGISTRY_CONTRACT_VERSION,
    FrozenStrategyRegistry,
    StrategyDefinition,
    StrategyDependencies,
    StrategyRegistry,
    _bundle_digest,
    composite_definition,
    rules_definition,
    vocabulary_digest,
)
from nakagai.strategies.base import Strategy
from nakagai.strategies.composite import CompositeStrategy
from nakagai.strategies.rules import RuleStrategy, Term, core_vocabulary
from tests.portfolio_fixtures import (
    COMPOSITE_PARAMS,
    DEFINITION_BASE_A,
    DEFINITION_BASE_B,
    DEFINITION_BASE_C,
    DEFINITION_BASE_D,
    PLAY_A_PARAMS,
    PRIVATE_RULES_PARAMS,
    SMA_CROSS_SPEC,
    base_definitions,
    counting_registry,
    strategy_registry,
)

PARAMS = PLAY_A_PARAMS
CORE_DIGEST = vocabulary_digest(core_vocabulary)


def sorted_definitions() -> tuple[StrategyDefinition, ...]:
    return tuple(sorted(base_definitions(), key=lambda item: item.definition_digest))


def reversed_definitions() -> tuple[StrategyDefinition, ...]:
    return tuple(reversed(sorted_definitions()))


def registry(definitions) -> FrozenStrategyRegistry:
    return FrozenStrategyRegistry.from_definitions(definitions)


class _Stub(Strategy):
    """A member with no grammar, so a test can declare any dependency it likes."""

    name = "stub"

    def on_bar(self, ctx):
        return ()


def stub_definition(name: str, digest: str, *, timeframes=("1h",),
                    reference_pairs=(), members=()) -> StrategyDefinition:
    def dependencies(params):
        return StrategyDependencies(
            timeframes=timeframes, reference_pairs=reference_pairs,
            vocabulary_digest=CORE_DIGEST,
        )

    def factory(params):
        built = _Stub(dict(params))
        built.name = name
        return built

    return StrategyDefinition(
        name=name, definition_digest=digest, dependencies=dependencies,
        factory=factory, ic_factor=None, members=members,
    )


# ------------------------------------------------------- the bundle is a value


def test_registry_bundle_is_order_independent_and_factories_are_fresh():
    left = registry(definitions=reversed_definitions())
    right = registry(definitions=sorted_definitions())
    assert left.registry_digest == right.registry_digest
    definition = left.resolve("private_rules")
    assert definition.factory(PARAMS) is not definition.factory(PARAMS)


def test_the_bundle_is_hashed_in_definition_digest_order():
    """The sort is not readable off the registry, so it is asserted through
    the one thing it exists for: the digest.

    `_bundle_digest` hashes the tuple it is handed, in that tuple's order, so
    the registry's own digest equals the digest of the definitions sorted by
    `definition_digest` and NOT the digest of the supply order. The first
    assertion is what makes the second a real check: these definitions are
    supplied in an order that is neither sorted, so a constructor that skipped
    the sort would produce a different digest here.
    """
    supplied = base_definitions()
    by_digest = tuple(sorted(supplied, key=lambda item: item.definition_digest))

    assert supplied != by_digest
    assert registry(supplied).registry_digest == _bundle_digest(by_digest)
    assert registry(supplied).registry_digest != _bundle_digest(supplied)


def test_the_registry_digest_follows_a_definition_digest():
    # donchian_break, because it is the one leaf no composite lowers onto. A
    # member's digest cannot be moved on its own: the bundle refuses a tree
    # that disagrees with the definition registered under that name.
    supplied = base_definitions()
    moved = tuple(
        dataclasses.replace(item, definition_digest="7e" * 32)
        if item.name == "donchian_break" else item
        for item in supplied
    )
    assert registry(supplied).registry_digest != registry(moved).registry_digest


def test_a_member_digest_cannot_move_without_the_bundle():
    supplied = base_definitions()
    moved = tuple(
        dataclasses.replace(item, definition_digest="7e" * 32)
        if item.name == "sma_cross" else item
        for item in supplied
    )
    with pytest.raises(ReplayInputError) as caught:
        registry(moved)
    assert caught.value.code == "member_digest_mismatch"


def test_the_registry_digest_follows_a_definition_name():
    supplied = base_definitions()
    renamed = tuple(
        dataclasses.replace(item, name="donchian_breakout")
        if item.name == "donchian_break" else item
        for item in supplied
    )
    assert registry(supplied).registry_digest != registry(renamed).registry_digest


def test_the_registry_digest_follows_the_contract_version(monkeypatch):
    before = registry(base_definitions()).registry_digest
    monkeypatch.setattr(
        "nakagai.engine.registry.REGISTRY_CONTRACT_VERSION",
        REGISTRY_CONTRACT_VERSION + "-next")
    assert registry(base_definitions()).registry_digest != before


def test_the_registry_digest_follows_the_core_vocabulary(monkeypatch):
    before = registry(base_definitions()).registry_digest
    monkeypatch.setattr("nakagai.engine.registry.vocabulary_digest",
                        lambda factory: "9f" * 32)
    assert registry(base_definitions()).registry_digest != before


def test_a_house_vocabulary_digests_differently():
    def house():
        return core_vocabulary().with_terms(
            Term("house_close", "series", {}, {}, lambda series, args: series))

    assert vocabulary_digest(house) != CORE_DIGEST
    assert len(CORE_DIGEST) == 64


def test_the_frozen_registry_is_a_strategy_registry():
    assert isinstance(strategy_registry(), StrategyRegistry)


# ------------------------------------------------------------ closed bundles


def test_two_definitions_under_one_name_are_refused():
    twice = (stub_definition("clash", "1a" * 32), stub_definition("clash", "2b" * 32))
    with pytest.raises(ReplayInputError) as caught:
        registry(twice)
    assert caught.value.code == "duplicate_value"


def test_two_definitions_under_one_digest_are_refused():
    twice = (stub_definition("left", "1a" * 32), stub_definition("right", "1a" * 32))
    with pytest.raises(ReplayInputError) as caught:
        registry(twice)
    assert caught.value.code == "duplicate_value"


def test_the_constructor_is_not_an_unchecked_back_door():
    twice = (stub_definition("clash", "1a" * 32), stub_definition("clash", "2b" * 32))
    with pytest.raises(ReplayInputError) as caught:
        FrozenStrategyRegistry(twice)
    assert caught.value.code == "duplicate_value"


def test_a_malformed_definition_digest_is_refused():
    with pytest.raises(ReplayInputError) as caught:
        stub_definition("short", "abc")
    assert caught.value.code == "invalid_identifier"


def test_a_member_the_bundle_never_registered_is_refused():
    member = stub_definition("absent_member", "3c" * 32)
    parent = stub_definition("parent", "4d" * 32, members=(member,))
    with pytest.raises(ReplayInputError) as caught:
        registry((parent,))
    assert caught.value.code == "unknown_member"


def test_a_member_registered_under_another_digest_is_refused():
    captured = stub_definition("leaf", "3c" * 32)
    registered = stub_definition("leaf", "5e" * 32)
    parent = stub_definition("parent", "4d" * 32, members=(captured,))
    with pytest.raises(ReplayInputError) as caught:
        registry((parent, registered))
    assert caught.value.code == "member_digest_mismatch"


def test_a_member_cycle_is_refused_before_any_factory_runs():
    built = []

    def counting(name, digest, members=()):
        definition = stub_definition(name, digest, members=members)

        def factory(params):
            built.append(name)
            return definition.factory(params)

        return dataclasses.replace(definition, factory=factory)

    left = counting("left", "6a" * 32)
    right = counting("right", "7b" * 32, members=(left,))
    knotted = dataclasses.replace(left, members=(right,))
    with pytest.raises(ReplayInputError) as caught:
        registry((knotted, right))
    assert caught.value.code == "member_cycle"
    assert built == []


def test_resolving_a_name_the_bundle_does_not_carry_is_refused():
    with pytest.raises(ReplayInputError) as caught:
        strategy_registry().resolve("nothing_here")
    assert caught.value.code == "unknown_strategy"


# ---------------------------------------------------------- pure dependencies


def test_dependency_resolution_does_not_construct_a_strategy():
    bundle, calls = counting_registry()
    bundle.resolve("sma_cross").dependencies(PARAMS)
    assert calls.factory_count == 0


def test_composite_dependency_resolution_constructs_no_member():
    bundle, calls = counting_registry()
    bundle.resolve("combo").dependencies(COMPOSITE_PARAMS)
    assert calls.factory_count == 0
    assert calls.dependency_count == 3


def test_a_rules_definition_declares_every_timeframe_its_spec_reads():
    declared = strategy_registry().resolve("private_rules").dependencies(
        PRIVATE_RULES_PARAMS)
    # 1h is the spec's own timeframe, 4h is reached only through a node `tf`,
    # and 15m is the base clock every strategy is evaluated on.
    assert declared.timeframes == ("15m", "1h", "4h")
    assert declared.reference_pairs == ()
    assert declared.vocabulary_digest == CORE_DIGEST


def test_a_composite_unions_every_member_dependency():
    combo = strategy_registry().resolve("combo")
    declared = combo.dependencies(COMPOSITE_PARAMS)
    assert declared.timeframes == ("15m", "1h", "4h")


def test_a_composite_unions_exact_member_reference_pairs():
    leaf = stub_definition("leaf", "3c" * 32, timeframes=("1d",),
                           reference_pairs=(("spy", "1d"),))
    combo = composite_definition("combo", DEFINITION_BASE_C, members={"leaf": leaf})
    params = {"spec": {"version": 1, "name": "combo",
                       "blocks": {"a": {"strategy": "leaf", "params": {}}},
                       "long": {"all": ["a"]}}}
    declared = registry((leaf, combo)).resolve("combo").dependencies(params)
    assert declared.timeframes == ("15m", "1d")
    assert declared.reference_pairs == (("SPY", "1d"),)


def test_a_rules_definition_declares_exact_symbol_timeframe_pairs():
    spec = {
        "version": 2,
        "name": "relative_scope",
        "timeframe": "15m",
        "long": {"all": [
            {
                "lhs": {"src": "close"},
                "op": ">",
                "rhs": {"src": "close", "sym": "SPY", "tf": "15m"},
            },
            {
                "lhs": {"src": "close"},
                "op": ">",
                "rhs": {"src": "close", "sym": "QQQ", "tf": "1d"},
            },
        ]},
        "risk": {
            "stop": {"kind": "atr", "n": 14, "mult": 2.0},
            "target": {"kind": "rr", "rr": 2.0},
        },
    }
    definition = rules_definition("relative_scope", "8a" * 32, spec=spec)
    declared = definition.dependencies({})
    assert declared.timeframes == ("15m",)
    assert declared.reference_pairs == (("QQQ", "1d"), ("SPY", "15m"))
    assert ("SPY", "1d") not in declared.reference_pairs
    assert ("QQQ", "15m") not in declared.reference_pairs


def test_a_composite_refuses_a_block_naming_an_unregistered_member():
    combo = strategy_registry().resolve("combo")
    params = {"spec": {"version": 1, "name": "combo",
                       "blocks": {"a": {"strategy": "nobody", "params": {}}},
                       "long": {"all": ["a"]}}}
    with pytest.raises(ReplayInputError) as caught:
        combo.dependencies(params)
    assert caught.value.code == "unknown_member"


def test_declared_timeframes_take_the_fixed_order_and_deduplicate():
    declared = StrategyDependencies(
        timeframes=("1d", "1h", "1d", "15m"), reference_pairs=(),
        vocabulary_digest=CORE_DIGEST)
    assert declared.timeframes == ("15m", "1h", "1d")


def test_declared_reference_pairs_normalize_deduplicate_and_sort():
    declared = StrategyDependencies(
        timeframes=("1h",),
        reference_pairs=(("spy", "15m"), ("QQQ", "1d"), ("SPY", "15m")),
        vocabulary_digest=CORE_DIGEST)
    assert declared.reference_pairs == (("QQQ", "1d"), ("SPY", "15m"))


@pytest.mark.parametrize("timeframes", [("30m",), ("",), (), ("1h", "5m")])
def test_an_unsupported_or_blank_timeframe_is_refused(timeframes):
    with pytest.raises(ReplayInputError) as caught:
        StrategyDependencies(timeframes=timeframes, reference_pairs=(),
                             vocabulary_digest=CORE_DIGEST)
    assert caught.value.code == "invalid_value"


def test_a_malformed_reference_pair_is_refused():
    with pytest.raises(ReplayInputError) as caught:
        StrategyDependencies(timeframes=("1h",), reference_pairs=(("", "1d"),),
                             vocabulary_digest=CORE_DIGEST)
    assert caught.value.code == "invalid_value"


# ------------------------------------------------------------ fresh runtimes


@pytest.mark.parametrize("name, params", [
    ("sma_cross", PLAY_A_PARAMS),
    ("donchian_break", {}),
    ("private_rules", PRIVATE_RULES_PARAMS),
    ("combo", COMPOSITE_PARAMS),
])
def test_every_factory_builds_a_fresh_runtime(name, params):
    definition = strategy_registry().resolve(name)
    first, second = definition.factory(params), definition.factory(params)
    assert first is not second
    assert first.name == second.name == name
    assert first.params is not second.params


def test_a_composite_builds_fresh_members_every_call():
    definition = strategy_registry().resolve("combo")
    first, second = definition.factory(COMPOSITE_PARAMS), definition.factory(
        COMPOSITE_PARAMS)
    assert isinstance(first, CompositeStrategy)
    assert set(first._members) == {"a", "b"}
    for block, member in first._members.items():
        assert member is not second._members[block]


def test_composite_members_are_built_in_declared_block_order():
    built = strategy_registry().resolve("combo").factory(COMPOSITE_PARAMS)
    assert list(built._members) == ["b", "a"]
    assert built._members["a"].name == "sma_cross"
    assert built._members["b"].name == "private_rules"


def test_the_definition_members_are_ordered_by_name():
    combo = strategy_registry().resolve("combo")
    assert [member.name for member in combo.members] == [
        "private_rules", "sma_cross"]


def test_two_runtimes_of_one_definition_share_no_mutable_state():
    definition = strategy_registry().resolve("combo")
    first, second = definition.factory(COMPOSITE_PARAMS), definition.factory(
        COMPOSITE_PARAMS)
    first._passing["long"] = True
    first._members["a"].spec["name"] = "mutated"
    assert second._passing["long"] is False
    assert second._members["a"].spec["name"] == "sma_cross"


def test_a_runtime_cannot_write_through_into_the_supplied_params():
    definition = strategy_registry().resolve("private_rules")
    built = definition.factory(PRIVATE_RULES_PARAMS)
    built.spec["timeframe"] = "1d"
    assert PRIVATE_RULES_PARAMS["spec"]["timeframe"] == "1h"
    assert definition.factory(PRIVATE_RULES_PARAMS).spec["timeframe"] == "1h"


def test_a_rules_runtime_reads_the_spec_the_definition_binds():
    built = strategy_registry().resolve("sma_cross").factory(PLAY_A_PARAMS)
    assert isinstance(built, RuleStrategy)
    assert built.spec == SMA_CROSS_SPEC
    assert built.params["fast_n"] == 10


def test_params_cannot_replace_a_spec_the_definition_binds():
    definition = strategy_registry().resolve("sma_cross")
    with pytest.raises(ReplayInputError) as caught:
        definition.factory({"spec": {"version": 2, "name": "other"}})
    assert caught.value.code == "invalid_value"


def test_a_definition_without_a_bound_spec_is_inert_without_one():
    built = strategy_registry().resolve("private_rules").factory({})
    assert isinstance(built, RuleStrategy)
    assert built.spec == {}
    assert built.name == "private_rules"


def test_a_rule_spec_definition_is_graded_and_a_composite_is_not():
    """A composite has no single margin series, so it declares no factor.

    Grading is not a parameter of `rules_definition`: the graded margin of a
    rule tree is `spec_margin` and there is exactly one of it, so a definition
    that could be built without it would report zero observations and read as
    a lens that measured nothing rather than one that never ran.
    """
    combo = strategy_registry().resolve("combo")
    assert (combo.ic_factor, combo.ic_timeframe) == (None, None)

    graded = strategy_registry().resolve("sma_cross")
    assert graded.ic_factor is not None
    assert graded.ic_timeframe(PARAMS) == "1h"


def test_a_definition_refuses_a_factor_without_its_observation_axis():
    """The two are one thing. A margin series says nothing without its axis."""
    with pytest.raises(ReplayInputError) as caught:
        StrategyDefinition(
            name="broken", definition_digest=DEFINITION_BASE_D,
            dependencies=lambda params: None, factory=lambda params: None,
            ic_factor=lambda *args: (), ic_timeframe=None,
        )
    assert caught.value.code == "invalid_value"


def test_a_definition_refuses_an_observation_axis_without_its_factor():
    with pytest.raises(ReplayInputError) as caught:
        StrategyDefinition(
            name="broken", definition_digest=DEFINITION_BASE_D,
            dependencies=lambda params: None, factory=lambda params: None,
            ic_factor=None, ic_timeframe=lambda params: "15m",
        )
    assert caught.value.code == "invalid_value"


def test_a_definition_refuses_a_factory_that_is_not_callable():
    with pytest.raises(ReplayInputError) as caught:
        StrategyDefinition(
            name="broken", definition_digest=DEFINITION_BASE_A,
            dependencies=lambda params: None, factory="not callable",
            ic_factor=None,
        )
    assert caught.value.code == "invalid_type"


def test_a_definition_refuses_a_member_that_is_not_a_definition():
    with pytest.raises(ReplayInputError) as caught:
        StrategyDefinition(
            name="broken", definition_digest=DEFINITION_BASE_B,
            dependencies=lambda params: None, factory=lambda params: None,
            ic_factor=None, members=("sma_cross",),
        )
    assert caught.value.code == "invalid_type"
