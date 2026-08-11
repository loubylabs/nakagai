"""The catalog loader over the shipped example specs (the content slot).

One door, and it returns VALUES. `catalog_definitions` reads a directory and
hands back frozen `StrategyDefinition`s a registry bundle takes; there is no
second loader minting `RuleStrategy` subclasses beside it, so a catalog entry
cannot exist as two different types in one process.
"""

from pathlib import Path

import pytest

from nakagai.strategies.catalog import catalog_definitions, load_entries
from nakagai.strategies.rules import RuleStrategy, Term, core_vocabulary

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


def test_empty_directory_loads_empty(tmp_path):
    assert load_entries(tmp_path, core_vocabulary) == {}
    assert catalog_definitions(tmp_path, core_vocabulary) == ()


def test_a_loader_refuses_to_guess_the_vocabulary():
    # No default, on purpose. One spec read under two grammars is two
    # strategies, and `load_entries` is @cache'd on its whole argument tuple,
    # so a defaulted call and an explicit one would also be two entries over
    # one directory. The test below is the half that proves that is reachable.
    with pytest.raises(TypeError):
        load_entries(SPECS)


def _house_vocabulary():
    """Stands in for the platform's: the core's terms plus one of its own."""
    return core_vocabulary().with_terms(
        Term("house_close", "series", {}, {}, lambda series, _args: series))


def test_two_vocabularies_over_one_directory_are_two_cache_entries():
    # Why the argument is required rather than merely recommended. The cache
    # key is the whole argument tuple, so one directory read under two
    # factories is two entries. That is correct once the caller chose; a
    # default silently choosing for it is what is not.
    core = load_entries(SPECS, core_vocabulary)
    assert core is load_entries(SPECS, core_vocabulary)
    assert load_entries(SPECS, _house_vocabulary) is not core


# ------------------------------------------------- catalog as frozen values


def test_catalog_definitions_name_and_bind_every_shipped_spec():
    definitions = catalog_definitions(SPECS, core_vocabulary)
    assert {item.name for item in definitions} == {"sma_cross", "rsi_reversion",
                                                   "macd_trend"}
    for definition in definitions:
        built = definition.factory({})
        # A plain RuleStrategy, not a minted Catalog_* subclass: the name
        # comes from the definition and travels on the instance.
        assert type(built) is RuleStrategy
        assert built.name == definition.name
        assert built.spec["name"] == definition.name


def test_a_catalog_definition_builds_a_fresh_runtime_every_call():
    definition = next(item for item in catalog_definitions(SPECS, core_vocabulary)
                      if item.name == "sma_cross")
    first, second = definition.factory({}), definition.factory({})
    assert first is not second
    assert first.spec is not second.spec


def test_catalog_definition_digests_are_stable_and_distinct():
    first = {item.name: item.definition_digest
             for item in catalog_definitions(SPECS, core_vocabulary)}
    second = {item.name: item.definition_digest
              for item in catalog_definitions(SPECS, core_vocabulary)}
    assert first == second
    assert len(set(first.values())) == len(first)


def test_a_house_vocabulary_gives_every_catalog_definition_another_digest():
    # The same reason the loaders refuse to guess a vocabulary: one spec read
    # under two grammars is two strategies, so it cannot be one digest.
    core = {item.name: item.definition_digest
            for item in catalog_definitions(SPECS, core_vocabulary)}
    house = {item.name: item.definition_digest
             for item in catalog_definitions(SPECS, _house_vocabulary)}
    assert core.keys() == house.keys()
    assert all(core[name] != house[name] for name in core)


def test_a_catalog_definition_carries_the_grammar_it_was_read_under():
    """The definition records its own grammar, so the replay reads one answer.

    The factory validates a spec under it, the IC lens grades under it, and the
    context a runtime decides through is built from it. A definition that did
    not carry it would leave the replay to guess, which is how a play comes to
    decide under one grammar and be graded under another.
    """
    for definition in catalog_definitions(SPECS, _house_vocabulary):
        assert definition.vocabulary_factory is _house_vocabulary
        # The built runtime speaks it too, which is what makes the field the
        # definition's one answer rather than a second, unrelated declaration.
        assert "house_close" in definition.factory({}).vocabulary.indicators


def test_a_catalog_definition_declares_the_frames_its_spec_reads():
    declared = {item.name: item.dependencies({}).timeframes
                for item in catalog_definitions(SPECS, core_vocabulary)}
    entries = load_entries(SPECS, core_vocabulary)
    for name, timeframes in declared.items():
        assert "15m" in timeframes
        assert entries[name]["spec"]["timeframe"] in timeframes
