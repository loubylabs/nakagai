"""The catalog loader over the shipped example specs (the content slot)."""

from pathlib import Path

import pytest

from nakagai.strategies.catalog import (
    catalog_definitions, load_catalog, load_entries,
)
from nakagai.strategies.rules import RuleStrategy, Term, core_vocabulary

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


def test_empty_directory_loads_empty(tmp_path):
    assert load_entries(tmp_path, core_vocabulary) == {}
    assert load_catalog(tmp_path, core_vocabulary) == {}


def test_example_specs_load_as_strategies():
    catalog = load_catalog(SPECS, core_vocabulary)
    assert set(catalog) == {"sma_cross", "rsi_reversion", "macd_trend"}
    for cls in catalog.values():
        assert issubclass(cls, RuleStrategy)


def test_example_strategies_instantiate():
    for name, cls in load_catalog(SPECS, core_vocabulary).items():
        strat = cls({})
        assert strat.name == name


@pytest.mark.parametrize("load", (load_entries, load_catalog))
def test_a_loader_refuses_to_guess_the_vocabulary(load):
    # No default, on purpose. Both loaders are @cache'd on their argument
    # tuple, so a defaulted call and an explicit one would be two entries over
    # the same directory: load_catalog would hand back two different
    # `Catalog_sma_cross` classes in one process, and nothing would raise. The
    # test below is the half that proves the caching makes that reachable.
    with pytest.raises(TypeError):
        load(SPECS)


def _house_vocabulary():
    """Stands in for the platform's: the core's terms plus one of its own."""
    return core_vocabulary().with_terms(
        Term("house_close", "series", {}, {}, lambda series, _args: series))


def test_two_vocabularies_over_one_directory_are_two_caches():
    # Why the argument is required rather than merely recommended. The cache
    # key is the whole argument tuple, so one directory read under two
    # factories is two entries and two `Catalog_sma_cross` classes, neither of
    # which is an instance of the other. That is correct once the caller chose;
    # a default silently choosing for it is what is not.
    core = load_catalog(SPECS, core_vocabulary)
    assert core is load_catalog(SPECS, core_vocabulary)
    house = load_catalog(SPECS, _house_vocabulary)
    assert house is not core
    assert house["sma_cross"] is not core["sma_cross"]
    assert not issubclass(house["sma_cross"], core["sma_cross"])


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


def test_a_catalog_definition_declares_the_frames_its_spec_reads():
    declared = {item.name: item.dependencies({}).timeframes
                for item in catalog_definitions(SPECS, core_vocabulary)}
    entries = load_entries(SPECS, core_vocabulary)
    for name, timeframes in declared.items():
        assert "15m" in timeframes
        assert entries[name]["spec"]["timeframe"] in timeframes
