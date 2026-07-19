"""The catalog loader over the shipped example specs (the content slot)."""

from pathlib import Path

from nakagai.strategies.catalog import load_catalog, load_entries
from nakagai.strategies.rules import RuleStrategy

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


def test_empty_directory_loads_empty(tmp_path):
    assert load_entries(tmp_path) == {}
    assert load_catalog(tmp_path) == {}


def test_example_specs_load_as_strategies():
    catalog = load_catalog(SPECS)
    assert set(catalog) == {"sma_cross", "rsi_reversion", "macd_trend"}
    for cls in catalog.values():
        assert issubclass(cls, RuleStrategy)


def test_example_strategies_instantiate():
    for name, cls in load_catalog(SPECS).items():
        strat = cls({})
        assert strat.name == name
