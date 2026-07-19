"""Composite over the example catalog: bound() is the only membership door."""

from pathlib import Path

import pytest

from nakagai.strategies.catalog import load_catalog
from nakagai.strategies.composite import CompositeStrategy, validate_composite_spec

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


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
    members = load_catalog(SPECS)
    assert validate_composite_spec(_two_member_spec(), members, allow_refs=False) == []


def test_validate_rejects_unknown_member():
    errs = validate_composite_spec(_two_member_spec(), {}, allow_refs=False)
    assert errs


def test_bound_composite_instantiates_members():
    bound = CompositeStrategy.bound(load_catalog(SPECS))
    strat = bound({"spec": _two_member_spec()})
    assert strat is not None


def test_unbound_composite_rejects_a_populated_spec():
    with pytest.raises(ValueError):
        CompositeStrategy({"spec": _two_member_spec()})
