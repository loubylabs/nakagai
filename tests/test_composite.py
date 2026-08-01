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


from nakagai.strategies.composite import validate_composite_blocks
from nakagai.strategies.rules import RuleStrategy

_LEG_SPEC = {"version": 2, "name": "rsi-leg", "timeframe": "1h",
             "long": {"all": [{"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 40}]},
             "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                      "target": {"kind": "rr", "rr": 2.0}}}

# Mirrors how load_catalog builds a catalog play: empty PARAMS, spec baked in.
_Bare = type("_Bare", (RuleStrategy,),
             {"name": "bare_play", "PARAMS": {}, "DEFAULT_PARAMS": {"spec": _LEG_SPEC}})
_MEMBERS = {"rules": RuleStrategy, "bare_play": _Bare}


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
