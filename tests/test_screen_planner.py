"""Cheap ScreenSpec planning over current discovery facts."""

from nakagai.screen.planner import (
    PlannedSymbol,
    plan_symbol,
    uses_facts,
    uses_technical,
)


def _low_float():
    return {
        "lhs": {"fact": "float_shares"},
        "op": "<",
        "rhs": 20_000_000,
    }


def _rsi_under_30():
    return {"lhs": {"ind": "rsi", "n": 14}, "op": "<", "rhs": 30}


def test_false_fact_prunes_an_all_group_before_technical_work():
    group = {"all": [_low_float(), _rsi_under_30()]}
    planned = plan_symbol(group, {"float_shares": 50_000_000})
    assert planned == PlannedSymbol(False, False, ())


def test_true_fact_settles_an_any_group_without_bars():
    group = {"any": [_low_float(), _rsi_under_30()]}
    planned = plan_symbol(group, {"float_shares": 5_000_000})
    assert planned == PlannedSymbol(True, False, ())


def test_missing_fact_stays_unknown_without_becoming_zero():
    planned = plan_symbol({"all": [_low_float()]}, {})
    assert planned == PlannedSymbol(None, False, ("float_shares",))


def test_non_finite_fact_stays_unknown():
    planned = plan_symbol({"all": [_low_float()]}, {"float_shares": float("nan")})
    assert planned == PlannedSymbol(None, False, ("float_shares",))


def test_technical_only_group_requests_technical_evaluation():
    planned = plan_symbol({"all": [_rsi_under_30()]}, {})
    assert planned == PlannedSymbol(None, True, ())


def test_nested_not_uses_three_valued_negation():
    group = {"not": {"all": [_low_float()]}}
    assert plan_symbol(group, {"float_shares": 50_000_000}) == PlannedSymbol(
        True, False, ())
    assert plan_symbol(group, {}) == PlannedSymbol(
        None, False, ("float_shares",))


def test_fact_math_is_resolved_in_the_cheap_pass():
    group = {"all": [{
        "lhs": {"op": "/", "args": [
            {"fact": "market_cap"}, {"fact": "shares_outstanding"},
        ]},
        "op": "<",
        "rhs": 10,
    }]}
    planned = plan_symbol(
        group, {"market_cap": 90_000_000, "shares_outstanding": 10_000_000})
    assert planned == PlannedSymbol(True, False, ())


def test_fact_division_by_zero_is_a_settled_non_match():
    group = {"all": [{
        "lhs": {"op": "/", "args": [
            {"fact": "market_cap"}, 0,
        ]},
        "op": ">",
        "rhs": 1,
    }]}
    planned = plan_symbol(group, {"market_cap": 90_000_000})
    assert planned == PlannedSymbol(False, False, ())


def test_non_finite_literal_is_a_settled_non_match():
    group = {"all": [{
        "lhs": float("inf"),
        "op": ">",
        "rhs": 1,
    }]}
    assert plan_symbol(group, {}) == PlannedSymbol(False, False, ())


def test_missing_fact_names_are_sorted_and_unique():
    group = {"all": [
        _low_float(),
        {"lhs": {"fact": "market_cap"}, "op": ">", "rhs": 1},
        {"lhs": {"fact": "float_shares"}, "op": ">", "rhs": 1},
    ]}
    assert plan_symbol(group, {}) == PlannedSymbol(
        None, False, ("float_shares", "market_cap"))


def test_usage_classifiers_distinguish_fact_and_technical_trees():
    mixed = {"all": [_low_float(), _rsi_under_30()]}
    assert uses_facts(mixed) is True
    assert uses_technical(mixed) is True
    assert uses_facts({"all": [_rsi_under_30()]}) is False
    assert uses_technical({"all": [_low_float()]}) is False
