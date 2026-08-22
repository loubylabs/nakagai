"""Every number a spec fixes becomes one typed, bounded, stably named input.

The name is the RuleSpec path, so a chart's saved settings survive a
recompilation of the same strategy; the type and the bounds come from the term
that declared the argument, so no input can be set to a value the engine would
refuse.
"""

import numpy as np
import pytest

from nakagai.strategies.rules import (
    PineCompileError, PineExpr, PineInput, PineLowering, lower_pine,
    validate_spec,
)
from nakagai.strategies.rules.vocabulary import Term, core_vocabulary, is_choice_rule


def _spec(lhs, op=">", rhs=0, **extra):
    return {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": lhs, "op": op, "rhs": rhs}]}, **extra}


def _by_name(program):
    return {item.name: item for item in program.inputs}


def _emit_close_with_n(ctx, call):
    return PineExpr(ctx.calc(call, f"close + {ctx.arg(call, 'n')} * 0"))


def test_numeric_rule_values_become_stable_typed_inputs(load_spec):
    # The sma input is reached by both sides of sma_cross, so its label drops
    # the block name; the risk input is reached by the risk block alone and
    # keeps it. See test_a_label_names_its_block_only_when_one_block_owns_it.
    program = lower_pine(load_spec("sma_cross"))
    assert PineInput(
        name="nk_long_all_0_lhs_sma_n", label="lhs · sma · n",
        kind="int", default=20, bounds=(2, 500),
    ) in program.inputs
    assert PineInput(
        name="nk_risk_stop_mult", label="Risk · stop · mult",
        kind="float", default=2.0, bounds=(0.1, 10.0),
    ) in program.inputs


def test_the_declared_bounds_decide_int_against_float():
    # `==` reads 2 and 2.0 as equal, so the types are asserted directly: an
    # input.int with float bounds does not compile on TradingView.
    inputs = _by_name(lower_pine(_spec({"ind": "bb", "field": "upper"})))
    n = inputs["nk_long_all_0_lhs_bb_n"]
    assert n.kind == "int"
    assert isinstance(n.default, int) and n.bounds == (2, 200)
    assert all(isinstance(bound, int) for bound in n.bounds)
    k = inputs["nk_long_all_0_lhs_bb_k"]
    assert k.kind == "float"
    assert isinstance(k.default, float) and k.bounds == (0.5, 5.0)
    assert all(isinstance(bound, float) for bound in k.bounds)


@pytest.mark.parametrize("bounds,default,kind,cast_bounds", [
    ((np.int64(2), np.int64(100)), 30, "int", (2, 100)),
    ((np.float64(0.5), np.float64(5.0)), 2.0, "float", (0.5, 5.0)),
], ids=["numpy-int-bounds", "numpy-float-bounds"])
def test_numpy_real_bounds_lower_to_typed_inputs(bounds, default, kind, cast_bounds):
    vocabulary = core_vocabulary().with_terms(
        Term("numpy_bounds", "series", {"n": bounds}, {"n": default},
             lambda series, _args: series, pine=PineLowering(_emit_close_with_n)))
    spec = _spec({"ind": "numpy_bounds"})
    assert validate_spec(spec, vocabulary) == []
    input_ = _by_name(lower_pine(spec, vocabulary))["nk_long_all_0_lhs_numpy_bounds_n"]
    assert input_.kind == kind
    assert input_.default == default
    assert input_.bounds == cast_bounds


@pytest.mark.parametrize("bounds", [
    (-(10 ** 400), 10 ** 400),
    (float("-inf"), float("inf")),
], ids=["huge", "infinite"])
def test_invalid_numeric_bounds_stop_pine_as_invalid_spec(bounds):
    vocabulary = core_vocabulary().with_terms(
        Term("bad_bounds", "series", {"n": bounds}, {"n": 30},
             lambda series, _args: series,
             pine=PineLowering(_emit_close_with_n)))
    spec = _spec({"ind": "bad_bounds"})
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec, vocabulary)
    assert exc.value.code == "invalid_spec"


@pytest.mark.parametrize("value", [9007199254740993, -9007199254740993],
                         ids=["positive", "negative"])
def test_unroundtrippable_integer_stops_pine_as_invalid_spec(value):
    vocabulary = core_vocabulary().with_terms(
        Term("large_numeric", "series", {"n": (-(10 ** 20), 10 ** 20)},
             {"n": 30}, lambda series, _args: series,
             pine=PineLowering(_emit_close_with_n)))
    spec = _spec({"ind": "large_numeric", "n": value})
    with pytest.raises(PineCompileError) as exc:
        lower_pine(spec, vocabulary)
    assert exc.value.code == "invalid_spec"


def test_an_integral_float_lowers_exactly_like_the_integer_it_equals():
    # Both spell the same canonical spec, so both owe the same program.
    assert lower_pine(_spec({"ind": "sma", "n": 20.0})) == \
        lower_pine(_spec({"ind": "sma", "n": 20}))


def test_a_bare_threshold_becomes_an_unbounded_input():
    program = lower_pine(_spec({"ind": "rsi"}, ">", 70))
    assert PineInput("nk_long_all_0_rhs", "Long · rhs", "int", 70) in program.inputs
    program = lower_pine(_spec({"ind": "rsi"}, ">", 70.5))
    assert PineInput("nk_long_all_0_rhs", "Long · rhs", "float", 70.5) \
        in program.inputs


@pytest.mark.parametrize("name", sorted(core_vocabulary().indicators))
def test_every_numeric_argument_of_an_indicator_reaches_one_input(name):
    term = core_vocabulary().indicators[name]
    program = lower_pine(_spec({"ind": name}))
    names = {item.name for item in program.inputs}
    for arg, rule in term.args.items():
        if is_choice_rule(rule):
            continue
        assert f"nk_long_all_0_lhs_{name}_{arg}" in names


def test_inputs_follow_the_source_order_of_the_spec(load_spec):
    program = lower_pine(load_spec("rsi_reversion"))
    assert [item.name for item in program.inputs] == [
        "nk_long_all_0_lhs_rsi_n", "nk_long_all_0_rhs",
        "nk_long_all_1_rhs_sma_n", "nk_short_all_0_rhs",
        "nk_risk_stop_n", "nk_risk_stop_mult", "nk_risk_target_rr"]


def test_identical_nodes_share_the_input_their_first_path_named(load_spec):
    # sma_cross reads sma(20) and sma(50) on both sides. Four knobs would mean
    # two of them silently doing nothing to the other side's calculation.
    program = lower_pine(load_spec("sma_cross"))
    assert [item.name for item in program.inputs] == [
        "nk_long_all_0_lhs_sma_n", "nk_long_all_0_rhs_sma_n",
        "nk_risk_stop_n", "nk_risk_stop_mult", "nk_risk_target_rr"]


def test_labels_drop_the_structural_parts_of_the_path():
    program = lower_pine(_spec({"ind": "sma"}, ">", {"ind": "ema"}))
    labels = {item.name: item.label for item in program.inputs}
    assert labels["nk_long_all_0_lhs_sma_n"] == "Long · lhs · sma · n"
    assert labels["nk_long_all_0_rhs_ema_n"] == "Long · rhs · ema · n"


def test_a_label_names_its_block_only_when_one_block_owns_it(load_spec):
    # sma_cross's two sides read the same pair of moving averages, so one input
    # drives both. Heading its row "Long" would tell a chart's user that moving
    # it retunes the long side, when it retunes the short side with it.
    labels = {item.name: item.label
              for item in lower_pine(load_spec("sma_cross")).inputs}
    assert labels["nk_long_all_0_lhs_sma_n"] == "lhs · sma · n"
    assert labels["nk_long_all_0_rhs_sma_n"] == "rhs · sma · n"
    # Reached by the risk block only, so it keeps its block name.
    assert labels["nk_risk_stop_n"] == "Risk · stop · n"
    assert not any(label.startswith("Long") or label.startswith("Short")
                   for label in labels.values())


def test_a_shared_argument_nested_under_of_also_drops_its_block(load_spec):
    # The sharing that drops a block name is the node memo's, so it reaches
    # every input under the node, not only the outermost term's own arguments.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma", "n": 10,
                                      "of": {"ind": "ema", "n": 5}},
                              "op": ">", "rhs": 0}]},
            "short": {"all": [{"lhs": {"ind": "sma", "n": 10,
                                       "of": {"ind": "ema", "n": 5}},
                               "op": "<", "rhs": 0}]}}
    labels = {item.name: item.label for item in lower_pine(spec).inputs}
    assert labels["nk_long_all_0_lhs_sma_n"] == "lhs · sma · n"
    assert labels["nk_long_all_0_lhs_of_ema_n"] == "lhs · of · ema · n"
    # The bare thresholds are one per side, so they keep their block names.
    assert labels["nk_long_all_0_rhs"] == "Long · rhs"
    assert labels["nk_short_all_0_rhs"] == "Short · rhs"


def test_labels_take_the_index_back_when_two_of_them_would_read_alike():
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [
                {"lhs": {"ind": "sma", "n": 20}, "op": ">", "rhs": 0},
                {"lhs": {"ind": "sma", "n": 50}, "op": ">", "rhs": 0},
            ]}}
    labels = [item.label for item in lower_pine(spec).inputs]
    assert len(labels) == len(set(labels))
    assert "Long · 0 · lhs · sma · n" in labels
    assert "Long · 1 · lhs · sma · n" in labels


def test_risk_inputs_carry_the_bounds_the_validator_enforces():
    program = lower_pine(_spec({"ind": "sma"},
                               risk={"stop": {"kind": "percent", "pct": 1.5},
                                     "target": {"kind": "percent", "pct": 3.0}}))
    assert PineInput("nk_risk_stop_pct", "Risk · stop · pct", "float", 1.5,
                     (0.05, 50.0)) in program.inputs
    assert PineInput("nk_risk_target_pct", "Risk · target · pct", "float", 3.0,
                     (0.05, 100.0)) in program.inputs
    program = lower_pine(_spec({"ind": "sma"}))
    assert PineInput("nk_risk_stop_n", "Risk · stop · n", "int", 14,
                     (2, 100)) in program.inputs
    assert PineInput("nk_risk_target_rr", "Risk · target · rr", "float", 2.0,
                     (0.1, 20.0)) in program.inputs


def test_exit_inputs_carry_the_bounds_the_validator_enforces():
    program = lower_pine(_spec(
        {"ind": "sma"},
        exits={"trailing": {"kind": "atr", "n": 10, "mult": 3.0},
               "time_stop": {"bars": 8}, "breakeven_at": {"rr": 1.0}}))
    assert PineInput("nk_exits_trailing_n", "Exits · trailing · n", "int", 10,
                     (2, 100)) in program.inputs
    assert PineInput("nk_exits_trailing_mult", "Exits · trailing · mult",
                     "float", 3.0, (0.5, 10.0)) in program.inputs
    assert PineInput("nk_exits_time_stop_bars", "Exits · time_stop · bars",
                     "int", 8, (1, 500)) in program.inputs
    assert PineInput("nk_exits_breakeven_at_rr", "Exits · breakeven_at · rr",
                     "float", 1.0, (0.1, 10.0)) in program.inputs


def test_an_argument_nested_under_of_is_named_by_its_own_path():
    program = lower_pine(_spec({"ind": "sma", "n": 10,
                                "of": {"ind": "ema", "n": 5}}))
    names = [item.name for item in program.inputs]
    assert "nk_long_all_0_lhs_of_ema_n" in names
    assert names.index("nk_long_all_0_lhs_of_ema_n") < \
        names.index("nk_long_all_0_lhs_sma_n")
