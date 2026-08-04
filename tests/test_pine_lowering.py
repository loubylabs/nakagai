"""The RuleSpec expression language lowered into one deterministic Pine program.

Everything here reads a PineProgram, never a rendered artifact. The program is
target-neutral by contract, so no test in this file may expect an `indicator()`
or a `strategy()` statement; those belong to the renderers.
"""

import json
import os
import subprocess
import sys

import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules import (
    PineExits, PineRisk, lower_pine, spec_hash,
)
from nakagai.strategies.rules.pine.lower import PINE_TIMEFRAMES, SpecLowerer
from nakagai.strategies.rules.spec import (
    DRIVING,
    STOP_ATR_MULT_DEFAULT, STOP_ATR_N_DEFAULT, STOP_PCT_DEFAULT,
    TARGET_PCT_DEFAULT, TARGET_RR_DEFAULT, TRAILING_ATR_MULT_DEFAULT,
    TRAILING_ATR_N_DEFAULT, TRAILING_PCT_DEFAULT,
)
from nakagai.strategies.rules.vocabulary import core_vocabulary, is_choice_rule

# The path prefix every operand in a one-condition long group carries.
LHS = "nk_long_all_0_lhs"


def _spec(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return {"version": 2, "name": "probe", "timeframe": timeframe,
            "long": {"all": [{"lhs": lhs, "op": op, "rhs": rhs}]}, **extra}


def _program(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return lower_pine(_spec(lhs, op, rhs, timeframe, **extra))


def _lines(program):
    """Every statement the program emits, function bodies opened up.

    A play whose timeframe is not the chart's puts the calculations behind its
    conditions inside ONE request function, which reaches PineProgram as a
    single multi-line statement. A test asking what was calculated has to look
    inside it, or it would only ever see 15m plays.
    """
    return [line.strip() for text in program.calculations
            for line in text.splitlines()]


def _midline(n):
    return f"(ta.highest(high, {n}) + ta.lowest(low, {n})) / 2"


# One row per indicator: the node, the calculations it owes, and the identifier
# the condition reads. The table is asserted complete against the vocabulary
# below, so a new indicator without a Pine form fails here rather than silently.
INDICATORS = [
    ({"ind": "sma"}, [f"nk_sma_1 = ta.sma(close, {LHS}_sma_n)"], "nk_sma_1"),
    ({"ind": "ema"}, [f"nk_ema_1 = ta.ema(close, {LHS}_ema_n)"], "nk_ema_1"),
    ({"ind": "rsi"}, [f"nk_rsi_1 = ta.rsi(close, {LHS}_rsi_n)"], "nk_rsi_1"),
    ({"ind": "roc"}, [f"nk_roc_1 = ta.roc(close, {LHS}_roc_n)"], "nk_roc_1"),
    ({"ind": "zscore"},
     [f"nk_zscore_1 = nk_div(close - ta.sma(close, {LHS}_zscore_n), "
      f"ta.stdev(close, {LHS}_zscore_n))"], "nk_zscore_1"),
    ({"ind": "highest"}, [f"nk_highest_1 = ta.highest(close, {LHS}_highest_n)"],
     "nk_highest_1"),
    ({"ind": "lowest"}, [f"nk_lowest_1 = ta.lowest(close, {LHS}_lowest_n)"],
     "nk_lowest_1"),
    ({"ind": "stdev"}, [f"nk_stdev_1 = ta.stdev(close, {LHS}_stdev_n)"],
     "nk_stdev_1"),
    ({"ind": "macd", "field": "hist"},
     [f"[nk_macd_1_macd, nk_macd_1_signal, nk_macd_1_hist] = "
      f"ta.macd(close, {LHS}_macd_fast, {LHS}_macd_slow, {LHS}_macd_signal)"],
     "nk_macd_1_hist"),
    ({"ind": "bb", "field": "upper"},
     [f"[nk_bb_1_mid, nk_bb_1_upper, nk_bb_1_lower] = "
      f"ta.bb(close, {LHS}_bb_n, {LHS}_bb_k)"], "nk_bb_1_upper"),
    ({"ind": "atr"}, [f"nk_atr_1 = ta.atr({LHS}_atr_n)"], "nk_atr_1"),
    # donchian excludes the current bar, exactly as indicators.donchian does
    # with its .shift(1); in Pine that is the [1] on each channel edge.
    ({"ind": "donchian", "field": "mid"},
     [f"nk_donchian_1_upper = ta.highest(high, {LHS}_donchian_n)[1]",
      f"nk_donchian_1_lower = ta.lowest(low, {LHS}_donchian_n)[1]",
      "nk_donchian_1_mid = (nk_donchian_1_upper + nk_donchian_1_lower) / 2"],
     "nk_donchian_1_mid"),
    # ta.supertrend reports -1 for an up-trend where indicators.supertrend
    # reports +1, so the direction field is negated rather than passed through.
    ({"ind": "supertrend", "field": "direction"},
     [f"[nk_supertrend_1_line, nk_supertrend_1_raw_direction] = "
      f"ta.supertrend({LHS}_supertrend_mult, {LHS}_supertrend_n)",
      "nk_supertrend_1_direction = -nk_supertrend_1_raw_direction"],
     "nk_supertrend_1_direction"),
    ({"ind": "vwap"}, ["nk_vwap_1 = ta.vwap"], "nk_vwap_1"),
    ({"ind": "stoch", "field": "d"},
     [f"nk_stoch_1_k = ta.stoch(close, high, low, {LHS}_stoch_n)",
      f"nk_stoch_1_d = ta.sma(nk_stoch_1_k, {LHS}_stoch_d)"], "nk_stoch_1_d"),
    ({"ind": "adx"},
     [f"[nk_adx_1_di_plus, nk_adx_1_di_minus, nk_adx_1_adx] = "
      f"ta.dmi({LHS}_adx_n, {LHS}_adx_n)"], "nk_adx_1_adx"),
    ({"ind": "obv"}, ["nk_obv_1 = ta.obv"], "nk_obv_1"),
    ({"ind": "ichimoku", "field": "senkou_b"},
     [f"nk_ichimoku_1_tenkan = {_midline(f'{LHS}_ichimoku_tenkan_n')}",
      f"nk_ichimoku_1_kijun = {_midline(f'{LHS}_ichimoku_kijun_n')}",
      "nk_ichimoku_1_senkou_a_base = "
      "(nk_ichimoku_1_tenkan + nk_ichimoku_1_kijun) / 2",
      f"nk_ichimoku_1_senkou_a = "
      f"nk_ichimoku_1_senkou_a_base[{LHS}_ichimoku_disp]",
      f"nk_ichimoku_1_senkou_b_base = {_midline(f'{LHS}_ichimoku_senkou_n')}",
      f"nk_ichimoku_1_senkou_b = "
      f"nk_ichimoku_1_senkou_b_base[{LHS}_ichimoku_disp]"],
     "nk_ichimoku_1_senkou_b"),
    ({"ind": "keltner", "field": "lower"},
     [f"nk_keltner_1_mid = ta.ema(close, {LHS}_keltner_n)",
      f"nk_keltner_1_band = {LHS}_keltner_mult * ta.atr({LHS}_keltner_n)",
      "nk_keltner_1_upper = nk_keltner_1_mid + nk_keltner_1_band",
      "nk_keltner_1_lower = nk_keltner_1_mid - nk_keltner_1_band"],
     "nk_keltner_1_lower"),
    ({"ind": "cci"}, [f"nk_cci_1 = ta.cci(hlc3, {LHS}_cci_n)"], "nk_cci_1"),
    ({"ind": "mfi"}, [f"nk_mfi_1 = ta.mfi(hlc3, {LHS}_mfi_n)"], "nk_mfi_1"),
    ({"ind": "wpr"}, [f"nk_wpr_1 = ta.wpr({LHS}_wpr_n)"], "nk_wpr_1"),
]


@pytest.mark.parametrize("node, statements, operand", INDICATORS,
                         ids=[row[0]["ind"] for row in INDICATORS])
def test_every_indicator_lowers_to_its_pine_form(node, statements, operand):
    program = _program(node)
    for statement in statements:
        assert statement in program.calculations
    assert (f"nk_long_entry = ({operand} > nk_long_all_0_rhs)"
            in program.calculations)


def test_the_indicator_table_covers_every_indicator_in_the_vocabulary():
    covered = {node["ind"] for node, _, _ in INDICATORS}
    assert covered == set(core_vocabulary().indicators)


# One row per (indicator, field) the grammar admits. The table above lowers one
# field per multi-field indicator; this one lowers all of them, so a mapping
# that answered only the fields a test happened to ask for is caught here.
FIELDS = {
    ("macd", "macd"): "nk_macd_1_macd",
    ("macd", "signal"): "nk_macd_1_signal",
    ("macd", "hist"): "nk_macd_1_hist",
    ("bb", "upper"): "nk_bb_1_upper",
    ("bb", "mid"): "nk_bb_1_mid",
    ("bb", "lower"): "nk_bb_1_lower",
    ("donchian", "upper"): "nk_donchian_1_upper",
    ("donchian", "lower"): "nk_donchian_1_lower",
    ("donchian", "mid"): "nk_donchian_1_mid",
    ("supertrend", "line"): "nk_supertrend_1_line",
    ("supertrend", "direction"): "nk_supertrend_1_direction",
    ("stoch", "k"): "nk_stoch_1_k",
    ("stoch", "d"): "nk_stoch_1_d",
    ("ichimoku", "tenkan"): "nk_ichimoku_1_tenkan",
    ("ichimoku", "kijun"): "nk_ichimoku_1_kijun",
    ("ichimoku", "senkou_a"): "nk_ichimoku_1_senkou_a",
    ("ichimoku", "senkou_b"): "nk_ichimoku_1_senkou_b",
    ("keltner", "upper"): "nk_keltner_1_upper",
    ("keltner", "mid"): "nk_keltner_1_mid",
    ("keltner", "lower"): "nk_keltner_1_lower",
}


@pytest.mark.parametrize("pair, operand", sorted(FIELDS.items()),
                         ids=[f"{i}.{f}" for i, f in sorted(FIELDS)])
def test_every_declared_field_of_every_indicator_reads_its_own_identifier(
        pair, operand):
    indicator, field = pair
    program = _program({"ind": indicator, "field": field})
    assert (f"nk_long_entry = ({operand} > nk_long_all_0_rhs)"
            in program.calculations)


def test_the_field_table_covers_every_field_choice_in_the_vocabulary():
    declared = {(name, field)
                for name, term in core_vocabulary().indicators.items()
                for rule in [term.args.get("field")] if is_choice_rule(rule)
                for field in rule}
    assert declared == set(FIELDS)


def test_sources_and_numbers_lower_to_plain_operands():
    program = _program({"src": "high"}, "<", {"src": "low"})
    assert "nk_long_entry = (high < low)" in program.calculations


def test_a_field_is_selected_from_one_tuple_calculation(load_spec):
    # macd_trend reads the macd line against its signal line: two operands, one
    # ta.macd. A second calculation would be a second, drifting MACD.
    program = lower_pine(load_spec("macd_trend"))
    tuples = [c for c in _lines(program) if "ta.macd(" in c]
    assert len(tuples) == 1
    assert ("bool nk_long_entry_native = "
            "(ta.crossover(nk_macd_1_macd, nk_macd_1_signal))"
            in _lines(program))


def test_crosses_lower_to_crossover_and_crossunder():
    above = _program({"ind": "rsi"}, "crosses_above", 30)
    below = _program({"ind": "rsi"}, "crosses_below", 70)
    assert ("nk_long_entry = (ta.crossover(nk_rsi_1, nk_long_all_0_rhs))"
            in above.calculations)
    assert ("nk_long_entry = (ta.crossunder(nk_rsi_1, nk_long_all_0_rhs))"
            in below.calculations)


def test_group_trees_lower_to_fully_parenthesized_boolean_expressions():
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [
                {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}},
                {"any": [
                    {"lhs": {"src": "high"}, "op": ">", "rhs": {"src": "open"}},
                    {"lhs": {"src": "low"}, "op": "<", "rhs": {"src": "open"}},
                ]},
            ]}}
    program = lower_pine(spec)
    # Comparisons bind tighter than `and`/`or` in Pine, so a condition needs no
    # parentheses of its own; every GROUP gets its own, at every depth.
    assert ("nk_long_entry = (close > open and (high > open or low < open))"
            in program.calculations)


def test_math_ops_lower_with_explicit_parentheses():
    node = {"op": "+", "args": [{"src": "high"}, {"src": "low"}, {"src": "close"}]}
    assert ("nk_long_entry = ((high + low + close) > nk_long_all_0_rhs)"
            in _program(node).calculations)
    minus = {"op": "-", "args": [{"src": "high"}, {"src": "low"}]}
    assert ("nk_long_entry = ((high - low) > nk_long_all_0_rhs)"
            in _program(minus).calculations)
    times = {"op": "*", "args": [{"src": "close"}, 2]}
    assert ("nk_long_entry = ((close * nk_long_all_0_lhs_args_1) > "
            "nk_long_all_0_rhs)" in _program(times).calculations)


def test_abs_lowers_to_math_abs():
    node = {"op": "abs", "args": [{"op": "-", "args": [{"src": "close"},
                                                       {"src": "open"}]}]}
    assert ("nk_long_entry = (math.abs((close - open)) > nk_long_all_0_rhs)"
            in _program(node).calculations)


def test_variadic_min_and_max_fold_left():
    node = {"op": "max", "args": [{"src": "high"}, {"src": "low"},
                                  {"src": "close"}]}
    assert ("nk_long_entry = (math.max(math.max(high, low), close) > "
            "nk_long_all_0_rhs)" in _program(node).calculations)
    node = {"op": "min", "args": [{"src": "high"}, {"src": "low"},
                                  {"src": "close"}]}
    assert ("nk_long_entry = (math.min(math.min(high, low), close) > "
            "nk_long_all_0_rhs)" in _program(node).calculations)


def test_division_goes_through_the_zero_safe_helper():
    # frame_eval._math maps a zero denominator to NaN rather than raising or
    # producing an infinity. The helper has to read the same way.
    node = {"op": "/", "args": [{"src": "close"}, {"src": "volume"}]}
    program = _program(node)
    assert ("nk_long_entry = (nk_div(close, volume) > nk_long_all_0_rhs)"
            in program.calculations)
    assert [h.id for h in program.helpers] == ["nk_div"]
    assert program.helpers[0].source == "nk_div(a, b) => b == 0 ? na : a / b"


def test_a_program_without_division_carries_no_helper():
    assert _program({"ind": "sma"}).helpers == ()


def test_timeframe_strings_cover_every_timeframe_the_engine_admits():
    assert PINE_TIMEFRAMES == {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    assert set(PINE_TIMEFRAMES) == set(DEFAULT_TIMEFRAMES.all)


def test_a_foreign_timeframe_operand_is_requested_gated_and_latched():
    # The whole visibility rule is in these four statements together. The
    # request carries no offset, which is only honest because the gate under it
    # fires where the requested bar CLOSES; the latch is what stops any other
    # bar reading that unoffset value, which off the gate is future data.
    program = _program({"ind": "sma", "n": 20, "tf": "1h"})
    assert ("nk_htf_1() =>\n"
            f"    nk_sma_1 = ta.sma(close, {LHS}_sma_n)\n"
            "    nk_sma_1") in program.calculations
    assert 'nk_visible_60 = time_close("60") == time_close' in program.calculations
    assert ('nk_sma_2_raw = request.security(syminfo.tickerid, "60", '
            "nk_htf_1(), lookahead=barmerge.lookahead_on, "
            "gaps=barmerge.gaps_off)") in program.calculations
    assert "var float nk_sma_2 = na" in program.calculations
    assert "if nk_visible_60\n    nk_sma_2 := nk_sma_2_raw" in program.calculations
    # Every consumer reads the latch, never the raw read. That is the property
    # a later edit could quietly break, so it is asserted over the whole
    # program rather than over the one line that happens to read it today.
    readers = [c for c in _lines(program) if "nk_sma_2_raw" in c]
    assert readers == ['nk_sma_2_raw = request.security(syminfo.tickerid, "60", '
                       "nk_htf_1(), lookahead=barmerge.lookahead_on, "
                       "gaps=barmerge.gaps_off)", "nk_sma_2 := nk_sma_2_raw"]
    assert "nk_long_entry = (nk_sma_2 > nk_long_all_0_rhs)" in program.calculations


def test_a_session_aligned_reference_is_read_one_bar_back_instead():
    # The offset is not decoration, it is the other half of the gate. A daily
    # gate fires where the new New York day OPENS, and the daily bar that day
    # belongs to has not happened yet, so the confirmed value there is the
    # previous one. closed_before hands the engine exactly that on the same bar.
    program = _program({"ind": "sma", "n": 20, "tf": "1d"})
    assert ("nk_htf_1() =>\n"
            f"    nk_sma_1 = ta.sma(close, {LHS}_sma_n)\n"
            "    nk_sma_1[1]") in program.calculations
    assert "nk_visible_d = nk_new_session()" in program.calculations
    assert "if nk_visible_d\n    nk_sma_2 := nk_sma_2_raw" in program.calculations


def test_a_foreign_multi_field_indicator_is_requested_once_as_a_tuple():
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "macd", "tf": "4h", "field": "macd"},
                              "op": "crosses_above",
                              "rhs": {"ind": "macd", "tf": "4h",
                                      "field": "signal"}}]}}
    program = lower_pine(spec)
    requests = [c for c in program.calculations if "request.security(" in c]
    assert len(requests) == 1
    assert requests[0] == (
        "[nk_macd_2_raw_macd, nk_macd_2_raw_signal, nk_macd_2_raw_hist] = "
        'request.security(syminfo.tickerid, "240", nk_htf_1(), '
        "lookahead=barmerge.lookahead_on, gaps=barmerge.gaps_off)")
    assert ("nk_htf_1() =>\n"
            f"    [nk_macd_1_macd, nk_macd_1_signal, nk_macd_1_hist] = "
            f"ta.macd(close, {LHS}_macd_fast, {LHS}_macd_slow, {LHS}_macd_signal)\n"
            "    [nk_macd_1_macd, nk_macd_1_signal, nk_macd_1_hist]"
            in program.calculations)
    # One gate for the whole tuple, and every member latched under it.
    assert ("if nk_visible_240\n"
            "    nk_macd_2_macd := nk_macd_2_raw_macd\n"
            "    nk_macd_2_signal := nk_macd_2_raw_signal\n"
            "    nk_macd_2_hist := nk_macd_2_raw_hist") in program.calculations
    assert ("nk_long_entry = (ta.crossover(nk_macd_2_macd, nk_macd_2_signal))"
            in program.calculations)


def test_two_lifts_of_the_same_timeframe_keep_their_own_locals():
    # Each request wraps its own function, so a calculation memoized inside one
    # of them must never be handed to the next: the identifier is local to the
    # function that declared it.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [
                {"lhs": {"ind": "sma", "n": 10, "tf": "1h",
                         "of": {"ind": "ema", "n": 5}},
                 "op": ">",
                 "rhs": {"ind": "sma", "n": 20, "tf": "1h",
                         "of": {"ind": "ema", "n": 5}}},
            ]}}
    program = lower_pine(spec)
    bodies = [c for c in program.calculations if c.startswith("nk_htf_")]
    assert len(bodies) == 2
    declared = [{line.split(" = ")[0].strip() for line in body.splitlines()
                 if " = " in line} for body in bodies]
    assert not declared[0] & declared[1]
    for body, names in zip(bodies, declared):
        assert any(name.startswith("nk_ema_") for name in names)
        assert body.splitlines()[-1].strip().removesuffix("[1]") in names


def test_identical_nodes_on_both_sides_share_one_calculation(load_spec):
    # sma_cross reads the same sma(20) and sma(50) on the long and the short
    # side. Two calculations would be the same number computed twice.
    # sma_cross is a 1h play, so both sides live inside ONE request function.
    # That is what makes the sharing survive the lift: a request per side would
    # give each of them a moving average of its own, computed twice and read
    # once, which is the exact regression this asserts against.
    program = lower_pine(load_spec("sma_cross"))
    assert len([c for c in _lines(program) if "ta.sma(" in c]) == 2
    assert len([c for c in program.calculations if "request.security(" in c]) == 1
    assert ("bool nk_long_entry_native = (ta.crossover(nk_sma_1, nk_sma_2))"
            in _lines(program))
    assert ("bool nk_short_entry_native = (ta.crossunder(nk_sma_1, nk_sma_2))"
            in _lines(program))


def test_repeated_compilation_is_byte_identical(load_spec):
    first = lower_pine(load_spec("sma_cross"))
    second = lower_pine(load_spec("sma_cross"))
    assert first == second


def test_the_program_carries_its_identity(load_spec):
    spec = load_spec("rsi_reversion")
    program = lower_pine(spec)
    assert program.title == "rsi_reversion"
    assert program.spec_hash == spec_hash(spec)
    assert program.generator_version == "1"


def test_the_program_holds_no_indicator_or_strategy_statement(load_spec):
    program = lower_pine(load_spec("sma_cross"))
    blob = "\n".join([*program.calculations,
                      *(helper.source for helper in program.helpers)])
    assert "indicator(" not in blob
    assert "strategy(" not in blob


def test_a_side_the_spec_omits_leaves_an_empty_decision():
    program = _program({"ind": "sma"})
    assert program.long_decision == "nk_long_entry"
    assert program.short_decision == ""


def test_the_default_risk_block_becomes_one_distance_calculation():
    program = _program({"ind": "sma"})
    assert program.risk == PineRisk(stop_kind="atr",
                                    stop="nk_risk_stop_distance",
                                    target_kind="rr",
                                    target="nk_risk_target_rr")
    assert ("nk_risk_stop_distance = ta.atr(nk_risk_stop_n) * nk_risk_stop_mult"
            in program.calculations)
    assert program.exits == PineExits()


def test_percent_risk_and_every_exit_become_expressions():
    program = _program(
        {"ind": "sma"},
        risk={"stop": {"kind": "percent", "pct": 1.5},
              "target": {"kind": "percent", "pct": 3.0}},
        exits={"exit": {"all": [{"lhs": {"src": "close"}, "op": "<",
                                 "rhs": {"src": "open"}}]},
               "trailing": {"kind": "atr", "n": 10, "mult": 3.0},
               "time_stop": {"bars": 8},
               "breakeven_at": {"rr": 1.0}})
    assert program.risk == PineRisk(stop_kind="percent", stop="nk_risk_stop_pct",
                                    target_kind="percent",
                                    target="nk_risk_target_pct")
    assert program.exits == PineExits(
        signal="nk_exit_signal",
        trailing_kind="atr",
        trailing="nk_exits_trailing_distance",
        time_stop_bars="nk_exits_time_stop_bars",
        breakeven_rr="nk_exits_breakeven_at_rr")
    assert "nk_exit_signal = (close < open)" in program.calculations
    assert ("nk_exits_trailing_distance = ta.atr(nk_exits_trailing_n) * "
            "nk_exits_trailing_mult" in program.calculations)


def test_a_percent_trailing_stop_carries_its_own_input():
    program = _program({"ind": "sma"},
                       exits={"trailing": {"kind": "percent", "pct": 2.5}})
    assert program.exits == PineExits(trailing_kind="percent",
                                      trailing="nk_exits_trailing_pct")


def test_a_kind_and_the_identifier_it_describes_are_separate_fields():
    # A renderer must never have to tell a Pine identifier from a literal by
    # reading the key it arrived under: "atr" is not something to emit.
    program = _program({"ind": "sma"})
    assert program.risk.stop_kind == "atr"
    assert program.risk.stop.startswith("nk_")
    assert program.exits.trailing_kind == ""
    assert program.exits.trailing == ""


def test_risk_and_exit_defaults_come_from_the_grammar_not_the_compiler():
    # A spec that names a kind and omits its numbers takes spec.py's defaults,
    # which is what the engine sizes a live stop with.
    program = _program({"ind": "sma"},
                       risk={"stop": {"kind": "atr"}, "target": {"kind": "rr"}},
                       exits={"trailing": {"kind": "atr"}})
    defaults = {item.name: item.default for item in program.inputs}
    assert defaults["nk_risk_stop_n"] == STOP_ATR_N_DEFAULT == 14
    assert defaults["nk_risk_stop_mult"] == STOP_ATR_MULT_DEFAULT == 2.0
    assert defaults["nk_risk_target_rr"] == TARGET_RR_DEFAULT == 2.0
    assert defaults["nk_exits_trailing_n"] == TRAILING_ATR_N_DEFAULT == 14
    assert defaults["nk_exits_trailing_mult"] == TRAILING_ATR_MULT_DEFAULT == 2.0
    percent = _program({"ind": "sma"},
                       risk={"stop": {"kind": "percent"},
                             "target": {"kind": "percent"}},
                       exits={"trailing": {"kind": "percent"}})
    defaults = {item.name: item.default for item in percent.inputs}
    assert defaults["nk_risk_stop_pct"] == STOP_PCT_DEFAULT == 2.0
    assert defaults["nk_risk_target_pct"] == TARGET_PCT_DEFAULT == 4.0
    assert defaults["nk_exits_trailing_pct"] == TRAILING_PCT_DEFAULT == 2.0


def test_assumptions_state_the_chart_and_the_request_semantics():
    program = _program({"ind": "sma", "tf": "1h"})
    assert program.assumptions[0] == (
        "The chart must be on 15m bars. Nakagai replays every play on its 15m "
        "driving cadence whatever the play's own timeframe says, so the script "
        "charts that cadence and requests the play's own timeframe rather than "
        "charting it.")
    assert ("1h values are latched on the chart bar that closes with the 1h "
            "bar, which is where the engine first reads them, so they neither "
            "repaint nor lead it." in program.assumptions)
    assert len(set(program.assumptions)) == len(program.assumptions)


def test_a_requested_intraday_frame_states_the_premise_it_rests_on():
    """The one thing about this export nobody has measured, said out loud.

    A requested value is TradingView's aggregate, and TradingView anchors an
    intraday aggregate to the chart's SESSION. Whether those bars are the
    engine's therefore depends entirely on the chart opening at 04:00 rather
    than 09:30 New York, which is what the extended-hours guard enforces. The
    artifact has to say so, because a reader who does not know it cannot check
    it, and every non-15m play is wrong if it is false.
    """
    def premise(spec):
        return [text for text in lower_pine(spec).assumptions
                if "premise of this export" in text]

    for timeframe, spelled in (("1h", "60"), ("4h", "240")):
        text, = premise(_spec({"ind": "sma"}, timeframe=timeframe))
        assert "anchors an intraday aggregate to the chart's session" in text
        assert "opens at 04:00 New York" in text
        assert f'plot time("{spelled}")' in text
    # A FOREIGN intraday reference rests on it just as much as a play's own.
    assert premise(_spec({"ind": "sma", "tf": "1h"}))
    # Nothing requested, nothing to premise: a 15m play reads the chart, and a
    # daily bar is a daily bar however the session is drawn.
    assert not premise(_spec({"ind": "sma"}))
    assert not premise(_spec({"ind": "sma"}, timeframe="1d"))


@pytest.mark.parametrize("timeframe", DEFAULT_TIMEFRAMES.all)
def test_the_chart_is_the_driving_cadence_whatever_the_play_asks_for(timeframe):
    # The defect this replaces: the lowerer charted spec["timeframe"], so a
    # non-15m play decided on bars the engine never decides on, at prices it
    # never pays. There is exactly one chart, and it is the engine's own
    # driving frame, read off the TimeframeSet rather than spelled here.
    spec = _spec({"ind": "sma"}, timeframe=timeframe)
    assert SpecLowerer(spec, core_vocabulary()).chart == DRIVING == "15m"
    assert lower_pine(spec).chart == DRIVING
    assert SpecLowerer(spec, core_vocabulary()).frame == timeframe


@pytest.mark.parametrize("timeframe, gate", [
    ("15m", ""), ("1h", "nk_visible_60"), ("4h", "nk_visible_240"),
    ("1d", "nk_frame_fresh")])
def test_only_a_play_off_the_driving_frame_carries_a_freshness_gate(timeframe,
                                                                    gate):
    # RuleStrategy._fresh, term for term: a 15m play is fresh on every driving
    # bar, an intraday play on the one its own bar closes on, and a
    # session-aligned play on the session open rather than on the calendar day.
    program = lower_pine(_spec({"ind": "sma"}, timeframe=timeframe))
    assert program.decision_gate == gate
    if gate:
        assert any(line.startswith(f"{gate} = ")
                   for line in _lines(program)), _lines(program)


def test_a_session_aligned_play_separates_visibility_from_freshness():
    # The two coincide on an intraday play and must not be collapsed on a daily
    # one: closed_before makes yesterday's bar visible when the New York date
    # arrives, and first_bar_of_session gates the signal on the 09:30 open,
    # which is a later bar. One identifier for both would signal at midnight.
    program = lower_pine(_spec({"ind": "sma"}, timeframe="1d"))
    assert "nk_visible_d = nk_new_session()" in program.calculations
    assert "nk_frame_fresh = nk_session_open_bar()" in program.calculations
    assert program.decision_gate == "nk_frame_fresh"
    assert "if nk_visible_d" in "\n".join(program.calculations)


def test_a_zero_safe_division_is_stated_as_an_assumption():
    node = {"op": "/", "args": [{"src": "close"}, {"src": "volume"}]}
    assert ("A zero denominator reads as na, so a condition over it is false."
            in _program(node).assumptions)


def test_a_session_anchored_term_warns_where_the_two_engines_differ():
    program = _program({"ind": "vwap"})
    assert program.warnings == (
        "ta.vwap anchors to the chart's own session, which follows the "
        "exchange's settings rather than the engine's New York session.",)
    assert _program({"ind": "sma"}).warnings == ()


def test_an_origin_dependent_cumulation_warns_the_same_way():
    # ta.obv counts from the start of the chart's history and indicators.obv
    # from the first bar the engine loaded, so the two lines share a shape but
    # sit at different levels: `obv > 0` is not the same condition on each.
    assert _program({"ind": "obv"}).warnings == (
        "ta.obv cumulates from the start of the chart's history, while the "
        "engine cumulates from the first bar it loaded, so the two lines "
        "share a shape but not a level.",)


def test_a_variable_history_offset_asks_the_renderer_for_max_bars_back():
    # ichimoku displaces its cloud by `disp`, an input rather than a constant,
    # and TradingView cannot infer a buffer from an offset it cannot read.
    # What it owes is a buffer SIZE: the lines index [disp], and reading
    # [100] needs 101 values, this bar's and a hundred behind it.
    program = _program({"ind": "ichimoku", "field": "senkou_a"})
    assert program.max_bars_back == core_vocabulary().indicators[
        "ichimoku"].args["disp"][1] + 1 == 101
    assert f"[{LHS}_ichimoku_disp]" in "\n".join(program.calculations)


def test_a_program_with_only_constant_offsets_needs_no_history_declared():
    # donchian's [1] is a constant, so TradingView sizes its own buffer.
    assert _program({"ind": "donchian", "field": "upper"}).max_bars_back == 0
    assert _program({"ind": "sma"}).max_bars_back == 0


# Compiled in a fresh interpreter, so the comparison spans two hash seeds
# rather than two calls that share one. Set iteration order is stable inside a
# single process whatever PYTHONHASHSEED says, which is exactly what makes an
# in-process determinism check unable to see an unsorted set.
_PROBE = """
import json, sys
from nakagai.strategies.rules import lower_pine
print(repr(lower_pine(json.loads(sys.argv[1]))))
"""

# One spec that touches every ordering decision the compiler makes: nodes
# shared across two blocks, a helper, a warning, a variable history offset, and
# inputs at several depths.
_DETERMINISM_SPEC = {
    "version": 2, "name": "determinism_probe", "timeframe": "15m",
    "long": {"all": [
        {"lhs": {"ind": "sma", "n": 20}, "op": "crosses_above",
         "rhs": {"ind": "sma", "n": 50}},
        {"lhs": {"op": "/", "args": [{"src": "close"}, {"ind": "vwap"}]},
         "op": ">", "rhs": 1},
        {"lhs": {"ind": "ichimoku", "field": "senkou_b"}, "op": "<",
         "rhs": {"src": "close"}},
    ]},
    "short": {"all": [
        {"lhs": {"ind": "sma", "n": 20}, "op": "crosses_below",
         "rhs": {"ind": "sma", "n": 50}},
        {"lhs": {"ind": "obv"}, "op": "<", "rhs": 0},
    ]},
}


def test_compilation_is_identical_across_processes_and_hash_seeds():
    payload = json.dumps(_DETERMINISM_SPEC)
    outputs = set()
    for seed in ("0", "1", "524287"):
        run = subprocess.run(
            [sys.executable, "-c", _PROBE, payload],
            capture_output=True, text=True, check=True,
            env={**os.environ, "PYTHONHASHSEED": seed})
        outputs.add(run.stdout)
    assert len(outputs) == 1
    assert outputs.pop().strip() == repr(lower_pine(_DETERMINISM_SPEC))
