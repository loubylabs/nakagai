"""The RuleSpec expression language lowered into one deterministic Pine program.

Everything here reads a PineProgram, never a rendered artifact. The program is
target-neutral by contract, so no test in this file may expect an `indicator()`
or a `strategy()` statement; those belong to the renderers.
"""

import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules import lower_pine, spec_hash
from nakagai.strategies.rules.pine.lower import PINE_TIMEFRAMES
from nakagai.strategies.rules.vocabulary import core_vocabulary

# The path prefix every operand in a one-condition long group carries.
LHS = "nk_long_all_0_lhs"


def _spec(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return {"version": 2, "name": "probe", "timeframe": timeframe,
            "long": {"all": [{"lhs": lhs, "op": op, "rhs": rhs}]}, **extra}


def _program(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return lower_pine(_spec(lhs, op, rhs, timeframe, **extra))


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


def test_sources_and_numbers_lower_to_plain_operands():
    program = _program({"src": "high"}, "<", {"src": "low"})
    assert "nk_long_entry = (high < low)" in program.calculations


def test_a_field_is_selected_from_one_tuple_calculation(load_spec):
    # macd_trend reads the macd line against its signal line: two operands, one
    # ta.macd. A second calculation would be a second, drifting MACD.
    program = lower_pine(load_spec("macd_trend"))
    tuples = [c for c in program.calculations if "ta.macd(" in c]
    assert len(tuples) == 1
    assert ("nk_long_entry = (ta.crossover(nk_macd_1_macd, nk_macd_1_signal))"
            in program.calculations)


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


def test_a_foreign_timeframe_operand_is_requested_on_its_own_bars():
    program = _program({"ind": "sma", "n": 20, "tf": "1h"})
    assert ("nk_htf_1() =>\n"
            f"    nk_sma_1 = ta.sma(close, {LHS}_sma_n)\n"
            "    nk_sma_1[1]") in program.calculations
    assert ('nk_sma_2 = request.security(syminfo.tickerid, "60", nk_htf_1(), '
            "lookahead=barmerge.lookahead_on, gaps=barmerge.gaps_off)"
            in program.calculations)
    assert "nk_long_entry = (nk_sma_2 > nk_long_all_0_rhs)" in program.calculations


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
        "[nk_macd_2_macd, nk_macd_2_signal, nk_macd_2_hist] = "
        'request.security(syminfo.tickerid, "240", nk_htf_1(), '
        "lookahead=barmerge.lookahead_on, gaps=barmerge.gaps_off)")
    assert ("nk_htf_1() =>\n"
            f"    [nk_macd_1_macd, nk_macd_1_signal, nk_macd_1_hist] = "
            f"ta.macd(close, {LHS}_macd_fast, {LHS}_macd_slow, {LHS}_macd_signal)\n"
            "    [nk_macd_1_macd[1], nk_macd_1_signal[1], nk_macd_1_hist[1]]"
            in program.calculations)
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
    program = lower_pine(load_spec("sma_cross"))
    assert len([c for c in program.calculations if "ta.sma(" in c]) == 2
    assert ("nk_long_entry = (ta.crossover(nk_sma_1, nk_sma_2))"
            in program.calculations)
    assert ("nk_short_entry = (ta.crossunder(nk_sma_1, nk_sma_2))"
            in program.calculations)


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
    assert program.risk == {"stop_kind": "atr",
                            "stop_distance": "nk_risk_stop_distance",
                            "target_kind": "rr", "target_rr": "nk_risk_target_rr"}
    assert ("nk_risk_stop_distance = ta.atr(nk_risk_stop_n) * nk_risk_stop_mult"
            in program.calculations)
    assert program.exits == {}


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
    assert program.risk == {"stop_kind": "percent", "stop_pct": "nk_risk_stop_pct",
                            "target_kind": "percent",
                            "target_pct": "nk_risk_target_pct"}
    assert program.exits == {
        "exit": "nk_exit_signal",
        "trailing_kind": "atr",
        "trailing_distance": "nk_exits_trailing_distance",
        "time_stop_bars": "nk_exits_time_stop_bars",
        "breakeven_rr": "nk_exits_breakeven_at_rr"}
    assert "nk_exit_signal = (close < open)" in program.calculations
    assert ("nk_exits_trailing_distance = ta.atr(nk_exits_trailing_n) * "
            "nk_exits_trailing_mult" in program.calculations)


def test_a_percent_trailing_stop_carries_its_own_input():
    program = _program({"ind": "sma"},
                       exits={"trailing": {"kind": "percent", "pct": 2.5}})
    assert program.exits == {"trailing_kind": "percent",
                             "trailing_pct": "nk_exits_trailing_pct"}


def test_assumptions_state_the_chart_and_the_request_semantics():
    program = _program({"ind": "sma", "tf": "1h"})
    assert program.assumptions[0] == (
        "The chart must be on 15m bars: the spec's own timeframe is charted "
        "rather than requested.")
    assert ("1h values are read with request.security on the last confirmed 1h "
            "bar, so they do not repaint." in program.assumptions)
    assert len(set(program.assumptions)) == len(program.assumptions)


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
