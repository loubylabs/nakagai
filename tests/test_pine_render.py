"""One lowered program, rendered into the two artifacts a user pastes.

Everything here reads a PineBundle. Two properties are load-bearing and each
has its own test rather than being inferred from the others:

- the two artifacts SHARE their decisions. Not "agree", share: the header, the
  inputs, the helpers, the calculations and the two decision identifiers are
  one rendered text embedded in both, so a strategy cannot drift into trading a
  bar the indicator never marked.
- generation is ATOMIC. A failure anywhere yields neither artifact, never one
  good half a caller could ship.

The Pine statements asserted below are exact on purpose. They are what a
TradingView user pastes, so a test that only checked "a stop is placed
somewhere" would pass on a stop with the wrong geometry.
"""

import pytest

from nakagai.strategies.rules import PineCompileError, compile_pine, spec_hash
from nakagai.strategies.rules.pine import render
from nakagai.strategies.rules.spec import TIMEFRAMES


def _spec(**extra) -> dict:
    """A one-condition two-sided spec, so both decisions are always present."""
    return {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"ind": "sma"}, "op": ">",
                              "rhs": {"src": "close"}}]},
            "short": {"all": [{"lhs": {"ind": "sma"}, "op": "<",
                               "rhs": {"src": "close"}}]},
            **extra}


def _header(source: str) -> str:
    """The header's prose with its comment markers and its wrapping undone.

    Asserting a sentence against the raw source would pass or fail on where
    textwrap happened to break the line, which is not what any of these tests
    is about.
    """
    out = []
    for line in source.splitlines():
        if not line.startswith("//"):
            break
        text = line.lstrip("/").strip()
        out.append(text[2:] if text.startswith("- ") else text)
    return " ".join(out)


def _shared(source: str) -> str:
    """The part of an artifact that both of them are supposed to be one copy of."""
    start = source.index("// --- Inputs ---")
    end = source.index("// --- ", source.index("nk_short_decision ="))
    return source[start:end]


# -- the two artifacts are one program ------------------------------------


def test_indicator_and_strategy_share_decision_identifiers(load_rule_spec):
    bundle = compile_pine(load_rule_spec("sma_cross"))
    for name in ("nk_long_decision", "nk_short_decision"):
        assert name in bundle.indicator
        assert name in bundle.strategy


def test_the_shared_half_is_one_text_rather_than_two_agreeing_ones(
        load_rule_spec):
    # Byte-for-byte, not "both mention nk_long_decision": two renderers each
    # building their own copy would pass the identifier test above on the day
    # they agreed and keep passing it on the day one of them changed.
    bundle = compile_pine(load_rule_spec("ifvg_reversal"))
    assert _shared(bundle.indicator) == _shared(bundle.strategy)
    assert "nk_fvg_nearest(" in _shared(bundle.indicator)


def test_each_artifact_holds_only_its_own_half(load_rule_spec):
    bundle = compile_pine(load_rule_spec("orb"))
    assert "strategy." not in bundle.indicator
    assert "alert(" not in bundle.strategy
    assert "plotshape(" in bundle.indicator


def test_rendering_is_atomic(monkeypatch, load_rule_spec):
    # The design's rule: either output failing suppresses the whole bundle. So
    # the strategy is built BEFORE anything is returned, and a caller can never
    # receive an indicator whose strategy did not render.
    def explode(*_args, **_kwargs):
        raise RuntimeError("the strategy half failed")

    monkeypatch.setattr(render, "_strategy", explode)
    with pytest.raises(RuntimeError):
        compile_pine(load_rule_spec("sma_cross"))


def test_a_spec_that_does_not_compile_yields_no_artifact_at_all():
    with pytest.raises(PineCompileError) as exc:
        compile_pine({"version": 2, "name": "probe", "timeframe": "15m"})
    assert exc.value.code == "pine_invalid_spec"


def test_repeated_compilation_is_byte_identical(load_rule_spec):
    first = compile_pine(load_rule_spec("ob_bounce"))
    assert first == compile_pine(load_rule_spec("ob_bounce"))


# -- the shared header and the runtime contract ---------------------------


def test_both_artifacts_open_with_the_version_and_their_identity(
        load_rule_spec):
    spec = load_rule_spec("sma_cross")
    bundle = compile_pine(spec)
    assert bundle.spec_hash == spec_hash(spec)
    assert bundle.generator_version == "1"
    for source in (bundle.indicator, bundle.strategy):
        assert source.splitlines()[0] == "//@version=6"
        assert "// Nakagai Pine export: sma_cross" in source
        assert "// Generator version: 1" in source
        assert f"// Spec hash: {bundle.spec_hash}" in source
        # The FULL hash, not a prefix: it is the artifact's whole claim about
        # which spec it came from.
        assert len(bundle.spec_hash) == 64


def test_the_header_says_a_tradingview_input_edit_leaves_the_hash_behind(
        load_rule_spec):
    bundle = compile_pine(load_rule_spec("orb"))
    for source in (bundle.indicator, bundle.strategy):
        assert ("Editing any input below creates a TradingView-local "
                "variation: its results no longer represent the spec hash "
                "above") in _header(source)


def test_the_runtime_guard_refuses_every_other_chart_timeframe():
    source = compile_pine(_spec()).indicator
    assert ("if barstate.isfirst and timeframe.in_seconds() != 15 * 60\n"
            '    runtime.error("Nakagai Pine exports require a standard '
            '15-minute chart.")') in source


def test_the_guard_names_the_one_chart_the_program_is_true_on(load_rule_spec):
    # A node without a `tf` is emitted on the chart's own bars rather than
    # requested, so a 1h play's arithmetic is 1h arithmetic and a 15m chart
    # would run it over the wrong bars. The guard is the program's own chart,
    # which is what makes it a guard rather than a decoration.
    source = compile_pine(load_rule_spec("sma_cross")).strategy
    assert ("if barstate.isfirst and timeframe.in_seconds() != 60 * 60\n"
            '    runtime.error("Nakagai Pine exports require a standard '
            '1-hour chart.")') in source
    daily = compile_pine(load_rule_spec("bollinger_breakout")).indicator
    assert "timeframe.in_seconds() != 24 * 60 * 60" in daily
    assert "require a standard daily chart." in daily


def test_every_timeframe_the_grammar_admits_has_a_chart_guard():
    assert set(render.CHART) == set(TIMEFRAMES)


def test_a_synthetic_chart_is_refused_at_runtime_and_warned_about_in_the_header():
    source = compile_pine(_spec()).indicator
    assert ("if barstate.isfirst and not chart.is_standard\n"
            '    runtime.error("Nakagai Pine exports require a standard candle '
            'chart.")') in source
    # chart.is_standard is the runtime half; naming the chart types is the
    # header's, because a user who has to be told WHICH chart to leave is
    # reading the source rather than watching for a runtime error.
    for name in ("Heikin Ashi", "Renko", "Line Break", "Kagi",
                 "Point and Figure", "Range"):
        assert name in _header(source)


def test_the_header_carries_the_assumptions_and_warnings_the_program_holds():
    bundle = compile_pine(_spec(
        long={"all": [{"lhs": {"ind": "vwap"}, "op": ">", "rhs": 0}]},
        short={"all": [{"lhs": {"ind": "vwap"}, "op": "<", "rhs": 0}]}))
    assert "ta.vwap anchors to the chart's own session" in bundle.warnings[-1]
    for source in (bundle.indicator, bundle.strategy):
        header = _header(source)
        assert "Every condition is read on the close of its bar." in header
        for text in (*bundle.assumptions, *bundle.warnings):
            assert text in header


def test_the_bundle_always_carries_the_five_fidelity_warnings():
    bundle = compile_pine(_spec())
    assert bundle.warnings == render.FIDELITY
    assert len(render.FIDELITY) == 5
    for source in (bundle.indicator, bundle.strategy):
        header = _header(source)
        assert "is never Nakagai evidence" in header
        for text in render.FIDELITY:
            assert text in header


def test_a_conditional_exit_says_it_closes_one_bar_later_than_the_engine(
        load_rule_spec):
    # The engine closes a manage() exit at the signal bar's own close;
    # TradingView fills the market order it becomes at the next bar's open.
    # Unstated, that reads as a bug in one of the two engines.
    plain = compile_pine(_spec())
    timed = compile_pine(load_rule_spec("orb"))
    assert render.NEXT_BAR_CLOSE not in plain.warnings
    assert render.NEXT_BAR_CLOSE in timed.warnings
    assert render.NEXT_BAR_CLOSE in _header(timed.strategy)
    assert render.NEXT_BAR_CLOSE not in _header(plain.indicator)


def test_a_play_on_a_longer_timeframe_says_where_its_entry_lands(
        load_rule_spec):
    # The engine drives on 15m whatever a play's own timeframe is, so a 1h play
    # signals one driving bar after the 1h bar closes. A 1h chart decides at
    # the close of that bar instead: the same rule, a different entry price.
    assert render.DRIVING_CADENCE not in compile_pine(_spec()).warnings
    for name in ("sma_cross", "bollinger_breakout"):
        bundle = compile_pine(load_rule_spec(name))
        assert render.DRIVING_CADENCE in bundle.warnings
        assert render.DRIVING_CADENCE in _header(bundle.strategy)


# -- inputs, helpers, calculations ----------------------------------------


def test_inputs_render_as_typed_bounded_pine_inputs(load_rule_spec):
    source = compile_pine(load_rule_spec("sma_cross")).indicator
    assert ('nk_long_all_0_lhs_sma_n = input.int(20, "lhs · sma · n", '
            "minval=2, maxval=500)") in source
    assert ('nk_risk_stop_mult = input.float(2.0, "Risk · stop · mult", '
            "minval=0.1, maxval=10.0)") in source
    # An int input with a float bound does not compile on TradingView, and a
    # float input whose default reads as an integer is a type error. Both are
    # decided by PineInput.kind, so both spellings are pinned.
    assert "input.int(20.0" not in source
    assert "input.float(2," not in source


def test_an_unbounded_threshold_declares_no_bounds():
    source = compile_pine(_spec(
        long={"all": [{"lhs": {"ind": "rsi"}, "op": ">", "rhs": 70}]},
        short={"all": [{"lhs": {"ind": "rsi"}, "op": "<", "rhs": 30}]},
    )).indicator
    assert 'nk_long_all_0_rhs = input.int(70, "Long · rhs")' in source


def test_helpers_render_once_each_in_dependency_order(load_rule_spec):
    source = compile_pine(load_rule_spec("orb")).indicator
    assert source.count("nk_new_session() =>") == 1
    assert source.index("nk_session_key() =>") < source.index("nk_new_session() =>")
    assert source.index("nk_new_session() =>") < \
        source.index("nk_opening_range_high(minutes) =>")


def test_a_variable_history_offset_declares_max_bars_back(load_rule_spec):
    # TradingView cannot infer a buffer from an offset it cannot read, and
    # answers "Pine cannot determine the referencing length of series" instead.
    deep = compile_pine(load_rule_spec("ob_bounce"))
    assert "max_bars_back=200" in deep.indicator
    assert "max_bars_back=200" in deep.strategy
    flat = compile_pine(load_rule_spec("sma_cross"))
    assert "max_bars_back" not in flat.indicator
    assert "max_bars_back" not in flat.strategy


# -- the decision and its geometry ----------------------------------------


def test_the_stop_and_target_are_measured_from_the_signal_bar_close():
    source = compile_pine(_spec()).indicator
    assert "nk_long_stop = close - nk_risk_stop_distance" in source
    assert ("nk_long_target = close + nk_risk_target_rr * "
            "(close - nk_long_stop)") in source
    assert "nk_short_stop = close + nk_risk_stop_distance" in source
    assert ("nk_short_target = close - nk_risk_target_rr * "
            "(nk_short_stop - close)") in source


def test_percent_stops_and_percent_targets_carry_their_own_geometry():
    source = compile_pine(_spec(
        risk={"stop": {"kind": "percent", "pct": 1.5},
              "target": {"kind": "percent", "pct": 3.0}})).indicator
    assert "nk_long_stop = close - close * nk_risk_stop_pct / 100" in source
    assert "nk_long_target = close + close * nk_risk_target_pct / 100" in source
    assert "nk_short_stop = close + close * nk_risk_stop_pct / 100" in source
    assert "nk_short_target = close - close * nk_risk_target_pct / 100" in source


def test_a_decision_stands_only_where_the_geometry_is_real():
    # rr_signal returns None for a NaN stop or one on the wrong side of the
    # reference, so the engine emits NO signal there. Without this the first
    # bars of a chart (where ta.atr is still na) would mark a decision, fire an
    # alert carrying NaN levels, and size a position from a NaN risk.
    source = compile_pine(_spec()).indicator
    assert ("nk_long_decision = nk_long_entry and not na(nk_long_stop) and "
            "nk_long_stop < close and nk_long_target > close") in source
    assert ("nk_short_decision = nk_short_entry and not na(nk_short_stop) and "
            "nk_short_stop > close and nk_short_target < close") in source


def test_a_side_the_spec_omits_renders_as_a_false_decision():
    source = compile_pine({
        "version": 2, "name": "probe", "timeframe": "15m",
        "long": {"all": [{"lhs": {"ind": "sma"}, "op": ">",
                          "rhs": {"src": "close"}}]}}).strategy
    assert "nk_short_decision = false" in source
    # ... and nothing downstream pretends the side exists.
    assert "nk_short_stop" not in source
    assert "strategy.short" not in source


# -- the indicator --------------------------------------------------------


def test_the_indicator_marks_decisions_without_placing_an_order():
    source = compile_pine(_spec()).indicator
    assert source.count("indicator(") == 1
    assert 'indicator("Nakagai · probe", overlay=true)' in source
    assert "plotshape(nk_long_decision" in source
    assert "plotshape(nk_short_decision" in source


def test_the_indicator_freezes_the_signal_levels_and_plots_them():
    source = compile_pine(_spec()).indicator
    for name in ("nk_reference", "nk_stop", "nk_target"):
        assert f"var float {name} = na" in source
        assert f"plot({name}," in source
    assert ("if nk_long_decision\n"
            "    nk_reference := close\n"
            "    nk_stop := nk_long_stop\n"
            "    nk_target := nk_long_target\n") in source


def test_exactly_one_alert_fires_and_the_long_side_wins_a_both_sides_bar():
    source = compile_pine(_spec()).indicator
    assert source.count("alert(") == 2       # one per side, in one if/else
    assert "if nk_long_decision" in source
    assert "else if nk_short_decision" in source
    body = source[source.index("if nk_long_decision"):]
    assert body.index('nk_signal_json("long"') < \
        body.index("else if nk_short_decision") < \
        body.index('nk_signal_json("short"')


def test_the_alert_body_is_the_nakagai_signal_schema(load_rule_spec):
    bundle = compile_pine(load_rule_spec("orb"))
    source = bundle.indicator
    assert '\\"schema\\":\\"nakagai.pine.signal.v1\\"' in source
    assert f'\\"spec_hash\\":\\"{bundle.spec_hash}\\"' in source
    for key, value in (("symbol", "syminfo.ticker"),
                       ("timeframe", "timeframe.period"),
                       ("direction", "direction")):
        assert f'\\"{key}\\":\\"' in source and value in source
    for key in ("bar_time", "reference", "stop", "target"):
        assert f'\\"{key}\\":' in source
    assert "str.tostring(time)" in source
    assert "alert.freq_once_per_bar_close" in source


# -- the strategy ---------------------------------------------------------


def test_strategy_contract_is_explicit(load_rule_spec):
    source = compile_pine(load_rule_spec("orb")).strategy
    assert "calc_on_order_fills=false" in source
    assert "calc_on_every_tick=false" in source
    assert "process_orders_on_close=false" in source
    assert "pyramiding=0" in source
    assert "if nk_long_decision" in source
    assert "else if nk_short_decision" in source


def test_the_strategy_holds_one_position_and_enters_only_when_flat():
    source = compile_pine(_spec()).strategy
    assert 'strategy("Nakagai · probe · strategy", overlay=true, ' \
        "initial_capital=10000, pyramiding=0" in source
    assert "if strategy.position_size == 0" in source
    assert 'strategy.entry("Nakagai long", strategy.long, qty=nk_long_size)' \
        in source
    assert 'strategy.entry("Nakagai short", strategy.short, qty=nk_short_size)' \
        in source


def test_the_strategy_sizes_from_current_equity_and_per_share_risk():
    source = compile_pine(_spec()).strategy
    assert ('nk_risk_fraction = input.float(1.0, "Sizing · risk per trade '
            '(% of equity)", minval=0.01, maxval=100.0)') in source
    # floor, and never a fraction of a share: the engine floors too, and a
    # position it rounds to zero is a trade neither engine takes.
    assert ("float nk_long_size = math.floor(strategy.equity * "
            "nk_risk_fraction / 100 / (close - nk_long_stop))") in source
    assert ("float nk_short_size = math.floor(strategy.equity * "
            "nk_risk_fraction / 100 / (nk_short_stop - close))") in source
    assert "if nk_long_size > 0" in source


def test_the_strategy_brackets_the_position_with_the_frozen_levels():
    source = compile_pine(_spec()).strategy
    assert ('strategy.exit("Nakagai long exit", from_entry="Nakagai long", '
            "stop=nk_stop, limit=nk_target)") in source
    assert ('strategy.exit("Nakagai short exit", from_entry="Nakagai short", '
            "stop=nk_stop, limit=nk_target)") in source
    # Submitted while the position is open, never on the signal bar: the engine
    # checks its levels before it fills a pending entry, so the earliest bar a
    # stop can take it out is the one after the fill.
    assert "if strategy.position_size > 0" in source
    assert "else if strategy.position_size < 0" in source


def test_a_conditional_exit_closes_the_matching_position(load_rule_spec):
    source = compile_pine(load_rule_spec("orb")).strategy
    assert ('    if nk_exit_signal\n'
            '        strategy.close("Nakagai long", '
            'comment="Nakagai exit rule")') in source
    assert ('    if nk_exit_signal\n'
            '        strategy.close("Nakagai short", '
            'comment="Nakagai exit rule")') in source


def test_a_time_stop_counts_the_fill_bar_as_held_bar_one(load_rule_spec):
    # manage() runs in the same loop pass as the fill, so held == 1 there.
    source = compile_pine(load_rule_spec("orb")).strategy
    assert ("bar_index - strategy.opentrades.entry_bar_index(0) + 1 >= "
            "nk_exits_time_stop_bars") in source
    assert 'strategy.close("Nakagai long", comment="Nakagai time stop")' in source


def test_break_even_moves_the_stop_to_the_fill_price_at_its_r_multiple():
    source = compile_pine(_spec(exits={"breakeven_at": {"rr": 1.0}})).strategy
    # The R is measured against the SIGNAL stop, which never moves, so a
    # ratchet cannot shrink the yardstick it is measured with.
    assert "var float nk_signal_stop = na" in source
    assert ("float nk_long_risk = math.abs(strategy.position_avg_price - "
            "nk_signal_stop)") in source
    assert ("if nk_long_risk > 0 and (close - strategy.position_avg_price) / "
            "nk_long_risk >= nk_exits_breakeven_at_rr") in source
    assert "nk_stop := math.max(nk_stop, strategy.position_avg_price)" in source
    assert "nk_stop := math.min(nk_stop, strategy.position_avg_price)" in source


def test_a_spec_without_a_break_even_keeps_no_signal_stop():
    assert "nk_signal_stop" not in compile_pine(_spec()).strategy


def test_trailing_stops_ratchet_toward_price_and_never_away():
    atr = compile_pine(_spec(
        exits={"trailing": {"kind": "atr", "n": 10, "mult": 3.0}})).strategy
    assert "float nk_long_trail = close - nk_exits_trailing_distance" in atr
    assert "nk_stop := math.max(nk_stop, nk_long_trail)" in atr
    assert "float nk_short_trail = close + nk_exits_trailing_distance" in atr
    assert "nk_stop := math.min(nk_stop, nk_short_trail)" in atr
    percent = compile_pine(_spec(
        exits={"trailing": {"kind": "percent", "pct": 2.5}})).strategy
    assert ("float nk_long_trail = close - close * nk_exits_trailing_pct / 100"
            in percent)
    assert ("float nk_short_trail = close + close * nk_exits_trailing_pct / 100"
            in percent)
    # A NaN candidate must not blank the stop; the engine's ratchet skips it.
    assert "if not na(nk_long_trail)" in atr


def test_an_exit_rule_and_a_time_stop_pre_empt_the_ratchets():
    # manage() returns EXIT before it reaches either ratchet, so a bar that is
    # closing the position does not also move its stop.
    source = compile_pine(_spec(exits={
        "exit": {"all": [{"lhs": {"src": "close"}, "op": "<",
                          "rhs": {"src": "open"}}]},
        "time_stop": {"bars": 8}, "breakeven_at": {"rr": 1.0},
        "trailing": {"kind": "atr"}})).strategy
    block = source[source.index("if strategy.position_size > 0"):]
    block = block[:block.index("else if strategy.position_size < 0")]
    assert block.splitlines()[1].strip() == "if nk_exit_signal"
    assert "    else if bar_index - strategy.opentrades" in block
    assert block.index("\n    else\n") < block.index("nk_long_risk")
    assert block.index("nk_long_risk") < block.index("nk_long_trail")


# -- house style ----------------------------------------------------------


@pytest.mark.parametrize("name", ["sma_cross", "orb", "ifvg_reversal",
                                  "ob_bounce", "bollinger_breakout"])
def test_no_dash_lookalike_reaches_an_artifact(name, load_rule_spec):
    # The middle dot in a label is a separator and stays; an em dash or an en
    # dash is house style, and a golden is the one place it would ship.
    bundle = compile_pine(load_rule_spec(name))
    for source in (bundle.indicator, bundle.strategy):
        assert "—" not in source
        assert "–" not in source
        assert " · " in source

