"""The stateful primitives, lowered into Pine that keeps their causality.

These are the terms with no TradingView equivalent: session windows, swing
confirmation, the fair-value-gap lifecycle, order blocks. Each one is a
translation of the engine's own algorithm, so the assertions here are about
WHEN a value becomes readable and WHAT it is computed from, rather than about
the shape of a call. A built-in with a similar name would pass a shape test and
trade differently.

Everything reads a PineProgram or a helper's source, never a rendered artifact.
"""

import dataclasses

import pytest

from nakagai.strategies.rules import lower_pine
from nakagai.strategies.rules.pine.lowerings import HELPERS
from nakagai.strategies.rules.pine.model import PineLowering
from nakagai.strategies.rules.vocabulary import core_vocabulary, is_choice_rule

LHS = "nk_long_all_0_lhs"
CLOSE_OVER_OPEN = {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}


def _spec(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return {"version": 2, "name": "probe", "timeframe": timeframe,
            "long": {"all": [{"lhs": lhs, "op": op, "rhs": rhs}]}, **extra}


def _program(lhs, op=">", rhs=0, timeframe="15m", **extra):
    return lower_pine(_spec(lhs, op, rhs, timeframe, **extra))


def _source(helper_id):
    return HELPERS[helper_id].source


# One row per primitive: the node, the calculations it owes, and the identifier
# the condition reads. Asserted complete against the vocabulary below, so a
# primitive without a Pine form fails here rather than silently.
PRIMITIVES = [
    ({"prim": "opening_range_high"},
     [f"nk_opening_range_high_1 = nk_opening_range_high("
      f"{LHS}_opening_range_high_minutes)"], "nk_opening_range_high_1"),
    ({"prim": "opening_range_low"},
     [f"nk_opening_range_low_1 = nk_opening_range_low("
      f"{LHS}_opening_range_low_minutes)"], "nk_opening_range_low_1"),
    ({"prim": "prev_session_high"},
     ["nk_prev_session_high_1 = nk_prev_session_high()"],
     "nk_prev_session_high_1"),
    ({"prim": "prev_session_low"},
     ["nk_prev_session_low_1 = nk_prev_session_low()"],
     "nk_prev_session_low_1"),
    ({"prim": "prev_session_close"},
     ["nk_prev_session_close_1 = nk_prev_session_close()"],
     "nk_prev_session_close_1"),
    ({"prim": "gap_pct"}, ["nk_gap_pct_1 = nk_gap_pct()"], "nk_gap_pct_1"),
    ({"prim": "rvol"}, [f"nk_rvol_1 = nk_rvol({LHS}_rvol_sessions)"],
     "nk_rvol_1"),
    ({"prim": "minutes_into_session"},
     ["nk_minutes_into_session_1 = nk_minutes_into_session()"],
     "nk_minutes_into_session_1"),
    ({"prim": "day_of_week"}, ["nk_day_of_week_1 = nk_day_of_week()"],
     "nk_day_of_week_1"),
    ({"prim": "swing_high"},
     [f"nk_swing_high_1 = nk_swing_high({LHS}_swing_high_k)"],
     "nk_swing_high_1"),
    ({"prim": "swing_low"},
     [f"nk_swing_low_1 = nk_swing_low({LHS}_swing_low_k)"], "nk_swing_low_1"),
    ({"prim": "leg_retrace"},
     [f"nk_leg_retrace_1 = nk_leg_retrace({LHS}_leg_retrace_k, true)"],
     "nk_leg_retrace_1"),
    ({"prim": "bars_since", "cond": CLOSE_OVER_OPEN},
     ["nk_bars_since_1 = nk_bars_since(close > open)"], "nk_bars_since_1"),
    ({"prim": "fvg_nearest", "field": "mid"},
     ["[nk_fvg_nearest_1_top, nk_fvg_nearest_1_bottom] = nk_fvg_nearest("
      f"{LHS}_fvg_nearest_min_size_atr, {LHS}_fvg_nearest_lookback, "
      "true, false)",
      "nk_fvg_nearest_1_mid = "
      "(nk_fvg_nearest_1_top + nk_fvg_nearest_1_bottom) / 2"],
     "nk_fvg_nearest_1_mid"),
    ({"prim": "order_block", "field": "bottom"},
     ["[nk_order_block_1_top, nk_order_block_1_bottom] = nk_order_block("
      f"{LHS}_order_block_body_atr, {LHS}_order_block_lookback, true)"],
     "nk_order_block_1_bottom"),
]


@pytest.mark.parametrize("node, statements, operand", PRIMITIVES,
                         ids=[row[0]["prim"] for row in PRIMITIVES])
def test_every_primitive_lowers_to_its_pine_form(node, statements, operand):
    program = _program(node)
    for statement in statements:
        assert statement in program.calculations
    assert (f"nk_long_entry = ({operand} > nk_long_all_0_rhs)"
            in program.calculations)


def test_the_primitive_table_covers_every_primitive_in_the_vocabulary():
    covered = {node["prim"] for node, _, _ in PRIMITIVES}
    assert covered == set(core_vocabulary().primitives)


# One row per (primitive, field) the grammar admits, so a mapping answering
# only the field a test happened to ask for is caught.
FIELDS = {
    ("fvg_nearest", "top"): "nk_fvg_nearest_1_top",
    ("fvg_nearest", "bottom"): "nk_fvg_nearest_1_bottom",
    ("fvg_nearest", "mid"): "nk_fvg_nearest_1_mid",
    ("order_block", "top"): "nk_order_block_1_top",
    ("order_block", "bottom"): "nk_order_block_1_bottom",
    ("order_block", "mid"): "nk_order_block_1_mid",
}


@pytest.mark.parametrize("pair, operand", sorted(FIELDS.items()),
                         ids=[f"{p}.{f}" for p, f in sorted(FIELDS)])
def test_every_declared_field_of_every_primitive_reads_its_own_identifier(
        pair, operand):
    primitive, field = pair
    program = _program({"prim": primitive, "field": field})
    assert (f"nk_long_entry = ({operand} > nk_long_all_0_rhs)"
            in program.calculations)


def test_the_field_table_covers_every_field_choice_in_the_vocabulary():
    declared = {(name, field)
                for name, term in core_vocabulary().primitives.items()
                for rule in [term.args.get("field")] if is_choice_rule(rule)
                for field in rule}
    assert declared == set(FIELDS)


def test_two_readings_of_one_gap_share_one_scan():
    # The top and the bottom of the nearest gap are one scan of the window, so
    # a spec reading both must not run it twice: a second scan is a second
    # window ATR and can identify a different gap.
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"prim": "fvg_nearest", "field": "top"},
                              "op": ">",
                              "rhs": {"prim": "fvg_nearest",
                                      "field": "bottom"}}]}}
    program = lower_pine(spec)
    assert len([c for c in program.calculations if "nk_fvg_nearest(" in c]) == 1
    assert ("nk_long_entry = (nk_fvg_nearest_1_top > nk_fvg_nearest_1_bottom)"
            in program.calculations)


# Two spellings of the same primitive that stay two nodes, so the program has
# two calls of one helper. minutes_into_session and day_of_week take no
# argument and no timeframe, so their two readings are one node by
# construction; the assertion below still holds, the memo does the work.
TWICE = {
    "opening_range_high": ({"minutes": 15}, {"minutes": 60}),
    "opening_range_low": ({"minutes": 15}, {"minutes": 60}),
    "prev_session_high": ({}, {"tf": "1h"}),
    "prev_session_low": ({}, {"tf": "1h"}),
    "prev_session_close": ({}, {"tf": "1h"}),
    "gap_pct": ({}, {"tf": "1h"}),
    "rvol": ({"sessions": 10}, {"sessions": 30}),
    "minutes_into_session": ({}, {}),
    "day_of_week": ({}, {}),
    "swing_high": ({"k": 2}, {"k": 5}),
    "swing_low": ({"k": 2}, {"k": 5}),
    "leg_retrace": ({"k": 2}, {"k": 5}),
    "bars_since": ({"cond": CLOSE_OVER_OPEN},
                   {"cond": {"lhs": {"src": "high"}, "op": ">",
                             "rhs": {"src": "low"}}}),
    "fvg_nearest": ({"lookback": 20}, {"lookback": 60}),
    "order_block": ({"lookback": 20}, {"lookback": 60}),
}


@pytest.mark.parametrize("name", sorted(TWICE))
def test_primitive_helper_is_emitted_once(name):
    first, second = TWICE[name]
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [
                {"lhs": {"prim": name, **first}, "op": ">", "rhs": 0},
                {"lhs": {"prim": name, **second}, "op": ">", "rhs": 0},
            ]}}
    program = lower_pine(spec)
    ids = [helper.id for helper in program.helpers]
    assert ids.count(f"nk_{name}") == 1
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", sorted(core_vocabulary().primitives))
def test_every_numeric_argument_of_a_primitive_reaches_one_input(name):
    term = core_vocabulary().primitives[name]
    node = {"prim": name}
    if name == "bars_since":
        node["cond"] = CLOSE_OVER_OPEN
    names = {item.name for item in _program(node).inputs}
    for arg, rule in term.args.items():
        if is_choice_rule(rule) or rule == "condition":
            continue
        assert f"nk_long_all_0_lhs_{name}_{arg}" in names


def test_a_choice_argument_never_becomes_an_input():
    # A choice picks which Pine the lowering writes, so a chart knob for it
    # would let a user pick a strategy the compiler never generated.
    program = _program({"prim": "fvg_nearest", "direction": "short",
                        "state": "inverted", "field": "bottom"})
    names = {item.name for item in program.inputs}
    assert not any("direction" in name or "state" in name or "field" in name
                   for name in names)
    # An inverted short reads a gap of the OPPOSITE original direction, so the
    # scan is asked for long-origin gaps that a later close has flipped.
    assert any("nk_fvg_nearest(" in c and "true, true)" in c
               for c in program.calculations)


def test_the_state_and_direction_pair_chooses_the_gaps_the_scan_keeps():
    # fvg_nearest's "open" reads a gap of the trade's own direction, and
    # "inverted" reads one of the opposite original direction whose far
    # boundary a later bar closed through. Four combinations, two literals.
    wanted = {("long", "open"): "true, false", ("short", "open"): "false, false",
              ("long", "inverted"): "false, true",
              ("short", "inverted"): "true, true"}
    for (direction, state), literals in wanted.items():
        program = _program({"prim": "fvg_nearest", "direction": direction,
                            "state": state})
        assert any(c.endswith(f"{literals})") and "nk_fvg_nearest(" in c
                   for c in program.calculations), (direction, state)


# -- the causal rules ------------------------------------------------------
# Each assertion below names the line of nakagai/strategies/rules/primitives.py
# (or the ICT module it calls) that the Pine mirrors.


def test_the_opening_range_is_invisible_until_its_own_window_elapses():
    # _opening_range: the level accumulates over `[start, edge)` and is
    # returned only `where(ts >= edge)`, so a session that closes inside the
    # range never gets one. Both halves have to be in the helper, or the level
    # reads a bar early and every opening-range breakout fires on the range's
    # own bars.
    source = _source("nk_opening_range_high")
    assert "if time >= start and time < edge" in source
    assert "time >= edge ? level : na" in source
    # The window opens at the BELL, from nk_session_open, not at the session's
    # own first bar. Neither engine's first bar is the open: the caches are not
    # RTH-only and a chart's extended session starts at 04:00, so measuring
    # from it aggregated a pre-market band and called it the opening range.
    assert "int start = nk_session_open()" in source
    assert "edge = start + minutes * 60000" in source


def test_the_opening_range_low_keeps_the_minimum_of_the_same_window():
    source = _source("nk_opening_range_low")
    assert "math.min(level, low)" in source
    assert "time >= edge ? level : na" in source


def test_the_previous_session_levels_roll_only_when_the_session_changes():
    # _prev_session: per_day.agg(...).shift(1), read back onto every bar of the
    # current session. The roll happens on the session's first bar, so a bar
    # never sees its own session's aggregate.
    for helper_id, aggregate in (("nk_prev_session_high", "math.max(current, high)"),
                                 ("nk_prev_session_low", "math.min(current, low)")):
        source = _source(helper_id)
        assert "if nk_new_session()" in source
        assert "previous := current" in source
        assert aggregate in source
        # The value read out is the PREVIOUS session's, on every bar.
        assert source.rstrip().endswith("previous")
    close = _source("nk_prev_session_close")
    assert "current := close" in close
    assert close.rstrip().endswith("previous")


def test_the_previous_session_levels_aggregate_regular_hours_only():
    # _prev_session masks with data/schema.rth_mask before the groupby, so the
    # extremes are the session's and not the widest off-hours print of its
    # date. Both engines carry pre-market and post-market bars, so an
    # unfiltered aggregate made "yesterday's high" a thin quote nothing
    # defended, and "yesterday's close" the 19:45 print rather than the 15:45
    # one every catalog play means by it.
    for helper_id in ("nk_prev_session_high", "nk_prev_session_low",
                      "nk_prev_session_close"):
        source = _source(helper_id)
        assert "if nk_in_session()" in source
        # Reset to na rather than to this bar's price: a session with no
        # regular bar at all answers na, which is what a masked pandas
        # aggregate returns for an all-NaN group. Seeding from the first bar of
        # the date would seed from a pre-market print instead.
        assert "        current := na" in source


def test_the_regular_session_is_the_wall_clock_window_the_engine_masks_on():
    # data/schema.rth_mask: [09:30, 16:00) of the bar's own New York session,
    # half-open on the right because a bar is labeled by its OPEN. 570 and 960
    # are those two times in minutes, read off hour() and minute() in New York
    # so the window survives a daylight-saving change.
    source = _source("nk_in_session")
    assert ('hour(time, "America/New_York") * 60 + '
            'minute(time, "America/New_York")' in source)
    assert "into >= 570 and into < 960" in source


def test_a_session_is_the_bar_s_new_york_calendar_day():
    # _session_keys: one group per New York calendar day, spelled engine-side
    # as that day's 09:30 open. A session key built on any other clock groups
    # the bars differently and moves every session-scoped primitive at once.
    key = _source("nk_session_key")
    assert 'tz = "America/New_York"' in key
    assert "year(time, tz) * 10000 + month(time, tz) * 100 + dayofmonth(time, tz)" \
        in key
    # The chart's first bar starts a session: there is no earlier key to differ
    # from, and the engine gives that partial session its own group.
    assert "na(key[1]) or key != key[1]" in _source("nk_new_session")


def test_the_session_open_is_the_bell_on_the_new_york_wall_clock():
    # data/schema.session_opens, which every session-anchored helper measures
    # from. 570 is 09:30 New York in minutes, and reading it off hour() and
    # minute() in that zone is what keeps the bell at 09:30 through a
    # daylight-saving change instead of drifting an hour, the same reasoning
    # session_opens writes down for its own naive-local construction.
    source = _source("nk_session_open")
    assert ('hour(time, "America/New_York") * 60 + '
            'minute(time, "America/New_York") - 570' in source)
    assert "time - into * 60000" in source


def test_the_gap_is_this_session_s_open_against_the_last_session_s_close():
    # gap_pct: 100 * (session open - prev session close) / prev session close.
    source = _source("nk_gap_pct")
    assert "session_open := open" in source
    assert "nk_prev_session_close()" in source
    assert "100 * (session_open - previous) / previous" in source
    # The open is the session's first REGULAR bar's, and it is na until that
    # bar has printed. Both halves are the fix for chrvsd/nakagai#276: the
    # first bar of the New York date is a pre-market print on a cache that is
    # not RTH-only, and pandas' transform broadcasts the group's first regular
    # open backwards over the pre-market, so the engine blanks those bars and
    # this must answer na on them for the same reason.
    assert "if nk_in_session() and na(session_open)" in source
    assert "        session_open := na" in source


def test_minutes_into_session_counts_from_the_bell_and_is_na_before_it():
    # minutes_into_session: (bar - session_opens).total_seconds() / 60, blanked
    # where that is negative. The blanking is half the primitive: a pre-market
    # bar reads negative against the bell and a negative passes every `< N`
    # gate a session play writes, so an unblanked chart would mark entries an
    # hour and a half before the engine's earliest.
    source = _source("nk_minutes_into_session")
    assert "(time - nk_session_open()) / 60000.0" in source
    assert "mins >= 0 ? mins : na" in source


def test_the_weekday_is_zero_on_monday_the_way_pandas_numbers_it():
    # day_of_week: 0 = Monday. Pine numbers Sunday 1 through Saturday 7.
    assert '(dayofweek(time, "America/New_York") + 5) % 7' \
        in _source("nk_day_of_week")


def test_relative_volume_compares_a_bar_with_its_own_clock_time():
    # rvol: bucket on the EXCHANGE clock, median of the trailing `sessions`
    # occurrences, shifted by one so a session is never in its own baseline,
    # and NaN until the bucket holds a full window.
    source = _source("nk_rvol")
    assert ('hour(time, "America/New_York") * 60 + '
            'minute(time, "America/New_York")' in source)
    assert "array.indexof(clocks, clock)" in source
    # Half a window is a different measurement, not a rough one.
    assert "float baseline = na" in source
    assert "if seen >= sessions" in source
    # Read before the write: this bar's volume enters the bucket only after the
    # baseline it is measured against has been taken.
    assert source.index("baseline") < source.index("array.set(ring")
    # A median, not a mean: one halted session inside the window would drag a
    # mean far enough to hide the next genuine surge.
    assert "array.sort(window)" in source
    assert "array.avg" not in source
    # A zero baseline is an absence of anything to compare against, not an
    # infinite surge.
    assert "baseline <= 0 ? na" in source


def test_a_swing_is_stamped_at_the_bar_that_confirms_it():
    # _swing: a swing at i is only KNOWN k bars later, so the pivot's price is
    # stamped at i + k and carried forward. In Pine the pivot under test sits
    # at [k], its neighbours at [k + i] and [k - i], and the level is only
    # reassigned on a confirmation.
    for helper_id, source_series, worse in (("nk_swing_high", "high", ">"),
                                            ("nk_swing_low", "low", "<")):
        source = _source(helper_id)
        assert f"pivot = {source_series}[k]" in source
        assert f"not (pivot {worse} {source_series}[k + i])" in source
        assert f"not (pivot {worse} {source_series}[k - i])" in source
        # _strict_extrema returns an all-false mask until both windows are
        # full, which is 2k bars of history.
        assert "bar_index >= 2 * k" in source
        # No else: an unconfirmed bar carries the last confirmed level.
        assert "level := pivot" in source
        assert source.rstrip().endswith("level")


def test_the_leg_retrace_reads_both_confirmed_swings_and_refuses_a_flat_range():
    # leg_retrace: (H - close) / (H - L) long, mirrored short, NaN when H <= L.
    source = _source("nk_leg_retrace")
    assert "nk_swing_high(k)" in source and "nk_swing_low(k)" in source
    assert "span > 0" in source
    assert "(hi - close) / span" in source
    assert "(close - lo) / span" in source


def test_bars_since_counts_from_the_last_time_the_condition_held():
    # bars_since: pos - pos.where(mask).ffill(), which is 0 on the bar the
    # condition holds and NaN before the first one. The no-hit value is not 0.
    source = _source("nk_bars_since")
    assert "hit ? 0.0" in source
    assert "na(since) ? na : since + 1" in source


def test_bars_since_lowers_the_condition_it_is_handed():
    program = _program({"prim": "bars_since",
                        "cond": {"lhs": {"ind": "rsi"}, "op": "<", "rhs": 30}},
                       "<", 5)
    assert (f"nk_rsi_1 = ta.rsi(close, {LHS}_cond_lhs_rsi_n)"
            in program.calculations)
    assert (f"nk_bars_since_1 = nk_bars_since(nk_rsi_1 < {LHS}_cond_rhs)"
            in program.calculations)


def test_a_condition_arg_named_anything_else_still_reaches_the_pine_walk(
        count_where_vocab):
    """The Pine walk keys its condition handling on the arg's declared TYPE.

    _term gated on the literal key "cond" twice: once to keep the condition
    out of the generic args merge, and once to decide whether to lower it into
    call.source. A condition-taking term whose arg is named anything else
    therefore reached its emit with source='' and a raw condition dict sitting
    in call.args, which is silent: the emit writes an empty call rather than
    refusing, so the artifact renders and the chart reads nothing.
    """
    spec = {"version": 2, "name": "probe", "timeframe": "15m",
            "long": {"all": [{"lhs": {"prim": "count_where",
                                      "when": CLOSE_OVER_OPEN},
                              "op": ">", "rhs": 0}]}}
    program = lower_pine(spec, count_where_vocab)
    assert ("nk_count_where_1 = nk_bars_since(close > open)"
            in program.calculations)

    # And the condition did not ALSO land in the args merge. Read off the
    # TermCall rather than off program.inputs: an earlier draft asserted that
    # no input was named "when", reasoning that the raw dict "would have become
    # an input the settings dialog cannot render". It would not. An arg becomes
    # a PineInput only where an emit calls ctx.arg(call, name), and no emit
    # asks that of a condition arg, so reverting the merge half left the
    # program byte-identical and the assertion passed either way. The dict does
    # reach call.args, which is where the claim is actually observable.
    seen = {}
    term = count_where_vocab.primitives["count_where"]

    def spy(ctx, call):
        seen["args"] = dict(call.args)
        return term.pine.emit(ctx, call)

    spied = core_vocabulary().with_terms(dataclasses.replace(
        term, pine=PineLowering(spy, helpers=term.pine.helpers)))
    lower_pine(spec, spied)
    assert seen, "the spy emit never ran, so the assertion below proves nothing"
    assert "when" not in seen["args"], seen["args"]


def test_a_gap_exists_only_once_its_third_candle_has_closed():
    # find_fvgs: the imbalance at i is read off i, i-1 and i-2, so the gap is
    # created only once its third candle has closed and never reaches forward
    # of the bar being tested.
    source = _source("nk_fvg_nearest")
    assert "low[back] > high[back + 2]" in source
    assert "high[back] < low[back + 2]" in source
    assert "min_size_atr * unit" in source


def test_a_gap_is_invalidated_by_the_same_boundary_tests_the_engine_uses():
    # find_fvgs: a LONG gap inverts when a later CLOSE is below its bottom and
    # fills when a later LOW reaches it; a SHORT gap mirrors on its top. Both
    # are read from bars after the gap only.
    source = _source("nk_fvg_nearest")
    assert "later_close_min < bottom" in source
    assert "later_low <= bottom" in source
    assert "later_close_max > top" in source
    assert "later_high >= top" in source
    assert "want_inverted ? inverted : (not inverted and not filled)" in source
    # The nearest gap to the last close wins.
    assert "math.min(math.abs(close - top), math.abs(close - bottom))" in source


def test_the_order_block_is_the_last_opposing_candle_before_the_displacement():
    # order_block: the most recent displacement candle in the window, then the
    # last opposing candle strictly before it, and only inside `lookback`.
    source = _source("nk_order_block")
    assert "for back = 0 to lookback - 1" in source
    assert "body >= body_atr * unit : body <= -body_atr * unit" in source
    assert "for back = shove + 1 to lookback - 1" in source
    assert "want_long ? body < 0 : body > 0" in source


def test_the_window_atr_is_the_mean_true_range_the_ict_helpers_use():
    # ict.primitives.atr: tr.dropna().tail(14).mean() over the LOOKBACK window,
    # so a window shorter than 15 bars averages what it has rather than
    # reaching outside itself.
    source = _source("nk_window_atr")
    assert "math.min(14, bars - 1)" in source
    assert "total / n" in source
    assert "ta.atr" not in source


@pytest.mark.parametrize("helper_id", sorted(HELPERS))
def test_no_helper_leans_on_a_look_alike_built_in(helper_id):
    # ta.pivothigh confirms on a different bar and carries forward differently,
    # ta.barssince and the session variables answer a different session, and
    # ta.vwap anchors where the exchange says. None of them is the engine's
    # measurement, and a substitution would pass every shape test above.
    source = _source(helper_id)
    for built_in in ("ta.pivothigh", "ta.pivotlow", "ta.barssince",
                     "ta.valuewhen", "session.", "ta.change("):
        assert built_in not in source, f"{helper_id} leans on {built_in}"


# -- history, warnings and the end-anchored cross --------------------------


def test_the_swing_history_reach_is_twice_its_confirmation_lag():
    # The pivot sits k bars back and its own left window reaches k further, so
    # the deepest offset is [2k]. Declaring only k would let TradingView size a
    # buffer the script reads past, and so would declaring 2k: what the script
    # owes is the BUFFER, and reading [20] needs 21 values.
    for name in ("swing_high", "swing_low", "leg_retrace"):
        program = _program({"prim": name})
        assert program.max_bars_back == 2 * core_vocabulary().primitives[
            name].args["k"][1] + 1 == 21


def test_an_end_anchored_window_declares_the_history_it_scans():
    # `lookback` counts BARS rather than an offset: the scan reads [0] through
    # [lookback - 1], so the count is already the buffer and one more would
    # declare a bar the script never indexes.
    for name in ("fvg_nearest", "order_block"):
        program = _program({"prim": name})
        assert program.max_bars_back == core_vocabulary().primitives[
            name].args["lookback"][1] == 200


def test_a_primitive_with_a_fixed_reach_declares_no_history():
    assert _program({"prim": "day_of_week"}).max_bars_back == 0
    assert _program({"prim": "prev_session_high"}).max_bars_back == 0


@pytest.mark.parametrize("name", sorted(core_vocabulary().primitives))
def test_no_primitive_warns_about_where_its_session_begins(name):
    """Every primitive, not a sample: the vocabulary itself is the list.

    Seven of these used to carry a warning saying that a chart's extended
    session opens at 04:00 while the engine's frame opens on the day's first
    actual print, so anything measured from a session's first bar or
    aggregated over its whole span differed between the two. That was true
    while the engine anchored on the first bar of the New York date. It no
    longer is: every session-scoped primitive is now anchored on the 09:30
    bell or aggregated over [09:30, 16:00) on both sides, so which pre-market
    bars a chart happens to print reaches nothing.

    The warning is gone rather than reworded, and this asserts the absence for
    the whole vocabulary so a new primitive cannot quietly reintroduce a
    fidelity gap that nobody wrote down. A term that genuinely diverges gets
    its own sentence naming what differs, the way emit_vwap and emit_obv do.
    """
    node = {"prim": name}
    if name == "bars_since":
        node["cond"] = CLOSE_OVER_OPEN
    assert _program(node).warnings == ()


DAILY_STAMP = ("A daily bar's timestamp is its session's start in New York, "
               "which is TradingView's convention for a US equity, so "
               "converting it gives the session's own weekday.")


def test_the_daily_weekday_states_the_bar_stamp_it_rests_on():
    # day_of_week reads a session frame's weekday off the UTC date of the
    # bar's label, and the Pine converts the bar's own timestamp to New York.
    # The two agree only while a daily bar is stamped at its session's start,
    # which is a fact about the chart rather than about the compiler, so it is
    # stated where a reader of the artifact looks for the others.
    program = _program({"prim": "day_of_week"}, "<", 4, "1d")
    assert DAILY_STAMP in program.assumptions


def test_an_intraday_chart_does_not_carry_the_daily_bar_assumption():
    # An intraday chart converts the bar's own timestamp, which is the
    # primitive's intraday reading exactly, and the spec's timeframe is pinned
    # by the first assumption, so the daily premise can never apply there.
    assert DAILY_STAMP not in _program({"prim": "day_of_week"}).assumptions
    assert DAILY_STAMP not in _program({"prim": "prev_session_high"}, ">", 0,
                                       "1d").assumptions


def test_a_cross_against_an_end_anchored_level_compares_both_bars_to_it():
    # frame_eval._cross_prev broadcasts an end-anchored operand rather than
    # shifting it: between two bars the nearest gap can become a DIFFERENT gap
    # at a different price, so the level one bar ago is another object, not
    # this level's history. ta.crossover shifts both sides, which is the
    # reading the engine deliberately does not use.
    # The condition carries its own parentheses: `and` binds looser than a
    # comparison, so a group joining it with `or` would otherwise mis-associate.
    program = _program({"src": "close"}, "crosses_above",
                       {"prim": "fvg_nearest", "field": "bottom"})
    assert ("nk_long_entry = ((close[1] <= nk_fvg_nearest_1_bottom and "
            "close > nk_fvg_nearest_1_bottom))") in program.calculations
    assert not any("ta.crossover(" in c for c in program.calculations)
    below = _program({"src": "close"}, "crosses_below", {"prim": "order_block"})
    assert ("nk_long_entry = ((close[1] >= nk_order_block_1_top and "
            "close < nk_order_block_1_top))") in below.calculations


def test_a_computed_cross_operand_is_named_before_its_history_is_read():
    # Pine indexes an identifier, not a parenthesized expression, so the side
    # whose previous bar the broadcast cross reads has to be bound first.
    program = _program({"op": "-", "args": [{"src": "high"}, {"src": "low"}]},
                       "crosses_above", {"prim": "order_block"})
    assert "nk_long_all_0_lhs = (high - low)" in program.calculations
    assert ("nk_long_entry = ((nk_long_all_0_lhs[1] <= nk_order_block_1_top and "
            "nk_long_all_0_lhs > nk_order_block_1_top))") in program.calculations


def test_an_ordinary_cross_still_reads_the_built_in():
    # The broadcast is the end-anchored reading only. Everything else shifts,
    # which is exactly what ta.crossover does.
    program = _program({"ind": "rsi"}, "crosses_above", 30)
    assert ("nk_long_entry = (ta.crossover(nk_rsi_1, nk_long_all_0_rhs))"
            in program.calculations)


def test_a_primitive_reads_the_same_program_on_every_compilation():
    spec = _spec({"prim": "fvg_nearest"}, ">", 0)
    assert lower_pine(spec) == lower_pine(spec)
