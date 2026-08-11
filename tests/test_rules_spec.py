import inspect
from pathlib import Path

import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules import (
    canonical_spec, describe_spec, spec_hash, validate_spec,
)
from nakagai.strategies.rules import spec as rules_spec
from nakagai.strategies.rules.spec import (
    TIMEFRAMES, _expr_text, validate_condition_group)

ORB = {
    "version": 2, "name": "orb-volume", "timeframe": "15m",
    "long": {"all": [
        {"lhs": {"src": "close"}, "op": "crosses_above",
         "rhs": {"prim": "opening_range_high", "minutes": 30}},
        {"lhs": {"src": "volume"}, "op": ">",
         "rhs": {"op": "*", "args": [1.5, {"ind": "sma", "n": 20, "of": {"src": "volume"}}]}},
        {"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": {"ind": "sma", "n": 50}},
    ]},
    "exits": {"time_stop": {"bars": 16},
              "trailing": {"kind": "atr", "n": 14, "mult": 2.5}},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}


def test_valid_v2_spec_passes():
    assert validate_spec(ORB) == []


def test_a_four_hour_spec_validates():
    """4h is a real timeframe, derived from cached 1h bars (nakagai/data/
    resample.py), so the grammar accepts it on the spec and on a leaf."""
    assert validate_spec({**ORB, "timeframe": "4h"}) == []
    on_a_leaf = {**ORB, "long": {"all": [{"lhs": {"src": "close", "tf": "4h"},
                                          "op": ">", "rhs": {"ind": "sma", "n": 20}}]}}
    assert validate_spec(on_a_leaf) == []


def test_the_grammar_takes_its_timeframes_from_the_schema():
    """One source of truth: spec.TIMEFRAMES is the engine's axis, not a second
    hardcoded copy that has to be edited alongside it."""
    assert TIMEFRAMES == DEFAULT_TIMEFRAMES.all == ("15m", "1h", "4h", "1d")


def test_version_required_and_v1_rejected():
    assert any("version" in e for e in validate_spec({**ORB, "version": 1}))
    spec = dict(ORB); spec.pop("version")
    assert any("version" in e for e in validate_spec(spec))


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"src": "closs"}, "op": ">", "rhs": 1}), "closs"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"ind": "nope"}, "op": ">", "rhs": 1}), "nope"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"op": "%", "args": [1, 2]}, "op": ">", "rhs": 1}), "%"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"prim": "wat"}, "op": ">", "rhs": 1}), "wat"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"src": "close", "tf": "2h"}, "op": ">", "rhs": 1}), "2h"),
    (lambda s: s.__setitem__("timeframe", "2h"), "timeframe"),
])
def test_rejections_name_the_problem(mutate, needle):
    import copy
    spec = copy.deepcopy(ORB)
    mutate(spec)
    errs = validate_spec(spec)
    assert errs and any(needle in e for e in errs)


def test_error_paths_are_precise():
    import copy
    spec = copy.deepcopy(ORB)
    spec["long"]["all"][1]["rhs"]["args"][1]["n"] = 9999
    errs = validate_spec(spec)
    assert any(e.startswith("long.all[1].rhs") for e in errs)


def test_depth_and_node_caps():
    deep = {"src": "close"}
    for _ in range(9):
        deep = {"op": "abs", "args": [deep]}
    spec = {"version": 2, "name": "d", "timeframe": "1h",
            "long": {"all": [{"lhs": deep, "op": ">", "rhs": 1}]},
            "risk": ORB["risk"]}
    assert any("depth" in e for e in validate_spec(spec))


def test_cross_lhs_must_be_series():
    spec = {"version": 2, "name": "c", "timeframe": "1h",
            "long": {"all": [{"lhs": 5, "op": "crosses_above", "rhs": {"src": "close"}}]},
            "risk": ORB["risk"]}
    assert any("cross" in e.lower() for e in validate_spec(spec))


def test_exits_validation():
    import copy
    spec = copy.deepcopy(ORB)
    spec["exits"]["time_stop"]["bars"] = 0
    assert any("time_stop" in e for e in validate_spec(spec))
    spec = copy.deepcopy(ORB)
    spec["exits"]["breakeven_at"] = {"rr": 50}
    assert any("breakeven_at" in e for e in validate_spec(spec))
    spec = copy.deepcopy(ORB)
    spec["exits"]["exit"] = {"any": [{"lhs": {"ind": "rsi", "n": 14}, "op": ">", "rhs": 70}]}
    assert validate_spec(spec) == []


def _set_stop_not_dict(s):
    s["risk"]["stop"] = "tight"


def _set_target_not_dict(s):
    s["risk"]["target"] = ["not", "a", "dict"]


def _set_stop_mult_string(s):
    s["risk"]["stop"]["mult"] = "two"


def _set_trailing_mult_string(s):
    s["exits"]["trailing"]["mult"] = "big"


def _set_stop_pct_string(s):
    s["risk"]["stop"] = {"kind": "percent", "pct": "lots"}


def _set_trailing_unknown_key(s):
    s["exits"]["trailing"] = {"kind": "percent", "n": 5}


@pytest.mark.parametrize("mutate,needle", [
    (_set_stop_not_dict, "risk.stop"),
    (_set_target_not_dict, "risk.target"),
    (_set_stop_mult_string, "risk.stop.mult"),
    (_set_trailing_mult_string, "exits.trailing.mult"),
    (_set_stop_pct_string, "risk.stop.pct"),
    (_set_trailing_unknown_key, "exits.trailing"),
])
def test_validator_never_raises_on_malformed_shapes(mutate, needle):
    """Malformed shapes fed as raw user JSON (POST /api/strategy-configs) or
    emitted by the NL compiler's model must come back as validation errors,
    never as an unhandled AttributeError/ValueError that turns into a 500 or
    aborts the compiler's retry loop into a 503."""
    import copy
    spec = copy.deepcopy(ORB)
    mutate(spec)
    errs = validate_spec(spec)
    assert errs and any(needle in e for e in errs)


def test_bars_since_condition_rejects_cross_ops():
    spec = {"version": 2, "name": "b", "timeframe": "1h",
            "long": {"all": [{"lhs": {"prim": "bars_since",
                                       "cond": {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": 1}},
                              "op": "<", "rhs": 5}]},
            "risk": ORB["risk"]}
    assert any("bars_since" in e for e in validate_spec(spec))


FVG = {"prim": "fvg_nearest", "direction": "long", "field": "top"}
OB = {"prim": "order_block", "direction": "long", "field": "top"}


def test_a_cross_may_still_have_an_end_anchored_level_on_the_right():
    """The supported shape, and the one the catalog uses: price crossing a
    level. Both bars of the cross see the level known at the current bar."""
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": FVG}]}}
    assert validate_spec(spec) == []


@pytest.mark.parametrize("node", [FVG, OB], ids=lambda n: n["prim"])
def test_an_end_anchored_level_on_the_left_of_a_cross_is_rejected(node):
    """fvg_nearest and order_block are one level read from the tail of the
    frame, which is what Term.end_anchored means. crossed_above's scalar branch
    only ever covered the RHS and the old eval_condition returned False
    outright for a non-Series LHS, so this spec was permanently dead;
    _cross_prev is symmetric, so it would now fire. series_required has to say
    so."""
    spec = {**ORB, "long": {"all": [
        {"lhs": node, "op": "crosses_above", "rhs": {"src": "close"}}]}}
    errs = validate_spec(spec)
    assert any(node["prim"] in e and "series" in e for e in errs), errs


NEST = [
    ({"op": "*", "args": [FVG, 1.0]}, "one math op"),
    ({"op": "+", "args": [{"op": "*", "args": [OB, 1.0]}, 0.0]}, "two math ops"),
    ({"ind": "ema", "n": 3, "of": FVG}, "an indicator's `of`"),
]


@pytest.mark.parametrize("side", ["lhs", "rhs"])
@pytest.mark.parametrize("node,how", NEST, ids=[h for _, h in NEST])
def test_an_end_anchored_level_nested_in_a_cross_operand_is_rejected(node, how, side):
    """series_required only ever looked at the operand's TOP node, so every one
    of these validated, on either side. That is not the supported reading read
    from one level down, it is the other reading: _cross_prev matches on the
    node itself, so nesting hides the primitive and the whole computed series
    gets .shift(1). It is also span-dependent, since an end-anchored primitive
    is NaN outside its span, which is exactly what makes the bars_since case
    below illegal."""
    other = {"src": "close"}
    cond = {"lhs": node if side == "lhs" else other, "op": "crosses_above",
            "rhs": node if side == "rhs" else other}
    errs = validate_spec({**ORB, "long": {"all": [cond]}})
    assert any("nested inside a cross" in e for e in errs), errs


def test_a_bare_end_anchored_level_on_the_right_is_not_caught_by_the_nesting_check():
    """The nesting check must not swallow the one shape the catalog uses.
    fvg_bounce, ifvg_reversal, ob_bounce and smc_confluence all depend on it."""
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": "crosses_above", "rhs": FVG}]}}
    assert validate_spec(spec) == []


@pytest.mark.parametrize("node", [FVG, OB], ids=lambda n: n["prim"])
def test_bars_since_over_an_end_anchored_level_is_rejected(node):
    """bars_since ffills over the whole mask, so it reaches rows outside the
    span the end-anchored primitives are evaluated over. Outside it they are
    NaN and the condition False, so the same bar would count differently
    depending on which walk-forward window replayed it."""
    spec = {**ORB, "long": {"all": [
        {"lhs": {"prim": "bars_since",
                 "cond": {"lhs": {"src": "close"}, "op": ">", "rhs": node}},
         "op": "<", "rhs": 5}]}}
    errs = validate_spec(spec)
    assert any(node["prim"] in e and "bars_since" in e for e in errs), errs


def test_a_session_aligned_spec_may_not_reference_another_timeframe():
    """A daily bar's label carries its session date, not the 16:00 NY bell, so
    there is no honest cutoff for carrying an intraday series onto it. The
    evaluator refuses at runtime, per symbol, inside the screener's per-symbol
    try/except; saying it here puts it where the NL retry loop can read it."""
    spec = {**ORB, "timeframe": "1d", "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"ind": "sma", "n": 50, "tf": "1h"}}]}}
    errs = validate_spec(spec)
    assert any("session-aligned" in e for e in errs), errs


def test_a_session_aligned_spec_with_no_foreign_reference_is_fine():
    spec = {**ORB, "timeframe": "1d", "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": {"ind": "sma", "n": 50}}]}}
    assert validate_spec(spec) == []


def test_an_intraday_spec_may_still_reference_a_daily_bar():
    """ORB itself does: the destination is 15m, which has a close time."""
    assert validate_spec(ORB) == []


def test_the_session_aligned_refusal_follows_a_tf_qualified_bars_since():
    """A bars_since with a tf evaluates its condition on THAT frame, so the
    destination timeframe moves out from under the subtree. The spec here is
    15m, but the reference inside the condition lands on 1d."""
    spec = {**ORB, "timeframe": "15m", "long": {"all": [
        {"lhs": {"prim": "bars_since", "tf": "1d",
                 "cond": {"lhs": {"src": "close"}, "op": ">",
                          "rhs": {"src": "close", "tf": "1h"}}},
         "op": "<", "rhs": 5}]}}
    errs = validate_spec(spec)
    assert any("session-aligned" in e for e in errs), errs


def test_the_session_aligned_refusal_covers_the_exit_group():
    spec = {**ORB, "timeframe": "1d",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 1}]},
            "exits": {"exit": {"all": [
                {"lhs": {"src": "close"}, "op": "<",
                 "rhs": {"ind": "sma", "n": 20, "tf": "15m"}}]}}}
    errs = validate_spec(spec)
    assert any("exits.exit" in e and "session-aligned" in e for e in errs), errs


DAILY = {"version": 2, "name": "daily", "timeframe": "1d", "risk": ORB["risk"]}
INTRADAY_ONLY = [
    {"prim": "opening_range_high", "minutes": 30},
    {"prim": "opening_range_low", "minutes": 30},
    {"prim": "minutes_into_session"},
    {"prim": "rvol", "sessions": 20},
]


def _daily(node):
    return {**DAILY, "long": {"all": [{"lhs": node, "op": ">", "rhs": 1}]}}


@pytest.mark.parametrize("node", INTRADAY_ONLY, ids=lambda n: n["prim"])
def test_an_intraday_only_primitive_is_refused_on_a_daily_driving_frame(node):
    """One daily bar IS the whole session, so these read nothing: the opening
    range never elapses (NaN forever), every bar sits 0 minutes into its
    session, and rvol's same-clock-time bucket swallows the entire series. The
    old rule only refused a foreign `tf`, so a 1d spec validated clean and then
    read something that did not mean what its name says."""
    errs = validate_spec(_daily(node))
    assert any(e.startswith("long.all[0].lhs") and node["prim"] in e
               for e in errs), errs


def test_rvol_names_the_door_it_closes():
    """Refusing rvol on daily bars is a decision, not an oversight: there it
    collapses to a trailing-median daily volume ratio, a different measurement
    wearing the same name. The message has to say a daily reading needs its own
    primitive, because the NL compiler retries against this text."""
    errs = validate_spec(_daily({"prim": "rvol"}))
    assert any("rvol" in e and "own name" in e for e in errs), errs


def test_day_of_week_is_still_fine_on_a_daily_driving_frame():
    """The trap in this rule. turnaround_tuesday is a shipped 1d catalog play
    whose entire premise is day_of_week, and on daily bars that reading is
    RIGHT: a daily bar is one session, so its weekday is exactly the calendar
    identity the primitive promises. Only the foreign-`tf` rule refuses it."""
    assert validate_spec(_daily({"prim": "day_of_week"})) == []


@pytest.mark.parametrize("tf", ["15m", "1h"])
@pytest.mark.parametrize("node", INTRADAY_ONLY, ids=lambda n: n["prim"])
def test_the_same_primitives_are_untouched_on_an_intraday_driving_frame(node, tf):
    assert validate_spec({**_daily(node), "timeframe": tf}) == []


def test_the_refusal_follows_a_tf_that_moves_the_frame_under_a_subtree():
    """The spec is 15m, but the indicator's `tf` puts its `of` subtree on daily
    bars. The primitive carries no tf of its own, so the own-`tf` check cannot
    see this; the walker, which tracks the effective frame, can."""
    spec = {**DAILY, "timeframe": "15m", "long": {"all": [
        {"lhs": {"ind": "sma", "n": 5, "tf": "1d",
                 "of": {"prim": "minutes_into_session"}},
         "op": ">", "rhs": 1}]}}
    errs = validate_spec(spec)
    assert any(e.startswith("long.all[0].lhs.of") and "minutes_into_session" in e
               for e in errs), errs


def test_the_refusal_covers_the_exit_group_too():
    spec = {**DAILY, "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": 1}]},
            "exits": {"exit": {"all": [
                {"lhs": {"prim": "minutes_into_session"}, "op": ">", "rhs": 120}]}}}
    errs = validate_spec(spec)
    assert any(e.startswith("exits.exit") and "minutes_into_session" in e
               for e in errs), errs


def test_the_foreign_tf_rule_is_unchanged_by_the_driving_frame_rule():
    """Two rules over two different sets. A weekday read off a 1h frame inside
    a 15m spec is still a category error, and a 15m spec is still free to look
    at a daily close."""
    errs = validate_spec({**DAILY, "timeframe": "15m", "long": {"all": [
        {"lhs": {"prim": "day_of_week", "tf": "1h"}, "op": ">", "rhs": 1}]}})
    assert any("day_of_week is session-scoped and takes no tf" in e for e in errs)
    assert validate_spec(ORB) == []


def test_a_daily_screen_refuses_an_intraday_only_primitive():
    """The screener path: its whole schema is one condition group, and its tf
    defaults to 1d, so this is the surface most likely to hit the rule."""
    group = {"all": [{"lhs": {"prim": "rvol"}, "op": ">", "rhs": 2}]}
    errs = validate_condition_group(group, "conditions", tf="1d")
    assert any(e.startswith("conditions.all[0].lhs") and "rvol" in e
               for e in errs), errs
    assert validate_condition_group(group, "conditions", tf="1h") == []


def test_describe_mentions_the_pieces():
    text = describe_spec(ORB)
    assert "orb-volume" in text and "15m" in text
    assert "opening_range_high" in text or "opening range high" in text
    assert "Stop:" in text and "Target:" in text
    assert "time stop" in text.lower()


def test_canonical_hash_stable_and_name_free():
    import copy
    a, b = copy.deepcopy(ORB), copy.deepcopy(ORB)
    b["name"] = "totally different"
    b["long"]["all"][2]["rhs"] = {"ind": "sma", "n": 50, "of": {"src": "close"}}  # explicit default `of`
    assert spec_hash(a) == spec_hash(b)
    assert len(spec_hash(a)) == 64
    assert "name" not in canonical_spec(a)


def test_hash_changes_when_logic_changes():
    import copy
    b = copy.deepcopy(ORB)
    b["long"]["all"][2]["rhs"]["n"] = 200
    assert spec_hash(ORB) != spec_hash(b)


def test_exits_exit_group_canonicalized_in_hash():
    import copy
    a, b = copy.deepcopy(ORB), copy.deepcopy(ORB)
    a["exits"]["exit"] = {"any": [
        {"lhs": {"src": "close"}, "op": "<", "rhs": {"ind": "sma", "n": 20}}]}
    b["exits"]["exit"] = {"any": [
        {"lhs": {"src": "close"}, "op": "<",
         "rhs": {"ind": "sma", "n": 20, "of": {"src": "close"}}}]}
    assert validate_spec(a) == [] and validate_spec(b) == []
    assert spec_hash(a) == spec_hash(b)


def test_trailing_defaults_materialized_in_hash():
    import copy
    a, b = copy.deepcopy(ORB), copy.deepcopy(ORB)
    a["exits"]["trailing"] = {"kind": "atr"}
    b["exits"]["trailing"] = {"kind": "atr", "n": 14, "mult": 2.0}
    assert spec_hash(a) == spec_hash(b)


def test_trailing_mult_changes_hash():
    import copy
    b = copy.deepcopy(ORB)
    b["exits"]["trailing"]["mult"] = 3.0
    assert spec_hash(ORB) != spec_hash(b)


def test_check_expr_has_no_bespoke_bars_since_branch():
    """Acceptance item 1, half one."""
    src = inspect.getsource(rules_spec._check_expr)
    assert '"bars_since"' not in src


def test_spec_module_has_zero_bars_since_special_cases():
    """The literal acceptance-item-1 search: spec.py has no == "bars_since"
    left anywhere, validator or describe."""
    src = Path(rules_spec.__file__).read_text()
    assert '== "bars_since"' not in src


# A second condition-taking term, count_where, registered only by the
# count_where_vocab fixture and named nowhere in nakagai/. Its condition arg is
# called "when", so a validator keyed on the primitive's name or on the literal
# arg key "cond" reaches none of it. Together these are the acceptance claim
# N3-D5 makes: all four guards, and the readable rendering, come to a term the
# validator has never heard of, with no new validator code. The evaluation half
# lives in test_rules_vocabulary.py, beside the injection it proves.


def _count_where_spec(node):
    return {"version": 2, "name": "x", "timeframe": "15m",
            "long": {"all": [{"lhs": node, "op": ">", "rhs": 3}]},
            "risk": ORB["risk"]}


def test_a_second_condition_taking_term_validates_with_no_new_validator_code(
        count_where_vocab):
    """The positive control the four refusals below need: a well-formed use of
    this term is ACCEPTED, so each guard is refusing its own case rather than
    the validator refusing count_where on sight."""
    node = {"prim": "count_where",
            "when": {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}}
    assert validate_spec(_count_where_spec(node), count_where_vocab) == []


def test_a_second_condition_taking_term_inherits_guard_1_shape(count_where_vocab):
    """A condition-typed arg may declare no default (N3-D13), so its absence is
    an error rather than a fallback."""
    errs = validate_spec(_count_where_spec({"prim": "count_where"}),
                         count_where_vocab)
    assert any("count_where needs when" in e for e in errs), errs


def test_a_second_condition_taking_term_inherits_guard_2_no_cross_ops(
        count_where_vocab):
    node = {"prim": "count_where",
            "when": {"lhs": {"src": "close"}, "op": "crosses_above",
                     "rhs": {"src": "open"}}}
    errs = validate_spec(_count_where_spec(node), count_where_vocab)
    assert any("count_where.when conditions use comparison ops only" in e
               for e in errs), errs


def test_a_second_condition_taking_term_inherits_guard_3_no_end_anchored(
        count_where_vocab):
    node = {"prim": "count_where",
            "when": {"lhs": {"src": "close"}, "op": ">", "rhs": FVG}}
    errs = validate_spec(_count_where_spec(node), count_where_vocab)
    assert any("fvg_nearest is anchored to the end of the frame and cannot "
               "sit inside count_where.when" in e for e in errs), errs


def test_a_second_condition_taking_term_inherits_guard_4_no_session_scoped_with_tf(
        count_where_vocab):
    node = {"prim": "count_where", "tf": "1h",
            "when": {"lhs": {"prim": "day_of_week"}, "op": "<", "rhs": 1}}
    errs = validate_spec(_count_where_spec(node), count_where_vocab)
    assert any("day_of_week is session-scoped and cannot sit inside "
               "count_where.when with tf" in e for e in errs), errs


def test_describe_renders_a_second_condition_taking_terms_condition_readably(
        count_where_vocab):
    """Describe is what a user approves before saving or backtesting an
    imported or NL-built strategy. The generic args path stringifies a value
    with f"{v}", which on a condition dict prints its repr; the condition args
    have to reach _condition_text by type instead."""
    node = {"prim": "count_where",
            "when": {"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}}
    assert (_expr_text(node, count_where_vocab)
            == "count_where(5, close is above open)")
