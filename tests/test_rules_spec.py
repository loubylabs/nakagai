import inspect
from datetime import time
from pathlib import Path

import numpy as np
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.strategies.rules import (
    canonical_spec, describe_spec, spec_hash, validate_spec,
)
from nakagai.strategies.rules.canon import canonical_expr
from nakagai.strategies.rules import spec as rules_spec
from nakagai.strategies.rules.spec import (
    MAX_DEPTH, TIMEFRAMES, _expr_text, group_text, validate_condition_group)
from nakagai.strategies.rules.strategy import (
    expression_reference_pairs, spec_reference_pairs,
)
from nakagai.strategies.rules.vocabulary import (
    Term, core_vocabulary,
)
from nakagai.strategies.rules.windows import PRIOR_DAY, WindowSpec

ORB = {
    "version": 2, "name": "orb-volume", "timeframe": "15m",
    "long": {"all": [
        {"lhs": {"src": "close"}, "op": "crosses_above",
         "rhs": {"prim": "gap_pct"}},
        {"lhs": {"src": "volume"}, "op": ">",
         "rhs": {"op": "*", "args": [1.5, {"ind": "sma", "n": 20, "of": {"src": "volume"}}]}},
        {"lhs": {"src": "close", "tf": "1d"}, "op": ">", "rhs": {"ind": "sma", "n": 50}},
    ]},
    "exits": {"time_stop": {"bars": 16},
              "trailing": {"kind": "atr", "n": 14, "mult": 2.5}},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

RELATIVE_SCOPE_GROUP = {
    "all": [
        {
            "lhs": {"src": "close"},
            "op": ">",
            "rhs": {"src": "close", "sym": "SPY"},
        },
        {
            "lhs": {
                "ind": "sma", "n": 20, "of": {"src": "close"},
                "sym": "SPY", "tf": "15m",
            },
            "op": ">",
            "rhs": {
                "ind": "sma", "n": 20,
                "of": {"src": "close", "sym": "QQQ", "tf": "1d"},
                "sym": "SPY", "tf": "15m",
            },
        },
    ],
}

RELATIVE_SCOPE_SPEC = {
    "version": 2,
    "name": "relative_scope",
    "timeframe": "15m",
    "long": RELATIVE_SCOPE_GROUP,
    "risk": {
        "stop": {"kind": "atr", "n": 14, "mult": 2.0},
        "target": {"kind": "rr", "rr": 2.0},
    },
}

LOW_IEX_DISCLOSURE = "US-equity extended-hours IEX data can be sparse."
LONDON = WindowSpec(
    "london", "Europe/London", time(8), time(16, 30), "weekday", "low_iex")
NY_AM = WindowSpec(
    "ny_am", "America/New_York", time(9, 30), time(12),
    "xnys_session", "standard")
NY_OPEN_15 = WindowSpec(
    "ny_open_15", "America/New_York", time(9, 30), time(9, 45),
    "xnys_session", "standard")
NY_OPEN_30 = WindowSpec(
    "ny_open_30", "America/New_York", time(9, 30), time(10),
    "xnys_session", "standard")
WINDOW_VOCABULARY = core_vocabulary().with_windows(
    LONDON, NY_AM, NY_OPEN_15, NY_OPEN_30, PRIOR_DAY)


def _rule_with(expr: dict, timeframe: str = "15m") -> dict:
    return {
        "version": 2,
        "name": "window-contract",
        "timeframe": timeframe,
        "long": {"all": [
            {"lhs": {"src": "close"}, "op": ">", "rhs": expr},
        ]},
        "risk": ORB["risk"],
    }


@pytest.mark.parametrize("expr,timeframe", [
    ({"ind": "highest", "of": {"src": "high"}, "window": "london"},
     "15m"),
    ({"ind": "last", "of": {"src": "close"}, "window": "prior_day"},
     "15m"),
    ({"ind": "highest", "of": {"src": "high"}, "tf": "15m",
      "window": "ny_open_15"}, "1h"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "ny_open_15"}, "15m"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "prior_day"}, "1d"),
], ids=["current", "required", "own-tf", "equal-width", "daily-prior"])
def test_rule_spec_accepts_every_window_contract(expr, timeframe):
    assert validate_spec(
        _rule_with(expr, timeframe), WINDOW_VOCABULARY) == []


@pytest.mark.parametrize("expr,timeframe,expected", [
    ({"ind": "highest", "of": {"src": "high"}, "window": "unknown"},
     "15m", "unknown window 'unknown'"),
    ({"src": "high", "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"op": "max", "args": [{"src": "high"}, 1], "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"prim": "gap_pct", "window": "london"},
     "15m", "window is only valid on an aggregate indicator"),
    ({"ind": "rsi", "window": "london"},
     "15m", "rsi does not support window aggregation"),
    ({"ind": "first", "of": {"src": "open"}},
     "15m", "first requires window"),
    ({"ind": "highest", "of": {"src": "high"}, "n": 20,
      "window": "london"},
     "15m", "highest cannot combine n with window"),
    ({"ind": "highest", "of": {"src": "high"},
      "window": "ny_open_30"},
     "1h", "window 'ny_open_30' spans 30 minutes, narrower than '1h' bars "
           "(60 minutes)"),
    ({"ind": "highest", "of": {"src": "high"}, "window": "ny_am"},
     "1d", "window 'ny_am' is intraday and cannot be resolved from "
           "session-aligned '1d' bars"),
], ids=[
    "unknown", "source", "math", "primitive", "non-aggregate",
    "required", "scope-conflict", "wide-fixed-frame", "daily-current",
])
def test_rule_spec_refuses_every_invalid_window_shape(expr, timeframe, expected):
    errors = validate_spec(_rule_with(expr, timeframe), WINDOW_VOCABULARY)
    assert errors == [f"long.all[0].rhs: {expected}"]


def test_windowed_canonical_form_carries_scope_without_a_rolling_default():
    node = {"ind": "highest", "of": {"src": "high"}, "window": "london"}
    assert canonical_expr(node, WINDOW_VOCABULARY) == {
        "ind": "highest", "of": {"src": "high"}, "window": "london",
    }


def test_adding_windows_does_not_move_an_unwindowed_spec_hash():
    assert spec_hash(ORB, core_vocabulary()) == spec_hash(ORB, WINDOW_VOCABULARY)


def test_window_readback_names_scope_and_discloses_only_low_iex_rows():
    london = _rule_with(
        {"ind": "highest", "of": {"src": "high"}, "window": "london"})
    standard = _rule_with(
        {"ind": "highest", "of": {"src": "high"}, "window": "ny_am"})

    london_group = group_text(london["long"], WINDOW_VOCABULARY)
    assert "highest(of=high) over london" in london_group
    assert LOW_IEX_DISCLOSURE in london_group
    assert LOW_IEX_DISCLOSURE in describe_spec(london, WINDOW_VOCABULARY)

    standard_group = group_text(standard["long"], WINDOW_VOCABULARY)
    assert "highest(of=high) over ny_am" in standard_group
    assert LOW_IEX_DISCLOSURE not in standard_group
    assert LOW_IEX_DISCLOSURE not in describe_spec(standard, WINDOW_VOCABULARY)


def test_valid_v2_spec_passes():
    assert validate_spec(ORB) == []


def test_a_four_hour_spec_validates():
    """4h is a real timeframe, derived from cached 1h bars (nakagai/data/
    resample.py), so the grammar accepts it on the spec and on a leaf."""
    wide_orb = {**ORB, "timeframe": "4h", "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 1}]}}
    assert validate_spec(wide_orb) == []
    on_a_leaf = {**ORB, "long": {"all": [{"lhs": {"src": "close", "tf": "4h"},
                                          "op": ">", "rhs": {"ind": "sma", "n": 20}}]}}
    assert validate_spec(on_a_leaf) == []


def _custom_numeric_vocabulary(*, default):
    base = core_vocabulary()
    replacement = Term(
        "custom_numeric", "primitive",
        {"n": (-(10 ** 401), 10 ** 401)},
        {"n": default}, lambda *_args: None,
    )
    return base.with_terms(replacement)


@pytest.mark.parametrize("node", [
    {"prim": "custom_numeric", "n": 10 ** 400},
    {"prim": "custom_numeric"},
], ids=["explicit", "default"])
def test_numeric_injected_terms_reject_values_without_a_canonical_form(node):
    vocabulary = _custom_numeric_vocabulary(default=10 ** 400)
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": node}]}}
    errs = validate_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1, errs
    assert "custom_numeric.n has invalid argument rule" in errs[0]


@pytest.mark.parametrize("value", [
    9007199254740993,
    -9007199254740993,
], ids=["positive", "negative"])
def test_integers_that_float_canonicalization_rounds_are_rejected(value):
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": value}]}}
    errs = validate_spec(spec)
    assert len(errs) == 1 and "number is out of range" in errs[0], errs


def test_adjacent_large_integers_cannot_collapse_to_one_hash():
    exact = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 9007199254740992}]}}
    rounded = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">", "rhs": 9007199254740993}]}}
    assert validate_spec(exact) == []
    errs = validate_spec(rounded)
    assert len(errs) == 1 and "number is out of range" in errs[0], errs


def test_a_required_numeric_argument_is_refused_before_evaluation():
    vocabulary = core_vocabulary().with_terms(
        Term("required_numeric", "primitive", {"n": (1, 10)}, {},
             lambda *_args: None))
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "required_numeric"}}]}}
    errs = validate_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "required_numeric needs n" in errs[0], errs


def test_an_invalid_injected_choice_default_is_refused_centrally():
    vocabulary = core_vocabulary().with_terms(
        Term("invalid_choice", "primitive", {"side": ("long", "short")},
             {"side": "sideways"}, lambda *_args: None))
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "invalid_choice"}}]}}
    errs = validate_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "invalid_choice.side default" in errs[0], errs


@pytest.mark.parametrize("rule", [
    {"n": ("low", "high")},
    {"n": "malformed"},
], ids=["choice-is-valid-but-no-numeric", "malformed"])
def test_a_non_numeric_argument_rule_is_reported_without_raising(rule):
    vocabulary = core_vocabulary().with_terms(
        Term("bad_rule", "primitive", rule, {}, lambda *_args: None))
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "bad_rule", "n": 1}}]}}
    errs = validate_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "bad_rule.n" in errs[0], errs


@pytest.mark.parametrize("bounds", [
    (-(10 ** 400), 10 ** 400),
    (float("-inf"), float("inf")),
], ids=["huge", "infinite"])
def test_invalid_injected_numeric_bounds_are_rejected_before_hashing(bounds):
    base = core_vocabulary()
    replacement = Term(
        "bad_bounds", "primitive", {"n": bounds}, {"n": 30},
        lambda *_args: None,
    )
    vocabulary = base.with_terms(replacement)
    spec = {**ORB, "long": {"all": [
        {"lhs": {"src": "close"}, "op": ">",
         "rhs": {"prim": "bad_bounds"}}]}}
    errs = validate_spec(spec, vocabulary=vocabulary)
    assert len(errs) == 1 and "invalid argument rule" in errs[0], errs


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


def _bars_since_spec(inner):
    return {"version": 2, "name": "b", "timeframe": "1h",
            "long": {"all": [{"lhs": {"prim": "bars_since", **inner},
                              "op": "<", "rhs": 5}]},
            "risk": ORB["risk"]}


def test_bars_since_missing_cond_entirely_is_refused():
    """The core term's ABSENT branch (`_check_args`), not guard 1."""
    errs = validate_spec(_bars_since_spec({}))
    assert any("bars_since needs cond" in e for e in errs), errs


@pytest.mark.parametrize("cond,shown", [
    (5, "5"),
    ("close > open", "'close > open'"),
    ([{"lhs": {"src": "close"}, "op": ">", "rhs": 1}], "[{'lhs'"),
    ({"lhs": {"src": "close"}, "op": ">"}, "{'lhs': {'src': 'close'}, 'op': '>'}"),
], ids=["int", "string", "list", "dict-missing-rhs"])
def test_bars_since_with_a_present_but_malformed_cond_is_refused(cond, shown):
    """Guard 1 on the SHIPPED term, reached through the real vocabulary rather
    than the count_where fixture. Each of these supplies `cond`, so the absent
    loop skips them and `_check_condition_arg` is the only thing that can
    refuse them. The string case is the one a hand-written spec produces, and
    the list case is what an NL compiler emits when it treats the arg as a
    group."""
    errs = validate_spec(_bars_since_spec({"cond": cond}))
    assert any("bars_since.cond must be a condition {lhs, op, rhs}, got " in e
               and shown in e for e in errs), errs


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
    {"prim": "minutes_into_session"},
    {"prim": "rvol", "sessions": 20},
]


def _daily(node):
    return {**DAILY, "long": {"all": [{"lhs": node, "op": ">", "rhs": 1}]}}


@pytest.mark.parametrize("node", INTRADAY_ONLY, ids=lambda n: n["prim"])
def test_an_intraday_only_primitive_is_refused_on_a_daily_driving_frame(node):
    """One daily bar IS the whole session, so these read the wrong shape.

    Every bar sits 0 minutes into its session, and rvol's same-clock-time
    bucket swallows the entire series.
    """
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


@pytest.mark.parametrize("tf", ["15m"])
@pytest.mark.parametrize("node", INTRADAY_ONLY, ids=lambda n: n["prim"])
def test_the_same_primitives_are_untouched_on_an_intraday_driving_frame(node, tf):
    assert validate_spec({**_daily(node), "timeframe": tf}) == []


@pytest.mark.parametrize("node", [
    {"prim": "minutes_into_session"},
    {"prim": "rvol", "sessions": 20},
], ids=lambda n: n["prim"])
def test_unrelated_intraday_primitives_are_untouched_on_an_hourly_frame(node):
    assert validate_spec({**_daily(node), "timeframe": "1h"}) == []


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
    assert "gap_pct" in text
    assert "Stop:" in text and "Target:" in text
    assert "time stop" in text.lower()


@pytest.mark.parametrize("node", [
    {"src": "close", "sym": "SPY"},
    {"ind": "sma", "n": 20, "sym": "SPY"},
    {"prim": "gap_pct", "sym": "SPY"},
    {"op": "+", "args": [
        {"src": "close", "sym": "SPY"},
        {"src": "close", "sym": "QQQ"},
    ]},
], ids=["source", "indicator", "primitive", "math-children"])
def test_symbol_scope_is_accepted_on_scoped_expressions(node):
    assert validate_spec(_rule_with(node)) == []


@pytest.mark.parametrize("value", [
    "spy", " SPY", "", "123", "ABCDEFGHIJK", "BRK/B", "NASDAQ:QQQ",
    "SP'Y", "ÅBC", None, 7,
], ids=[
    "lowercase", "whitespace", "empty", "leading-digit", "too-long",
    "slash", "colon", "quote", "non-ascii", "none", "number",
])
def test_symbol_scope_rejects_noncanonical_shapes(value):
    errors = validate_spec(_rule_with({"src": "close", "sym": value}))
    assert errors == [
        "long.all[0].rhs: sym must match [A-Z][A-Z0-9.-]{0,9}, "
        f"got {value!r}"
    ]


def test_math_and_rule_roots_refuse_symbol_scope():
    math_errors = validate_spec(_rule_with(
        {"op": "+", "args": [{"src": "close"}, 1], "sym": "SPY"}))
    assert math_errors == ["long.all[0].rhs: math nodes take only op/args"]
    root_errors = validate_spec({**ORB, "sym": "SPY"})
    assert root_errors == ["unknown keys ['sym']"]


@pytest.mark.parametrize(("node", "text"), [
    ({"src": "close"}, "close"),
    ({"src": "close", "sym": "SPY"}, "SPY:close"),
    ({"src": "close", "sym": "QQQ", "tf": "1d"}, "QQQ:close[1d]"),
    ({"ind": "sma", "n": 20, "sym": "SPY", "tf": "15m"},
     "SPY:sma(20)[15m]"),
    ({"ind": "sma", "n": 20, "sym": "SPY", "tf": "15m",
      "of": {"src": "close", "sym": "QQQ", "tf": "1d"}},
     "SPY:sma(20, of=QQQ:close[1d])[15m]"),
    ({"op": "+", "args": [
        {"src": "close", "sym": "SPY"},
        {"src": "close", "sym": "QQQ", "tf": "1d"},
     ]}, "(SPY:close + QQQ:close[1d])"),
    ({"prim": "gap_pct", "sym": "SPY", "tf": "15m"},
     "SPY:gap_pct[15m]"),
], ids=[
    "driving-source", "source", "source-timeframe", "indicator",
    "nested-symbol-and-timeframe", "math", "primitive",
])
def test_symbol_scope_readback_is_explicit(node, text):
    assert _expr_text(node, core_vocabulary()) == text


def test_explicit_default_source_readback_is_controlled_by_term_metadata():
    explicit_source = Term(
        "kama", "series",
        {"n": (1, 100), "fast": (1, 100), "slow": (1, 100)}, {},
        lambda series, args: series,
        render_explicit_source=True,
    )
    vocabulary = core_vocabulary().with_terms(explicit_source)

    assert _expr_text(
        {"ind": "sma", "n": 20, "of": {"src": "close"}},
        vocabulary,
    ) == "sma(20)"
    assert _expr_text(
        {"ind": "kama", "n": 10, "fast": 2, "slow": 30,
         "of": {"src": "close"}},
        vocabulary,
    ) == "kama(10, 2, 30, of=close)"
    assert _expr_text(
        {"ind": "kama", "n": 10, "fast": 2, "slow": 30},
        vocabulary,
    ) == "kama(10, 2, 30)"


def test_explicit_source_rendering_is_core_neutral_term_metadata():
    arbitrary = Term(
        "external_series", "series", {"n": (1, 100)}, {},
        lambda series, args: series,
        render_explicit_source=True,
    )
    vocabulary = core_vocabulary().with_terms(arbitrary)
    assert _expr_text(
        {"ind": "external_series", "n": 7, "of": {"src": "close"}},
        vocabulary,
    ) == "external_series(7, of=close)"

    renderer_source = inspect.getsource(rules_spec._expr_text)
    assert "nakagai_platform" not in renderer_source
    assert "'kama'" not in renderer_source
    assert '"kama"' not in renderer_source
    assert "adapter" not in renderer_source


def test_relative_scope_description_matches_the_frozen_readback():
    assert validate_spec(RELATIVE_SCOPE_SPEC) == []
    assert describe_spec(RELATIVE_SCOPE_SPEC) == (
        'Strategy "relative_scope" on 15m bars.\n'
        "Enter long when ALL of:\n"
        "  - close is above SPY:close\n"
        "  - SPY:sma(20)[15m] is above "
        "SPY:sma(20, of=QQQ:close[1d])[15m]\n"
        "Stop: 2x ATR(14) from entry. Target: 2x the risked distance."
    )


def test_expression_reference_pairs_follow_lexical_symbol_and_timeframe_scope():
    node = {
        "ind": "sma", "n": 20, "sym": "SPY", "tf": "15m",
        "of": {
            "op": "+", "args": [
                {"src": "close"},
                {"src": "close", "sym": "QQQ", "tf": "1d"},
            ],
        },
    }
    assert expression_reference_pairs(node, "1h") == (
        ("QQQ", "1d"), ("SPY", "15m"),
    )


def test_child_symbol_override_inherits_its_parent_timeframe():
    node = {
        "ind": "sma", "n": 20, "tf": "1d",
        "of": {"src": "close", "sym": "QQQ"},
    }
    assert expression_reference_pairs(node, "15m") == (("QQQ", "1d"),)


def test_child_timeframe_override_inherits_its_parent_symbol():
    node = {
        "ind": "sma", "n": 20, "sym": "SPY",
        "of": {"src": "close", "tf": "1d"},
    }
    assert expression_reference_pairs(node, "15m") == (
        ("SPY", "15m"), ("SPY", "1d"),
    )


def test_spec_reference_pairs_cover_entries_exits_and_condition_arguments():
    spec = {
        **RELATIVE_SCOPE_SPEC,
        "exits": {"exit": {"all": [{
            "lhs": {
                "prim": "bars_since",
                "cond": {
                    "lhs": {"src": "close", "sym": "SPY", "tf": "15m"},
                    "op": ">",
                    "rhs": {"src": "open"},
                },
            },
            "op": ">",
            "rhs": 2,
        }]}},
    }
    assert validate_spec(spec) == []
    assert spec_reference_pairs(spec) == (("QQQ", "1d"), ("SPY", "15m"))


def test_canonical_hash_stable_and_name_free():
    import copy
    a, b = copy.deepcopy(ORB), copy.deepcopy(ORB)
    b["name"] = "totally different"
    b["long"]["all"][2]["rhs"] = {"ind": "sma", "n": 50, "of": {"src": "close"}}  # explicit default `of`
    assert spec_hash(a) == spec_hash(b)
    assert len(spec_hash(a)) == 64
    assert "name" not in canonical_spec(a)


def test_no_symbol_scope_keeps_the_existing_canonical_identity():
    assert spec_hash(ORB) == (
        "591450193c6785a71b7cd369ab7ddad38ddbb428c3f8bcae75ab19e2fd6563ec"
    )
    assert "sym" not in repr(canonical_spec(ORB))


def test_explicit_symbol_scope_is_canonical_and_identity_bearing():
    left = _rule_with({"src": "close", "sym": "SPY"})
    right = _rule_with({"sym": "SPY", "src": "close"})
    driving = _rule_with({"src": "close"})
    assert validate_spec(left) == []
    assert canonical_expr(left["long"]["all"][0]["rhs"], core_vocabulary()) == {
        "src": "close", "sym": "SPY",
    }
    assert spec_hash(left) == spec_hash(right)
    assert spec_hash(left) != spec_hash(driving)


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


def test_a_second_condition_taking_term_refuses_an_absent_condition_arg(
        count_where_vocab):
    """The ABSENT branch, which is _check_args' condition loop, NOT guard 1.

    A condition-typed arg may declare no default (N3-D13), so its absence is
    an error rather than a fallback. This test is named for the branch it
    reaches rather than for the guard it was once named for: the two branches
    used to emit the identical message, this was the only test either had, and
    deleting guard 1's `errs.append` outright left all 1169 tests green. The
    two shape cases below are what actually reach guard 1, and they are told
    apart from this one by the message as well as by the input.
    """
    errs = validate_spec(_count_where_spec({"prim": "count_where"}),
                         count_where_vocab)
    assert any("count_where needs when" in e for e in errs), errs


def test_a_second_condition_taking_term_inherits_guard_1_shape_non_dict(
        count_where_vocab):
    """Guard 1 proper. The arg is PRESENT, so `_check_args`' absent loop skips
    it and only `_check_condition_arg`'s shape refusal can produce this."""
    node = {"prim": "count_where", "when": 5}
    errs = validate_spec(_count_where_spec(node), count_where_vocab)
    assert any("count_where.when must be a condition {lhs, op, rhs}, got 5" in e
               for e in errs), errs


def test_a_second_condition_taking_term_inherits_guard_1_shape_missing_a_key(
        count_where_vocab):
    """Guard 1's other half: a dict, but not a condition. Left unrefused this
    validates clean, is saved, and raises inside FrameEval.condition_series at
    backtest and scan time, where detect_events swallows it and the symbol
    reports zero events."""
    node = {"prim": "count_where", "when": {"lhs": {"src": "close"}, "op": ">"}}
    errs = validate_spec(_count_where_spec(node), count_where_vocab)
    assert any("count_where.when must be a condition {lhs, op, rhs}, got "
               "{'lhs': {'src': 'close'}, 'op': '>'}" in e for e in errs), errs


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


# --- `not` (N3-D6, N3-D7, N3-D11) -------------------------------------------

LEAF_A = {"lhs": {"src": "close"}, "op": ">", "rhs": 1000}
LEAF_B = {"lhs": {"src": "volume"}, "op": ">", "rhs": 1_000_000}
NOT_RISK = ORB["risk"]


def test_not_validates_over_a_group():
    spec = {"version": 2, "name": "x", "timeframe": "15m",
            "long": {"not": {"any": [LEAF_A, LEAF_B]}}, "risk": NOT_RISK}
    assert validate_spec(spec) == []


def test_not_over_a_bare_leaf_is_refused_naming_the_accepted_form():
    """N3-D6: one accepted shape."""
    spec = {"version": 2, "name": "x", "timeframe": "15m",
            "long": {"not": LEAF_A}, "risk": NOT_RISK}
    errs = validate_spec(spec)
    assert any("expected a group" in e and '"all"' in e for e in errs), errs


def test_not_may_contain_not_directly():
    """N3-D7."""
    spec = {"version": 2, "name": "x", "timeframe": "15m",
            "long": {"not": {"not": {"all": [LEAF_A]}}}, "risk": NOT_RISK}
    assert validate_spec(spec) == []


def test_not_counts_against_max_depth():
    group = {"all": [LEAF_A]}
    for _ in range(MAX_DEPTH + 2):
        group = {"not": group}
    spec = {"version": 2, "name": "x", "timeframe": "1h", "long": group,
            "risk": NOT_RISK}
    errs = validate_spec(spec)
    assert any("group depth exceeds" in e for e in errs), errs


def test_not_readback_matches_the_frozen_shape_flat():
    """N3-D11, example 1, verbatim."""
    text = group_text({"not": {"any": [LEAF_A, LEAF_B]}})
    assert text == ("NOT ANY of:\n"
                    "  - close is above 1000\n"
                    "  - volume is above 1e+06")


def test_not_readback_matches_the_frozen_shape_nested():
    """N3-D11, example 2, verbatim. The one that distinguishes a renderer that
    scopes correctly from one that prefixes NOT onto the whole tree."""
    leaf_c = {"lhs": {"src": "close"}, "op": "<", "rhs": 3}
    text = group_text({"all": [{"not": {"any": [LEAF_A, LEAF_B]}}, leaf_c]})
    assert text == ("ALL of:\n"
                    "  NOT ANY of:\n"
                    "    - close is above 1000\n"
                    "    - volume is above 1e+06\n"
                    "  - close is below 3")


def test_a_nested_group_two_levels_deep_indents_by_exactly_two_per_level():
    """The pre-existing bug the frozen goldens exposed, pinned on a plain
    all/any tree with no `not` in it: the string-.replace() scheme
    double-counted and put the grandchild leaf six spaces deep."""
    text = group_text({"all": [{"any": [LEAF_A]}, LEAF_B]})
    assert text == ("ALL of:\n"
                    "  ANY of:\n"
                    "    - close is above 1000\n"
                    "  - volume is above 1e+06")


def test_double_negation_canonicalizes_structurally_not_simplified():
    """N3-D7's second half: {"not": {"not": G}} hashes differently from G."""
    spec_dbl = {"version": 2, "name": "x", "timeframe": "1h",
                "long": {"not": {"not": {"all": [LEAF_A]}}}, "risk": NOT_RISK}
    spec_plain = {"version": 2, "name": "x", "timeframe": "1h",
                  "long": {"all": [LEAF_A]}, "risk": NOT_RISK}
    assert spec_hash(spec_dbl) != spec_hash(spec_plain)
