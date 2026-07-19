import pytest

from nakagai.strategies.rules import (
    canonical_spec, describe_spec, spec_hash, validate_spec,
)

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


def test_version_required_and_v1_rejected():
    assert any("version" in e for e in validate_spec({**ORB, "version": 1}))
    spec = dict(ORB); spec.pop("version")
    assert any("version" in e for e in validate_spec(spec))


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"src": "closs"}, "op": ">", "rhs": 1}), "closs"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"ind": "nope"}, "op": ">", "rhs": 1}), "nope"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"op": "%", "args": [1, 2]}, "op": ">", "rhs": 1}), "%"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"prim": "wat"}, "op": ">", "rhs": 1}), "wat"),
    (lambda s: s["long"]["all"].__setitem__(0, {"lhs": {"src": "close", "tf": "4h"}, "op": ">", "rhs": 1}), "4h"),
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
