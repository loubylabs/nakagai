import copy
import hashlib
import json
import math
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


def test_node06_baseline_fixture_matches_its_transferred_digest():
    payload = (FIXTURES / "node06-before.json").read_bytes()
    recorded = (FIXTURES / "node06-before.sha256").read_text().strip()
    assert recorded == (
        f"{hashlib.sha256(payload).hexdigest()}  node06-before.json"
    )


def _assert_trade_behavior_equal(before: dict, after: dict) -> None:
    assert after["behavior"] == before["behavior"]
    assert after["floats"].keys() == before["floats"].keys()
    for field, expected in before["floats"].items():
        assert math.isclose(
            after["floats"][field], expected,
            rel_tol=1e-11, abs_tol=0.0,
        ), field


def test_node06_baseline_separates_exact_behavior_floats_and_identity():
    """The transferred artifact can prove behavioral identity independently
    from the grammar-derived identifiers that Node 06 intentionally moves."""
    baseline = json.loads((FIXTURES / "node06-before.json").read_text())
    replay_ids = set()
    trade_ids = set()

    for row in baseline["plays"].values():
        request = row["request"]
        assert request["replay_id"] not in replay_ids
        replay_ids.add(request["replay_id"])
        for trade in row["trades"]:
            _assert_trade_behavior_equal(trade, copy.deepcopy(trade))
            assert trade["identity"]["replay_id"] == request["replay_id"]
            assert trade["identity"]["play_id"] == request["plays"][0]["play_id"]
            assert trade["identity"]["trade_id"] not in trade_ids
            trade_ids.add(trade["identity"]["trade_id"])

    assert len(replay_ids) == len(baseline["plays"])
    assert len(trade_ids) == sum(
        len(row["trades"]) for row in baseline["plays"].values())


def test_node06_trade_comparison_is_exact_for_nonfloats():
    baseline = json.loads((FIXTURES / "node06-before.json").read_text())
    trade = next(
        trade
        for row in baseline["plays"].values()
        for trade in row["trades"]
    )
    changed = copy.deepcopy(trade)
    changed["behavior"]["qty"] += 1

    with pytest.raises(AssertionError):
        _assert_trade_behavior_equal(trade, changed)


def test_node06_trade_comparison_uses_per_field_float_tolerance():
    baseline = json.loads((FIXTURES / "node06-before.json").read_text())
    trade = next(
        trade
        for row in baseline["plays"].values()
        for trade in row["trades"]
        if trade["floats"]["entry"] != 0.0
    )
    within = copy.deepcopy(trade)
    outside = copy.deepcopy(trade)
    within["floats"]["entry"] *= 1.0 + 0.5e-11
    outside["floats"]["entry"] *= 1.0 + 2.0e-11

    _assert_trade_behavior_equal(trade, within)
    with pytest.raises(AssertionError, match="entry"):
        _assert_trade_behavior_equal(trade, outside)
