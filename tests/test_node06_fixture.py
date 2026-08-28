import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest

from tests.node06_current_capture import capture_current


FIXTURES = Path(__file__).parent / "fixtures"


def test_node06_baseline_fixture_matches_its_transferred_digest():
    payload = (FIXTURES / "node06-before.json").read_bytes()
    recorded = (FIXTURES / "node06-before.sha256").read_text().strip()
    assert recorded == (
        f"{hashlib.sha256(payload).hexdigest()}  node06-before.json"
    )


def test_node06_current_corpus_matches_its_provenance_digest():
    path = FIXTURES / "node06-current-corpus.json.gz"
    recorded = (FIXTURES / "node06-current-corpus.json.sha256").read_text().strip()
    assert recorded == f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        corpus = json.load(handle)
    assert corpus["provenance"] == {
        "baseline_core_commit": "c9e0c3a65cd31738a4f2908b42771f8d5a837973",
        "captured_on": "2026-08-28",
        "platform_commit": "9a2a8459defdbb892a01fe54a545aea42dd09237",
    }
    assert len(corpus["specs"]) == 57
    assert len(corpus["schedules"]) == 7


def _assert_trade_behavior_equal(before: dict, after: dict) -> None:
    assert after["behavior"] == before["behavior"]
    assert after["floats"].keys() == before["floats"].keys()
    for field, expected in before["floats"].items():
        assert math.isclose(
            after["floats"][field], expected,
            rel_tol=1e-11, abs_tol=0.0,
        ), field


def _record_identity(mapping: dict, old: str, new: str) -> None:
    if old in mapping:
        assert mapping[old] == new
    else:
        mapping[old] = new


def test_node06_current_output_preserves_behavior_and_moves_identity_one_to_one():
    """Replay all 57 transferred plays through current production core."""
    baseline = json.loads((FIXTURES / "node06-before.json").read_text())
    current = capture_current(baseline)
    assert set(current) == set(baseline["plays"])

    definition_map = {}
    play_definition_map = {}
    replay_map = {}
    trade_map = {}

    for name, before in baseline["plays"].items():
        after = current[name]
        assert after["canonical_spec_hash"] == before["canonical_spec_hash"]
        assert len(after["trades"]) == len(before["trades"])
        for old_trade, new_trade in zip(before["trades"], after["trades"]):
            _assert_trade_behavior_equal(old_trade, new_trade)
            assert (new_trade["identity"]["replay_id"]
                    == after["request"]["replay_id"])
            assert (new_trade["identity"]["play_id"]
                    == after["request"]["plays"][0]["play_id"])
            _record_identity(
                trade_map,
                old_trade["identity"]["trade_id"],
                new_trade["identity"]["trade_id"],
            )

        _record_identity(
            definition_map,
            before["definition_digest"], after["definition_digest"])
        _record_identity(
            play_definition_map,
            before["request"]["plays"][0]["definition_digest"],
            after["request"]["plays"][0]["definition_digest"],
        )
        _record_identity(
            replay_map,
            before["request"]["replay_id"], after["request"]["replay_id"])

    expected_trades = sum(
        len(row["trades"]) for row in baseline["plays"].values())
    assert [len(mapping) for mapping in (
        definition_map, play_definition_map, replay_map, trade_map,
    )] == [57, 57, 57, expected_trades]
    for identity_map in (
            definition_map, play_definition_map, replay_map, trade_map):
        assert all(old != new for old, new in identity_map.items())
        assert len(set(identity_map.values())) == len(identity_map)


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
