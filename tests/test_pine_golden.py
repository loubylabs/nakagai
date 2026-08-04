"""The exact source a TradingView user pastes, frozen byte for byte.

Six plays, chosen so that between them they reach every shape of the language
that changes what the renderer writes:

    sma_cross           ordinary indicators, crosses, and a chart that is not
                        15m, so the runtime guard has to name its own
    orb                 session primitives, a conditional exit, and a time stop
    ifvg_reversal       a foreign timeframe, bars_since, swings, and fair value
                        gap state
    ob_bounce           order block state and the deepest history buffer
    bollinger_breakout  a multi-field indicator, a daily play, and a trailing
                        exit
    discount_pullback   a play whose own timeframe is not the chart's AND that
                        reads a second one, so the tree is split between a
                        native subtree and a chart-level composition

A golden diff is not a failure by itself; it is the renderer saying the text a
user pastes changed. Read the diff, decide whether the new text is right, and
only then regenerate.
"""

import json
from pathlib import Path

import pytest

from nakagai.strategies.rules import compile_pine

GOLDEN = Path(__file__).resolve().parent / "golden" / "pine"
PLAYS = ("sma_cross", "orb", "ifvg_reversal", "ob_bounce",
         "bollinger_breakout", "discount_pullback")


@pytest.mark.parametrize("name", PLAYS)
@pytest.mark.parametrize("artifact", ("indicator", "strategy"))
def test_the_golden_artifact_is_what_the_compiler_writes(name, artifact,
                                                         load_rule_spec):
    bundle = compile_pine(load_rule_spec(name))
    assert getattr(bundle, artifact) == \
        (GOLDEN / f"{name}.{artifact}.pine").read_text()


def test_the_golden_set_is_exactly_the_representative_plays():
    assert {path.name for path in GOLDEN.glob("*.pine")} == {
        f"{name}.{artifact}.pine"
        for name in PLAYS for artifact in ("indicator", "strategy")}


def test_the_sma_cross_fixture_has_not_drifted_from_the_shipped_play(
        load_spec, load_rule_spec):
    # The golden reads its spec from tests/fixtures/rules so that a catalog
    # edit cannot silently rewrite a frozen artifact. sma_cross is also shipped
    # content, and a golden claiming to be the shipped play while quietly
    # differing from it would be worse than either.
    assert load_rule_spec("sma_cross") == load_spec("sma_cross")


@pytest.mark.parametrize("name", PLAYS)
def test_every_golden_input_spec_is_canonical_json(name):
    path = Path(__file__).resolve().parent / "fixtures" / "rules" / f"{name}.json"
    assert path.read_text() == json.dumps(json.loads(path.read_text()),
                                          indent=2) + "\n"
