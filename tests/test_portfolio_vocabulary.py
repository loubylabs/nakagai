"""One grammar per play, from the definition to the trade and to the estimate.

A definition names the grammar its spec is read under. Three things then have
to read that same grammar or the replay reports a number nobody asked for:

- the FACTORY, which builds the runtime and validates its spec;
- the CONTEXT, whose `ctx.fe` is what `RuleStrategy` evaluates its conditions
  through, so it decides the entries;
- the IC LENS, which grades the same spec's margin.

The two failures below are the reachable ones, and they fail differently. An
ADDED term is loud: the evaluator has no such name and the replay aborts. A
REDEFINED term is silent, because `vocabulary_digest` covers what a term
DECLARES rather than what it computes, so two grammars that differ only in an
implementation carry one digest, one base digest, one play digest, one replay
identity, and one registry digest. Nothing in the contract moves. The only
observable is the trades themselves, which is why they are what is asserted.

Every grammar here is built by a module-level function rather than a lambda:
`vocabulary_digest` is cached on the factory object, so a fresh lambda per call
would be a fresh cache entry per call and the digest claims below would be
measuring the cache instead of the declarations.
"""

import math

import pandas as pd
import pytest

from nakagai.engine import PortfolioBars, run_portfolio
from nakagai.engine.canonical import definition_digest
from nakagai.engine.portfolio_types import (
    PlayRequest,
    ReplayWindow,
    StrategyRuntimeError,
)
from nakagai.engine.registry import (
    FrozenStrategyRegistry,
    composite_definition,
    rules_definition,
    spec_base_digest,
    vocabulary_digest,
)
from nakagai.strategies.rules.vocabulary import Term, Vocabulary, core_vocabulary
from tests.portfolio_fixtures import (
    base_request,
    frictionless_execution,
    replay_account,
    rules_frames,
    rules_replay,
    rules_schedule,
)

# A spec whose only condition names a term core does not have. Under the core
# grammar the evaluator cannot resolve `house_floor` at all.
ADDED_TERM_SPEC = {
    "version": 2, "name": "added_term", "timeframe": "15m",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"ind": "house_floor"}}]},
    "risk": {"stop": {"kind": "percent", "pct": 1.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

# A spec whose only condition names a term core DOES have, so it resolves under
# either grammar and the difference is the number that comes back.
REDEFINED_TERM_SPEC = {
    "version": 2, "name": "redefined_term", "timeframe": "15m",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"ind": "sma", "n": 20}}]},
    "risk": {"stop": {"kind": "percent", "pct": 1.0},
             "target": {"kind": "rr", "rr": 2.0}},
}


def wave(count: int) -> list[float]:
    """A close series that rises and falls, so a real spec can cross a level."""
    return [100.0 + 4.0 * math.sin(step / 3.0) + 0.05 * step
            for step in range(count)]


def added_term_vocabulary() -> Vocabulary:
    """Core, plus one term core has no name for."""
    return core_vocabulary().with_terms(
        Term("house_floor", "series", {}, {},
             lambda series, _args: pd.Series(0.0, index=series.index),
             doc="a floor every positive close clears"))


def _redefined(fn) -> Vocabulary:
    """Core with `sma` REPLACED, declaration for declaration.

    `with_terms` refuses a name the vocabulary already holds, which is correct
    for adding and is not what this does: the term keeps `sma`'s name, kind,
    argument schema, defaults, and causal flags, and changes only the function
    behind them. That is precisely the change no digest can see.
    """
    core = core_vocabulary()
    existing = core.indicators["sma"]
    replacement = Term(
        existing.name, existing.kind, existing.args, existing.defaults, fn,
        doc=existing.doc, end_anchored=existing.end_anchored,
        session_scoped=existing.session_scoped,
        driving_frame_intraday=existing.driving_frame_intraday,
        pine=existing.pine)
    return Vocabulary({**core.indicators, "sma": replacement}, core.primitives)


def sma_reads_zero() -> Vocabulary:
    """`sma` answers zero, so a positive close is always above it."""
    return _redefined(lambda series, _args: pd.Series(0.0, index=series.index))


def sma_reads_double() -> Vocabulary:
    """`sma` answers twice the close, so a positive close is never above it."""
    return _redefined(lambda series, _args: series * 2.0)


# ------------------------------------------------------------- an added term


def test_a_spec_using_an_added_term_replays_under_its_own_grammar():
    """The loud half: the play runs rather than aborting at the first bar.

    Under the core grammar `ctx.fe` has no `house_floor`, and `on_bar` dies on
    the lookup at the first close the play is asked at, which the boundary
    reports as `strategy_raised`. The definition declares the grammar that has
    the term, so the only way this passes is if that grammar reached the
    context the runtime decides through.
    """
    result = rules_replay(ADDED_TERM_SPEC, wave(60),
                          vocabulary_factory=added_term_vocabulary)

    assert result.trades
    assert result.slices[0].signals > 0


def test_the_core_grammar_cannot_read_an_added_term():
    """The control: `house_floor` is genuinely not a name core has.

    Without this the test above could pass for the wrong reason, if the term it
    adds had happened to be one core already carried. The refusal lands at
    CONSTRUCTION here rather than at a bar, because a definition built under
    the core grammar hands that grammar to `RuleStrategy`, which validates its
    own spec against it. The pre-fix failure was the later one: the factory
    accepted the spec under the definition's grammar and then the context
    evaluated it under core's, so the lookup died at the first close.
    """
    with pytest.raises(StrategyRuntimeError) as raised:
        rules_replay(ADDED_TERM_SPEC, wave(60))

    assert raised.value.code == "strategy_raised"
    assert raised.value.details["operation"] == "construct"


# ---------------------------------------------------------- a redefined term


def test_two_implementations_of_one_term_produce_two_different_replays():
    """The silent half: only the trades can tell these two apart.

    `sma` reading zero puts every positive close above it, so the play signals
    and trades. `sma` reading twice the close puts every close below it, so the
    play never signals. A replay that evaluated under the core grammar would
    run core's real moving average both times and return one identical answer.
    """
    trading = rules_replay(REDEFINED_TERM_SPEC, wave(60),
                           vocabulary_factory=sma_reads_zero)
    silent = rules_replay(REDEFINED_TERM_SPEC, wave(60),
                          vocabulary_factory=sma_reads_double)

    assert trading.trades
    assert silent.trades == ()
    assert trading.result_digest != silent.result_digest


def test_no_digest_moves_between_two_implementations_of_one_term():
    """Why the test above has to assert on the trades.

    Everything a caller could match on is identical across the pair: the
    vocabulary digest, the definition's base digest, the play's own digest
    binding that base to its params, and the replay identity derived from it.
    A replay under the wrong grammar therefore announces nothing at all.
    """
    left, right = sma_reads_zero, sma_reads_double

    assert vocabulary_digest(left) == vocabulary_digest(right)
    assert (spec_base_digest(REDEFINED_TERM_SPEC, left)
            == spec_base_digest(REDEFINED_TERM_SPEC, right))
    trading = rules_replay(REDEFINED_TERM_SPEC, wave(60), vocabulary_factory=left)
    silent = rules_replay(REDEFINED_TERM_SPEC, wave(60), vocabulary_factory=right)
    assert trading.request == silent.request


def two_grammar_inputs():
    """One request over two plays whose definitions read two grammars.

    The two specs are identical except for their names, which is deliberate on
    both counts: the same condition means the bodies differ only in the grammar
    behind `sma`, and the different names give them different base digests, so
    one bundle can hold both. Two definitions sharing a digest is a refusal, and
    a grammar that redefines rather than adds moves no digest at all.
    """
    specs = {
        "reads_zero": sma_reads_zero,
        "reads_double": sma_reads_double,
    }
    definitions, plays = [], []
    for name, factory in specs.items():
        spec = {**REDEFINED_TERM_SPEC, "name": name}
        base = spec_base_digest(spec, factory)
        definitions.append(rules_definition(name, base, spec=spec,
                                            vocabulary_factory=factory))
        plays.append(PlayRequest(
            play_id=f"play-{name}", strategy=name,
            definition_digest=definition_digest(base, {}), params={},
            priority=100))
    closes = wave(60)
    schedule = rules_schedule(len(closes))
    intervals = schedule.base_intervals
    request = base_request(
        plays=tuple(plays), symbols=("SPY",),
        window=ReplayWindow(
            train_start=intervals[0].open_ts,
            train_end=intervals[20].open_ts,
            test_start=intervals[20].open_ts,
            test_end=intervals[-1].close_ts,
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=intervals[-1].close_ts,
        account=replay_account(),
        execution=frictionless_execution(),
    )
    return (request, PortfolioBars(rules_frames(closes, schedule)),
            FrozenStrategyRegistry.from_definitions(tuple(definitions)), schedule)


def test_two_plays_in_one_replay_each_decide_under_their_own_grammar():
    """Per RUNTIME, not per replay. One context each, one grammar each.

    Both plays carry the same condition over the same bars in the same replay,
    and only the implementation of `sma` separates them. A replay that resolved
    one grammar for the whole run, or keyed it on the symbol rather than the
    play, would hand both runtimes whichever it picked and the two would either
    both trade or both stay silent.
    """
    result = run_portfolio(*two_grammar_inputs())

    assert {row.play_id for row in result.trades} == {"play-reads_zero"}
    signals = {row.play_id: row.signals for row in result.slices}
    assert signals["play-reads_double"] == 0
    assert signals["play-reads_zero"] > 0


# --------------------------------------------------------------- composites


def composite_inputs():
    """One play over a composite whose single member reads an added term.

    A composite has no graded factor in Phase 1, so nothing but the trades can
    show which grammar it was read under. Its members evaluate through the
    composite's OWN context, so the grammar the composite definition carries is
    the one every member in its tree gets.
    """
    member = rules_definition(
        "added_term", spec_base_digest(ADDED_TERM_SPEC, added_term_vocabulary),
        spec=ADDED_TERM_SPEC, vocabulary_factory=added_term_vocabulary)
    combo_spec = {
        "version": 1, "name": "combo",
        "blocks": {"a": {"strategy": "added_term", "params": {}}},
        "long": {"all": ["a"]},
        "risk": {"stop": {"kind": "percent", "pct": 1.0},
                 "target": {"kind": "rr", "rr": 2.0}},
    }
    base = "6e" * 32
    combo = composite_definition(
        "combo", base, members={"added_term": member},
        vocabulary_factory=added_term_vocabulary)
    params = {"spec": combo_spec}
    closes = wave(60)
    schedule = rules_schedule(len(closes))
    intervals = schedule.base_intervals
    request = base_request(
        plays=(PlayRequest(play_id="play-combo", strategy="combo",
                           definition_digest=definition_digest(base, params),
                           params=params, priority=100),),
        symbols=("SPY",),
        window=ReplayWindow(
            train_start=intervals[0].open_ts,
            train_end=intervals[20].open_ts,
            test_start=intervals[20].open_ts,
            test_end=intervals[-1].close_ts,
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=intervals[-1].close_ts,
        account=replay_account(),
        execution=frictionless_execution(),
    )
    return (request, PortfolioBars(rules_frames(closes, schedule)),
            FrozenStrategyRegistry.from_definitions((combo, member)), schedule)


def test_a_composite_reads_its_members_under_the_grammar_it_declares():
    """The composite half of the same wiring.

    `composite_definition` takes a vocabulary factory and refuses a member read
    under another, so the tree already agrees about its grammar. What this pins
    is that the agreed grammar reaches the context: the members evaluate their
    conditions through `ctx.fe`, so a composite handed the core grammar aborts
    on the first bar however its own definition was built.
    """
    result = run_portfolio(*composite_inputs())

    assert result.trades
    assert result.slices[0].signals > 0


def test_a_definition_carries_the_grammar_it_was_built_under():
    """The seam itself: the field the replay reads is the one a builder set."""
    definition = rules_definition(
        "added_term", spec_base_digest(ADDED_TERM_SPEC, added_term_vocabulary),
        spec=ADDED_TERM_SPEC, vocabulary_factory=added_term_vocabulary)

    assert definition.vocabulary_factory is added_term_vocabulary
    assert rules_definition("plain", "0" * 64).vocabulary_factory is core_vocabulary


# --------------------------------------------- one grammar, decided and graded


def test_the_graded_factor_reads_the_same_grammar_the_entries_did():
    """Decided and graded under one grammar, in one result.

    The IC lens builds its evaluator from the definition's own factory, so it
    could always read an added term. What this asserts is the pair: the same
    replay that traded on `house_floor` also reports a real measurement over
    it, rather than a graded factor that ran beside entries taken under
    another grammar.

    The counts are hand-derived from the fixture. Sixty base intervals are
    ordinals 0 to 59, the first twenty are warmup, so the observations are
    ordinals 20 to 59. A horizon of `k` bars needs a close at `ordinal + k`, so
    it keeps the observations up to ordinal `59 - k`: thirty-nine at one bar,
    thirty-five at five, and twenty at twenty.
    """
    result = rules_replay(ADDED_TERM_SPEC, wave(60),
                          vocabulary_factory=added_term_vocabulary)

    (row,) = result.slices
    assert row.trades > 0
    assert [item.observations for item in row.ic] == [39, 35, 20]
