"""Causality and isolation: what one runtime may see, and what is only its own.

Two questions, and they are separate. VISIBILITY asks which rows a context
carries, and `tests/test_portfolio_bars.py` already pins the cut itself. What
lands here is the rest of the causal contract:

- the availability cut walked across every close of a replay rather than
  spot-checked, so an off-by-one at one boundary cannot hide behind a passing
  sample;
- the EMISSION GATE, which is a different question from visibility. A bar being
  readable does not entitle a higher-timeframe play to decide on it; the
  schedule's `fresh_context_at` names the one base close where it may, and this
  file is where core stops reconstructing that instant from a label and a
  `TimeframeSet` delta;
- training as immutable history, with no mutable runtime alive to observe it;
- isolation, which is what makes two play symbols two replays rather than one.

The daylight-saving pair is the sharpest thing here and it needs its reason
stated. `spring_schedule` and `fall_schedule` straddle a transition across a
weekend, the way XNYS does, so on those every bucket still satisfies
`label + 4h == period_end` and the wrong rule and the right one agree at every
close. The bucket schedules do not: they put the change inside a bucket, which
is the only shape that can tell the two apart, and they tell them apart in
opposite directions.

One hole is knowingly left open and is not this file's to close. A gate says
WHEN a play may decide; `FrameEval._positions` says WHICH higher-timeframe bar
the deciding row reads, and it still answers from `label + delta`. It cannot
show a bar the cut withheld, because it is built over the already-cut frames,
so this is not a leak. It is the other direction: on a bucket that spans a
transition forward, the bucket is released and fresh an hour before that
arithmetic counts it, so the play is entitled to decide and can read nothing.
Fixing it means changing what a `FrameEval` is constructed from, which is a
Phase 1 contract this task does not own. The gate is fixed here; the alignment
is measured and reported.
"""

import dataclasses

import pandas as pd
import pytest

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.bars import ReplayDependencies
from nakagai.engine.context import build_scheduled_context
from nakagai.engine.portfolio_types import StrategyOutputError
from nakagai.strategies.base import MarketContext, Strategy
from nakagai.strategies.composite.strategy import CompositeStrategy
from nakagai.strategies.rules import RuleStrategy
from tests.portfolio_fixtures import (
    FactoryCalls,
    ScriptedPlay,
    base_dependencies,
    base_request,
    base_schedule,
    bucket_dependencies,
    counting_definitions,
    fall_bucket_request,
    fall_bucket_schedule,
    prepared_for,
    replay_fixture,
    scripted_name,
    spring_bucket_request,
    spring_bucket_schedule,
    ts,
)

# A play that decides on the four-hour bucket, so its emission gate is the
# bucket's freshness and nothing else. The condition is built from sources
# alone and the geometry is a percent, so no indicator warmup and no ATR can
# make a bar go quiet for a reason other than the gate.
BUCKET_SPEC = {
    "version": 2, "name": "bucket", "timeframe": "4h",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">", "rhs": {"src": "open"}}]},
    "risk": {"stop": {"kind": "percent", "pct": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

FIRST_TEST_CLOSE = ts("2026-11-27T14:45:00Z")
# The default window warms up on the whole 2026-11-25 session, so the first
# test context carries 26 training bars behind its one test bar.
TRAIN_INTERVALS = 26


def _label_gate(ctx: MarketContext, timeframe: str) -> bool:
    """What `label + delta` says, which is the rule the schedule replaces.

    Spelled out here rather than imported, so these tests state BOTH answers
    and a fixture where the two agree cannot pass for a fixture that proves
    anything. This is `strategies/util.fresh_bar` before the schedule existed:
    the last visible bar's label plus a fixed absolute delta, called a close.
    """
    bars = ctx.bars[timeframe]
    if bars.empty:
        return False
    reconstructed = bars.index[-1] + DEFAULT_TIMEFRAMES.deltas[timeframe]
    return (ctx.now - reconstructed) < DEFAULT_TIMEFRAMES.step


def _gates(request, schedule, dependencies, spec):
    """Every close a play decided on `spec`'s frame may emit at, two ways.

    The real gate is taken from the real strategy rather than read off the
    context, so these cover the whole chain: the schedule answers, the context
    carries the answer, and the play obeys it.
    """
    timeframe = spec["timeframe"]
    strategy = RuleStrategy({"spec": spec})
    validated, prepared = prepared_for(request, schedule, dependencies)
    scheduled, reconstructed = [], []
    for interval in validated.test_intervals:
        context = build_scheduled_context(
            prepared, "SPY", interval.close_ts, validated, dependencies)
        if strategy._fresh(context):
            scheduled.append(interval.close_ts)
        if _label_gate(context, timeframe):
            reconstructed.append(interval.close_ts)
    return (scheduled, reconstructed)


# ------------------------------------------------------- the availability cut


def test_every_context_of_a_replay_shows_exactly_the_released_rows():
    """The cut, at every close of the window rather than at a chosen one.

    A prefix rule is exactly the kind that is right in the middle of a range
    and wrong at one end, so the whole range is walked and each timeframe's
    labels are compared against the schedule's own `available_at`, in both
    directions: every released bar present, and no bar whose release is still
    ahead.
    """
    dependencies = base_dependencies()
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    for interval in validated.test_intervals:
        now = interval.close_ts
        context = build_scheduled_context(
            prepared, "SPY", now, validated, dependencies)
        assert list(context.bars["15m"].index) == [
            row.open_ts for row in validated.base_intervals if row.close_ts <= now]
        for timeframe in ("1h", "4h", "1d"):
            rows = validated.context_bars(timeframe)
            assert list(context.bars[timeframe].index) == [
                row.label_ts for row in rows if row.available_at <= now]
            assert not {row.label_ts for row in rows if row.available_at > now} & set(
                context.bars[timeframe].index)


def test_an_external_dependency_joins_only_after_its_own_availability():
    """A symbol a play reads without trading is released the same way.

    Availability is a property of the bar and the schedule, never of who is
    reading it, so the external symbol's hourly bar appears at its own
    `available_at` and not at the traded symbols'.
    """
    dependencies = ReplayDependencies(
        timeframes=("15m", "1h"), external_symbols=("IWM",))
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    before = build_scheduled_context(
        prepared, "IWM", ts("2026-11-27T14:45:00Z"), validated, dependencies)
    after = build_scheduled_context(
        prepared, "IWM", ts("2026-11-27T15:00:00Z"), validated, dependencies)
    assert list(before.bars["1h"].index) == [ts("2026-11-25T14:00:00Z")]
    assert list(after.bars["1h"].index) == [ts("2026-11-25T14:00:00Z"),
                                            ts("2026-11-27T14:00:00Z")]


def test_a_replay_builds_a_context_only_for_the_symbols_it_trades():
    """An external dependency is hydrated, and no runtime is handed one.

    Phase 1 gives a runtime its own symbol's context and nothing else, so a
    declared external symbol reaches the bar preflight and stops there.
    """
    calls = []
    replay_fixture(
        calls=calls,
        dependencies=ReplayDependencies(
            timeframes=("15m",), external_symbols=("IWM",)),
    )
    assert {call.symbol for call in calls} == {"QQQ", "SPY"}


def test_the_mappings_a_runtime_receives_cannot_be_rebound():
    """A strategy may not replace an answer the door owns.

    A narrower claim than isolation, which is one context per runtime and is
    pinned below. The one that matters here is `fresh`: a play that could
    rebind its own gate could decide at a close the schedule never released it
    for, which is the rule this door exists to enforce.
    """
    validated, prepared = prepared_for(
        base_request(), base_schedule(), base_dependencies())
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), validated, base_dependencies())
    for mapping in (context.bars, context.cursor, context.fresh):
        with pytest.raises(TypeError):
            mapping["15m"] = None


def test_a_context_carries_only_the_frames_its_replay_declared():
    """No undeclared key, in any of the three mappings.

    A frame nobody declared is a frame nobody hydrated, so reading one is a
    spec asking for data this replay never had. It raises rather than reading
    empty, which would look to a play like a market with no history.
    """
    dependencies = ReplayDependencies(timeframes=("15m", "1h"), external_symbols=())
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), validated, dependencies)
    assert set(context.bars) == {"15m", "1h"}
    assert set(context.cursor) == {"15m", "1h"}
    # The driving frame has no gate, so it is absent from `fresh` on purpose.
    assert set(context.fresh) == {"1h"}
    assert context.tfs.all == ("15m", "1h")
    for undeclared in ("4h", "1d"):
        with pytest.raises(KeyError):
            context.bars[undeclared]
        with pytest.raises(KeyError):
            context.fresh[undeclared]


def test_the_cursor_indexes_the_newest_row_each_frame_has_released():
    """`ctx.cursor[tf]` is the position of the last visible row, per frame.

    The rules grammar reads its series AT the cursor, so one off by a row would
    evaluate a condition on a bar the schedule had not released, or on the one
    before the deciding close. Per frame rather than shared, because the two
    move at different rates: the base frame gains a row at every close and an
    hourly frame only when its own bar becomes available.

    The numbers are the schedule's own. By 15:00 on the half day, twenty-six
    intervals of 2026-11-25 and two of 2026-11-27 have closed, so the base
    cursor is 27; both hourly bars have been released, so the hourly cursor
    is 1.
    """
    dependencies = ReplayDependencies(timeframes=("15m", "1h"), external_symbols=())
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)

    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), validated, dependencies)

    assert dict(context.cursor) == {"15m": 27, "1h": 1}
    for timeframe, position in context.cursor.items():
        assert position == len(context.bars[timeframe]) - 1


def test_one_plays_write_cannot_reach_another_plays_prices():
    """Two plays at one close, and the first one writes into its bars.

    Copy-on-write is what makes a zero-copy prefix safe to hand out, and it
    protects the ENGINE only: the write copies away from the engine's frame and
    then mutates the object it was made on. So a shared context object is a
    channel between plays even when nothing can be rebound, and the fix is one
    context per runtime rather than one per symbol.

    Both halves are asserted. Play B reads the true open at every close it was
    evaluated on, and play A reads it too, which is the engine's own frame
    surviving into the next interval.
    """
    calls = []
    replay_fixture(
        plays=(ScriptedPlay(play_id="play-a", writes_first_open=-1.0),
               ScriptedPlay(play_id="play-b")),
        calls=calls,
    )
    reads = {play: {call.first_base_open for call in calls if call.play_id == play}
             for play in ("play-a", "play-b")}
    assert reads == {"play-a": {100.0}, "play-b": {100.0}}


# -------------------------------------------------------- the emission gate


def test_the_gate_is_the_schedules_freshness_and_not_the_bars_availability():
    """Readable and decidable are two different questions.

    The prior session's daily bar is readable from the first base close of the
    next session onward, and it entitles a daily play to decide on exactly one
    of those closes. A gate that read visibility instead would fire on all
    fourteen.
    """
    dependencies = base_dependencies()
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    readable, fresh = [], []
    for interval in validated.test_intervals:
        context = build_scheduled_context(
            prepared, "SPY", interval.close_ts, validated, dependencies)
        if len(context.bars["1d"]):
            readable.append(interval.close_ts)
        if context.fresh["1d"]:
            fresh.append(interval.close_ts)
    assert len(readable) == 14
    assert fresh == [ts("2026-11-27T14:45:00Z")]


def test_a_context_bar_the_schedule_never_calls_fresh_gates_nothing():
    """The four-hour bucket of an early close has no freshness at all.

    Its period ends three hours after the half day does, so no scheduled base
    close falls inside its freshness window and the schedule leaves
    `fresh_context_at` null. Null is not a timestamp that never matches by
    accident; it is the schedule saying this bar entitles no decision.
    """
    dependencies = base_dependencies()
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    assert [row.fresh_context_at for row in validated.context_bars("4h")] == [None]
    for interval in validated.test_intervals:
        context = build_scheduled_context(
            prepared, "SPY", interval.close_ts, validated, dependencies)
        assert context.fresh["4h"] is False


def test_a_rule_play_gates_on_the_context_rather_than_on_a_reconstruction():
    """The consumer side, isolated from any schedule.

    Two contexts identical in every other respect, differing only in what they
    declare fresh, and the gate follows the declaration. A strategy that still
    reconstructed the boundary would answer the same way twice.
    """
    tfs = TimeframeSet(driving="15m", higher=("4h",),
                       deltas=DEFAULT_TIMEFRAMES.deltas,
                       session_aligned=DEFAULT_TIMEFRAMES.session_aligned)
    index = pd.DatetimeIndex([ts("2026-11-27T14:30:00Z")], name="ts")
    frame = pd.DataFrame({"open": [100.0], "high": [100.2], "low": [99.8],
                          "close": [100.1], "volume": [1_000.0]}, index=index)
    strategy = RuleStrategy({"spec": BUCKET_SPEC})
    for declared in (True, False):
        context = MarketContext(
            symbol="SPY", now=ts("2026-11-27T14:45:00Z"), tfs=tfs,
            bars={"15m": frame, "4h": frame}, fresh={"4h": declared})
        assert strategy._fresh(context) is declared


def test_the_spring_transition_gate_fires_once_where_the_label_rule_fires_five_times():
    """00:00 Eastern to 04:00 Eastern on 2026-03-08 is three absolute hours.

    So `label + 4h` lands an hour PAST the bucket's own end, and the reading
    "within one base bar after that instant" is satisfied by every close from
    the bucket's real end up to it. The play emits on five closes where the
    schedule entitles it to one, and it emits them before the bucket it is
    deciding on has been superseded.
    """
    scheduled, reconstructed = _gates(
        spring_bucket_request(), spring_bucket_schedule(), bucket_dependencies(),
        BUCKET_SPEC)
    assert scheduled == [ts("2026-03-08T08:00:00Z"), ts("2026-03-08T12:00:00Z")]
    assert reconstructed == [
        ts("2026-03-08T08:00:00Z"), ts("2026-03-08T08:15:00Z"),
        ts("2026-03-08T08:30:00Z"), ts("2026-03-08T08:45:00Z"),
        ts("2026-03-08T09:00:00Z"), ts("2026-03-08T12:00:00Z"),
    ]


def test_the_autumn_transition_gate_fires_where_the_label_rule_never_fires():
    """00:00 Eastern to 04:00 Eastern on 2026-11-01 is five absolute hours.

    So `label + 4h` lands an hour SHORT of the bucket's end, at an instant when
    the bucket is not available yet and the frame is still empty. By the time
    it is readable the reconstructed close is an hour in the past, so the gate
    never fires at all and the play goes silent for the whole bucket.
    """
    scheduled, reconstructed = _gates(
        fall_bucket_request(), fall_bucket_schedule(), bucket_dependencies(),
        BUCKET_SPEC)
    assert scheduled == [ts("2026-11-01T09:00:00Z"), ts("2026-11-01T13:00:00Z")]
    assert reconstructed == [ts("2026-11-01T13:00:00Z")]


# ------------------------------------------------- training as causal history


def test_no_runtime_observes_a_training_interval():
    """Training is history a context carries, never a bar a runtime is shown.

    The first callback of the replay lands on the first test close, and the
    context it lands with already holds the whole training session. So the
    warmup is causal and the runtime that reads it is younger than the bars it
    is reading.
    """
    calls = []
    replay_fixture(calls=calls)
    assert min(call.now for call in calls) == FIRST_TEST_CLOSE
    first = calls[0]
    assert first.operation == "on_bar"
    assert first.now == FIRST_TEST_CLOSE
    assert first.visible == (("15m", TRAIN_INTERVALS + 1),)
    assert first.last_base_label == ts("2026-11-27T14:30:00Z")


def test_the_first_test_context_is_warm_from_the_training_range():
    """The indicator a strategy reads at `test_start` was computed on training.

    Only one test bar has closed, so a mean over 27 bars is 26 parts training
    history. The frames rise by a tenth per bar from 100.05, which makes the
    expected value a literal rather than a restatement of the frame.
    """
    dependencies = base_dependencies()
    validated, prepared = prepared_for(base_request(), base_schedule(), dependencies)
    context = build_scheduled_context(
        prepared, "SPY", FIRST_TEST_CLOSE, validated, dependencies)
    closes = context.bars["15m"]["close"]
    assert len(closes) == TRAIN_INTERVALS + 1
    assert context.bars["15m"].index[0] == base_request().window.train_start
    assert closes.iloc[0] == pytest.approx(100.05)
    assert closes.iloc[-1] == pytest.approx(102.65)
    assert closes.mean() == pytest.approx(101.35)


# ------------------------------------------------------------ runtime scoping


def test_a_stateful_runtime_is_built_once_per_play_symbol_and_shares_no_state():
    """Four runtimes for two plays over two symbols, and none of them shared.

    The per-instance sequence is what proves it. A runtime shared between two
    symbols would count straight through, so its second symbol's first call
    would be numbered two; four isolated runtimes each start at one.
    """
    built, calls = FactoryCalls(), []
    replay_fixture(calls=calls, wrap=counting_definitions(built))
    assert built.factory_count == 4
    assert len({id(runtime) for runtime in built.built}) == 4
    first_seen: dict[tuple[str, str], int] = {}
    for call in calls:
        first_seen.setdefault((call.play_id, call.symbol), call.sequence)
    assert set(first_seen) == {(play, symbol)
                               for play in ("play-a", "play-b")
                               for symbol in ("QQQ", "SPY")}
    assert set(first_seen.values()) == {1}


def _composite_over(definition):
    """A definition whose factory builds a real composite over the original.

    The one shape `scripted_definition` cannot produce on its own, and the one
    spec:853-854 asks to be proven: a runtime with a recursive member tree. The
    composite is the real `CompositeStrategy`, its member factory is the
    original definition's, and both are rebuilt on every call, so the member's
    own per-instance counter reports whether the tree was rebuilt per play
    symbol or shared across them.
    """
    spec = {
        "version": 1, "name": "combo", "window_bars": 1,
        "blocks": {"leg": {"strategy": "member", "params": {}}},
        "long": {"all": ["leg"]},
        "risk": {"stop": {"kind": "percent", "pct": 2.0},
                 "target": {"kind": "rr", "rr": 2.0}},
    }

    def factory(params):
        return CompositeStrategy(
            {"spec": spec}, name=definition.name,
            members={"member": lambda block: definition.factory(params)})

    return dataclasses.replace(definition, factory=factory)


def test_a_composite_rebuilds_its_member_tree_for_every_play_symbol():
    """Member state cannot cross a play symbol, one level down from the runtime.

    A composite's members are where the leak would actually happen: the votes,
    the memo, and the ratchet all live on the member rather than on the tree
    above it. Each of the four members here counts its own calls, and each
    counts from one.
    """
    calls = []
    replay_fixture(
        plays=tuple(ScriptedPlay(play_id=play_id) for play_id in ("play-a", "play-b")),
        calls=calls, wrap=_composite_over,
    )
    first_seen: dict[tuple[str, str], int] = {}
    for call in calls:
        first_seen.setdefault((call.play_id, call.symbol), call.sequence)
    assert len(first_seen) == 4
    assert set(first_seen.values()) == {1}
    # Four members, each asked at all fourteen closes of the default window.
    assert len(calls) == 56


def test_a_factory_returning_something_other_than_a_strategy_is_refused():
    """The type, before the name, because the name check cannot tell them apart.

    A mapping has no `name`, so a name comparison alone would report a mismatch
    against null and send an operator to a definition whose name was never the
    problem.
    """
    def _returns_a_mapping(definition):
        return dataclasses.replace(
            definition, factory=lambda params: {"name": definition.name})

    with pytest.raises(StrategyOutputError) as raised:
        replay_fixture(wrap=_returns_a_mapping)
    assert raised.value.code == "invalid_type"
    assert raised.value.details["seen"] == "dict"
    assert raised.value.details["strategy"] == scripted_name("play-a")


def test_a_runtime_whose_declared_name_disagrees_with_its_definition_is_refused():
    """A factory may not hand back a strategy calling itself something else.

    Nothing downstream would notice: the name only ever surfaces in an error's
    details, so a mislabeled runtime would replay silently and blame the wrong
    definition the one time something went wrong.
    """
    with pytest.raises(StrategyOutputError) as raised:
        replay_fixture(plays=(ScriptedPlay(play_id="play-a", runtime_name="impostor"),))
    assert raised.value.code == "strategy_name_mismatch"
    assert raised.value.details["strategy"] == scripted_name("play-a")
    assert raised.value.details["declared"] == "impostor"
    assert raised.value.details["play_id"] == "play-a"


def test_a_class_default_parameter_is_never_shared_between_runtimes():
    """`DEFAULT_PARAMS` is class state, and two runtimes may not hold one node.

    A shallow merge would hand every instance of a strategy the SAME nested
    object, so one play symbol appending to it would be read by every other
    one, in every other replay in the process.
    """

    class _Nested(Strategy):
        name = "nested"
        DEFAULT_PARAMS = {"levels": [1.0, 2.0]}

        def on_bar(self, ctx):
            return ()

    first, second = _Nested(), _Nested()
    first.params["levels"].append(3.0)
    assert second.params["levels"] == [1.0, 2.0]
    assert _Nested.DEFAULT_PARAMS["levels"] == [1.0, 2.0]
