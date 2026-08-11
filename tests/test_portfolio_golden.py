"""The one public replay: `run_portfolio`, end to end, on whole results.

Every other portfolio test file points at one component. This one points at the
boundary a caller actually has, and it asks three kinds of question.

- THE SHAPE OF THE DOOR. One entry point, four required arguments, no defaults
  and no keyword controls. A caller cannot reach a second replay, a singleton
  adapter, or a knob that changes arithmetic.
- THE ORDER OF THE PREFLIGHT. Request, schedule, registry definitions,
  dependency closure, complete bar set, and only then a runtime. Every refusal
  below is proven to land with `factory_count == 0`, because "refused before a
  strategy was built" is the contract and a count is the only thing that can
  show it.
- THE WHOLE RESULT. A result is one value, and the assertions here are on the
  value rather than on an intermediate: the trades, the rejections, the equity
  points, the slices, the benchmark, the metrics, and the digest over all of
  them. Determinism is asserted by comparing two runs rather than by pasting a
  hash, so nothing here can pass by having been stamped from its own output.
"""

import dataclasses
import inspect

import pandas as pd
import pytest

from nakagai.engine import (
    ExitReason,
    PortfolioBars,
    PortfolioReplayRequest,
    PortfolioReplayResult,
    RejectionReason,
    ReplayInputError,
    ReplaySchedule,
    StrategyOutputError,
    StrategyRuntimeError,
    decode_replay_result,
    encode_replay_result,
    expected_replay_id,
    result_digest,
    run_portfolio,
)
from nakagai.engine.bars import ReplayDependencies
from nakagai.engine.registry import (
    FrozenStrategyRegistry,
    StrategyRegistry,
    dependencies_for,
)
from tests.portfolio_fixtures import (
    BarPlan,
    ManagePlan,
    ReplayWindow,
    ScriptedPlay,
    SignalPlan,
    base_execution,
    base_request,
    base_schedule,
    counting_registry,
    flat_frame,
    frames_for,
    frictionless_execution,
    ic_plays,
    ic_window,
    ramp_frame,
    replay_account,
    replay_inputs,
    replay_result,
    scripted_definition,
    scripted_params,
    scripted_play_request,
    single_symbol_benchmark,
    strategy_registry,
    ts,
    without_pair,
)

# The default window trades the 2026-11-27 half day: fourteen 15-minute
# intervals from 14:30Z to 18:00Z.
FIRST_OPEN = ts("2026-11-27T14:30:00Z")
TEST_END = ts("2026-11-27T18:00:00Z")
LAST_INTERVAL = 13
POINT_COUNT = 16  # the opening anchor, fourteen closes, and the post-close mark

# What `base_definitions` declares for `base_request`'s two plays, written out
# rather than derived. `sma_cross` is evaluated on the hourly frame and
# `donchian_break` on the daily one, and both read the base frame to decide at,
# so the union is these three and never the fourth supported timeframe. A
# literal is what makes `dependencies_for` checkable: a closure taken from the
# code under test could not disagree with it.
DECLARED = ReplayDependencies(timeframes=("15m", "1h", "1d"), external_symbols=())


def opens(ordinal: int) -> pd.Timestamp:
    return FIRST_OPEN + pd.Timedelta(minutes=15 * ordinal)


def closes(ordinal: int) -> pd.Timestamp:
    return opens(ordinal) + pd.Timedelta(minutes=15)


def catalog_frames() -> dict:
    """One valid frame for every pair `base_request` and its bundle declare."""
    return frames_for(base_request(), base_schedule(), DECLARED)


def one_play(**overrides) -> tuple[ScriptedPlay, ...]:
    """One play trading SPY, signalling once at the first close by default."""
    plan = SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0)
    return (dataclasses.replace(
        ScriptedPlay(play_id="play-a", signals=(plan,)), **overrides),)


def attributed(rows) -> tuple:
    return tuple((row.play_id, row.symbol, row.signal_ordinal) for row in rows)


def slice_of(result: PortfolioReplayResult, play_id: str, symbol: str):
    return next(row for row in result.slices
                if row.play_id == play_id and row.symbol == symbol)


# ------------------------------------------------------------ the public door


def test_run_portfolio_takes_exactly_the_four_contract_arguments():
    """One entry point with four required values and nothing else.

    A default would let a caller replay without naming a schedule or a
    registry; a variadic or a keyword-only control would let one reach a second
    behavior through the same name. The annotations are pinned too, because the
    contract is the four TYPES rather than four positions.
    """
    signature = inspect.signature(run_portfolio)

    assert [(name, parameter.kind, parameter.default, parameter.annotation)
            for name, parameter in signature.parameters.items()] == [
        ("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
         inspect.Parameter.empty, PortfolioReplayRequest),
        ("bars", inspect.Parameter.POSITIONAL_OR_KEYWORD,
         inspect.Parameter.empty, PortfolioBars),
        ("registry", inspect.Parameter.POSITIONAL_OR_KEYWORD,
         inspect.Parameter.empty, StrategyRegistry),
        ("schedule", inspect.Parameter.POSITIONAL_OR_KEYWORD,
         inspect.Parameter.empty, ReplaySchedule),
    ]
    assert signature.return_annotation is PortfolioReplayResult


# ----------------------------------------------------------- preflight order


def test_preflight_finishes_before_strategy_construction():
    """The brief's own case: a missing dependency frame builds nothing.

    `dependencies_for` asks each definition what it reads, which is pure, so the
    dependency count rises while the factory count stays at zero. Asserting both
    is what tells "the preflight ran and refused" from "the registry was never
    consulted at all".
    """
    registry, calls = counting_registry()
    bars = PortfolioBars(without_pair(catalog_frames(), "SPY", "1h"))

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(base_request(), bars, registry, base_schedule())

    assert raised.value.code == "missing_required_bar"
    assert calls.factory_count == 0
    assert calls.dependency_count > 0


def broken_identity() -> PortfolioReplayRequest:
    """A request whose replay identity no longer matches its own formula."""
    request = base_request()
    return dataclasses.replace(request, replay_id=f"replay:{'a' * 64}")


def broken_candidate() -> PortfolioReplayRequest:
    """A wrong candidate whose replay identity is re-derived AROUND it.

    `expected_replay_id` hashes the candidate as an opaque string, so simply
    replacing the candidate breaks the replay identity too and the replay check
    would catch it first. Re-deriving the replay identity from the corrupted
    candidate makes the request self-consistent everywhere except the one place
    this exists to test, so only the candidate check can refuse it.
    """
    request = dataclasses.replace(base_request(),
                                  candidate_id=f"candidate:{'b' * 64}")
    return dataclasses.replace(request, replay_id=expected_replay_id(request))


def edited_play(**fields) -> PortfolioReplayRequest:
    """A request whose first play is edited and whose identities re-derive.

    Re-derived on purpose. Editing a play in place breaks the candidate
    projection, so `validate_request` would refuse first and the registry step
    this exists to reach would never run.
    """
    plays = base_request().plays
    return base_request(
        plays=(dataclasses.replace(plays[0], **fields), plays[1]))


def unknown_strategy() -> PortfolioReplayRequest:
    return edited_play(strategy="never_registered")


def mismatched_definition_digest() -> PortfolioReplayRequest:
    return edited_play(definition_digest="c" * 64)


@pytest.mark.parametrize(
    ("build", "code"),
    [
        (broken_identity, "identity_mismatch"),
        (broken_candidate, "identity_mismatch"),
        (unknown_strategy, "unknown_strategy"),
        (mismatched_definition_digest, "definition_digest_mismatch"),
    ],
    ids=["replay_id", "candidate_id", "unknown_strategy", "definition_digest"],
)
def test_no_strategy_is_constructed_when_a_preflight_step_refuses(build, code):
    """Every refusal ahead of the runtime leaves the factories untouched.

    The request and registry steps are the ones a runtime could most easily
    have been built before, since neither needs a bar. Parametrized rather than
    written once, so a reordering that moved construction earlier fails on the
    first step it overtook rather than only on the last.
    """
    registry, calls = counting_registry()

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(build(), PortfolioBars(catalog_frames()), registry,
                      base_schedule())

    assert raised.value.code == code
    assert calls.factory_count == 0


def test_a_schedule_that_disagrees_with_its_own_digest_builds_no_strategy():
    """The schedule step, which sits between the request and the registry.

    The identity is left alone and one interval is dropped from the body, so
    the request still agrees with the identity and only the recomputed digest
    disagrees. That is the transport failure the digest exists for, and it
    lands before the registry is consulted at all.
    """
    registry, calls = counting_registry()
    schedule = base_schedule()

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(
            base_request(), PortfolioBars(catalog_frames()), registry,
            dataclasses.replace(schedule,
                                base_intervals=schedule.base_intervals[:-1]))

    assert raised.value.code == "schedule_digest_mismatch"
    assert calls.factory_count == 0


def test_a_definition_digest_binds_the_params_the_play_actually_carries():
    """The one check that stops a play naming any params under any definition.

    `PlayRequest.definition_digest` is the SHA-256 of the canonical pair
    `(StrategyDefinition.definition_digest, params)`, so editing either half
    alone breaks it. Here the definition, its name, and the play's own digest
    are untouched and only the PARAMS move, which nothing else in the request
    can see: the registry digest deliberately covers names and base digests,
    and the candidate identity re-derives around whatever the play carries.
    """
    plays = base_request().plays
    swapped = dataclasses.replace(
        plays[0], params={**dict(plays[0].params), "fast_n": 11})

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(base_request(plays=(swapped, plays[1])),
                      PortfolioBars(catalog_frames()), strategy_registry(),
                      base_schedule())

    assert raised.value.code == "definition_digest_mismatch"
    assert raised.value.details["play_id"] == swapped.play_id
    assert raised.value.details["actual"] == plays[0].definition_digest


def test_the_dependency_closure_is_the_union_the_definitions_declare():
    """Only the declared frames are required, and every one of them is.

    Two claims in one, and they fail differently. A frame the closure omits is
    refused as a surplus the replay never declared; a frame it names and the
    caller withholds is refused as absent. The fourth supported timeframe is
    the control: nothing in this bundle reads it, so supplying it is the
    surplus case.
    """
    surplus = frames_for(base_request(), base_schedule(),
                         ReplayDependencies(timeframes=("15m", "1h", "4h", "1d"),
                                            external_symbols=()))

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(base_request(), PortfolioBars(surplus), strategy_registry(),
                      base_schedule())

    assert raised.value.details["field"] == "unexpected_frame"
    assert raised.value.details["timeframe"] == "4h"


def daily_only_inputs():
    """A one-play portfolio whose definition declares the daily frame alone.

    Built here rather than through `replay_inputs`, because that helper takes
    the closure as a `ReplayDependencies`, which refuses a tuple without the
    base timeframe. A DEFINITION is under no such rule: `StrategyDependencies`
    asks only for one supported frame, and declaring `1d` alone is exactly what
    a daily play does.
    """
    play = ScriptedPlay(play_id="play-a", signals=(
        SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),))
    params = scripted_params(False)
    request = base_request(
        plays=(scripted_play_request(play, params),), symbols=("SPY",),
        account=replay_account(), execution=frictionless_execution())
    registry = FrozenStrategyRegistry.from_definitions(
        (scripted_definition(play, timeframes=("1d",)),))
    return (request, registry)


def test_a_daily_only_play_still_requires_the_base_timeframe():
    """The base frame is injected into the closure whatever a play declares.

    The account fills, marks, and settles on the base clock, so a portfolio of
    daily-only plays still reads it. Without the injection the union would be
    a tuple `ReplayDependencies` refuses outright, and a valid request would
    fail as a contract error rather than run.
    """
    request, registry = daily_only_inputs()

    assert dependencies_for(request, registry).timeframes == ("15m", "1d")


def test_a_daily_only_portfolio_replays_through_the_public_door():
    """And the injected frame is genuinely read: the trade fills on it."""
    request, registry = daily_only_inputs()
    frames = frames_for(
        request, base_schedule(),
        ReplayDependencies(timeframes=("15m", "1d"), external_symbols=()),
        build=lambda labels: flat_frame(labels, 100.0))

    result = run_portfolio(request, PortfolioBars(frames), registry,
                           base_schedule())

    assert result.metrics.all_trades.n_trades == 1
    assert result.trades[0].entry_ts == opens(1)


# ------------------------------------------------------- the singleton case


def test_singleton_is_a_one_candidate_portfolio():
    """One play, one symbol, one signal: the smallest valid portfolio.

    The same door and the same result type as a four-runtime replay, which is
    the architecture's claim that a singleton is a degenerate case rather than
    a second path. The reconciliation between the metric cohort and the trade
    list is what proves the result is one value rather than two views assembled
    separately.
    """
    result = replay_result(symbol_order=("SPY",), plays=one_play())

    assert len(result.request.plays) == 1
    assert result.metrics.all_trades.n_trades == len(result.trades)
    assert len(result.slices) == 1
    assert result.slices[0].trades == len(result.trades)


def test_the_singleton_trade_carries_every_hand_derived_value():
    """One whole trade, every field a literal a reader can check.

    A tenth of a percent of 100,000 is 100 of risk; a stop a dollar under a
    flat 100.0 tape is a dollar of protective distance, so the account buys 100
    shares for 10,000. Nothing moves, so the position survives to the forced
    close at the window's end and books exactly nothing.
    """
    result = replay_result(symbol_order=("SPY",), plays=one_play())

    (trade,) = result.trades
    assert (trade.play_id, trade.symbol, trade.direction) == (
        "play-a", "SPY", "long")
    assert (trade.qty, trade.entry, trade.exit) == (100, 100.0, 100.0)
    assert (trade.signal_ts, trade.entry_ts, trade.exit_ts) == (
        closes(0), opens(1), TEST_END)
    assert (trade.gross_pnl, trade.fees, trade.net_pnl) == (0.0, 0.0, 0.0)
    assert trade.exit_reason is ExitReason.END_OF_WINDOW
    assert trade.trade_id.startswith("trade:")
    assert trade.replay_id == result.request.replay_id


def test_the_result_echoes_the_arithmetic_the_engine_executed():
    from nakagai.engine import ARITHMETIC_VERSION

    result = replay_result(symbol_order=("SPY",), plays=one_play())

    assert result.arithmetic_version == ARITHMETIC_VERSION == "2"
    assert result.fill_mode == "pessimistic"
    assert result.schedule_identity == result.request.schedule_identity


# ------------------------------------------------- shared capital contention


def test_two_affordable_entries_contend_for_one_account():
    """Four candidates and one seat: three are refused for capacity.

    The point of the portfolio topology. Each of these entries is affordable in
    isolation, and under the retired one-account-per-row model all four would
    have filled with the full starting equity behind them. Here the account is
    one, so the funding order decides and the losers are recorded rather than
    dropped.
    """
    result = replay_result(account=replay_account(max_open_positions=1))

    assert attributed(result.trades) == (("play-a", "QQQ", 0),)
    assert attributed(result.rejections) == (
        ("play-b", "QQQ", 2), ("play-a", "SPY", 1), ("play-b", "SPY", 3))
    assert {row.reason for row in result.rejections} == {
        RejectionReason.PORTFOLIO_CAPACITY}
    assert result.metrics.n_rejections == 3


def test_capacity_five_accepts_the_first_five_ordered_candidates():
    """A wider account fills every candidate and refuses none."""
    result = replay_result(account=replay_account(max_open_positions=5))

    assert len(result.trades) == 4
    assert result.rejections == ()
    assert result.metrics.all_trades.n_trades == 4


def test_a_lower_capacity_produces_the_exact_corresponding_subset():
    """Two seats take the first two of the same funding order, not any two."""
    result = replay_result(account=replay_account(max_open_positions=2))

    assert attributed(result.trades) == (
        ("play-a", "QQQ", 0), ("play-b", "QQQ", 2))
    assert attributed(result.rejections) == (
        ("play-a", "SPY", 1), ("play-b", "SPY", 3))


def test_an_account_that_cannot_fund_a_second_entry_records_the_cash_reason():
    """Capacity is not the only refusal: cash is its own reason and its own
    pair of numbers.

    One percent of 100,000 is 1,000 of risk, and a dollar of protective
    distance sizes that to 1,000 shares at 100, which is the whole account. The
    first candidate takes every settled dollar and the second is refused with
    both figures on the record. Capacity is five here, so nothing else could
    have produced this refusal.
    """
    both = tuple(SignalPlan(symbol=symbol, at=closes(0), stop=99.0, target=103.0)
                 for symbol in ("QQQ", "SPY"))
    result = replay_result(
        plays=(ScriptedPlay(play_id="play-a", signals=both),),
        account=replay_account(risk_pct=0.01, max_open_positions=5))

    assert attributed(result.trades) == (("play-a", "QQQ", 0),)
    (rejection,) = result.rejections
    assert rejection.reason is RejectionReason.UNSETTLED_CASH
    assert (rejection.required_cash, rejection.available_cash) == (100_000.0, 0.0)
    assert rejection.open_positions == 1


# ------------------------------------------------------------- reordering


def test_reordering_the_request_cannot_change_one_byte_of_the_result():
    """Symbols, plays, param keys, and the bar mapping all reverse.

    The digest is over the whole result, so this covers the trades, the
    rejections, the equity points, the slices, the benchmark, and the metrics
    at once. Comparing two runs rather than one pasted hash means the assertion
    cannot be satisfied by stamping it from an output.
    """
    forward = replay_result(symbol_order=("SPY", "QQQ"), reverse_param_keys=False)
    reversed_order = replay_result(symbol_order=("QQQ", "SPY"),
                                   reverse_param_keys=True, reverse_plays=True)

    assert forward.result_digest == reversed_order.result_digest
    assert forward.request.replay_id == reversed_order.request.replay_id
    assert [row.trade_id for row in forward.trades] == [
        row.trade_id for row in reversed_order.trades]
    assert forward.trades


def test_a_repeated_run_over_the_same_inputs_is_the_same_result():
    """Local repetition is byte equality, not incidental equality."""
    first = replay_result(plays=one_play(), symbol_order=("SPY",))
    second = replay_result(plays=one_play(), symbol_order=("SPY",))

    assert encode_replay_result(first) == encode_replay_result(second)
    assert first.result_digest == second.result_digest


def test_the_digest_is_the_result_with_its_own_digest_field_omitted():
    """Recomputable by a settler that never saw the run.

    The field is excluded from its own input, so a receiver can recompute the
    digest over the decoded value and compare. A digest that hashed itself
    could never be checked at all.
    """
    result = replay_result(plays=one_play(), symbol_order=("SPY",))

    assert result_digest(result) == result.result_digest
    assert result_digest(dataclasses.replace(result, result_digest="0" * 64)) == (
        result.result_digest)


def test_a_result_survives_the_transport_codec_unchanged():
    """Strict JSON out, the same value and the same digest back in."""
    result = replay_result(plays=one_play(), symbol_order=("SPY",))

    round_tripped = decode_replay_result(encode_replay_result(result))

    assert round_tripped == result
    assert result_digest(round_tripped) == result.result_digest


# ------------------------------------------------------- stateful composites


def test_a_stateful_play_on_two_symbols_keeps_two_separate_runtimes():
    """One runtime per `(play_id, symbol)`, proven by what each one counted.

    The scripted strategy counts its own calls, and a signal is emitted only on
    the call the plan names. Both symbols signal at the same close, which one
    shared instance could not do without its counter running through.
    """
    result = replay_result(plays=(ScriptedPlay(
        play_id="play-a",
        signals=tuple(SignalPlan(symbol=symbol, at=closes(0), stop=99.0,
                                 target=103.0)
                      for symbol in ("QQQ", "SPY"))),))

    assert attributed(result.trades) == (
        ("play-a", "QQQ", 0), ("play-a", "SPY", 1))
    assert {slice_of(result, "play-a", symbol).signals
            for symbol in ("QQQ", "SPY")} == {1}


def test_one_definition_under_two_play_ids_is_two_runtimes():
    """The same strategy body, twice, with no state crossing between them."""
    plan = tuple(SignalPlan(symbol=symbol, at=closes(0), stop=99.0, target=103.0)
                 for symbol in ("QQQ", "SPY"))
    result = replay_result(plays=(
        ScriptedPlay(play_id="play-a", signals=plan),
        ScriptedPlay(play_id="play-b", signals=plan),
    ))

    assert len(result.trades) == 4
    assert {(row.play_id, row.symbol) for row in result.slices} == {
        (play_id, symbol)
        for play_id in ("play-a", "play-b") for symbol in ("QQQ", "SPY")}


# ------------------------------------------------- missing and malformed bars


def test_a_missing_scheduled_bar_refuses_the_whole_replay():
    """Strict refusal: no forward fill, no dropped symbol, no partial result."""
    inputs = replay_inputs(plays=one_play(), symbol_order=("SPY",))
    frames = dict(inputs.bars)
    frames[("SPY", "15m")] = frames[("SPY", "15m")].iloc[:-1]

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(inputs.request, PortfolioBars(frames), inputs.registry,
                      inputs.schedule)

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["field"] == "labels"


def test_a_malformed_frame_refuses_the_whole_replay():
    """A bar whose high is not its highest price is a data defect, not a bar."""
    inputs = replay_inputs(plays=one_play(), symbol_order=("SPY",))
    frames = dict(inputs.bars)
    broken = frames[("SPY", "15m")].copy(deep=True)
    broken.iloc[0, broken.columns.get_loc("high")] = 1.0
    frames[("SPY", "15m")] = broken

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(inputs.request, PortfolioBars(frames), inputs.registry,
                      inputs.schedule)

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["field"] == "high"


def test_a_benchmark_symbol_with_no_bars_refuses_before_any_runtime():
    """An explicit benchmark adds a symbol to the required set."""
    registry, calls = counting_registry()
    request = base_request(benchmark=single_symbol_benchmark("IWM"))

    with pytest.raises(ReplayInputError) as raised:
        run_portfolio(request, PortfolioBars(catalog_frames()), registry,
                      base_schedule())

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["symbol"] == "IWM"
    assert calls.factory_count == 0


# --------------------------------------------------------- schedule boundaries


def test_the_window_ends_the_replay_even_though_the_schedule_runs_on():
    """The clock is the schedule; the loop is the window.

    The schedule carries forty intervals and this window opens on fourteen of
    them, so the equity series has the anchor, those fourteen closes, and the
    post-close point, and nothing from the tail.
    """
    result = replay_result(plays=one_play(), symbol_order=("SPY",))

    assert len(result.equity) == POINT_COUNT
    assert result.equity[0].ts == FIRST_OPEN
    assert [point.ts for point in result.equity[-2:]] == [TEST_END, TEST_END]
    assert [point.point_ordinal for point in result.equity] == list(
        range(POINT_COUNT))


def test_a_signal_at_the_last_close_has_no_open_to_fill_at():
    """The window boundary expires a pending intent rather than filling it."""
    result = replay_result(signal_at=closes(LAST_INTERVAL))

    assert result.trades == ()
    assert {row.reason for row in result.rejections} == {
        RejectionReason.WINDOW_ENDED}
    assert {row.event_ts for row in result.rejections} == {TEST_END}


# ------------------------------------------------------------- T plus one


def test_exit_proceeds_are_unsettled_at_the_final_point():
    """A credit raised on the schedule's last session never settles.

    The forced liquidation happens at the window's end, and this schedule has
    no session after it, so the proceeds contribute to equity and stay
    unsettled. That is the honest answer rather than a settlement the calendar
    cannot support.
    """
    result = replay_result(plays=one_play(), symbol_order=("SPY",))

    final = result.equity[-1]
    assert (final.settled_cash, final.unsettled_cash) == (90_000.0, 10_000.0)
    assert final.portfolio_equity == 100_000.0
    assert final.open_positions == 0


def test_proceeds_are_unavailable_that_session_and_settled_the_next_one():
    """The whole T+1 rule in one replay, across the Thanksgiving holiday.

    The window opens on 2026-11-25 at 17:00Z and runs to the 2026-11-27 half
    day's close. One percent of 100,000 over a dollar of protective distance is
    1,000 shares at 100, which is the whole account.

    - 17:15 the play signals and fills, spending every settled dollar;
    - 17:30 it manages out, raising 100,000 of credit that settles on the next
      EXCHANGE session, which is 2026-11-27 and not 2026-11-26;
    - 17:45 it signals again, and the next open is the same instant. That is
      still 2026-11-25, the proceeds have not settled, and the entry is refused
      for cash with nothing available;
    - 2026-11-27 at 14:45 it signals a third time. That instant is both the
      first interval's close and the second interval's open, the session's
      credits settled when the session's first interval began, and the entry
      fills.

    A ledger that settled immediately would have funded the second entry from
    the first one's own money, and one that settled on the next CALENDAR day
    would have settled on 2026-11-26, which this exchange never traded.
    """
    window = ReplayWindow(
        train_start=ts("2026-11-25T14:30:00Z"),
        train_end=ts("2026-11-25T17:00:00Z"),
        test_start=ts("2026-11-25T17:00:00Z"),
        test_end=TEST_END,
    )
    result = replay_result(
        symbol_order=("SPY",), window=window,
        account=replay_account(risk_pct=0.01),
        plays=one_play(
            signals=(
                SignalPlan(symbol="SPY", at=ts("2026-11-25T17:15:00Z"),
                           stop=99.0, target=103.0),
                SignalPlan(symbol="SPY", at=ts("2026-11-25T17:45:00Z"),
                           stop=99.0, target=103.0),
                SignalPlan(symbol="SPY", at=ts("2026-11-27T14:45:00Z"),
                           stop=99.0, target=103.0),
            ),
            manages=(ManagePlan(symbol="SPY", at=ts("2026-11-25T17:30:00Z"),
                                action="exit"),),
        ),
    )

    assert [(row.entry_ts, row.exit_ts, row.exit_reason) for row in result.trades] == [
        (ts("2026-11-25T17:15:00Z"), ts("2026-11-25T17:30:00Z"), ExitReason.MANAGE),
        (ts("2026-11-27T14:45:00Z"), TEST_END, ExitReason.END_OF_WINDOW),
    ]
    (refused,) = result.rejections
    assert (refused.reason, refused.event_ts, refused.available_cash) == (
        RejectionReason.UNSETTLED_CASH, ts("2026-11-25T17:45:00Z"), 0.0)
    assert result.equity[-1].unsettled_cash == 100_000.0


# ---------------------------------------------------------------- geometry


def test_a_bar_touching_both_levels_takes_the_stop():
    """Pessimistic mode, at the whole-result level.

    The entry bar opens between the levels and trades through both. OHLC cannot
    say which came first, so the stop is assumed and the trade is a loss.
    """
    result = replay_result(
        symbol_order=("SPY",), plays=one_play(),
        bars=(BarPlan(symbol="SPY", at=opens(1), open=100.0, high=104.0,
                      low=98.0, close=100.0),),
    )

    (trade,) = result.trades
    assert trade.exit_reason is ExitReason.STOP
    assert trade.exit == 99.0
    assert trade.net_pnl < 0.0
    assert result.metrics.all_trades.n_wins == 0


def test_a_fill_slipped_onto_its_target_never_becomes_a_position():
    """The raw open brackets protectively and the slipped fill does not.

    `base_execution` charges two basis points with a one-cent floor, so a buy
    at a raw open of 100.0 prints at 100.02. A target at 100.01 sits above the
    raw open and below the fill, which is a position whose own target is
    already inside it. Nothing here is a cash or capacity refusal: the account
    holds 100,000 and five seats.

    The refusal is recorded with its own reason rather than dropped, which is
    what tells a refused entry from one that was never proposed, and the reason
    names the geometry rather than the account.
    """
    result = replay_result(
        symbol_order=("SPY",), execution=base_execution(),
        plays=one_play(signals=(SignalPlan(symbol="SPY", at=closes(0),
                                           stop=99.0, target=100.01),)),
    )

    assert result.trades == ()
    (refused,) = result.rejections
    assert refused.reason is RejectionReason.INVALID_PROTECTIVE_GEOMETRY
    assert (refused.required_cash, refused.available_cash) == (None, None)
    assert slice_of(result, "play-a", "SPY").signals == 1


def test_a_gap_beyond_the_stop_books_at_the_open_it_could_have_traded():
    """The bar opens under the stop, so the open is the fill and not the level."""
    result = replay_result(
        symbol_order=("SPY",), plays=one_play(),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=97.0, high=97.0,
                      low=97.0, close=97.0),),
    )

    (trade,) = result.trades
    assert (trade.exit_reason, trade.exit, trade.exit_ts) == (
        ExitReason.STOP_GAP, 97.0, opens(2))


# -------------------------------------------------------- exception mapping


def test_a_strategy_that_raises_aborts_the_replay_as_a_runtime_error():
    """Never an empty signal list, and never a partial result."""
    with pytest.raises(StrategyRuntimeError) as raised:
        replay_result(symbol_order=("SPY",),
                      plays=one_play(on_bar_raises="boom"))

    assert raised.value.code == "strategy_raised"
    assert raised.value.details["operation"] == "on_bar"


def test_a_malformed_signal_sequence_aborts_as_a_strategy_output_error():
    with pytest.raises(StrategyOutputError) as raised:
        replay_result(symbol_order=("SPY",),
                      plays=one_play(on_bar_returns="not a sequence"))

    assert raised.value.code == "invalid_type"


def test_a_factory_that_raises_aborts_before_any_interval_is_replayed():
    with pytest.raises(StrategyRuntimeError) as raised:
        replay_result(symbol_order=("SPY",),
                      plays=one_play(factory_raises="no runtime"))

    assert raised.value.details["operation"] == "construct"


# ------------------------------------------------------ multi-signal output


def test_every_returned_signal_is_recorded_in_the_order_it_was_returned():
    """Two signals from one call. The first takes the seat and the second is
    refused for it, so both appear and neither is silently dropped."""
    result = replay_result(
        symbol_order=("SPY",),
        plays=one_play(signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),
            SignalPlan(symbol="SPY", at=closes(0), stop=98.0, target=104.0),
        )),
    )

    assert slice_of(result, "play-a", "SPY").signals == 2
    assert [row.reason for row in result.rejections] == [
        RejectionReason.PENDING_INTENT_OCCUPIED]
    assert len(result.trades) == 1


# ------------------------------------------------------------- benchmark


def test_the_equal_weight_benchmark_holds_the_request_symbols():
    """A flat tape leaves the basket exactly where it started, for any count."""
    result = replay_result(plays=one_play())

    assert result.benchmark.spec.kind == "equal_weight_request_symbols"
    assert result.benchmark.total_return == 0.0
    assert {point.benchmark_equity for point in result.equity} == {100_000.0}
    assert result.metrics.benchmark_return == result.benchmark.total_return


def test_an_explicit_benchmark_symbol_never_touches_the_account():
    """A named benchmark is marked, never traded, and moves nothing.

    IWM is not in the request's symbols, so no play can trade it, and its close
    is planted at 150 on the last interval while SPY stays flat at 100. The
    basket therefore ends fifty percent up while the account ends exactly where
    it started, which is a benchmark that was read and an account that was not
    touched by it.
    """
    result = replay_result(
        symbol_order=("SPY",), plays=one_play(),
        benchmark=single_symbol_benchmark("IWM"),
        bars=(BarPlan(symbol="IWM", at=opens(LAST_INTERVAL), open=100.0,
                      high=150.0, low=100.0, close=150.0),),
    )

    assert result.benchmark.spec.symbol == "IWM"
    assert result.benchmark.total_return == 0.5
    assert {row.symbol for row in result.trades} == {"SPY"}
    assert result.equity[-1].portfolio_equity == 100_000.0
    assert result.equity[-1].benchmark_equity == 150_000.0


# ------------------------------------------------------- play symbol slices


def test_every_canonical_play_symbol_owns_exactly_one_slice():
    """Four runtimes, four slices, including the ones that never spoke."""
    result = replay_result(plays=(ScriptedPlay(
        play_id="play-a",
        signals=(SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),)),
        ScriptedPlay(play_id="play-b")))

    assert len(result.slices) == 4
    assert slice_of(result, "play-b", "QQQ").signals == 0
    assert slice_of(result, "play-b", "QQQ").trades == 0
    assert slice_of(result, "play-a", "SPY").trades == 1


def test_slice_totals_reconcile_to_the_parent_totals():
    """The attribution identity: the slices add up to the portfolio.

    Trades, fees, and pre-cost PnL are plain sums at both levels, so they
    reconcile term for term. A slice that dropped or duplicated a trade could
    not.
    """
    result = replay_result(account=replay_account(max_open_positions=2),
                           execution=None)

    assert sum(row.trades for row in result.slices) == len(result.trades)
    assert sum(row.signals for row in result.slices) == 4
    assert sum(sum(row.rejection_counts.values()) for row in result.slices) == (
        result.metrics.n_rejections)
    assert sum(row.fees for row in result.slices) == result.metrics.fees


# ----------------------------------------------------------- IC tail isolation


def test_ic_lives_on_slices_and_never_on_the_parent():
    """No portfolio IC exists and none is derivable from what is reported."""
    result = replay_result(plays=ic_plays(margin=lambda position, at: float(-position)),
                           window=ic_window(), build=ramp_frame)

    assert not hasattr(result.metrics, "ic")
    for row in result.slices:
        assert [item.horizon_bars for item in row.ic] == [1, 5, 20]


def test_the_ic_tail_is_read_as_an_outcome_and_never_shown_to_a_strategy():
    """The observations end at `test_end` while the forward return runs on.

    Twenty-five observations for a window that tests ordinals 1 through 25 of
    the schedule's forty intervals, and the twenty-bar horizon reaches a close
    for only the first nineteen of them. A lens that let the factor see the
    tail would report twenty-five at every horizon.
    """
    result = replay_result(plays=ic_plays(margin=lambda position, at: float(-position)),
                           window=ic_window(), build=ramp_frame)

    row = slice_of(result, "play-a", "SPY")
    assert [item.observations for item in row.ic] == [25, 25, 19]


def test_an_ungraded_definition_reports_no_measurement_rather_than_zero():
    """Zero observations at every horizon, which is not the same as a null
    correlation over a real sample."""
    result = replay_result(plays=ic_plays(timeframe=None), window=ic_window())

    for row in result.slices:
        assert [item.correlation for item in row.ic] == [None, None, None]
        assert [item.observations for item in row.ic] == [0, 0, 0]
