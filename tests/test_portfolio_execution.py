"""The replay chronology: what happens at an interval, and in what order.

Every test here runs a real replay through `replay_fixture`, which assembles
the actual request, schedule, registry, bars, and runtime. Nothing in this file
reimplements a step of the loop, so an assertion that passes is a statement
about the engine rather than about a stand-in for it.

Two kinds of assertion carry the weight:

- ORDER goldens, written as one exact tuple of `(play, symbol, signal ordinal)`
  in event order. The fixtures give two plays ONE priority on purpose, which
  makes signal order play-major and funding order symbol-major, so a golden
  that merely restated the code would have to state two different orders and
  get both right;
- ARITHMETIC goldens, written as exact literals. The default bars are flat, so
  every quantity and cash figure is a whole number a reader can check by hand,
  and a test that needs a level reached puts one bar where it is reached.
"""

import dataclasses

import pandas as pd
import pytest

from nakagai.engine.portfolio_types import (
    ExitReason,
    RejectionReason,
    ReplayInputError,
    ReplayWindow,
    StrategyOutputError,
    StrategyRuntimeError,
)
from tests.portfolio_fixtures import (
    BarPlan,
    ManagePlan,
    ScriptedPlay,
    SignalPlan,
    base_dependencies,
    base_execution,
    canonical_event_bytes,
    replay_account,
    replay_fixture,
    ts,
)

# The default window trades the 2026-11-27 half day: fourteen 15-minute
# intervals from 14:30Z to 18:00Z, which is also the schedule's last close.
FIRST_OPEN = ts("2026-11-27T14:30:00Z")
TEST_END = ts("2026-11-27T18:00:00Z")
LAST_INTERVAL = 13
MARK_COUNT = 16  # the opening anchor, fourteen closes, and the post-close mark


def opens(ordinal: int) -> pd.Timestamp:
    """The open of test interval `ordinal` under the default window."""
    return FIRST_OPEN + pd.Timedelta(minutes=15 * ordinal)


def closes(ordinal: int) -> pd.Timestamp:
    return opens(ordinal) + pd.Timedelta(minutes=15)


def attributed(rows) -> tuple:
    """Each event as `(play, symbol, signal ordinal)`, in its own event order."""
    return tuple((row.play_id, row.symbol, row.signal_ordinal) for row in rows)


def one_play(**overrides) -> tuple[ScriptedPlay, ...]:
    """One play trading SPY, signalling once at the first close by default."""
    plan = SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0)
    return (dataclasses.replace(
        ScriptedPlay(play_id="play-a", signals=(plan,)), **overrides,
    ),)


# ------------------------------------------------------------ event ordering


def test_signals_number_play_major_while_positions_close_symbol_major():
    """The two canonical orders are different orders, and both are pinned here.

    Evaluation visits plays then symbols, so the ordinals run a, a, b, b. Every
    step that later visits POSITIONS visits them by symbol first, so the
    end-of-window closes run QQQ, QQQ, SPY, SPY and carry ordinals 0, 2, 1, 3.
    An engine that used one order for both would fail one half of this.
    """
    events = replay_fixture()

    assert attributed(events.trades) == (
        ("play-a", "QQQ", 0), ("play-b", "QQQ", 2),
        ("play-a", "SPY", 1), ("play-b", "SPY", 3),
    )
    assert {trade.qty for trade in events.trades} == {100}
    assert {trade.exit_reason for trade in events.trades} == {
        ExitReason.END_OF_WINDOW}
    assert dict(events.signal_counts) == {
        ("play-a", "QQQ"): 1, ("play-a", "SPY"): 1,
        ("play-b", "QQQ"): 1, ("play-b", "SPY"): 1,
    }


def test_eligible_intents_fund_in_priority_symbol_signal_order():
    """One seat, four candidates: the seat goes to the first in funding order
    and the other three refuse in that same order.

    Funding order is symbol-major, so the survivor is the QQQ intent of the
    lower play id and the refusals carry signal ordinals 2, 1, 3. A play-major
    funding order would leave 1, 2, 3.
    """
    events = replay_fixture(account=replay_account(max_open_positions=1))

    assert attributed(events.trades) == (("play-a", "QQQ", 0),)
    assert attributed(events.rejections) == (
        ("play-b", "QQQ", 2), ("play-a", "SPY", 1), ("play-b", "SPY", 3),
    )
    assert {row.reason for row in events.rejections} == {
        RejectionReason.PORTFOLIO_CAPACITY}
    assert {row.event_ts for row in events.rejections} == {opens(1)}
    assert {row.open_positions for row in events.rejections} == {1}


def test_request_collection_order_cannot_change_result_bytes():
    """Supply order carries no meaning: symbols, plays, param keys, and the bar
    mapping's own insertion order all reverse without moving one byte."""
    left = replay_fixture(symbol_order=("SPY", "QQQ"), reverse_param_keys=False)
    right = replay_fixture(symbol_order=("QQQ", "SPY"), reverse_param_keys=True,
                           reverse_plays=True)

    assert canonical_event_bytes(left) == canonical_event_bytes(right)
    assert left.trades and left.marks


# --------------------------------------------------------- interval sequence


def test_a_position_filled_at_this_open_exits_on_its_own_bar():
    """Spec step 7 protects a position filled at THIS interval's open.

    The entry bar reaches the stop it was filled with. Checking exits before
    filling would leave the position open, so this trade would come back as an
    end-of-window close at 18:00 rather than a stop at 15:00.
    """
    events = replay_fixture(
        symbol_order=("SPY",), plays=one_play(),
        bars=(BarPlan(symbol="SPY", at=opens(1), open=100.0, high=100.0,
                      low=98.5, close=99.5),),
    )

    (trade,) = events.trades
    assert trade.exit_reason == ExitReason.STOP
    assert (trade.entry_ts, trade.exit_ts) == (opens(1), closes(1))
    assert (trade.entry, trade.exit) == (100.0, 99.0)
    assert (trade.mae, trade.mfe) == (1.0, 0.0)
    assert trade.r_multiple == -1.0


def test_a_carried_position_that_reaches_its_stop_exits_at_the_close():
    """A level the OPEN was not beyond, reached inside the bar, is step 7's.

    Step 3 has to decline this one. Acting on it there would stamp the open
    rather than the close, take a trade ordinal ahead of the fills at that open,
    and credit an excursion that had only ever seen the opening print.
    """
    events = replay_fixture(
        symbol_order=("SPY",), plays=one_play(),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=100.0, high=100.0,
                      low=98.5, close=99.5),),
    )

    (trade,) = events.trades
    assert (trade.exit_reason, trade.exit_ts, trade.exit) == (
        ExitReason.STOP, closes(2), 99.0)
    assert (trade.mae, trade.mfe) == (1.0, 0.0)


def test_a_gap_exit_books_at_the_open_and_frees_its_seat_first():
    """Spec step 3 runs before spec step 6, and only on the gap reasons.

    The carried position's bar opens below its stop, so it exits at that open
    rather than at the close, and the seat it releases is free for the intent
    funded at the same open. An engine that funded first would refuse that
    intent for capacity.

    The excursion is the other half. A gap exit folds the opening print alone,
    because that print is the first and last price the position saw on this bar,
    so the low it never traded at cannot deepen its adverse excursion.
    """
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=(
            ScriptedPlay(play_id="play-a", signals=(
                SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),)),
            ScriptedPlay(play_id="play-b", signals=(
                SignalPlan(symbol="SPY", at=closes(1), stop=93.0, target=105.0),)),
        ),
        account=replay_account(max_open_positions=1),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=96.0, high=96.0,
                      low=94.0, close=95.0),),
    )

    gapped, funded = events.trades
    assert (gapped.play_id, gapped.exit_reason) == ("play-a", ExitReason.STOP_GAP)
    assert (gapped.exit_ts, gapped.exit) == (opens(2), 96.0)
    assert (gapped.mae, gapped.mfe) == (4.0, 0.0)
    assert (funded.play_id, funded.entry_ts, funded.qty) == ("play-b", opens(2), 33)
    assert events.rejections == ()


def test_a_gap_above_the_target_books_at_the_open_too():
    """The other gap reason, on the other side of the position.

    A bar that opens above a long's target never offered that target, so the
    open is the reference and the open is the instant. Recognizing only the stop
    gap would send this one to step 7 and stamp it at the close, behind the
    fills at this open.
    """
    events = replay_fixture(
        symbol_order=("SPY",), plays=one_play(),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=104.0, high=105.0,
                      low=103.5, close=104.5),),
    )

    (trade,) = events.trades
    assert (trade.exit_reason, trade.exit_ts, trade.exit) == (
        ExitReason.TARGET_GAP, opens(2), 104.0)
    # The opening print alone: the 105 high it never traded at cannot flatter
    # the excursion of a position that had already left.
    assert (trade.mae, trade.mfe) == (0.0, 4.0)


def test_a_short_target_exit_credits_the_high_it_traded_through():
    """The mirror of the long stop, and the only exit that credits an extreme.

    Pessimistic ordering assumes the adverse extreme came first, so a target
    exit folds it. For a short that extreme is the HIGH, and reading the low
    there would report an adverse excursion of zero on a bar that ran half a
    point against the position.
    """
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=(ScriptedPlay(play_id="play-a", signals=(
            SignalPlan(symbol="SPY", at=closes(0), direction="short",
                       stop=101.0, target=97.0),)),),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=100.0, high=100.5,
                      low=96.5, close=97.0),),
    )

    (trade,) = events.trades
    assert (trade.direction, trade.exit_reason) == ("short", ExitReason.TARGET)
    assert (trade.exit_ts, trade.exit, trade.qty) == (closes(2), 97.0, 100)
    assert (trade.mae, trade.mfe) == (0.5, 3.0)
    assert trade.r_multiple == 3.0


def test_gap_proceeds_cannot_fund_a_fill_at_the_same_open():
    """The exit at this open credits T+1 cash, so the intent funded at the same
    open sees the settled balance from before it: zero, here."""
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=(
            ScriptedPlay(play_id="play-a", signals=(
                SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),)),
            ScriptedPlay(play_id="play-b", signals=(
                SignalPlan(symbol="SPY", at=closes(1), stop=95.0, target=105.0),)),
        ),
        account=replay_account(10_000.0, risk_pct=0.01),
        bars=(BarPlan(symbol="SPY", at=opens(2), open=96.0, high=96.0,
                      low=96.0, close=96.0),),
    )

    (refusal,) = events.rejections
    assert refusal.reason == RejectionReason.UNSETTLED_CASH
    assert (refusal.required_cash, refusal.available_cash) == (9_216.0, 0.0)
    assert (refusal.event_ts, refusal.open_positions) == (opens(2), 0)


def test_every_candidate_at_one_open_is_sized_from_one_frozen_equity():
    """Four fills at one open, all sized from the equity frozen before any of
    them. Re-marking between fills costs the account its slippage and fees, and
    would size the later candidates at 908 shares instead of 909."""
    events = replay_fixture(price=10.0, execution=base_execution(),
                            signal_stop=9.9, signal_target=10.3)

    assert len(events.trades) == 4
    assert {trade.qty for trade in events.trades} == {909}


def test_due_credits_settle_before_the_open_they_fund():
    """A credit raised on one session funds nothing until the next session's
    open, and funds a fill there.

    The refusal and the fill are the same candidate at the same price with the
    same budget. Only the settlement date differs, so this is the T+1 rule and
    nothing else.
    """
    signal = SignalPlan(symbol="SPY", at=ts("2026-11-25T15:15:00Z"),
                        stop=99.0, target=103.0)
    events = replay_fixture(
        symbol_order=("SPY",),
        window=ReplayWindow(
            train_start=ts("2026-11-25T14:30:00Z"),
            train_end=ts("2026-11-25T14:45:00Z"),
            test_start=ts("2026-11-25T14:45:00Z"),
            test_end=TEST_END,
        ),
        plays=(
            ScriptedPlay(play_id="play-a", signals=(
                SignalPlan(symbol="SPY", at=ts("2026-11-25T15:00:00Z"),
                           stop=99.0, target=103.0),
            ), manages=(
                ManagePlan(symbol="SPY", at=ts("2026-11-25T15:15:00Z"),
                           action="exit"),
            )),
            ScriptedPlay(play_id="play-b", signals=(
                signal,
                dataclasses.replace(signal, at=ts("2026-11-25T21:00:00Z")),
            )),
        ),
        account=replay_account(10_000.0, risk_pct=0.01),
    )

    (refusal,) = events.rejections
    assert refusal.reason == RejectionReason.UNSETTLED_CASH
    assert (refusal.event_ts, refusal.available_cash) == (
        ts("2026-11-25T15:15:00Z"), 0.0)
    assert [(trade.play_id, trade.entry_ts, trade.exit_reason)
            for trade in events.trades] == [
        ("play-a", ts("2026-11-25T15:00:00Z"), ExitReason.MANAGE),
        ("play-b", ts("2026-11-27T14:30:00Z"), ExitReason.END_OF_WINDOW),
    ]


def test_management_runs_after_the_entry_bar_and_before_evaluation():
    """Spec step 10 before spec step 11, and no management before the fill.

    The position is filled at the open of the second interval, so its first
    management call is that interval's close, and at every close after it the
    engine manages before it evaluates.
    """
    calls: list = []
    events = replay_fixture(
        symbol_order=("SPY",), calls=calls,
        plays=one_play(manages=(
            ManagePlan(symbol="SPY", at=closes(1), stop=99.5),
            ManagePlan(symbol="SPY", at=closes(2), action="exit"),
        )),
    )

    assert [(call.operation, call.now) for call in calls][:6] == [
        ("on_bar", closes(0)),
        ("manage", closes(1)), ("on_bar", closes(1)),
        ("manage", closes(2)), ("on_bar", closes(2)),
        ("on_bar", closes(3)),
    ]
    (trade,) = events.trades
    assert (trade.exit_reason, trade.exit_ts) == (ExitReason.MANAGE, closes(2))
    assert (trade.initial_stop, trade.final_stop) == (99.0, 99.5)


def test_a_context_shows_only_what_the_schedule_has_released():
    """Availability is the schedule's answer, never label arithmetic.

    The hourly bar labeled 14:00 on the trading session is available at 15:00,
    so it is invisible at the 14:45 close and visible at the 15:00 one. The
    four-hour bucket of an early close becomes available after the session ends
    and is therefore never visible at all. The last base bar is always the one
    closing at `now`, so nothing later than the deciding bar is readable.
    """
    calls: list = []
    events = replay_fixture(symbol_order=("SPY",), calls=calls,
                            dependencies=base_dependencies(),
                            plays=(ScriptedPlay(play_id="play-a"),))

    # Every canonical play symbol is counted, including one that never signals.
    assert dict(events.signal_counts) == {("play-a", "SPY"): 0}
    assert [(call.now, call.visible, call.last_base_label)
            for call in calls][:2] == [
        (closes(0), (("15m", 27), ("1h", 1), ("4h", 0), ("1d", 1)), opens(0)),
        (closes(1), (("15m", 28), ("1h", 2), ("4h", 0), ("1d", 1)), opens(1)),
    ]
    assert {dict(call.visible)["4h"] for call in calls} == {0}


def test_every_returned_signal_is_processed_in_its_returned_order():
    """Two signals from one call are two proposals. The first takes the play
    symbol's one pending seat and the second is recorded rather than dropped."""
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=one_play(signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),
            SignalPlan(symbol="SPY", at=closes(0), stop=98.0, target=104.0),
        )),
    )

    (trade,) = events.trades
    assert (trade.signal_ordinal, trade.initial_stop) == (0, 99.0)
    (refusal,) = events.rejections
    assert refusal.reason == RejectionReason.PENDING_INTENT_OCCUPIED
    assert (refusal.signal_ordinal, refusal.signal_ts, refusal.event_ts) == (
        1, closes(0), closes(0))
    # Counted where it was emitted, not where it was acted on: a refused signal
    # is still a signal this play symbol produced.
    assert events.signal_counts[("play-a", "SPY")] == 2


def test_a_held_play_symbol_records_position_occupied_at_the_close():
    """A signal on an occupied play symbol is refused where it was decided, so
    its event timestamp is the deciding close rather than a later open."""
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=one_play(signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),
            SignalPlan(symbol="SPY", at=closes(2), stop=99.0, target=103.0),
        )),
    )

    (refusal,) = events.rejections
    assert refusal.reason == RejectionReason.POSITION_OCCUPIED
    assert (refusal.signal_ordinal, refusal.event_ts) == (1, closes(2))
    assert refusal.open_positions == 1


# --------------------------------------------------------------- window close


def test_the_final_close_marks_then_expires_then_liquidates_then_marks():
    """The last close carries two marks and the whole liquidation between them.

    The first is the ordinary close mark, taken while the position is still
    open. The second is the post-liquidation mark at the same instant. A
    liquidation that ran before the ordinary mark would leave both at zero
    positions.
    """
    events = replay_fixture(
        symbol_order=("SPY",),
        plays=(
            ScriptedPlay(play_id="play-a", signals=(
                SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),)),
            ScriptedPlay(play_id="play-b", signals=(
                SignalPlan(symbol="SPY", at=closes(LAST_INTERVAL),
                           stop=99.0, target=103.0),)),
        ),
    )

    assert len(events.marks) == MARK_COUNT
    assert events.marks[0].ts == FIRST_OPEN
    assert events.marks[0].snapshot.settled_cash == 100_000.0
    assert [mark.ts for mark in events.marks[-2:]] == [TEST_END, TEST_END]
    assert [mark.snapshot.open_positions for mark in events.marks[-2:]] == [1, 0]
    assert [mark.snapshot.portfolio_equity for mark in events.marks[-2:]] == [
        100_000.0, 100_000.0]
    # The liquidation raises cash on the schedule's LAST session, so nothing
    # can settle it: it counts toward equity and stays unsettled forever, which
    # is the honest answer when the replay's calendar ends first.
    assert (events.marks[-1].snapshot.settled_cash,
            events.marks[-1].snapshot.unsettled_cash) == (90_000.0, 10_000.0)
    (expired,) = events.rejections
    assert (expired.play_id, expired.reason, expired.event_ts) == (
        "play-b", RejectionReason.WINDOW_ENDED, TEST_END)
    (trade,) = events.trades
    assert (trade.exit_reason, trade.exit_ts) == (
        ExitReason.END_OF_WINDOW, TEST_END)


def test_pending_intents_expire_in_funding_order():
    """Four intents decided at the last close have no eligibility open inside
    the window, so all four expire, in the order the account would have funded
    them."""
    events = replay_fixture(signal_at=closes(LAST_INTERVAL))

    assert attributed(events.rejections) == (
        ("play-a", "QQQ", 0), ("play-b", "QQQ", 2),
        ("play-a", "SPY", 1), ("play-b", "SPY", 3),
    )
    assert {row.reason for row in events.rejections} == {
        RejectionReason.WINDOW_ENDED}
    assert events.trades == ()


def test_expiry_order_puts_the_lower_priority_play_first():
    """Priority outranks symbol, which the equal-priority goldens cannot show.

    Play b acts first here, so it is evaluated first and numbered first as well.
    Dropping priority from the order would interleave the two plays by symbol
    instead: QQQ from b, QQQ from a, then the two SPY intents.
    """
    signals = tuple(SignalPlan(symbol=symbol, at=closes(LAST_INTERVAL),
                               stop=99.0, target=103.0)
                    for symbol in ("QQQ", "SPY"))
    events = replay_fixture(plays=(
        ScriptedPlay(play_id="play-a", priority=100, signals=signals),
        ScriptedPlay(play_id="play-b", priority=50, signals=signals),
    ))

    assert attributed(events.rejections) == (
        ("play-b", "QQQ", 0), ("play-b", "SPY", 1),
        ("play-a", "QQQ", 2), ("play-a", "SPY", 3),
    )


def test_only_intervals_opening_inside_the_window_are_replayed():
    """The schedule runs on to the IC tail. The event loop stops at the window,
    so an intent whose one eligibility open lies in the tail expires."""
    events = replay_fixture(
        symbol_order=("SPY",),
        window=ReplayWindow(
            train_start=ts("2026-11-25T14:30:00Z"),
            train_end=FIRST_OPEN,
            test_start=FIRST_OPEN,
            test_end=ts("2026-11-27T17:00:00Z"),
        ),
        plays=one_play(signals=(
            SignalPlan(symbol="SPY", at=ts("2026-11-27T17:00:00Z"),
                       stop=99.0, target=103.0),)),
    )

    assert len(events.marks) == 12
    assert events.marks[-1].ts == ts("2026-11-27T17:00:00Z")
    (expired,) = events.rejections
    assert (expired.reason, expired.event_ts) == (
        RejectionReason.WINDOW_ENDED, ts("2026-11-27T17:00:00Z"))
    assert events.trades == ()


# ------------------------------------------------------------ error boundary


def test_strategy_exception_is_not_an_empty_signal_list():
    """A raising strategy leaves the replay raising, rather than returning what
    it had already booked. The same script books a trade when it does not
    raise, so the refusal is the exception and not an absence of signals."""
    booked = replay_fixture(symbol_order=("SPY",), plays=one_play())
    assert len(booked.trades) == 1

    with pytest.raises(StrategyRuntimeError) as caught:
        replay_fixture(symbol_order=("SPY",),
                       plays=one_play(on_bar_raises="boom",
                                      raises_at=closes(5)))

    assert caught.value.code == "strategy_raised"
    assert caught.value.details["operation"] == "on_bar"
    assert caught.value.details["symbol"] == "SPY"
    # The sixth close, so the replay had already filled a position and booked
    # nothing of it: a partial result existed and did not escape.
    assert caught.value.details["event_ts"] == closes(5).isoformat()


def test_a_wrapped_strategy_error_names_its_runtime_without_a_traceback():
    with pytest.raises(StrategyRuntimeError) as caught:
        replay_fixture(symbol_order=("SPY",), plays=one_play(on_bar_raises="boom"))

    details = caught.value.details
    assert dict(details) == {
        "operation": "on_bar", "error": "RuntimeError",
        "strategy": "scripted-play-a", "symbol": "SPY", "play_id": "play-a",
        "event_ts": closes(0).isoformat(),
    }


@pytest.mark.parametrize(("returned", "field"), [
    (("not a signal",), "signals[0]"),
    ("SPY", "on_bar"),
])
def test_a_malformed_return_is_a_strategy_output_error(returned, field):
    with pytest.raises(StrategyOutputError) as caught:
        replay_fixture(symbol_order=("SPY",),
                       plays=one_play(on_bar_returns=returned))

    assert caught.value.code == "invalid_type"
    assert caught.value.details["field"] == field


def test_a_malformed_management_decision_is_a_strategy_output_error():
    """The management half of the strategy boundary, reached from the loop."""
    with pytest.raises(StrategyOutputError) as caught:
        replay_fixture(symbol_order=("SPY",),
                       plays=one_play(manage_returns="hold"))

    assert caught.value.code == "invalid_type"
    assert caught.value.details["field"] == "manage"


@pytest.mark.parametrize(("behavior", "operation"), [
    ({"factory_raises": "no runtime"}, "construct"),
    ({"manage_raises": "no decision"}, "manage"),
])
def test_construction_and_management_failures_name_their_operation(
    behavior, operation,
):
    with pytest.raises(StrategyRuntimeError) as caught:
        replay_fixture(symbol_order=("SPY",), plays=one_play(**behavior))

    assert caught.value.code == "strategy_raised"
    assert caught.value.details["operation"] == operation
    assert caught.value.details["play_id"] == "play-a"


def test_bars_prepared_under_another_closure_are_refused():
    """The one seam the prepared bars cannot police themselves.

    They carry no record of the closure they were hydrated under, so a caller
    that prepares one and drives the loop with another would otherwise fail as a
    bare KeyError mid-replay, from outside the closed taxonomy.
    """
    with pytest.raises(ReplayInputError) as caught:
        replay_fixture(symbol_order=("SPY",), plays=one_play(),
                       drive_dependencies=base_dependencies())

    assert caught.value.code == "mismatched_dependencies"
    assert caught.value.details["timeframe"] == "1h"
