"""The one settlement authority: cash, capacity, fills, exits, and marks.

Every assertion here is against a stated number rather than against a repeat
of the implementation's arithmetic. Where a value is exact under binary64 the
test states it exactly, and `pytest.approx` appears only where slippage and
per-share fees make an exact literal unreadable.

The frictionless fixtures are deliberate. A cash assertion carrying an
unrelated slippage term proves nothing about settlement, so cost behavior gets
its own ledger and its own tests further down.
"""

from datetime import date

import pytest

from nakagai.engine.canonical import rejection_id, trade_id
from nakagai.engine.portfolio import (
    EntryProposal,
    LedgerSnapshot,
    Reservation,
    _Ledger,
)
from nakagai.engine.portfolio_types import (
    ExitReason,
    RejectionReason,
    ReplayInputError,
)
from nakagai.engine.schedule import validate_schedule
from tests.portfolio_fixtures import (
    CLOSE_TS,
    ENTRY_TS,
    LAST_CLOSE_TS,
    NEXT_OPEN_TS,
    SESSION_ONE,
    SESSION_TWO,
    SIGNAL_TS,
    base_execution,
    base_schedule,
    entry_intent,
    funded_ledger,
    ledger_request,
    opened_position,
    replay_ambiguous_long,
)

HOLIDAY = date(2026, 11, 26)


# --------------------------------------------------------- one shared pool


def test_two_symbols_contend_for_one_cash_pool():
    ledger = funded_ledger(1000.0)
    first_proposal = EntryProposal(
        intent=entry_intent(ledger, "SPY"), raw_open=100.0, fill=100.0,
        quantity=7, entry_fee=0.0, required_cash=700.0,
    )
    second_proposal = EntryProposal(
        intent=entry_intent(ledger, "QQQ"), raw_open=100.0, fill=100.0,
        quantity=7, entry_fee=0.0, required_cash=700.0,
    )
    first = ledger.reserve(first_proposal, frozen_equity=1000.0)
    second = ledger.reserve(second_proposal, frozen_equity=1000.0)
    assert first.accepted is True
    assert second.reason == RejectionReason.UNSETTLED_CASH


def test_an_unsettled_cash_refusal_reports_the_two_cash_values():
    """Required is what the fill would cost; available is settled cash as it
    stood immediately before this proposal, not after it."""
    ledger = funded_ledger(1000.0)
    ledger.reserve(_proposal(ledger, "SPY", 1000.0), 1000.0)

    refusal = ledger.reserve(_proposal(ledger, "QQQ", 1000.0), 1000.0)

    assert refusal == Reservation(
        accepted=False, reason=RejectionReason.UNSETTLED_CASH,
        required_cash=700.0, available_cash=300.0,
    )


def test_a_refused_proposal_leaves_cash_and_capacity_untouched():
    """A candidate can fit the risk budget and still exceed settled cash,
    because equity counts positions and unsettled proceeds that cannot fund
    anything. The refusal has to cost the account nothing at all."""
    ledger = funded_ledger(500.0)

    refusal = ledger.reserve(_proposal(ledger, "SPY", 100_000.0), 100_000.0)

    assert refusal.accepted is False
    assert refusal.reason == RejectionReason.UNSETTLED_CASH
    assert ledger.settled_cash == 500.0
    assert ledger.open_positions == 0


def test_a_retry_after_a_refusal_reserves_exactly_what_the_first_try_asked():
    """A refusal consumes nothing, so the same candidate re-proposed against
    the same frozen equity is the same candidate, not a second charge."""
    ledger = funded_ledger(1000.0)
    intent = entry_intent(ledger, "SPY")
    first = ledger.propose(intent, 100.0, 1000.0)
    refused = ledger.reserve(
        ledger.propose(intent, 100.0, 100.0), 100.0,
    )

    second = ledger.propose(intent, 100.0, 1000.0)

    assert refused.reason == RejectionReason.ZERO_QUANTITY
    assert second == first
    assert ledger.reserve(second, 1000.0).accepted is True
    assert ledger.settled_cash == 300.0


# ------------------------------------------------------------------ sizing


def test_quantity_is_the_floored_risk_budget_over_protective_distance():
    """One percent of 100,000 is 1,000, and 1.4 of protective distance buys
    714 whole shares with change left over that never becomes a fraction."""
    ledger = funded_ledger(100_000.0)

    proposal = ledger.propose(entry_intent(ledger, "SPY"), 100.0, 100_000.0)

    assert proposal.quantity == 714
    assert proposal.required_cash == 71_400.0


def test_every_candidate_at_one_open_sizes_from_the_same_frozen_equity():
    """The second candidate is sized after the first one spent 71,400. Its
    quantity must not move, or fill order would change position size."""
    ledger = funded_ledger(100_000.0)
    first = ledger.propose(entry_intent(ledger, "SPY"), 100.0, 100_000.0)
    ledger.reserve(first, 100_000.0)
    ledger.open(first, ENTRY_TS)

    second = ledger.propose(entry_intent(ledger, "QQQ"), 100.0, 100_000.0)

    assert ledger.settled_cash == 28_600.0
    assert second.quantity == first.quantity == 714


def test_a_proposal_sized_against_another_equity_is_refused():
    """The ledger re-derives the candidate from the frozen equity it is given.
    A proposal built from post-fill equity is a different candidate and cannot
    reach the cash pool by being handed over with the earlier number."""
    ledger = funded_ledger(100_000.0)
    stale = ledger.propose(entry_intent(ledger, "SPY"), 100.0, 28_600.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.reserve(stale, 100_000.0)

    assert exc.value.code == "inconsistent_proposal"
    assert ledger.settled_cash == 100_000.0


def test_a_risk_budget_below_one_share_records_zero_quantity():
    ledger = funded_ledger(100.0)

    refusal = ledger.reserve(_proposal(ledger, "SPY", 100.0), 100.0)

    assert refusal.reason == RejectionReason.ZERO_QUANTITY
    assert refusal.required_cash is None
    assert ledger.settled_cash == 100.0


def test_a_negative_frozen_equity_cannot_size_a_position():
    ledger = funded_ledger(1000.0)

    proposal = ledger.propose(entry_intent(ledger, "SPY"), 100.0, -5_000.0)

    assert proposal.quantity == 0
    assert ledger.reserve(proposal, -5_000.0).reason == RejectionReason.ZERO_QUANTITY


# --------------------------------------------- the two entry geometry gates


def test_a_long_opening_on_its_stop_is_refused_though_slippage_lifts_the_fill():
    """The raw open printed before the entry existed, so a stop it already
    crossed could not have triggered. Buying slippage moves the modeled fill
    back above the stop, and the fill gate alone would let this through."""
    ledger = funded_ledger(100_000.0, execution=base_execution())
    intent = entry_intent(ledger, "SPY", stop=95.0, target=105.0)

    proposal = ledger.propose(intent, 95.0, 100_000.0)
    refusal = ledger.reserve(proposal, 100_000.0)

    assert proposal.fill == pytest.approx(95.019)
    assert refusal.reason == RejectionReason.INVALID_PROTECTIVE_GEOMETRY
    assert ledger.settled_cash == 100_000.0


def test_a_short_opening_on_its_stop_is_refused_though_slippage_drops_the_fill():
    """The mirror case, and the one the retired engine filled: selling
    slippage moves the fill down to 104.979, which sits inside a 95 to 105
    protective range, so only the raw open can refuse it."""
    ledger = funded_ledger(100_000.0, execution=base_execution())
    intent = entry_intent(
        ledger, "SPY", direction="short", stop=105.0, target=95.0,
    )

    proposal = ledger.propose(intent, 105.0, 100_000.0)
    refusal = ledger.reserve(proposal, 100_000.0)

    assert proposal.fill == pytest.approx(104.979)
    assert refusal.reason == RejectionReason.INVALID_PROTECTIVE_GEOMETRY
    assert ledger.settled_cash == 100_000.0


def test_a_fill_slipped_onto_its_target_is_refused_after_the_raw_open_passes():
    """The second gate, in the direction the first cannot see. The raw open at
    104.99 brackets correctly; the modeled fill at 105.01 does not."""
    ledger = funded_ledger(100_000.0, execution=base_execution())
    intent = entry_intent(ledger, "SPY", stop=95.0, target=105.0)

    proposal = ledger.propose(intent, 104.99, 100_000.0)
    refusal = ledger.reserve(proposal, 100_000.0)

    assert proposal.fill == pytest.approx(105.010998)
    assert refusal.reason == RejectionReason.INVALID_PROTECTIVE_GEOMETRY


def test_broken_geometry_is_named_before_capacity_or_cash_could_be():
    """Gate order is part of the contract: an account with no room and no cash
    still reports the geometry that was wrong with the candidate itself."""
    ledger = funded_ledger(100.0, max_open_positions=1)
    _fill_one(ledger, "QQQ", frozen_equity=100.0, entry_ref=10.0,
              raw_open=10.0, stop=9.86, target=10.4)
    intent = entry_intent(ledger, "SPY", stop=95.0, target=105.0)

    refusal = ledger.reserve(ledger.propose(intent, 95.0, 100.0), 100.0)

    assert ledger.open_positions == 1
    assert refusal.reason == RejectionReason.INVALID_PROTECTIVE_GEOMETRY


# ---------------------------------------------------------------- capacity


def test_the_account_refuses_a_fill_past_max_open_positions():
    ledger = funded_ledger(100_000.0, max_open_positions=2)
    _fill_one(ledger, "SPY", frozen_equity=1000.0)
    _fill_one(ledger, "QQQ", frozen_equity=1000.0)

    refusal = ledger.reserve(
        _proposal(ledger, "SPY", 1000.0, play_id="play-b",
                  strategy="donchian_break"),
        1000.0,
    )

    assert refusal.reason == RejectionReason.PORTFOLIO_CAPACITY
    assert refusal.required_cash is None
    assert ledger.open_positions == 2
    assert ledger.settled_cash == 98_600.0


def test_one_play_symbol_holds_one_position():
    ledger = funded_ledger(100_000.0)
    _fill_one(ledger, "SPY", frozen_equity=1000.0)

    refusal = ledger.reserve(_proposal(ledger, "SPY", 1000.0), 1000.0)

    assert refusal.reason == RejectionReason.POSITION_OCCUPIED
    assert ledger.open_positions == 1


def test_the_same_symbol_under_two_plays_is_two_positions():
    ledger = funded_ledger(100_000.0)
    _fill_one(ledger, "SPY", frozen_equity=1000.0)
    _fill_one(ledger, "SPY", frozen_equity=1000.0, play_id="play-b",
              strategy="donchian_break")

    assert ledger.position_keys() == (("play-a", "SPY"), ("play-b", "SPY"))


# -------------------------------------------------------- T+1 settlement


def test_exit_proceeds_settle_at_the_next_exchange_session():
    """2026-11-26 is Thanksgiving. A Wednesday sale settles on the Friday,
    because the settlement clock is the schedule's sessions rather than the
    next calendar day or the next weekday."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)
    ledger.close(key, 101.0, CLOSE_TS, ExitReason.MANAGE, 0.0)

    ledger.settle_due(SESSION_ONE)
    same_session = (ledger.settled_cash, ledger.unsettled_cash)
    ledger.settle_due(HOLIDAY)
    holiday = (ledger.settled_cash, ledger.unsettled_cash)
    ledger.settle_due(SESSION_TWO)

    assert same_session == (99_300.0, 707.0)
    assert holiday == (99_300.0, 707.0)
    assert (ledger.settled_cash, ledger.unsettled_cash) == (100_007.0, 0.0)


def test_exit_proceeds_cannot_fund_a_fill_before_they_settle():
    """Unsettled cash counts toward equity and not toward affordability. The
    account is worth 100,007 and can spend 99,300 of it."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=100_000.0)
    ledger.close(key, 101.0, CLOSE_TS, ExitReason.MANAGE, 0.0)
    ledger.settle_due(SESSION_ONE)

    refusal = ledger.reserve(
        _proposal(ledger, "QQQ", 100_000.0), 100_000.0,
    )

    assert ledger.settled_cash == 28_600.0
    assert ledger.unsettled_cash == 72_114.0
    assert refusal == Reservation(
        accepted=False, reason=RejectionReason.UNSETTLED_CASH,
        required_cash=71_400.0, available_cash=28_600.0,
    )


def test_a_credit_raised_on_the_last_session_never_settles():
    """The schedule is the calendar. There is no session after the final one,
    so proceeds from a closing trade stay unsettled for the whole replay
    rather than being guessed onto an invented date."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, entry_ts=NEXT_OPEN_TS,
        signal_ts=NEXT_OPEN_TS, eligible_after=NEXT_OPEN_TS,
    )
    ledger.close(key, 101.0, LAST_CLOSE_TS, ExitReason.END_OF_WINDOW, 0.0)

    ledger.settle_due(SESSION_TWO)

    assert ledger.unsettled_cash == 707.0
    assert ledger.settled_cash == 99_300.0


def test_settlement_takes_a_session_date_and_not_an_instant():
    """A settlement session is a calendar date. Passing the timestamp of the
    interval open would compare a moment against a date and settle by
    whichever timezone the caller happened to be holding."""
    ledger = funded_ledger(100_000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.settle_due(NEXT_OPEN_TS)

    assert exc.value.code == "invalid_type"


def test_an_exit_fill_refuses_a_direction_outside_the_contract():
    ledger = funded_ledger(100_000.0)

    with pytest.raises(ReplayInputError):
        ledger.exit_fill("flat", 100.0)


def test_an_exit_fee_refuses_anything_that_is_not_a_share_count():
    """The fee model takes the magnitude of what it is given, so a float or a
    negative count would price a plausible fee instead of refusing."""
    ledger = funded_ledger(100_000.0, execution=base_execution())

    with pytest.raises(ReplayInputError):
        ledger.exit_fee(-200)
    with pytest.raises(ReplayInputError):
        ledger.exit_fee(200.0)
    assert ledger.exit_fee(200) == pytest.approx(2.0)


def test_a_direction_outside_the_contract_never_becomes_a_position():
    """A signal validates at the strategy boundary and an intent checks only
    its symbol, so the entry path is where the ledger has to learn that a
    direction is one of the two it knows. Without this the short geometry
    branch passes, the short pricing branch sells to open, cash leaves, and
    the fill posts no collateral at all."""
    ledger = funded_ledger(100_000.0)
    intent = entry_intent(
        ledger, "SPY", direction="sell_short", stop=101.4, target=96.0,
    )

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(intent, 100.0, 100_000.0)

    assert exc.value.code == "invalid_value"
    assert ledger.settled_cash == 100_000.0
    assert ledger.open_positions == 0


def test_settling_the_same_session_twice_does_not_double_count():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)
    ledger.close(key, 101.0, CLOSE_TS, ExitReason.MANAGE, 0.0)

    ledger.settle_due(SESSION_TWO)
    ledger.settle_due(SESSION_TWO)

    assert ledger.settled_cash == 100_007.0
    assert ledger.unsettled_cash == 0.0


# --------------------------------------------------- gap and intrabar exits


def test_same_bar_stop_and_target_uses_pessimistic_exit():
    closed = replay_ambiguous_long(stop=95.0, target=105.0, low=94.0, high=106.0)
    assert closed.exit_reason == "stop"
    assert closed.exit == 95.0


def test_a_long_gapping_below_its_stop_fills_at_the_open():
    """The level was never available. The bar's first print is the first price
    this position could have traded at, so the open is the fill."""
    closed = replay_ambiguous_long(
        stop=95.0, target=105.0, low=90.0, high=96.0, raw_open=100.0,
        gap_open=94.0,
    )
    assert closed.exit_reason == ExitReason.STOP_GAP
    assert closed.exit == 94.0


def test_a_long_gapping_above_its_target_fills_at_the_open():
    closed = replay_ambiguous_long(
        stop=95.0, target=105.0, low=105.5, high=108.0, raw_open=100.0,
        gap_open=106.0,
    )
    assert closed.exit_reason == ExitReason.TARGET_GAP
    assert closed.exit == 106.0


def test_a_short_gapping_above_its_stop_fills_at_the_open():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    hit = ledger.protective_exit(key, 102.0, 103.0, 101.5)

    assert hit == (102.0, ExitReason.STOP_GAP)


def test_a_short_gapping_below_its_target_fills_at_the_open():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    hit = ledger.protective_exit(key, 95.0, 95.5, 94.0)

    assert hit == (95.0, ExitReason.TARGET_GAP)


def test_a_long_gapping_past_its_target_is_a_gap_even_when_the_stop_is_reachable():
    """The precedence that separates causality from pessimism. This bar opens
    above the target AND trades below the stop, so the pessimistic rule would
    take the stop. It cannot: the open is the first print of the bar and the
    position was gone at 106 before 94 was ever available."""
    closed = replay_ambiguous_long(
        stop=95.0, target=105.0, low=94.0, high=108.0, gap_open=106.0,
    )
    assert closed.exit_reason == ExitReason.TARGET_GAP
    assert closed.exit == 106.0


def test_a_long_gapping_past_its_stop_is_a_gap_even_when_the_target_is_reachable():
    """The mirror, where causality and pessimism agree on the stop but not on
    the price. Filling at 95 rather than 94 would model the gap as free."""
    closed = replay_ambiguous_long(
        stop=95.0, target=105.0, low=92.0, high=106.0, gap_open=94.0,
    )
    assert closed.exit_reason == ExitReason.STOP_GAP
    assert closed.exit == 94.0


def test_a_short_gapping_past_its_target_is_a_gap_even_when_the_stop_is_reachable():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    hit = ledger.protective_exit(key, 95.0, 102.0, 94.0)

    assert hit == (95.0, ExitReason.TARGET_GAP)


def test_a_short_gapping_past_its_stop_is_a_gap_even_when_the_target_is_reachable():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    hit = ledger.protective_exit(key, 102.0, 103.0, 95.0)

    assert hit == (102.0, ExitReason.STOP_GAP)


def test_an_intrabar_stop_fills_at_the_level_and_not_the_extreme():
    closed = replay_ambiguous_long(stop=95.0, target=110.0, low=90.0, high=101.0)
    assert closed.exit_reason == ExitReason.STOP
    assert closed.exit == 95.0


def test_an_intrabar_target_fills_at_the_level_and_not_the_extreme():
    closed = replay_ambiguous_long(stop=90.0, target=105.0, low=99.0, high=112.0)
    assert closed.exit_reason == ExitReason.TARGET
    assert closed.exit == 105.0


def test_a_bar_reaching_neither_level_leaves_the_position_open():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)

    assert ledger.protective_exit(key, 100.0, 100.5, 99.5) is None
    assert ledger.open_positions == 1


def test_a_position_filled_at_this_open_is_exited_by_this_same_bar():
    """Protection is active from the fill, so the entry bar's own low counts.
    A bar that opens inside the range and closes past the stop books a loss
    here rather than surviving to be marked at a price it could not hold."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)

    hit = ledger.protective_exit(key, 100.0, 100.2, 98.0)
    trade = ledger.close(key, hit[0], CLOSE_TS, hit[1], 0.0)

    assert hit == (98.6, ExitReason.STOP)
    assert trade.entry_ts == ENTRY_TS
    assert trade.exit_ts == CLOSE_TS
    assert trade.net_pnl == pytest.approx(-9.8)


def test_a_ratcheted_stop_becomes_the_level_that_exits():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)
    ledger.adjust(key, stop=99.5, target=None)

    hit = ledger.protective_exit(key, 100.0, 100.2, 99.0)
    trade = ledger.close(key, hit[0], CLOSE_TS, hit[1], 0.0)

    assert hit == (99.5, ExitReason.STOP)
    assert trade.initial_stop == 98.6
    assert trade.final_stop == 99.5
    assert trade.final_target == 104.0


# ------------------------------------------------------------- trade values


def test_a_long_trade_books_gross_pnl_fees_and_r_multiple():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)

    trade = ledger.close(key, 101.0, CLOSE_TS, ExitReason.TARGET, 0.0)

    assert (trade.qty, trade.entry, trade.exit) == (7, 100.0, 101.0)
    assert trade.gross_pnl == 7.0
    assert trade.fees == 0.0
    assert trade.net_pnl == 7.0
    assert trade.r_multiple == pytest.approx(0.7142857142857143)
    assert trade.trade_ordinal == 0
    assert trade.trade_id == trade_id(ledger.replay_id, "play-a", "SPY", 0)


def test_a_short_trade_books_gross_pnl_from_the_entry_side():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    trade = ledger.close(key, 99.0, CLOSE_TS, ExitReason.TARGET, 0.0)

    assert trade.direction == "short"
    assert trade.gross_pnl == 7.0
    assert trade.net_pnl == 7.0
    assert ledger.unsettled_cash == 707.0


def test_a_short_that_loses_credits_less_than_its_collateral():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, frozen_equity=1000.0, direction="short",
        stop=101.4, target=96.0,
    )

    trade = ledger.close(key, 101.0, CLOSE_TS, ExitReason.STOP, 0.0)

    assert trade.gross_pnl == -7.0
    assert ledger.unsettled_cash == 693.0


def test_trade_fees_are_one_entry_fill_plus_one_exit_fill():
    """A dollar per fill and half a cent a share, charged twice for a round
    trip because the model prices one fill at a time."""
    ledger = funded_ledger(100_000.0, execution=base_execution())
    key, proposal = opened_position(ledger, frozen_equity=100_000.0)
    exit_fee = ledger.exit_fee(proposal.quantity)

    trade = ledger.close(key, 101.0, CLOSE_TS, ExitReason.TARGET, exit_fee)

    assert proposal.quantity == 704
    assert proposal.entry_fee == pytest.approx(4.52)
    assert exit_fee == pytest.approx(4.52)
    assert trade.fees == pytest.approx(9.04)
    assert trade.net_pnl == pytest.approx(trade.gross_pnl - 9.04)


def test_slippage_moves_the_entry_and_the_exit_in_opposite_directions():
    ledger = funded_ledger(100_000.0, execution=base_execution())
    key, proposal = opened_position(ledger, frozen_equity=100_000.0)

    trade = ledger.close(
        key, ledger.exit_fill("long", 101.0), CLOSE_TS, ExitReason.TARGET, 0.0,
    )

    assert proposal.raw_open == 100.0
    assert proposal.fill == pytest.approx(100.02)
    assert trade.exit == pytest.approx(100.9798)
    assert trade.gross_pnl == pytest.approx(675.6992)


def test_a_gap_exit_credits_only_the_open_it_filled_at():
    """The one print between the fill and the exit is the open. Folding the
    bar's full range would credit a position with an excursion it was no
    longer open for, which is how a favorable extreme flatters a trade that
    had already gone."""
    closed = replay_ambiguous_long(
        stop=95.0, target=105.0, low=90.0, high=110.0, gap_open=94.0,
    )

    assert closed.exit_reason == ExitReason.STOP_GAP
    assert closed.mae == pytest.approx(1.2)
    assert closed.mfe == 0.0


def test_a_stop_exit_does_not_credit_the_favorable_extreme():
    """OHLC cannot prove the high came before the stop, and pessimistic
    ordering already assumed it did not."""
    closed = replay_ambiguous_long(
        stop=95.0, target=110.0, low=90.0, high=108.0,
    )

    assert closed.exit_reason == ExitReason.STOP
    assert closed.mae == 1.0
    assert closed.mfe == 0.0


def test_a_target_exit_credits_the_adverse_extreme_it_survived():
    closed = replay_ambiguous_long(
        stop=90.0, target=105.0, low=96.0, high=112.0,
    )

    assert closed.exit_reason == ExitReason.TARGET
    assert closed.mae == pytest.approx(0.4)
    assert closed.mfe == pytest.approx(0.5)


def test_excursion_is_reported_in_r_against_the_initial_stop():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)
    ledger.observe(key, 100.7, 99.3)

    trade = ledger.close(key, 100.0, CLOSE_TS, ExitReason.MANAGE, 0.0)

    assert trade.mae == pytest.approx(0.5)
    assert trade.mfe == pytest.approx(0.5)


# --------------------------------------------------------------- snapshots


def test_a_snapshot_reconciles_equity_to_cash_and_liquidation_value():
    ledger = funded_ledger(100_000.0)
    opened_position(ledger, frozen_equity=100_000.0)

    flat = ledger.snapshot(CLOSE_TS, {"SPY": 100.0})
    up = ledger.snapshot(CLOSE_TS, {"SPY": 101.0})

    assert flat == LedgerSnapshot(
        settled_cash=28_600.0, unsettled_cash=0.0, short_collateral=0.0,
        positions_liquidation_value=71_400.0, portfolio_equity=100_000.0,
        gross_exposure=71_400.0, open_positions=1,
    )
    assert up.portfolio_equity == 100_714.0
    assert up.gross_exposure == 72_114.0


def test_a_snapshot_that_does_not_reconcile_cannot_be_built():
    """The account identity belongs to the value, not to one call site.

    Every equity point copies these fields straight through, so a snapshot
    whose stated equity disagreed with its own parts would publish a curve that
    ties back to nothing. One cent is enough to refuse.
    """
    with pytest.raises(ReplayInputError) as exc:
        LedgerSnapshot(
            settled_cash=28_600.0, unsettled_cash=0.0, short_collateral=0.0,
            positions_liquidation_value=71_400.0, portfolio_equity=100_000.01,
            gross_exposure=71_400.0, open_positions=1,
        )

    assert exc.value.code == "unreconciled_equity"


def test_a_short_marks_at_its_collateral_plus_unrealized_pnl():
    ledger = funded_ledger(100_000.0)
    opened_position(
        ledger, frozen_equity=100_000.0, direction="short",
        stop=101.4, target=96.0,
    )

    mark = ledger.snapshot(CLOSE_TS, {"SPY": 99.0})

    assert mark.short_collateral == 71_400.0
    assert mark.positions_liquidation_value == 72_114.0
    assert mark.portfolio_equity == 100_714.0


def test_a_marked_position_pays_its_exit_costs_before_it_is_realized():
    """Hypothetical exit slippage and fees move the mark and stay out of the
    realized fee total until the position actually closes."""
    ledger = funded_ledger(100_000.0, execution=base_execution())
    _, proposal = opened_position(ledger, frozen_equity=100_000.0)

    mark = ledger.snapshot(CLOSE_TS, {"SPY": 100.0})

    assert mark.positions_liquidation_value == pytest.approx(70_381.4)
    assert mark.gross_exposure == 70_400.0
    assert proposal.quantity == 704


def test_equity_after_a_round_trip_is_starting_equity_plus_net_pnl():
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=100_000.0)
    trade = ledger.close(key, 101.0, CLOSE_TS, ExitReason.TARGET, 0.0)

    mark = ledger.snapshot(CLOSE_TS, {})

    assert trade.net_pnl == 714.0
    assert mark == LedgerSnapshot(
        settled_cash=28_600.0, unsettled_cash=72_114.0, short_collateral=0.0,
        positions_liquidation_value=0.0, portfolio_equity=100_714.0,
        gross_exposure=0.0, open_positions=0,
    )


def test_a_position_still_open_at_the_final_close_is_marked_not_dropped():
    ledger = funded_ledger(100_000.0)
    opened_position(ledger, frozen_equity=1000.0)
    _fill_one(ledger, "QQQ", frozen_equity=1000.0)

    mark = ledger.snapshot(LAST_CLOSE_TS, {"SPY": 103.0, "QQQ": 99.0})

    assert mark.open_positions == 2
    assert mark.positions_liquidation_value == 1414.0
    assert mark.portfolio_equity == 100_014.0


def test_a_snapshot_refuses_a_symbol_it_was_given_no_mark_for():
    ledger = funded_ledger(100_000.0)
    opened_position(ledger, frozen_equity=1000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.snapshot(CLOSE_TS, {"QQQ": 100.0})

    assert exc.value.code == "missing_mark"


def test_a_snapshot_refuses_while_a_credit_is_already_due():
    """Marking before settling would count matured cash twice, once as
    settled and once as unsettled. The ledger refuses instead of silently
    reporting the larger number."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)
    ledger.close(key, 101.0, CLOSE_TS, ExitReason.MANAGE, 0.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.snapshot(NEXT_OPEN_TS, {})

    assert exc.value.code == "settlement_not_applied"


def test_a_snapshot_refuses_while_a_reservation_is_still_open():
    ledger = funded_ledger(100_000.0)
    proposal = _proposal(ledger, "SPY", 1000.0)
    ledger.reserve(proposal, 1000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.snapshot(CLOSE_TS, {})

    assert exc.value.code == "reservation_outstanding"


# ----------------------------------------------------------------- ordering


def test_eligible_proposals_sort_by_priority_then_symbol_then_ordinal():
    """Play priority first, then uppercase symbol, then the stable signal
    ordinal. The ordinal case cannot arise from one play-symbol in a replay,
    but the order has to be total for the sort to be deterministic at all."""
    ledger = funded_ledger(100_000.0)
    late = _proposal(ledger, "SPY", 1000.0, signal_ordinal=9)
    early = _proposal(ledger, "SPY", 1000.0, signal_ordinal=2)
    other_symbol = _proposal(ledger, "QQQ", 1000.0, signal_ordinal=7)
    low_priority = _proposal(
        ledger, "QQQ", 1000.0, signal_ordinal=0, play_id="play-b",
        strategy="donchian_break",
    )
    supplied = (low_priority, late, other_symbol, early)

    ordered = ledger.funding_order(supplied)

    assert [(row.intent.play_id, row.intent.symbol, row.intent.signal_ordinal)
            for row in ordered] == [
        ("play-a", "QQQ", 7), ("play-a", "SPY", 2), ("play-a", "SPY", 9),
        ("play-b", "QQQ", 0),
    ]
    assert ledger.funding_order(tuple(reversed(supplied))) == ordered


def test_positions_are_visited_in_priority_symbol_and_entry_ordinal_order():
    ledger = funded_ledger(100_000.0)
    _fill_one(ledger, "SPY", frozen_equity=1000.0, play_id="play-b",
              strategy="donchian_break")
    _fill_one(ledger, "SPY", frozen_equity=1000.0)
    _fill_one(ledger, "QQQ", frozen_equity=1000.0)

    assert ledger.position_keys() == (
        ("play-a", "QQQ"), ("play-a", "SPY"), ("play-b", "SPY"),
    )


# --------------------------------------------------------------- rejections


def test_window_ended_rejects_a_pending_intent_at_the_final_close():
    ledger = funded_ledger(100_000.0)
    intent = entry_intent(ledger, "SPY")

    rejection = ledger.reject(
        intent, RejectionReason.WINDOW_ENDED, LAST_CLOSE_TS, None, None,
    )

    assert rejection.reason == RejectionReason.WINDOW_ENDED
    assert rejection.event_ts == LAST_CLOSE_TS
    assert rejection.signal_ts == SIGNAL_TS
    assert (rejection.required_cash, rejection.available_cash) == (None, None)
    assert rejection.open_positions == 0


def test_rejection_identity_and_ordinals_are_derived_not_counted_by_luck():
    ledger = funded_ledger(100_000.0)
    first = ledger.reject(
        entry_intent(ledger, "SPY"), RejectionReason.POSITION_OCCUPIED,
        SIGNAL_TS, None, None,
    )
    second = ledger.reject(
        entry_intent(ledger, "QQQ", signal_ordinal=4),
        RejectionReason.PORTFOLIO_CAPACITY, ENTRY_TS, None, None,
    )

    assert first.rejection_ordinal == 0
    assert second.rejection_ordinal == 1
    assert first.rejection_id == rejection_id(
        ledger.replay_id, "play-a", "SPY", 0, RejectionReason.POSITION_OCCUPIED,
    )
    assert second.rejection_id == rejection_id(
        ledger.replay_id, "play-a", "QQQ", 4, RejectionReason.PORTFOLIO_CAPACITY,
    )


def test_a_rejection_moves_neither_cash_nor_capacity():
    ledger = funded_ledger(100_000.0)
    _fill_one(ledger, "SPY", frozen_equity=1000.0)
    before = (ledger.settled_cash, ledger.unsettled_cash, ledger.open_positions)

    ledger.reject(
        entry_intent(ledger, "QQQ"), RejectionReason.PORTFOLIO_CAPACITY,
        ENTRY_TS, None, None,
    )

    assert (ledger.settled_cash, ledger.unsettled_cash,
            ledger.open_positions) == before


def test_only_an_unsettled_cash_rejection_may_carry_cash():
    ledger = funded_ledger(100_000.0)

    with pytest.raises(ReplayInputError):
        ledger.reject(
            entry_intent(ledger, "SPY"), RejectionReason.PORTFOLIO_CAPACITY,
            ENTRY_TS, 700.0, 300.0,
        )


# ---------------------------------------------------- refusals and binary64


def test_a_nonfinite_mark_never_reaches_an_equity_value():
    ledger = funded_ledger(100_000.0)
    opened_position(ledger, frozen_equity=1000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.snapshot(CLOSE_TS, {"SPY": float("nan")})

    assert exc.value.code == "nonfinite_binary64"


def test_a_nonfinite_frozen_equity_never_sizes_a_position():
    ledger = funded_ledger(100_000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(entry_intent(ledger, "SPY"), 100.0, float("inf"))

    assert exc.value.code == "nonfinite_binary64"


def test_a_boolean_is_not_a_price():
    ledger = funded_ledger(100_000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(entry_intent(ledger, "SPY"), True, 100_000.0)

    assert exc.value.code == "invalid_binary64"


def test_a_fill_without_a_reservation_is_refused():
    ledger = funded_ledger(100_000.0)
    proposal = _proposal(ledger, "SPY", 1000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.open(proposal, ENTRY_TS)

    assert exc.value.code == "unreserved_fill"
    assert ledger.open_positions == 0


def test_a_close_the_contract_refuses_leaves_the_position_open():
    """An exit that cannot become a trade must not settle anyway. Crediting
    cash and dropping the position for a record nobody got would lose the
    position silently and leave the account richer for it."""
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(ledger, frozen_equity=1000.0)

    with pytest.raises(ReplayInputError):
        ledger.close(key, 101.0, ENTRY_TS, ExitReason.MANAGE, 0.0)

    assert ledger.open_positions == 1
    assert (ledger.settled_cash, ledger.unsettled_cash) == (99_300.0, 0.0)
    assert ledger.close(key, 101.0, CLOSE_TS, ExitReason.MANAGE, 0.0).trade_ordinal == 0


def test_one_key_cannot_hold_two_reservations():
    ledger = funded_ledger(100_000.0)
    proposal = _proposal(ledger, "SPY", 1000.0)
    ledger.reserve(proposal, 1000.0)

    with pytest.raises(ReplayInputError) as exc:
        ledger.reserve(proposal, 1000.0)

    assert exc.value.code == "reservation_outstanding"
    assert ledger.settled_cash == 99_300.0


def test_an_intent_from_another_replay_is_refused():
    ledger = funded_ledger(100_000.0)
    stranger = entry_intent(funded_ledger(50_000.0), "SPY")

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(stranger, 100.0, 100_000.0)

    assert exc.value.code == "foreign_intent"


def test_an_intent_naming_an_unrequested_play_is_refused():
    ledger = funded_ledger(100_000.0)
    intent = entry_intent(ledger, "SPY", play_id="play-z")

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(intent, 100.0, 100_000.0)

    assert exc.value.code == "unknown_play"


def test_an_intent_naming_another_strategy_than_its_play_is_refused():
    ledger = funded_ledger(100_000.0)
    intent = entry_intent(ledger, "SPY", strategy="donchian_break")

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(intent, 100.0, 100_000.0)

    assert exc.value.code == "strategy_mismatch"


def test_an_intent_naming_an_untraded_symbol_is_refused():
    ledger = funded_ledger(100_000.0)
    intent = entry_intent(ledger, "IWM")

    with pytest.raises(ReplayInputError) as exc:
        ledger.propose(intent, 100.0, 100_000.0)

    assert exc.value.code == "unknown_symbol"


def test_a_ledger_refuses_a_schedule_validated_for_another_request():
    request = ledger_request(1000.0)
    other = ledger_request(2000.0)

    with pytest.raises(ReplayInputError):
        _Ledger(request, validate_schedule(other, base_schedule()))


# ------------------------------------------------------------------ helpers


def _proposal(ledger, symbol, frozen_equity, raw_open=100.0, **intent_fields):
    """One priced candidate, straight from the ledger's own pipeline."""
    return ledger.propose(
        entry_intent(ledger, symbol, **intent_fields), raw_open, frozen_equity,
    )


def _fill_one(ledger, symbol, *, frozen_equity, raw_open=100.0, **intent_fields):
    key, _ = opened_position(
        ledger, symbol, raw_open=raw_open, frozen_equity=frozen_equity,
        **intent_fields,
    )
    return key
