"""The two series a user reads: the account's equity curve and its benchmark.

Every number here is stated as an exact literal a reader can check by hand,
because these are the values every downstream metric and every product surface
draws from. The default bars are flat, so a replay that trades nothing holds
both series at starting equity to the bit, and a test that wants a price to
move puts one bar where it moves.

Two properties carry most of the weight:

- RECONCILIATION. `portfolio_equity` is settled cash plus unsettled credits
  plus what the open positions would liquidate for, at every point, with no
  tolerance. A curve that reconciles only to a rounding is a curve nobody can
  tie back to an account statement;
- ALIGNMENT. The benchmark is marked at the account's own instants, taken from
  the validated schedule rather than from a calendar, so the two series answer
  at the same boundaries and one cannot silently slip a bar against the other.

The benchmark goldens use dyadic prices (a symbol doubles, halves, or moves by
half) so the equal-weight average and its total return are exact under
binary64. That is deliberate: a golden that needed `approx` could not tell a
correct basket from one that normalized against the wrong bar.
"""

import dataclasses
from datetime import date

import pandas as pd
import pytest

from nakagai.engine.bars import PortfolioBars, prepare_portfolio_bars
from nakagai.engine.benchmark import _equity_series
from nakagai.engine.portfolio_types import (
    FeeSpec,
    ReplayInputError,
    ReplayWindow,
)
from nakagai.engine.schedule import validate_schedule
from tests.portfolio_fixtures import (
    BASE_ONLY,
    BarPlan,
    FactoryCalls,
    ScriptedPlay,
    SignalPlan,
    base_execution,
    base_request,
    base_schedule,
    canonical_curve_bytes,
    counting_definitions,
    frames_for,
    frictionless_execution,
    replay_account,
    replay_curve,
    replay_parts,
    single_symbol_benchmark,
    ts,
)

# The default window trades the 2026-11-27 half day: fourteen 15-minute
# intervals from 14:30Z to 18:00Z, which is also the schedule's last close.
FIRST_OPEN = ts("2026-11-27T14:30:00Z")
TEST_END = ts("2026-11-27T18:00:00Z")
# The opening anchor, fourteen closes, and the post-close point.
POINT_COUNT = 16
LAST_CLOSE_POINT = 14
FINAL_POINT = 15
STARTING_EQUITY = 100_000.0
HOLIDAY = date(2026, 11, 26)


def opens(ordinal: int) -> pd.Timestamp:
    """The open of test interval `ordinal` under the default window."""
    return FIRST_OPEN + pd.Timedelta(minutes=15 * ordinal)


def closes(ordinal: int) -> pd.Timestamp:
    return opens(ordinal) + pd.Timedelta(minutes=15)


def quiet_plays() -> tuple[ScriptedPlay, ...]:
    """One play that signals nothing, so the account never leaves cash."""
    return (ScriptedPlay(play_id="play-a"),)


def long_short_plays() -> tuple[ScriptedPlay, ...]:
    """One long on SPY and one short on QQQ, both filled at the second open.

    The protective levels are deliberately far away (a 1,000 target on the long
    and a 1.0 target on the short), so the closing move below cannot exit
    either position and both are still open at the final close.
    """
    return (
        ScriptedPlay(play_id="play-a", signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=1_000.0),)),
        ScriptedPlay(play_id="play-b", signals=(
            SignalPlan(symbol="QQQ", at=closes(0), stop=101.0, target=1.0,
                       direction="short"),)),
    )


# SPY closes the window at 150 and QQQ at 50. The long gains 5,000, the short
# gains 5,000, and the equal-weight basket of the two ends exactly where it
# started, which is what makes the pair worth marking together.
CLOSING_MOVE = (
    BarPlan(symbol="SPY", at=opens(13), open=100.0, high=150.0, low=100.0,
            close=150.0),
    BarPlan(symbol="QQQ", at=opens(13), open=100.0, high=100.0, low=50.0,
            close=50.0),
)


def fee_only_execution():
    """A dollar a fill, no slippage, so a cost shows up as a whole number."""
    return dataclasses.replace(
        frictionless_execution(), fees=FeeSpec(per_fill=1.0, per_share=0.0),
    )


def account_fields(point) -> tuple:
    """Every account value on one point, and nothing from the benchmark."""
    return (
        point.ts, point.point_ordinal, point.settled_cash, point.unsettled_cash,
        point.short_collateral, point.positions_liquidation_value,
        point.portfolio_equity, point.gross_exposure, point.open_positions,
    )


# --------------------------------------------------------- the frozen instants


def test_the_series_is_the_anchor_then_every_close_then_the_post_close():
    """Sixteen points for a fourteen-interval half day.

    The first is the opening anchor at `test_start`, which is not a close mark,
    and the last two share `test_end`: one before the window's liquidation and
    one after it. An engine that marked only closes would produce fourteen.
    """
    parts = replay_parts(plays=quiet_plays())

    curve = _equity_series(parts.schedule, parts.events, parts.prepared)

    assert len(curve.equity) == POINT_COUNT
    assert tuple(point.point_ordinal for point in curve.equity) == tuple(
        range(POINT_COUNT))
    assert curve.equity[0].ts == FIRST_OPEN
    assert tuple(point.ts for point in curve.equity[1:FINAL_POINT]) == tuple(
        closes(ordinal) for ordinal in range(14))
    assert curve.equity[LAST_CLOSE_POINT].ts == TEST_END
    assert curve.equity[FINAL_POINT].ts == TEST_END
    assert {point.replay_id for point in curve.equity} == {
        parts.request.replay_id}


def test_the_curve_clock_is_the_schedule_and_never_a_calendar():
    """A window spanning the Thanksgiving holiday marks no point inside it.

    Thirty scheduled intervals, sixteen on 2026-11-25 and fourteen on the
    2026-11-27 half day, with 2026-11-26 simply absent. A curve that stepped a
    calendar day, or assumed a 6.5 hour session, would put points where this
    exchange never traded.
    """
    window = ReplayWindow(
        train_start=ts("2026-11-25T14:30:00Z"),
        train_end=ts("2026-11-25T17:00:00Z"),
        test_start=ts("2026-11-25T17:00:00Z"),
        test_end=TEST_END,
    )

    curve = replay_curve(plays=quiet_plays(), window=window)

    assert len(curve.equity) == 32
    assert curve.equity[0].ts == ts("2026-11-25T17:00:00Z")
    assert curve.equity[16].ts == ts("2026-11-25T21:00:00Z")
    assert curve.equity[17].ts == ts("2026-11-27T14:45:00Z")
    assert curve.equity[-1].ts == TEST_END
    assert not [point for point in curve.equity if point.ts.date() == HOLIDAY]


# ----------------------------------------------------------- reconciliation


@pytest.mark.parametrize(
    "execution", [None, base_execution()], ids=["frictionless", "with_costs"],
)
def test_every_point_reconciles_to_cash_plus_marked_positions(execution):
    """The account identity, exactly, at every point including the boundaries.

    Run twice: once frictionless and once through the real slippage and fee
    models, because costs are what make the liquidation value stop being a
    round number. Neither run gets a tolerance.
    """
    curve = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE,
                         execution=execution)

    assert len(curve.equity) == POINT_COUNT
    for point in curve.equity:
        assert point.portfolio_equity == (
            point.settled_cash + point.unsettled_cash
            + point.positions_liquidation_value
        )
    assert curve.equity[LAST_CLOSE_POINT].open_positions == 2


def test_a_quiet_replay_holds_both_series_at_starting_equity_exactly():
    """No signal, no trade, no deposit: nothing moves either series.

    The account has no door that adds cash, so a replay that never trades ends
    on the number it started with. The prices here are chosen to make that a
    real claim about the benchmark too: at 30.0 a share and a 1,000 account, a
    shadow basket that bought fractional units and re-multiplied them would
    report 1000.0000000000001 at its own anchor. Normalizing each symbol
    against its own first open reports 1,000 exactly, for any symbol count and
    any starting equity.
    """
    parts = replay_parts(plays=quiet_plays(), price=30.0,
                         account=replay_account(1_000.0))

    curve = _equity_series(parts.schedule, parts.events, parts.prepared)

    assert parts.events.trades == ()
    assert {point.portfolio_equity for point in curve.equity} == {1_000.0}
    assert {point.benchmark_equity for point in curve.equity} == {1_000.0}
    assert {point.settled_cash for point in curve.equity} == {1_000.0}
    assert curve.benchmark.total_return == 0.0


def test_a_long_and_a_short_mark_against_the_close_that_moved():
    """One golden point, every field derived by hand.

    Both positions fill at 100.0 for 100 shares: the long spends 10,000 and the
    short reserves 10,000 as collateral, leaving 80,000 settled. At the final
    close SPY prints 150 and QQQ prints 50, so the long liquidates for 15,000
    and the short for its 10,000 collateral plus 5,000 of unrealized gain.
    Gross exposure marks both sides at the raw close: 15,000 plus 5,000.
    """
    curve = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)

    point = curve.equity[LAST_CLOSE_POINT]
    assert account_fields(point) == (
        TEST_END, LAST_CLOSE_POINT, 80_000.0, 0.0, 10_000.0, 30_000.0,
        110_000.0, 20_000.0, 2,
    )


def test_the_final_liquidation_moves_value_into_unsettled_cash():
    """The last two points bracket the window's forced exit.

    They share `test_end`. Before it, two open positions worth 30,000; after
    it, no position and 30,000 of T+1 credits that this schedule has no session
    left to settle. Equity is 110,000 across both, which is starting equity
    plus the two trades' net PnL and nothing else.
    """
    parts = replay_parts(plays=long_short_plays(), bars=CLOSING_MOVE)

    curve = _equity_series(parts.schedule, parts.events, parts.prepared)

    final = curve.equity[FINAL_POINT]
    assert account_fields(final) == (
        TEST_END, FINAL_POINT, 80_000.0, 30_000.0, 0.0, 0.0, 110_000.0, 0.0, 0,
    )
    assert sum(trade.net_pnl for trade in parts.events.trades) == 10_000.0
    assert final.portfolio_equity - STARTING_EQUITY == 10_000.0


def test_a_mark_carries_the_exit_costs_a_position_has_not_paid_yet():
    """A dollar per fill, charged twice in the mark and once in the cash.

    The entry cost 10,000 of notional plus its own dollar, so settled cash is
    89,999. The mark values the position at what it would raise if it closed
    now, which is 10,000 less the exit fee it has not paid. Marked equity is
    therefore 99,998: the round trip's whole cost, visible before the trade
    realizes any of it.
    """
    plays = (ScriptedPlay(play_id="play-a", signals=(
        SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=1_000.0),)),)

    curve = replay_curve(plays=plays, execution=fee_only_execution())

    point = curve.equity[LAST_CLOSE_POINT]
    assert account_fields(point) == (
        TEST_END, LAST_CLOSE_POINT, 89_999.0, 0.0, 0.0, 9_999.0, 99_998.0,
        10_000.0, 1,
    )


# ------------------------------------------------------------- the benchmark


def test_the_anchor_values_the_basket_at_the_first_test_opens():
    """Point zero carries starting equity on both sides.

    The shadow account initializes from the first raw opens and is valued at
    those same opens, so the anchor is the denominator of the first session's
    return rather than a mark of anything.
    """
    curve = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)

    assert curve.equity[0].ts == FIRST_OPEN
    assert curve.equity[0].benchmark_equity == STARTING_EQUITY
    assert curve.equity[0].portfolio_equity == STARTING_EQUITY


def test_the_basket_is_anchored_at_the_first_test_open_and_not_the_warm_up():
    """The warm-up range is not part of the opportunity set.

    The schedule carries the whole 2026-11-25 session before this window opens,
    and the prepared frames carry its bars. Here SPY warms up at 50 and trades
    the entire test window at 100, so a basket anchored at the schedule's first
    bar would report the account's benchmark up 50% before the first signal was
    even evaluated.
    """
    warm_up = (BarPlan(symbol="SPY", at=ts("2026-11-25T14:30:00Z"), open=50.0,
                       high=50.0, low=50.0, close=50.0),)

    curve = replay_curve(plays=quiet_plays(), bars=warm_up)

    assert {point.benchmark_equity for point in curve.equity} == {STARTING_EQUITY}
    assert curve.benchmark.total_return == 0.0


def test_the_equal_weight_basket_averages_the_symbols_it_trades():
    """Half the money in each symbol, held: SPY doubles the money it holds,
    QQQ halves it, and the basket ends exactly where it began.

    The account made 10,000 over the same window, so a benchmark reading any
    part of the account, or averaging one symbol twice, would not land on
    100,000 here.
    """
    curve = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)

    assert curve.equity[LAST_CLOSE_POINT].benchmark_equity == 100_000.0
    assert curve.equity[FINAL_POINT].benchmark_equity == 100_000.0
    assert curve.equity[LAST_CLOSE_POINT].portfolio_equity == 110_000.0
    assert curve.benchmark.total_return == 0.0
    assert curve.benchmark.spec.kind == "equal_weight_request_symbols"


def test_the_basket_reports_the_return_of_the_opportunity_set():
    """One symbol up half, one flat: the basket is up a quarter.

    (1.5 + 1.0) / 2 is exactly 1.25 under binary64, so 125,000 and 0.25 are
    exact literals rather than approximations.
    """
    move = (BarPlan(symbol="SPY", at=opens(13), open=100.0, high=150.0,
                    low=100.0, close=150.0),)

    curve = replay_curve(plays=quiet_plays(), bars=move)

    assert curve.equity[FINAL_POINT].benchmark_equity == 125_000.0
    assert curve.benchmark.total_return == 0.25


def test_the_basket_normalizes_by_the_first_open_and_never_rebalances():
    """Units bought once and held, so a later move prices the ORIGINAL units.

    SPY doubles on the first interval and returns to 100; QQQ then doubles on
    the last. A basket holding its first-open units is up half at both ends and
    flat in between. One that renormalized against the previous close, or
    rebalanced to equal weights along the way, would report 187,500 at the end
    instead of 150,000.
    """
    move = (
        BarPlan(symbol="SPY", at=opens(0), open=100.0, high=200.0, low=100.0,
                close=200.0),
        BarPlan(symbol="QQQ", at=opens(13), open=100.0, high=200.0, low=100.0,
                close=200.0),
    )

    curve = replay_curve(plays=quiet_plays(), bars=move)

    assert curve.equity[0].benchmark_equity == STARTING_EQUITY
    assert curve.equity[1].benchmark_equity == 150_000.0
    assert curve.equity[2].benchmark_equity == 100_000.0
    assert curve.equity[LAST_CLOSE_POINT].benchmark_equity == 150_000.0
    assert curve.benchmark.total_return == 0.5


def test_an_explicit_single_symbol_benchmark_tracks_only_that_symbol():
    """A declared SPY-style benchmark on a symbol no play trades.

    IWM doubles while the traded symbols go their own ways, and the benchmark
    follows IWM alone. It never becomes a traded symbol: no trade, no position,
    and no request symbol names it.
    """
    move = (*CLOSING_MOVE, BarPlan(symbol="IWM", at=opens(13), open=100.0,
                                   high=200.0, low=100.0, close=200.0))

    parts = replay_parts(plays=long_short_plays(), bars=move,
                         benchmark=single_symbol_benchmark("IWM"))
    curve = _equity_series(parts.schedule, parts.events, parts.prepared)

    assert curve.equity[FINAL_POINT].benchmark_equity == 200_000.0
    assert curve.benchmark.total_return == 1.0
    assert curve.benchmark.spec.symbol == "IWM"
    assert parts.request.symbols == ("QQQ", "SPY")
    assert {trade.symbol for trade in parts.events.trades} == {"QQQ", "SPY"}


def test_a_single_symbol_benchmark_changes_no_account_value():
    """The shadow account has no cash, no capacity, and no fills.

    Every account field of every point is identical to the same replay under
    the default basket, so declaring a benchmark cannot move a number a user
    would read as their own money. The basket run carries no IWM frame at all,
    which is the other half of the claim: an undeclared symbol is not hydrated.
    """
    move = (*CLOSING_MOVE, BarPlan(symbol="IWM", at=opens(13), open=100.0,
                                   high=200.0, low=100.0, close=200.0))

    basket = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)
    single = replay_curve(plays=long_short_plays(), bars=move,
                          benchmark=single_symbol_benchmark("IWM"))

    assert [account_fields(point) for point in single.equity] == [
        account_fields(point) for point in basket.equity]
    assert single.equity[FINAL_POINT].benchmark_equity == 200_000.0
    assert basket.equity[FINAL_POINT].benchmark_equity == 100_000.0


# ------------------------------------------------- refusals before a factory


def test_a_benchmark_the_bars_never_carried_refuses_before_a_factory_runs():
    """A declared benchmark symbol is a manifest dependency, never an option.

    Preflight refuses the whole replay, and it refuses before a strategy is
    constructed, so no play has run when the operator hears about it.
    """
    calls = FactoryCalls()

    with pytest.raises(ReplayInputError) as raised:
        replay_curve(plays=quiet_plays(),
                     benchmark=single_symbol_benchmark("IWM"),
                     drop_pairs=(("IWM", "15m"),),
                     wrap=counting_definitions(calls))

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["symbol"] == "IWM"
    assert calls.factory_count == 0


def test_a_benchmark_frame_missing_one_scheduled_bar_refuses():
    """Incomplete is refused exactly like absent: no forward fill, no gap.

    A benchmark with one bar missing would otherwise be marked against the
    wrong interval for the whole rest of the window.
    """
    request = base_request(benchmark=single_symbol_benchmark("IWM"))
    schedule = base_schedule()
    frames = frames_for(request, schedule, BASE_ONLY)
    frame = frames[("IWM", "15m")]
    frames[("IWM", "15m")] = frame.drop(index=frame.index[5])

    with pytest.raises(ReplayInputError) as raised:
        prepare_portfolio_bars(
            request, PortfolioBars(frames),
            validate_schedule(request, schedule), BASE_ONLY,
        )

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["field"] == "labels"


def test_a_curve_computed_over_another_requests_bars_refuses():
    """The bars do not carry the request they were prepared under.

    Pairing one replay's curve with another replay's bars is the one miswiring
    nothing upstream can see, and it would otherwise surface as a bare
    `KeyError` from outside the closed replay taxonomy.
    """
    parts = replay_parts(plays=quiet_plays())
    other = validate_schedule(
        base_request(benchmark=single_symbol_benchmark("IWM")), base_schedule())

    with pytest.raises(ReplayInputError) as raised:
        _equity_series(other, parts.events, parts.prepared)

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["symbol"] == "IWM"


def test_a_benchmark_price_that_is_not_positive_refuses():
    """A zero print is a refusal rather than a division.

    The benchmark reads its prices through the same raw bar reader the account
    fills against, which is where a nonpositive print is refused. A benchmark
    symbol no play trades never reaches that reader through the replay loop, so
    this is the one path on which a zero print could otherwise become a
    division by zero.
    """
    zero = (BarPlan(symbol="IWM", at=opens(0), open=0.0, high=0.0, low=0.0,
                    close=0.0),)

    with pytest.raises(ReplayInputError) as raised:
        replay_curve(plays=quiet_plays(), bars=zero,
                     benchmark=single_symbol_benchmark("IWM"))

    assert raised.value.code == "invalid_value"


# ------------------------------------------------- alignment and determinism


def test_a_mark_series_that_misses_a_scheduled_point_refuses():
    """The benchmark's instants come from the schedule, not from the marks.

    Two statements of one chronology have to agree, so a replay that dropped
    the post-close mark is refused rather than silently marked one point short
    with the benchmark values shifted onto the wrong instants.
    """
    parts = replay_parts(plays=quiet_plays())
    short = dataclasses.replace(parts.events, marks=parts.events.marks[:-1])

    with pytest.raises(ReplayInputError) as raised:
        _equity_series(parts.schedule, short, parts.prepared)

    assert raised.value.code == "misaligned_marks"


def test_a_mark_taken_at_an_unscheduled_instant_refuses():
    parts = replay_parts(plays=quiet_plays())
    moved = dataclasses.replace(parts.events, marks=(
        dataclasses.replace(parts.events.marks[0], ts=closes(0)),
        *parts.events.marks[1:],
    ))

    with pytest.raises(ReplayInputError) as raised:
        _equity_series(parts.schedule, moved, parts.prepared)

    assert raised.value.code == "misaligned_marks"
    assert raised.value.details["expected"] == FIRST_OPEN.isoformat()


def test_the_order_the_symbols_are_supplied_in_cannot_move_the_curve():
    """The basket sums in the request's canonical symbol order.

    Floating point addition does not commute, so a basket that summed in
    supplied order would produce two different curves for one candidate.
    """
    forward = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE,
                           symbol_order=("SPY", "QQQ"))
    reversed_order = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE,
                                  symbol_order=("QQQ", "SPY"))

    assert canonical_curve_bytes(forward) == canonical_curve_bytes(
        reversed_order)


def test_repeating_a_replay_repeats_the_curve_byte_for_byte():
    first = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)
    second = replay_curve(plays=long_short_plays(), bars=CLOSING_MOVE)

    assert canonical_curve_bytes(first) == canonical_curve_bytes(second)
    assert first.equity == second.equity
    assert first.benchmark == second.benchmark
