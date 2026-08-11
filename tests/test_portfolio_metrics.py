"""Every number a user reads, and the arithmetic that has to be right.

A wrong formula here is worse than a crash: it is a number that looks like a
measurement, so every golden below is derived from the architecture's own
expression and stated as a value a reader can check by hand, never copied out
of a run.

Three habits carry the file:

- HAND-COMPUTABLE INPUTS. The replay fixtures trade at flat prices with whole
  share counts, and the many-session fixtures move equity by doubling and
  halving. Every return, moment sum, and PnL is then exact under binary64, so a
  golden can be an equality rather than a tolerance and a tolerance means the
  test is comparing two DIFFERENT derivations on purpose.
- INDEPENDENT EXPECTATIONS. Where a value cannot be exact, the expected number
  is assembled from hand-derived moments in a different operation order than
  the implementation uses. `sqrt(126)` for a Sharpe and `sqrt(756)` for a
  Sortino are the same statistic reached another way, so a test cannot pass by
  restating the code.
- ULP RECONCILIATION. Parent and slice totals are compared through
  `_within_one_ulp` rather than a decimal approximation, because the identity
  the architecture states is a bitwise one under a fixed operation order and a
  loose comparison would let a real disagreement through.
"""

import dataclasses
import math
from datetime import date

import pandas as pd
import pytest

from nakagai.engine.metrics import (
    ARITHMETIC_VERSION,
    _portfolio_metrics,
    _slice_accumulators,
    _within_one_ulp,
)
from nakagai.engine.portfolio_types import (
    PortfolioMetrics,
    RejectionReason,
    ReplayInputError,
)
from tests.portfolio_fixtures import (
    BarPlan,
    ScriptedPlay,
    SignalPlan,
    base_execution,
    daily_curve,
    daily_metrics,
    daily_session_dates,
    daily_validated,
    replay_metrics,
    ts,
)

# The default window trades the 2026-11-27 half day: fourteen 15-minute
# intervals from 14:30Z to 18:00Z.
FIRST_OPEN = ts("2026-11-27T14:30:00Z")
STARTING_EQUITY = 100_000.0
WINDOW_SECONDS = 12_600.0


def opens(ordinal: int) -> pd.Timestamp:
    return FIRST_OPEN + pd.Timedelta(minutes=15 * ordinal)


def closes(ordinal: int) -> pd.Timestamp:
    return opens(ordinal) + pd.Timedelta(minutes=15)


# ---------------------------------------------------------- replay fixtures

# One replay carrying every trade shape the cohorts have to tell apart. Both
# plays signal both symbols at the first close and fill at the next open, at
# 100.0 with a 1.00 protective distance, which is 100 shares against a 100
# risk budget. What each position then meets is one edited bar:
#
#   play-a SPY long   target 103 reached in interval 3   +300, R  3.0
#   play-b SPY short  stop   101 reached in interval 3   -100, R -1.0
#   play-a QQQ long   stop    99 reached in interval 5   -100, R -1.0
#   play-b QQQ short  neither, closed at the window end     0, R  0.0
#
# The last is a break-even trade, which counts as a trade, is not a win, and
# belongs to neither gross sum. Three more signals never become trades: play-a
# signals SPY twice more while it holds SPY, and signals QQQ at the final
# close, where no open remains to fill it.
MIXED_BARS = (
    BarPlan(symbol="SPY", at=opens(3), open=100.0, high=103.0, low=100.0,
            close=100.0),
    BarPlan(symbol="QQQ", at=opens(5), open=100.0, high=100.0, low=99.0,
            close=100.0),
)


def mixed_plays() -> tuple[ScriptedPlay, ...]:
    return (
        ScriptedPlay(play_id="play-a", priority=100, signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),
            SignalPlan(symbol="SPY", at=closes(1), stop=99.0, target=103.0),
            SignalPlan(symbol="SPY", at=closes(2), stop=99.0, target=103.0),
            SignalPlan(symbol="QQQ", at=closes(0), stop=99.0, target=103.0),
            SignalPlan(symbol="QQQ", at=closes(13), stop=99.0, target=103.0),
        )),
        ScriptedPlay(play_id="play-b", priority=200, signals=(
            SignalPlan(symbol="SPY", at=closes(0), stop=101.0, target=97.0,
                       direction="short"),
            SignalPlan(symbol="QQQ", at=closes(0), stop=101.0, target=97.0,
                       direction="short"),
        )),
    )


def winning_plays() -> tuple[ScriptedPlay, ...]:
    """One long that reaches its target, so nothing ever lost."""
    return (ScriptedPlay(play_id="play-a", signals=(
        SignalPlan(symbol="SPY", at=closes(0), stop=99.0, target=103.0),)),)


def quiet_plays() -> tuple[ScriptedPlay, ...]:
    """One play that signals nothing, so the account never leaves cash."""
    return (ScriptedPlay(play_id="play-a"),)


def mixed_lenses():
    return replay_metrics(plays=mixed_plays(), bars=MIXED_BARS)


def totals(lenses, play_id: str, symbol: str):
    return lenses.slices[(play_id, symbol)]


def stats_fields(stats) -> tuple:
    return (
        stats.n_trades, stats.n_wins, stats.win_rate, stats.gross_profit,
        stats.gross_loss, stats.profit_factor, stats.profit_factor_state,
        stats.expectancy_r,
    )


# ------------------------------------------------------------- trade cohorts


def test_the_cohorts_split_one_replay_by_direction():
    """Every trade statistic, for all trades and for each direction.

    The whole cohort contract in one golden: a break-even trade counts as a
    trade and as neither a win nor a sum, gross profit and gross loss are sums
    of NET trade PnL rather than pre-cost PnL, and expectancy is the mean R
    over every trade including the break-even one.
    """
    metrics = mixed_lenses().metrics

    assert stats_fields(metrics.all_trades) == (
        4, 1, 0.25, 300.0, 200.0, 1.5, "finite", 0.25)
    assert stats_fields(metrics.long_trades) == (
        2, 1, 0.5, 300.0, 100.0, 3.0, "finite", 1.0)
    assert stats_fields(metrics.short_trades) == (
        2, 0, 0.0, 0.0, 100.0, 0.0, "finite", -0.5)


def test_a_cohort_that_never_lost_reports_an_infinite_state_and_no_ratio():
    """Strict JSON carries no infinity, so the state carries the meaning.

    The same replay proves the empty cohort beside it: no short traded, so its
    sums are zero, its state is unavailable, and its rate and expectancy are
    null rather than zero.
    """
    metrics = replay_metrics(plays=winning_plays(), bars=MIXED_BARS).metrics

    assert stats_fields(metrics.all_trades) == (
        1, 1, 1.0, 300.0, 0.0, None, "infinite", 3.0)
    assert stats_fields(metrics.short_trades) == (
        0, 0, None, 0.0, 0.0, None, "unavailable", None)


def test_undefined_ratios_are_null():
    """No trade at all: every ratio is null and every sum is zero."""
    metrics = replay_metrics(plays=quiet_plays()).metrics

    assert metrics.all_trades.profit_factor is None
    assert metrics.all_trades.profit_factor_state == "unavailable"
    assert metrics.all_trades.win_rate is None
    assert metrics.all_trades.expectancy_r is None
    assert metrics.all_trades.gross_profit == 0.0
    assert metrics.all_trades.gross_loss == 0.0
    assert metrics.exposure_pct == 0.0
    assert metrics.avg_holding_hours == 0.0


# ------------------------------------------------ parent and slice identities


def test_every_play_symbol_slice_reconciles_to_portfolio_totals():
    """The additive identity, on counts and on every money column.

    Slices are the attribution of the parent's own trades and refused signals,
    so a total that did not reconcile would mean one of the two was reading
    something the other could not see.
    """
    lenses = mixed_lenses()
    rows = tuple(lenses.slices.values())
    metrics = lenses.metrics

    assert len(rows) == len({(row.play_id, row.symbol) for row in rows})
    assert sum(row.trades for row in rows) == metrics.all_trades.n_trades
    assert sum(sum(row.rejection_counts.values()) for row in rows) == (
        metrics.n_rejections)
    assert sum(row.signals for row in rows) == 7
    assert _within_one_ulp(
        sum(row.net_pnl for row in rows), metrics.net_pnl)
    assert _within_one_ulp(
        sum(row.pre_cost_pnl for row in rows), metrics.pre_cost_pnl)
    assert _within_one_ulp(sum(row.fees for row in rows), metrics.fees)
    assert _within_one_ulp(
        sum(row.gross_profit for row in rows), metrics.all_trades.gross_profit)
    assert _within_one_ulp(
        sum(row.gross_loss for row in rows), metrics.all_trades.gross_loss)


def test_the_parent_pnl_identity_ties_the_trades_to_the_curve():
    """`net_pnl = pre_cost_pnl - fees`, and it lands the curve where it ended.

    The first half is a sum over trades and holds exactly. The second is a
    statement about the ACCOUNT, and it is asserted at the account's own scale:
    the ledger reached 100,070 through dozens of cash operations, so its last
    bit is a bit of 100,070 rather than of the 70 the trades add up to.
    Differencing the two first and comparing that would demand a precision the
    equity mark never carried.

    Run under the real slippage and fee models, because costs are what stop
    every intermediate from being a round number. The frictionless run beside
    it keeps the whole identity exact.
    """
    lenses = replay_metrics(plays=mixed_plays(), bars=MIXED_BARS,
                            execution=base_execution())
    metrics = lenses.metrics
    frictionless = mixed_lenses().metrics

    assert metrics.starting_equity == STARTING_EQUITY
    assert metrics.ending_equity == lenses.curve.equity[-1].portfolio_equity
    assert metrics.fees > 0.0
    assert metrics.net_pnl == metrics.pre_cost_pnl - metrics.fees
    assert _within_one_ulp(metrics.starting_equity + metrics.net_pnl,
                           metrics.ending_equity)
    assert (frictionless.pre_cost_pnl, frictionless.fees,
            frictionless.net_pnl) == (100.0, 0.0, 100.0)
    assert frictionless.ending_equity - frictionless.starting_equity == 100.0


def test_a_slice_exists_for_every_canonical_play_symbol_pair():
    """Including the pairs that never signalled, in canonical order.

    An absent key would make a reader guess what it meant, and a slice that
    appeared only where something happened would let a play with no signals
    disappear from its own attribution.
    """
    lenses = replay_metrics(plays=mixed_plays(), bars=MIXED_BARS)

    assert tuple(lenses.slices) == (
        ("play-a", "QQQ"), ("play-a", "SPY"),
        ("play-b", "QQQ"), ("play-b", "SPY"),
    )
    quiet = replay_metrics(plays=quiet_plays())
    assert tuple(quiet.slices) == (("play-a", "QQQ"), ("play-a", "SPY"))
    for row in quiet.slices.values():
        assert (row.signals, row.trades, row.net_pnl) == (0, 0, 0.0)
        assert row.win_rate is None and row.expectancy_r is None


def test_each_slice_carries_its_own_trades_and_refusals():
    """One golden per pair: what it signalled, traded, refused, and made."""
    lenses = mixed_lenses()

    long_spy = totals(lenses, "play-a", "SPY")
    assert (long_spy.strategy, long_spy.signals, long_spy.trades) == (
        "scripted-play-a", 3, 1)
    assert (long_spy.gross_profit, long_spy.gross_loss) == (300.0, 0.0)
    assert (long_spy.pre_cost_pnl, long_spy.fees, long_spy.net_pnl) == (
        300.0, 0.0, 300.0)
    assert (long_spy.win_rate, long_spy.expectancy_r) == (1.0, 3.0)
    assert dict(long_spy.rejection_counts) == {
        RejectionReason.POSITION_OCCUPIED: 2}

    long_qqq = totals(lenses, "play-a", "QQQ")
    assert (long_qqq.signals, long_qqq.trades, long_qqq.net_pnl) == (2, 1, -100.0)
    assert (long_qqq.win_rate, long_qqq.expectancy_r) == (0.0, -1.0)
    assert dict(long_qqq.rejection_counts) == {RejectionReason.WINDOW_ENDED: 1}

    short_qqq = totals(lenses, "play-b", "QQQ")
    assert (short_qqq.signals, short_qqq.trades) == (1, 1)
    assert (short_qqq.gross_profit, short_qqq.gross_loss) == (0.0, 0.0)
    assert (short_qqq.win_rate, short_qqq.expectancy_r) == (0.0, 0.0)
    assert dict(short_qqq.rejection_counts) == {}


def test_rejections_are_counted_at_the_parent_and_attributed_by_reason():
    lenses = mixed_lenses()

    assert lenses.metrics.n_rejections == 3
    assert [row.reason for row in lenses.events.rejections] == [
        RejectionReason.POSITION_OCCUPIED, RejectionReason.POSITION_OCCUPIED,
        RejectionReason.WINDOW_ENDED]


def test_within_one_ulp_admits_the_adjacent_bits_and_nothing_further():
    """The helper the reconciliation assertions ride on, pinned itself."""
    value = 100.1
    assert _within_one_ulp(value, value)
    assert _within_one_ulp(math.nextafter(value, math.inf), value)
    assert _within_one_ulp(math.nextafter(value, -math.inf), value)
    two_up = math.nextafter(math.nextafter(value, math.inf), math.inf)
    assert not _within_one_ulp(two_up, value)
    assert not _within_one_ulp(math.nan, value)
    assert _within_one_ulp(0.0, -0.0)


# --------------------------------------------------------- exposure and time


def test_exposure_counts_concurrent_holding_and_can_exceed_the_window():
    """Four positions over a 3.5 hour window hold for six hours in total.

    2,700 seconds each for the two SPY trades, 4,500 for the QQQ long, and
    11,700 for the QQQ short carried to the window end. The fraction is above
    one because the positions overlapped, which is exactly what a portfolio
    does and what a per-symbol account could never show.
    """
    metrics = mixed_lenses().metrics

    assert metrics.exposure_pct == 21_600.0 / WINDOW_SECONDS
    assert metrics.exposure_pct > 1.0
    assert metrics.avg_holding_hours == 1.5


# -------------------------------------------------------------- curve shapes


def test_the_curve_metrics_are_hand_calculable():
    """One doubling and one halving over two sessions, every point counted.

    The six points are 100,000, then 200,000 twice, then 100,000 three times.
    The peak is 200,000 from the second point on, so three points sit at a
    drawdown of exactly one half and three at zero. Max drawdown is 0.5 and the
    ulcer index is the root mean square of all six, `sqrt(3 * 0.25 / 6)`.
    """
    metrics = daily_metrics((1.0, -0.5))

    assert metrics.starting_equity == STARTING_EQUITY
    assert metrics.ending_equity == STARTING_EQUITY
    assert metrics.total_return == 0.0
    assert metrics.max_drawdown == 0.5
    assert metrics.ulcer_index == math.sqrt(0.125)
    assert metrics.cagr == 0.0
    assert metrics.calmar == 0.0


def test_a_curve_that_never_drew_down_has_no_calmar():
    """No denominator, so the field is null rather than zero or infinite."""
    metrics = daily_metrics((0.0, 0.0))

    assert metrics.max_drawdown == 0.0
    assert metrics.ulcer_index == 0.0
    assert metrics.calmar is None


def test_cagr_annualizes_the_exact_elapsed_seconds_of_the_window():
    """A window of exactly one year makes the exponent exactly one.

    365.25 days is 31,557,600 seconds, so a test range spanning exactly that
    reports a CAGR equal to its own total return, to the bit. A year measured
    as 365 or 360 days would move the exponent off one and the two fields
    would part company.
    """
    validated = daily_validated(263, last_intervals=24)
    window = validated.request.window

    metrics = daily_metrics((0.0,) * 261 + (1.0,), last_intervals=24)

    assert window.test_end - window.test_start == pd.Timedelta(
        days=365, hours=6)
    assert metrics.daily_n == 262
    assert metrics.total_return == 1.0
    assert metrics.cagr == 1.0


@pytest.mark.parametrize(
    "final_return, expected_return",
    [(-1.0, -1.0), (-1.5, -1.5)],
    ids=["wiped_out", "below_zero"],
)
def test_an_account_at_or_below_zero_reports_a_total_loss(final_return,
                                                          expected_return):
    """CAGR is -1 rather than a root of a nonpositive number."""
    metrics = daily_metrics((final_return,))

    assert metrics.total_return == expected_return
    assert metrics.cagr == -1.0
    assert metrics.max_drawdown == -expected_return
    assert metrics.calmar == -1.0 / -expected_return


def test_a_cagr_beyond_binary64_refuses_rather_than_reporting_an_infinity():
    """Doubling in half an hour annualizes past every representable float.

    The window is one 30-minute session, so the exponent is 17,532. There is no
    honest value to report and the contract admits no infinity, so the replay
    refuses inside the closed taxonomy instead of raising an OverflowError out
    of it.
    """
    with pytest.raises(ReplayInputError) as raised:
        daily_metrics((1.0,))

    assert raised.value.code == "nonfinite_binary64"
    assert raised.value.details["field"] == "cagr"


# ------------------------------------------------------------ daily sampling


def test_daily_sampling_takes_the_last_point_of_each_session():
    """The post-close point, not the last close, ends the final session.

    The two share `test_end`: one is taken before the window's forced
    liquidation and one after it. A sampler reading the earlier of them would
    report the session flat here instead of doubled.
    """
    metrics = daily_metrics((0.0, 0.0), final=200_000.0)

    assert metrics.daily_n == 2
    assert metrics.daily_sum == 1.0
    assert metrics.ending_equity == 200_000.0


def test_the_first_daily_return_is_measured_against_starting_equity():
    """One test session, so its only return is the window's total return."""
    lenses = mixed_lenses()
    metrics = lenses.metrics

    assert metrics.daily_n == 1
    assert metrics.daily_sum == metrics.total_return
    assert metrics.daily_sum_sq == metrics.total_return ** 2
    assert metrics.daily_sum_cube == metrics.total_return ** 3
    assert metrics.daily_sum_fourth == metrics.total_return ** 4
    assert metrics.daily_sum_sq_down == 0.0


def test_a_session_the_curve_never_marked_refuses():
    """Two statements of one chronology, so a disagreement is a refusal.

    A curve paired with a schedule it did not come from would otherwise sample
    whichever sessions happened to line up and report a shorter series than the
    window actually contains.
    """
    validated = daily_validated(3)
    curve = daily_curve(validated, (0.0, 0.0))
    missing = dataclasses.replace(
        curve, equity=curve.equity[:1] + curve.equity[3:])

    with pytest.raises(ReplayInputError) as raised:
        _portfolio_metrics(missing, (), (), validated)

    assert raised.value.code == "misaligned_marks"


def test_a_daily_return_with_no_denominator_refuses():
    """A session that ended at zero cannot divide the next one."""
    with pytest.raises(ReplayInputError) as raised:
        daily_metrics((-1.0, 0.0))

    assert raised.value.code == "nonfinite_binary64"
    assert raised.value.details["field"] == "daily_return"


# ------------------------------------------------------- pooled statistics

# Sixty daily returns of +1.0 and -0.5, alternating: the account doubles and
# halves and ends where it began. Every power sum below is exact under
# binary64 because both returns are dyadic:
#
#   n 60   sum 15    sum of squares 37.5   downside squares 7.5
#   cubes 26.25      fourths 31.875
#
# so mean is 0.25, population variance is 0.5625 and its root is exactly 0.75.
# Two equally weighted values are symmetric about their mean, which is why the
# skew of this series is exactly zero and its biased kurtosis is exactly one.
ALTERNATING = (1.0, -0.5) * 30

# Forty returns of +1.0 and twenty of -0.5, in a repeating triple so the
# drawdown stays one half. Mean is 0.5, variance is 0.5, and the third moment
# is -0.25, so this series has the skew the alternating one cannot.
SKEWED = (1.0, 1.0, -0.5) * 20

BIAS = math.sqrt(60 * 59) / 58
EXCESS = 59 / (58 * 57)


def test_the_six_poolable_sums_are_exact():
    metrics = daily_metrics(ALTERNATING)

    assert metrics.daily_n == 60
    assert metrics.daily_sum == 15.0
    assert metrics.daily_sum_sq == 37.5
    assert metrics.daily_sum_sq_down == 7.5
    assert metrics.daily_sum_cube == 26.25
    assert metrics.daily_sum_fourth == 31.875


def test_sharpe_and_sortino_annualize_the_pooled_moments():
    """Mean over deviation, annualized by the root of 252.

    Both expectations are reached another way: a mean of 0.5 over a population
    standard deviation of `sqrt(0.5)` is `sqrt(252 / 2)`, and the same mean over
    a downside deviation of `sqrt(1 / 12)` is `sqrt(252 * 3)`.
    """
    metrics = daily_metrics(SKEWED)

    assert metrics.daily_n == 60
    assert metrics.sharpe == pytest.approx(math.sqrt(126), rel=1e-15)
    assert metrics.sortino == pytest.approx(math.sqrt(756), rel=1e-15)


def test_skew_and_kurtosis_are_bias_corrected_and_non_excess():
    """A normal sample would score 3.0, and both carry the small-sample lift.

    The biased moments are exact here: `g1` is `-sqrt(2) / 2` and `g2` is 1.5.
    The reported values apply the architecture's corrections to those, so a
    reading that dropped either correction lands somewhere else entirely.
    """
    metrics = daily_metrics(SKEWED)

    assert metrics.skew == pytest.approx(-math.sqrt(2) / 2 * BIAS, rel=1e-15)
    assert metrics.kurtosis == pytest.approx(
        3.0 + EXCESS * (61 * (1.5 - 3.0) + 6), rel=1e-15)


def test_a_symmetric_series_has_no_skew_at_all():
    """Two equally weighted returns are symmetric, so the third moment is zero.

    Exactly zero, not nearly: the sums are dyadic and the central moment
    cancels bit for bit.
    """
    metrics = daily_metrics(ALTERNATING)

    assert metrics.skew == 0.0
    assert metrics.kurtosis == pytest.approx(
        3.0 + EXCESS * (61 * (1.0 - 3.0) + 6), rel=1e-15)


def test_psr_reads_the_higher_moments_of_the_same_pool():
    """The probability the true Sharpe clears zero, from hand-built moments.

    The expectation is assembled here from the series' own moments and the
    standard normal integral, in a different order than the implementation
    reaches it. A PSR built on the BIASED kurtosis would move the denominator
    and land off this number.
    """
    metrics = daily_metrics(ALTERNATING)
    kurtosis = 3.0 + EXCESS * (61 * (1.0 - 3.0) + 6)
    sharpe = 0.25 / 0.75
    z = sharpe * math.sqrt(59) / math.sqrt(
        1.0 + (kurtosis - 1.0) / 4.0 * sharpe ** 2)

    assert metrics.psr == pytest.approx(
        0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), rel=1e-12)
    assert 0.99 < metrics.psr < 1.0


def test_fifty_nine_daily_returns_report_no_pooled_statistic():
    """One observation short of the floor is null on all five fields."""
    below = daily_metrics(ALTERNATING[:59])
    at = daily_metrics(ALTERNATING)

    assert below.daily_n == 59
    assert below.daily_sum_sq == pytest.approx(37.25)
    assert (below.sharpe, below.sortino, below.psr, below.skew,
            below.kurtosis) == (None, None, None, None, None)
    assert at.daily_n == 60
    assert all(value is not None for value in (
        at.sharpe, at.sortino, at.psr, at.skew, at.kurtosis))


@pytest.mark.parametrize(
    "constant, sortino",
    [(-0.5, -math.sqrt(252)), (0.5, None)],
    ids=["all_down", "all_up"],
)
def test_a_series_with_no_variance_reports_no_sharpe(constant, sortino):
    """Zero variance is a division by zero, not an infinitely good result.

    Sortino survives the all-down case on purpose: its denominator is the
    downside deviation, which is 0.5 here, so the ratio is defined even where
    the Sharpe's is not.
    """
    metrics = daily_metrics((constant,) * 60)

    assert metrics.daily_n == 60
    assert metrics.sharpe is None
    assert (metrics.skew, metrics.kurtosis, metrics.psr) == (None, None, None)
    if sortino is None:
        assert metrics.sortino is None
    else:
        assert metrics.sortino == pytest.approx(sortino, rel=1e-15)


# A return large enough that its own fourth power crowds the top of binary64.
# Absurd as a market move and perfectly legal as an input: every equity point
# below is finite, so nothing upstream refuses them. Two such returns sum past
# the largest float, and one an order of magnitude larger cannot be raised to
# the fourth at all. The window is the exact year, so the CAGR is finite and
# the pooled sums are what the metrics reach next.
LARGE_RETURN = 1e77
HUGE_RETURN = 1e78


def test_a_moment_sum_beyond_binary64_refuses():
    """An infinite pooled sum is not a value a result can carry.

    The six sums travel on the result so that windows can be pooled, and the
    contract admits no infinity, so a fourth power that sums past the largest
    float refuses at the field that broke rather than travelling as an `inf`
    nothing downstream could serialize.
    """
    with pytest.raises(ReplayInputError) as raised:
        daily_metrics((LARGE_RETURN, LARGE_RETURN) + (0.0,) * 260,
                      last_intervals=24)

    assert raised.value.code == "nonfinite_binary64"
    assert raised.value.details["field"] == "daily_sum_fourth"


def test_a_power_that_cannot_be_taken_refuses_inside_the_taxonomy():
    """Raising to the fourth can RAISE rather than return an infinity.

    The same replay under a slightly larger return, and it has to fail the
    same way: an `OverflowError` reaching a caller is an untyped failure from
    outside the closed error contract.
    """
    with pytest.raises(ReplayInputError) as raised:
        daily_metrics((HUGE_RETURN,) + (0.0,) * 261, last_intervals=24)

    assert raised.value.code == "nonfinite_binary64"
    assert raised.value.details["field"] == "daily_moment"


# ---------------------------------------------------------- reading, not re-deriving


def test_the_benchmark_return_is_read_from_the_curve():
    """One benchmark number, decided by the curve and copied here.

    The fixture's benchmark series is flat while its reported total return is
    not, so a metric that recomputed the return from the equity points would
    report zero and a metric that reads the curve reports what the curve said.
    """
    validated = daily_validated(3)
    curve = daily_curve(validated, (0.0, 0.25), benchmark_return=0.4242)

    metrics = _portfolio_metrics(curve, (), (), validated)

    assert metrics.benchmark_return == 0.4242
    assert curve.equity[0].benchmark_equity == curve.equity[-1].benchmark_equity
    assert metrics.total_return == 0.25


def test_the_replay_metrics_read_the_curve_the_replay_produced():
    lenses = mixed_lenses()

    assert lenses.metrics.benchmark_return == lenses.curve.benchmark.total_return
    assert lenses.metrics.ending_equity == (
        lenses.curve.equity[-1].portfolio_equity)


# ------------------------------------------------------------- typed refusals


def test_metrics_computed_under_another_arithmetic_version_refuse():
    """These formulas are version 2 and say so rather than relabelling."""
    execution = dataclasses.replace(base_execution(), arithmetic_version="3")
    validated = daily_validated(2, execution=execution)
    curve = daily_curve(validated, (0.0,))

    with pytest.raises(ReplayInputError) as raised:
        _portfolio_metrics(curve, (), (), validated)

    assert raised.value.code == "unsupported_arithmetic_version"
    assert raised.value.details["expected"] == ARITHMETIC_VERSION


def test_a_curve_carrying_no_point_refuses():
    """The opening anchor is the denominator of nearly everything here."""
    validated = daily_validated(2)
    curve = dataclasses.replace(daily_curve(validated, (0.0,)), equity=())

    with pytest.raises(ReplayInputError) as raised:
        _portfolio_metrics(curve, (), (), validated)

    assert raised.value.details["field"] == "equity"


def test_a_curve_that_started_with_nothing_refuses():
    """Total return, every drawdown, and the CAGR all divide by this number.

    One guard at the value rather than three at the divisions, so a curve that
    could not produce a meaning refuses instead of reporting one.
    """
    validated = daily_validated(2)
    curve = daily_curve(validated, (0.0,))
    empty = dataclasses.replace(
        curve.equity[0], settled_cash=0.0, portfolio_equity=0.0)
    started_flat = dataclasses.replace(
        curve, equity=(empty, *curve.equity[1:]))

    with pytest.raises(ReplayInputError) as raised:
        _portfolio_metrics(started_flat, (), (), validated)

    assert raised.value.details["field"] == "starting_equity"


def test_signal_counts_that_miss_a_canonical_pair_refuse():
    """Every canonical play symbol reports its own signal count.

    A missing key would silently become a slice claiming no signal, which is
    the same shape as a play that genuinely never signalled.
    """
    lenses = mixed_lenses()
    counts = dict(lenses.events.signal_counts)
    del counts[("play-b", "QQQ")]
    events = dataclasses.replace(lenses.events, signal_counts=counts)

    with pytest.raises(ReplayInputError) as raised:
        _slice_accumulators(events, lenses.schedule)

    assert raised.value.code == "mismatched_signal_counts"


def test_a_trade_naming_an_unknown_pair_refuses():
    """An unattributable trade is refused, never dropped from the slices."""
    lenses = mixed_lenses()
    stray = dataclasses.replace(lenses.events.trades[0], play_id="ghost")
    events = dataclasses.replace(
        lenses.events, trades=(stray, *lenses.events.trades[1:]))

    with pytest.raises(ReplayInputError) as raised:
        _slice_accumulators(events, lenses.schedule)

    assert raised.value.code == "unknown_attribution_key"
    assert raised.value.details["play_id"] == "ghost"


def test_a_rejection_naming_an_unknown_pair_refuses():
    lenses = mixed_lenses()
    stray = dataclasses.replace(lenses.events.rejections[0], symbol="IWM")
    events = dataclasses.replace(lenses.events, rejections=(stray,))

    with pytest.raises(ReplayInputError) as raised:
        _slice_accumulators(events, lenses.schedule)

    assert raised.value.code == "unknown_attribution_key"
    assert raised.value.details["symbol"] == "IWM"


# ------------------------------------------------------------ binary64 output


def test_every_metric_is_a_plain_binary64_or_a_plain_count():
    """No numpy scalar, no bool, no integer where a float belongs.

    A numpy float survives arithmetic and comparison and then encodes as an
    unsupported canonical value, which is a failure a long way from its cause.
    """
    metrics = mixed_lenses().metrics
    counts = {"n_rejections", "daily_n"}

    for field in dataclasses.fields(PortfolioMetrics):
        value = getattr(metrics, field.name)
        if field.name.endswith("_trades"):
            continue
        if field.name in counts:
            assert type(value) is int, field.name
        else:
            assert value is None or type(value) is float, field.name


def test_no_metric_or_slice_value_is_a_negative_zero():
    """Negative zero is a different canonical encoding of the same number.

    `float.hex()` spells it `-0x0.0p+0`, so a result carrying one hashes
    differently from the identical result that carried a positive zero. The
    break-even trade in this fixture is where one would appear.
    """
    lenses = mixed_lenses()
    values = [getattr(lenses.metrics, field.name)
              for field in dataclasses.fields(PortfolioMetrics)]
    for row in lenses.slices.values():
        values.extend((row.gross_profit, row.gross_loss, row.pre_cost_pnl,
                       row.net_pnl, row.fees, row.win_rate, row.expectancy_r))

    zeros = [value for value in values
             if isinstance(value, float) and value == 0.0]
    assert zeros
    assert all(math.copysign(1.0, value) > 0.0 for value in zeros)


# ------------------------------------------------------------ fixture claims


def test_the_many_session_fixture_runs_real_weekday_sessions():
    """The synthetic schedule is data core reads, and it is well formed.

    Stated here rather than assumed: consecutive weekdays, two intervals each,
    and one test session for every daily return the metric tests ask for.
    """
    validated = daily_validated(4)

    assert daily_session_dates(4) == (
        date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8))
    assert len(validated.schedule.base_intervals) == 8
    assert len(validated.test_intervals) == 6
    assert validated.request.window.test_start == ts("2026-01-06T14:30:00Z")
    assert validated.request.window.test_end == ts("2026-01-08T15:00:00Z")
