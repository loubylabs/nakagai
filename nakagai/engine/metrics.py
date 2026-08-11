"""One metric authority: every number a replay reports about itself.

These are the values a user acts on, so the rules here are stricter than the
arithmetic alone would need.

- ONE DERIVATION PER VALUE. Anything the ledger or the curve already decided is
  read rather than recomputed. Starting and ending equity are the curve's own
  first and last marks, and `benchmark_return` is the number the benchmark lens
  already published. A second derivation of either would be a second authority
  that can disagree with the first, and the disagreement would surface as a
  metric nobody can reconcile rather than as a refusal.
- ONE FOLD, TWO READERS. The parent cohorts and the play-symbol accumulators
  fold the same trades through the same `_Cohort`, in trade ordinal order, so
  neither can come to hold a term the other does not. What that buys is exact
  reconciliation on every SUM: pre-cost PnL, fees, the two gross sums, and the
  counts add the same values in the same order at both levels.
  `net_pnl` is the one column where it does not, and the reason is structural
  rather than associative. A slice reports `p_k - f_k` and the parent reports
  `(sum p) - (sum f)`, which are different expressions of the same quantity, so
  a reader summing the slices can land a few ULPs from the parent and the gap
  grows with the number of slices. The alternative would be to define a slice's
  net PnL as something a user cannot check against that slice's own two
  columns, which is a worse trade: the per-row identity is the one a person
  actually verifies by hand.
- NULL, NEVER A NONFINITE. An undefined ratio is `None`. Strict JSON and
  Postgres carry no infinity and no NaN, so a profit factor with nothing lost
  reports its STATE instead of a value, and a statistic whose denominator
  vanished reports nothing at all. Zero is a measurement and null is the
  absence of one; the two never stand in for each other.
- NEVER A NEGATIVE ZERO. `float.hex()` spells it `-0x0.0p+0`, so a result
  carrying one hashes differently from the identical result that carried a
  positive zero. Losses accumulate as positive magnitudes and no sum is
  negated at the end, which is what keeps the sign of every zero here positive.

The pooled higher moments come from `nakagai.stats.pooled_moments` and
`probabilistic_sharpe_ratio`, which own the bias corrections and the PSR
formula for this repository and for the platform. Re-deriving them here would
be the second copy that module's own docstring warns about, and the copies
would drift in exactly one of the two places.

Nothing here rounds. The approved IC correlation is the one value in a replay
that rounds to four decimals, and it belongs to the IC lens.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType

from pandas import Timestamp

from nakagai.engine.benchmark import ReplayCurve
from nakagai.engine.execution import ReplayEvents
from nakagai.engine.portfolio import PositionKey
from nakagai.engine.portfolio_types import (
    POOLED_MIN_DAILY,
    EquityPoint,
    PortfolioMetrics,
    PortfolioReplayRequest,
    PortfolioTrade,
    RejectionReason,
    ReplayRejection,
    TradeStats,
    _fail,
    _require_binary64,
    _require_instance,
    _require_items,
    _require_positive,
)
from nakagai.engine.schedule import ValidatedSchedule
from nakagai.stats import pooled_moments, probabilistic_sharpe_ratio

# The chronology, the cost model, and these formulas are one arithmetic. A
# request naming another version is refused rather than reported under a label
# that does not describe how its numbers were reached.
ARITHMETIC_VERSION = "2"

# Trading days in a year, for annualizing a daily statistic, and the seconds in
# a Julian year, for annualizing a window. They answer different questions and
# are deliberately not the same constant.
_TRADING_DAYS = 252
_ANNUALIZE = math.sqrt(_TRADING_DAYS)
_SECONDS_PER_YEAR = 365.25 * 86_400.0
_SECONDS_PER_HOUR = 3_600.0


def _within_ulps(actual: float, expected: float, ulps: int) -> bool:
    """Whether `actual` is `expected` or within `ulps` representable steps.

    Built from `math.nextafter` rather than a chosen epsilon, so the bound is
    the arithmetic's own granularity at that magnitude instead of a decimal
    tolerance that means something different at 0.001 and at 100,000. A NaN
    compares false against both bounds, which is the right answer here.

    A cross-level total needs a bound that scales, and the reason is the one in
    the module docstring: summing the slices' `net_pnl` evaluates
    `sum(p_k - f_k)` while the parent evaluates `(sum p) - (sum f)`. Each slice
    can contribute its own rounding, so the honest bound is the slice count,
    not one. Every identity that is a plain sum at both levels stays at one.
    """
    if ulps < 1:
        raise _fail("invalid_value", "an ULP bound is at least one step",
                    field="ulps")
    lower = expected
    upper = expected
    for _ in range(ulps):
        lower = math.nextafter(lower, -math.inf)
        upper = math.nextafter(upper, math.inf)
    return lower <= actual <= upper


def _within_one_ulp(actual: float, expected: float) -> bool:
    """Whether `actual` is `expected` or one of its two neighbours.

    The tightest bound, for the identities that reconcile term for term: the
    counts, the fees, the pre-cost PnL, and the two gross sums.
    """
    return _within_ulps(actual, expected, 1)


# ------------------------------------------------------------- trade cohorts


@dataclass
class _Cohort:
    """One set of trades, folded in trade ordinal order.

    The one fold in the module. The parent's three cohorts and every
    play-symbol accumulator use it, so a slice total and a parent total are
    the same additions over the same values rather than two implementations
    that happen to agree today.

    `gross_profit` and `gross_loss` are the industry terms and follow the
    architecture's definition rather than their names: both are sums over NET
    trade PnL, so costs are already inside them, and a break-even trade counts
    as a trade while contributing to neither.
    """

    n_trades: int = 0
    n_wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    pre_cost_pnl: float = 0.0
    fees: float = 0.0
    r_total: float = 0.0

    def fold(self, trade: PortfolioTrade) -> None:
        self.n_trades += 1
        self.pre_cost_pnl += trade.gross_pnl
        self.fees += trade.fees
        self.r_total += trade.r_multiple
        if trade.net_pnl > 0.0:
            self.n_wins += 1
            self.gross_profit += trade.net_pnl
        elif trade.net_pnl < 0.0:
            # Accumulated as a magnitude rather than negated at the end, so a
            # cohort that never lost reports a positive zero.
            self.gross_loss += -trade.net_pnl

    @property
    def net_pnl(self) -> float:
        return self.pre_cost_pnl - self.fees

    @property
    def win_rate(self) -> float | None:
        return None if self.n_trades == 0 else self.n_wins / self.n_trades

    @property
    def expectancy_r(self) -> float | None:
        return None if self.n_trades == 0 else self.r_total / self.n_trades

    @property
    def profit_factor_state(self) -> str:
        if self.gross_loss > 0.0:
            return "finite"
        return "infinite" if self.gross_profit > 0.0 else "unavailable"

    @property
    def profit_factor(self) -> float | None:
        """The ratio, or null where there is no denominator to divide by.

        A cohort that never lost has no finite profit factor, and the state
        above is what says which kind of nothing this is.
        """
        return None if self.gross_loss <= 0.0 else self.gross_profit / self.gross_loss


def _trade_stats(cohort: _Cohort) -> TradeStats:
    return TradeStats(
        n_trades=cohort.n_trades,
        n_wins=cohort.n_wins,
        win_rate=cohort.win_rate,
        gross_profit=cohort.gross_profit,
        gross_loss=cohort.gross_loss,
        profit_factor=cohort.profit_factor,
        profit_factor_state=cohort.profit_factor_state,
        expectancy_r=cohort.expectancy_r,
    )


# -------------------------------------------------------- slice accumulators


@dataclass
class _SliceTotals:
    """One canonical play symbol's attribution, still mutable.

    Deliberately not a `PortfolioSlice`. A slice is frozen and carries its
    three IC estimates, and the IC lens computes the complete map for every
    pair before any of them exists, so freezing one here would only mean
    rebuilding it there. This carries everything a slice needs except those
    estimates, and the IC lens performs the one construction.
    """

    play_id: str
    strategy: str
    symbol: str
    signals: int = 0
    rejection_counts: dict[RejectionReason, int] = field(default_factory=dict)
    cohort: _Cohort = field(default_factory=_Cohort)

    @property
    def trades(self) -> int:
        return self.cohort.n_trades

    @property
    def gross_profit(self) -> float:
        return self.cohort.gross_profit

    @property
    def gross_loss(self) -> float:
        return self.cohort.gross_loss

    @property
    def pre_cost_pnl(self) -> float:
        return self.cohort.pre_cost_pnl

    @property
    def fees(self) -> float:
        return self.cohort.fees

    @property
    def net_pnl(self) -> float:
        return self.cohort.net_pnl

    @property
    def win_rate(self) -> float | None:
        return self.cohort.win_rate

    @property
    def expectancy_r(self) -> float | None:
        return self.cohort.expectancy_r


def _slice_accumulators(events: ReplayEvents,
                        schedule: ValidatedSchedule) -> Mapping[PositionKey, _SliceTotals]:
    """One accumulator per canonical `(play_id, symbol)`, in canonical order.

    Every pair the request declares is present, including the ones that never
    signalled: an absent key would make a reader decide for themselves what it
    meant, and a slice that appeared only where something happened would let a
    silent play vanish from its own attribution.

    The canonical order here is the replay's evaluation order, play
    `(priority, play_id)` then symbol. The result codec applies its own
    `(play_id, symbol)` sort before hashing, so this order is a convenience for
    the caller rather than a claim about the serialized order.
    """
    _require_instance(events, "events", ReplayEvents)
    _require_instance(schedule, "schedule", ValidatedSchedule)
    request = schedule.request
    strategies = {play.play_id: play.strategy for play in request.plays}
    totals = {
        (play.play_id, symbol): _SliceTotals(
            play_id=play.play_id, strategy=play.strategy, symbol=symbol)
        for play in request.plays for symbol in request.symbols
    }
    _require_signal_counts(events.signal_counts, totals)
    for key, count in events.signal_counts.items():
        totals[key].signals = count
    for trade in events.trades:
        _attributed(totals, trade, "trades").cohort.fold(trade)
    for rejection in events.rejections:
        counts = _attributed(totals, rejection, "rejections").rejection_counts
        counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
    return MappingProxyType(totals)


def _require_signal_counts(counts: Mapping[PositionKey, int],
                           totals: Mapping[PositionKey, _SliceTotals]) -> None:
    """The loop's signal counts cover the request's pairs, exactly.

    The chronology builds one entry per canonical play symbol and the request
    declares the same set, so a disagreement means the two were not produced by
    one replay. Reading through a default instead would report a real signal
    count of zero, which is indistinguishable from a play that never spoke.
    """
    if set(counts) != set(totals):
        raise _fail(
            "mismatched_signal_counts",
            "the signal counts and the request name different play symbols",
            field="signal_counts", expected=len(totals), actual=len(counts),
        )


def _attributed(totals: Mapping[PositionKey, _SliceTotals], row,
                field_name: str) -> _SliceTotals:
    """The accumulator a trade or rejection belongs to, or a refusal.

    A row naming a pair the request never declared cannot be attributed, and
    dropping it would break the very identity the slices exist to prove.
    """
    key = (row.play_id, row.symbol)
    if key not in totals:
        raise _fail(
            "unknown_attribution_key", "a row names a play symbol the request "
            "does not declare", field=field_name, play_id=row.play_id,
            symbol=row.symbol,
        )
    return totals[key]


# --------------------------------------------------------- portfolio metrics


def _portfolio_metrics(curve: ReplayCurve, trades: tuple[PortfolioTrade, ...],
                       rejections: tuple[ReplayRejection, ...],
                       schedule: ValidatedSchedule) -> PortfolioMetrics:
    """One replay's complete metric set, under arithmetic version 2.

    The curve rather than a bare equity series, because two of these values
    belong to the curve and are copied rather than derived again: the account's
    own first and last marks, and the one benchmark return the benchmark lens
    published.
    """
    _require_instance(curve, "curve", ReplayCurve)
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_items(trades, "trades", PortfolioTrade)
    _require_items(rejections, "rejections", ReplayRejection)
    request = schedule.request
    _require_arithmetic_version(request)
    points = curve.equity
    if len(points) < 2:
        # The series is the opening anchor, one point per test close, and the
        # post-close mark, so two is the floor a real replay cannot go under.
        # Requiring it here is what gives the ulcer index a nonzero sample.
        raise _fail(
            "invalid_value", "a replay marks its opening anchor and its close",
            field="equity", points=len(points),
        )
    starting = _require_positive(points[0].portfolio_equity, "starting_equity")
    ending = points[-1].portfolio_equity
    cohorts = {"all": _Cohort(), "long": _Cohort(), "short": _Cohort()}
    held = 0.0
    for trade in trades:
        cohorts["all"].fold(trade)
        cohorts[trade.direction].fold(trade)
        held += (trade.exit_ts - trade.entry_ts).total_seconds()
    whole = cohorts["all"]
    # The one span two metrics divide by. It is positive because the window
    # type refuses `test_end <= test_start` when it is constructed, so neither
    # the exposure fraction below nor the annualization inside `_cagr` needs a
    # guard of its own.
    elapsed = (request.window.test_end - request.window.test_start).total_seconds()
    max_drawdown, ulcer_index = _drawdowns(points)
    cagr = _cagr(starting, ending, elapsed)
    pool = _daily_pool(_daily_returns(points, schedule, starting))
    sharpe, sortino, psr, skew, kurtosis = _pooled_statistics(pool)
    return PortfolioMetrics(
        all_trades=_trade_stats(whole),
        long_trades=_trade_stats(cohorts["long"]),
        short_trades=_trade_stats(cohorts["short"]),
        n_rejections=len(rejections),
        pre_cost_pnl=whole.pre_cost_pnl,
        fees=whole.fees,
        net_pnl=whole.net_pnl,
        starting_equity=starting,
        ending_equity=ending,
        total_return=ending / starting - 1.0,
        benchmark_return=curve.benchmark.total_return,
        max_drawdown=max_drawdown,
        ulcer_index=ulcer_index,
        cagr=cagr,
        calmar=cagr / max_drawdown if max_drawdown > 0.0 else None,
        exposure_pct=held / elapsed,
        avg_holding_hours=(
            0.0 if whole.n_trades == 0
            else held / whole.n_trades / _SECONDS_PER_HOUR
        ),
        daily_n=pool.n,
        daily_sum=pool.total,
        daily_sum_sq=pool.total_sq,
        daily_sum_sq_down=pool.total_sq_down,
        daily_sum_cube=pool.total_cube,
        daily_sum_fourth=pool.total_fourth,
        sharpe=sharpe,
        sortino=sortino,
        psr=psr,
        skew=skew,
        kurtosis=kurtosis,
    )


def _require_arithmetic_version(request: PortfolioReplayRequest) -> None:
    declared = request.execution.arithmetic_version
    if declared != ARITHMETIC_VERSION:
        raise _fail(
            "unsupported_arithmetic_version",
            "these are the metrics of one arithmetic version",
            field="arithmetic_version", expected=ARITHMETIC_VERSION,
            actual=declared,
        )


# ----------------------------------------------------------- the curve shape


def _drawdowns(points: Sequence[EquityPoint]) -> tuple[float, float]:
    """Maximum drawdown over the whole series, ulcer index over its own sample.

    One pass and one running peak, and two different samples on purpose. The
    architecture defines maximum drawdown as the maximum of the drawdown at
    each point, and the ulcer index as the root mean square drawdown "across
    the starting point and every close point", which is the anchor and one
    point per test interval. The final post-close mark is a third kind of
    point, taken after the window's forced liquidation, and it is outside that
    enumeration.

    The contrast is deliberate rather than shorthand: the daily sampling rule
    is written as "the last post-event equity point", which is what makes the
    same post-close mark the sample there and not here. Averaging it into the
    ulcer index would let the cost of liquidating at the window's end deepen a
    statistic that is meant to describe how long the account spent underwater.

    The peak includes the point being measured, so a new high draws down by
    exactly zero and no drawdown is ever negative. It starts at the anchor,
    which is positive, so the denominator cannot vanish.
    """
    peak = points[0].portfolio_equity
    deepest = 0.0
    squares = 0.0
    underwater = len(points) - 1
    for index, point in enumerate(points):
        peak = max(peak, point.portfolio_equity)
        drawdown = (peak - point.portfolio_equity) / peak
        deepest = max(deepest, drawdown)
        if index < underwater:
            squares += drawdown ** 2
    return (deepest, math.sqrt(squares / underwater))


def _cagr(starting: float, ending: float, elapsed: float) -> float:
    """The window's return, annualized over its own elapsed seconds.

    An account at or below zero has no root to take and reports a total loss.
    Above it, a large gain over a short window can annualize past the largest
    binary64: there is no honest value to report and the contract admits no
    infinity, so it refuses inside the closed taxonomy rather than raising an
    `OverflowError` out of it.
    """
    if ending <= 0.0:
        return -1.0
    # The window contract makes this positive: `test_start < test_end` is
    # checked when the window is constructed, so the elapsed span is never
    # zero and never negative.
    years = elapsed / _SECONDS_PER_YEAR
    try:
        growth = (ending / starting) ** (1.0 / years)
    except OverflowError:
        raise _fail("nonfinite_binary64", "value must be finite",
                    field="cagr") from None
    return _require_binary64(growth - 1.0, "cagr")


# --------------------------------------------------------- daily sampling


def _daily_returns(points: Sequence[EquityPoint], schedule: ValidatedSchedule,
                   starting: float) -> tuple[float, ...]:
    """One return per exchange session the window tests.

    The sessions come from the schedule rather than from the points, so the
    sample is the exchange's own calendar and a curve that failed to mark a
    session is a refusal instead of a silently shorter series. Inside a
    session the LAST point wins, which is what distinguishes the final
    post-close point from the close it shares a timestamp with.

    The first return divides by starting equity and every later one by the
    previous session's own last point.
    """
    sessions: list[date] = []
    closing: dict[Timestamp, date] = {}
    for interval in schedule.test_intervals:
        if not sessions or sessions[-1] != interval.session_date:
            sessions.append(interval.session_date)
        closing[interval.close_ts] = interval.session_date
    marked: dict[date, float] = {}
    for point in points:
        session = closing.get(point.ts)
        if session is not None:
            marked[session] = point.portfolio_equity
    returns = []
    previous = starting
    for session in sessions:
        if session not in marked:
            raise _fail(
                "misaligned_marks", "a scheduled session carries no equity point",
                field="equity", session_date=session.isoformat(),
            )
        value = marked[session]
        if previous <= 0.0:
            # Zero has no quotient at all, and a negative denominator has a
            # quotient that lies: an account climbing from -1,000 to -500 would
            # report a 50 percent LOSS. Equity below zero is reachable, since a
            # short can lose more than its collateral, so this is refused
            # rather than reported backwards.
            raise _fail(
                "nonfinite_binary64", "value must be finite",
                field="daily_return", session_date=session.isoformat(),
            )
        returns.append(value / previous - 1.0)
        previous = value
    return tuple(returns)


@dataclass(frozen=True)
class _DailyPool:
    """The six values that add across windows where a ratio cannot.

    Every pooled statistic below is recovered from these, which is why they
    are stored on the result rather than the moments: thirteen windows of sums
    pool into one sample, and thirteen Sharpes do not.
    """

    n: int
    total: float
    total_sq: float
    total_sq_down: float
    total_cube: float
    total_fourth: float


def _daily_pool(returns: Sequence[float]) -> _DailyPool:
    """The six sums, accumulated in session order.

    A power of a very large return can leave binary64 outright, which raises
    an `OverflowError` from the exponentiation rather than producing an
    infinity. That is refused as the nonfinite value it is, so no arithmetic
    here escapes the closed replay taxonomy. Sums that merely reach infinity
    do not raise, and the statistics built on them go null instead.
    """
    total = 0.0
    total_sq = 0.0
    total_sq_down = 0.0
    total_cube = 0.0
    total_fourth = 0.0
    try:
        for value in returns:
            total += value
            total_sq += value ** 2
            # A minimum acceptable return of zero, divided later by the FULL
            # count. That is what makes the downside sum poolable: a
            # denominator that varied with the count of losing days would not
            # add.
            total_sq_down += min(value, 0.0) ** 2
            total_cube += value ** 3
            total_fourth += value ** 4
    except OverflowError:
        raise _fail("nonfinite_binary64", "value must be finite",
                    field="daily_moment") from None
    return _DailyPool(
        n=len(returns), total=total, total_sq=total_sq,
        total_sq_down=total_sq_down, total_cube=total_cube,
        total_fourth=total_fourth,
    )


def _pooled_statistics(
    pool: _DailyPool,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Sharpe, Sortino, PSR, skew, and kurtosis, or nulls.

    All five are null below sixty daily returns. An annualized ratio on twenty
    returns is a rounding of noise, and the correction terms in the higher
    moments are written for a sample rather than a handful.

    Above the floor each still answers for its own denominator. The Sharpe and
    the three moment statistics need a positive population variance, which is
    what `pooled_moments` returns nothing without. The Sortino needs a positive
    downside deviation instead, and those are different questions: a series
    that lost the same amount every day has no variance and a perfectly good
    downside deviation.

    Two notes on the shared derivations, because they are bits rather than
    formulas. The architecture writes biased skew as `m3 / m2**1.5` and
    `pooled_moments` evaluates the same quantity as `m3 / sqrt(m2)**3`; it
    writes the PSR denominator's third term as `(kurtosis - 1) * sr**2 / 4` and
    `probabilistic_sharpe_ratio` evaluates `(kurtosis - 1) / 4 * sharpe**2`.
    Both pairs are the same expression associated differently, and both can
    round differently in the last place.

    The shared functions are still the right ones to call. The platform pools
    windows through exactly these, so a per-window statistic reported here and
    a pooled one reported there agree bit for bit, which a private copy of the
    algebra could not promise. Reassociating them to the spec's parenthesization
    is one owner decision made once for both repositories, in `nakagai/stats.py`,
    rather than a divergence introduced here.
    """
    if pool.n < POOLED_MIN_DAILY:
        return (None, None, None, None, None)
    mean = pool.total / pool.n
    downside = math.sqrt(pool.total_sq_down / pool.n)
    sortino = None if downside == 0.0 else mean / downside * _ANNUALIZE
    moments = pooled_moments(pool.n, pool.total, pool.total_sq,
                             pool.total_cube, pool.total_fourth)
    if moments is None:
        return (None, _finite(sortino), None, None, None)
    return (
        _finite(moments.sharpe * _ANNUALIZE),
        _finite(sortino),
        _finite(probabilistic_sharpe_ratio(moments)),
        _finite(moments.skew),
        _finite(moments.kurtosis),
    )


def _finite(value: float | None) -> float | None:
    """A measurement, or nothing where an intermediate left the reals."""
    return None if value is None or not math.isfinite(value) else value
