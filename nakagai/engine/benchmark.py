"""The two series a user reads: the account's equity curve and its benchmark.

Everything downstream of a replay draws from these points. A metric, a chart, a
sweep ranking, and a Lab equity line are all views of one tuple of
`EquityPoint` values, so the arithmetic here is where a wrong number becomes a
wrong number everywhere at once. Four rules keep that from happening.

- The ACCOUNT side is copied, never recomputed. `_Ledger.snapshot` already
  decided what the account was worth at an instant, and it decided it by summing
  over its own positions in canonical order. Recomputing any of it here would
  be a second settlement authority that could disagree with the first.
- The BENCHMARK side never touches the account. It is a shadow basket with no
  cash, no fees, no slippage, no settlement, and no capacity policy: it divides
  starting equity equally at the first test open, buys normalized units, and
  holds them. Its symbols are declared by the request and are never inferred,
  never defaulted to SPY, and never chosen from product configuration.
- Both series are marked at the SAME instants, and those instants come from the
  validated schedule rather than from a calendar or from the marks themselves.
  The two are two statements of one chronology, so `_equity_series` refuses when
  they disagree instead of pairing benchmark values with the wrong closes.
- Every price the benchmark reads is a RAW print, and every value it produces
  crosses `_require_binary64` at the site that produced it.

One arithmetic note, because it is load bearing rather than stylistic. The
basket is valued as `starting_equity * (sum(price / first_open) / n)` rather
than by summing `units * price` over fractional unit counts. The two are the
same in exact arithmetic and they are not the same in binary64: at the anchor
every ratio is exactly 1.0, so their sum is exactly `n`, `n / n` is exactly 1.0,
and the anchor is exactly starting equity for any symbol count and any starting
equity. Fractional units drift by an ULP or two at that same anchor, which
would put the first point of the benchmark series slightly off the number the
result reports as its own starting equity.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pandas import Timestamp

from nakagai.engine.bars import (
    BASE_TIMEFRAME,
    _MISSING_BAR,
    _ValidatedPortfolioBars,
)
from nakagai.engine.execution import ReplayEvents, ReplayMark, _RawBase
from nakagai.engine.portfolio_types import (
    BenchmarkResult,
    EquityPoint,
    PortfolioReplayRequest,
    ReplayInputError,
    _fail,
    _require_binary64,
    _require_instance,
)
from nakagai.engine.schedule import ValidatedSchedule


@dataclass(frozen=True)
class ReplayCurve:
    """One replay's equity points and the one benchmark return they carry.

    The benchmark appears twice on purpose and never disagrees with itself:
    every point carries the basket's value at that instant, and the result
    reports the single total return of the last of them.
    """

    equity: tuple[EquityPoint, ...]
    benchmark: BenchmarkResult


@dataclass(frozen=True)
class _ScheduledMark:
    """One recorded instant, and which raw price of which bar prices it.

    The anchor reads an OPEN and every other point reads a CLOSE, which is the
    whole difference between the opening anchor and a close mark.
    """

    ts: Timestamp
    column: str
    row: int


def _equity_series(schedule: ValidatedSchedule, events: ReplayEvents,
                   prepared: _ValidatedPortfolioBars) -> ReplayCurve:
    """The equity curve and the benchmark, on the frozen schedule's instants."""
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_instance(events, "events", ReplayEvents)
    _require_instance(prepared, "prepared", _ValidatedPortfolioBars)
    request = schedule.request
    scheduled = _scheduled_marks(schedule)
    _require_aligned(events.marks, scheduled)
    values = _benchmark_series(request, prepared, scheduled)
    points = tuple(
        _equity_point(request.replay_id, ordinal, mark, value)
        for ordinal, (mark, value) in enumerate(
            zip(events.marks, values, strict=True))
    )
    return ReplayCurve(
        equity=points,
        benchmark=BenchmarkResult(
            spec=request.benchmark,
            total_return=_require_binary64(
                points[-1].benchmark_equity / request.account.starting_equity - 1.0,
                "benchmark_return",
            ),
        ),
    )


def _equity_point(replay_id: str, ordinal: int, mark: ReplayMark,
                  benchmark_equity: float) -> EquityPoint:
    """One point: the account exactly as the ledger marked it, plus the basket.

    Every account field is a copy. `portfolio_equity` in particular is the
    ledger's own sum over its own positions, so the reconciliation identity a
    reader checks on this point is the identity the account settled under
    rather than an addition performed again here with a different rounding.
    """
    snapshot = mark.snapshot
    return EquityPoint(
        replay_id=replay_id,
        ts=mark.ts,
        point_ordinal=ordinal,
        settled_cash=snapshot.settled_cash,
        unsettled_cash=snapshot.unsettled_cash,
        short_collateral=snapshot.short_collateral,
        positions_liquidation_value=snapshot.positions_liquidation_value,
        portfolio_equity=snapshot.portfolio_equity,
        gross_exposure=snapshot.gross_exposure,
        open_positions=snapshot.open_positions,
        benchmark_equity=benchmark_equity,
    )


# ------------------------------------------------------------- the instants


def _scheduled_marks(schedule: ValidatedSchedule) -> tuple[_ScheduledMark, ...]:
    """Every instant a replay records, derived from the schedule alone.

    The opening anchor at `test_start`, priced at the first test interval's raw
    opens; one point at every test interval's close, priced at that interval's
    raw close; and the final post-close point at `test_end`, which reads the
    same last close as the point before it because no bar exists after it.

    Derived here rather than read off the marks the loop produced, so the two
    are independent statements of one chronology. `_require_aligned` then makes
    a disagreement a refusal, which is what stops a dropped or extra mark from
    silently sliding every benchmark value onto the wrong instant.
    """
    window = schedule.request.window
    closes = tuple(
        _ScheduledMark(row.close_ts, "close",
                       schedule.closed_base_count(row.close_ts) - 1)
        for row in schedule.test_intervals
    )
    return (
        _ScheduledMark(window.test_start, "open", closes[0].row),
        *closes,
        _ScheduledMark(window.test_end, "close", closes[-1].row),
    )


def _require_aligned(marks: Sequence[ReplayMark],
                     scheduled: Sequence[_ScheduledMark]) -> None:
    """The account's marks land on exactly the schedule's instants, in order."""
    if len(marks) != len(scheduled):
        raise _fail(
            "misaligned_marks", "the replay marked a different number of instants "
            "than the schedule declares", field="marks",
            expected=len(scheduled), actual=len(marks),
        )
    for mark, row in zip(marks, scheduled, strict=True):
        if mark.ts != row.ts:
            raise _fail(
                "misaligned_marks", "a mark was taken at an instant the schedule "
                "does not declare", field="marks",
                expected=row.ts.isoformat(), actual=mark.ts.isoformat(),
            )


# ------------------------------------------------------------- the benchmark


def _benchmark_symbols(request: PortfolioReplayRequest) -> tuple[str, ...]:
    """What the shadow account holds, as the request declared it.

    Partitioned on the explicit kind. An equal-weight basket holds the request's
    own canonical trading symbols; a single-symbol benchmark holds the one it
    names, whether or not any play trades it. There is no third branch and no
    fallback, because a benchmark nobody declared is a benchmark nobody agreed
    to compare against.
    """
    spec = request.benchmark
    if spec.kind == "single_symbol":
        return (spec.symbol,)
    return request.symbols


def _benchmark_bases(request: PortfolioReplayRequest,
                     prepared: _ValidatedPortfolioBars) -> tuple[_RawBase, ...]:
    """One raw base reader per benchmark symbol, in canonical order.

    The bar preflight already requires every benchmark symbol's base frame and
    refuses a replay without one before a strategy is constructed. What it
    cannot see is a curve computed over ANOTHER request's prepared bars, which
    the prepared value carries no record of, so that is what this refuses. The
    alternative is a bare `KeyError` out of the closed replay taxonomy.
    """
    pairs = frozenset(prepared.pairs)
    built = []
    for symbol in _benchmark_symbols(request):
        if (symbol, BASE_TIMEFRAME) not in pairs:
            raise ReplayInputError(
                _MISSING_BAR,
                "the benchmark's base frame was never prepared",
                {"field": "benchmark", "symbol": symbol,
                 "timeframe": BASE_TIMEFRAME},
            )
        built.append(_RawBase(prepared.frame(symbol, BASE_TIMEFRAME)))
    return tuple(built)


def _benchmark_series(request: PortfolioReplayRequest,
                      prepared: _ValidatedPortfolioBars,
                      scheduled: Sequence[_ScheduledMark]) -> tuple[float, ...]:
    """The shadow basket's value at every recorded instant.

    Equal weight, bought once at the first test opens and never rebalanced: each
    symbol contributes its own price normalized by its own first open, and the
    basket is starting equity times the mean of those ratios. Summing in the
    request's canonical symbol order is what makes the value independent of the
    order the caller supplied its symbols in, since binary64 addition does not
    commute.
    """
    bases = _benchmark_bases(request, prepared)
    anchor = scheduled[0]
    opens = tuple(_price(base, anchor) for base in bases)
    equity = request.account.starting_equity
    count = float(len(bases))
    values = []
    for mark in scheduled:
        total = 0.0
        for base, first_open in zip(bases, opens, strict=True):
            total = _require_binary64(
                total + _price(base, mark) / first_open, "benchmark_ratio")
        values.append(_require_binary64(equity * (total / count), "benchmark_equity"))
    return tuple(values)


def _price(base: _RawBase, mark: _ScheduledMark) -> float:
    """One raw print, at the bar and the column this instant reads.

    Read through the same `_RawBase` the replay loop fills and marks against,
    rather than through a second column reader of this module's own, so a
    benchmark cannot come to disagree with the account about what a bar said.
    That reader already proves every price finite and positive, which is what
    normalizing against a first open needs; requiring it again here would be a
    check nothing could ever fail.
    """
    return getattr(base.at(mark.row), mark.column)
