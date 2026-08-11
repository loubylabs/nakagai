"""The IC lens: one play's graded margin against that symbol's own outcome.

This is the only place in a replay where a bar after `test_end` is read, and
the only place a number rounds. Both are deliberate and both are narrow.

- TWO CLOCKS, ONE PAIR. A pair is a FACTOR and an OUTCOME, and they are read
  through different doors on purpose. The factor is causal: the definition's
  graded function is handed a `CausalFactorBars` cut at `test_end`, which is
  the same prefix `build_scheduled_context` would have shown a strategy at the
  window's last close. The outcome is realized: `close[t + k] / close[t] - 1`
  reads the prepared frame's own tail, which runs on to `ic_tail_end`. Nothing
  carries a tail bar back the other way. `build_scheduled_context` refuses a
  `now` past `test_end`, so a strategy cannot reach one through its own door
  either.
- ONE AXIS PER PLAY. A play graded on the base timeframe is observed at every
  scheduled base close it tested; a play graded on an hourly, four-hour, or
  daily frame is observed at that frame's own `fresh_context_at`, which is the
  one close the schedule entitles it to decide at. The ordinal `t` and the
  outcome series follow the axis, so a daily play's five-bar horizon is five
  DAYS ahead rather than five base intervals.
- ATTRIBUTABLE, NEVER AGGREGATED. An estimate belongs to one
  `(replay_id, play_id, symbol, horizon)`. There is no portfolio IC and none
  is derivable from what this returns: averaging correlations across plays and
  symbols answers no defined question, so the shape simply does not exist.
- THE ONE ROUNDING. The coefficient rounds to four decimals, once, here.
  Every other number a replay reports is unrounded, and a reader who finds a
  rounded value anywhere else has found a defect.

The observation COUNT is the load-bearing field. A lens that never ran and a
lens that ran and found nothing both report `correlation=None`; only the count
tells them apart, so a null coefficient always travels with the sample size
that produced it.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd
from pandas import Timestamp

from nakagai.engine.bars import (
    BASE_TIMEFRAME,
    _MISSING_BAR,
    _ValidatedPortfolioBars,
    _require_timeframe,
)
from nakagai.engine.metrics import _SliceTotals
from nakagai.engine.portfolio import PositionKey
from nakagai.engine.portfolio_types import (
    IC_HORIZONS,
    IC_MIN_OBSERVATIONS,
    IcEstimate,
    PlayRequest,
    PortfolioSlice,
    StrategyOutputError,
    _fail,
    _require_instance,
)
from nakagai.engine.registry import StrategyDefinition, StrategyRegistry
from nakagai.engine.schedule import ValidatedSchedule
from nakagai.strategies.base import strategy_operation

# Four decimal places, applied once, to the one value in a replay that rounds.
IC_DECIMALS = 4


class CausalFactorBars:
    """One symbol's frames as they stood at `test_end`, and nothing later.

    A graded factor gets this and no other access to data. Each frame is the
    prefix the schedule had released by the window's last close: base bars
    that had fully closed, context bars whose `available_at` had arrived. The
    cut is a prefix because a prepared frame's labels ARE the schedule's
    labels, in the schedule's order.

    `labels` is the other half of the contract, and it is not the same thing
    as the observation timestamps the factor is called with. A timestamp says
    WHEN an observation was taken and a label says WHICH ROW carries it, and
    the two differ by construction: a base observation is taken at an
    interval's close and labeled at its open, and a daily observation is taken
    at the next session's first close and labeled at its own session's Eastern
    midnight. A factor evaluating a series has to index by the label; a factor
    reasoning about time reads the timestamp.

    The frames are views over the engine's own, which is safe for the same
    reason `build_scheduled_context`'s are: copy-on-write sends a write to a
    copy rather than back into the engine. It protects the ENGINE and not two
    consumers from each other, so the lens builds one of these per play symbol
    rather than sharing one.
    """

    __slots__ = ("_symbol", "_timeframe", "_frames", "_labels")

    def __init__(self, symbol: str, timeframe: str,
                 frames: Mapping[str, pd.DataFrame],
                 labels: tuple[Timestamp, ...]) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._frames = MappingProxyType(dict(frames))
        self._labels = tuple(labels)

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timeframe(self) -> str:
        """The observation axis: the frame the selected labels belong to."""
        return self._timeframe

    @property
    def frames(self) -> Mapping[str, pd.DataFrame]:
        """Every timeframe the definition declared, each cut at `test_end`."""
        return self._frames

    @property
    def labels(self) -> tuple[Timestamp, ...]:
        """The axis row of each observation, ascending, one per timestamp."""
        return self._labels

    def frame(self, timeframe: str) -> pd.DataFrame:
        """One causal frame, or `KeyError` for a timeframe nobody declared."""
        return self._frames[timeframe]


@dataclass(frozen=True)
class _Observation:
    """One selected instant: when it was taken, which row, and where.

    `ordinal` is the row's position in the FULL prepared series rather than in
    the causal view, because that is what the outcome is indexed by: the
    factor stops at `test_end` and `close[t + k]` does not.
    """

    at: Timestamp
    label: Timestamp
    ordinal: int


def _ic_map(
    schedule: ValidatedSchedule, prepared: _ValidatedPortfolioBars,
    registry: StrategyRegistry,
) -> Mapping[PositionKey, tuple[IcEstimate, IcEstimate, IcEstimate]]:
    """Three estimates for every canonical play symbol, before any slice.

    Complete by construction: every pair the request declares is a key,
    including the pairs whose definition grades nothing. A slice cannot be
    built without its entry, so a missing key is a refusal rather than a
    silently absent measurement.

    The order is the replay's own, play `(priority, play_id)` then symbol,
    which is the order `_slice_accumulators` produces. The result codec sorts
    before hashing, so this is a convenience for the caller rather than a
    claim about the serialized order.
    """
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_instance(prepared, "prepared", _ValidatedPortfolioBars)
    _require_instance(registry, "registry", StrategyRegistry)
    request = schedule.request
    estimates: dict[PositionKey, tuple[IcEstimate, ...]] = {}
    for play in request.plays:
        definition = registry.resolve(play.strategy)
        graded = _graded_axis(definition, play)
        if graded is None:
            for symbol in request.symbols:
                estimates[(play.play_id, symbol)] = _ungraded()
            continue
        axis, timeframes = graded
        # The selected instants belong to the schedule and the axis, not to a
        # symbol, so they are chosen once for the play and read by each of its
        # symbols. With none selected there is nothing to grade and nothing is
        # asked: a factor called with an empty series would be answering a
        # question no observation posed.
        observations = _observations(schedule, axis)
        for symbol in request.symbols:
            estimates[(play.play_id, symbol)] = (
                _estimates(schedule, prepared, definition, play, symbol,
                           axis, timeframes, observations)
                if observations else _ungraded()
            )
    return MappingProxyType(estimates)


def _ungraded() -> tuple[IcEstimate, IcEstimate, IcEstimate]:
    """No graded margin contract, so no correlation and no sample at all.

    Zero rather than a null count: nothing was measured, and a reader has to
    be able to tell that from a measurement that dropped every pair.
    """
    return tuple(
        IcEstimate(horizon_bars=horizon, correlation=None, observations=0)
        for horizon in IC_HORIZONS
    )


def _graded_axis(definition: StrategyDefinition,
                 play: PlayRequest) -> tuple[str, tuple[str, ...]] | None:
    """The play's observation axis and its declared frames, or nothing.

    Both come from the definition's own pure functions, and both are called
    once per play rather than once per play symbol: they answer for the
    params, and the symbol is not one of their inputs.

    The axis has to be one of the frames the definition declared. Nothing
    hydrates a frame a definition never asked for, so grading on one would
    mean reading bars the schedule never validated, which is refused here
    rather than surfacing as a lookup failure inside the factor.
    """
    if definition.ic_factor is None:
        return None
    named = {"strategy": definition.name, "play_id": play.play_id}
    with strategy_operation("dependencies", **named):
        declared = definition.dependencies(play.params)
    with strategy_operation("ic_timeframe", **named):
        axis = definition.ic_timeframe(play.params)
    axis = _require_timeframe(axis, "ic_timeframe")
    if axis not in declared.timeframes:
        raise _fail(
            "undeclared_ic_timeframe",
            "a graded factor observes a timeframe its definition never declares",
            field="ic_timeframe", strategy=definition.name,
            play_id=play.play_id, timeframe=axis,
            declared=declared.timeframes,
        )
    return (axis, declared.timeframes)


def _estimates(
    schedule: ValidatedSchedule, prepared: _ValidatedPortfolioBars,
    definition: StrategyDefinition, play: PlayRequest, symbol: str,
    axis: str, timeframes: tuple[str, ...],
    observations: tuple[_Observation, ...],
) -> tuple[IcEstimate, IcEstimate, IcEstimate]:
    """One play symbol's three estimates, from one call into its factor.

    One view and one call per play symbol, never one shared between them. A
    causal frame is a slice of the engine's own, and copy-on-write separates a
    writer from the ENGINE rather than from another reader holding the same
    slice; a symbol's outcomes are its own for the same reason.
    """
    bars = CausalFactorBars(
        symbol=symbol, timeframe=axis,
        frames=_causal_frames(schedule, prepared, symbol, timeframes),
        labels=tuple(row.label for row in observations),
    )
    timestamps = tuple(row.at for row in observations)
    with strategy_operation("ic_factor", strategy=definition.name,
                            play_id=play.play_id, symbol=symbol):
        returned = definition.ic_factor(play.params, symbol, bars, timestamps)
    margins = _require_margins(returned, len(timestamps), definition, play, symbol)
    closes = tuple(
        float(value) for value in prepared.frame(symbol, axis)["close"].to_numpy())
    return tuple(
        _estimate(horizon, margins, observations, closes)
        for horizon in IC_HORIZONS
    )


def _observations(schedule: ValidatedSchedule,
                  axis: str) -> tuple[_Observation, ...]:
    """The instants this axis was measured at, ascending, inside the window.

    The membership interval is open on the left and closed on the right,
    because the replay evaluates a bar at the close of an interval whose OPEN
    lies in `[test_start, test_end)`. Written as the architecture writes it,
    against the observation timestamp itself, so the boundary this test-drives
    is the one the contract states rather than an equivalent restatement of it.

    A base observation is a scheduled interval close. A higher-timeframe
    observation is a context bar's `fresh_context_at`, which is the emission
    gate: a bar is READABLE for every base close after it is released and
    entitles a decision at exactly one of them. A bar whose freshness is null,
    which is what an early close does to the noon four-hour bucket, entitles
    no decision and is therefore never an observation.

    Ordering needs no sort. Base closes rise strictly by schedule validation,
    and a context bar's freshness sits inside `[period_end, period_end + one
    base bar)` over strictly rising period ends.
    """
    window = schedule.request.window
    if axis == BASE_TIMEFRAME:
        rows = enumerate(schedule.base_intervals)
        return tuple(
            _Observation(at=row.close_ts, label=row.open_ts, ordinal=ordinal)
            for ordinal, row in rows
            if window.test_start < row.close_ts <= window.test_end
        )
    return tuple(
        _Observation(at=row.fresh_context_at, label=row.label_ts, ordinal=ordinal)
        for ordinal, row in enumerate(schedule.context_bars(axis))
        if row.fresh_context_at is not None
        and window.test_start < row.fresh_context_at <= window.test_end
    )


def _causal_frames(schedule: ValidatedSchedule, prepared: _ValidatedPortfolioBars,
                   symbol: str, timeframes: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Every declared frame, cut where the window's last close left it."""
    at = schedule.request.window.test_end
    frames: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        try:
            frame = prepared.frame(symbol, timeframe)
        except KeyError:
            raise _fail(
                _MISSING_BAR,
                "a graded definition declares a frame this replay never prepared",
                field="frame", symbol=symbol, timeframe=timeframe,
            ) from None
        frames[timeframe] = frame.iloc[:_causal_rows(schedule, timeframe, at)]
    return frames


def _causal_rows(schedule: ValidatedSchedule, timeframe: str, at: Timestamp) -> int:
    """How many rows of `timeframe` the schedule had released at `at`."""
    if timeframe == BASE_TIMEFRAME:
        return schedule.closed_base_count(at)
    return schedule.available_context_count(timeframe, at)


def _require_margins(returned: object, expected: int,
                     definition: StrategyDefinition, play: PlayRequest,
                     symbol: str) -> tuple[float | None, ...]:
    """One finite margin or one null per requested timestamp, in order.

    Order is the one clause of the contract nothing here can check, because
    the order is the lens's own: the factor is handed ascending timestamps and
    answers positionally. Length and value are checkable and both are refused
    as strategy output, which is what the architecture calls them.
    """
    named = {"operation": "ic_factor", "strategy": definition.name,
             "play_id": play.play_id, "symbol": symbol}
    if isinstance(returned, (str, bytes)) or not isinstance(returned, Sequence):
        raise StrategyOutputError(
            "invalid_type", "a graded factor returns one margin per timestamp",
            {**named, "seen": type(returned).__name__},
        )
    if len(returned) != expected:
        raise StrategyOutputError(
            "invalid_ic_margins",
            "a graded factor returns one margin per timestamp",
            {**named, "expected": expected, "actual": len(returned)},
        )
    margins: list[float | None] = []
    for value in returned:
        if value is None:
            margins.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StrategyOutputError(
                "invalid_type", "a margin is a finite number or nothing",
                {**named, "seen": type(value).__name__},
            )
        number = float(value)
        if not math.isfinite(number):
            raise StrategyOutputError(
                "nonfinite_binary64", "value must be finite",
                {**named, "field": "ic_margin"},
            )
        margins.append(number)
    return tuple(margins)


def _estimate(horizon: int, margins: tuple[float | None, ...],
              observations: tuple[_Observation, ...],
              closes: tuple[float, ...]) -> IcEstimate:
    """One horizon: pair, drop, count, correlate.

    A pair survives only when the factor graded the observation AND the series
    reaches a close `horizon` bars later. The two drop for different reasons
    and both leave the count smaller, which is why the count travels with the
    coefficient rather than being recoverable from the window.
    """
    factor: list[float] = []
    outcome: list[float] = []
    for margin, row in zip(margins, observations, strict=True):
        if margin is None:
            continue
        realized = _forward_return(closes, row.ordinal, horizon)
        if realized is None:
            continue
        factor.append(margin)
        outcome.append(realized)
    return IcEstimate(
        horizon_bars=horizon,
        correlation=_rank_correlation(factor, outcome),
        observations=len(factor),
    )


def _forward_return(closes: tuple[float, ...], ordinal: int,
                    horizon: int) -> float | None:
    """`close[t + k] / close[t] - 1`, or nothing where the series stops.

    The series is the symbol's own prepared frame on the observation axis, so
    it runs to `ic_tail_end` and the outcome may legitimately land after
    `test_end`. Past the last declared tail bar there is no realized return to
    report, and the pair is dropped rather than extrapolated.
    """
    ahead = ordinal + horizon
    if ahead >= len(closes):
        return None
    base = closes[ordinal]
    if base == 0.0:
        # A quotient that does not exist. Zero is a price a validated bar may
        # carry, and an infinity is not a return.
        return None
    realized = closes[ahead] / base - 1.0
    return realized if math.isfinite(realized) else None


def _rank_correlation(factor: Sequence[float],
                      outcome: Sequence[float]) -> float | None:
    """Spearman: Pearson over average ranks, rounded to four decimals once.

    Null below the minimum pair count, because a coefficient over a handful of
    overlapping forward returns is a rounding of noise rather than a
    measurement. Null again when either rank series has fewer than two
    distinct values, which is the case a constant factor and a flat tape both
    reach: there is no ranking to correlate and the quotient has no
    denominator.

    Ties take their AVERAGE rank. An ordinal rule would answer a different
    question, since it would let the order two equal grades happened to arrive
    in decide the coefficient.

    That distinct-value test is also what makes the division below safe, and
    it is the only guard here for exactly that reason. A rank series with two
    distinct values has two ranks that differ from their own mean, so its
    spread is strictly positive; a second check on the spread would be an
    unreachable branch restating this one in arithmetic.

    Zero is added to the rounded value on purpose: `float.hex` spells negative
    zero differently, so a result carrying one hashes differently from the
    identical result that carried a positive zero.
    """
    count = len(factor)
    if count < IC_MIN_OBSERVATIONS:
        return None
    left = _average_ranks(factor)
    right = _average_ranks(outcome)
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    covariance = 0.0
    spread_left = 0.0
    spread_right = 0.0
    for one, other in zip(left, right, strict=True):
        gap_left = one - mean_left
        gap_right = other - mean_right
        covariance += gap_left * gap_right
        spread_left += gap_left * gap_left
        spread_right += gap_right * gap_right
    coefficient = covariance / math.sqrt(spread_left * spread_right)
    if not math.isfinite(coefficient):
        return None
    return round(coefficient, IC_DECIMALS) + 0.0


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Ascending ranks from one, with every tied run sharing its average."""
    order = sorted(range(len(values)), key=lambda position: values[position])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2.0 + 1.0
        for position in order[start:end + 1]:
            ranks[position] = average
        start = end + 1
    return tuple(ranks)


# ------------------------------------------------------------ the frozen rows


def _portfolio_slices(
    schedule: ValidatedSchedule, totals: Mapping[PositionKey, _SliceTotals],
    ic: Mapping[PositionKey, tuple[IcEstimate, IcEstimate, IcEstimate]],
) -> tuple[PortfolioSlice, ...]:
    """One frozen slice per accumulator, each consumed exactly once.

    The last construction in a replay's attribution, and the reason
    `_slice_accumulators` stops short of it: a slice carries its three
    estimates, and the complete IC map exists before any of them, so freezing
    a slice earlier would only mean rebuilding it here.

    The two mappings must name the same play symbols. A pair in one and not
    the other means the accumulators and the lens ran over different replays,
    and reading through a default would report a real measurement as an absent
    one, or attach one pair's estimates to another pair's totals.
    """
    _require_instance(schedule, "schedule", ValidatedSchedule)
    if set(totals) != set(ic):
        raise _fail(
            "mismatched_ic_map",
            "the accumulators and the IC map name different play symbols",
            field="ic", expected=len(totals), actual=len(ic),
        )
    replay_id = schedule.request.replay_id
    return tuple(
        PortfolioSlice(
            replay_id=replay_id,
            play_id=row.play_id,
            strategy=row.strategy,
            symbol=row.symbol,
            signals=row.signals,
            trades=row.trades,
            rejection_counts=row.rejection_counts,
            gross_profit=row.gross_profit,
            gross_loss=row.gross_loss,
            pre_cost_pnl=row.pre_cost_pnl,
            net_pnl=row.net_pnl,
            fees=row.fees,
            win_rate=row.win_rate,
            expectancy_r=row.expectancy_r,
            ic=ic[key],
        )
        for key, row in totals.items()
    )
