"""The IC lens: which instants are measured, and what the measurement is.

Two failures are possible here and only one of them is loud. A lens that
crashes is found immediately; a lens that runs and reports `correlation=None,
observations=0` for every slice reads exactly like an honest measurement that
found nothing. Most of this file exists to tell those two apart, which is why
so many assertions are on OBSERVATION COUNTS rather than on correlations: a
count is the one number a silent no-op cannot fake.

Three habits carry the file:

- GOLDENS DERIVED FROM THE SCHEDULE. `base_schedule` carries forty base
  intervals, twenty-six on 2026-11-25 and fourteen on the 2026-11-27 half day.
  Every expected count below is worked out from those forty ordinals and the
  window boundaries, stated in the test that uses it, and never read back out
  of a run.
- CORRELATIONS DERIVED FROM THE FORMULA. `bar_frame` climbs by a constant step,
  so the forward return `close[t+k]/close[t] - 1` is `0.1k / close[t]`, which
  STRICTLY DECREASES in `t`. The forward rank series is therefore exactly the
  reverse of the observation order, and a scripted factor's Spearman follows
  from `1 - 6*sum(d^2) / (n * (n^2 - 1))` with the ranks written out by hand.
- THE TAIL IS THE SUBJECT. Every fixture here uses a window that ends before
  `ic_tail_end`, so "the factor stopped at `test_end`" and "the forward return
  read on past it" are two different, separately observable facts rather than
  one boundary doing both jobs.
"""

import dataclasses
import math
from pathlib import Path

import pandas as pd
import pytest

from nakagai.engine.bars import ReplayDependencies
from nakagai.engine.canonical import definition_digest
from nakagai.engine.ic import CausalFactorBars, _ic_map, _portfolio_slices
from nakagai.engine.portfolio_types import (
    PlayRequest,
    PortfolioMetrics,
    PortfolioSlice,
    ReplayInputError,
    StrategyOutputError,
    StrategyRuntimeError,
)
from nakagai.engine.registry import (
    FrozenStrategyRegistry,
    composite_definition,
    rules_definition,
)
from nakagai.strategies.catalog import catalog_definitions
from nakagai.strategies.rules import core_vocabulary
from tests.portfolio_fixtures import (
    BASE_ONLY,
    DEFINITION_BASE_A,
    DEFINITION_BASE_C,
    DEFINITION_BASE_D,
    FIRST_CLOSE,
    IC_OBSERVATIONS,
    IC_RULES_SPEC,
    IC_TAIL_LIMITED_OBSERVATIONS,
    IC_TEST_END,
    IC_TEST_START,
    SMA_CROSS_SPEC,
    BarPlan,
    ScriptedPlay,
    SignalPlan,
    bar_frame,
    base_request,
    base_schedule,
    ic_plays,
    ic_window,
    prepared_for,
    ramp_frame,
    replay_ic,
    ts,
)

CATALOG_SPECS = (Path(__file__).resolve().parents[1]
                 / "nakagai" / "strategies" / "catalog" / "specs")


# ------------------------------------------------------------------ margins


def descending(position: int, at: pd.Timestamp) -> float:
    """A factor that falls exactly as the forward return falls.

    `bar_frame` makes the forward return strictly decreasing in the
    observation position, so this grades every observation in the same order
    and the rank correlation is exactly one at every horizon.
    """
    return float(-position)


def swapped(position: int, at: pd.Timestamp) -> float:
    """`descending`, with its first two observations exchanged.

    Exchanging one adjacent pair of ranks makes `sum(d^2)` exactly two, which
    is the smallest departure from a perfect ranking and lands the coefficient
    a long way from a four-decimal boundary.
    """
    exchanged = {0: 1, 1: 0}
    return float(-exchanged.get(position, position))


def tied_at_the_top(position: int, at: pd.Timestamp) -> float:
    """Nine rising grades where the last two are equal.

    Under AVERAGE ranks the tied pair share rank 9.5; under an ordinal rule
    they would take 9 and 10 and the coefficient would be exactly -1.
    """
    return float(min(position, 8))


def constant(position: int, at: pd.Timestamp) -> float:
    return 7.0


def blank_after(cutoff: int):
    """A factor that grades nothing from `cutoff` onwards."""
    return lambda position, at: None if position >= cutoff else float(-position)


# ------------------------------------------------------------ small helpers


def ic_replay(**overrides):
    """One replay over rising bars, on the IC window, grading every close."""
    overrides.setdefault("plays", ic_plays(margin=descending))
    overrides.setdefault("window", ic_window())
    overrides.setdefault("build", bar_frame)
    return replay_ic(**overrides)


def counts_of(result, play_id: str = "play-a", symbol: str = "SPY") -> tuple:
    row = next(row for row in result.slices
               if row.play_id == play_id and row.symbol == symbol)
    return tuple(item.observations for item in row.ic)


def correlations_of(result, play_id: str = "play-a", symbol: str = "SPY") -> tuple:
    row = next(row for row in result.slices
               if row.play_id == play_id and row.symbol == symbol)
    return tuple(item.correlation for item in row.ic)


def only_call(factor_calls: list, play_id: str, symbol: str):
    return next(call for call in factor_calls
                if call.play_id == play_id and call.symbol == symbol)


def rules_ic(definition, params):
    """The IC map of one real rules definition, with no replay loop.

    `_ic_map` reads the schedule, the prepared bars, and the registry, and
    nothing the chronology produces, so a wiring test needs none of the loop.
    """
    request = base_request(
        plays=(PlayRequest(
            play_id="play-r", strategy=definition.name,
            definition_digest=definition_digest(
                definition.definition_digest, params),
            params=params, priority=100,
        ),),
        symbols=("SPY",),
        window=ic_window(),
    )
    schedule = base_schedule()
    validated, prepared = prepared_for(request, schedule, BASE_ONLY,
                                       build=ramp_frame)
    registry = FrozenStrategyRegistry.from_definitions((definition,))
    return _ic_map(validated, prepared, registry)


# ------------------------------------------------- membership on the base axis


def test_every_play_symbol_slice_carries_three_estimates_in_horizon_order():
    """Two plays and two symbols make four slices, and each carries 1, 5, 20.

    The shape claim the architecture makes about a result, checked at the
    result rather than at the map: a pair that never signalled still owns a
    slice and still owns three estimates.
    """
    result = ic_replay()

    assert len(result.slices) == 4
    assert {(row.play_id, row.symbol) for row in result.slices} == {
        (play_id, symbol)
        for play_id in ("play-a", "play-b") for symbol in ("QQQ", "SPY")
    }
    for row in result.slices:
        assert [item.horizon_bars for item in row.ic] == [1, 5, 20]
        assert row.trades == 0 and row.signals == 0


def test_the_factor_is_asked_for_the_scheduled_base_closes_it_tested():
    """Twenty-five closes, ascending, the first after `test_start`.

    The window opens the test at 14:45 and ends it at 21:00, so the schedule's
    ordinals 1 through 25 are tested and their closes run 15:00 through 21:00.
    """
    factor_calls = []
    ic_replay(factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert len(call.timestamps) == IC_OBSERVATIONS
    assert call.timestamps[0] == ts("2026-11-25T15:00:00Z")
    assert call.timestamps[-1] == IC_TEST_END
    assert list(call.timestamps) == sorted(call.timestamps)
    assert call.timeframe == "15m"


def test_an_interval_opening_exactly_at_test_end_is_never_an_observation():
    """The final interval OPENING before `test_end` is the last observation.

    Testing 14:45 through 15:00 covers exactly ordinal 1, which opens at 14:45
    and closes at 15:00. Ordinal 2 opens at 15:00, exactly on `test_end`, and
    closes at 15:15; both of its timestamps are absent.
    """
    factor_calls = []
    ic_replay(window=ic_window(test_end=ts("2026-11-25T15:00:00Z")),
              factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.timestamps == (ts("2026-11-25T15:00:00Z"),)
    assert call.labels == (ts("2026-11-25T14:45:00Z"),)


def test_a_close_landing_exactly_on_test_start_is_never_an_observation():
    """The membership interval is open on the left.

    Starting the test at 15:15 leaves ordinal 2 closing exactly there. That
    close belongs to an interval that opened before the test range, so the
    first observation is ordinal 3's close at 15:30.
    """
    factor_calls = []
    ic_replay(window=ic_window(test_start=ts("2026-11-25T15:15:00Z")),
              factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert ts("2026-11-25T15:15:00Z") not in call.timestamps
    assert call.timestamps[0] == ts("2026-11-25T15:30:00Z")
    # Ordinals 3 through 25 rather than 1 through 25.
    assert len(call.timestamps) == IC_OBSERVATIONS - 2


# ------------------------------------------ membership on a higher timeframe


HOURLY = ReplayDependencies(timeframes=("15m", "1h"), reference_pairs=())
FOUR_HOURLY = ReplayDependencies(timeframes=("15m", "4h"), reference_pairs=())
DAILY = ReplayDependencies(timeframes=("15m", "1d"), reference_pairs=())


def test_an_hourly_play_observes_its_own_freshness_and_not_a_base_close():
    """One observation, at the hourly bar's own `fresh_context_at`.

    `base_schedule` carries two hourly bars. The first is fresh at 15:00 on
    2026-11-25, inside this window; the second is fresh at 15:00 on 2026-11-27,
    long after `test_end`, so it is excluded.

    The counts are the second half of the claim. The hourly series has two
    rows, so the observation sits at ordinal 0 and only a one-bar horizon finds
    a close ahead of it. A lens reading forward returns off the BASE series
    instead would have found one at five and twenty bars too.
    """
    factor_calls = []
    result = ic_replay(dependencies=HOURLY, plays=ic_plays(timeframe="1h"),
                       factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.timestamps == (ts("2026-11-25T15:00:00Z"),)
    assert call.labels == (ts("2026-11-25T14:00:00Z"),)
    assert counts_of(result) == (1, 0, 0)


def test_freshness_landing_exactly_on_test_end_is_an_observation():
    """The membership interval is closed on the right.

    The hourly bar is fresh at 15:00, and this window ends there.
    """
    factor_calls = []
    ic_replay(dependencies=HOURLY, plays=ic_plays(timeframe="1h"),
              window=ic_window(test_end=ts("2026-11-25T15:00:00Z")),
              factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.timestamps == (ts("2026-11-25T15:00:00Z"),)


def test_freshness_landing_exactly_on_test_start_is_not_an_observation():
    """The membership interval is open on the left for a context bar too.

    Opening the test at 15:00 leaves the hourly bar fresh exactly there. It
    entitled a decision at the close the test range begins after, not at one
    inside it, so it is excluded and the factor is never asked anything.
    """
    factor_calls = []
    result = ic_replay(dependencies=HOURLY, plays=ic_plays(timeframe="1h"),
                       window=ic_window(test_start=ts("2026-11-25T15:00:00Z")),
                       factor_calls=factor_calls)

    assert factor_calls == []
    assert counts_of(result) == (0, 0, 0)


def test_a_context_bar_without_freshness_is_never_an_observation():
    """The noon four-hour bucket of an early close entitles no decision.

    Its period runs to 21:00 on the half day, three hours after the session
    ends, so no scheduled base close falls inside its freshness window and it
    carries no `fresh_context_at` at all. That is the only four-hour bar the
    schedule holds, so the factor is never asked anything.
    """
    factor_calls = []
    result = ic_replay(dependencies=FOUR_HOURLY, plays=ic_plays(timeframe="4h"),
                       window=None, factor_calls=factor_calls)

    assert factor_calls == []
    assert counts_of(result) == (0, 0, 0)
    assert correlations_of(result) == (None, None, None)


def test_a_daily_bar_is_observed_at_the_next_sessions_first_close():
    """The prior session's daily bar, at the first freshness of the next one.

    The daily bar for 2026-11-25 is available at the 2026-11-27 open and fresh
    at that session's first base close, 14:45. Only a window that tests the
    half day can see it, so this one uses the default window rather than the
    IC window, where that instant lies past `test_end`.
    """
    factor_calls = []
    ic_replay(dependencies=DAILY, plays=ic_plays(timeframe="1d"),
              window=base_request().window, factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.timestamps == (ts("2026-11-27T14:45:00Z"),)
    assert call.labels == (ts("2026-11-25T05:00:00Z"),)


def test_a_daily_bar_whose_freshness_follows_test_end_is_excluded():
    """The same bar, under a window that ends before it becomes fresh."""
    factor_calls = []
    result = ic_replay(dependencies=DAILY, plays=ic_plays(timeframe="1d"),
                       factor_calls=factor_calls)

    assert factor_calls == []
    assert counts_of(result) == (0, 0, 0)


# ------------------------------------------------------------ tail isolation


def test_the_factor_never_receives_a_bar_past_test_end():
    """Twenty-six causal rows out of forty prepared ones.

    `test_end` is the close of ordinal 25, so twenty-six base intervals have
    closed and the causal view's last LABEL is ordinal 25's open at 20:45. The
    prepared frame carries all forty, because ordinals 26 through 39 are the
    tail the forward returns read.
    """
    factor_calls = []
    result = ic_replay(factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.rows == (("15m", 26),)
    assert call.last_label == ts("2026-11-25T20:45:00Z")
    assert len(result.prepared.frame("SPY", "15m").index) == 40


def test_a_context_frame_is_cut_where_its_own_availability_stopped():
    """The hourly view carries one bar out of two, and the base clock is not
    what decides that.

    `base_schedule` holds two hourly bars, and the second becomes available at
    15:00 on 2026-11-27, long after this window's last close. A view cut by the
    BASE clock instead of by the timeframe's own availability would carry it
    anyway: twenty-six base intervals have closed and the hourly frame is only
    two rows long, so that cut takes everything.
    """
    factor_calls = []
    ic_replay(dependencies=HOURLY, plays=ic_plays(timeframe="1h"),
              factor_calls=factor_calls)

    call = only_call(factor_calls, "play-a", "SPY")
    assert call.rows == (("15m", 26), ("1h", 1))
    assert call.last_label == ts("2026-11-25T14:00:00Z")


def test_no_strategy_context_reaches_a_tail_bar():
    """Every call into a strategy stopped at the window's own last close."""
    calls = []
    ic_replay(calls=calls)

    assert calls, "the replay made no strategy calls at all"
    assert max(call.now for call in calls) == IC_TEST_END
    assert min(call.now for call in calls) > IC_TEST_START
    assert max(call.last_base_label for call in calls) == ts(
        "2026-11-25T20:45:00Z")


def test_forward_returns_read_the_declared_tail_and_stop_at_it():
    """Twenty-five, twenty-five, and nineteen pairs.

    Observations sit at ordinals 1 through 25 of a forty-row series. A one-bar
    horizon needs ordinal `t + 1` and a five-bar horizon `t + 5`, and both
    exist for every observation. A twenty-bar horizon needs `t + 20`, which
    runs off the end of the schedule for `t` above 19, so six observations lose
    their outcome and nineteen keep one.
    """
    result = ic_replay()

    assert counts_of(result) == (
        IC_OBSERVATIONS, IC_OBSERVATIONS, IC_TAIL_LIMITED_OBSERVATIONS)
    assert correlations_of(result) == (1.0, 1.0, 1.0)


def test_a_window_ending_at_the_tail_boundary_has_no_far_horizon_at_all():
    """The default window tests the last fourteen ordinals and has no tail.

    It ends at `ic_tail_end`, so the observations run from ordinal 26 to 39.
    Thirteen of them reach one bar ahead, nine reach five, and none reaches
    twenty. That last zero is the only honest answer when the declared tail
    supplies nothing.
    """
    result = ic_replay(window=base_request().window)

    assert counts_of(result) == (13, 9, 0)
    correlations = correlations_of(result)
    assert correlations[0] is not None
    assert correlations[1] is None
    assert correlations[2] is None


# ---------------------------------------------------------- the coefficient


def test_the_coefficient_rounds_to_four_decimal_places():
    """One exchanged adjacent pair, at three horizons.

    The forward return falls with the observation position, so `descending`
    ranks perfectly and `swapped` departs from it by exactly one adjacent
    exchange: `sum(d^2)` is 2 at every horizon.

    At one and five bars there are 25 pairs, so
    `1 - 6*2 / (25 * 624) = 1 - 12/15600 = 0.99923076923...`, which rounds to
    0.9992. At twenty bars there are 19, so
    `1 - 6*2 / (19 * 360) = 1 - 12/6840 = 0.99824561403...`, which rounds to
    0.9982.
    """
    result = ic_replay(plays=ic_plays(margin=swapped))

    assert correlations_of(result) == (0.9992, 0.9992, 0.9982)
    # The rounding is real rather than incidental: the unrounded coefficients
    # differ from what is reported.
    assert 1.0 - 12.0 / 15600.0 != 0.9992
    assert 1.0 - 12.0 / 6840.0 != 0.9982


def test_ties_take_their_average_rank():
    """Ten observations whose last two grades are equal.

    The window tests ordinals 1 through 10, so every horizon keeps all ten
    pairs. The forward ranks are 10 down to 1. Under AVERAGE ranks the factor
    ranks are 1 through 8 then 9.5 twice, which gives
    `-82 / sqrt(82 * 82.5) = -0.996965...`, reported as -0.997. An ordinal tie
    rule would rank them 9 and 10 and report exactly -1.0.
    """
    result = ic_replay(plays=ic_plays(margin=tied_at_the_top),
                       window=ic_window(test_end=ts("2026-11-25T17:15:00Z")))

    assert counts_of(result) == (10, 10, 10)
    assert correlations_of(result) == (-0.997, -0.997, -0.997)


def test_a_constant_factor_reports_no_coefficient():
    """One distinct rank on the factor side, and every pair still counted."""
    result = ic_replay(plays=ic_plays(margin=constant))

    assert correlations_of(result) == (None, None, None)
    assert counts_of(result) == (
        IC_OBSERVATIONS, IC_OBSERVATIONS, IC_TAIL_LIMITED_OBSERVATIONS)


def test_a_constant_forward_return_reports_no_coefficient():
    """A flat tape moves nothing, so every forward return is the same zero."""
    result = ic_replay(build=None)

    assert correlations_of(result) == (None, None, None)
    assert counts_of(result) == (
        IC_OBSERVATIONS, IC_OBSERVATIONS, IC_TAIL_LIMITED_OBSERVATIONS)


def test_a_null_margin_drops_only_its_own_pair():
    """Five ungraded observations, chosen where no horizon had lost one yet.

    Positions 20 through 24 are ordinals 21 through 25, which a twenty-bar
    horizon had already dropped for want of an outcome. So the near horizons
    fall from 25 to 20 and the far one stays at 19, which tells the two drop
    reasons apart.
    """
    result = ic_replay(plays=ic_plays(margin=blank_after(20)))

    assert counts_of(result) == (20, 20, IC_TAIL_LIMITED_OBSERVATIONS)


def test_below_ten_pairs_there_is_no_coefficient():
    """Nine observations, all of them counted and none of them correlated."""
    result = ic_replay(window=ic_window(test_end=ts("2026-11-25T17:00:00Z")))

    assert counts_of(result) == (9, 9, 9)
    assert correlations_of(result) == (None, None, None)


def test_exactly_ten_pairs_is_enough():
    """The floor is inclusive, so one more observation reports a number."""
    result = ic_replay(window=ic_window(test_end=ts("2026-11-25T17:15:00Z")))

    assert counts_of(result) == (10, 10, 10)
    assert correlations_of(result) == (1.0, 1.0, 1.0)


def test_every_estimate_is_a_plain_binary64_or_a_plain_count():
    """No numpy scalar and no bool, at either field."""
    result = ic_replay(plays=ic_plays(margin=swapped))

    for row in result.slices:
        for item in row.ic:
            assert type(item.observations) is int
            assert type(item.horizon_bars) is int
            if item.correlation is not None:
                assert type(item.correlation) is float
                assert math.isfinite(item.correlation)


# ------------------------------------------------------------- one symbol


def test_a_factor_reads_only_the_symbol_it_grades():
    """A price planted in one symbol's bars never reaches the other's view."""
    planted = BarPlan(symbol="QQQ", at=ts("2026-11-25T16:00:00Z"),
                      open=777.0, high=777.2, low=776.8, close=777.0)
    factor_calls = []
    ic_replay(bars=(planted,), factor_calls=factor_calls)

    assert only_call(factor_calls, "play-a", "QQQ").highest_close == 777.0
    assert only_call(factor_calls, "play-a", "SPY").highest_close < 200.0
    assert {(call.play_id, call.symbol) for call in factor_calls} == {
        (play_id, symbol)
        for play_id in ("play-a", "play-b") for symbol in ("QQQ", "SPY")
    }


# ------------------------------------------------- definitions without a factor


def test_a_definition_with_no_graded_factor_reports_no_measurement():
    """Zero observations and no correlation, at all three horizons."""
    result = ic_replay(plays=ic_plays(timeframe=None))

    for row in result.slices:
        assert [item.observations for item in row.ic] == [0, 0, 0]
        assert [item.correlation for item in row.ic] == [None, None, None]


def test_a_composite_carries_no_graded_factor():
    """Its members' margins are not one series, so it grades nothing."""
    member = rules_definition("ic_rules", DEFINITION_BASE_A, spec=IC_RULES_SPEC)
    combo = composite_definition("combo", DEFINITION_BASE_C,
                                 members={"ic_rules": member})

    assert combo.ic_factor is None
    assert combo.ic_timeframe is None


# ------------------------------------------------- the rules definition wiring


def test_a_rules_definition_binds_the_graded_spec_margin():
    """The binding exists at the definition, and the axis arrives with it."""
    definition = rules_definition("ic_rules", DEFINITION_BASE_A,
                                  spec=IC_RULES_SPEC)

    assert definition.ic_factor is not None
    assert definition.ic_timeframe({}) == "15m"


def test_a_rules_play_reports_a_real_measurement():
    """The wiring is what this pins, and the count is what proves it ran.

    A definition that bound no factor would report zero observations at every
    horizon, which is indistinguishable from a lens that ran and found
    nothing. The counts here are the schedule's own: 25, 25, and 19.

    The coefficient is exact rather than merely present, which is what pins
    the ORDER the margins come back in. `ramp_frame` widens `close - open`
    strictly with the row, so the graded margin ranks the observations in
    their own order, while the forward return falls strictly with the row. The
    two rankings are exact reverses and the coefficient is -1 at every
    horizon; a margin series handed back in any other order could not be.
    """
    definition = rules_definition("ic_rules", DEFINITION_BASE_A,
                                  spec=IC_RULES_SPEC)
    estimates = rules_ic(definition, {})[("play-r", "SPY")]

    assert [item.observations for item in estimates] == [
        IC_OBSERVATIONS, IC_OBSERVATIONS, IC_TAIL_LIMITED_OBSERVATIONS]
    assert [item.correlation for item in estimates] == [-1.0, -1.0, -1.0]


def test_a_private_rules_play_grades_the_spec_its_params_carry():
    """A definition binding no spec reads one out of the play's params."""
    definition = rules_definition("private_rules", DEFINITION_BASE_D)
    estimates = rules_ic(definition, {"spec": IC_RULES_SPEC})[("play-r", "SPY")]

    assert [item.observations for item in estimates] == [
        IC_OBSERVATIONS, IC_OBSERVATIONS, IC_TAIL_LIMITED_OBSERVATIONS]
    assert [item.correlation for item in estimates] == [-1.0, -1.0, -1.0]


def test_an_inert_rules_definition_grades_on_the_base_axis():
    """No spec at all is a strategy that reads nothing and grades nothing.

    It must still resolve to a declared axis: a definition whose axis named a
    timeframe its own dependencies never declared would abort the replay
    rather than report an empty measurement.
    """
    definition = rules_definition("private_rules", DEFINITION_BASE_D)

    assert definition.ic_timeframe({}) == "15m"
    assert definition.dependencies({}).timeframes == ("15m",)


def test_every_catalog_definition_binds_the_graded_spec_margin():
    """The shipped catalog is wired by construction, not by a second call."""
    definitions = catalog_definitions(CATALOG_SPECS, core_vocabulary)

    assert definitions
    for definition in definitions:
        assert definition.ic_factor is not None
        assert definition.ic_timeframe is not None


# ------------------------------------------------------- the factor contract


def test_a_factor_returning_the_wrong_number_of_margins_aborts():
    with pytest.raises(StrategyOutputError) as raised:
        ic_replay(plays=ic_plays(ic_returns=(0.5, 0.5)))

    assert raised.value.code == "invalid_ic_margins"
    assert raised.value.details["expected"] == IC_OBSERVATIONS


def test_a_factor_returning_a_nonfinite_margin_aborts():
    margins = (float("nan"),) * IC_OBSERVATIONS

    with pytest.raises(StrategyOutputError) as raised:
        ic_replay(plays=ic_plays(ic_returns=margins))

    assert raised.value.code == "nonfinite_binary64"


def test_a_factor_returning_something_that_is_not_a_margin_aborts():
    with pytest.raises(StrategyOutputError) as raised:
        ic_replay(plays=ic_plays(ic_returns=("high",) * IC_OBSERVATIONS))

    assert raised.value.code == "invalid_type"


def test_a_factor_that_raises_aborts_the_replay():
    with pytest.raises(StrategyRuntimeError) as raised:
        ic_replay(plays=ic_plays(ic_raises="no margin here"))

    assert raised.value.code == "strategy_raised"
    assert raised.value.details["operation"] == "ic_factor"


def test_a_declared_frame_this_replay_never_prepared_refuses():
    """A graded axis whose bars nobody hydrated is a refusal, not an empty map.

    `sma_cross` is evaluated on the hourly frame, so its axis is one the
    definition legitimately declares. This replay prepared the base frame
    alone, and reading an unprepared one would mean grading bars that were
    never validated against the schedule.
    """
    definition = rules_definition("sma_cross", DEFINITION_BASE_A,
                                  spec=SMA_CROSS_SPEC)

    with pytest.raises(ReplayInputError) as raised:
        rules_ic(definition, {})

    assert raised.value.code == "missing_required_bar"
    assert raised.value.details["timeframe"] == "1h"


def test_an_axis_the_definition_never_declared_refuses():
    """A graded axis outside the play's own data closure is a refusal.

    Nothing hydrates a frame the definition did not declare, so grading on one
    would read bars that were never validated against the schedule.
    """
    with pytest.raises(ReplayInputError) as raised:
        ic_replay(plays=ic_plays(timeframe="4h"), dependencies=BASE_ONLY)

    assert raised.value.code == "undeclared_ic_timeframe"


# ------------------------------------------------------ the frozen slices


def test_each_accumulator_becomes_exactly_one_slice():
    """One slice per canonical pair, carrying that pair's own totals."""
    result = ic_replay()

    assert len(result.slices) == len(result.totals)
    for row in result.slices:
        total = result.totals[(row.play_id, row.symbol)]
        assert row.strategy == total.strategy
        assert row.signals == total.signals
        assert row.net_pnl == total.net_pnl
        assert row.fees == total.fees


def test_each_slice_carries_the_estimates_measured_for_its_own_pair():
    """The frozen row's three estimates are the map's entry for THAT pair.

    Measured a SECOND time here rather than read back off the same slices. The
    fixture's `slice_ic` is those slices keyed by pair, so comparing a slice
    against it is comparing a value with itself and cannot fail. `_ic_map` is
    pure, so calling it over the same schedule, bars, and registry answers the
    same question independently.

    The four pairs are built to DISAGREE, which is what makes a mis-keying
    observable at all. The two plays grade different numbers of observations,
    and a 777 print planted in QQQ's tape moves its forward returns away from
    SPY's, so no two of the four estimate triples are equal. Over the default
    fixture, where both symbols carry one tape and both plays grade alike, all
    four would be identical and a map keyed by the wrong pair would pass.
    """
    planted = BarPlan(symbol="QQQ", at=ts("2026-11-25T16:00:00Z"),
                      open=777.0, high=777.2, low=776.8, close=777.0)
    result = ic_replay(bars=(planted,), plays=(
        ScriptedPlay(play_id="play-a", priority=100, ic_timeframe="15m",
                     ic_margin=descending),
        ScriptedPlay(play_id="play-b", priority=100, ic_timeframe="15m",
                     ic_margin=blank_after(15)),
    ))

    measured = _ic_map(result.schedule, result.prepared, result.registry)

    assert len({row.ic for row in result.slices}) == 4
    for row in result.slices:
        assert row.ic == measured[(row.play_id, row.symbol)]


def test_a_slice_carries_the_replays_own_identity():
    result = ic_replay()

    assert {row.replay_id for row in result.slices} == {result.request.replay_id}


def test_a_slice_reports_its_own_cohort_and_an_empty_one_reports_nothing():
    """One play signals on one symbol, so one of the four pairs traded.

    The frozen row is where an empty cohort's nulls have to be right: a pair
    that never traded reports no rate and no expectancy, and one that did
    reports both. The accumulator holds the same values, so this is the copy
    being checked rather than the arithmetic.
    """
    result = replay_ic(plays=(
        ScriptedPlay(play_id="play-a", priority=100, signals=(
            SignalPlan(symbol="SPY", at=FIRST_CLOSE, stop=99.0, target=103.0),)),
        ScriptedPlay(play_id="play-b", priority=100, signals=()),
    ))

    traded = [row for row in result.slices if row.trades > 0]
    assert [(row.play_id, row.symbol) for row in traded] == [("play-a", "SPY")]
    for row in traded:
        total = result.totals[(row.play_id, row.symbol)]
        assert (row.win_rate, row.expectancy_r) == (
            total.win_rate, total.expectancy_r)
        assert (row.gross_profit, row.gross_loss) == (
            total.gross_profit, total.gross_loss)
        assert row.win_rate is not None
    for row in result.slices:
        if row.trades == 0:
            assert (row.win_rate, row.expectancy_r) == (None, None)
            assert (row.gross_profit, row.gross_loss) == (0.0, 0.0)


def test_an_ic_map_that_misses_a_pair_refuses():
    """A slice cannot be built without the estimates that belong to it."""
    result = ic_replay()
    short = {key: value for key, value in result.slice_ic.items()
             if key != ("play-b", "QQQ")}

    with pytest.raises(ReplayInputError) as raised:
        _portfolio_slices(result.schedule, result.totals, short)

    assert raised.value.code == "mismatched_ic_map"


def test_the_portfolio_reports_no_cross_strategy_coefficient():
    """IC is slice detail. No parent metric carries one."""
    names = {field.name for field in dataclasses.fields(PortfolioMetrics)}

    assert not any("ic" == name or name.startswith("ic_") for name in names)
    assert "ic" in {field.name for field in dataclasses.fields(PortfolioSlice)}


# --------------------------------------------------------- the causal view


def test_the_causal_view_refuses_a_timeframe_it_does_not_carry():
    view = CausalFactorBars(symbol="SPY", timeframe="15m", frames={}, labels=())

    with pytest.raises(KeyError):
        view.frame("1h")
