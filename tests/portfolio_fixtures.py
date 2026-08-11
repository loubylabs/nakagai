"""Shared builders for the portfolio replay tests.

Every builder returns a complete, contract-valid value. A test that needs an
invalid value asks for the valid one and replaces exactly the field under
test, so one failure names one cause. Parent identities are derived here the
way platform derives them: build the draft, ask core for the candidate
identity, then ask core for the replay identity.

Later Phase 1 tasks extend this module with bars, registries, and a replay
helper. It stays a thin assembler over the real core values and never grows a
second replay implementation.
"""

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import field
from datetime import date

import pandas as pd

from nakagai.engine.bars import PortfolioBars, ReplayDependencies
from nakagai.engine.canonical import (
    definition_digest,
    expected_candidate_id,
    expected_replay_id,
    rejection_id,
    result_digest,
    schedule_digest,
    trade_id,
)
from nakagai.engine.portfolio import EntryProposal, _Ledger
from nakagai.engine.portfolio_types import (
    AccountPolicy,
    BenchmarkResult,
    BenchmarkSpec,
    EntryIntent,
    EquityPoint,
    ExchangeScheduleIdentity,
    ExecutionPolicy,
    ExitReason,
    FeeSpec,
    IcEstimate,
    PlayRequest,
    PortfolioMetrics,
    PortfolioReplayRequest,
    PortfolioReplayResult,
    PortfolioSlice,
    PortfolioTrade,
    RejectionReason,
    ReplayRejection,
    ReplaySchedule,
    ReplayWindow,
    ScheduledBaseInterval,
    ScheduledContextBar,
    Signal,
    SlippageSpec,
    TradeStats,
)
from nakagai.engine.registry import (
    FrozenStrategyRegistry,
    StrategyDefinition,
    composite_definition,
    rules_definition,
)
from nakagai.engine.schedule import ValidatedSchedule, validate_schedule

BATCH_ID = "0198b1c2-3d4e-7f80-8123-456789abcdef"
REGISTRY_DIGEST = "1f" * 32
DEFINITION_BASE_A = "2a" * 32
DEFINITION_BASE_B = "3b" * 32
DEFINITION_BASE_C = "4c" * 32
DEFINITION_BASE_D = "5d" * 32
CALENDAR_VERSION = "exchange_calendars:4.5.6:nakagai-rth-v1"
PLACEHOLDER_DIGEST = "0" * 64

# Two real XNYS sessions around Thanksgiving 2026, chosen so the fixture is a
# correct example for the ten tasks that build on it. 2026-11-26 is the
# holiday and is simply absent, 2026-11-27 is the 13:00 Eastern early close,
# and both dates are EST, so a full session is 14:30Z to 21:00Z and the half
# day is 14:30Z to 18:00Z.
SESSION_ONE = date(2026, 11, 25)
SESSION_TWO = date(2026, 11, 27)
SESSION_ONE_INTERVALS = 26
SESSION_TWO_INTERVALS = 14

PLAY_A_PARAMS = {"fast_n": 10, "slow_n": 30, "allow_short": False}
PLAY_B_PARAMS = {"lookback": 20, "labels": ("z", "a"), "nested": {"z": 1.5, "a": None}}


def ts(text: str) -> pd.Timestamp:
    """One timestamp literal, so every fixture timestamp reads the same way."""
    return pd.Timestamp(text)


def base_window() -> ReplayWindow:
    """Warm up on the full session, trade the half day that follows it."""
    return ReplayWindow(
        train_start=ts("2026-11-25T14:30:00Z"),
        train_end=ts("2026-11-27T14:30:00Z"),
        test_start=ts("2026-11-27T14:30:00Z"),
        test_end=ts("2026-11-27T18:00:00Z"),
    )


def base_identity(digest: str = PLACEHOLDER_DIGEST) -> ExchangeScheduleIdentity:
    return ExchangeScheduleIdentity(
        calendar_id="XNYS",
        calendar_version=CALENDAR_VERSION,
        schedule_digest=digest,
        timezone="America/New_York",
        base_timeframe="15m",
    )


def base_intervals() -> tuple[ScheduledBaseInterval, ...]:
    """One full session and one early close, with the holiday between absent."""
    built = []
    for session, count in (
        (SESSION_ONE, SESSION_ONE_INTERVALS), (SESSION_TWO, SESSION_TWO_INTERVALS),
    ):
        for ordinal in range(count):
            open_ts = ts(f"{session.isoformat()}T14:30:00Z") + pd.Timedelta(
                minutes=15 * ordinal,
            )
            built.append(ScheduledBaseInterval(
                session_date=session,
                interval_ordinal=ordinal,
                open_ts=open_ts,
                close_ts=open_ts + pd.Timedelta(minutes=15),
            ))
    return tuple(built)


def base_context_bars() -> tuple[ScheduledContextBar, ...]:
    """One bar of every supported timeframe and source, ordered canonically.

    Each row follows the label semantics the architecture freezes. The hourly
    bars use their cached UTC left edge and become fresh at the base close that
    lands on their period end. The four-hour bar is the Eastern noon bucket of
    the early close: it runs to 16:00 Eastern, which is three hours after the
    half day ends, so no scheduled base close falls inside its freshness
    window and it carries no `fresh_context_at`. The daily bar is available at
    the next scheduled session open and fresh at that session's first base
    close, which is why the final session has no daily bar at all: nothing
    later in this schedule could make one available.
    """
    return (
        ScheduledContextBar(
            timeframe="1h", session_date=SESSION_ONE,
            label_ts=ts("2026-11-25T14:00:00Z"),
            period_start=ts("2026-11-25T14:00:00Z"),
            period_end=ts("2026-11-25T15:00:00Z"),
            available_at=ts("2026-11-25T15:00:00Z"),
            fresh_context_at=ts("2026-11-25T15:00:00Z"),
            source="fetched_left_edge",
        ),
        ScheduledContextBar(
            timeframe="1h", session_date=SESSION_TWO,
            label_ts=ts("2026-11-27T14:00:00Z"),
            period_start=ts("2026-11-27T14:00:00Z"),
            period_end=ts("2026-11-27T15:00:00Z"),
            available_at=ts("2026-11-27T15:00:00Z"),
            fresh_context_at=ts("2026-11-27T15:00:00Z"),
            source="fetched_left_edge",
        ),
        ScheduledContextBar(
            timeframe="4h", session_date=SESSION_TWO,
            label_ts=ts("2026-11-27T17:00:00Z"),
            period_start=ts("2026-11-27T17:00:00Z"),
            period_end=ts("2026-11-27T21:00:00Z"),
            available_at=ts("2026-11-27T21:00:00Z"),
            fresh_context_at=None,
            source="derived_1h_et_midnight",
        ),
        ScheduledContextBar(
            timeframe="1d", session_date=SESSION_ONE,
            label_ts=ts("2026-11-25T05:00:00Z"),
            period_start=ts("2026-11-25T14:30:00Z"),
            period_end=ts("2026-11-25T21:00:00Z"),
            available_at=ts("2026-11-27T14:30:00Z"),
            fresh_context_at=ts("2026-11-27T14:45:00Z"),
            source="session_aligned",
        ),
    )


def schedule_with(
    intervals: tuple[ScheduledBaseInterval, ...] | None = None,
    context_bars: tuple[ScheduledContextBar, ...] | None = None,
) -> ReplaySchedule:
    """A schedule whose identity carries its own recomputed digest.

    Every refusal fixture goes through here rather than editing a built
    schedule in place. Editing in place would leave the old digest on the
    identity, so `validate_schedule` would refuse for the digest and the test
    would pass without ever reaching the rule it names.
    """
    draft = ReplaySchedule(
        identity=base_identity(),
        base_intervals=base_intervals() if intervals is None else intervals,
        context_bars=base_context_bars() if context_bars is None else context_bars,
    )
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def base_schedule() -> ReplaySchedule:
    return schedule_with()


def base_validated_schedule() -> ValidatedSchedule:
    """The base schedule, already through `validate_schedule`."""
    return validate_schedule(base_request(), base_schedule())


def base_plays() -> tuple[PlayRequest, ...]:
    """Deliberately supplied out of canonical order."""
    return (
        PlayRequest(
            play_id="play-b",
            strategy="donchian_break",
            definition_digest=definition_digest(DEFINITION_BASE_B, PLAY_B_PARAMS),
            params=PLAY_B_PARAMS,
            priority=200,
        ),
        PlayRequest(
            play_id="play-a",
            strategy="sma_cross",
            definition_digest=definition_digest(DEFINITION_BASE_A, PLAY_A_PARAMS),
            params=PLAY_A_PARAMS,
            priority=100,
        ),
    )


def base_account() -> AccountPolicy:
    return AccountPolicy(
        starting_equity=100_000.0,
        risk_pct=0.01,
        max_open_positions=5,
        max_positions_per_play_symbol=1,
        settlement_model="cash_t1",
    )


def base_execution() -> ExecutionPolicy:
    return ExecutionPolicy(
        arithmetic_version="2",
        fill_mode="pessimistic",
        slippage=SlippageSpec(bps=2.0, min_per_share=0.01),
        fees=FeeSpec(per_fill=1.0, per_share=0.005),
        funding_order="play_priority_symbol_signal",
        missing_bar_policy="strict",
    )


def base_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        kind="equal_weight_request_symbols",
        symbol=None,
        weighting="equal",
        rebalance="never",
    )


def base_request(**overrides) -> PortfolioReplayRequest:
    """A complete request whose parent identities match their own formulas.

    Overrides apply before the identities are derived, so a varied request
    stays self-consistent. A test that wants a mismatched identity replaces it
    on the returned value.
    """
    draft = PortfolioReplayRequest(
        request_version=1,
        replay_id=f"replay:{PLACEHOLDER_DIGEST}",
        candidate_id=f"candidate:{PLACEHOLDER_DIGEST}",
        batch_id=BATCH_ID,
        registry_digest=REGISTRY_DIGEST,
        plays=base_plays(),
        symbols=("qqq", "SPY"),
        window=base_window(),
        schedule_identity=base_schedule().identity,
        ic_horizons=(1, 5, 20),
        ic_tail_end=ts("2026-11-27T18:00:00Z"),
        account=base_account(),
        execution=base_execution(),
        benchmark=base_benchmark(),
    )
    if overrides:
        draft = dataclasses.replace(draft, **overrides)
    named = dataclasses.replace(draft, candidate_id=expected_candidate_id(draft))
    return dataclasses.replace(named, replay_id=expected_replay_id(named))


def base_trade(request: PortfolioReplayRequest) -> PortfolioTrade:
    return PortfolioTrade(
        trade_id=trade_id(request.replay_id, "play-a", "SPY", 0),
        replay_id=request.replay_id,
        trade_ordinal=0,
        play_id="play-a",
        strategy="sma_cross",
        symbol="SPY",
        signal_ordinal=0,
        direction="long",
        qty=12,
        signal_ts=ts("2026-11-27T15:00:00Z"),
        entry_ts=ts("2026-11-27T15:00:00Z"),
        entry=100.5,
        exit_ts=ts("2026-11-27T16:00:00Z"),
        exit=103.0,
        initial_stop=98.0,
        final_stop=99.5,
        initial_target=106.0,
        final_target=106.0,
        gross_pnl=30.0,
        fees=2.0,
        net_pnl=28.0,
        r_multiple=1.12,
        mae=0.4,
        mfe=1.6,
        setup_tags=("trend", "pullback"),
        exit_reason=ExitReason.TARGET,
    )


def base_rejection(request: PortfolioReplayRequest) -> ReplayRejection:
    return ReplayRejection(
        rejection_id=rejection_id(
            request.replay_id, "play-b", "QQQ", 1, RejectionReason.UNSETTLED_CASH,
        ),
        replay_id=request.replay_id,
        rejection_ordinal=0,
        play_id="play-b",
        strategy="donchian_break",
        symbol="QQQ",
        signal_ordinal=1,
        signal_ts=ts("2026-11-27T15:00:00Z"),
        event_ts=ts("2026-11-27T15:15:00Z"),
        reason=RejectionReason.UNSETTLED_CASH,
        required_cash=1_212.0,
        available_cash=980.25,
        open_positions=1,
    )


def base_equity(request: PortfolioReplayRequest) -> tuple[EquityPoint, ...]:
    return (
        EquityPoint(
            replay_id=request.replay_id, ts=request.window.test_start, point_ordinal=0,
            settled_cash=100_000.0, unsettled_cash=0.0, short_collateral=0.0,
            positions_liquidation_value=0.0, portfolio_equity=100_000.0,
            gross_exposure=0.0, open_positions=0, benchmark_equity=100_000.0,
        ),
        EquityPoint(
            replay_id=request.replay_id, ts=request.window.test_end, point_ordinal=1,
            settled_cash=99_774.0, unsettled_cash=254.0, short_collateral=0.0,
            positions_liquidation_value=0.0, portfolio_equity=100_028.0,
            gross_exposure=0.0, open_positions=0, benchmark_equity=100_120.0,
        ),
    )


def _slice_for(
    request: PortfolioReplayRequest, play_id: str, strategy: str, symbol: str,
    *, signals: int, trades: int, rejection_counts: dict, net_pnl: float,
) -> PortfolioSlice:
    traded = trades > 0
    return PortfolioSlice(
        replay_id=request.replay_id,
        play_id=play_id,
        strategy=strategy,
        symbol=symbol,
        signals=signals,
        trades=trades,
        rejection_counts=rejection_counts,
        gross_profit=net_pnl if net_pnl > 0 else 0.0,
        gross_loss=-net_pnl if net_pnl < 0 else 0.0,
        pre_cost_pnl=net_pnl + (2.0 if traded else 0.0),
        net_pnl=net_pnl,
        fees=2.0 if traded else 0.0,
        win_rate=1.0 if traded else None,
        expectancy_r=1.12 if traded else None,
        ic=(
            IcEstimate(horizon_bars=1, correlation=0.1234, observations=12),
            IcEstimate(horizon_bars=5, correlation=None, observations=4),
            IcEstimate(horizon_bars=20, correlation=None, observations=0),
        ),
    )


def base_slices(request: PortfolioReplayRequest) -> tuple[PortfolioSlice, ...]:
    strategies = {"play-a": "sma_cross", "play-b": "donchian_break"}
    built = []
    for play in request.plays:
        for symbol in request.symbols:
            traded = play.play_id == "play-a" and symbol == "SPY"
            rejected = play.play_id == "play-b" and symbol == "QQQ"
            built.append(_slice_for(
                request, play.play_id, strategies[play.play_id], symbol,
                signals=1 if traded or rejected else 0,
                trades=1 if traded else 0,
                rejection_counts={RejectionReason.UNSETTLED_CASH: 1} if rejected else {},
                net_pnl=28.0 if traded else 0.0,
            ))
    return tuple(built)


def base_metrics() -> PortfolioMetrics:
    winner = TradeStats(
        n_trades=1, n_wins=1, win_rate=1.0, gross_profit=28.0, gross_loss=0.0,
        profit_factor=None, profit_factor_state="infinite", expectancy_r=1.12,
    )
    empty = TradeStats(
        n_trades=0, n_wins=0, win_rate=None, gross_profit=0.0, gross_loss=0.0,
        profit_factor=None, profit_factor_state="unavailable", expectancy_r=None,
    )
    return PortfolioMetrics(
        all_trades=winner, long_trades=winner, short_trades=empty,
        n_rejections=1, pre_cost_pnl=30.0, fees=2.0, net_pnl=28.0,
        starting_equity=100_000.0, ending_equity=100_028.0,
        total_return=0.00028, benchmark_return=0.0012,
        max_drawdown=0.0004, ulcer_index=0.0002, cagr=0.1, calmar=250.0,
        exposure_pct=0.15, avg_holding_hours=1.0,
        daily_n=1, daily_sum=0.00028, daily_sum_sq=7.84e-08,
        daily_sum_sq_down=0.0, daily_sum_cube=2.1952e-11,
        daily_sum_fourth=6.14656e-15,
        sharpe=None, sortino=None, psr=None, skew=None, kurtosis=None,
    )


def base_result(request: PortfolioReplayRequest | None = None) -> PortfolioReplayResult:
    """A complete result whose digest field carries its own recomputed digest."""
    request = base_request() if request is None else request
    draft = PortfolioReplayResult(
        request=request,
        arithmetic_version="2",
        fill_mode="pessimistic",
        schedule_identity=request.schedule_identity,
        result_digest=PLACEHOLDER_DIGEST,
        trades=(base_trade(request),),
        rejections=(base_rejection(request),),
        equity=base_equity(request),
        slices=base_slices(request),
        benchmark=BenchmarkResult(spec=request.benchmark, total_return=0.0012),
        metrics=base_metrics(),
    )
    return dataclasses.replace(draft, result_digest=result_digest(draft))


# ------------------------------------------------ daylight saving schedules

# Two more real XNYS pairs, each straddling one 2026 transition, because a
# boundary rule that reads a UTC hour is correct on exactly one side of a
# transition and silently wrong on the other. Every timestamp below is a
# literal: a fixture that computed these from the same arithmetic the
# validator uses could not fail when that arithmetic is wrong.
#
# 2026-03-08 moves New York from EST to EDT, so the Friday before opens at
# 14:30Z and the Monday after opens at 13:30Z. 2026-11-01 moves it back, so
# that Friday opens at 13:30Z and that Monday at 14:30Z. The same 12:00
# Eastern four-hour bucket is therefore labeled 17:00Z under EST and 16:00Z
# under EDT, and a daily bar labeled at Eastern midnight moves the same way.
SPRING_SESSION_ONE = date(2026, 3, 6)
SPRING_SESSION_TWO = date(2026, 3, 9)
FALL_SESSION_ONE = date(2026, 10, 30)
FALL_SESSION_TWO = date(2026, 11, 2)
FULL_SESSION_INTERVALS = 26


def session_intervals(
    session: date, first_open: pd.Timestamp, count: int,
) -> tuple[ScheduledBaseInterval, ...]:
    """One session's contiguous 15-minute regular-session grid."""
    return tuple(
        ScheduledBaseInterval(
            session_date=session,
            interval_ordinal=ordinal,
            open_ts=first_open + pd.Timedelta(minutes=15 * ordinal),
            close_ts=first_open + pd.Timedelta(minutes=15 * (ordinal + 1)),
        )
        for ordinal in range(count)
    )


def _dst_schedule(
    first: date, first_open: str, second: date, second_open: str,
    hour_labels: tuple[str, str], four_hour: tuple[tuple[str, str], ...],
    daily_label: str, daily_period: tuple[str, str],
    daily_available: str, daily_fresh: str,
) -> ReplaySchedule:
    intervals = (
        session_intervals(first, ts(first_open), FULL_SESSION_INTERVALS)
        + session_intervals(second, ts(second_open), FULL_SESSION_INTERVALS)
    )
    context = (
        ScheduledContextBar(
            timeframe="1h", session_date=first, label_ts=ts(hour_labels[0]),
            period_start=ts(hour_labels[0]),
            period_end=ts(hour_labels[0]) + pd.Timedelta(hours=1),
            available_at=ts(hour_labels[0]) + pd.Timedelta(hours=1),
            fresh_context_at=ts(hour_labels[0]) + pd.Timedelta(hours=1),
            source="fetched_left_edge",
        ),
        ScheduledContextBar(
            timeframe="1h", session_date=second, label_ts=ts(hour_labels[1]),
            period_start=ts(hour_labels[1]),
            period_end=ts(hour_labels[1]) + pd.Timedelta(hours=1),
            available_at=ts(hour_labels[1]) + pd.Timedelta(hours=1),
            fresh_context_at=ts(hour_labels[1]) + pd.Timedelta(hours=1),
            source="fetched_left_edge",
        ),
    ) + tuple(
        ScheduledContextBar(
            timeframe="4h", session_date=session, label_ts=ts(label),
            period_start=ts(label), period_end=ts(end),
            available_at=ts(end), fresh_context_at=ts(end),
            source="derived_1h_et_midnight",
        )
        for session, (label, end) in zip((first, second), four_hour, strict=True)
    ) + (
        ScheduledContextBar(
            timeframe="1d", session_date=first, label_ts=ts(daily_label),
            period_start=ts(daily_period[0]), period_end=ts(daily_period[1]),
            available_at=ts(daily_available), fresh_context_at=ts(daily_fresh),
            source="session_aligned",
        ),
    )
    draft = ReplaySchedule(
        identity=base_identity(), base_intervals=intervals, context_bars=context,
    )
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def spring_schedule() -> ReplaySchedule:
    """The 2026-03-06 EST session and the 2026-03-09 EDT session."""
    return _dst_schedule(
        SPRING_SESSION_ONE, "2026-03-06T14:30:00Z",
        SPRING_SESSION_TWO, "2026-03-09T13:30:00Z",
        hour_labels=("2026-03-06T14:00:00Z", "2026-03-09T13:00:00Z"),
        four_hour=(
            ("2026-03-06T17:00:00Z", "2026-03-06T21:00:00Z"),
            ("2026-03-09T16:00:00Z", "2026-03-09T20:00:00Z"),
        ),
        daily_label="2026-03-06T05:00:00Z",
        daily_period=("2026-03-06T14:30:00Z", "2026-03-06T21:00:00Z"),
        daily_available="2026-03-09T13:30:00Z",
        daily_fresh="2026-03-09T13:45:00Z",
    )


def fall_schedule() -> ReplaySchedule:
    """The 2026-10-30 EDT session and the 2026-11-02 EST session."""
    return _dst_schedule(
        FALL_SESSION_ONE, "2026-10-30T13:30:00Z",
        FALL_SESSION_TWO, "2026-11-02T14:30:00Z",
        hour_labels=("2026-10-30T13:00:00Z", "2026-11-02T14:00:00Z"),
        four_hour=(
            ("2026-10-30T16:00:00Z", "2026-10-30T20:00:00Z"),
            ("2026-11-02T17:00:00Z", "2026-11-02T21:00:00Z"),
        ),
        daily_label="2026-10-30T04:00:00Z",
        daily_period=("2026-10-30T13:30:00Z", "2026-10-30T20:00:00Z"),
        daily_available="2026-11-02T14:30:00Z",
        daily_fresh="2026-11-02T14:45:00Z",
    )


def spring_request() -> PortfolioReplayRequest:
    """Warm up on the EST session, trade the EDT session that follows."""
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-03-06T14:30:00Z"),
            train_end=ts("2026-03-09T13:30:00Z"),
            test_start=ts("2026-03-09T13:30:00Z"),
            test_end=ts("2026-03-09T20:00:00Z"),
        ),
        schedule_identity=spring_schedule().identity,
        ic_tail_end=ts("2026-03-09T20:00:00Z"),
    )


def fall_request() -> PortfolioReplayRequest:
    """Warm up on the EDT session, trade the EST session that follows."""
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-10-30T13:30:00Z"),
            train_end=ts("2026-11-02T14:30:00Z"),
            test_start=ts("2026-11-02T14:30:00Z"),
            test_end=ts("2026-11-02T21:00:00Z"),
        ),
        schedule_identity=fall_schedule().identity,
        ic_tail_end=ts("2026-11-02T21:00:00Z"),
    )


# ------------------------------------------------------------- bar fixtures


def base_dependencies() -> ReplayDependencies:
    """Every supported timeframe and no external symbol."""
    return ReplayDependencies(
        timeframes=("15m", "1h", "4h", "1d"), external_symbols=(),
    )


def bar_frame(labels, base: float = 100.0) -> pd.DataFrame:
    """A valid OHLCV frame on exactly `labels`, gently rising from `base`.

    Geometry is deliberately generous (the high clears every other price by
    0.15 and the low undercuts every other price by the same), so a test that
    breaks the geometry has to break it on purpose.
    """
    index = pd.DatetimeIndex(list(labels), tz="UTC", name="ts")
    opens = [base + 0.1 * step for step in range(len(index))]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [price + 0.2 for price in opens],
            "low": [price - 0.2 for price in opens],
            "close": [price + 0.05 for price in opens],
            "volume": [1_000.0 + step for step in range(len(index))],
        },
        index=index,
        dtype="float64",
    )


def scheduled_labels(schedule: ReplaySchedule, timeframe: str,
                     boundary: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    """Every scheduled label of `timeframe` that starts before `boundary`."""
    if timeframe == "15m":
        return tuple(row.open_ts for row in schedule.base_intervals
                     if row.open_ts < boundary)
    return tuple(row.label_ts for row in schedule.context_bars
                 if row.timeframe == timeframe and row.label_ts < boundary)


def frames_for(request: PortfolioReplayRequest, schedule: ReplaySchedule,
               dependencies: ReplayDependencies) -> dict:
    """One valid frame for every pair the request and dependencies declare."""
    built = {}
    for symbol in request.symbols:
        for timeframe in dependencies.timeframes:
            built[(symbol, timeframe)] = bar_frame(
                scheduled_labels(schedule, timeframe, request.ic_tail_end),
            )
    for symbol in dependencies.external_symbols:
        for timeframe in dependencies.timeframes:
            built.setdefault((symbol, timeframe), bar_frame(
                scheduled_labels(schedule, timeframe, request.window.test_end),
            ))
    benchmark = request.benchmark.symbol
    if benchmark is not None:
        built.setdefault((benchmark, "15m"), bar_frame(
            scheduled_labels(schedule, "15m", request.window.test_end),
        ))
    return built


def base_frames() -> dict:
    return frames_for(base_request(), base_schedule(), base_dependencies())


def base_bars() -> PortfolioBars:
    return PortfolioBars(base_frames())


def without_pair(frames: dict, symbol: str, timeframe: str) -> dict:
    kept = dict(frames)
    del kept[(symbol, timeframe)]
    return kept


# -------------------------------------------------------- registry fixtures

# Real RuleSpecs, not stubs. A definition whose factory cannot build a working
# strategy would let a registry test pass while the replay it feeds cannot
# run, and every timeframe below is one a dependency test then has to find.
SMA_CROSS_SPEC = {
    "version": 2, "name": "sma_cross", "timeframe": "1h",
    "long": {"all": [{"lhs": {"ind": "sma", "n": 10}, "op": "crosses_above",
                      "rhs": {"ind": "sma", "n": 30}}]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

DONCHIAN_SPEC = {
    "version": 2, "name": "donchian_break", "timeframe": "1d",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"ind": "highest", "n": 20}}]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

# A private play: the spec travels in params rather than being bound to the
# definition, and it reads a second timeframe through a node `tf`, so its
# declared dependencies cannot come from `timeframe` alone.
PRIVATE_RULES_SPEC = {
    "version": 2, "name": "private_rules", "timeframe": "1h",
    "long": {"all": [{"lhs": {"src": "close", "tf": "4h"}, "op": ">",
                      "rhs": {"ind": "sma", "n": 20}}]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

PRIVATE_RULES_PARAMS = {"spec": PRIVATE_RULES_SPEC}

# Block ids out of alphabetical order on purpose: members are built and voted
# in declared block order, never in sorted order.
COMPOSITE_PARAMS = {
    "spec": {
        "version": 1, "name": "combo",
        "blocks": {
            "b": {"strategy": "private_rules", "params": PRIVATE_RULES_PARAMS},
            "a": {"strategy": "sma_cross", "params": {}},
        },
        "long": {"all": ["a", "b"]},
        "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
                 "target": {"kind": "rr", "rr": 2.0}},
    },
}


def base_definitions(
    wrap: Callable[[StrategyDefinition], StrategyDefinition] | None = None,
) -> tuple[StrategyDefinition, ...]:
    """The bundle the portfolio tests replay, supplied out of canonical order.

    `wrap` decorates every definition as it is built, so a caller that wants
    counting or spying gets it on the composite's members too. Wrapping after
    the fact could not: the composite captures its member definitions when it
    is built, so a later replacement would never reach the members it builds.
    """
    hook = (lambda definition: definition) if wrap is None else wrap
    sma = hook(rules_definition("sma_cross", DEFINITION_BASE_A, spec=SMA_CROSS_SPEC))
    donchian = hook(rules_definition(
        "donchian_break", DEFINITION_BASE_B, spec=DONCHIAN_SPEC))
    private = hook(rules_definition("private_rules", DEFINITION_BASE_D))
    combo = hook(composite_definition(
        "combo", DEFINITION_BASE_C,
        members={"sma_cross": sma, "private_rules": private},
    ))
    return (donchian, combo, private, sma)


def strategy_registry() -> FrozenStrategyRegistry:
    return FrozenStrategyRegistry.from_definitions(base_definitions())


@dataclasses.dataclass
class FactoryCalls:
    """What the registry asked of the definitions, and what it got back."""

    factory_count: int = 0
    dependency_count: int = 0
    built: list = field(default_factory=list)


def counting_registry() -> tuple[FrozenStrategyRegistry, FactoryCalls]:
    """The base bundle with every factory and dependency call counted."""
    calls = FactoryCalls()
    definitions = base_definitions(lambda item: _counted(item, calls))
    return (FrozenStrategyRegistry.from_definitions(definitions), calls)


def _counted(definition: StrategyDefinition,
             calls: FactoryCalls) -> StrategyDefinition:
    def factory(params: Mapping):
        calls.factory_count += 1
        built = definition.factory(params)
        calls.built.append(built)
        return built

    def dependencies(params: Mapping):
        calls.dependency_count += 1
        return definition.dependencies(params)

    return dataclasses.replace(
        definition, factory=factory, dependencies=dependencies)


# ---------------------------------------------------------- ledger fixtures

# The ledger tests drive one account directly, so their timestamps come from
# the schedule above rather than from a replay loop. Interval 0 of the full
# session closes at the same instant interval 1 opens, which is exactly the
# chronology the architecture freezes: decide at a close, fill at the next
# scheduled open.
SIGNAL_TS = ts("2026-11-25T14:45:00Z")
ENTRY_TS = ts("2026-11-25T14:45:00Z")
CLOSE_TS = ts("2026-11-25T15:00:00Z")
# The half day that follows the Thanksgiving holiday. A credit raised on
# SESSION_ONE settles here, two calendar days later, because 2026-11-26 is not
# an exchange session.
NEXT_OPEN_TS = ts("2026-11-27T14:30:00Z")
LAST_CLOSE_TS = ts("2026-11-27T18:00:00Z")


def frictionless_execution() -> ExecutionPolicy:
    """Zero slippage and zero fees, so a ledger test can state one price.

    Cost behavior gets its own fixtures and its own tests. Mixing the two
    would mean every cash assertion carried a slippage term nobody was
    testing, and a broken fee model could hide inside a rounded expectation.
    """
    return dataclasses.replace(
        base_execution(),
        slippage=SlippageSpec(bps=0.0, min_per_share=0.0),
        fees=FeeSpec(per_fill=0.0, per_share=0.0),
    )


def ledger_request(
    cash: float = 100_000.0, *, risk_pct: float = 0.01,
    max_open_positions: int = 5, execution: ExecutionPolicy | None = None,
) -> PortfolioReplayRequest:
    """The base request with the account and cost policy a ledger test wants."""
    return base_request(
        account=AccountPolicy(
            starting_equity=cash,
            risk_pct=risk_pct,
            max_open_positions=max_open_positions,
            max_positions_per_play_symbol=1,
            settlement_model="cash_t1",
        ),
        execution=frictionless_execution() if execution is None else execution,
    )


def funded_ledger(cash: float = 100_000.0, **overrides) -> _Ledger:
    """One ledger holding `cash`, on the base schedule."""
    request = ledger_request(cash, **overrides)
    return _Ledger(request, validate_schedule(request, base_schedule()))


def entry_intent(
    ledger: _Ledger, symbol: str = "SPY", *, play_id: str = "play-a",
    strategy: str = "sma_cross", direction: str = "long",
    entry_ref: float = 100.0, stop: float = 98.6, target: float = 104.0,
    signal_ordinal: int = 0, signal_ts: pd.Timestamp = SIGNAL_TS,
    eligible_after: pd.Timestamp = ENTRY_TS,
) -> EntryIntent:
    """One intent belonging to `ledger`, long from 100.0 by default.

    The default geometry is not arbitrary. A 1.4 stop distance sizes to seven
    shares against a 1,000 account at one percent risk, which is the smallest
    contention the ledger tests need: two of them cannot both fit.
    """
    return EntryIntent(
        replay_id=ledger.replay_id,
        play_id=play_id,
        strategy=strategy,
        symbol=symbol,
        signal=Signal(
            symbol=symbol.upper(), direction=direction, entry_ref=entry_ref,
            stop=stop, target=target, confidence=0.7,
            setup_tags=("trend",), rationale="fixture",
        ),
        signal_ordinal=signal_ordinal,
        signal_ts=signal_ts,
        order_type="market_next_open",
        eligible_after=eligible_after,
        expires_after_intervals=1,
    )


def opened_position(
    ledger: _Ledger, symbol: str = "SPY", *, raw_open: float = 100.0,
    frozen_equity: float | None = None, entry_ts: pd.Timestamp = ENTRY_TS,
    **intent_fields,
) -> tuple[tuple[str, str], EntryProposal]:
    """Propose, reserve, and fill one position, asserting the reservation held.

    Returns the position key and the proposal it was filled from, so a caller
    can assert against the exact quantity and cash the ledger committed.
    """
    intent = entry_intent(ledger, symbol, **intent_fields)
    equity = ledger.settled_cash if frozen_equity is None else frozen_equity
    proposal = ledger.propose(intent, raw_open, equity)
    reservation = ledger.reserve(proposal, equity)
    assert reservation.accepted is True, reservation.reason
    ledger.open(proposal, entry_ts)
    return (proposal.key, proposal)


def replay_ambiguous_long(
    *, stop: float, target: float, low: float, high: float,
    raw_open: float = 100.0, gap_open: float | None = None,
) -> PortfolioTrade:
    """One long, opened at `raw_open`, exited by one bar's own geometry.

    The bar is the whole point: `low` and `high` are chosen by the caller to
    reach one level, both, or neither, and the trade that comes back reports
    which one the ledger settled at. `gap_open` gives the exit bar an opening
    print away from the entry, which is how a gap case differs from an
    intrabar one.
    """
    ledger = funded_ledger(100_000.0)
    key, _ = opened_position(
        ledger, raw_open=raw_open, entry_ref=raw_open, stop=stop, target=target,
    )
    bar_open = raw_open if gap_open is None else gap_open
    ledger.observe(key, high, low)
    hit = ledger.protective_exit(key, bar_open, high, low)
    assert hit is not None, "the fixture bar reaches no protective level"
    fill, reason = hit
    return ledger.close(
        key, fill, CLOSE_TS, reason, ledger.exit_fee(ledger.view(key).qty),
    )
