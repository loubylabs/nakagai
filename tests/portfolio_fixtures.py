"""Shared builders for the portfolio replay tests.

Every builder returns a complete, contract-valid value. A test that needs an
invalid value asks for the valid one and replaces exactly the field under
test, so one failure names one cause. Parent identities are derived here the
way platform derives them: build the draft, ask core for the candidate
identity, then ask core for the replay identity.

`replay_parts` at the foot of the module runs a real replay: `replay_inputs`
assembles the four values `run_portfolio` takes, and the replay itself goes
through the public path's own two steps, so `replay_fixture`, `replay_curve`,
`replay_metrics`, `replay_ic`, and `replay_result` are five views of one drive
through `run_portfolio`. It is a thin assembler over the real core values and
never a second replay implementation.
"""

import dataclasses
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from datetime import date, timedelta

import pandas as pd

from nakagai.engine.bars import (
    PortfolioBars,
    ReplayDependencies,
    prepare_portfolio_bars,
    _ValidatedPortfolioBars,
)
from nakagai.engine.canonical import (
    canonical_replay_bytes,
    definition_digest,
    expected_candidate_id,
    expected_replay_id,
    rejection_id,
    result_digest,
    schedule_digest,
    trade_id,
    _projection,
)
from nakagai.engine.benchmark import ReplayCurve
from nakagai.engine.execution import ReplayEvents
from nakagai.engine.metrics import _portfolio_metrics, _slice_accumulators
from nakagai.engine.portfolio import EntryProposal, PositionKey, _Ledger
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
    ManagementDecision,
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
    StrategyDependencies,
    composite_definition,
    rules_definition,
    spec_base_digest,
    vocabulary_digest,
)
from nakagai.engine.replay import _finalize_result, _replay, run_portfolio
from nakagai.engine.schedule import ValidatedSchedule, validate_schedule
from nakagai.strategies.base import HOLD, MarketContext, Strategy
from nakagai.strategies.rules.vocabulary import core_vocabulary

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


def single_symbol_benchmark(symbol: str) -> BenchmarkSpec:
    """The declared alternative: one named symbol, whether or not it trades."""
    return BenchmarkSpec(
        kind="single_symbol", symbol=symbol, weighting="equal", rebalance="never",
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


# ------------------------------------ a transition INSIDE a four-hour bucket

# The two pairs above straddle a transition across a weekend, which is what
# XNYS does: the clocks move at 02:00 on a Sunday, so no exchange session and
# no bucket derived from one ever spans the change. Both label conventions
# therefore still satisfy `label + 4h == period_end` on every row, which makes
# those schedules silent about the rule the architecture actually freezes: a
# four-hour bucket is four EASTERN WALL-CLOCK hours, not four absolute ones.
#
# The only bucket that can span a US transition is the one anchored at Eastern
# midnight, on the Sunday itself. So these two carry a continuous session on
# the transition day. That is an ordinary core input rather than a liberty:
# core takes the schedule as data and never consults an installed calendar, and
# a continuously traded calendar is simply the only one that can put a base
# interval on the far side of the change from its own bucket's label.
#
# Fall back, 2026-11-01: 00:00 Eastern is 04:00Z and 04:00 Eastern is 09:00Z,
# five absolute hours later, so `label + 4h` lands an hour SHORT of the
# bucket's own end. Spring forward, 2026-03-08: 00:00 Eastern is 05:00Z and
# 04:00 Eastern is 08:00Z, three absolute hours, so `label + 4h` lands an hour
# PAST it. Every timestamp below is a literal, for the same reason the pairs
# above are: a fixture computed from the arithmetic under test cannot fail.
SPRING_BUCKET_SESSION = date(2026, 3, 8)
FALL_BUCKET_SESSION = date(2026, 11, 1)


def _bucket_schedule(
    session: date, first_open: str, count: int,
    buckets: tuple[tuple[str, str], ...],
) -> ReplaySchedule:
    """One continuous session and the four-hour buckets it derives.

    Only 4h context bars, because the base timeframe and the spanning bucket
    are the whole subject. A schedule carries exactly the timeframes a replay
    declares, so leaving 1h and 1d out is a smaller fixture rather than a
    partial one.
    """
    intervals = session_intervals(session, ts(first_open), count)
    context = tuple(
        ScheduledContextBar(
            timeframe="4h", session_date=session, label_ts=ts(start),
            period_start=ts(start), period_end=ts(end),
            available_at=ts(end), fresh_context_at=ts(end),
            source="derived_1h_et_midnight",
        )
        for start, end in buckets
    )
    draft = ReplaySchedule(
        identity=base_identity(), base_intervals=intervals, context_bars=context,
    )
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def spring_bucket_schedule() -> ReplaySchedule:
    """2026-03-08, 00:00 through 08:00 Eastern: 05:00Z through 12:00Z."""
    return _bucket_schedule(
        SPRING_BUCKET_SESSION, "2026-03-08T05:00:00Z", 28,
        (("2026-03-08T05:00:00Z", "2026-03-08T08:00:00Z"),
         ("2026-03-08T08:00:00Z", "2026-03-08T12:00:00Z")),
    )


def fall_bucket_schedule() -> ReplaySchedule:
    """2026-11-01, 00:00 through 08:00 Eastern: 04:00Z through 13:00Z."""
    return _bucket_schedule(
        FALL_BUCKET_SESSION, "2026-11-01T04:00:00Z", 36,
        (("2026-11-01T04:00:00Z", "2026-11-01T09:00:00Z"),
         ("2026-11-01T09:00:00Z", "2026-11-01T13:00:00Z")),
    )


def spring_bucket_request() -> PortfolioReplayRequest:
    """Trade from 07:00Z, so the 08:00Z bucket end lands inside the test."""
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-03-08T05:00:00Z"),
            train_end=ts("2026-03-08T07:00:00Z"),
            test_start=ts("2026-03-08T07:00:00Z"),
            test_end=ts("2026-03-08T12:00:00Z"),
        ),
        schedule_identity=spring_bucket_schedule().identity,
        ic_tail_end=ts("2026-03-08T12:00:00Z"),
    )


def fall_bucket_request() -> PortfolioReplayRequest:
    """Trade from 08:00Z, so the 09:00Z bucket end lands inside the test."""
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-11-01T04:00:00Z"),
            train_end=ts("2026-11-01T08:00:00Z"),
            test_start=ts("2026-11-01T08:00:00Z"),
            test_end=ts("2026-11-01T13:00:00Z"),
        ),
        schedule_identity=fall_bucket_schedule().identity,
        ic_tail_end=ts("2026-11-01T13:00:00Z"),
    )


def bucket_dependencies() -> ReplayDependencies:
    """The base timeframe and the four-hour bucket, and nothing else."""
    return ReplayDependencies(timeframes=("15m", "4h"), external_symbols=())


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


def ramp_frame(labels, base: float = 100.0) -> pd.DataFrame:
    """One flat opening print and a close that widens away from it, row by row.

    Built so a REAL spec margin is hand-derivable. `close - open` rises
    strictly with the row, so a graded `close > open` ranks the rows in their
    own order, while the forward return `0.01k / close[t]` falls strictly with
    the row because the close climbs. The two rankings are exact reverses, so
    the rank correlation of that spec over this frame is exactly -1 at every
    horizon.
    """
    index = pd.DatetimeIndex(list(labels), tz="UTC", name="ts")
    opens = [base] * len(index)
    closes = [base + 0.01 * step for step in range(len(index))]
    return pd.DataFrame(
        {
            "open": opens,
            "high": [price + 0.2 for price in closes],
            "low": [base - 0.2] * len(index),
            "close": closes,
            "volume": [1_000.0 + step for step in range(len(index))],
        },
        index=index,
        dtype="float64",
    )


def flat_frame(labels, price: float = 100.0) -> pd.DataFrame:
    """A frame whose every bar is one flat print at `price`.

    Nothing moves, so a protective level is reached only where a test puts a bar
    that reaches it, and every quantity, cash figure, and PnL in a replay test
    is an exact literal a reader can check by hand.
    """
    index = pd.DatetimeIndex(list(labels), tz="UTC", name="ts")
    prices = [price] * len(index)
    return pd.DataFrame(
        {
            "open": prices, "high": prices, "low": prices, "close": prices,
            "volume": [1_000.0] * len(index),
        },
        index=index,
        dtype="float64",
    )


def frames_for(request: PortfolioReplayRequest, schedule: ReplaySchedule,
               dependencies: ReplayDependencies, *,
               build: Callable[[tuple], pd.DataFrame] = bar_frame) -> dict:
    """One valid frame for every pair the request and dependencies declare."""
    built = {}
    for symbol in request.symbols:
        for timeframe in dependencies.timeframes:
            built[(symbol, timeframe)] = build(
                scheduled_labels(schedule, timeframe, request.ic_tail_end),
            )
    for symbol in dependencies.external_symbols:
        for timeframe in dependencies.timeframes:
            built.setdefault((symbol, timeframe), build(
                scheduled_labels(schedule, timeframe, request.window.test_end),
            ))
    benchmark = request.benchmark.symbol
    if benchmark is not None:
        built.setdefault((benchmark, "15m"), build(
            scheduled_labels(schedule, "15m", request.window.test_end),
        ))
    return built


def base_frames() -> dict:
    return frames_for(base_request(), base_schedule(), base_dependencies())


def base_bars() -> PortfolioBars:
    return PortfolioBars(base_frames())


def prepared_for(
    request: PortfolioReplayRequest, schedule: ReplaySchedule,
    dependencies: ReplayDependencies, *,
    build: Callable[[tuple], pd.DataFrame] = bar_frame,
) -> tuple[ValidatedSchedule, _ValidatedPortfolioBars]:
    """One validated schedule and the engine-owned bars that match it.

    The two travel together everywhere: a prepared frame's rows are the
    schedule's labels, in the schedule's order, so a caller holding one without
    the other cannot read a row. Built through the real doors, so a fixture
    cannot hand a context module bars the bar preflight would have refused.
    """
    validated = validate_schedule(request, schedule)
    frames = frames_for(request, schedule, dependencies, build=build)
    return (validated, prepare_portfolio_bars(
        request, PortfolioBars(frames), validated, dependencies))


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

# A spec evaluated on the BASE timeframe, so its IC observations are the
# scheduled base closes rather than a handful of context freshness instants.
# The wiring test needs a real spec on the axis that carries enough
# observations to reach a correlation at all.
#
# Two raw prices and no indicator, on purpose. An indicator leaves its first
# rows undefined, and an ungraded observation drops its own pair, so a moving
# average would quietly cost the first observations and make an exact count
# impossible to state. Over `ramp_frame` this spec's margin rises strictly
# with the row, which is what makes its coefficient an exact number.
IC_RULES_SPEC = {
    "version": 2, "name": "ic_rules", "timeframe": "15m",
    "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                      "rhs": {"src": "open"}}]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 2.0}},
}

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


def counting_definitions(
    calls: FactoryCalls,
) -> Callable[[StrategyDefinition], StrategyDefinition]:
    """A `wrap` for `replay_fixture` that counts and collects every runtime."""
    return lambda definition: _counted(definition, calls)


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
    hit = ledger.protective_exit(key, bar_open, high, low)
    assert hit is not None, "the fixture bar reaches no protective level"
    fill, reason = hit
    _fold_exit_excursion(
        ledger, key, reason, bar_open=bar_open, low=low, stop=stop, target=target,
    )
    return ledger.close(
        key, fill, CLOSE_TS, reason, ledger.exit_fee(ledger.view(key).qty),
    )


def _fold_exit_excursion(
    ledger: _Ledger, key: tuple[str, str], reason: ExitReason, *,
    bar_open: float, low: float, stop: float, target: float,
) -> None:
    """What one long exit is allowed to have observed, for this bar.

    The rules are the architecture's, and they are causal rather than
    generous. A gap exit folds only the raw open, because that print is the
    first and last observation before the position left. A stop exit folds the
    open and the stop and never the favorable extreme, because OHLC cannot
    prove that extreme came first, and that holds whether or not the target
    was also inside the bar. Only a target exit folds an extreme, the adverse
    one, because pessimistic ordering already assumed it came first.

    A surviving position is the other case entirely: it folds the bar's full
    high and low, through `_Ledger.observe`, because it was live throughout.
    That belongs to the replay chronology rather than here.
    """
    ledger.observe(key, bar_open, bar_open)
    if reason is ExitReason.STOP:
        ledger.observe(key, stop, stop)
    elif reason is ExitReason.TARGET:
        ledger.observe(key, target, target)
        ledger.observe(key, low, low)


# ---------------------------------------------------------- replay fixtures

# `replay_fixture` runs a REAL replay. `replay_inputs` assembles the four values
# `run_portfolio` takes and the drive goes through the public path's own two
# steps, so it is a thin assembler rather than a second engine.

# The default window trades the 2026-11-27 half day, whose first interval opens
# at 14:30Z and whose last closes at 18:00Z.
FIRST_CLOSE = ts("2026-11-27T14:45:00Z")

# Only the base timeframe, which is what a replay reads to fill, mark, and
# settle. A test that needs a context timeframe asks for one.
BASE_ONLY = ReplayDependencies(timeframes=("15m",), external_symbols=())

# Two plays under ONE priority, so the two canonical orders are genuinely
# different: signal ordinals run play-major and funding runs symbol-major.
DEFAULT_PLAY_IDS = ("play-a", "play-b")

_UNSET = object()


@dataclasses.dataclass(frozen=True)
class SignalPlan:
    """One scripted signal: which symbol, which deciding close, what geometry.

    The entry reference is not named here. It is the deciding raw close the
    context carries, which is what the strategy boundary demands of a real
    signal, so a scripted play cannot claim a price it was not looking at.
    """

    symbol: str
    at: pd.Timestamp
    stop: float
    target: float
    direction: str = "long"


@dataclasses.dataclass(frozen=True)
class ManagePlan:
    """One scripted management decision at one close."""

    symbol: str
    at: pd.Timestamp
    action: str = "hold"
    stop: float | None = None
    target: float | None = None


@dataclasses.dataclass(frozen=True)
class BarPlan:
    """One base bar replaced, so a test can put a level where it wants it."""

    symbol: str
    at: pd.Timestamp
    open: float
    high: float
    low: float
    close: float


@dataclasses.dataclass(frozen=True)
class StrategyCall:
    """One call the replay made into one runtime, and what it could see.

    The call log is the chronology as the strategies experienced it, which is
    the only place an ordering between two different operations at one instant
    is observable at all.
    """

    operation: str
    play_id: str
    symbol: str
    now: pd.Timestamp
    visible: tuple[tuple[str, int], ...]
    last_base_label: pd.Timestamp | None
    # This runtime's OWN call count, counting from one. It is the only field
    # here that is per instance rather than per event, so it is what tells two
    # runtimes apart from one shared between them: two isolated runtimes each
    # start at one, and one shared runtime counts straight through.
    sequence: int = 0
    # The first base open this call could see. A price rather than a count, so
    # a play that writes into its own context is observable from another
    # play's reading of the same bar.
    first_base_open: float | None = None


@dataclasses.dataclass(frozen=True)
class ScriptedPlay:
    """One play's whole behavior, as data.

    The failure knobs raise ordinary exceptions from ordinary strategy code, so
    the replay's error boundary is exercised the way a real defect would.
    """

    play_id: str
    priority: int = 100
    signals: tuple[SignalPlan, ...] = ()
    manages: tuple[ManagePlan, ...] = ()
    on_bar_returns: object = _UNSET
    manage_returns: object = _UNSET
    on_bar_raises: str | None = None
    manage_raises: str | None = None
    factory_raises: str | None = None
    raises_at: pd.Timestamp | None = None
    # What the built runtime CALLS itself, when that is not the name its
    # definition is registered under. Only a test that wants the two to
    # disagree sets it; a real definition names its own runtime.
    runtime_name: str | None = None
    # A careless play: write this price into row zero of the driving frame on
    # every call, IN PLACE. Copy-on-write sends that write to a copy of the
    # engine's data, so what it reaches is whatever else holds the same frame
    # object, which is the only way one play's write can become another's.
    writes_first_open: float | None = None
    # The graded factor, as data. `ic_timeframe` is the observation axis and
    # its presence is what makes the definition graded at all; a play that
    # leaves it None carries no factor, which is the composite's case and the
    # default. `ic_margin` grades one observation from its position in the
    # selected series and the instant it was taken at.
    ic_timeframe: str | None = None
    ic_margin: Callable[[int, pd.Timestamp], float | None] | None = None
    # A raw value to hand back instead of one margin per timestamp, so the
    # lens's own contract checks have something to refuse.
    ic_returns: object = _UNSET
    ic_raises: str | None = None


class ScriptedStrategy(Strategy):
    """A strategy that emits exactly what its script names, and nothing else.

    A test double for the strategy CONTRACT and never for the replay: every
    value it returns still crosses `validate_signal_sequence` or
    `validate_management_decision` on the way back in, so a scripted play cannot
    smuggle a signal past a gate a real play has to pass.
    """

    name = "scripted"

    def __init__(self, params: dict | None = None, *, play: ScriptedPlay,
                 name: str, calls: list | None = None):
        super().__init__(params)
        self.name = name
        self._play = play
        self._calls = calls
        # Per instance and never shared: the counter a leaked runtime would
        # carry across the play symbol it does not belong to.
        self._sequence = 0

    def on_bar(self, ctx: MarketContext):
        self._record("on_bar", ctx)
        if self._play.writes_first_open is not None:
            ctx.bars[ctx.tfs.driving].iloc[0, 0] = self._play.writes_first_open
        self._raise_if_due(self._play.on_bar_raises, ctx.now)
        if self._play.on_bar_returns is not _UNSET:
            return self._play.on_bar_returns
        close = float(ctx.driving_bars["close"].iloc[-1])
        return tuple(
            Signal(symbol=ctx.symbol, direction=plan.direction, entry_ref=close,
                   stop=plan.stop, target=plan.target, confidence=0.5,
                   setup_tags=("scripted",), rationale=f"{self.name} plan")
            for plan in self._play.signals
            if plan.symbol == ctx.symbol and plan.at == ctx.now
        )

    def manage(self, position, ctx: MarketContext) -> ManagementDecision:
        self._record("manage", ctx)
        self._raise_if_due(self._play.manage_raises, ctx.now)
        if self._play.manage_returns is not _UNSET:
            return self._play.manage_returns
        for plan in self._play.manages:
            if plan.symbol == ctx.symbol and plan.at == ctx.now:
                return ManagementDecision(action=plan.action, stop=plan.stop,
                                          target=plan.target)
        return HOLD

    def _raise_if_due(self, message: str | None, now: pd.Timestamp) -> None:
        if message is None:
            return
        if self._play.raises_at is None or self._play.raises_at == now:
            raise RuntimeError(message)

    def _record(self, operation: str, ctx: MarketContext) -> None:
        # Counted whether or not anyone is listening, so a replay driven
        # without a call log still isolates the same way one driven with it does.
        self._sequence += 1
        if self._calls is None:
            return
        base = ctx.bars[ctx.tfs.driving]
        self._calls.append(StrategyCall(
            operation=operation, play_id=self._play.play_id, symbol=ctx.symbol,
            now=ctx.now,
            visible=tuple((tf, len(ctx.bars[tf])) for tf in ctx.tfs.all),
            last_base_label=base.index[-1] if len(base.index) else None,
            sequence=self._sequence,
            first_base_open=float(base["open"].iloc[0]) if len(base.index) else None,
        ))


def scripted_name(play_id: str) -> str:
    return f"scripted-{play_id}"


def scripted_digest(play_id: str) -> str:
    """One stable base digest per play, derived from its name.

    Derived rather than positional, so reversing the order the plays are
    supplied in cannot quietly hand one play another's definition.
    """
    return hashlib.sha256(scripted_name(play_id).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class FactorCall:
    """One question the IC lens asked a graded factor, and what it could see.

    The causal view is the whole subject, so the row counts and the last label
    of each frame are recorded rather than the frames themselves: a tail bar
    inside the view moves `rows` and `last_label`, and nothing else here can.
    """

    play_id: str
    symbol: str
    timeframe: str
    timestamps: tuple[pd.Timestamp, ...]
    labels: tuple[pd.Timestamp, ...]
    rows: tuple[tuple[str, int], ...]
    last_label: pd.Timestamp | None
    # The highest close in the causal view of the axis frame. A price rather
    # than a count, so one symbol's bars are distinguishable from another's.
    highest_close: float | None


def scripted_definition(play: ScriptedPlay, *, timeframes: tuple[str, ...] = ("15m",),
                        external_symbols: tuple[str, ...] = (),
                        calls: list | None = None,
                        factor_calls: list | None = None) -> StrategyDefinition:
    """One definition per scripted play, built the way a real one is.

    Both halves of the closure are parameters, and `replay_fixture` feeds them
    the same `ReplayDependencies` it prepares bars from. A definition that
    declared less than the fixture hydrated would be invisible today and would
    surface at the C11 re-point, where `dependencies_for` derives the closure
    from these declarations instead: the bar set would change and the golden
    bytes would move.

    The graded factor arrives with its axis or not at all, which is the
    definition's own invariant: a play that names no `ic_timeframe` carries no
    factor and its slice reports no measurement.
    """
    name = scripted_name(play.play_id)
    digest = vocabulary_digest(core_vocabulary)

    def dependencies(params: Mapping) -> StrategyDependencies:
        return StrategyDependencies(timeframes=tuple(timeframes),
                                    external_symbols=tuple(external_symbols),
                                    vocabulary_digest=digest)

    def factory(params: Mapping) -> Strategy:
        if play.factory_raises is not None:
            raise RuntimeError(play.factory_raises)
        return ScriptedStrategy(params, play=play, calls=calls,
                                name=play.runtime_name or name)

    def ic_timeframe(params: Mapping) -> str:
        return play.ic_timeframe

    def ic_factor(params: Mapping, symbol: str, bars, timestamps: tuple):
        if factor_calls is not None:
            factor_calls.append(_factor_call(play, symbol, bars, timestamps))
        if play.ic_raises is not None:
            raise RuntimeError(play.ic_raises)
        if play.ic_returns is not _UNSET:
            return play.ic_returns
        grade = play.ic_margin or (lambda position, at: float(position))
        return tuple(grade(position, at) for position, at in enumerate(timestamps))

    graded = play.ic_timeframe is not None
    return StrategyDefinition(
        name=name, definition_digest=scripted_digest(play.play_id),
        dependencies=dependencies, factory=factory,
        ic_factor=ic_factor if graded else None,
        ic_timeframe=ic_timeframe if graded else None,
    )


def _factor_call(play: ScriptedPlay, symbol: str, bars,
                 timestamps: tuple) -> FactorCall:
    frame = bars.frame(bars.timeframe)
    return FactorCall(
        play_id=play.play_id,
        symbol=symbol,
        timeframe=bars.timeframe,
        timestamps=tuple(timestamps),
        labels=tuple(bars.labels),
        rows=tuple((tf, len(bars.frame(tf))) for tf in sorted(bars.frames)),
        last_label=frame.index[-1] if len(frame.index) else None,
        highest_close=float(frame["close"].max()) if len(frame.index) else None,
    )


def replay_account(equity: float = 100_000.0, *, risk_pct: float = 0.001,
                   max_open_positions: int = 5) -> AccountPolicy:
    """An account that funds four flat-price fills at once.

    One tenth of a percent of a hundred thousand is a hundred dollars of risk,
    which over a one dollar stop is a hundred shares at a hundred dollars: ten
    thousand per fill, forty thousand for the default four.
    """
    return AccountPolicy(
        starting_equity=equity, risk_pct=risk_pct,
        max_open_positions=max_open_positions,
        max_positions_per_play_symbol=1, settlement_model="cash_t1",
    )


def default_plays(*, at: pd.Timestamp = FIRST_CLOSE, stop: float = 99.0,
                  target: float = 103.0) -> tuple[ScriptedPlay, ...]:
    """Two plays that both signal on both symbols at one close."""
    signals = tuple(SignalPlan(symbol=symbol, at=at, stop=stop, target=target)
                    for symbol in ("QQQ", "SPY"))
    return tuple(ScriptedPlay(play_id=play_id, priority=100, signals=signals)
                 for play_id in DEFAULT_PLAY_IDS)


def with_base_bars(frames: dict, plans) -> dict:
    """The same frames with named base bars replaced, one plan per bar.

    A plan naming a label the schedule never carries is a mistake in the test
    rather than a new bar, so it raises here instead of silently extending an
    index the bar preflight would then refuse for a different reason.
    """
    edited = dict(frames)
    for plan in plans:
        key = (plan.symbol.upper(), "15m")
        frame = edited[key].copy(deep=True)
        if plan.at not in frame.index:
            raise AssertionError(f"the schedule carries no base bar at {plan.at}")
        frame.loc[plan.at, ["open", "high", "low", "close"]] = [
            plan.open, plan.high, plan.low, plan.close,
        ]
        edited[key] = frame
    return edited


@dataclasses.dataclass(frozen=True)
class ReplayInputs:
    """The four values `run_portfolio` takes, assembled and matched.

    Separated from the replay itself so a test can edit one input after the
    assembly and before the door: withholding a bar, breaking a frame's
    geometry, or naming another schedule are all things a caller can do and the
    public boundary has to refuse.
    """

    request: PortfolioReplayRequest
    bars: PortfolioBars
    registry: FrozenStrategyRegistry
    schedule: ReplaySchedule


@dataclasses.dataclass(frozen=True)
class ReplayParts:
    """One replay's validated inputs, its chronology, and its finished result.

    Both views come from ONE drive through the public path: `_replay` performs
    every preflight and runs the loop, and `_finalize_result` is the same
    assembly `run_portfolio` performs on top of it. Nothing here is a second
    engine, and a test asserting on `events` and a test asserting on `result`
    are looking at one replay from two distances.
    """

    request: PortfolioReplayRequest
    schedule: ValidatedSchedule
    prepared: _ValidatedPortfolioBars
    events: ReplayEvents
    registry: FrozenStrategyRegistry
    result: PortfolioReplayResult


def replay_inputs(
    *,
    plays: tuple[ScriptedPlay, ...] | None = None,
    symbol_order: tuple[str, ...] = ("SPY", "QQQ"),
    reverse_param_keys: bool = False,
    reverse_plays: bool = False,
    signal_at: pd.Timestamp = FIRST_CLOSE,
    signal_stop: float = 99.0,
    signal_target: float = 103.0,
    bars: tuple[BarPlan, ...] = (),
    drop_pairs: tuple[tuple[str, str], ...] = (),
    price: float = 100.0,
    account: AccountPolicy | None = None,
    execution: ExecutionPolicy | None = None,
    benchmark: BenchmarkSpec | None = None,
    dependencies: ReplayDependencies | None = None,
    window: ReplayWindow | None = None,
    build: Callable[[tuple], pd.DataFrame] | None = None,
    wrap: Callable[[StrategyDefinition], StrategyDefinition] | None = None,
    calls: list | None = None,
    factor_calls: list | None = None,
) -> ReplayInputs:
    """The request, bars, registry, and schedule for one scripted replay.

    Three knobs vary orderings the canonical contract is supposed to ignore, and
    they exist for the determinism tests: `symbol_order` supplies the request's
    symbols and its bar mapping in either order, `reverse_plays` supplies the
    plays and the registry's definitions in either order, and
    `reverse_param_keys` supplies each play's parameter object in either order.

    `drop_pairs` withholds a frame the request declares, which is what proves a
    preflight refusal lands before any play is constructed.

    `dependencies` is what the scripted definitions DECLARE, and the frames are
    built from the same closure. `run_portfolio` derives the closure from those
    declarations, so the two agree by construction rather than by a fixture
    telling the bar preflight one thing and the loop another.

    `wrap` decorates every definition on its way into the registry, the way
    `base_definitions` does, so a caller that wants each construction counted
    or each runtime collected gets it without a second assembler here.

    `build` supplies the bar geometry. The default is one flat print at
    `price`, which is what makes every cash figure an exact literal; a caller
    measuring a correlation asks for a frame whose closes move, because a flat
    series has one distinct forward return and no rank to correlate.
    """
    hook = (lambda definition: definition) if wrap is None else wrap
    scripted = (default_plays(at=signal_at, stop=signal_stop, target=signal_target)
                if plays is None else tuple(plays))
    declared = BASE_ONLY if dependencies is None else dependencies
    params = scripted_params(reverse_param_keys)
    supplied = tuple(reversed(scripted)) if reverse_plays else scripted
    request = base_request(
        plays=tuple(scripted_play_request(play, params) for play in supplied),
        symbols=tuple(symbol_order),
        account=replay_account() if account is None else account,
        execution=frictionless_execution() if execution is None else execution,
        **({} if benchmark is None else {"benchmark": benchmark}),
        **({} if window is None else {"window": window}),
    )
    schedule = base_schedule()
    geometry = (lambda labels: flat_frame(labels, price)) if build is None else build
    frames = with_base_bars(
        frames_for(request, schedule, declared, build=geometry),
        bars,
    )
    for symbol, timeframe in drop_pairs:
        frames = without_pair(frames, symbol, timeframe)
    if reverse_param_keys:
        frames = dict(reversed(list(frames.items())))
    registry = FrozenStrategyRegistry.from_definitions(tuple(
        hook(scripted_definition(
            play, timeframes=declared.timeframes,
            external_symbols=declared.external_symbols, calls=calls,
            factor_calls=factor_calls))
        for play in supplied
    ))
    return ReplayInputs(request=request, bars=PortfolioBars(frames),
                        registry=registry, schedule=schedule)


def replay_parts(**overrides) -> ReplayParts:
    """One complete replay through the public entry point's own composition."""
    inputs = replay_inputs(**overrides)
    run = _replay(inputs.request, inputs.bars, inputs.registry, inputs.schedule)
    return ReplayParts(request=run.request, schedule=run.schedule,
                       prepared=run.prepared, events=run.events,
                       registry=run.registry, result=_finalize_result(run))


def replay_fixture(**overrides) -> ReplayEvents:
    """What one replay's chronology produced, for a test about the loop."""
    return replay_parts(**overrides).events


def replay_result(**overrides) -> PortfolioReplayResult:
    """The finished result of one replay, exactly as a caller receives it."""
    return replay_parts(**overrides).result


def replay_curve(**overrides) -> ReplayCurve:
    """The equity and benchmark series the finished result carries."""
    result = replay_parts(**overrides).result
    return ReplayCurve(equity=result.equity, benchmark=result.benchmark)


@dataclasses.dataclass(frozen=True)
class ReplayLenses:
    """One replay, plus every lens taken over it.

    The metric lens reads the curve lens, so a test that asserts on a metric
    can also point at the equity point it came from without assembling the
    replay twice.
    """

    request: PortfolioReplayRequest
    schedule: ValidatedSchedule
    events: ReplayEvents
    curve: ReplayCurve
    metrics: PortfolioMetrics
    slices: Mapping[PositionKey, object]


def replay_metrics(**overrides) -> ReplayLenses:
    """The metrics the finished result carries, plus the accumulators behind it.

    `metrics` and `curve` are read off the result rather than recomputed, so a
    test asserting on them is asserting on what a caller receives. The
    accumulators are the one value a result does not carry: a slice is frozen
    with its IC estimates, and a test about attribution wants the totals before
    that.
    """
    parts = replay_parts(**overrides)
    result = parts.result
    return ReplayLenses(
        request=parts.request,
        schedule=parts.schedule,
        events=parts.events,
        curve=ReplayCurve(equity=result.equity, benchmark=result.benchmark),
        metrics=result.metrics,
        slices=_slice_accumulators(parts.events, parts.schedule),
    )


# ------------------------------------------------------- the IC lens fixtures

# The IC window trades one session and leaves the rest of the schedule as
# tail, which the default window cannot do: it ends at `ic_tail_end`, so no
# forward return there can read past the last close it tested.
#
# Every number below is a literal read off `base_schedule`. That schedule
# carries forty base intervals, twenty-six on 2026-11-25 and fourteen on the
# 2026-11-27 half day, with the holiday between them absent. Warming up on the
# first interval and testing through the full session's last close selects
# ordinals 1 through 25, and leaves ordinals 26 through 39 as outcome-only
# bars. So a one-bar and a five-bar horizon reach a close for all twenty-five
# observations, while a twenty-bar horizon reaches one for the first nineteen.
IC_TRAIN_START = ts("2026-11-25T14:30:00Z")
IC_TEST_START = ts("2026-11-25T14:45:00Z")
IC_TEST_END = ts("2026-11-25T21:00:00Z")
IC_OBSERVATIONS = 25
IC_TAIL_LIMITED_OBSERVATIONS = 19


def ic_window(test_end: pd.Timestamp = IC_TEST_END,
              test_start: pd.Timestamp = IC_TEST_START) -> ReplayWindow:
    """Warm up on one base interval and test through `test_end`."""
    return ReplayWindow(
        train_start=IC_TRAIN_START, train_end=test_start,
        test_start=test_start, test_end=test_end,
    )


def ic_plays(*, timeframe: str = "15m",
             margin: Callable[[int, pd.Timestamp], float | None] | None = None,
             play_ids: tuple[str, ...] = DEFAULT_PLAY_IDS,
             **fields) -> tuple[ScriptedPlay, ...]:
    """Plays that trade nothing and grade every observation they are given.

    Nothing signals, so the ledger, the curve, and the trade cohorts stay
    empty and an IC test measures the IC lens rather than the chronology.
    """
    return tuple(
        ScriptedPlay(play_id=play_id, priority=100, signals=(),
                     ic_timeframe=timeframe, ic_margin=margin, **fields)
        for play_id in play_ids
    )


@dataclasses.dataclass(frozen=True)
class ReplayIc:
    """One replay, its accumulators, its frozen slices, and its inputs.

    The same drive as `replay_metrics`, read at the IC lens instead. The slices
    come off the finished result and the accumulators are the one value a
    result does not carry.

    `slice_ic` is named for what it IS: the estimates read back off those
    slices, keyed for convenience. It is NOT an independent measurement, and a
    test that compares a slice against it is comparing a value with itself. Use
    it to build a well-shaped map to perturb; to check that a slice carries the
    estimates the lens measured for its own pair, call `_ic_map` over
    `schedule`, `prepared`, and `registry`, which is why all three travel here.
    """

    request: PortfolioReplayRequest
    schedule: ValidatedSchedule
    prepared: _ValidatedPortfolioBars
    registry: FrozenStrategyRegistry
    events: ReplayEvents
    totals: Mapping[PositionKey, object]
    slice_ic: Mapping[PositionKey, tuple[IcEstimate, IcEstimate, IcEstimate]]
    slices: tuple[PortfolioSlice, ...]


def replay_ic(**overrides) -> ReplayIc:
    """The frozen slices the finished result carries, plus what produced them.

    Nothing here measures the lens a second time: a second `_ic_map` would call
    every graded factor again, which would double the factor calls a test
    counts. A test that wants an independent measurement asks for one itself,
    over the `schedule`, `prepared`, and `registry` returned here.
    """
    parts = replay_parts(**overrides)
    slices = parts.result.slices
    return ReplayIc(
        request=parts.request,
        schedule=parts.schedule,
        prepared=parts.prepared,
        registry=parts.registry,
        events=parts.events,
        totals=_slice_accumulators(parts.events, parts.schedule),
        slice_ic={(row.play_id, row.symbol): row.ic for row in slices},
        slices=slices,
    )


# ------------------------------------------ one long session for a real spec

# The rules behaviours need a run of many base intervals over hand-authored
# closes: a fourteen-period ATR has to warm up before a trade can run its
# course afterwards. The two real XNYS sessions above are twenty-six intervals
# and fourteen, which is not enough for both. Core reads the schedule as data
# and never consults an installed calendar, so one continuous synthetic session
# is an ordinary input rather than a claim about what XNYS trades.
RULES_SESSION_OPEN = "T14:30:00Z"
RULES_SESSION_INTERVALS = 26
RULES_WARMUP_INTERVALS = 20


def rules_schedule(intervals: int) -> ReplaySchedule:
    """`intervals` base bars over as many weekday sessions as they need.

    Twenty-six a session, opening at 14:30Z, which is a full regular session's
    worth. It rolls rather than running on, because an interval belongs to the
    exchange session it opens in and a sixty-bar run from one open would put
    its later intervals on the next Eastern date.
    """
    built: list[ScheduledBaseInterval] = []
    sessions = daily_session_dates(
        -(-max(intervals, 1) // RULES_SESSION_INTERVALS))
    for session in sessions:
        built.extend(session_intervals(
            session, ts(f"{session.isoformat()}{RULES_SESSION_OPEN}"),
            min(RULES_SESSION_INTERVALS, intervals - len(built))))
    draft = ReplaySchedule(identity=base_identity(),
                           base_intervals=tuple(built), context_bars=())
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def rules_frames(closes: Sequence[float], schedule: ReplaySchedule,
                 symbol: str = "SPY") -> dict:
    """One bar per scheduled interval, opening and closing at `closes[i]`.

    A fifth of a point either side, so a level placed between two closes is
    reached by the bar that spans it and by no other. Open equal to close makes
    every fill price the caller's own number: an entry fills at the next bar's
    open, which is that bar's stated close.
    """
    index = pd.DatetimeIndex([row.open_ts for row in schedule.base_intervals],
                             tz="UTC", name="ts")
    values = pd.Series(list(closes), index=index, dtype="float64")
    return {(symbol, "15m"): pd.DataFrame(
        {"open": values, "high": values + 0.2, "low": values - 0.2,
         "close": values, "volume": 1_000.0}, index=index)}


def rules_replay(spec: Mapping, closes: Sequence[float], *,
                 warmup: int = RULES_WARMUP_INTERVALS, symbol: str = "SPY",
                 bars: tuple[BarPlan, ...] = (),
                 account: AccountPolicy | None = None,
                 execution: ExecutionPolicy | None = None,
                 vocabulary_factory: Callable = core_vocabulary,
                 ) -> PortfolioReplayResult:
    """One real RuleSpec replayed through the public entry point.

    Nothing is scripted: a real definition builds a real `RuleStrategy` through
    a real factory, and it goes through the same door a portfolio of forty
    plays goes through. The first `warmup` intervals are train and every
    interval after them is tested, which is what gives an indicator its history
    without letting a train bar produce a trade.

    `bars` replaces named base bars outright, which is how a test puts a level
    inside one bar's range: the default geometry is deliberately narrow, so a
    bar that reaches a level reaches it because a test said so.

    `vocabulary_factory` is the grammar the definition is read under, and it
    reaches the base digest and the definition together, the way a platform
    bundle builds one. It is what lets a test replay one spec under two
    grammars and compare what came back.
    """
    schedule = rules_schedule(len(closes))
    intervals = schedule.base_intervals
    params: dict = {}
    base = spec_base_digest(spec, vocabulary_factory)
    request = base_request(
        plays=(PlayRequest(
            play_id="play-r", strategy=spec["name"],
            definition_digest=definition_digest(base, params),
            params=params, priority=100),),
        symbols=(symbol,),
        window=ReplayWindow(
            train_start=intervals[0].open_ts,
            train_end=intervals[warmup].open_ts,
            test_start=intervals[warmup].open_ts,
            test_end=intervals[-1].close_ts,
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=intervals[-1].close_ts,
        account=replay_account() if account is None else account,
        execution=frictionless_execution() if execution is None else execution,
    )
    registry = FrozenStrategyRegistry.from_definitions(
        (rules_definition(spec["name"], base, spec=spec,
                          vocabulary_factory=vocabulary_factory),))
    frames = with_base_bars(rules_frames(closes, schedule, symbol), bars)
    return run_portfolio(request, PortfolioBars(frames), registry, schedule)


def rules_interval(ordinal: int) -> pd.Timestamp:
    """The open of `rules_schedule`'s interval `ordinal`.

    Read off a schedule rather than computed, because the run rolls onto a new
    weekday session every twenty-six intervals and plain arithmetic from the
    first open would name an instant no session covers.
    """
    return rules_schedule(ordinal + 1).base_intervals[ordinal].open_ts


# ------------------------------------------------- many-session metric input

# The replay fixtures above trade one half day, which is one daily return. The
# pooled statistics need sixty, and the annualization needs a window long
# enough to state in years, so those tests drive the metric lens over a
# schedule of many short sessions and a curve built by hand. Core reads the
# schedule as data and never consults an installed calendar, so a synthetic
# run of weekday sessions is an ordinary input rather than a claim about XNYS.
#
# Two intervals a session, 14:30Z to 15:00Z. That is 09:30 Eastern in winter
# and 10:30 in summer, and both land on the session's own date, which is the
# only thing `validate_schedule` asks of an open.
DAILY_FIRST_SESSION = date(2026, 1, 5)
DAILY_SESSION_OPEN = "T14:30:00Z"
DAILY_SESSION_INTERVALS = 2


def daily_session_dates(count: int) -> tuple[date, ...]:
    """`count` consecutive weekday sessions from the first Monday of 2026."""
    built: list[date] = []
    day = DAILY_FIRST_SESSION
    while len(built) < count:
        if day.weekday() < 5:
            built.append(day)
        day += timedelta(days=1)
    return tuple(built)


def daily_schedule(sessions: int, *, last_intervals: int | None = None) -> ReplaySchedule:
    """A schedule of `sessions` short weekday sessions and no context bars.

    `last_intervals` lengthens the final session alone, which is how a window
    whose test range spans an exact number of years is built: session opens are
    a whole number of days apart, so only the last close can supply the hours.
    """
    dates = daily_session_dates(sessions)
    intervals: list[ScheduledBaseInterval] = []
    for position, session in enumerate(dates):
        count = DAILY_SESSION_INTERVALS
        if last_intervals is not None and position == len(dates) - 1:
            count = last_intervals
        intervals.extend(session_intervals(
            session, ts(f"{session.isoformat()}{DAILY_SESSION_OPEN}"), count))
    draft = ReplaySchedule(
        identity=base_identity(), base_intervals=tuple(intervals), context_bars=(),
    )
    return dataclasses.replace(draft, identity=base_identity(schedule_digest(draft)))


def daily_request(sessions: int, *, last_intervals: int | None = None,
                  **overrides) -> PortfolioReplayRequest:
    """Warm up on the first session, test every session after it."""
    schedule = daily_schedule(sessions, last_intervals=last_intervals)
    intervals = schedule.base_intervals
    return base_request(
        window=ReplayWindow(
            train_start=intervals[0].open_ts,
            train_end=intervals[DAILY_SESSION_INTERVALS].open_ts,
            test_start=intervals[DAILY_SESSION_INTERVALS].open_ts,
            test_end=intervals[-1].close_ts,
        ),
        schedule_identity=schedule.identity,
        ic_tail_end=intervals[-1].close_ts,
        **overrides,
    )


def daily_validated(sessions: int, *, last_intervals: int | None = None,
                    **overrides) -> ValidatedSchedule:
    return validate_schedule(
        daily_request(sessions, last_intervals=last_intervals, **overrides),
        daily_schedule(sessions, last_intervals=last_intervals),
    )


def _daily_point(request: PortfolioReplayRequest, ordinal: int,
                 at: pd.Timestamp, equity: float) -> EquityPoint:
    """One point holding `equity` and nothing else, reconciled by hand.

    A positive account is all settled cash; a nonpositive one is carried as
    liquidation value, because settled cash cannot be negative. Either way the
    point states the same account identity a real snapshot does.
    """
    return EquityPoint(
        replay_id=request.replay_id,
        ts=at,
        point_ordinal=ordinal,
        settled_cash=equity if equity > 0.0 else 0.0,
        unsettled_cash=0.0,
        short_collateral=0.0,
        positions_liquidation_value=0.0 if equity > 0.0 else equity,
        portfolio_equity=equity,
        gross_exposure=0.0,
        open_positions=0,
        benchmark_equity=request.account.starting_equity,
    )


def daily_curve(validated: ValidatedSchedule, returns: Sequence[float], *,
                benchmark_return: float = 0.0,
                final: float | None = None) -> ReplayCurve:
    """One curve whose sessions realize `returns`, one return per test session.

    Every point of a session carries that session's closing equity, so a
    drawdown is readable off the returns alone. `final` gives the post-close
    point at `test_end` a value of its own, which is what tells a sampler that
    reads the last point of a session from one that reads the last close.

    The benchmark series is deliberately flat while `benchmark_return` is
    whatever the caller says: the two disagree so that a metric recomputing the
    benchmark from the points cannot pass for one reading the curve's own
    reported return.
    """
    request = validated.request
    starting = request.account.starting_equity
    equity = starting
    values: list[float] = []
    for factor in returns:
        equity = equity * (1.0 + factor)
        values.append(equity)
    sessions: list[date] = []
    for interval in validated.test_intervals:
        if not sessions or sessions[-1] != interval.session_date:
            sessions.append(interval.session_date)
    if len(sessions) != len(values):
        raise AssertionError(
            f"the schedule tests {len(sessions)} sessions and the fixture "
            f"supplied {len(values)} returns")
    by_session = dict(zip(sessions, values, strict=True))
    points = [_daily_point(request, 0, request.window.test_start, starting)]
    for interval in validated.test_intervals:
        points.append(_daily_point(
            request, len(points), interval.close_ts,
            by_session[interval.session_date]))
    points.append(_daily_point(
        request, len(points), request.window.test_end,
        values[-1] if final is None else final))
    return ReplayCurve(
        equity=tuple(points),
        benchmark=BenchmarkResult(spec=request.benchmark,
                                  total_return=benchmark_return),
    )


def daily_metrics(returns: Sequence[float], *, last_intervals: int | None = None,
                  **curve_fields) -> PortfolioMetrics:
    """The metrics of a curve that traded nothing and returned `returns`."""
    validated = daily_validated(len(returns) + 1, last_intervals=last_intervals)
    return _portfolio_metrics(
        daily_curve(validated, returns, **curve_fields), (), (), validated)


def canonical_event_bytes(events: ReplayEvents) -> bytes:
    """The canonical bytes of one replay's events.

    The C6-level answer to "are these two replays the same replay?", built from
    the same codec and the same contract projection the result digest uses, so
    two replays agree here exactly when they will agree there.
    """
    return canonical_replay_bytes({
        "trades": [_projection(row) for row in events.trades],
        "rejections": [_projection(row) for row in events.rejections],
        "marks": [{"ts": mark.ts, "snapshot": _projection(mark.snapshot)}
                  for mark in events.marks],
        "signals": [{"play_id": play_id, "symbol": symbol, "count": count}
                    for (play_id, symbol), count in events.signal_counts.items()],
    })


def canonical_curve_bytes(curve: ReplayCurve) -> bytes:
    """The canonical bytes of one replay's equity curve and benchmark.

    Built from the same codec and the same contract projection the result
    digest uses, so two curves agree here exactly when they will agree there.
    """
    return canonical_replay_bytes({
        "equity": [_projection(point) for point in curve.equity],
        "benchmark": _projection(curve.benchmark),
    })


def scripted_params(reverse: bool) -> dict:
    """One play's parameters, supplied in canonical order or against it."""
    params = {"alpha": 1, "beta": 2.5}
    return dict(reversed(list(params.items()))) if reverse else params


def scripted_play_request(play: ScriptedPlay, params: Mapping) -> PlayRequest:
    return PlayRequest(
        play_id=play.play_id,
        strategy=scripted_name(play.play_id),
        definition_digest=definition_digest(scripted_digest(play.play_id), params),
        params=params,
        priority=play.priority,
    )
