"""The public portfolio replay contract.

One account, one causal chronology, one canonical result. `run_portfolio` is
the only replay entry point core has, and everything a caller needs to build
its four arguments, hash them, transport them, and read the result is exported
here. Internal helpers stay behind their modules and are not re-exported: there
is no second replay API, no singleton adapter, and no deprecated name.

`__all__` is the whole surface. The names in `_DEFERRED` resolve on first
access rather than at import, and the reason is a one-way layering that Python
enforces between modules but not between packages.

`portfolio_types` owns the canonical value contract and imports nothing from
`nakagai.strategies`. `strategies/base.py` imports IT, and the replay modules
import `strategies.base` back, which is a clean stack of three layers. It stops
being clean the moment this file sits above all of them: importing
`nakagai.engine.portfolio_types` runs this module first, so an eager import of
the replay half here would reach `strategies.base` while `strategies.base` was
still executing its own import block, and a bare `import nakagai.strategies`
would die on a circular import.

Deferring the top layer costs nothing a caller can observe. Every name below is
importable, `from nakagai.engine import run_portfolio` works, `dir()` lists the
full set, and nothing outside `__all__` resolves at all.
"""

from importlib import import_module

from nakagai.engine.bars import PortfolioBars
from nakagai.engine.canonical import (
    canonical_replay_bytes,
    decode_replay_request,
    decode_replay_result,
    decode_replay_schedule,
    definition_digest,
    encode_replay_request,
    encode_replay_result,
    encode_replay_schedule,
    expected_candidate_id,
    expected_replay_id,
    rejection_id,
    result_digest,
    schedule_digest,
    trade_id,
)
from nakagai.engine.portfolio_types import (
    ARITHMETIC_VERSION,
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
    JSONScalar,
    JSONValue,
    ManagementDecision,
    PlayRequest,
    PortfolioMetrics,
    PortfolioReplayRequest,
    PortfolioReplayResult,
    PortfolioSlice,
    PortfolioTrade,
    PositionView,
    RejectionReason,
    ReplayInputError,
    ReplayRejection,
    ReplaySchedule,
    ReplayWindow,
    ScheduledBaseInterval,
    ScheduledContextBar,
    Signal,
    SlippageSpec,
    StrategyOutputError,
    StrategyRuntimeError,
    TradeStats,
)

# The replay half: every module below reaches `nakagai.strategies`, so it can
# only be imported once this package has finished initializing.
_DEFERRED = {
    "FrozenStrategyRegistry": "nakagai.engine.registry",
    "StrategyDefinition": "nakagai.engine.registry",
    "StrategyDependencies": "nakagai.engine.registry",
    "StrategyRegistry": "nakagai.engine.registry",
    "composite_definition": "nakagai.engine.registry",
    "rules_definition": "nakagai.engine.registry",
    "run_portfolio": "nakagai.engine.replay",
    "spec_definition_digest": "nakagai.engine.registry",
}

__all__ = [
    "ARITHMETIC_VERSION",
    "AccountPolicy",
    "BenchmarkResult",
    "BenchmarkSpec",
    "EntryIntent",
    "EquityPoint",
    "ExchangeScheduleIdentity",
    "ExecutionPolicy",
    "ExitReason",
    "FeeSpec",
    "FrozenStrategyRegistry",
    "IcEstimate",
    "JSONScalar",
    "JSONValue",
    "ManagementDecision",
    "PlayRequest",
    "PortfolioBars",
    "PortfolioMetrics",
    "PortfolioReplayRequest",
    "PortfolioReplayResult",
    "PortfolioSlice",
    "PortfolioTrade",
    "PositionView",
    "RejectionReason",
    "ReplayInputError",
    "ReplayRejection",
    "ReplaySchedule",
    "ReplayWindow",
    "ScheduledBaseInterval",
    "ScheduledContextBar",
    "Signal",
    "SlippageSpec",
    "StrategyDefinition",
    "StrategyDependencies",
    "StrategyOutputError",
    "StrategyRegistry",
    "StrategyRuntimeError",
    "TradeStats",
    "canonical_replay_bytes",
    "composite_definition",
    "decode_replay_request",
    "decode_replay_result",
    "decode_replay_schedule",
    "definition_digest",
    "encode_replay_request",
    "encode_replay_result",
    "encode_replay_schedule",
    "expected_candidate_id",
    "expected_replay_id",
    "rejection_id",
    "result_digest",
    "rules_definition",
    "run_portfolio",
    "schedule_digest",
    "spec_definition_digest",
    "trade_id",
]


def __getattr__(name: str) -> object:
    """Resolve one deferred export, then cache it in this module's namespace."""
    module = _DEFERRED.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_DEFERRED))
