"""The one public replay: everything a caller can reach lives behind this door.

`run_portfolio` is the whole of core's replay API. It is deterministic and side
effect free: it reads no cache, writes no file, generates no parent identifier,
consults no installed calendar, inspects no environment, and discovers no
strategy. The caller supplies a complete request, immutable bars, a frozen
registry, and the schedule that is the replay's clock, and gets back one
canonical result.

The order of what happens here is a contract rather than an implementation
detail, and it reads top to bottom:

1. the REQUEST, whose parent identities must match their own formulas;
2. the SCHEDULE, checked against that request and hashed against its own body;
3. the REGISTRY, where every play resolves to a definition and its digest is
   proven to bind that definition to that play's params;
4. the DEPENDENCY CLOSURE, asked of each definition's pure function;
5. the COMPLETE BAR SET, refused whole on any absent, surplus, or malformed
   frame;
6. and only then the RUNTIME, whose construction is the first thing in a
   replay that runs strategy code.

Nothing before step 6 calls a factory, which is what "refused before a strategy
instance is created" means. It is a property of this composition rather than of
any one of the doors, so the tests assert it with a construction count on every
refusal above rather than by reading the source.

The result is assembled once, at the end, from lenses that each read the one
event stream: the equity curve and its benchmark, the play-symbol accumulators,
the complete IC map, the frozen slices, and the portfolio metrics. The digest
is the SHA-256 of the canonical result bytes with its own digest field omitted,
so a settler that never saw the run can recompute it from the decoded value.
"""

import dataclasses

from nakagai.engine.bars import (
    PortfolioBars,
    _ValidatedPortfolioBars,
    prepare_portfolio_bars,
)
from nakagai.engine.benchmark import _equity_series
from nakagai.engine.canonical import (
    expected_candidate_id,
    expected_replay_id,
    result_digest,
)
from nakagai.engine.execution import ReplayEvents, _PortfolioRuntime
from nakagai.engine.ic import _ic_map, _portfolio_slices
from nakagai.engine.metrics import _portfolio_metrics, _slice_accumulators
from nakagai.engine.portfolio_types import (
    ARITHMETIC_VERSION,
    PortfolioReplayRequest,
    PortfolioReplayResult,
    ReplaySchedule,
    _fail,
    _require_instance,
)
from nakagai.engine.registry import (
    StrategyRegistry,
    dependencies_for,
    validate_registry,
)
from nakagai.engine.schedule import ValidatedSchedule, validate_schedule

# The digest field cannot carry its own value while it is being computed, and
# the result type refuses anything that is not a digest, so the draft carries
# this and `_finalize_result` replaces it with the real one.
_UNSET_DIGEST = "0" * 64


def run_portfolio(
    request: PortfolioReplayRequest,
    bars: PortfolioBars,
    registry: StrategyRegistry,
    schedule: ReplaySchedule,
) -> PortfolioReplayResult:
    """Replay one account over one set of plays and symbols, or refuse."""
    return _finalize_result(_replay(request, bars, registry, schedule))


@dataclasses.dataclass(frozen=True)
class _ReplayRun:
    """One replay's validated inputs and the chronology they produced.

    Internal, and never returned from `run_portfolio`. It exists because the
    lenses that build a result each need a different one of these: the curve
    needs the prepared bars, the accumulators need the schedule, and the IC
    lens needs the registry. Carrying them together is what lets the final
    assembly be one function reading one value.
    """

    request: PortfolioReplayRequest
    schedule: ValidatedSchedule
    registry: StrategyRegistry
    prepared: _ValidatedPortfolioBars
    events: ReplayEvents


def _replay(
    request: PortfolioReplayRequest,
    bars: PortfolioBars,
    registry: StrategyRegistry,
    schedule: ReplaySchedule,
) -> _ReplayRun:
    """The preflight, in contract order, and then the one chronology.

    Every statement above `_PortfolioRuntime` is a refusal door, and none of
    them constructs a strategy. `dependencies_for` is the closest call: it asks
    each definition what it reads, which the registry contract requires to be
    pure and to build nothing.

    The closure is computed once and travels to both the bar preflight and the
    runtime, so the two cannot disagree about what this replay reads. Handing
    the runtime a second closure is the miswiring `_PortfolioRuntime` refuses,
    and this composition is what makes it unreachable from the public door.
    """
    validated_request = validate_request(request)
    validated_schedule = validate_schedule(validated_request, schedule)
    validated_registry = validate_registry(validated_request, registry)
    dependencies = dependencies_for(validated_request, validated_registry)
    prepared = prepare_portfolio_bars(
        validated_request, bars, validated_schedule, dependencies)
    runtime = _PortfolioRuntime(validated_request, validated_schedule,
                                validated_registry, prepared, dependencies)
    return _ReplayRun(
        request=validated_request,
        schedule=validated_schedule,
        registry=validated_registry,
        prepared=prepared,
        events=runtime.run(),
    )


def validate_request(request: PortfolioReplayRequest) -> PortfolioReplayRequest:
    """The request step: the parent identities agree with their own formulas.

    `PortfolioReplayRequest` already proves its own field shapes when it is
    built, so what is left is the pair of derived identities. Platform mints
    both through the same two functions core calls here, and core derives every
    child identifier from `replay_id`, so an identity that does not recompute
    would silently produce trades and rejections belonging to a replay nobody
    asked for.

    The candidate is checked as well as the replay, and it is not implied by
    it: `replay_id` hashes the candidate as an opaque string, so a candidate
    that disagreed with its own projection would still produce a self-
    consistent replay identity.
    """
    _require_instance(request, "request", PortfolioReplayRequest)
    for field, expected in (
        ("candidate_id", expected_candidate_id(request)),
        ("replay_id", expected_replay_id(request)),
    ):
        actual = getattr(request, field)
        if actual != expected:
            raise _fail(
                "identity_mismatch", "an identity does not match its own formula",
                field=field, expected=expected, actual=actual,
            )
    return request


def _finalize_result(run: _ReplayRun) -> PortfolioReplayResult:
    """One result from one event stream, and its digest over all of it.

    The order of the four lenses is load bearing in one place. The COMPLETE IC
    map is built before any slice, because a slice carries its three estimates
    and the map answers for every canonical play symbol at once; freezing a
    slice earlier would mean rebuilding it. `_portfolio_slices` then consumes
    each accumulator exactly once, so the attribution a reader checks by hand
    is the one addition rather than two that happen to agree.

    The two provenance fields come from different places on purpose. The result
    reports the arithmetic version the ENGINE executed, and `_portfolio_metrics`
    refuses a request that named a different one, so the pair cannot disagree
    silently. `fill_mode` is echoed from the request because its type admits
    exactly one value, which leaves nothing for a second constant to state.
    """
    curve = _equity_series(run.schedule, run.events, run.prepared)
    totals = _slice_accumulators(run.events, run.schedule)
    ic = _ic_map(run.schedule, run.prepared, run.registry)
    draft = PortfolioReplayResult(
        request=run.request,
        arithmetic_version=ARITHMETIC_VERSION,
        fill_mode=run.request.execution.fill_mode,
        schedule_identity=run.request.schedule_identity,
        result_digest=_UNSET_DIGEST,
        trades=run.events.trades,
        rejections=run.events.rejections,
        equity=curve.equity,
        slices=_portfolio_slices(run.schedule, totals, ic),
        benchmark=curve.benchmark,
        metrics=_portfolio_metrics(curve, run.events.trades,
                                   run.events.rejections, run.schedule),
    )
    return dataclasses.replace(draft, result_digest=result_digest(draft))
