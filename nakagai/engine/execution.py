"""One causal chronology: what a replay does at an interval, and in what order.

Every other module in the replay answers a question about one thing. This one
decides WHEN each of them is asked, which is the whole product: two replays over
the same inputs agree byte for byte because the order of events is fixed here
and derives from nothing else.

The order is the architecture's, step for step:

1. settle every credit whose New York settlement session has arrived;
2. freeze the open marks and the pre-event equity every candidate sizes from;
3. exit positions the OPEN was already beyond, at that open;
4-6. select the intents eligible at this open, sort them into funding order,
   and process each past every account rule, filling or refusing;
7-8. exit positions the rest of the bar reached a level on, INCLUDING the ones
   filled at this open, then fold the surviving positions' excursions;
9-10. build one causal context per symbol at the close, then manage every open
   position in position order;
11-12. evaluate every play symbol in canonical order, number every signal it
   returns, and either open its pending seat or record why it could not;
13. mark the account at the close.

At the final close the ordinary mark is taken first, then pending intents
expire in funding order, then the remaining positions liquidate in position
order, and one post-close mark closes the series.

Nothing here is ordered by a dictionary, a set, a supplied collection, a clock,
or a random value. Where several things happen at one instant, the order comes
from `_Ledger`'s one canonical key, and where several plays are evaluated, it
comes from the request's own canonical play and symbol order.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from nakagai.engine.bars import (
    BASE_TIMEFRAME,
    ReplayDependencies,
    _ValidatedPortfolioBars,
)
from nakagai.engine.context import build_scheduled_context
from nakagai.engine.portfolio import LedgerSnapshot, PositionKey, _Ledger
from nakagai.engine.portfolio_types import (
    EntryIntent,
    ExitReason,
    PortfolioReplayRequest,
    PortfolioTrade,
    RejectionReason,
    ReplayRejection,
    ScheduledBaseInterval,
    Signal,
    StrategyOutputError,
    _fail,
    _require_instance,
    _set_positive,
)
from nakagai.engine.registry import StrategyRegistry
from nakagai.engine.schedule import ValidatedSchedule
from nakagai.strategies.base import (
    MarketContext,
    Strategy,
    call_manage,
    call_on_bar,
    strategy_operation,
)

# `_Ledger.protective_exit` answers for the whole bar, so one predicate covers
# two separate steps of this chronology. A reason naming the open belongs to
# step 3 and fills at that open; the other two belong to step 7, after this
# interval's own fills, and fill at the close.
_GAP_REASONS = (ExitReason.STOP_GAP, ExitReason.TARGET_GAP)


@dataclass(frozen=True)
class _Bar:
    """One symbol's RAW prices at one scheduled interval.

    Raw is the point. These four decide what the market did; the ledger's cost
    models decide what it cost, and a slipped price never enters here.
    """

    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        _set_positive(self, "open", "high", "low", "close")


class _RawBase:
    """One symbol's base bars, read by scheduled row rather than by label.

    A prepared frame's labels ARE the schedule's base opens, in the schedule's
    order, so row `i` is scheduled interval `i` and no lookup by timestamp is
    needed in the loop at all.
    """

    __slots__ = ("_columns",)

    def __init__(self, frame: pd.DataFrame) -> None:
        self._columns = tuple(
            frame[column].to_numpy() for column in ("open", "high", "low", "close")
        )

    def at(self, row: int) -> _Bar:
        return _Bar(*(float(column[row]) for column in self._columns))


@dataclass(frozen=True)
class ReplayMark:
    """The account at one recorded instant.

    The equity lens turns these into `EquityPoint` values: the point ordinal is
    this value's position in the series, and the benchmark is marked at exactly
    these instants. The two final marks share a timestamp on purpose, one from
    before the window's liquidation and one from after it.
    """

    ts: pd.Timestamp
    snapshot: LedgerSnapshot


@dataclass(frozen=True)
class ReplayEvents:
    """Everything one replay's chronology produced, each in its own event order.

    `signal_counts` carries one entry for every canonical play symbol, whether
    or not it ever signalled, so an attribution lens never has to decide what an
    absent key meant.
    """

    trades: tuple[PortfolioTrade, ...]
    rejections: tuple[ReplayRejection, ...]
    marks: tuple[ReplayMark, ...]
    signal_counts: Mapping[PositionKey, int]


class _PortfolioRuntime:
    """One replay, driven once, through one ledger and one set of runtimes.

    Constructed after every preflight has passed, so building it is the first
    thing in a replay that runs strategy code: one instance per
    `(replay_id, play_id, symbol)`, none shared, none reused.
    """

    def __init__(self, request: PortfolioReplayRequest,
                 schedule: ValidatedSchedule, registry: StrategyRegistry,
                 prepared: _ValidatedPortfolioBars,
                 dependencies: ReplayDependencies) -> None:
        _require_instance(request, "request", PortfolioReplayRequest)
        _require_instance(schedule, "schedule", ValidatedSchedule)
        _require_instance(prepared, "prepared", _ValidatedPortfolioBars)
        _require_instance(dependencies, "dependencies", ReplayDependencies)
        if not isinstance(registry, StrategyRegistry):
            raise _fail("invalid_type", "value must be a strategy registry",
                        field="registry")
        if schedule.request != request:
            raise _fail(
                "mismatched_schedule", "the schedule was validated for another request",
                field="schedule",
            )
        _require_prepared_closure(prepared, request, dependencies)
        self._request = request
        self._schedule = schedule
        self._prepared = prepared
        self._dependencies = dependencies
        self._symbols = request.symbols
        self._plays = {play.play_id: play for play in request.plays}
        # Canonical evaluation order: play `(priority, play_id)`, which the
        # request already sorted itself into, then uppercase symbol.
        self._keys: tuple[PositionKey, ...] = tuple(
            (play.play_id, symbol)
            for play in request.plays for symbol in request.symbols
        )
        self._base = {symbol: _RawBase(prepared.frame(symbol, BASE_TIMEFRAME))
                      for symbol in request.symbols}
        self._ledger = _Ledger(request, schedule)
        self._trades: list[PortfolioTrade] = []
        self._rejections: list[ReplayRejection] = []
        self._marks: list[ReplayMark] = []
        self._pending: dict[PositionKey, EntryIntent] = {}
        self._counts: dict[PositionKey, int] = {key: 0 for key in self._keys}
        self._signal_ordinal = 0
        self._runtimes = self._build_runtimes(registry)

    # ------------------------------------------------------------ the loop

    def run(self) -> ReplayEvents:
        intervals = self._schedule.test_intervals
        # The opening anchor, before any event: all starting equity in settled
        # cash and no position. It is the denominator of the first session's
        # return and drawdown, and it is not a close mark.
        self._mark(self._request.window.test_start,
                   _prices(self._bars_at(intervals[0]), "open"))
        for interval in intervals:
            bars = self._bars_at(interval)
            self._ledger.settle_due(interval.session_date)
            frozen_equity = self._freeze(interval, bars)
            self._gap_exits(interval, bars)
            self._fund(interval, bars, frozen_equity)
            self._protect(interval, bars)
            contexts = self._contexts(interval)
            self._manage(interval, bars, contexts)
            self._evaluate(interval, bars, contexts)
            self._mark(interval.close_ts, _prices(bars, "close"))
        self._close_window(self._bars_at(intervals[-1]))
        return ReplayEvents(
            trades=tuple(self._trades),
            rejections=tuple(self._rejections),
            marks=tuple(self._marks),
            signal_counts=MappingProxyType(
                {key: self._counts[key] for key in self._keys}),
        )

    # ------------------------------------------------------- interval steps

    def _freeze(self, interval: ScheduledBaseInterval,
                bars: Mapping[str, _Bar]) -> float:
        """Step 2: the equity every candidate at this open is sized from.

        Taken before this interval's gap exits and before its fills, so neither
        an exit at this open nor an earlier candidate's fill can change what a
        later candidate can afford.
        """
        return self._ledger.snapshot(
            interval.open_ts, _prices(bars, "open")).portfolio_equity

    def _gap_exits(self, interval: ScheduledBaseInterval,
                   bars: Mapping[str, _Bar]) -> None:
        """Step 3: exit the positions this open was already beyond a level of.

        The open is the reference, so the trade stamps the OPEN rather than the
        close, and its proceeds are unsettled cash that cannot fund a fill at
        this same open. The intrabar reasons are step 7's and are left alone
        here: acting on one now would stamp the wrong instant and take a trade
        ordinal ahead of this interval's fills.
        """
        for key in self._ledger.position_keys():
            bar = bars[key[1]]
            hit = self._ledger.protective_exit(key, bar.open, bar.high, bar.low)
            if hit is None or hit[1] not in _GAP_REASONS:
                continue
            # The first and last observation this position made on this bar:
            # it left at the open, so the rest of the bar is not its history.
            self._ledger.observe(key, bar.open, bar.open)
            self._settle_out(key, hit[0], interval.open_ts, hit[1])

    def _fund(self, interval: ScheduledBaseInterval, bars: Mapping[str, _Bar],
              frozen_equity: float) -> None:
        """Steps 4 to 6: select, sort, and process this open's candidates.

        An intent is eligible at exactly one open and expires there whatever the
        account decides, so it leaves the pending seat either way.
        """
        eligible = tuple(intent for intent in self._pending.values()
                         if intent.eligible_after == interval.open_ts)
        proposals = [
            self._ledger.propose(intent, bars[intent.symbol].open, frozen_equity)
            for intent in eligible
        ]
        for proposal in self._ledger.funding_order(proposals):
            del self._pending[proposal.key]
            reservation = self._ledger.reserve(proposal, frozen_equity)
            if reservation.accepted:
                self._ledger.open(proposal, interval.open_ts)
                continue
            self._rejections.append(self._ledger.reject(
                proposal.intent, reservation.reason, interval.open_ts,
                reservation.required_cash, reservation.available_cash,
            ))

    def _protect(self, interval: ScheduledBaseInterval,
                 bars: Mapping[str, _Bar]) -> None:
        """Steps 7 and 8: intrabar exits, then the survivors' excursions.

        Every position present after the fills is tested, including one filled
        at this interval's own open: its protection was live from the fill, so
        an entry bar that runs to its stop is a loss rather than an anomaly.

        A gap reason cannot reach here. A carried position beyond a level left
        at step 3, and a new fill passed the raw-open gate, so the open lies
        strictly inside its protective range. It is skipped rather than booked
        at the wrong instant, and the next interval's step 3 would take it.
        """
        for key in self._ledger.position_keys():
            bar = bars[key[1]]
            hit = self._ledger.protective_exit(key, bar.open, bar.high, bar.low)
            if hit is None or hit[1] in _GAP_REASONS:
                continue
            fill, reason = hit
            view = self._ledger.view(key)
            # What this exit is allowed to have seen, and no more. The open
            # happened, and the level that triggered happened. A stop credits
            # no favorable extreme, because OHLC cannot prove that extreme came
            # first; a target credits the ADVERSE extreme, because pessimistic
            # ordering already assumed it did.
            self._ledger.observe(key, bar.open, bar.open)
            if reason is ExitReason.STOP:
                self._ledger.observe(key, view.live_stop, view.live_stop)
            else:
                self._ledger.observe(key, view.live_target, view.live_target)
                adverse = bar.low if view.direction == "long" else bar.high
                self._ledger.observe(key, adverse, adverse)
            self._settle_out(key, fill, interval.close_ts, reason)
        for key in self._ledger.position_keys():
            # A position that survived the bar was live throughout it, so it
            # folds the whole raw range.
            bar = bars[key[1]]
            self._ledger.observe(key, bar.high, bar.low)

    def _contexts(self, interval: ScheduledBaseInterval) -> dict[str, MarketContext]:
        """Step 9: one point-in-time context per symbol at this close.

        Per symbol rather than per play symbol: a context is a view of one
        symbol's released bars, so two plays reading the same symbol at the same
        instant are entitled to exactly the same view of it.
        """
        return {
            symbol: build_scheduled_context(
                self._prepared, symbol, interval.close_ts, self._schedule,
                self._dependencies,
            )
            for symbol in self._symbols
        }

    def _manage(self, interval: ScheduledBaseInterval, bars: Mapping[str, _Bar],
                contexts: Mapping[str, MarketContext]) -> None:
        """Step 10: manage every open position, in position order.

        A replacement lands before a requested close, so a decision that
        ratchets and exits on one bar records the tighter level it exited under.
        """
        for key in self._ledger.position_keys():
            play_id, symbol = key
            close = bars[symbol].close
            decision = call_manage(
                self._runtimes[key], self._ledger.view(key), contexts[symbol],
                deciding_close=close, play_id=play_id,
                event_ts=interval.close_ts.isoformat(),
            )
            if decision.stop is not None or decision.target is not None:
                self._ledger.adjust(key, decision.stop, decision.target)
            if decision.action == "exit":
                view = self._ledger.view(key)
                self._settle_out(
                    key, self._ledger.exit_fill(view.direction, close),
                    interval.close_ts, ExitReason.MANAGE,
                )

    def _evaluate(self, interval: ScheduledBaseInterval, bars: Mapping[str, _Bar],
                  contexts: Mapping[str, MarketContext]) -> None:
        """Steps 11 and 12: evaluate every runtime, then number every signal.

        Evaluation is unconditional. A play symbol that is occupied, out of
        cash, or out of capacity is still asked, because its state is its own
        and the declined opportunity is what the result is for.
        """
        for key in self._keys:
            play_id, symbol = key
            signals = call_on_bar(
                self._runtimes[key], contexts[symbol],
                deciding_close=bars[symbol].close, play_id=play_id,
                event_ts=interval.close_ts.isoformat(),
            )
            for signal in signals:
                self._record_signal(key, signal, interval)

    def _record_signal(self, key: PositionKey, signal: Signal,
                       interval: ScheduledBaseInterval) -> None:
        """One signal: its replay-wide ordinal, then its seat or its refusal.

        The ordinal is assigned to EVERY signal, in this canonical order, before
        anything decides whether the account can act on it. A refusal at the
        deciding close carries that close as its event timestamp, because that
        is where the account declined it.
        """
        play_id, symbol = key
        intent = EntryIntent(
            replay_id=self._request.replay_id,
            play_id=play_id,
            strategy=self._plays[play_id].strategy,
            symbol=symbol,
            signal=signal,
            signal_ordinal=self._signal_ordinal,
            signal_ts=interval.close_ts,
            order_type="market_next_open",
            eligible_after=self._eligible_after(interval),
            expires_after_intervals=1,
        )
        self._signal_ordinal += 1
        self._counts[key] += 1
        if self._ledger.holds(key):
            reason = RejectionReason.POSITION_OCCUPIED
        elif key in self._pending:
            reason = RejectionReason.PENDING_INTENT_OCCUPIED
        else:
            self._pending[key] = intent
            return
        self._rejections.append(self._ledger.reject(
            intent, reason, interval.close_ts, None, None))

    def _close_window(self, bars: Mapping[str, _Bar]) -> None:
        """The final close, after its ordinary mark: expire, liquidate, mark.

        Nothing crosses a reporting-window boundary, so the sweep is total: no
        pending intent, no position, and no strategy runtime survives it.
        """
        end = self._request.window.test_end
        for intent in self._ledger.pending_order(tuple(self._pending.values())):
            self._rejections.append(self._ledger.reject(
                intent, RejectionReason.WINDOW_ENDED, end, None, None))
        self._pending.clear()
        for key in self._ledger.position_keys():
            view = self._ledger.view(key)
            close = bars[key[1]].close
            self._settle_out(key, self._ledger.exit_fill(view.direction, close),
                             end, ExitReason.END_OF_WINDOW)
        self._mark(end, _prices(bars, "close"))

    # ------------------------------------------------------------ internals

    def _build_runtimes(self, registry: StrategyRegistry) -> dict[PositionKey, Strategy]:
        """One fresh strategy per play symbol, built at `test_start`.

        Resolution is core's own refusal and stays outside the wrapper; only the
        call INTO the definition's factory is strategy code, and anything it
        raises is a runtime error naming the play symbol it was building.

        Every construction is verified against its definition, because a
        factory is the one place a bundle's promises stop being checkable by
        the registry: it never calls one, so a definition that hands back a
        strategy calling itself something else is only discoverable here.
        Nothing downstream would notice on its own. The declared name reaches
        an operator only through an error's details, so a mislabeled runtime
        replays silently and blames another definition the one time something
        does go wrong.
        """
        stamp = self._request.window.test_start.isoformat()
        runtimes: dict[PositionKey, Strategy] = {}
        for key in self._keys:
            play = self._plays[key[0]]
            definition = registry.resolve(play.strategy)
            with strategy_operation("construct", strategy=play.strategy,
                                    play_id=key[0], symbol=key[1], event_ts=stamp):
                runtime = definition.factory(play.params)
            declared = getattr(runtime, "name", None)
            if declared != definition.name:
                raise StrategyOutputError(
                    "strategy_name_mismatch",
                    "a factory returned a strategy that names another definition",
                    {"operation": "construct", "strategy": definition.name,
                     "declared": declared if isinstance(declared, str) else None,
                     "play_id": key[0], "symbol": key[1], "event_ts": stamp},
                )
            runtimes[key] = runtime
        return runtimes

    def _bars_at(self, interval: ScheduledBaseInterval) -> dict[str, _Bar]:
        """Every traded symbol's raw bar for one scheduled interval."""
        row = self._schedule.closed_base_count(interval.close_ts) - 1
        return {symbol: self._base[symbol].at(row) for symbol in self._symbols}

    def _eligible_after(self, interval: ScheduledBaseInterval) -> pd.Timestamp:
        """The one open a signal decided at this close may fill at.

        The next SCHEDULED interval, which is the next open the symbol has,
        whatever weekend, holiday, or early close sits in between. Past the last
        interval the schedule carries there is no such open, and the intent is
        stamped at this close instead: that instant is the window's own end, no
        test interval opens on it, and the intent expires unfilled.
        """
        following = self._schedule.closed_base_count(interval.close_ts)
        intervals = self._schedule.base_intervals
        if following < len(intervals):
            return intervals[following].open_ts
        return interval.close_ts

    def _settle_out(self, key: PositionKey, fill: float, ts: pd.Timestamp,
                    reason: ExitReason) -> None:
        """The one door a position leaves by. `fill` is already slipped."""
        qty = self._ledger.view(key).qty
        self._trades.append(self._ledger.close(
            key, fill, ts, reason, self._ledger.exit_fee(qty)))

    def _mark(self, ts: pd.Timestamp, marks: Mapping[str, float]) -> None:
        self._marks.append(ReplayMark(ts=ts, snapshot=self._ledger.snapshot(ts, marks)))


def _prices(bars: Mapping[str, _Bar], field: str) -> dict[str, float]:
    """One raw price per symbol, in the shape the ledger marks against."""
    return {symbol: getattr(bar, field) for symbol, bar in bars.items()}


def _require_prepared_closure(
    prepared: _ValidatedPortfolioBars, request: PortfolioReplayRequest,
    dependencies: ReplayDependencies,
) -> None:
    """Every frame this chronology will read was actually prepared.

    The prepared bars do not carry the closure they were prepared under, so a
    caller that hydrates one `ReplayDependencies` and drives the loop with
    another is only discoverable here. Without this the failure is a bare
    `KeyError` out of `prepared.frame`, mid-replay, from outside the closed
    taxonomy; with it, it is a typed refusal before a strategy is constructed.
    """
    pairs = frozenset(prepared.pairs)
    absent = tuple(
        (symbol, timeframe)
        for symbol in request.symbols for timeframe in dependencies.timeframes
        if (symbol, timeframe) not in pairs
    )
    if absent:
        raise _fail(
            "mismatched_dependencies",
            "the bars were prepared under another dependency closure",
            field="prepared", symbol=absent[0][0], timeframe=absent[0][1],
        )
