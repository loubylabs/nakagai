"""One ledger, one cash pool: the settlement authority for a portfolio replay.

Every play and every symbol in a replay contends here for the same settled
cash and the same position capacity. Nothing else in core moves cash, opens a
position, closes one, or decides what a fill costs, so two candidates cannot
end up disagreeing about what the account could afford.

Four rules run through the module:

- every cash, quantity, notional, fee, slippage, PnL, and equity value crosses
  `_require_binary64` at the site that produces it, so a nonfinite value is a
  refusal where it was created rather than a NaN discovered later in a metric;
- a refusal costs nothing. A rejected proposal leaves settled cash, capacity,
  and every counter but the rejection ordinal exactly as it found them, so a
  retry is a repeat rather than a second charge;
- the raw bar decides what happened and slippage only decides what it cost. A
  level that the raw print never reached cannot be filled at any price;
- a strategy sees a `PositionView`. The mutable position belongs to the ledger.

The module holds no clock. It settles what it is told is due, prices what it
is told is eligible, and refuses anything it cannot reconcile; the chronology
that decides those things is the replay event loop.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from pandas import Timestamp

from nakagai.data.schema import EXCHANGE_TZ
from nakagai.engine.canonical import rejection_id, trade_id
from nakagai.engine.costs import FeeModel, SlippageModel
from nakagai.engine.portfolio_types import (
    DIRECTIONS,
    EntryIntent,
    ExitReason,
    FeeSpec,
    PortfolioReplayRequest,
    PortfolioTrade,
    PositionView,
    RejectionReason,
    ReplayRejection,
    SlippageSpec,
    _fail,
    _require_binary64,
    _require_choice,
    _require_date,
    _require_enum,
    _require_instance,
    _require_optional_binary64,
    _require_ordinal,
    _require_positive,
    _require_positive_int,
    _require_timestamp,
    _set,
    _set_binary64,
    _set_nonnegative,
    _set_positive,
    brackets_protectively,
)
from nakagai.engine.schedule import ValidatedSchedule

# The key every position, reservation, and capacity rule is stored under. One
# play may trade many symbols and one symbol may be traded by many plays; the
# pair is what holds at most one position.
type PositionKey = tuple[str, str]


def _slippage_model(spec: SlippageSpec) -> SlippageModel:
    """The request's slippage policy as the model that prices a fill."""
    _require_instance(spec, "slippage", SlippageSpec)
    return SlippageModel(bps=spec.bps, min_per_share=spec.min_per_share)


def _fee_model(spec: FeeSpec) -> FeeModel:
    """The request's fee policy as the model that prices a fill."""
    _require_instance(spec, "fees", FeeSpec)
    return FeeModel(per_fill=spec.per_fill, per_share=spec.per_share)


def _entry_fill(direction: str, reference: float, slippage: SlippageModel) -> float:
    """What an opening fill prints. A long buys and a short sells to open."""
    priced = slippage.buy(reference) if direction == "long" else slippage.sell(reference)
    return _require_positive(priced, "fill")


def _exit_fill(direction: str, reference: float, slippage: SlippageModel) -> float:
    """What a closing fill prints. A long sells and a short buys to cover."""
    priced = slippage.sell(reference) if direction == "long" else slippage.buy(reference)
    return _require_positive(priced, "exit")


def _protective_exit(
    direction: str, stop: float, target: float,
    open_price: float, high: float, low: float,
) -> tuple[float, ExitReason] | None:
    """The raw level this bar's geometry triggered, and why, or nothing.

    Two rules, in this order, and the order carries the whole model.

    1. GAP. A bar whose open is already beyond a level never offered that
       level: its first print is the first price the position could have
       traded at, so the open is the reference. Filling at the level instead
       would model overnight and open-gap risk as exactly zero.

    2. INTRABAR. Otherwise the bar traded through from the inside and OHLC
       cannot say which of stop and target came first. Pessimistic mode
       assumes the stop. A bar that opens between the levels and reaches both
       is an intrabar case, not a gap, so rule 1 cannot relax rule 2.

    The value returned is the RAW reference. Slippage prices it afterwards and
    never moves the test that produced it.
    """
    if direction == "long":
        if open_price <= stop:
            return (open_price, ExitReason.STOP_GAP)
        if open_price >= target:
            return (open_price, ExitReason.TARGET_GAP)
        if low <= stop:
            return (stop, ExitReason.STOP)
        if high >= target:
            return (target, ExitReason.TARGET)
        return None
    if open_price >= stop:
        return (open_price, ExitReason.STOP_GAP)
    if open_price <= target:
        return (open_price, ExitReason.TARGET_GAP)
    if high >= stop:
        return (stop, ExitReason.STOP)
    if low <= target:
        return (target, ExitReason.TARGET)
    return None


@dataclass(frozen=True)
class EntryProposal:
    """One candidate, priced and sized at one eligibility open.

    Built only by `_Ledger.propose`, which is the single place the entry side
    of the cost pipeline runs. `reserve` rebuilds the proposal from the frozen
    equity it is given and refuses anything that does not match, so a
    candidate sized against post-fill equity cannot reach the cash pool by
    being handed over beside the earlier number.

    This is also where the ledger's entry path learns that a direction is one
    of the two it knows. A `Signal` deliberately does not validate itself and
    an `EntryIntent` checks only its symbol, so without this every direction
    test downstream would be reading an unchecked string, and the one that
    decides whether a fill posts collateral would silently answer "long".
    """

    intent: EntryIntent
    raw_open: float
    fill: float
    quantity: int
    entry_fee: float
    required_cash: float

    def __post_init__(self) -> None:
        _require_instance(self.intent, "intent", EntryIntent)
        _require_choice(self.intent.signal.direction, "direction", DIRECTIONS)
        _set_positive(self, "raw_open", "fill")
        _set(self, "quantity", _require_ordinal(self.quantity, "quantity"))
        _set_nonnegative(self, "entry_fee", "required_cash")

    @property
    def key(self) -> PositionKey:
        return (self.intent.play_id, self.intent.symbol)

    @property
    def direction(self) -> str:
        return self.intent.signal.direction


@dataclass(frozen=True)
class Reservation:
    """The account's verdict on one proposal.

    Its cash nullability copies `ReplayRejection` exactly, so a refusal can
    become a recorded rejection without a caller deciding which reasons carry
    cash. Only an unsettled-cash refusal does.
    """

    accepted: bool
    reason: RejectionReason | None
    required_cash: float | None
    available_cash: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise _fail("invalid_type", "acceptance must be a boolean", field="accepted")
        if self.reason is not None:
            _set(self, "reason", _require_enum(self.reason, "reason", RejectionReason))
        _set(self, "required_cash",
             _require_optional_binary64(self.required_cash, "required_cash"))
        _set(self, "available_cash",
             _require_optional_binary64(self.available_cash, "available_cash"))
        if self.accepted != (self.reason is None):
            raise _fail(
                "invalid_value", "a reservation is accepted or it names its reason",
                field="reason",
            )
        carries_cash = self.reason is RejectionReason.UNSETTLED_CASH
        present = self.required_cash is not None and self.available_cash is not None
        if carries_cash and not present:
            raise _fail(
                "invalid_value", "an unsettled cash refusal reports both cash values",
                field="required_cash",
            )
        if not carries_cash and (self.required_cash is not None
                                 or self.available_cash is not None):
            raise _fail(
                "invalid_value", "only an unsettled cash refusal reports cash",
                field="required_cash",
            )


@dataclass(frozen=True)
class LedgerSnapshot:
    """The account at one instant, in the shape an `EquityPoint` needs.

    `portfolio_equity` is settled cash plus unsettled credits plus what the
    open positions would liquidate for. That identity is the account mark, not
    a derived convenience, so it is summed here once and never recomputed
    downstream.
    """

    settled_cash: float
    unsettled_cash: float
    short_collateral: float
    positions_liquidation_value: float
    portfolio_equity: float
    gross_exposure: float
    open_positions: int

    def __post_init__(self) -> None:
        _set_nonnegative(self, "settled_cash", "short_collateral", "gross_exposure")
        _set_binary64(self, "unsettled_cash", "positions_liquidation_value",
                      "portfolio_equity")
        _set(self, "open_positions",
             _require_ordinal(self.open_positions, "open_positions"))


@dataclass(frozen=True)
class _Credit:
    """Cash raised by a close, and the session it becomes spendable on.

    A null session means the schedule carries no session after the one the
    close landed in. The credit counts toward equity and never settles, which
    is the honest answer: the replay's calendar simply ends first.
    """

    amount: float
    settles: date | None


@dataclass
class _Position:
    """One open position. Mutable, ledger-owned, never handed to a strategy."""

    intent: EntryIntent
    qty: int
    entry_ts: Timestamp
    entry: float
    entry_fee: float
    collateral: float
    live_stop: float
    live_target: float
    lowest: float
    highest: float

    @property
    def direction(self) -> str:
        return self.intent.signal.direction

    @property
    def initial_stop(self) -> float:
        return self.intent.signal.stop

    @property
    def initial_target(self) -> float:
        return self.intent.signal.target

    def observe(self, high: float, low: float) -> None:
        """Fold one raw observation into the running extremes."""
        self.highest = max(self.highest, high)
        self.lowest = min(self.lowest, low)


class _Ledger:
    """The one account a replay settles through.

    Reservation and fill are two calls because the account commits cash before
    the position exists, and the pair must not come apart: `reserve` holds the
    accepted proposal under its key and `open` consumes exactly that value, so
    a fill can neither arrive without its cash nor arrive twice. `snapshot`
    refuses while any reservation is outstanding rather than marking an
    account whose cash has left and whose position has not arrived.
    """

    def __init__(self, request: PortfolioReplayRequest,
                 schedule: ValidatedSchedule) -> None:
        _require_instance(request, "request", PortfolioReplayRequest)
        _require_instance(schedule, "schedule", ValidatedSchedule)
        if schedule.request != request:
            raise _fail(
                "mismatched_schedule", "the schedule was validated for another request",
                field="schedule",
            )
        self._request = request
        self._schedule = schedule
        self._plays = {play.play_id: play for play in request.plays}
        self._symbols = frozenset(request.symbols)
        self._slippage = _slippage_model(request.execution.slippage)
        self._fees = _fee_model(request.execution.fees)
        self._risk_pct = request.account.risk_pct
        self._max_open_positions = request.account.max_open_positions
        self._starting_equity = _require_positive(
            request.account.starting_equity, "starting_equity",
        )
        self._settled = self._starting_equity
        self._credits: list[_Credit] = []
        self._positions: dict[PositionKey, _Position] = {}
        self._reserved: dict[PositionKey, EntryProposal] = {}
        self._trade_ordinal = 0
        self._rejection_ordinal = 0

    # ------------------------------------------------------------- reading

    @property
    def replay_id(self) -> str:
        return self._request.replay_id

    @property
    def starting_equity(self) -> float:
        return self._starting_equity

    @property
    def settled_cash(self) -> float:
        return self._settled

    @property
    def unsettled_cash(self) -> float:
        total = 0.0
        for credit in self._credits:
            total = _require_binary64(total + credit.amount, "unsettled_cash")
        return total

    @property
    def open_positions(self) -> int:
        """Positions the account is committed to, filled or being filled.

        Reservations count. In the replay chronology a reservation is consumed
        by the fill that follows it immediately, so this is the open position
        count at every point a caller can observe it, and counting the
        outstanding one keeps capacity honest in between.
        """
        return len(self._positions) + len(self._reserved)

    def position_keys(self) -> tuple[PositionKey, ...]:
        """Every open position, in canonical visiting order.

        Play priority, then uppercase symbol, then the entry's signal ordinal.
        Whenever one step visits several positions it visits them in this
        order, so same-timestamp trade and rejection ordinals do not depend on
        dictionary order even where the account arithmetic commutes.
        """
        return tuple(sorted(self._positions, key=self._position_order))

    def view(self, key: PositionKey) -> PositionView:
        """The immutable position a strategy is allowed to see."""
        position = self._require_position(key)
        return PositionView(
            direction=position.direction,
            qty=position.qty,
            entry_ts=position.entry_ts,
            entry=position.entry,
            initial_stop=position.initial_stop,
            initial_target=position.initial_target,
            live_stop=position.live_stop,
            live_target=position.live_target,
        )

    def holds(self, key: PositionKey) -> bool:
        return key in self._positions

    # ------------------------------------------------------------ settling

    def settle_due(self, session_date: date) -> None:
        """Credit every proceed whose settlement session has arrived.

        Called at an interval open, before anything at that open is priced, so
        cash raised on an earlier session can fund a fill on this one and cash
        raised today cannot.
        """
        due = _require_date(session_date, "session_date")
        held: list[_Credit] = []
        for credit in self._credits:
            if credit.settles is not None and credit.settles <= due:
                self._settled = _require_binary64(
                    self._settled + credit.amount, "settled_cash",
                )
            else:
                held.append(credit)
        self._credits = held

    # ------------------------------------------------------------- pricing

    def propose(self, intent: EntryIntent, raw_open: float,
                frozen_equity: float) -> EntryProposal:
        """Price and size one eligible intent at one raw open.

        Every candidate at one open is sized from the same frozen equity, so
        an earlier fill cannot change a later candidate's quantity. Sizing is
        deliberately tolerant here: a candidate whose geometry is broken or
        whose budget buys nothing still becomes a proposal, and `reserve`
        names which rule refused it.
        """
        _require_instance(intent, "intent", EntryIntent)
        self._require_own(intent)
        signal = intent.signal
        raw = _require_positive(raw_open, "raw_open")
        equity = _require_binary64(frozen_equity, "frozen_equity")
        fill = _entry_fill(signal.direction, raw, self._slippage)
        quantity = self._risk_quantity(equity, fill, signal.stop)
        entry_fee = _require_binary64(self._fees.charge(quantity), "entry_fee")
        notional = _require_binary64(quantity * fill, "entry_notional")
        return EntryProposal(
            intent=intent,
            raw_open=raw,
            fill=fill,
            quantity=quantity,
            entry_fee=entry_fee,
            required_cash=_require_binary64(notional + entry_fee, "required_cash"),
        )

    def funding_order(
        self, proposals: Sequence[EntryProposal],
    ) -> tuple[EntryProposal, ...]:
        """Eligible proposals in the order the account funds them.

        Play priority, then uppercase symbol, then the stable signal ordinal.
        The supplied order carries no meaning, so process scheduling, dict
        iteration, and request collection order cannot reach the result.
        """
        for index, proposal in enumerate(proposals):
            _require_instance(proposal, f"proposals[{index}]", EntryProposal)
            self._require_own(proposal.intent)
        return tuple(sorted(proposals,
                            key=lambda item: self._intent_order(item.intent)))

    def pending_order(
        self, intents: Sequence[EntryIntent],
    ) -> tuple[EntryIntent, ...]:
        """Intents in that same funding order, priced or not.

        The window-end sweep holds intents whose eligibility open never arrived,
        so they never became proposals. They still expire in funding order, and
        it is the one this ledger orders everything by rather than a second
        spelling of it.
        """
        for index, intent in enumerate(intents):
            _require_instance(intent, f"intents[{index}]", EntryIntent)
            self._require_own(intent)
        return tuple(sorted(intents, key=self._intent_order))

    def exit_fill(self, direction: str, reference: float) -> float:
        """What closing `direction` at `reference` actually prints."""
        return _exit_fill(
            _require_choice(direction, "direction", DIRECTIONS),
            _require_positive(reference, "reference"), self._slippage,
        )

    def exit_fee(self, qty: int) -> float:
        """What one closing fill of `qty` shares costs.

        `FeeModel.charge` takes the magnitude of what it is given, so a float
        or a negative share count would price a plausible fee rather than
        refuse. The quantity is a whole positive number of shares here.
        """
        return _require_binary64(
            self._fees.charge(_require_positive_int(qty, "qty")), "exit_fee",
        )

    def protective_exit(
        self, key: PositionKey, open_price: float, high: float, low: float,
    ) -> tuple[float, ExitReason] | None:
        """The actual exit fill this bar's raw geometry triggered, and why.

        The levels are the position's live ones, so a ratcheted stop is the
        stop that exits. The raw open, high, and low decide whether a level
        was crossed; slippage prices the reference afterwards and never moves
        the test.

        This answers for the whole bar, so it collapses two separate steps of
        the replay chronology into one predicate. A caller applying open-gap
        exits to positions carried into an interval (step 3) must filter the
        result to `STOP_GAP` and `TARGET_GAP` and act on nothing else; the
        intrabar reasons belong to step 7, after this interval's fills.
        """
        position = self._require_position(key)
        hit = _protective_exit(
            position.direction, position.live_stop, position.live_target,
            _require_positive(open_price, "open"),
            _require_positive(high, "high"),
            _require_positive(low, "low"),
        )
        if hit is None:
            return None
        reference, reason = hit
        return (self.exit_fill(position.direction, reference), reason)

    # ------------------------------------------------------------ committing

    def reserve(self, proposal: EntryProposal, frozen_equity: float) -> Reservation:
        """Run one proposal past every account rule, holding its cash if it passes.

        The gates run in the architecture's order: protective geometry at the
        raw open, then at the modeled fill, then the sizing outcome, then the
        one-position rule, then portfolio capacity, then settled cash. A
        candidate that fails several is reported under the first, and a
        candidate that fails any consumes neither cash nor capacity.
        """
        _require_instance(proposal, "proposal", EntryProposal)
        expected = self.propose(proposal.intent, proposal.raw_open, frozen_equity)
        if proposal != expected:
            raise _fail(
                "inconsistent_proposal",
                "the proposal does not price and size from this frozen equity",
                field="proposal", play_id=proposal.intent.play_id,
                symbol=proposal.intent.symbol,
            )
        key = proposal.key
        if key in self._reserved:
            raise _fail(
                "reservation_outstanding", "this play and symbol already holds cash",
                field="proposal", play_id=key[0], symbol=key[1],
            )
        signal = proposal.intent.signal
        # The raw open is a causal gate rather than a protective exit. A print
        # that broke a level before the entry existed cannot have triggered
        # protection that did not exist yet, and direction-aware slippage
        # moving the modeled fill back inside the range does not change that.
        # The second gate is the mirror: geometry that survives the raw open
        # can still be broken by the fill the account would actually get.
        if not brackets_protectively(proposal.direction, proposal.raw_open,
                                     signal.stop, signal.target):
            return _refused(RejectionReason.INVALID_PROTECTIVE_GEOMETRY)
        if not brackets_protectively(proposal.direction, proposal.fill,
                                     signal.stop, signal.target):
            return _refused(RejectionReason.INVALID_PROTECTIVE_GEOMETRY)
        if proposal.quantity <= 0:
            return _refused(RejectionReason.ZERO_QUANTITY)
        if key in self._positions:
            return _refused(RejectionReason.POSITION_OCCUPIED)
        if self.open_positions >= self._max_open_positions:
            return _refused(RejectionReason.PORTFOLIO_CAPACITY)
        available = self._settled
        if available < proposal.required_cash:
            return Reservation(
                accepted=False, reason=RejectionReason.UNSETTLED_CASH,
                required_cash=proposal.required_cash, available_cash=available,
            )
        self._settled = _require_binary64(
            available - proposal.required_cash, "settled_cash",
        )
        self._reserved[key] = proposal
        return Reservation(accepted=True, reason=None, required_cash=None,
                           available_cash=None)

    def open(self, proposal: EntryProposal, entry_ts: Timestamp) -> PositionView:
        """Fill one reserved proposal, with its protection active immediately.

        A short's slipped opening notional becomes that position's collateral;
        its entry fee is consumed and is never collateral.
        """
        _require_instance(proposal, "proposal", EntryProposal)
        key = proposal.key
        if self._reserved.get(key) != proposal:
            raise _fail(
                "unreserved_fill", "this fill holds no matching reservation",
                field="proposal", play_id=key[0], symbol=key[1],
            )
        # Validate before releasing the reservation. A refused timestamp must
        # not leave the account holding cash against a position it never got.
        stamp = _require_timestamp(entry_ts, "entry_ts")
        del self._reserved[key]
        # Partitioned on "long", the way every other direction test in this
        # module is. Asking whether a direction is short instead would make
        # this the one site that reads an unrecognized value as a long, and
        # the failure would be a short holding no collateral at all.
        collateral = 0.0 if proposal.direction == "long" else _require_binary64(
            proposal.quantity * proposal.fill, "short_collateral",
        )
        signal = proposal.intent.signal
        self._positions[key] = _Position(
            intent=proposal.intent,
            qty=proposal.quantity,
            entry_ts=stamp,
            entry=proposal.fill,
            entry_fee=proposal.entry_fee,
            collateral=collateral,
            live_stop=signal.stop,
            live_target=signal.target,
            # Seeded at the fill, so a position that closes on its own entry
            # bar reports a measured zero rather than an unset sentinel.
            lowest=proposal.fill,
            highest=proposal.fill,
        )
        return self.view(key)

    def observe(self, key: PositionKey, high: float, low: float) -> None:
        """Fold one raw observation into a position's excursion extremes.

        Raw prices, never slipped fills: excursion is a claim about where the
        market went, and the caller decides which observations a given exit
        may credit.
        """
        self._require_position(key).observe(
            _require_positive(high, "high"), _require_positive(low, "low"),
        )

    def adjust(self, key: PositionKey, stop: float | None,
               target: float | None) -> PositionView:
        """Apply a validated management replacement to the live levels.

        A null value keeps the live level. The decision itself was already
        judged at the strategy boundary; what is checked here is the pair the
        position lands in, because the ledger will exit against these levels
        and a crossed pair has no protective meaning.
        """
        position = self._require_position(key)
        live_stop = position.live_stop if stop is None else _require_positive(stop, "stop")
        live_target = (position.live_target if target is None
                       else _require_positive(target, "target"))
        opens_outward = (live_stop < live_target if position.direction == "long"
                         else live_target < live_stop)
        if not opens_outward:
            raise _fail(
                "crossed_protective_levels", "the live stop and target cross each other",
                field="stop", direction=position.direction,
            )
        position.live_stop = live_stop
        position.live_target = live_target
        return self.view(key)

    def close(self, key: PositionKey, exit: float, exit_ts: Timestamp,
              exit_reason: ExitReason, exit_fee: float) -> PortfolioTrade:
        """Settle one position out and record the trade it became.

        A long credits its sale proceeds less the exit fee. A short credits
        the collateral it reserved plus its gross PnL, less the exit fee, so a
        short that lost more than its collateral credits less than it held.
        Either credit settles on T+1 and cannot fund a fill before then.
        """
        position = self._require_position(key)
        fill = _require_positive(exit, "exit")
        stamp = _require_timestamp(exit_ts, "exit_ts")
        reason = _require_enum(exit_reason, "exit_reason", ExitReason)
        closing_fee = _require_binary64(exit_fee, "exit_fee")
        qty = position.qty
        if position.direction == "long":
            gross_pnl = _require_binary64((fill - position.entry) * qty, "gross_pnl")
            credit = _require_binary64(qty * fill - closing_fee, "credit")
        else:
            gross_pnl = _require_binary64((position.entry - fill) * qty, "gross_pnl")
            credit = _require_binary64(
                position.collateral + gross_pnl - closing_fee, "credit",
            )
        fees = _require_binary64(position.entry_fee + closing_fee, "fees")
        net_pnl = _require_binary64(gross_pnl - fees, "net_pnl")
        # The signal's stop, never the ratcheted one. R has to mean the same
        # thing in r_multiple and in the excursions, or one trade's own fields
        # disagree about the size of 1R.
        risk_per_share = _require_positive(
            abs(position.entry - position.initial_stop), "protective_risk",
        )
        risk = _require_positive(risk_per_share * qty, "protective_risk")
        # Build the trade before the account moves. A trade the contract
        # refuses must leave the position open and the cash where it was,
        # rather than settling a close that produced no record.
        trade = PortfolioTrade(
            trade_id=trade_id(self.replay_id, key[0], key[1],
                              position.intent.signal_ordinal),
            replay_id=self.replay_id,
            trade_ordinal=self._trade_ordinal,
            play_id=key[0],
            strategy=position.intent.strategy,
            symbol=key[1],
            signal_ordinal=position.intent.signal_ordinal,
            direction=position.direction,
            qty=qty,
            signal_ts=position.intent.signal_ts,
            entry_ts=position.entry_ts,
            entry=position.entry,
            exit_ts=stamp,
            exit=fill,
            initial_stop=position.initial_stop,
            final_stop=position.live_stop,
            initial_target=position.initial_target,
            final_target=position.live_target,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            r_multiple=_require_binary64(net_pnl / risk, "r_multiple"),
            mae=self._excursion(position, adverse=True, risk_per_share=risk_per_share),
            mfe=self._excursion(position, adverse=False, risk_per_share=risk_per_share),
            setup_tags=position.intent.signal.setup_tags,
            exit_reason=reason,
        )
        del self._positions[key]
        self._credits.append(_Credit(credit, self._settlement_session(stamp)))
        self._trade_ordinal += 1
        return trade

    def reject(self, intent: EntryIntent, reason: RejectionReason,
               event_ts: Timestamp, required_cash: float | None,
               available_cash: float | None) -> ReplayRejection:
        """Record one signal the account declined. Nothing else moves.

        Cash and capacity are untouched by construction: this method reads the
        account and writes only the rejection ordinal.
        """
        _require_instance(intent, "intent", EntryIntent)
        self._require_own(intent)
        rejection = ReplayRejection(
            rejection_id=rejection_id(
                self.replay_id, intent.play_id, intent.symbol,
                intent.signal_ordinal, reason,
            ),
            replay_id=self.replay_id,
            rejection_ordinal=self._rejection_ordinal,
            play_id=intent.play_id,
            strategy=intent.strategy,
            symbol=intent.symbol,
            signal_ordinal=intent.signal_ordinal,
            signal_ts=intent.signal_ts,
            event_ts=event_ts,
            reason=reason,
            required_cash=required_cash,
            available_cash=available_cash,
            open_positions=self.open_positions,
        )
        self._rejection_ordinal += 1
        return rejection

    # ------------------------------------------------------------- marking

    def snapshot(self, ts: Timestamp, marks: Mapping[str, float]) -> LedgerSnapshot:
        """Mark the whole account at one instant against one raw price per symbol.

        Hypothetical exit slippage and fees are inside the liquidation value,
        because that is what the positions are worth to this account; they do
        not become realized fees until the positions actually close.
        """
        stamp = _require_timestamp(ts, "ts")
        if not isinstance(marks, Mapping):
            raise _fail("invalid_type", "marks must be a mapping", field="marks")
        if self._reserved:
            raise _fail(
                "reservation_outstanding",
                "the account cannot be marked between a reservation and its fill",
                field="ts",
            )
        session = stamp.tz_convert(EXCHANGE_TZ).date()
        if any(credit.settles is not None and credit.settles <= session
               for credit in self._credits):
            # Marking before settling would count matured cash as settled and
            # unsettled at once. The order is the caller's to get right.
            raise _fail(
                "settlement_not_applied", "a credit is due at this point and unsettled",
                field="ts",
            )
        collateral = 0.0
        liquidation = 0.0
        exposure = 0.0
        for key in self.position_keys():
            position = self._positions[key]
            mark = _require_positive(self._mark_for(marks, key[1]), "mark")
            collateral = _require_binary64(
                collateral + position.collateral, "short_collateral",
            )
            liquidation = _require_binary64(
                liquidation + self._liquidation_value(position, mark),
                "positions_liquidation_value",
            )
            exposure = _require_binary64(
                exposure + abs(position.qty * mark), "gross_exposure",
            )
        unsettled = self.unsettled_cash
        return LedgerSnapshot(
            settled_cash=self._settled,
            unsettled_cash=unsettled,
            short_collateral=collateral,
            positions_liquidation_value=liquidation,
            portfolio_equity=_require_binary64(
                self._settled + unsettled + liquidation, "portfolio_equity",
            ),
            gross_exposure=exposure,
            open_positions=len(self._positions),
        )

    # ------------------------------------------------------------- internals

    def _liquidation_value(self, position: _Position, mark: float) -> float:
        """What one position would raise if it closed at this raw mark."""
        fee = self.exit_fee(position.qty)
        fill = _exit_fill(position.direction, mark, self._slippage)
        if position.direction == "long":
            return _require_binary64(position.qty * fill - fee, "liquidation_value")
        return _require_binary64(
            position.collateral + (position.entry - fill) * position.qty - fee,
            "liquidation_value",
        )

    def _mark_for(self, marks: Mapping[str, float], symbol: str) -> float:
        if symbol not in marks:
            raise _fail(
                "missing_mark", "no mark was supplied for an open position",
                field="marks", symbol=symbol,
            )
        return marks[symbol]

    def _risk_quantity(self, equity: float, fill: float, stop: float) -> int:
        """Whole shares the risk budget buys at this fill and this stop.

        Fees affect what the account can afford and never the protective-risk
        denominator, so this is budget over per-share risk, floored. A budget
        or a distance that cannot produce a share yields zero, which `reserve`
        reports as `zero_quantity` rather than raising here.
        """
        risk_per_share = _require_binary64(abs(fill - stop), "protective_risk")
        budget = _require_binary64(self._risk_pct * equity, "risk_budget")
        if risk_per_share <= 0.0 or budget <= 0.0:
            return 0
        return max(0, math.floor(_require_binary64(budget / risk_per_share, "quantity")))

    def _excursion(self, position: _Position, *, adverse: bool,
                   risk_per_share: float) -> float:
        """One excursion as a nonnegative R magnitude against the initial stop.

        A long suffers on the way down and gains on the way up; a short is the
        mirror. Both are measured from the slipped entry fill to a raw
        observed extreme.
        """
        against_the_long = adverse if position.direction == "long" else not adverse
        travel = (position.entry - position.lowest if against_the_long
                  else position.highest - position.entry)
        return _require_binary64(
            max(0.0, _require_binary64(travel, "excursion")) / risk_per_share,
            "excursion",
        )

    def _settlement_session(self, exit_ts: Timestamp) -> date | None:
        """The session a close raised on this instant becomes spendable.

        Weekends and exchange holidays are not settlement days, so the answer
        comes from the schedule's own sessions. Past the last session it
        carries there is no answer, and a credit with none never settles.
        """
        session = exit_ts.tz_convert(EXCHANGE_TZ).date()
        if session >= self._schedule.sessions[-1]:
            return None
        return self._schedule.next_session(session)

    def _require_position(self, key: object) -> _Position:
        if not isinstance(key, tuple) or len(key) != 2:
            raise _fail("invalid_type", "a position key is a play and a symbol",
                        field="key")
        position = self._positions.get(key)
        if position is None:
            raise _fail(
                "unknown_position", "this play and symbol holds no position",
                field="key", play_id=str(key[0]), symbol=str(key[1]),
            )
        return position

    def _require_own(self, intent: EntryIntent) -> None:
        """Refuse an intent that does not belong to this replay.

        Trades and rejections are stamped with this ledger's replay identity,
        so an intent from another replay would be silently reattributed here.
        """
        if intent.replay_id != self.replay_id:
            raise _fail("foreign_intent", "the intent belongs to another replay",
                        field="replay_id")
        play = self._plays.get(intent.play_id)
        if play is None:
            raise _fail("unknown_play", "the request declares no such play",
                        field="play_id", play_id=intent.play_id)
        if play.strategy != intent.strategy:
            raise _fail(
                "strategy_mismatch", "the intent names another strategy than its play",
                field="strategy", play_id=intent.play_id, strategy=intent.strategy,
            )
        if intent.symbol not in self._symbols:
            raise _fail("unknown_symbol", "the request does not trade this symbol",
                        field="symbol", symbol=intent.symbol)

    def _intent_order(self, intent: EntryIntent) -> tuple[int, str, int]:
        """Play priority, uppercase symbol, signal ordinal: the ONE order.

        Funding an eligible proposal, expiring a pending intent, and visiting an
        open position are three views of the same question, so they read one
        key. Two spellings of it could disagree at a same-timestamp contention,
        which is exactly where the ordering is load bearing.
        """
        return (self._plays[intent.play_id].priority, intent.symbol,
                intent.signal_ordinal)

    def _position_order(self, key: PositionKey) -> tuple[int, str, int]:
        return self._intent_order(self._positions[key].intent)


def _refused(reason: RejectionReason) -> Reservation:
    """A refusal that carries no cash, which is every reason but one."""
    return Reservation(accepted=False, reason=reason, required_cash=None,
                       available_cash=None)
