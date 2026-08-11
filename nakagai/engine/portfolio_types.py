"""Frozen portfolio replay values, typed replay errors, and their validation.

Every public value here copies the Phase 1 portfolio architecture field for
field, in declaration order, with its exact literals and nullability. The rest
of core and all of platform import these values; nothing downstream
reimplements this validation, this normalization, or the hashing built on top
of it in `nakagai.engine.canonical`.

Three rules run through the whole module:

- every price, cash, risk, cost, and metric value is a finite IEEE 754
  binary64 `float`, and a bool is never accepted where a number is required;
- every semantic timestamp is timezone aware, normalizes to UTC, and carries
  at most microsecond precision, while a schedule session date is a strict
  `datetime.date`;
- ordered collections are stored as tuples and exposed mappings as
  `MappingProxyType` over a copied dict, so a caller cannot reach back into a
  constructed value.

Validation runs in `__post_init__` and normalizes in place, so a constructed
value is always canonical. Invalid input raises `ReplayInputError` before any
strategy is built.
"""

import math
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum, StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Literal

from pandas import NaT, Timestamp

from nakagai.strategies.base import Signal

type JSONScalar = bool | int | float | str | None
type JSONValue = JSONScalar | tuple[JSONValue, ...] | Mapping[str, JSONValue]

_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_MAX_JSON_DEPTH = 64
# The canonical codec emits these two keys for its tagged scalars, so no input
# object may carry them. Refusing at construction keeps a value that cannot be
# hashed from ever existing.
_RESERVED_KEYS = ("$date", "$float")

REQUEST_VERSION = 1
CANDIDATE_ID_PREFIX = "candidate:"
REPLAY_ID_PREFIX = "replay:"
TRADE_ID_PREFIX = "trade:"
REJECTION_ID_PREFIX = "rejection:"
IC_HORIZONS = (1, 5, 20)
IC_MIN_OBSERVATIONS = 10
POOLED_MIN_DAILY = 60
DIRECTIONS = ("long", "short")
CONTEXT_TIMEFRAMES = ("1h", "4h", "1d")
CONTEXT_SOURCES = ("fetched_left_edge", "derived_1h_et_midnight", "session_aligned")
BENCHMARK_KINDS = ("equal_weight_request_symbols", "single_symbol")
PROFIT_FACTOR_STATES = ("finite", "infinite", "unavailable")


class _CodedError(Exception):
    """Base of the closed replay exception taxonomy.

    Every replay failure carries a stable machine code, an operator message,
    and canonical JSON details, so a caller can branch on the code and persist
    the details without reformatting them.
    """

    def __init__(self, code: str, message: str, details: Mapping[str, JSONValue]) -> None:
        if not isinstance(code, str) or not isinstance(message, str):
            raise TypeError("replay error code and message must be strings")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = MappingProxyType(dict(_canonical_details(details)))


class ReplayInputError(_CodedError, ValueError):
    """A request, schedule, or bar input is not a valid replay input."""


class StrategyOutputError(_CodedError, ValueError):
    """A strategy returned a value outside the strict strategy contract."""


class StrategyRuntimeError(_CodedError, RuntimeError):
    """Strategy construction or evaluation raised, so the replay is aborted."""


def _fail(code: str, message: str, **details: JSONValue) -> ReplayInputError:
    return ReplayInputError(code, message, details)


def _canonical_details(details: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    if not isinstance(details, Mapping):
        raise TypeError("replay error details must be a mapping")
    frozen: dict[str, JSONValue] = {}
    for key, value in details.items():
        if not isinstance(key, str):
            raise TypeError("replay error detail keys must be strings")
        frozen[key] = _frozen_json_value(value, field=key, depth=0)
    return frozen


def _frozen_json_value(value: object, *, field: str, depth: int) -> JSONValue:
    """One deep freeze for every arbitrary JSON value core accepts.

    Strategy parameters and error details are the two places where a caller
    hands core a shape core did not declare, so both come through here.
    """
    if depth > _MAX_JSON_DEPTH:
        raise _fail("canonical_value_too_deep", "value nests too deeply", field=field)
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail("nonfinite_binary64", "value must be finite", field=field)
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key, item in value.items():
            if isinstance(key, Enum):
                key = key.value
            if not isinstance(key, str) or isinstance(key, bool):
                raise _fail("invalid_type", "JSON object keys must be strings", field=field)
            if key in _RESERVED_KEYS:
                raise _fail(
                    "reserved_canonical_key", "the codec reserves this key",
                    field=field, key=key,
                )
            if key in frozen:
                raise _fail("duplicate_value", "duplicate normalized key", field=field, key=key)
            frozen[key] = _frozen_json_value(item, field=f"{field}.{key}", depth=depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _frozen_json_value(item, field=f"{field}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    raise _fail(
        "invalid_type", "value is not a JSON value",
        field=field, seen=type(value).__name__,
    )


# --------------------------------------------------------------- field checks


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or isinstance(value, Enum):
        raise _fail("invalid_type", "value must be a string", field=field)
    return value


def _require_name(value: object, field: str) -> str:
    """A nonblank identifier-like string with no surrounding or control space."""
    text = _require_str(value, field)
    if not text or text != text.strip() or any(ord(ch) < 0x20 for ch in text):
        raise _fail("invalid_value", "value must be a nonblank trimmed name", field=field)
    return text


def _require_symbol(value: object, field: str) -> str:
    text = _require_str(value, field)
    if not text or any(ch.isspace() for ch in text):
        raise _fail("invalid_value", "symbol must be nonblank and unspaced", field=field)
    return text.upper()


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail("invalid_type", "value must be an integer", field=field)
    return value


def _require_ordinal(value: object, field: str) -> int:
    ordinal = _require_int(value, field)
    if ordinal < 0:
        raise _fail("invalid_value", "value must not be negative", field=field)
    return ordinal


def _require_positive_int(value: object, field: str) -> int:
    count = _require_int(value, field)
    if count < 1:
        raise _fail("invalid_value", "value must be positive", field=field)
    return count


def _require_binary64(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("invalid_binary64", "value must be binary64", field=field)
    number = float(value)
    if not math.isfinite(number):
        raise _fail("nonfinite_binary64", "value must be finite", field=field)
    return number


def _require_nonnegative(value: object, field: str) -> float:
    number = _require_binary64(value, field)
    if number < 0.0:
        raise _fail("invalid_value", "value must not be negative", field=field)
    return number


def _require_positive(value: object, field: str) -> float:
    number = _require_binary64(value, field)
    if number <= 0.0:
        raise _fail("invalid_value", "value must be positive", field=field)
    return number


def _require_optional_binary64(value: object, field: str) -> float | None:
    return None if value is None else _require_binary64(value, field)


def _require_timestamp(value: object, field: str) -> Timestamp:
    if value is NaT or not isinstance(value, datetime):
        raise _fail("invalid_type", "value must be a timestamp", field=field)
    if value.tzinfo is None or value.utcoffset() is None:
        raise _fail("invalid_timestamp", "timestamp must carry an offset", field=field)
    stamp = value if isinstance(value, Timestamp) else Timestamp(value)
    if stamp.nanosecond:
        raise _fail(
            "invalid_timestamp", "timestamp precision is finer than a microsecond",
            field=field,
        )
    return stamp.tz_convert("UTC")


def _require_optional_timestamp(value: object, field: str) -> Timestamp | None:
    return None if value is None else _require_timestamp(value, field)


def _require_date(value: object, field: str) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise _fail("invalid_type", "value must be a calendar date", field=field)
    return value


def _require_choice(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if isinstance(value, Enum):
        value = value.value
    text = _require_str(value, field)
    if text not in allowed:
        raise _fail(
            "invalid_value", "value is not a supported setting",
            field=field, allowed=allowed,
        )
    return text


def _require_exact_int(value: object, field: str, expected: int) -> int:
    number = _require_int(value, field)
    if number != expected:
        raise _fail("invalid_value", "value is not the supported constant", field=field)
    return number


def _require_enum(value: object, field: str, kind: type[Enum]) -> Enum:
    try:
        return kind(value)
    except ValueError:
        raise _fail(
            "invalid_value", "value is not a member of the closed vocabulary",
            field=field,
        ) from None


def _require_digest(value: object, field: str) -> str:
    text = _require_str(value, field)
    if not _HEX64.match(text):
        raise _fail(
            "invalid_identifier", "value must be 64 lowercase hexadecimal characters",
            field=field,
        )
    return text


def _require_prefixed_digest(value: object, field: str, prefix: str) -> str:
    text = _require_str(value, field)
    if not text.startswith(prefix):
        raise _fail("invalid_identifier", "identifier prefix is wrong", field=field)
    _require_digest(text[len(prefix):], field)
    return text


def _require_uuid7(value: object, field: str) -> str:
    text = _require_str(value, field)
    try:
        parsed = uuid.UUID(text)
    except ValueError:
        raise _fail("invalid_identifier", "value is not a UUID", field=field) from None
    if parsed.version != 7 or parsed.variant != uuid.RFC_4122 or str(parsed) != text:
        raise _fail(
            "invalid_identifier", "value is not a lowercase canonical UUIDv7",
            field=field,
        )
    return text


def _require_items(value: object, field: str, kind: type) -> tuple:
    if not isinstance(value, tuple):
        raise _fail("invalid_type", "value must be a tuple", field=field)
    for index, item in enumerate(value):
        if not isinstance(item, kind):
            raise _fail(
                "invalid_type", "collection holds the wrong value type",
                field=f"{field}[{index}]", expected=kind.__name__,
            )
    return value


def _require_params(value: object, field: str) -> Mapping[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise _fail("invalid_type", "value must be a mapping", field=field)
    return _frozen_json_value(value, field=field, depth=0)


def _require_tags(value: object, field: str) -> tuple[str, ...]:
    tags = _require_items(value, field, str)
    return tuple(_require_name(tag, field) for tag in tags)


def _set(target: object, field: str, value: object) -> None:
    object.__setattr__(target, field, value)


def _set_names(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_name(getattr(target, field), field))


def _set_timestamps(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_timestamp(getattr(target, field), field))


def _set_ordinals(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_ordinal(getattr(target, field), field))


def _set_binary64(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_binary64(getattr(target, field), field))


def _set_positive(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_positive(getattr(target, field), field))


def _set_nonnegative(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_nonnegative(getattr(target, field), field))


def _set_optional_binary64(target: object, *names: str) -> None:
    for field in names:
        _set(target, field, _require_optional_binary64(getattr(target, field), field))


def _require_instance(value: object, field: str, kind: type):
    if not isinstance(value, kind):
        raise _fail(
            "invalid_type", "value must be the declared contract type",
            field=field, expected=kind.__name__,
        )
    return value


# -------------------------------------------------------------- request types


@dataclass(frozen=True)
class ReplayWindow:
    train_start: Timestamp
    train_end: Timestamp
    test_start: Timestamp
    test_end: Timestamp

    def __post_init__(self) -> None:
        _set_timestamps(self, "train_start", "train_end", "test_start", "test_end")
        if not self.train_start < self.train_end:
            raise _fail("invalid_window", "train range must be nonempty")
        if self.train_end != self.test_start:
            raise _fail("invalid_window", "test range must begin where train ends")
        if not self.test_start < self.test_end:
            raise _fail("invalid_window", "test range must be nonempty")


@dataclass(frozen=True)
class ExchangeScheduleIdentity:
    calendar_id: Literal["XNYS"]
    calendar_version: str
    schedule_digest: str
    timezone: Literal["America/New_York"]
    base_timeframe: Literal["15m"]

    def __post_init__(self) -> None:
        _set(self, "calendar_id", _require_choice(self.calendar_id, "calendar_id", ("XNYS",)))
        _set_names(self, "calendar_version")
        _set(self, "schedule_digest", _require_digest(self.schedule_digest, "schedule_digest"))
        _set(self, "timezone", _require_choice(self.timezone, "timezone", ("America/New_York",)))
        _set(self, "base_timeframe",
             _require_choice(self.base_timeframe, "base_timeframe", ("15m",)))


@dataclass(frozen=True)
class ScheduledBaseInterval:
    session_date: date
    interval_ordinal: int
    open_ts: Timestamp
    close_ts: Timestamp

    def __post_init__(self) -> None:
        _set(self, "session_date", _require_date(self.session_date, "session_date"))
        _set_ordinals(self, "interval_ordinal")
        _set_timestamps(self, "open_ts", "close_ts")
        if not self.open_ts < self.close_ts:
            raise _fail("invalid_value", "interval must open before it closes")


@dataclass(frozen=True)
class ScheduledContextBar:
    timeframe: Literal["1h", "4h", "1d"]
    session_date: date
    label_ts: Timestamp
    period_start: Timestamp
    period_end: Timestamp
    available_at: Timestamp
    fresh_context_at: Timestamp | None
    source: Literal["fetched_left_edge", "derived_1h_et_midnight", "session_aligned"]

    def __post_init__(self) -> None:
        _set(self, "timeframe", _require_choice(self.timeframe, "timeframe", CONTEXT_TIMEFRAMES))
        _set(self, "session_date", _require_date(self.session_date, "session_date"))
        _set_timestamps(self, "label_ts", "period_start", "period_end", "available_at")
        _set(self, "fresh_context_at",
             _require_optional_timestamp(self.fresh_context_at, "fresh_context_at"))
        _set(self, "source", _require_choice(self.source, "source", CONTEXT_SOURCES))
        if not self.period_start < self.period_end:
            raise _fail("invalid_value", "context period must be nonempty")
        if self.available_at < self.period_end:
            raise _fail("invalid_value", "a context bar cannot precede its own period end")
        if self.fresh_context_at is not None and self.fresh_context_at < self.available_at:
            raise _fail("invalid_value", "freshness cannot precede availability")


@dataclass(frozen=True)
class ReplaySchedule:
    identity: ExchangeScheduleIdentity
    base_intervals: tuple[ScheduledBaseInterval, ...]
    context_bars: tuple[ScheduledContextBar, ...]

    def __post_init__(self) -> None:
        _require_instance(self.identity, "identity", ExchangeScheduleIdentity)
        _require_items(self.base_intervals, "base_intervals", ScheduledBaseInterval)
        _require_items(self.context_bars, "context_bars", ScheduledContextBar)
        if not self.base_intervals:
            raise _fail("invalid_value", "a schedule needs at least one base interval")


@dataclass(frozen=True)
class PlayRequest:
    play_id: str
    strategy: str
    definition_digest: str
    params: Mapping[str, JSONValue]
    priority: int

    def __post_init__(self) -> None:
        _set_names(self, "play_id", "strategy")
        _set(self, "definition_digest",
             _require_digest(self.definition_digest, "definition_digest"))
        _set(self, "params", _require_params(self.params, "params"))
        _set(self, "priority", _require_int(self.priority, "priority"))


@dataclass(frozen=True)
class AccountPolicy:
    starting_equity: float
    risk_pct: float
    max_open_positions: int
    max_positions_per_play_symbol: Literal[1]
    settlement_model: Literal["cash_t1"]

    def __post_init__(self) -> None:
        _set_positive(self, "starting_equity", "risk_pct")
        if self.risk_pct > 1.0:
            raise _fail("invalid_value", "risk_pct must lie in (0, 1]", field="risk_pct")
        _set(self, "max_open_positions",
             _require_positive_int(self.max_open_positions, "max_open_positions"))
        _set(self, "max_positions_per_play_symbol", _require_exact_int(
            self.max_positions_per_play_symbol, "max_positions_per_play_symbol", 1,
        ))
        _set(self, "settlement_model",
             _require_choice(self.settlement_model, "settlement_model", ("cash_t1",)))


@dataclass(frozen=True)
class SlippageSpec:
    bps: float
    min_per_share: float

    def __post_init__(self) -> None:
        _set_nonnegative(self, "bps", "min_per_share")


@dataclass(frozen=True)
class FeeSpec:
    per_fill: float
    per_share: float

    def __post_init__(self) -> None:
        _set_nonnegative(self, "per_fill", "per_share")


@dataclass(frozen=True)
class ExecutionPolicy:
    arithmetic_version: str
    fill_mode: Literal["pessimistic"]
    slippage: SlippageSpec
    fees: FeeSpec
    funding_order: Literal["play_priority_symbol_signal"]
    missing_bar_policy: Literal["strict"]

    def __post_init__(self) -> None:
        _set_names(self, "arithmetic_version")
        _set(self, "fill_mode", _require_choice(self.fill_mode, "fill_mode", ("pessimistic",)))
        _require_instance(self.slippage, "slippage", SlippageSpec)
        _require_instance(self.fees, "fees", FeeSpec)
        _set(self, "funding_order", _require_choice(
            self.funding_order, "funding_order", ("play_priority_symbol_signal",),
        ))
        _set(self, "missing_bar_policy",
             _require_choice(self.missing_bar_policy, "missing_bar_policy", ("strict",)))


@dataclass(frozen=True)
class BenchmarkSpec:
    kind: Literal["equal_weight_request_symbols", "single_symbol"]
    symbol: str | None
    weighting: Literal["equal"]
    rebalance: Literal["never"]

    def __post_init__(self) -> None:
        _set(self, "kind", _require_choice(self.kind, "kind", BENCHMARK_KINDS))
        _set(self, "weighting", _require_choice(self.weighting, "weighting", ("equal",)))
        _set(self, "rebalance", _require_choice(self.rebalance, "rebalance", ("never",)))
        if self.kind == "single_symbol":
            _set(self, "symbol", _require_symbol(self.symbol, "symbol"))
        elif self.symbol is not None:
            raise _fail(
                "invalid_value", "an equal weight benchmark names no symbol",
                field="symbol",
            )


@dataclass(frozen=True)
class PortfolioReplayRequest:
    request_version: Literal[1]
    replay_id: str
    candidate_id: str
    batch_id: str
    registry_digest: str
    plays: tuple[PlayRequest, ...]
    symbols: tuple[str, ...]
    window: ReplayWindow
    schedule_identity: ExchangeScheduleIdentity
    ic_horizons: tuple[Literal[1], Literal[5], Literal[20]]
    ic_tail_end: Timestamp
    account: AccountPolicy
    execution: ExecutionPolicy
    benchmark: BenchmarkSpec

    def __post_init__(self) -> None:
        _set(self, "request_version",
             _require_exact_int(self.request_version, "request_version", REQUEST_VERSION))
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set(self, "candidate_id",
             _require_prefixed_digest(self.candidate_id, "candidate_id", CANDIDATE_ID_PREFIX))
        _set(self, "batch_id", _require_uuid7(self.batch_id, "batch_id"))
        _set(self, "registry_digest",
             _require_digest(self.registry_digest, "registry_digest"))
        _set(self, "plays", _canonical_plays(self.plays))
        _set(self, "symbols", _canonical_symbols(self.symbols))
        _require_instance(self.window, "window", ReplayWindow)
        _require_instance(self.schedule_identity, "schedule_identity", ExchangeScheduleIdentity)
        if not isinstance(self.ic_horizons, tuple) or self.ic_horizons != IC_HORIZONS:
            raise _fail(
                "invalid_value", "ic_horizons must be the supported horizons",
                field="ic_horizons", allowed=IC_HORIZONS,
            )
        _set_timestamps(self, "ic_tail_end")
        _require_instance(self.account, "account", AccountPolicy)
        _require_instance(self.execution, "execution", ExecutionPolicy)
        _require_instance(self.benchmark, "benchmark", BenchmarkSpec)
        if self.ic_tail_end < self.window.test_end:
            raise _fail(
                "invalid_window", "the IC tail cannot end before the test range",
                field="ic_tail_end",
            )


def _canonical_plays(value: object) -> tuple[PlayRequest, ...]:
    plays = _require_items(value, "plays", PlayRequest)
    if not plays:
        raise _fail("invalid_value", "a request needs at least one play", field="plays")
    seen: set[str] = set()
    for play in plays:
        if play.play_id in seen:
            raise _fail(
                "duplicate_value", "play_id is repeated",
                field="plays", play_id=play.play_id,
            )
        seen.add(play.play_id)
    return tuple(sorted(plays, key=lambda play: (play.priority, play.play_id)))


def _canonical_symbols(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise _fail("invalid_type", "value must be a tuple", field="symbols")
    symbols = [_require_symbol(symbol, "symbols") for symbol in value]
    if not symbols:
        raise _fail("invalid_value", "a request needs at least one symbol", field="symbols")
    if len(set(symbols)) != len(symbols):
        raise _fail("duplicate_value", "symbol is repeated", field="symbols")
    return tuple(sorted(symbols))


# ---------------------------------------------- signal and internal intent


@dataclass(frozen=True)
class EntryIntent:
    replay_id: str
    play_id: str
    strategy: str
    symbol: str
    signal: Signal
    signal_ordinal: int
    signal_ts: Timestamp
    order_type: Literal["market_next_open"]
    eligible_after: Timestamp
    expires_after_intervals: Literal[1]

    def __post_init__(self) -> None:
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set_names(self, "play_id", "strategy")
        _set(self, "symbol", _require_symbol(self.symbol, "symbol"))
        _require_instance(self.signal, "signal", Signal)
        _set_ordinals(self, "signal_ordinal")
        _set_timestamps(self, "signal_ts", "eligible_after")
        _set(self, "order_type",
             _require_choice(self.order_type, "order_type", ("market_next_open",)))
        _set(self, "expires_after_intervals",
             _require_exact_int(self.expires_after_intervals, "expires_after_intervals", 1))
        if self.signal.symbol != self.symbol:
            raise _fail(
                "invalid_value", "the signal names another symbol",
                field="signal", symbol=self.symbol,
            )
        if self.eligible_after < self.signal_ts:
            raise _fail(
                "invalid_value", "eligibility cannot precede the deciding close",
                field="eligible_after",
            )


@dataclass(frozen=True)
class PositionView:
    direction: Literal["long", "short"]
    qty: int
    entry_ts: Timestamp
    entry: float
    initial_stop: float
    initial_target: float
    live_stop: float
    live_target: float

    def __post_init__(self) -> None:
        _set(self, "direction", _require_choice(self.direction, "direction", DIRECTIONS))
        _set(self, "qty", _require_positive_int(self.qty, "qty"))
        _set_timestamps(self, "entry_ts")
        _set_positive(self, "entry", "initial_stop", "initial_target",
                      "live_stop", "live_target")
        _require_entry_geometry(self.direction, self.entry, self.initial_stop,
                                self.initial_target)


@dataclass(frozen=True)
class ManagementDecision:
    action: Literal["hold", "exit"]
    stop: float | None
    target: float | None

    def __post_init__(self) -> None:
        _set(self, "action", _require_choice(self.action, "action", ("hold", "exit")))
        for field in ("stop", "target"):
            value = getattr(self, field)
            _set(self, field, None if value is None else _require_positive(value, field))


def _require_entry_geometry(direction: str, entry: float, stop: float, target: float) -> None:
    """The protective geometry every accepted entry passed at its fill."""
    ordered = stop < entry < target if direction == "long" else target < entry < stop
    if not ordered:
        raise _fail(
            "invalid_value", "protective levels do not bracket the entry fill",
            field="initial_stop", direction=direction,
        )


# ---------------------------------------------------------------- result types


class ExitReason(StrEnum):
    STOP_GAP = "stop_gap"
    TARGET_GAP = "target_gap"
    STOP = "stop"
    TARGET = "target"
    MANAGE = "manage"
    END_OF_WINDOW = "end_of_window"


@dataclass(frozen=True)
class PortfolioTrade:
    trade_id: str
    replay_id: str
    trade_ordinal: int
    play_id: str
    strategy: str
    symbol: str
    signal_ordinal: int
    direction: Literal["long", "short"]
    qty: int
    signal_ts: Timestamp
    entry_ts: Timestamp
    entry: float
    exit_ts: Timestamp
    exit: float
    initial_stop: float
    final_stop: float
    initial_target: float
    final_target: float
    gross_pnl: float
    fees: float
    net_pnl: float
    r_multiple: float
    mae: float
    mfe: float
    setup_tags: tuple[str, ...]
    exit_reason: ExitReason

    def __post_init__(self) -> None:
        _set(self, "trade_id",
             _require_prefixed_digest(self.trade_id, "trade_id", TRADE_ID_PREFIX))
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set_ordinals(self, "trade_ordinal", "signal_ordinal")
        _set_names(self, "play_id", "strategy")
        _set(self, "symbol", _require_symbol(self.symbol, "symbol"))
        _set(self, "direction", _require_choice(self.direction, "direction", DIRECTIONS))
        _set(self, "qty", _require_positive_int(self.qty, "qty"))
        _set_timestamps(self, "signal_ts", "entry_ts", "exit_ts")
        _set_positive(self, "entry", "exit", "initial_stop", "final_stop",
                      "initial_target", "final_target")
        _set_binary64(self, "gross_pnl", "net_pnl", "r_multiple")
        _set_nonnegative(self, "fees", "mae", "mfe")
        _set(self, "setup_tags", _require_tags(self.setup_tags, "setup_tags"))
        _set(self, "exit_reason", _require_enum(self.exit_reason, "exit_reason", ExitReason))
        if self.entry_ts < self.signal_ts:
            raise _fail("invalid_value", "entry cannot precede its signal", field="entry_ts")
        if self.exit_ts <= self.entry_ts:
            raise _fail("invalid_value", "exit cannot precede its entry", field="exit_ts")
        _require_entry_geometry(self.direction, self.entry, self.initial_stop,
                                self.initial_target)


class RejectionReason(StrEnum):
    POSITION_OCCUPIED = "position_occupied"
    PENDING_INTENT_OCCUPIED = "pending_intent_occupied"
    INVALID_PROTECTIVE_GEOMETRY = "invalid_protective_geometry"
    ZERO_QUANTITY = "zero_quantity"
    PORTFOLIO_CAPACITY = "portfolio_capacity"
    UNSETTLED_CASH = "unsettled_cash"
    WINDOW_ENDED = "window_ended"


@dataclass(frozen=True)
class ReplayRejection:
    rejection_id: str
    replay_id: str
    rejection_ordinal: int
    play_id: str
    strategy: str
    symbol: str
    signal_ordinal: int
    signal_ts: Timestamp
    event_ts: Timestamp
    reason: RejectionReason
    required_cash: float | None
    available_cash: float | None
    open_positions: int

    def __post_init__(self) -> None:
        _set(self, "rejection_id", _require_prefixed_digest(
            self.rejection_id, "rejection_id", REJECTION_ID_PREFIX,
        ))
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set_ordinals(self, "rejection_ordinal", "signal_ordinal", "open_positions")
        _set_names(self, "play_id", "strategy")
        _set(self, "symbol", _require_symbol(self.symbol, "symbol"))
        _set_timestamps(self, "signal_ts", "event_ts")
        _set(self, "reason", _require_enum(self.reason, "reason", RejectionReason))
        _set_optional_binary64(self, "required_cash", "available_cash")
        if self.event_ts < self.signal_ts:
            raise _fail("invalid_value", "an event cannot precede its signal", field="event_ts")
        carries_cash = self.reason is RejectionReason.UNSETTLED_CASH
        present = self.required_cash is not None and self.available_cash is not None
        absent = self.required_cash is None and self.available_cash is None
        if carries_cash and not present:
            raise _fail(
                "invalid_value", "an unsettled cash rejection reports both cash values",
                field="required_cash",
            )
        if not carries_cash and not absent:
            raise _fail(
                "invalid_value", "only an unsettled cash rejection reports cash",
                field="required_cash", reason=self.reason.value,
            )


@dataclass(frozen=True)
class EquityPoint:
    replay_id: str
    ts: Timestamp
    point_ordinal: int
    settled_cash: float
    unsettled_cash: float
    short_collateral: float
    positions_liquidation_value: float
    portfolio_equity: float
    gross_exposure: float
    open_positions: int
    benchmark_equity: float

    def __post_init__(self) -> None:
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set_timestamps(self, "ts")
        _set_ordinals(self, "point_ordinal", "open_positions")
        _set_binary64(self, "settled_cash", "unsettled_cash",
                      "positions_liquidation_value", "portfolio_equity",
                      "benchmark_equity")
        _set_nonnegative(self, "short_collateral", "gross_exposure")


@dataclass(frozen=True)
class IcEstimate:
    horizon_bars: Literal[1, 5, 20]
    correlation: float | None
    observations: int

    def __post_init__(self) -> None:
        if self.horizon_bars not in IC_HORIZONS or isinstance(self.horizon_bars, bool):
            raise _fail(
                "invalid_value", "horizon_bars must be a supported horizon",
                field="horizon_bars", allowed=IC_HORIZONS,
            )
        _set_optional_binary64(self, "correlation")
        _set_ordinals(self, "observations")
        if self.correlation is None:
            return
        if not -1.0 <= self.correlation <= 1.0:
            raise _fail(
                "invalid_value", "a rank correlation lies in [-1, 1]", field="correlation",
            )
        if self.observations < IC_MIN_OBSERVATIONS:
            raise _fail(
                "invalid_value", "a correlation needs its minimum pair count",
                field="correlation", observations=self.observations,
            )


@dataclass(frozen=True)
class PortfolioSlice:
    replay_id: str
    play_id: str
    strategy: str
    symbol: str
    signals: int
    trades: int
    rejection_counts: Mapping[RejectionReason, int]
    gross_profit: float
    gross_loss: float
    pre_cost_pnl: float
    net_pnl: float
    fees: float
    win_rate: float | None
    expectancy_r: float | None
    ic: tuple[IcEstimate, IcEstimate, IcEstimate]

    def __post_init__(self) -> None:
        _set(self, "replay_id",
             _require_prefixed_digest(self.replay_id, "replay_id", REPLAY_ID_PREFIX))
        _set_names(self, "play_id", "strategy")
        _set(self, "symbol", _require_symbol(self.symbol, "symbol"))
        _set_ordinals(self, "signals", "trades")
        _set(self, "rejection_counts", _canonical_rejection_counts(self.rejection_counts))
        _set_nonnegative(self, "gross_profit", "gross_loss", "fees")
        _set_binary64(self, "pre_cost_pnl", "net_pnl")
        _set_optional_binary64(self, "win_rate", "expectancy_r")
        _require_cohort_nullability(self.trades, self.win_rate, self.expectancy_r)
        _set(self, "ic", _canonical_ic(self.ic))


def _canonical_rejection_counts(value: object) -> Mapping[RejectionReason, int]:
    if not isinstance(value, Mapping):
        raise _fail("invalid_type", "value must be a mapping", field="rejection_counts")
    counts: dict[RejectionReason, int] = {}
    for reason, count in value.items():
        key = _require_enum(reason, "rejection_counts", RejectionReason)
        if key in counts:
            raise _fail(
                "duplicate_value", "duplicate normalized key", field="rejection_counts",
            )
        counts[key] = _require_ordinal(count, f"rejection_counts.{key.value}")
    return MappingProxyType(counts)


def _canonical_ic(value: object) -> tuple[IcEstimate, ...]:
    estimates = _require_items(value, "ic", IcEstimate)
    if tuple(estimate.horizon_bars for estimate in estimates) != IC_HORIZONS:
        raise _fail(
            "invalid_value", "ic carries one estimate per horizon in order",
            field="ic", allowed=IC_HORIZONS,
        )
    return estimates


def _require_cohort_nullability(
    trades: int, win_rate: float | None, expectancy_r: float | None,
) -> None:
    """No trades means no rate and no expectancy, and any trade means both."""
    if trades == 0 and (win_rate is not None or expectancy_r is not None):
        raise _fail(
            "invalid_value", "an empty cohort reports no rate and no expectancy",
            field="win_rate",
        )
    if trades > 0 and (win_rate is None or expectancy_r is None):
        raise _fail(
            "invalid_value", "a nonempty cohort reports a rate and an expectancy",
            field="win_rate",
        )
    if win_rate is not None and not 0.0 <= win_rate <= 1.0:
        raise _fail("invalid_value", "a win rate lies in [0, 1]", field="win_rate")


@dataclass(frozen=True)
class BenchmarkResult:
    spec: BenchmarkSpec
    total_return: float

    def __post_init__(self) -> None:
        _require_instance(self.spec, "spec", BenchmarkSpec)
        _set_binary64(self, "total_return")


@dataclass(frozen=True)
class TradeStats:
    n_trades: int
    n_wins: int
    win_rate: float | None
    gross_profit: float
    gross_loss: float
    profit_factor: float | None
    profit_factor_state: Literal["finite", "infinite", "unavailable"]
    expectancy_r: float | None

    def __post_init__(self) -> None:
        _set_ordinals(self, "n_trades", "n_wins")
        _set_optional_binary64(self, "win_rate", "profit_factor", "expectancy_r")
        _set_nonnegative(self, "gross_profit", "gross_loss")
        _set(self, "profit_factor_state", _require_choice(
            self.profit_factor_state, "profit_factor_state", PROFIT_FACTOR_STATES,
        ))
        if self.n_wins > self.n_trades:
            raise _fail("invalid_value", "wins cannot exceed trades", field="n_wins")
        _require_cohort_nullability(self.n_trades, self.win_rate, self.expectancy_r)
        if self.gross_loss > 0.0:
            expected = "finite"
        elif self.gross_profit > 0.0:
            expected = "infinite"
        else:
            expected = "unavailable"
        if self.profit_factor_state != expected:
            raise _fail(
                "invalid_value", "profit factor state does not follow the gross sums",
                field="profit_factor_state", expected=expected,
            )
        if (self.profit_factor is None) is (expected == "finite"):
            raise _fail(
                "invalid_value", "only a finite profit factor carries a value",
                field="profit_factor",
            )


@dataclass(frozen=True)
class PortfolioMetrics:
    all_trades: TradeStats
    long_trades: TradeStats
    short_trades: TradeStats
    n_rejections: int
    pre_cost_pnl: float
    fees: float
    net_pnl: float
    starting_equity: float
    ending_equity: float
    total_return: float
    benchmark_return: float
    max_drawdown: float
    ulcer_index: float
    cagr: float
    calmar: float | None
    exposure_pct: float
    avg_holding_hours: float
    daily_n: int
    daily_sum: float
    daily_sum_sq: float
    daily_sum_sq_down: float
    daily_sum_cube: float
    daily_sum_fourth: float
    sharpe: float | None
    sortino: float | None
    psr: float | None
    skew: float | None
    kurtosis: float | None

    def __post_init__(self) -> None:
        for field in ("all_trades", "long_trades", "short_trades"):
            _require_instance(getattr(self, field), field, TradeStats)
        _set_ordinals(self, "n_rejections", "daily_n")
        _set_binary64(self, "pre_cost_pnl", "net_pnl", "ending_equity", "total_return",
                      "benchmark_return", "cagr", "daily_sum", "daily_sum_cube")
        _set_positive(self, "starting_equity")
        _set_nonnegative(self, "fees", "max_drawdown", "ulcer_index", "exposure_pct",
                         "avg_holding_hours", "daily_sum_sq", "daily_sum_sq_down",
                         "daily_sum_fourth")
        _set_optional_binary64(self, "calmar", "sharpe", "sortino", "psr", "skew",
                               "kurtosis")
        if self.cagr < -1.0:
            raise _fail("invalid_value", "cagr cannot fall below total loss", field="cagr")
        if (self.calmar is None) is (self.max_drawdown > 0.0):
            raise _fail(
                "invalid_value", "calmar exists exactly when drawdown is positive",
                field="calmar",
            )
        if self.daily_n < POOLED_MIN_DAILY:
            pooled = ("sharpe", "sortino", "psr", "skew", "kurtosis")
            if any(getattr(self, field) is not None for field in pooled):
                raise _fail(
                    "invalid_value", "pooled statistics need their minimum sample",
                    field="daily_n", minimum=POOLED_MIN_DAILY,
                )


@dataclass(frozen=True)
class PortfolioReplayResult:
    request: PortfolioReplayRequest
    arithmetic_version: str
    fill_mode: str
    schedule_identity: ExchangeScheduleIdentity
    result_digest: str
    trades: tuple[PortfolioTrade, ...]
    rejections: tuple[ReplayRejection, ...]
    equity: tuple[EquityPoint, ...]
    slices: tuple[PortfolioSlice, ...]
    benchmark: BenchmarkResult
    metrics: PortfolioMetrics

    def __post_init__(self) -> None:
        _require_instance(self.request, "request", PortfolioReplayRequest)
        _set_names(self, "arithmetic_version", "fill_mode")
        _require_instance(self.schedule_identity, "schedule_identity",
                          ExchangeScheduleIdentity)
        _set(self, "result_digest", _require_digest(self.result_digest, "result_digest"))
        _require_items(self.trades, "trades", PortfolioTrade)
        _require_items(self.rejections, "rejections", ReplayRejection)
        _require_items(self.equity, "equity", EquityPoint)
        _require_items(self.slices, "slices", PortfolioSlice)
        _require_instance(self.benchmark, "benchmark", BenchmarkResult)
        _require_instance(self.metrics, "metrics", PortfolioMetrics)
        if self.schedule_identity != self.request.schedule_identity:
            raise _fail(
                "invalid_value", "the result echoes another schedule identity",
                field="schedule_identity",
            )
        if not self.equity:
            raise _fail("invalid_value", "a result carries its opening equity point",
                        field="equity")
        _require_event_order(self.trades, "trades", "trade_ordinal")
        _require_event_order(self.rejections, "rejections", "rejection_ordinal")
        _require_event_order(self.equity, "equity", "point_ordinal")
        marks = tuple(point.ts for point in self.equity)
        if any(later < earlier for earlier, later in pairwise(marks)):
            raise _fail(
                "invalid_value", "equity points are appended in chronological order",
                field="equity",
            )
        self._require_attribution()

    def _require_attribution(self) -> None:
        replay_id = self.request.replay_id
        plays = {play.play_id: play.strategy for play in self.request.plays}
        symbols = set(self.request.symbols)
        seen_slices: set[tuple[str, str]] = set()
        for name, rows in (
            ("trades", self.trades), ("rejections", self.rejections),
            ("equity", self.equity), ("slices", self.slices),
        ):
            for row in rows:
                if row.replay_id != replay_id:
                    raise _fail(
                        "invalid_value", "a child row names another replay",
                        field=name, replay_id=row.replay_id,
                    )
                play_id = getattr(row, "play_id", None)
                if play_id is None:
                    continue
                if plays.get(play_id) != row.strategy or row.symbol not in symbols:
                    raise _fail(
                        "invalid_value", "a child row names an unknown attribution key",
                        field=name, play_id=play_id, symbol=row.symbol,
                    )
        for row in self.slices:
            key = (row.play_id, row.symbol)
            if key in seen_slices:
                raise _fail(
                    "duplicate_value", "one slice per play and symbol",
                    field="slices", play_id=row.play_id, symbol=row.symbol,
                )
            seen_slices.add(key)


def _require_event_order(rows: tuple, field: str, ordinal: str) -> None:
    """Event ordinals start at zero and increment once per recorded event."""
    if tuple(getattr(row, ordinal) for row in rows) != tuple(range(len(rows))):
        raise _fail(
            "invalid_value", "event ordinals must start at zero and be contiguous",
            field=field,
        )
