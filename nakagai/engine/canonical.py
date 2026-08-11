"""The one canonical codec, identifier formulas, and strict transport.

Two encodings live here and they answer different questions.

`canonical_replay_bytes` answers "are these two results the same result?" It is
the hashing encoding: object keys sort lexically, a finite binary64 value
becomes its exact `float.hex()` inside a tagged `{"$float": ...}` object, a
calendar date becomes a tagged `{"$date": ...}` object, and a timestamp
becomes one UTC spelling. Nothing about it depends on a runtime's choice of
decimal rendering, so local, remote, and Fly agree byte for byte or they
disagree loudly.

`encode_replay_*` and `decode_replay_*` answer "how does this cross a wire?"
They are ordinary strict JSON: real numbers, ISO timestamp strings, and
`YYYY-MM-DD` dates, so an API, a worker envelope, and a database column can
carry them. A receiver recomputes the canonical bytes and the digest from the
decoded values rather than trusting the transport text.

Identifiers are derived here and nowhere else. Platform calls
`expected_candidate_id` and `expected_replay_id` rather than reimplementing
either formula, and core derives every child identifier from the accepted
request.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from functools import cache
from types import UnionType
from typing import Literal, get_args, get_origin, get_type_hints

from pandas import Timestamp

from nakagai.engine.portfolio_types import (
    CANDIDATE_ID_PREFIX,
    REJECTION_ID_PREFIX,
    REPLAY_ID_PREFIX,
    TRADE_ID_PREFIX,
    JSONValue,
    PortfolioReplayRequest,
    PortfolioReplayResult,
    PortfolioSlice,
    RejectionReason,
    ReplaySchedule,
    _fail,
    _MAX_JSON_DEPTH,
    _require_digest,
    _require_encodable,
    _require_enum,
    _require_instance,
    _require_name,
    _require_ordinal,
    _require_params,
    _require_prefixed_digest,
    _require_symbol,
    _require_timestamp,
    _RESERVED_KEYS,
)

_DATE_TEXT = re.compile(r"\A\d{4}-\d{2}-\d{2}\Z", re.ASCII)
_TIMESTAMP_TEXT = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z\Z", re.ASCII,
)


# ------------------------------------------------------------ canonical bytes


def canonical_replay_bytes(value: object) -> bytes:
    normalized = _canonical_value(value, path="$")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_value(value: object, *, path: str, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise _fail("canonical_value_too_deep", "value nests too deeply", field=path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return _enum_text(value, path)
    if isinstance(value, str):
        return _require_encodable(value, path)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return {"$float": _require_finite(value, path).hex()}
    if isinstance(value, datetime):
        return _timestamp_text(value, path)
    if isinstance(value, date):
        return {"$date": _date_text(value)}
    if isinstance(value, Mapping):
        return _canonical_object(value, path=path, depth=depth)
    if isinstance(value, (tuple, list)):
        return [
            _canonical_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise _fail(
        "unsupported_canonical_value", "value has no canonical encoding",
        field=path, seen=type(value).__name__,
    )


def _canonical_object(value: Mapping, *, path: str, depth: int) -> dict:
    encoded: dict[str, object] = {}
    for key, item in value.items():
        name = _canonical_key(key, path)
        if name in encoded:
            raise _fail(
                "duplicate_value", "two keys normalize to one canonical key",
                field=path, key=name,
            )
        encoded[name] = _canonical_value(item, path=f"{path}.{name}", depth=depth + 1)
    return encoded


def _canonical_key(key: object, path: str) -> str:
    if isinstance(key, Enum):
        key = _enum_text(key, path)
    if not isinstance(key, str) or isinstance(key, bool):
        raise _fail(
            "invalid_type", "canonical object keys must be strings",
            field=path, seen=type(key).__name__,
        )
    if key in _RESERVED_KEYS:
        raise _fail(
            "reserved_canonical_key", "the codec reserves this key for tagged scalars",
            field=path, key=key,
        )
    return _require_encodable(key, path)


def _enum_text(value: Enum, path: str) -> str:
    if not isinstance(value.value, str):
        raise _fail(
            "unsupported_canonical_value", "an enum encodes as its string value",
            field=path, seen=type(value).__name__,
        )
    return value.value


def _require_finite(value: float, path: str) -> float:
    """A finite binary64 whose zero, if it is one, is positive.

    NaN and infinity have no canonical encoding, so they are refused. Negative
    zero has one, and that is the problem: `float.hex()` spells it
    `-0x0.0p+0`, so a result carrying one hashes differently from the identical
    result that carried a positive zero. Two workers computing the same numbers
    would report different digests and the disagreement would read as a real
    divergence.

    Refused rather than normalized, and the difference is deliberate. Folding
    `-0.0` onto `0.0` inside the encoder would make two different bit patterns
    hash alike, and the encoder's whole job is to be injective: it exists so
    that a value which differs in its last bit differs in its bytes. So the
    refusal sits UPSTREAM of the spelling. Every result crosses this function
    on its way to a digest, which makes it the one chokepoint where a stray
    negative zero anywhere in a trade, a rejection, or an equity point becomes
    a typed error instead of a silent hash divergence.

    The producers hold up their end: the metric fold accumulates losses as
    positive magnitudes and the IC lens adds zero to its rounded coefficient,
    so nothing in a correct replay ever reaches this branch.
    """
    number = float(value)
    if not math.isfinite(number):
        raise _fail("nonfinite_binary64", "value must be finite", field=path)
    if number == 0.0 and math.copysign(1.0, number) < 0.0:
        raise _fail("negative_zero", "a canonical zero is positive", field=path)
    return number


def _timestamp_text(value: datetime, path: str) -> str:
    stamp = _require_timestamp(value, path)
    return (
        f"{stamp.year:04d}-{stamp.month:02d}-{stamp.day:02d}"
        f"T{stamp.hour:02d}:{stamp.minute:02d}:{stamp.second:02d}"
        f".{stamp.microsecond:06d}Z"
    )


def _date_text(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}-{value.day:02d}"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_replay_bytes(value)).hexdigest()


# ---------------------------------------------------------------- projections


def _projection(value: object) -> object:
    """The contract shape of one value: named fields, ordered arrays, leaves.

    Leaves stay as their Python values, so the canonical codec is the only
    thing that decides how a date, a timestamp, or a binary64 is spelled.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _projection(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {key: _projection(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_projection(item) for item in value]
    return value


def _schedule_projection(schedule: ReplaySchedule) -> dict:
    """The schedule without its identity, which would otherwise self-reference."""
    return {
        "base_intervals": [
            (row.session_date, row.interval_ordinal, row.open_ts, row.close_ts)
            for row in schedule.base_intervals
        ],
        "context_bars": [
            (row.timeframe, row.session_date, row.label_ts, row.period_start,
             row.period_end, row.available_at, row.fresh_context_at, row.source)
            for row in schedule.context_bars
        ],
    }


def _candidate_projection(request: PortfolioReplayRequest) -> dict:
    """What makes one candidate one candidate, across all of its windows.

    Batch, replay, window, schedule digest, and IC tail are deliberately
    absent: a reporting window must not fork the candidate identity.
    """
    return {
        "request_version": request.request_version,
        "plays": [_projection(play) for play in request.plays],
        "symbols": list(request.symbols),
        "account": _projection(request.account),
        "execution": _projection(request.execution),
        "benchmark": _projection(request.benchmark),
        "registry_digest": request.registry_digest,
        "calendar_id": request.schedule_identity.calendar_id,
        "calendar_version": request.schedule_identity.calendar_version,
        "ic_horizons": list(request.ic_horizons),
    }


def _result_projection(result: PortfolioReplayResult) -> dict:
    """The complete semantic result, with only its own digest field omitted.

    Slice order is the one canonical sort applied here. Plays, symbols, event
    ordinals, and IC horizons are already canonically ordered by the value
    types themselves, so re-sorting them could only ever be a no-op.
    """
    return {
        "request": _projection(result.request),
        "arithmetic_version": result.arithmetic_version,
        "fill_mode": result.fill_mode,
        "schedule_identity": _projection(result.schedule_identity),
        "trades": [_projection(row) for row in result.trades],
        "rejections": [_projection(row) for row in result.rejections],
        "equity": [_projection(row) for row in result.equity],
        "slices": [_projection(row) for row in sorted(result.slices, key=_slice_order)],
        "benchmark": _projection(result.benchmark),
        "metrics": _projection(result.metrics),
    }


def _slice_order(row: PortfolioSlice) -> tuple[str, str]:
    return (row.play_id, row.symbol)


# ---------------------------------------------------------------- identifiers


def schedule_digest(schedule: ReplaySchedule) -> str:
    _require_instance(schedule, "schedule", ReplaySchedule)
    return _digest(_schedule_projection(schedule))


def expected_candidate_id(request: PortfolioReplayRequest) -> str:
    _require_instance(request, "request", PortfolioReplayRequest)
    return CANDIDATE_ID_PREFIX + _digest(_candidate_projection(request))


def expected_replay_id(request: PortfolioReplayRequest) -> str:
    _require_instance(request, "request", PortfolioReplayRequest)
    return REPLAY_ID_PREFIX + _digest((
        request.batch_id,
        request.candidate_id,
        _projection(request.window),
        _projection(request.schedule_identity),
        request.ic_tail_end,
    ))


def trade_id(replay_id: str, play_id: str, symbol: str, signal_ordinal: int) -> str:
    return TRADE_ID_PREFIX + _digest(
        _child_key(replay_id, play_id, symbol, signal_ordinal),
    )


def rejection_id(
    replay_id: str, play_id: str, symbol: str, signal_ordinal: int,
    reason: RejectionReason,
) -> str:
    key = _child_key(replay_id, play_id, symbol, signal_ordinal)
    return REJECTION_ID_PREFIX + _digest(
        key + (_require_enum(reason, "reason", RejectionReason),),
    )


def _child_key(
    replay_id: str, play_id: str, symbol: str, signal_ordinal: int,
) -> tuple:
    """One signal is one child, so its parts are validated before hashing."""
    return (
        _require_prefixed_digest(replay_id, "replay_id", REPLAY_ID_PREFIX),
        _require_name(play_id, "play_id"),
        _require_canonical_symbol(symbol, "symbol"),
        _require_ordinal(signal_ordinal, "signal_ordinal"),
    )


def _require_canonical_symbol(value: object, field: str) -> str:
    symbol = _require_symbol(value, field)
    if symbol != value:
        raise _fail(
            "invalid_value", "symbol must already be canonical", field=field,
        )
    return symbol


def definition_digest(base_digest: str, params: Mapping[str, JSONValue]) -> str:
    return _digest((
        _require_digest(base_digest, "base_digest"),
        _require_params(params, "params"),
    ))


def result_digest(result: PortfolioReplayResult) -> str:
    _require_instance(result, "result", PortfolioReplayResult)
    return _digest(_result_projection(result))


# ------------------------------------------------------------------ transport


def encode_replay_request(request: PortfolioReplayRequest) -> dict[str, JSONValue]:
    _require_instance(request, "request", PortfolioReplayRequest)
    return _transport_value(request, path="$")


def decode_replay_request(value: Mapping[str, object]) -> PortfolioReplayRequest:
    return _decode_dataclass(PortfolioReplayRequest, value, path="$")


def encode_replay_schedule(schedule: ReplaySchedule) -> dict[str, JSONValue]:
    _require_instance(schedule, "schedule", ReplaySchedule)
    return _transport_value(schedule, path="$")


def decode_replay_schedule(value: Mapping[str, object]) -> ReplaySchedule:
    return _decode_dataclass(ReplaySchedule, value, path="$")


def encode_replay_result(result: PortfolioReplayResult) -> dict[str, JSONValue]:
    _require_instance(result, "result", PortfolioReplayResult)
    return _transport_value(result, path="$")


def decode_replay_result(value: Mapping[str, object]) -> PortfolioReplayResult:
    return _decode_dataclass(PortfolioReplayResult, value, path="$")


def _transport_value(value: object, *, path: str, depth: int = 0):
    if depth > _MAX_JSON_DEPTH:
        raise _fail("canonical_value_too_deep", "value nests too deeply", field=path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return _enum_text(value, path)
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _require_finite(value, path)
    if isinstance(value, datetime):
        return _timestamp_text(value, path)
    if isinstance(value, date):
        return _date_text(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _transport_value(
                getattr(value, f.name), path=f"{path}.{f.name}", depth=depth + 1,
            )
            for f in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            _canonical_key(key, path): _transport_value(
                item, path=f"{path}.{key}", depth=depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _transport_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise _fail(
        "unsupported_canonical_value", "value has no transport encoding",
        field=path, seen=type(value).__name__,
    )


@cache
def _annotations(cls: type) -> tuple[tuple[str, object], ...]:
    hints = get_type_hints(cls)
    return tuple((f.name, hints[f.name]) for f in fields(cls))


def _decode_dataclass(cls: type, value: object, *, path: str):
    if not isinstance(value, Mapping):
        raise _fail(
            "invalid_type", "value must be a JSON object",
            field=path, expected=cls.__name__,
        )
    declared = _annotations(cls)
    names = {name for name, _ in declared}
    unknown = sorted(key for key in value if key not in names)
    if unknown:
        raise _fail("invalid_value", "value carries unknown keys", field=path, keys=unknown)
    missing = sorted(name for name in names if name not in value)
    if missing:
        raise _fail("invalid_value", "value is missing keys", field=path, keys=missing)
    return cls(**{
        name: _decode_annotation(annotation, value[name], path=f"{path}.{name}")
        for name, annotation in declared
    })


def _decode_annotation(annotation: object, value: object, *, path: str):
    if annotation is Timestamp:
        return _decode_timestamp(value, path)
    if annotation is date:
        return _decode_date(value, path)
    if annotation is float:
        return _decode_number(value, path)
    if annotation is int:
        return _decode_int(value, path)
    if annotation is str:
        return _decode_text(value, path)
    if annotation is JSONValue:
        return _decode_json_value(value, path=path)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return _require_enum(_decode_text(value, path), path, annotation)
    if is_dataclass(annotation):
        return _decode_dataclass(annotation, value, path=path)
    origin = get_origin(annotation)
    if origin is Literal:
        return _decode_literal(value, path, get_args(annotation))
    if origin is UnionType:
        return _decode_optional(value, path, get_args(annotation))
    if origin is tuple:
        return _decode_tuple(value, path, get_args(annotation))
    if origin is not None and issubclass(origin, Mapping):
        return _decode_mapping(value, path, get_args(annotation))
    raise _fail(
        "invalid_type", "the contract has no transport rule for this field",
        field=path, expected=str(annotation),
    )


def _decode_timestamp(value: object, path: str) -> Timestamp:
    text = _decode_text(value, path)
    if not _TIMESTAMP_TEXT.match(text):
        raise _fail(
            "invalid_timestamp", "a transported timestamp is exact UTC microseconds",
            field=path,
        )
    try:
        return Timestamp(text)
    except ValueError:
        raise _fail("invalid_timestamp", "value is not a real instant", field=path) from None


def _decode_date(value: object, path: str) -> date:
    text = _decode_text(value, path)
    if not _DATE_TEXT.match(text):
        raise _fail("invalid_value", "a session date is strict YYYY-MM-DD", field=path)
    try:
        return date(int(text[0:4]), int(text[5:7]), int(text[8:10]))
    except ValueError:
        raise _fail("invalid_value", "value is not a real calendar date", field=path) from None


def _decode_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("invalid_binary64", "value must be a JSON number", field=path)
    return _require_finite(float(value), path)


def _decode_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail("invalid_type", "value must be a JSON integer", field=path)
    return value


def _decode_text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise _fail("invalid_type", "value must be a JSON string", field=path)
    return _require_encodable(value, path)


def _decode_literal(value: object, path: str, allowed: tuple) -> object:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value not in allowed:
        raise _fail(
            "invalid_value", "value is not a supported setting",
            field=path, allowed=list(allowed),
        )
    return value


def _decode_optional(value: object, path: str, allowed: tuple) -> object:
    present = tuple(item for item in allowed if item is not type(None))
    if len(present) != len(allowed) - 1 or len(present) != 1:
        raise _fail(
            "invalid_type", "the contract has no transport rule for this field",
            field=path, expected=str(allowed),
        )
    if value is None:
        return None
    return _decode_annotation(present[0], value, path=path)


def _decode_tuple(value: object, path: str, allowed: tuple) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise _fail("invalid_type", "value must be a JSON array", field=path)
    if len(allowed) == 2 and allowed[1] is Ellipsis:
        return tuple(
            _decode_annotation(allowed[0], item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if len(value) != len(allowed):
        raise _fail(
            "invalid_value", "value has the wrong number of entries",
            field=path, expected=len(allowed),
        )
    return tuple(
        _decode_annotation(annotation, item, path=f"{path}[{index}]")
        for index, (annotation, item) in enumerate(zip(allowed, value, strict=True))
    )


def _decode_mapping(value: object, path: str, allowed: tuple) -> dict:
    if not isinstance(value, Mapping):
        raise _fail("invalid_type", "value must be a JSON object", field=path)
    key_annotation, value_annotation = allowed
    decoded: dict = {}
    for key, item in value.items():
        name = _decode_text(key, path)
        decoded_key = (
            name if key_annotation is str
            else _decode_annotation(key_annotation, name, path=path)
        )
        if decoded_key in decoded:
            raise _fail(
                "duplicate_value", "two keys normalize to one canonical key",
                field=path, key=name,
            )
        decoded[decoded_key] = _decode_annotation(
            value_annotation, item, path=f"{path}.{name}",
        )
    return decoded


def _decode_json_value(value: object, *, path: str, depth: int = 0) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise _fail("canonical_value_too_deep", "value nests too deeply", field=path)
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _decode_number(value, path)
    if isinstance(value, (list, tuple)):
        return tuple(
            _decode_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return {
            _decode_text(key, path): _decode_json_value(
                item, path=f"{path}.{key}", depth=depth + 1,
            )
            for key, item in value.items()
        }
    raise _fail(
        "invalid_type", "value is not a JSON value",
        field=path, seen=type(value).__name__,
    )
