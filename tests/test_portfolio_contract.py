"""The frozen portfolio contract: fields, canonical bytes, and identifiers.

Everything downstream of Task C1 imports these values, so this module pins the
architecture literally. The field map reproduces the specification class by
class, the codec goldens pin exact bytes, and the identifier goldens pin exact
strings. A change that moves any of them is a contract change and has to be
argued for, rather than absorbed.
"""

import dataclasses
import inspect
import json
import math
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Literal, get_type_hints

import pandas as pd
import pytest

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
from nakagai.strategies.base import Direction
from pandas import Timestamp
from tests.portfolio_fixtures import (
    PLACEHOLDER_DIGEST,
    base_account,
    base_benchmark,
    base_execution,
    base_identity,
    base_intervals,
    base_metrics,
    base_plays,
    base_request,
    base_result,
    base_schedule,
    base_window,
    ts,
)

CANDIDATE_GOLDEN = (
    "candidate:713c60c014a9506a8c7ecb7b8fa61e746192abc68f2b3747eaf23785efc89100"
)
REPLAY_GOLDEN = (
    "replay:5d4bc974357618a640b90ca8ef1f74d3d19871283d38240af7885f2892562595"
)
TRADE_GOLDEN = (
    "trade:1b2511d285e8c23d88289f6c17b198e9f400c0c3891febfde77c84ac6e8947f2"
)
REJECTION_GOLDEN = (
    "rejection:3b96ef68973c6ef597935f8660a7ca363e5896d245c29dbe1ab3b6992889dfbb"
)
SCHEDULE_DIGEST_GOLDEN = (
    "6a3a37dc10d6ba92955e7a840d2a4e5aa58a750b1164d3ef4bd432ce69d51c21"
)
RESULT_DIGEST_GOLDEN = (
    "d0e90f1dd5ffd1f4cbdff74878e27a9aad96afb065d01f22989ceda5ce547e1f"
)

EMPTY = inspect.Parameter.empty

PUBLIC_FIELDS = {
    ReplayWindow: (
        ("train_start", Timestamp), ("train_end", Timestamp),
        ("test_start", Timestamp), ("test_end", Timestamp),
    ),
    ExchangeScheduleIdentity: (
        ("calendar_id", Literal["XNYS"]), ("calendar_version", str),
        ("schedule_digest", str), ("timezone", Literal["America/New_York"]),
        ("base_timeframe", Literal["15m"]),
    ),
    ScheduledBaseInterval: (
        ("session_date", date), ("interval_ordinal", int),
        ("open_ts", Timestamp), ("close_ts", Timestamp),
    ),
    ScheduledContextBar: (
        ("timeframe", Literal["1h", "4h", "1d"]), ("session_date", date),
        ("label_ts", Timestamp), ("period_start", Timestamp),
        ("period_end", Timestamp), ("available_at", Timestamp),
        ("fresh_context_at", Timestamp | None),
        ("source", Literal[
            "fetched_left_edge", "derived_1h_et_midnight", "session_aligned",
        ]),
    ),
    ReplaySchedule: (
        ("identity", ExchangeScheduleIdentity),
        ("base_intervals", tuple[ScheduledBaseInterval, ...]),
        ("context_bars", tuple[ScheduledContextBar, ...]),
    ),
    PlayRequest: (
        ("play_id", str), ("strategy", str), ("definition_digest", str),
        ("params", Mapping[str, JSONValue]), ("priority", int),
    ),
    AccountPolicy: (
        ("starting_equity", float), ("risk_pct", float),
        ("max_open_positions", int),
        ("max_positions_per_play_symbol", Literal[1]),
        ("settlement_model", Literal["cash_t1"]),
    ),
    SlippageSpec: (("bps", float), ("min_per_share", float)),
    FeeSpec: (("per_fill", float), ("per_share", float)),
    ExecutionPolicy: (
        ("arithmetic_version", str), ("fill_mode", Literal["pessimistic"]),
        ("slippage", SlippageSpec), ("fees", FeeSpec),
        ("funding_order", Literal["play_priority_symbol_signal"]),
        ("missing_bar_policy", Literal["strict"]),
    ),
    BenchmarkSpec: (
        ("kind", Literal["equal_weight_request_symbols", "single_symbol"]),
        ("symbol", str | None), ("weighting", Literal["equal"]),
        ("rebalance", Literal["never"]),
    ),
    PortfolioReplayRequest: (
        ("request_version", Literal[1]), ("replay_id", str),
        ("candidate_id", str), ("batch_id", str), ("registry_digest", str),
        ("plays", tuple[PlayRequest, ...]), ("symbols", tuple[str, ...]),
        ("window", ReplayWindow),
        ("schedule_identity", ExchangeScheduleIdentity),
        ("ic_horizons", tuple[Literal[1], Literal[5], Literal[20]]),
        ("ic_tail_end", Timestamp), ("account", AccountPolicy),
        ("execution", ExecutionPolicy), ("benchmark", BenchmarkSpec),
    ),
    EntryIntent: (
        ("replay_id", str), ("play_id", str), ("strategy", str),
        ("symbol", str), ("signal", Signal), ("signal_ordinal", int),
        ("signal_ts", Timestamp), ("order_type", Literal["market_next_open"]),
        ("eligible_after", Timestamp), ("expires_after_intervals", Literal[1]),
    ),
    PositionView: (
        ("direction", Literal["long", "short"]), ("qty", int),
        ("entry_ts", Timestamp), ("entry", float), ("initial_stop", float),
        ("initial_target", float), ("live_stop", float), ("live_target", float),
    ),
    ManagementDecision: (
        ("action", Literal["hold", "exit"]), ("stop", float | None),
        ("target", float | None),
    ),
    PortfolioTrade: (
        ("trade_id", str), ("replay_id", str), ("trade_ordinal", int),
        ("play_id", str), ("strategy", str), ("symbol", str),
        ("signal_ordinal", int), ("direction", Literal["long", "short"]),
        ("qty", int), ("signal_ts", Timestamp), ("entry_ts", Timestamp),
        ("entry", float), ("exit_ts", Timestamp), ("exit", float),
        ("initial_stop", float), ("final_stop", float),
        ("initial_target", float), ("final_target", float),
        ("gross_pnl", float), ("fees", float), ("net_pnl", float),
        ("r_multiple", float), ("mae", float), ("mfe", float),
        ("setup_tags", tuple[str, ...]), ("exit_reason", ExitReason),
    ),
    ReplayRejection: (
        ("rejection_id", str), ("replay_id", str),
        ("rejection_ordinal", int), ("play_id", str), ("strategy", str),
        ("symbol", str), ("signal_ordinal", int), ("signal_ts", Timestamp),
        ("event_ts", Timestamp), ("reason", RejectionReason),
        ("required_cash", float | None), ("available_cash", float | None),
        ("open_positions", int),
    ),
    EquityPoint: (
        ("replay_id", str), ("ts", Timestamp), ("point_ordinal", int),
        ("settled_cash", float), ("unsettled_cash", float),
        ("short_collateral", float), ("positions_liquidation_value", float),
        ("portfolio_equity", float), ("gross_exposure", float),
        ("open_positions", int), ("benchmark_equity", float),
    ),
    IcEstimate: (
        ("horizon_bars", Literal[1, 5, 20]), ("correlation", float | None),
        ("observations", int),
    ),
    PortfolioSlice: (
        ("replay_id", str), ("play_id", str), ("strategy", str),
        ("symbol", str), ("signals", int), ("trades", int),
        ("rejection_counts", Mapping[RejectionReason, int]),
        ("gross_profit", float), ("gross_loss", float),
        ("pre_cost_pnl", float), ("net_pnl", float), ("fees", float),
        ("win_rate", float | None), ("expectancy_r", float | None),
        ("ic", tuple[IcEstimate, IcEstimate, IcEstimate]),
    ),
    BenchmarkResult: (("spec", BenchmarkSpec), ("total_return", float)),
    TradeStats: (
        ("n_trades", int), ("n_wins", int), ("win_rate", float | None),
        ("gross_profit", float), ("gross_loss", float),
        ("profit_factor", float | None),
        ("profit_factor_state", Literal["finite", "infinite", "unavailable"]),
        ("expectancy_r", float | None),
    ),
    PortfolioMetrics: (
        ("all_trades", TradeStats), ("long_trades", TradeStats),
        ("short_trades", TradeStats), ("n_rejections", int),
        ("pre_cost_pnl", float), ("fees", float), ("net_pnl", float),
        ("starting_equity", float), ("ending_equity", float),
        ("total_return", float), ("benchmark_return", float),
        ("max_drawdown", float), ("ulcer_index", float), ("cagr", float),
        ("calmar", float | None), ("exposure_pct", float),
        ("avg_holding_hours", float), ("daily_n", int),
        ("daily_sum", float), ("daily_sum_sq", float),
        ("daily_sum_sq_down", float), ("daily_sum_cube", float),
        ("daily_sum_fourth", float), ("sharpe", float | None),
        ("sortino", float | None), ("psr", float | None),
        ("skew", float | None), ("kurtosis", float | None),
    ),
    PortfolioReplayResult: (
        ("request", PortfolioReplayRequest), ("arithmetic_version", str),
        ("fill_mode", str), ("schedule_identity", ExchangeScheduleIdentity),
        ("result_digest", str), ("trades", tuple[PortfolioTrade, ...]),
        ("rejections", tuple[ReplayRejection, ...]),
        ("equity", tuple[EquityPoint, ...]),
        ("slices", tuple[PortfolioSlice, ...]), ("benchmark", BenchmarkResult),
        ("metrics", PortfolioMetrics),
    ),
}

PUBLIC_SIGNATURES = {
    canonical_replay_bytes: ((("value", object, EMPTY),), bytes),
    schedule_digest: ((("schedule", ReplaySchedule, EMPTY),), str),
    expected_candidate_id: ((("request", PortfolioReplayRequest, EMPTY),), str),
    expected_replay_id: ((("request", PortfolioReplayRequest, EMPTY),), str),
    trade_id: ((
        ("replay_id", str, EMPTY), ("play_id", str, EMPTY),
        ("symbol", str, EMPTY), ("signal_ordinal", int, EMPTY),
    ), str),
    rejection_id: ((
        ("replay_id", str, EMPTY), ("play_id", str, EMPTY),
        ("symbol", str, EMPTY), ("signal_ordinal", int, EMPTY),
        ("reason", RejectionReason, EMPTY),
    ), str),
    definition_digest: ((
        ("base_digest", str, EMPTY), ("params", Mapping[str, JSONValue], EMPTY),
    ), str),
    result_digest: ((("result", PortfolioReplayResult, EMPTY),), str),
    decode_replay_request: (
        (("value", Mapping[str, object], EMPTY),), PortfolioReplayRequest,
    ),
    encode_replay_request: (
        (("request", PortfolioReplayRequest, EMPTY),), dict[str, JSONValue],
    ),
    decode_replay_schedule: (
        (("value", Mapping[str, object], EMPTY),), ReplaySchedule,
    ),
    encode_replay_schedule: (
        (("schedule", ReplaySchedule, EMPTY),), dict[str, JSONValue],
    ),
    decode_replay_result: (
        (("value", Mapping[str, object], EMPTY),), PortfolioReplayResult,
    ),
    encode_replay_result: (
        (("result", PortfolioReplayResult, EMPTY),), dict[str, JSONValue],
    ),
}


class _AliasKey(Enum):
    """A plain enum whose value equals a string key it must not collide with.

    A `StrEnum` member and its own value are one dict key already, so only a
    plain enum can prove the codec catches a collision that appears when keys
    normalize.
    """

    UNSETTLED_CASH = "unsettled_cash"


def _signature(target) -> tuple:
    signature = inspect.signature(target)
    return (
        tuple((p.name, p.annotation, p.default) for p in signature.parameters.values()),
        signature.return_annotation,
    )


def _round_trip(encoded: dict) -> dict:
    """Send an encoded value through real strict JSON, the way a worker does."""
    return json.loads(json.dumps(encoded, allow_nan=False))


# ---------------------------------------------------------------- codec bytes


def test_canonical_scalars_are_tagged_and_normalized():
    value = {
        "day": date(2026, 11, 27),
        "price": 100.25,
        "ts": pd.Timestamp("2026-08-10T09:30:00-04:00"),
    }
    assert canonical_replay_bytes(value) == (
        b'{"day":{"$date":"2026-11-27"},'
        b'"price":{"$float":"0x1.9100000000000p+6"},'
        b'"ts":"2026-08-10T13:30:00.000000Z"}'
    )


def test_canonical_bytes_sort_object_keys_and_keep_array_order():
    value = {"b": 1, "a": ("z", "a"), "A": [3, 2]}
    assert canonical_replay_bytes(value) == b'{"A":[3,2],"a":["z","a"],"b":1}'


def test_canonical_bytes_carry_enums_booleans_and_nulls_verbatim():
    value = {
        "reason": RejectionReason.UNSETTLED_CASH,
        "exit": ExitReason.STOP_GAP,
        "flag": True,
        "absent": None,
        "count": 3,
    }
    assert canonical_replay_bytes(value) == (
        b'{"absent":null,"count":3,"exit":"stop_gap",'
        b'"flag":true,"reason":"unsettled_cash"}'
    )


def test_canonical_bytes_are_stable_across_repeated_calls():
    request = base_request()
    first = canonical_replay_bytes({"symbols": request.symbols})
    assert first == canonical_replay_bytes({"symbols": request.symbols})


def test_canonical_floats_keep_every_bit_that_decimal_text_would_lose():
    near = 0.1 + 0.2
    assert canonical_replay_bytes(near) != canonical_replay_bytes(0.3)
    assert canonical_replay_bytes(-0.0) != canonical_replay_bytes(0.0)


def test_canonical_integers_and_floats_are_distinguishable():
    assert canonical_replay_bytes(1) == b"1"
    assert canonical_replay_bytes(1.0) == b'{"$float":"0x1.0000000000000p+0"}'


def test_canonical_timestamps_normalize_every_offset_to_one_spelling():
    same_instant = (
        pd.Timestamp("2026-08-10T13:30:00Z"),
        pd.Timestamp("2026-08-10T09:30:00-04:00"),
        datetime(2026, 8, 10, 15, 30, tzinfo=timezone(timedelta(hours=2))),
    )
    encoded = {canonical_replay_bytes(value) for value in same_instant}
    assert encoded == {b'"2026-08-10T13:30:00.000000Z"'}


def test_canonical_timestamps_keep_microseconds():
    stamp = pd.Timestamp("2026-08-10T13:30:00.123456Z")
    assert canonical_replay_bytes(stamp) == b'"2026-08-10T13:30:00.123456Z"'


CODEC_REFUSALS = {
    "nan": float("nan"),
    "positive_infinity": float("inf"),
    "negative_infinity": float("-inf"),
    "naive_timestamp": pd.Timestamp("2026-08-10T13:30:00"),
    "naive_datetime": datetime(2026, 8, 10, 13, 30),
    "sub_microsecond": pd.Timestamp("2026-08-10T13:30:00.123456789Z"),
    "not_a_time": pd.NaT,
    "set": {"a", "b"},
    "frozenset": frozenset({"a"}),
    "bytes": b"raw",
    "bytearray": bytearray(b"raw"),
    "unknown_object": object(),
    "complex": 1 + 2j,
    "reserved_date_key": {"$date": "2026-11-27"},
    "reserved_float_key": {"$float": "0x1.0p+0"},
    "nested_reserved_key": {"params": {"inner": {"$float": 1}}},
    "non_string_key": {1: "one"},
    "boolean_key": {True: "one"},
    "duplicate_normalized_key": {_AliasKey.UNSETTLED_CASH: 1, "unsettled_cash": 2},
    "generator": (item for item in (1, 2)),
    "dataclass_instance": base_window(),
    "lone_surrogate": "\ud800",
    "lone_surrogate_in_object": {"note": "bad \udfff text"},
    "lone_surrogate_key": {"\ud800": 1},
}


@pytest.mark.parametrize("value", CODEC_REFUSALS.values(), ids=CODEC_REFUSALS)
def test_canonical_codec_refuses_uncanonical_input(value):
    with pytest.raises(ReplayInputError):
        canonical_replay_bytes(value)


def test_a_lone_surrogate_stays_inside_the_closed_taxonomy():
    """`json.loads` produces one, and it has no UTF-8 encoding.

    Every door has to refuse it: an untyped `UnicodeEncodeError` out of the
    hasher would bypass the code platform branches on.
    """
    surrogate = json.loads('"\\ud800"')
    with pytest.raises(ReplayInputError) as raised:
        canonical_replay_bytes({"note": surrogate})
    assert raised.value.code == "invalid_value"
    with pytest.raises(ReplayInputError):
        dataclasses.replace(base_plays()[0], strategy=surrogate)
    with pytest.raises(ReplayInputError):
        dataclasses.replace(base_plays()[0], params={"note": surrogate})
    encoded = {**encode_replay_request(base_request()), "registry_digest": surrogate}
    with pytest.raises(ReplayInputError):
        decode_replay_request(encoded)


def test_canonical_codec_refuses_unbounded_nesting():
    deep = {"leaf": 1}
    for _ in range(200):
        deep = {"nested": deep}
    with pytest.raises(ReplayInputError) as raised:
        canonical_replay_bytes(deep)
    assert raised.value.code == "canonical_value_too_deep"


def test_canonical_refusal_names_the_path_that_failed():
    with pytest.raises(ReplayInputError) as raised:
        canonical_replay_bytes({"outer": {"inner": (1, float("nan"))}})
    assert raised.value.details["field"] == "$.outer.inner[1]"


# ------------------------------------------------------------ frozen contract


@pytest.mark.parametrize("cls", PUBLIC_FIELDS, ids=lambda cls: cls.__name__)
def test_public_dataclass_fields_match_the_architecture(cls):
    expected = PUBLIC_FIELDS[cls]
    hints = get_type_hints(cls)
    declared = tuple((f.name, hints[f.name]) for f in dataclasses.fields(cls))
    assert declared == expected


@pytest.mark.parametrize("cls", PUBLIC_FIELDS, ids=lambda cls: cls.__name__)
def test_public_values_are_frozen(cls):
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen


@pytest.mark.parametrize("target", PUBLIC_SIGNATURES, ids=lambda t: t.__name__)
def test_public_helper_signatures_match_the_contract(target):
    assert _signature(target) == PUBLIC_SIGNATURES[target]


REPLAY_ERRORS = {
    ReplayInputError: ValueError,
    StrategyOutputError: ValueError,
    StrategyRuntimeError: RuntimeError,
}


@pytest.mark.parametrize("error", REPLAY_ERRORS, ids=lambda error: error.__name__)
def test_replay_errors_take_a_required_code_message_and_details(error):
    parameters = tuple(inspect.signature(error).parameters.values())
    assert tuple((p.name, p.annotation, p.default) for p in parameters) == (
        ("code", str, EMPTY),
        ("message", str, EMPTY),
        ("details", Mapping[str, JSONValue], EMPTY),
    )
    assert issubclass(error, REPLAY_ERRORS[error])
    raised = error("some_code", "some message", {"field": "risk_pct"})
    assert (raised.code, raised.message) == ("some_code", "some message")
    assert raised.details == {"field": "risk_pct"}
    assert str(raised) == "some message"


def test_replay_input_error_copies_and_freezes_its_details():
    supplied = {"field": "risk_pct", "path": ["a", "b"], "seen": {"k": 1.5}}
    error = ReplayInputError("invalid_value", "risk_pct out of range", supplied)
    supplied["field"] = "mutated"
    assert error.code == "invalid_value"
    assert error.message == "risk_pct out of range"
    assert error.details["field"] == "risk_pct"
    assert error.details["path"] == ("a", "b")
    with pytest.raises(TypeError):
        error.details["field"] = "mutated"
    with pytest.raises(TypeError):
        error.details["seen"]["k"] = 2.0


def test_replay_input_error_details_are_canonical_json_values():
    with pytest.raises(ReplayInputError):
        ReplayInputError("bad", "details carry only JSON values", {"seen": object()})


def test_ordered_collections_are_tuples_and_mappings_are_read_only():
    request = base_request()
    assert isinstance(request.plays, tuple)
    assert isinstance(request.symbols, tuple)
    assert isinstance(request.plays[0].params, MappingProxyType)
    with pytest.raises(TypeError):
        request.plays[0].params["fast_n"] = 2


def test_request_normalizes_caller_order_and_symbol_case():
    request = base_request()
    assert request.symbols == ("QQQ", "SPY")
    assert tuple(play.play_id for play in request.plays) == ("play-a", "play-b")


def test_request_identity_ignores_caller_collection_order():
    request = base_request()
    shuffled = dataclasses.replace(
        request,
        plays=tuple(reversed(request.plays)),
        symbols=tuple(reversed(request.symbols)),
    )
    assert shuffled == request
    assert expected_candidate_id(shuffled) == expected_candidate_id(request)


def test_int_valued_binary64_fields_normalize_to_float():
    account = dataclasses.replace(base_account(), starting_equity=100_000)
    assert isinstance(account.starting_equity, float)
    assert account.starting_equity == 100_000.0


def test_offset_timestamps_normalize_to_utc_before_storage():
    window = dataclasses.replace(
        base_window(), train_start=pd.Timestamp("2026-08-10T09:30:00-04:00"),
    )
    assert str(window.train_start.tz) == "UTC"
    assert window.train_start == pd.Timestamp("2026-08-10T13:30:00Z")


REQUEST_REFUSALS = {
    "duplicate_symbols": {"symbols": ("SPY", "spy")},
    "empty_symbols": {"symbols": ()},
    "blank_symbol": {"symbols": ("SPY", "  ")},
    "padded_symbol": {"symbols": ("SPY", " QQQ")},
    "empty_plays": {"plays": ()},
    "unknown_request_version": {"request_version": 2},
    "boolean_request_version": {"request_version": True},
    "wrong_ic_horizons": {"ic_horizons": (1, 5, 10)},
    "short_ic_horizons": {"ic_horizons": (1, 5)},
    "tail_before_test_end": {"ic_tail_end": ts("2026-11-27T17:45:00Z")},
    "naive_tail": {"ic_tail_end": pd.Timestamp("2026-11-27T18:00:00")},
    "sub_microsecond_tail": {"ic_tail_end": pd.Timestamp("2026-11-27T18:00:00.000000001Z")},
    "date_for_timestamp": {"ic_tail_end": date(2026, 11, 27)},
    "short_registry_digest": {"registry_digest": "1f" * 31},
    "uppercase_registry_digest": {"registry_digest": "1F" * 32},
}


@pytest.mark.parametrize("override", REQUEST_REFUSALS.values(), ids=REQUEST_REFUSALS)
def test_request_refuses_uncanonical_values(override):
    with pytest.raises(ReplayInputError):
        base_request(**override)


def test_play_requests_refuse_blank_names_and_loose_digests():
    play = base_plays()[0]
    for override in (
        {"play_id": " "}, {"play_id": ""}, {"play_id": " play-a"},
        {"strategy": ""}, {"definition_digest": "not-a-digest"},
        {"priority": 1.0}, {"params": [("fast_n", 10)]},
    ):
        with pytest.raises(ReplayInputError):
            dataclasses.replace(play, **override)


def test_play_parameters_refuse_the_keys_the_codec_reserves():
    play = base_plays()[0]
    for reserved in ({"$float": 1.0}, {"nested": {"$date": "2026-08-10"}}):
        with pytest.raises(ReplayInputError) as raised:
            dataclasses.replace(play, params=reserved)
        assert raised.value.code == "reserved_canonical_key"


def test_request_refuses_duplicate_play_ids():
    play = base_plays()[0]
    with pytest.raises(ReplayInputError):
        base_request(plays=(play, dataclasses.replace(play, priority=300)))


IDENTIFIER_REFUSALS = {
    "unprefixed_replay_id": {"replay_id": PLACEHOLDER_DIGEST},
    "wrong_prefix": {"replay_id": f"candidate:{PLACEHOLDER_DIGEST}"},
    "short_digest": {"replay_id": "replay:" + "0" * 63},
    "uppercase_digest": {"replay_id": "replay:" + "A" * 64},
    "unprefixed_candidate_id": {"candidate_id": PLACEHOLDER_DIGEST},
    "uuid4_batch_id": {"batch_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479"},
    "uppercase_batch_id": {"batch_id": "0198B1C2-3D4E-7F80-8123-456789ABCDEF"},
    "braced_batch_id": {"batch_id": "{0198b1c2-3d4e-7f80-8123-456789abcdef}"},
    "urn_batch_id": {"batch_id": "urn:uuid:0198b1c2-3d4e-7f80-8123-456789abcdef"},
    "undashed_batch_id": {"batch_id": "0198b1c23d4e7f808123456789abcdef"},
    "reserved_variant_batch_id": {"batch_id": "0198b1c2-3d4e-7f80-c123-456789abcdef"},
    "not_a_uuid": {"batch_id": "batch-1"},
}


@pytest.mark.parametrize("override", IDENTIFIER_REFUSALS.values(), ids=IDENTIFIER_REFUSALS)
def test_request_refuses_malformed_identifiers(override):
    with pytest.raises(ReplayInputError):
        dataclasses.replace(base_request(), **override)


WINDOW_REFUSALS = {
    "train_after_test": {"train_start": ts("2026-11-27T14:45:00Z")},
    "empty_train": {"train_start": ts("2026-11-27T14:30:00Z")},
    "gap_between_train_and_test": {"train_end": ts("2026-11-27T14:15:00Z")},
    "empty_test": {"test_end": ts("2026-11-27T14:30:00Z")},
    "reversed_test": {"test_end": ts("2026-11-27T14:15:00Z")},
}


@pytest.mark.parametrize("override", WINDOW_REFUSALS.values(), ids=WINDOW_REFUSALS)
def test_window_refuses_invalid_boundaries(override):
    with pytest.raises(ReplayInputError):
        dataclasses.replace(base_window(), **override)


POLICY_REFUSALS = {
    "zero_equity": (base_account, {"starting_equity": 0.0}),
    "negative_equity": (base_account, {"starting_equity": -1.0}),
    "boolean_equity": (base_account, {"starting_equity": True}),
    "nan_equity": (base_account, {"starting_equity": float("nan")}),
    "infinite_equity": (base_account, {"starting_equity": float("inf")}),
    "zero_risk": (base_account, {"risk_pct": 0.0}),
    "risk_above_one": (base_account, {"risk_pct": 1.5}),
    "zero_capacity": (base_account, {"max_open_positions": 0}),
    "boolean_capacity": (base_account, {"max_open_positions": True}),
    "float_capacity": (base_account, {"max_open_positions": 5.0}),
    "two_per_play_symbol": (base_account, {"max_positions_per_play_symbol": 2}),
    "unsupported_settlement": (base_account, {"settlement_model": "margin_t0"}),
    "unsupported_fill_mode": (base_execution, {"fill_mode": "optimistic"}),
    "blank_arithmetic_version": (base_execution, {"arithmetic_version": " "}),
    "unsupported_funding_order": (base_execution, {"funding_order": "symbol_only"}),
    "unsupported_missing_bar_policy": (base_execution, {"missing_bar_policy": "skip"}),
    "unsupported_benchmark_kind": (base_benchmark, {"kind": "cap_weighted"}),
    "unsupported_weighting": (base_benchmark, {"weighting": "cap"}),
    "unsupported_rebalance": (base_benchmark, {"rebalance": "monthly"}),
    "equal_weight_with_symbol": (base_benchmark, {"symbol": "SPY"}),
    "single_symbol_without_symbol": (base_benchmark, {"kind": "single_symbol"}),
    "wrong_calendar": (base_identity, {"calendar_id": "XLON"}),
    "wrong_timezone": (base_identity, {"timezone": "UTC"}),
    "wrong_base_timeframe": (base_identity, {"base_timeframe": "5m"}),
    "blank_calendar_version": (base_identity, {"calendar_version": ""}),
}


@pytest.mark.parametrize(
    ("builder", "override"), POLICY_REFUSALS.values(), ids=POLICY_REFUSALS,
)
def test_policy_values_refuse_unsupported_settings(builder, override):
    with pytest.raises(ReplayInputError):
        dataclasses.replace(builder(), **override)


def test_negative_costs_are_refused_but_zero_costs_are_allowed():
    assert SlippageSpec(bps=0.0, min_per_share=0.0).bps == 0.0
    assert FeeSpec(per_fill=0.0, per_share=0.0).per_fill == 0.0
    for kwargs in ({"bps": -1.0, "min_per_share": 0.0}, {"bps": 1.0, "min_per_share": -0.01}):
        with pytest.raises(ReplayInputError):
            SlippageSpec(**kwargs)
    for kwargs in ({"per_fill": -1.0, "per_share": 0.0}, {"per_fill": 1.0, "per_share": -0.005}):
        with pytest.raises(ReplayInputError):
            FeeSpec(**kwargs)


def test_single_symbol_benchmark_normalizes_its_symbol():
    spec = BenchmarkSpec(
        kind="single_symbol", symbol="spy", weighting="equal", rebalance="never",
    )
    assert spec.symbol == "SPY"


def test_rejection_cash_is_populated_only_for_unsettled_cash():
    result = base_result()
    rejection = result.rejections[0]
    with pytest.raises(ReplayInputError):
        dataclasses.replace(rejection, reason=RejectionReason.WINDOW_ENDED)
    with pytest.raises(ReplayInputError):
        dataclasses.replace(rejection, required_cash=None)
    moved = dataclasses.replace(
        rejection, reason=RejectionReason.WINDOW_ENDED,
        required_cash=None, available_cash=None,
    )
    assert moved.reason is RejectionReason.WINDOW_ENDED


def test_trade_stats_state_follows_its_gross_sums():
    stats = base_metrics().all_trades
    with pytest.raises(ReplayInputError):
        dataclasses.replace(stats, profit_factor_state="finite")
    with pytest.raises(ReplayInputError):
        dataclasses.replace(stats, gross_loss=10.0, profit_factor=None)
    finite = dataclasses.replace(
        stats, gross_loss=14.0, profit_factor=2.0, profit_factor_state="finite",
    )
    assert finite.profit_factor == 2.0


def test_an_empty_cohort_reports_no_rate_no_expectancy_and_two_zero_sums():
    empty = base_metrics().short_trades
    assert empty.n_trades == 0
    for override in (
        {"gross_profit": 5.0}, {"gross_loss": 5.0},
        {"win_rate": 0.0}, {"expectancy_r": 0.0},
    ):
        with pytest.raises(ReplayInputError):
            dataclasses.replace(empty, **override)
    quiet_slice = next(row for row in base_result().slices if row.trades == 0)
    with pytest.raises(ReplayInputError):
        dataclasses.replace(quiet_slice, gross_profit=5.0)


def test_metrics_refuse_pooled_statistics_below_sixty_daily_returns():
    metrics = base_metrics()
    assert metrics.daily_n < 60
    with pytest.raises(ReplayInputError):
        dataclasses.replace(metrics, sharpe=1.4)
    with pytest.raises(ReplayInputError):
        dataclasses.replace(metrics, max_drawdown=0.0)


def test_direction_members_and_reason_strings_normalize_to_the_contract():
    """The engine passes enums, transport passes strings, storage sees one shape."""
    trade = base_result().trades[0]
    from_enum = dataclasses.replace(trade, direction=Direction.LONG)
    assert from_enum.direction == "long"
    assert type(from_enum.direction) is str
    assert from_enum == trade
    from_text = dataclasses.replace(trade, exit_reason="target")
    assert from_text.exit_reason is ExitReason.TARGET
    with pytest.raises(ReplayInputError):
        dataclasses.replace(trade, exit_reason="liquidated")
    with pytest.raises(ReplayInputError):
        dataclasses.replace(trade, direction="flat")


def test_exit_and_rejection_reasons_are_closed_lowercase_vocabularies():
    assert tuple(reason.value for reason in ExitReason) == (
        "stop_gap", "target_gap", "stop", "target", "manage", "end_of_window",
    )
    assert tuple(reason.value for reason in RejectionReason) == (
        "position_occupied", "pending_intent_occupied",
        "invalid_protective_geometry", "zero_quantity", "portfolio_capacity",
        "unsettled_cash", "window_ended",
    )


# ------------------------------------------------------------- identifiers


def test_parent_and_child_identifiers_are_pinned():
    request = base_request()
    assert expected_candidate_id(request) == CANDIDATE_GOLDEN
    assert expected_replay_id(request) == REPLAY_GOLDEN
    assert trade_id(request.replay_id, "play-a", "SPY", 0) == TRADE_GOLDEN
    assert rejection_id(
        request.replay_id, "play-a", "SPY", 1,
        RejectionReason.UNSETTLED_CASH,
    ) == REJECTION_GOLDEN


def test_identifiers_carry_their_prefix_and_a_lowercase_sha256():
    request = base_request()
    for identifier, prefix in (
        (expected_candidate_id(request), "candidate:"),
        (expected_replay_id(request), "replay:"),
        (trade_id(request.replay_id, "play-a", "SPY", 0), "trade:"),
        (rejection_id(
            request.replay_id, "play-a", "SPY", 0, RejectionReason.WINDOW_ENDED,
        ), "rejection:"),
    ):
        assert identifier.startswith(prefix)
        digest = identifier[len(prefix):]
        assert len(digest) == 64
        assert digest == digest.lower()
        int(digest, 16)


def test_the_request_carries_the_identities_its_own_formulas_produce():
    """Pinned constants, because the fixture derives both from these formulas.

    Comparing the request against the same functions that filled it would
    compare each function with itself and could never fail.
    """
    request = base_request()
    assert request.candidate_id == CANDIDATE_GOLDEN
    assert request.replay_id == REPLAY_GOLDEN


CANDIDATE_INVARIANT = {
    "batch_id": {"batch_id": "0198b1c2-3d4e-7f80-8123-000000000000"},
    "window": {"window": dataclasses.replace(
        base_window(), train_start=ts("2026-11-25T14:45:00Z"),
    )},
    "schedule_digest": {"schedule_identity": base_identity("9c" * 32)},
    "ic_tail_end": {"ic_tail_end": ts("2026-11-27T18:15:00Z")},
}


@pytest.mark.parametrize("override", CANDIDATE_INVARIANT.values(), ids=CANDIDATE_INVARIANT)
def test_candidate_identity_survives_its_excluded_fields(override):
    request = base_request()
    varied = base_request(**override)
    assert varied != request
    assert expected_candidate_id(varied) == expected_candidate_id(request)


CANDIDATE_SENSITIVE = {
    "registry_digest": {"registry_digest": "4c" * 32},
    "symbols": {"symbols": ("SPY", "QQQ", "IWM")},
    "account": {"account": dataclasses.replace(base_account(), risk_pct=0.02)},
    "execution": {"execution": dataclasses.replace(
        base_execution(), fees=FeeSpec(per_fill=2.0, per_share=0.005),
    )},
    "benchmark": {"benchmark": BenchmarkSpec(
        kind="single_symbol", symbol="SPY", weighting="equal", rebalance="never",
    )},
    "calendar_version": {"schedule_identity": dataclasses.replace(
        base_identity("9c" * 32), calendar_version="exchange_calendars:9.9.9:nakagai-rth-v1",
    )},
}


@pytest.mark.parametrize("override", CANDIDATE_SENSITIVE.values(), ids=CANDIDATE_SENSITIVE)
def test_candidate_identity_moves_with_every_projected_field(override):
    assert expected_candidate_id(base_request(**override)) != expected_candidate_id(base_request())


def test_candidate_identity_moves_with_play_parameters_and_priority():
    request = base_request()
    retuned = dataclasses.replace(
        request.plays[0], params={**dict(request.plays[0].params), "fast_n": 11},
    )
    varied = base_request(plays=(retuned, request.plays[1]))
    assert expected_candidate_id(varied) != expected_candidate_id(request)

    reprioritized = base_request(
        plays=(dataclasses.replace(request.plays[0], priority=900), request.plays[1]),
    )
    assert expected_candidate_id(reprioritized) != expected_candidate_id(request)


@pytest.mark.parametrize("override", CANDIDATE_INVARIANT.values(), ids=CANDIDATE_INVARIANT)
def test_replay_identity_moves_with_every_window_scoped_field(override):
    assert expected_replay_id(base_request(**override)) != expected_replay_id(base_request())


def test_child_identifiers_separate_every_component():
    replay = base_request().replay_id
    other = f"replay:{'ab' * 32}"
    baseline = trade_id(replay, "play-a", "SPY", 0)
    assert trade_id(other, "play-a", "SPY", 0) != baseline
    assert trade_id(replay, "play-b", "SPY", 0) != baseline
    assert trade_id(replay, "play-a", "QQQ", 0) != baseline
    assert trade_id(replay, "play-a", "SPY", 1) != baseline
    assert rejection_id(
        replay, "play-a", "SPY", 0, RejectionReason.WINDOW_ENDED,
    ) != rejection_id(replay, "play-a", "SPY", 0, RejectionReason.ZERO_QUANTITY)


def test_child_identifiers_refuse_uncanonical_arguments():
    replay = base_request().replay_id
    with pytest.raises(ReplayInputError):
        trade_id("not-a-replay-id", "play-a", "SPY", 0)
    with pytest.raises(ReplayInputError):
        trade_id(replay, "play-a", "spy", 0)
    with pytest.raises(ReplayInputError):
        trade_id(replay, "play-a", "SPY", -1)
    with pytest.raises(ReplayInputError):
        trade_id(replay, "play-a", "SPY", True)
    with pytest.raises(ReplayInputError):
        rejection_id(replay, "play-a", "SPY", 0, "unsettled_cash_typo")


def test_definition_digest_binds_the_base_digest_to_its_parameters():
    base = "2a" * 32
    assert definition_digest(base, {"fast_n": 10}) == definition_digest(base, {"fast_n": 10})
    assert definition_digest(base, {"fast_n": 10}) != definition_digest(base, {"fast_n": 11})
    assert definition_digest(base, {"fast_n": 10}) != definition_digest("3b" * 32, {"fast_n": 10})
    digest = definition_digest(base, {})
    assert len(digest) == 64 and digest == digest.lower()
    with pytest.raises(ReplayInputError):
        definition_digest("not-a-digest", {})


def test_definition_digest_ignores_parameter_key_order():
    ordered = definition_digest("2a" * 32, {"a": 1, "b": 2})
    assert ordered == definition_digest("2a" * 32, {"b": 2, "a": 1})


# ------------------------------------------------------------------ digests


def test_schedule_digest_is_pinned_and_excludes_its_own_identity():
    schedule = base_schedule()
    assert schedule_digest(schedule) == SCHEDULE_DIGEST_GOLDEN
    assert schedule.identity.schedule_digest == SCHEDULE_DIGEST_GOLDEN
    relabelled = dataclasses.replace(schedule, identity=base_identity("7d" * 32))
    assert schedule_digest(relabelled) == SCHEDULE_DIGEST_GOLDEN


def test_schedule_digest_moves_with_every_scheduled_boundary():
    schedule = base_schedule()
    trimmed = dataclasses.replace(schedule, base_intervals=base_intervals()[:-1])
    assert schedule_digest(trimmed) != SCHEDULE_DIGEST_GOLDEN
    without_context = dataclasses.replace(schedule, context_bars=())
    assert schedule_digest(without_context) != SCHEDULE_DIGEST_GOLDEN


def test_result_digest_is_pinned_and_omits_only_itself():
    result = base_result()
    assert result_digest(result) == RESULT_DIGEST_GOLDEN
    assert result.result_digest == RESULT_DIGEST_GOLDEN
    restamped = dataclasses.replace(result, result_digest="0" * 64)
    assert result_digest(restamped) == RESULT_DIGEST_GOLDEN


def test_result_refuses_child_rows_it_cannot_attribute_or_order():
    result = base_result()
    with pytest.raises(ReplayInputError):
        dataclasses.replace(result, trades=(dataclasses.replace(
            result.trades[0], symbol="IWM",
        ),))
    with pytest.raises(ReplayInputError):
        dataclasses.replace(result, trades=(dataclasses.replace(
            result.trades[0], trade_ordinal=1,
        ),))
    with pytest.raises(ReplayInputError):
        dataclasses.replace(result, slices=result.slices + result.slices[:1])
    with pytest.raises(ReplayInputError):
        dataclasses.replace(result, equity=tuple(
            dataclasses.replace(point, point_ordinal=index)
            for index, point in enumerate(reversed(result.equity))
        ))
    with pytest.raises(ReplayInputError):
        dataclasses.replace(result, schedule_identity=base_identity("5e" * 32))


def test_result_digest_ignores_the_order_slices_were_collected_in():
    result = base_result()
    reordered = dataclasses.replace(result, slices=tuple(reversed(result.slices)))
    assert reordered.slices != result.slices
    assert result_digest(reordered) == RESULT_DIGEST_GOLDEN


def test_result_digest_moves_with_every_semantic_part():
    result = base_result()
    for override in (
        {"trades": ()},
        {"rejections": ()},
        {"equity": result.equity[:1]},
        {"benchmark": dataclasses.replace(result.benchmark, total_return=0.5)},
        {"metrics": dataclasses.replace(result.metrics, net_pnl=27.0)},
        {"arithmetic_version": "3"},
    ):
        assert result_digest(dataclasses.replace(result, **override)) != RESULT_DIGEST_GOLDEN


# ---------------------------------------------------------------- transport


def test_request_round_trips_through_strict_json():
    request = base_request()
    encoded = _round_trip(encode_replay_request(request))
    assert decode_replay_request(encoded) == request
    assert expected_replay_id(decode_replay_request(encoded)) == request.replay_id


def test_schedule_round_trips_through_strict_json():
    schedule = base_schedule()
    encoded = _round_trip(encode_replay_schedule(schedule))
    decoded = decode_replay_schedule(encoded)
    assert decoded == schedule
    assert schedule_digest(decoded) == SCHEDULE_DIGEST_GOLDEN


def test_result_round_trips_through_strict_json():
    result = base_result()
    encoded = _round_trip(encode_replay_result(result))
    decoded = decode_replay_result(encoded)
    assert decoded == result
    assert result_digest(decoded) == RESULT_DIGEST_GOLDEN
    assert canonical_replay_bytes(encoded["metrics"]["total_return"]) == (
        canonical_replay_bytes(result.metrics.total_return)
    )


def test_transport_encodes_dates_timestamps_enums_and_nulls_as_strict_json():
    encoded = encode_replay_schedule(base_schedule())
    assert encoded["base_intervals"][0]["session_date"] == "2026-11-25"
    assert encoded["base_intervals"][0]["open_ts"] == "2026-11-25T14:30:00.000000Z"
    half_day_bucket = encoded["context_bars"][2]
    assert half_day_bucket["timeframe"] == "4h"
    assert half_day_bucket["fresh_context_at"] is None
    assert encoded["context_bars"][-1]["source"] == "session_aligned"
    assert encoded["context_bars"][-1]["fresh_context_at"] == "2026-11-27T14:45:00.000000Z"
    trade = encode_replay_result(base_result())["trades"][0]
    assert trade["exit_reason"] == "target"
    assert trade["setup_tags"] == ["trend", "pullback"]
    assert isinstance(trade["entry"], float)


def test_transport_keeps_every_bit_of_a_binary64_field():
    request = base_request(account=dataclasses.replace(
        base_account(), risk_pct=0.1 + 0.2,
    ))
    decoded = decode_replay_request(_round_trip(encode_replay_request(request)))
    assert decoded.account.risk_pct == 0.1 + 0.2
    assert canonical_replay_bytes(decoded.account.risk_pct) == (
        canonical_replay_bytes(0.1 + 0.2)
    )


def test_transport_decoding_refuses_unknown_and_missing_keys():
    encoded = encode_replay_schedule(base_schedule())
    with pytest.raises(ReplayInputError):
        decode_replay_schedule({**encoded, "surprise": 1})
    without = {k: v for k, v in encoded.items() if k != "context_bars"}
    with pytest.raises(ReplayInputError):
        decode_replay_schedule(without)


TRANSPORT_REFUSALS = {
    "offset_timestamp_text": {"ic_tail_end": "2026-11-27T13:00:00.000000-05:00"},
    "second_precision_text": {"ic_tail_end": "2026-11-27T18:00:00Z"},
    "nanosecond_text": {"ic_tail_end": "2026-11-27T18:00:00.000000001Z"},
    "epoch_number": {"ic_tail_end": 1786809600},
    "float_for_int": {"request_version": 1.0},
    "string_for_int": {"request_version": "1"},
    "null_for_required": {"registry_digest": None},
    "list_for_mapping": {"account": []},
}


@pytest.mark.parametrize("override", TRANSPORT_REFUSALS.values(), ids=TRANSPORT_REFUSALS)
def test_transport_decoding_refuses_loose_json(override):
    encoded = {**encode_replay_request(base_request()), **override}
    with pytest.raises(ReplayInputError):
        decode_replay_request(encoded)


SESSION_DATE_REFUSALS = (
    "2026-8-10", " 2026-08-10", "2026-08-10 ", "2026-08-10T00:00:00Z",
    "2026-08-10Z", "2026-222", "2026-W33-1", "2026-02-30", "20260810",
    "+2026-08-10", "2026-08-1O",
)


@pytest.mark.parametrize("text", SESSION_DATE_REFUSALS)
def test_transport_decoding_refuses_loose_session_dates(text):
    encoded = encode_replay_schedule(base_schedule())
    intervals = [dict(interval) for interval in encoded["base_intervals"]]
    intervals[0]["session_date"] = text
    with pytest.raises(ReplayInputError):
        decode_replay_schedule({**encoded, "base_intervals": intervals})


def test_transport_decoding_refuses_nonfinite_json_numbers():
    encoded = encode_replay_request(base_request())
    account = {**encoded["account"], "starting_equity": math.nan}
    with pytest.raises(ReplayInputError):
        decode_replay_request({**encoded, "account": account})


def test_encoded_request_is_serializable_by_a_strict_json_writer():
    encoded = encode_replay_request(base_request())
    text = json.dumps(encoded, allow_nan=False)
    assert json.loads(text) == encoded
