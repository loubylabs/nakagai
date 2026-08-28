"""Hermetic current-core replay of the transferred Node 06 platform corpus.

The compressed corpus was captured on 2026-08-28 from platform commit
9a2a8459defdbb892a01fe54a545aea42dd09237. It carries the 57 normalized
RuleSpecs, the one curated composite needed by the registry closure, the exact
house grammar declarations, and seven deduplicated encoded schedules. Core CI
therefore needs no platform checkout or import to replay the same inputs.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import fields, replace
from datetime import time
from enum import Enum
from functools import cache
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pandas as pd

from nakagai.data.schema import EXCHANGE_TZ
from nakagai.engine import (
    FrozenStrategyRegistry,
    PlayRequest,
    PortfolioBars,
    PortfolioTrade,
    composite_definition,
    decode_replay_request,
    decode_replay_schedule,
    definition_digest,
    encode_replay_request,
    expected_candidate_id,
    expected_replay_id,
    rules_definition,
    run_portfolio,
    spec_base_digest,
)
from nakagai.engine.registry import dependencies_for
from nakagai.strategies.rules import Term, WindowSpec, core_vocabulary
from nakagai.strategies.rules.canon import spec_hash


FIXTURE = Path(__file__).parent / "fixtures" / "node06-current-corpus.json.gz"
IDENTITY_FIELDS = frozenset({"trade_id", "replay_id", "play_id"})
FLOAT_FIELDS = frozenset(
    name for name, annotation in get_type_hints(PortfolioTrade).items()
    if annotation is float
)


@cache
def _corpus() -> dict:
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _unused_house_term(*_args, **_kwargs):
    raise AssertionError(
        "the Node 06 corpus unexpectedly executed a platform-only term")


@cache
def _house_vocabulary():
    corpus = _corpus()
    terms = []
    for row in corpus["extra_terms"]:
        args = {
            name: tuple(value) if isinstance(value, list) else value
            for name, value in row["args"].items()
        }
        terms.append(Term(
            name=row["name"], kind=row["kind"], args=args,
            defaults=row["defaults"], fn=_unused_house_term,
            end_anchored=row["end_anchored"],
            session_scoped=row["session_scoped"],
            driving_frame_intraday=row["driving_frame_intraday"],
            window_reduce=row["window_reduce"],
            window_required=row["window_required"],
        ))
    windows = tuple(WindowSpec(
        name=row["name"], tz=row["tz"],
        start=time.fromisoformat(row["start"]),
        end=time.fromisoformat(row["end"]),
        recurrence=row["recurrence"], confidence=row["confidence"],
    ) for row in corpus["windows"])
    return core_vocabulary().with_terms(*terms).with_windows(*windows)


def _adapter_base_digest(adapter: str) -> str:
    return spec_base_digest({"$unbound_adapter": adapter}, _house_vocabulary)


def _registry(corpus: dict) -> FrozenStrategyRegistry:
    catalog = {
        name: rules_definition(
            name, spec_base_digest(spec, _house_vocabulary),
            spec=spec, vocabulary_factory=_house_vocabulary,
        )
        for name, spec in corpus["specs"].items()
    }
    rules = rules_definition(
        "rules", _adapter_base_digest("rules"),
        spec=None, vocabulary_factory=_house_vocabulary,
    )
    members = {**catalog, "rules": rules}
    composite = composite_definition(
        "composite", _adapter_base_digest("composite"),
        members=members, vocabulary_factory=_house_vocabulary,
    )
    curated = {
        name: composite_definition(
            name,
            definition_digest(
                _adapter_base_digest("composite"), {"spec": spec}),
            members=members, vocabulary_factory=_house_vocabulary,
        )
        for name, spec in corpus["composites"].items()
    }
    return FrozenStrategyRegistry.from_definitions(
        (*catalog.values(), rules, composite, *curated.values()))


def _request(before: dict, definition, registry, schedule):
    prior = decode_replay_request(before["request"])
    play = PlayRequest(
        play_id=prior.plays[0].play_id,
        strategy=prior.plays[0].strategy,
        params=prior.plays[0].params,
        priority=prior.plays[0].priority,
        definition_digest=definition_digest(
            definition.definition_digest, prior.plays[0].params),
    )
    draft = replace(
        prior,
        replay_id="replay:" + "0" * 64,
        candidate_id="candidate:" + "0" * 64,
        registry_digest=registry.registry_digest,
        plays=(play,),
        schedule_identity=schedule.identity,
    )
    identified = replace(draft, candidate_id=expected_candidate_id(draft))
    return replace(identified, replay_id=expected_replay_id(identified))


def _labels(schedule, timeframe: str, boundary: pd.Timestamp):
    if timeframe == "15m":
        return tuple(row.open_ts for row in schedule.base_intervals
                     if row.open_ts < boundary)
    return tuple(row.label_ts for row in schedule.context_bars
                 if row.timeframe == timeframe and row.label_ts < boundary)


def _tape(index: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
    """The exact deterministic tape used by the platform baseline capture."""
    local = index.tz_convert(EXCHANGE_TZ)
    day = (local.normalize()
           - pd.Timestamp("2025-01-01", tz=EXCHANGE_TZ)).days
    phase = np.asarray(day % 6, dtype=int)
    minute = np.asarray(
        local.hour * 60 + local.minute - (9 * 60 + 30), dtype=int)
    close = 100.0 + 0.03 * np.asarray(day, dtype=float)

    if timeframe == "15m":
        opening = minute < 30
        morning = (minute >= 30) & (minute < 240)
        afternoon = (minute >= 240) & (minute < 375)
        close = np.where(opening, 100.0, close)
        close = np.where((phase == 1) & opening, 105.0, close)
        close = np.where((phase == 0) & (minute == 30), 107.0, close)
        close = np.where((phase == 0) & (minute == 45), 100.0, close)
        close = np.where((phase == 0) & (minute >= 60) & morning, 107.0, close)
        close = np.where((phase == 1) & (minute == 30), 105.0, close)
        close = np.where(
            (phase == 1) & (minute >= 45) & (minute < 75), 94.0, close)
        close = np.where((phase == 1) & (minute >= 75) & morning, 108.0, close)
        close = np.where((phase == 2) & afternoon, 119.0, close)
        close = np.where((phase == 3) & opening, 120.0, close)
        close = np.where((phase == 3) & (minute == 15), 121.0, close)
        close = np.where((phase == 3) & (minute >= 30) & morning, 119.0, close)
        close = np.where((phase == 4) & afternoon, 108.0, close)
        close = np.where(phase == 5, 100.0, close)

    spread = np.where(
        (timeframe == "15m") & (phase == 5), 0.1,
        np.where(timeframe == "15m", 0.6, 1.5),
    )
    high = close + spread
    low = close - spread
    volume = np.where((phase == 0) & (minute == 60), 100_000.0, 100.0)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": volume},
        index=index,
    )


def _frames(request, schedule, registry) -> PortfolioBars:
    dependencies = dependencies_for(request, registry)
    return PortfolioBars({
        (symbol, timeframe): _tape(
            pd.DatetimeIndex(
                _labels(schedule, timeframe, request.ic_tail_end),
                tz="UTC", name="ts"),
            timeframe,
        )
        for symbol in request.symbols
        for timeframe in dependencies.timeframes
    })


def _json_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in sorted(value.items())}
    return value


def _trade_row(trade: PortfolioTrade) -> dict:
    values = {field.name: getattr(trade, field.name) for field in fields(trade)}
    return {
        "behavior": _json_value({
            name: value for name, value in values.items()
            if name not in IDENTITY_FIELDS | FLOAT_FIELDS
        }),
        "identity": _json_value({
            name: values[name] for name in IDENTITY_FIELDS}),
        "floats": _json_value({name: values[name] for name in FLOAT_FIELDS}),
    }


def capture_current(baseline: dict) -> dict[str, dict]:
    """Replay every transferred play through current production core."""
    corpus = _corpus()
    registry = _registry(corpus)
    out = {}
    for name, before in baseline["plays"].items():
        spec = corpus["specs"][name]
        definition = registry.resolve(name)
        schedule_digest = before["request"]["schedule_identity"][
            "schedule_digest"]
        schedule = decode_replay_schedule(corpus["schedules"][schedule_digest])
        request = _request(before, definition, registry, schedule)
        result = run_portfolio(
            request, _frames(request, schedule, registry), registry, schedule)
        out[name] = {
            "canonical_spec_hash": spec_hash(spec, _house_vocabulary()),
            "definition_digest": definition.definition_digest,
            "request": _json_value(encode_replay_request(request)),
            "trades": [_trade_row(trade) for trade in result.trades],
        }
    return out
