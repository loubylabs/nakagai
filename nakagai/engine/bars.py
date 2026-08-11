"""The bar boundary: one read-only mapping in, engine-owned frames out.

Two copies stand between a caller's DataFrames and the replay, and both are
deliberate. `PortfolioBars` deep copies every frame it is handed, so a caller
mutating its own dictionary afterwards changes nothing here, and it returns a
deep copy from `__getitem__`, so a caller reading a frame cannot write through
into the mapping. `prepare_portfolio_bars` then takes a second, engine-owned
copy that no caller has ever held a reference to.

Preparation is the strict-refusal preflight. Every declared frame must exist,
carry a UTC index that is exactly the labels the schedule declares up to that
symbol's boundary, hold the five binary64 OHLCV columns, and satisfy bar
geometry. Any failure raises `ReplayInputError(code="missing_required_bar")`
and the whole replay is refused, before a strategy could have been built. The
engine never forward-fills a price, drops a symbol, or settles a partial
portfolio.

Two boundaries exist because two kinds of symbol have different jobs. A traded
symbol is read by the IC lens after the trading replay ends, so it carries
every scheduled label through `ic_tail_end`. A benchmark or external
dependency is only ever read inside the test range, so it stops at `test_end`.
A label at or after a symbol's boundary could never have become available to
it, so requiring one would demand data nothing can read.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from nakagai.data.schema import BAR_COLUMNS
from nakagai.engine.portfolio_types import (
    CONTEXT_TIMEFRAMES,
    PortfolioReplayRequest,
    ReplayInputError,
    _fail,
    _require_instance,
    _require_str,
    _require_symbol,
)
from nakagai.engine.schedule import ValidatedSchedule

BASE_TIMEFRAME = "15m"
BAR_TIMEFRAMES = (BASE_TIMEFRAME, *CONTEXT_TIMEFRAMES)

_MISSING_BAR = "missing_required_bar"


def _refuse(message: str, field: str, **details) -> ReplayInputError:
    return ReplayInputError(_MISSING_BAR, message, {"field": field, **details})


def _require_timeframe(value: object, field: str) -> str:
    text = _require_str(value, field).lower()
    if text not in BAR_TIMEFRAMES:
        raise _fail(
            "invalid_value", "value is not a supported timeframe",
            field=field, allowed=BAR_TIMEFRAMES,
        )
    return text


def _bar_key(key: object) -> tuple[str, str]:
    if not isinstance(key, tuple):
        raise _fail(
            "invalid_type", "a bar key is a (symbol, timeframe) pair", field="bars",
            seen=type(key).__name__,
        )
    if len(key) != 2:
        raise _fail(
            "invalid_value", "a bar key is a (symbol, timeframe) pair", field="bars",
            seen=len(key),
        )
    return (_require_symbol(key[0], "symbol"), _require_timeframe(key[1], "timeframe"))


def _key_order(key: tuple[str, str]) -> tuple[str, int]:
    return (key[0], BAR_TIMEFRAMES.index(key[1]))


class PortfolioBars(Mapping[tuple[str, str], pd.DataFrame]):
    """The public, read-only bar input: `(symbol, timeframe)` to one frame.

    There is no mutation method and no way to reach the stored frames. A key
    normalizes on the way in and on every lookup, so `("spy", "15M")` and
    `("SPY", "15m")` are one entry rather than two, and supplying both is a
    refusal rather than a silent overwrite.
    """

    def __init__(self, frames: Mapping[tuple[str, str], pd.DataFrame]) -> None:
        if not isinstance(frames, Mapping):
            raise _fail("invalid_type", "bars must be a mapping", field="bars")
        owned: dict[tuple[str, str], pd.DataFrame] = {}
        for key, frame in frames.items():
            normalized = _bar_key(key)
            if normalized in owned:
                raise _fail(
                    "duplicate_value", "two keys normalize to one pair", field="bars",
                    symbol=normalized[0], timeframe=normalized[1],
                )
            if not isinstance(frame, pd.DataFrame):
                raise _fail(
                    "invalid_type", "a bar value must be a DataFrame", field="bars",
                    symbol=normalized[0], timeframe=normalized[1],
                )
            owned[normalized] = frame.copy(deep=True)
        self._frames = MappingProxyType(
            {key: owned[key] for key in sorted(owned, key=_key_order)},
        )

    def __getitem__(self, key: tuple[str, str]) -> pd.DataFrame:
        return self._frames[_bar_key(key)].copy(deep=True)

    def __contains__(self, key: object) -> bool:
        return _bar_key(key) in self._frames

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def _engine_copy(self) -> dict[tuple[str, str], pd.DataFrame]:
        """The engine's own frames, which no caller holds a reference to."""
        return {key: frame.copy(deep=True) for key, frame in self._frames.items()}


@dataclass(frozen=True)
class ReplayDependencies:
    """The resolved data closure one replay needs, beyond its traded symbols.

    `timeframes` is the union every play declares, and it always contains the
    base timeframe: the account fills, marks, and settles on the base clock
    whatever a strategy chooses to read. `external_symbols` are the symbols a
    play reads without trading them.
    """

    timeframes: tuple[str, ...]
    external_symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timeframes, tuple):
            raise _fail("invalid_type", "value must be a tuple", field="timeframes")
        if not isinstance(self.external_symbols, tuple):
            raise _fail("invalid_type", "value must be a tuple", field="external_symbols")
        declared = {_require_timeframe(value, "timeframes") for value in self.timeframes}
        if BASE_TIMEFRAME not in declared:
            raise _fail(
                "invalid_value", "every replay depends on the base timeframe",
                field="timeframes", required=BASE_TIMEFRAME,
            )
        symbols = {_require_symbol(value, "external_symbols")
                   for value in self.external_symbols}
        object.__setattr__(self, "timeframes", tuple(
            value for value in BAR_TIMEFRAMES if value in declared))
        object.__setattr__(self, "external_symbols", tuple(sorted(symbols)))


class _ValidatedPortfolioBars:
    """Engine-owned frames, validated against the schedule. Never escapes core.

    Nothing here copies again. Every frame was copied out of `PortfolioBars`
    and validated before this object existed, and pandas copy-on-write means a
    slice handed to a strategy cannot write back through into one of them.
    """

    def __init__(self, frames: dict[tuple[str, str], pd.DataFrame]) -> None:
        self._frames = MappingProxyType(frames)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._frames)

    def frame(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """One prepared frame, or `KeyError` for a pair nobody declared."""
        return self._frames[(_require_symbol(symbol, "symbol"),
                             _require_timeframe(timeframe, "timeframe"))]


def prepare_portfolio_bars(
    request: PortfolioReplayRequest,
    bars: PortfolioBars,
    schedule: ValidatedSchedule,
    dependencies: ReplayDependencies,
) -> _ValidatedPortfolioBars:
    """Take the engine's copy and refuse anything the schedule disagrees with."""
    _require_instance(request, "request", PortfolioReplayRequest)
    _require_instance(bars, "bars", PortfolioBars)
    _require_instance(schedule, "schedule", ValidatedSchedule)
    _require_instance(dependencies, "dependencies", ReplayDependencies)
    if schedule.request != request:
        raise _fail(
            "invalid_value", "the schedule was validated for another request",
            field="schedule",
        )
    required = _required_labels(request, schedule, dependencies)
    owned = bars._engine_copy()
    prepared: dict[tuple[str, str], pd.DataFrame] = {}
    for key in sorted(required, key=_key_order):
        frame = owned.get(key)
        if frame is None:
            raise _refuse(
                "a declared frame is absent", "frame",
                symbol=key[0], timeframe=key[1],
            )
        _validate_frame(frame, key)
        _require_scheduled_labels(frame, key, required[key])
        prepared[key] = frame
    surplus = sorted(key for key in owned if key not in required)
    if surplus:
        raise _refuse(
            "a frame was supplied that this replay never declared",
            "unexpected_frame", symbol=surplus[0][0], timeframe=surplus[0][1],
        )
    return _ValidatedPortfolioBars(prepared)


def _required_labels(
    request: PortfolioReplayRequest, schedule: ValidatedSchedule,
    dependencies: ReplayDependencies,
) -> dict[tuple[str, str], tuple[pd.Timestamp, ...]]:
    """Every `(symbol, timeframe)` this replay reads, and its exact labels."""
    required: dict[tuple[str, str], tuple[pd.Timestamp, ...]] = {}
    for symbol in request.symbols:
        for timeframe in dependencies.timeframes:
            required[(symbol, timeframe)] = _scheduled_labels(
                schedule, timeframe, request.ic_tail_end)
    context_boundary = request.window.test_end
    for symbol in dependencies.external_symbols:
        for timeframe in dependencies.timeframes:
            required.setdefault((symbol, timeframe), _scheduled_labels(
                schedule, timeframe, context_boundary))
    benchmark = request.benchmark.symbol
    if benchmark is not None:
        required.setdefault((benchmark, BASE_TIMEFRAME), _scheduled_labels(
            schedule, BASE_TIMEFRAME, context_boundary))
    return required


def _scheduled_labels(
    schedule: ValidatedSchedule, timeframe: str, boundary: pd.Timestamp,
) -> tuple[pd.Timestamp, ...]:
    if timeframe == BASE_TIMEFRAME:
        return tuple(row.open_ts for row in schedule.base_intervals
                     if row.open_ts < boundary)
    return tuple(row.label_ts for row in schedule.context_bars(timeframe)
                 if row.label_ts < boundary)


def _validate_frame(frame: pd.DataFrame, key: tuple[str, str]) -> None:
    """A UTC monotonic unique index and five finite binary64 OHLCV columns."""
    symbol, timeframe = key
    index = frame.index
    # A missing label needs no clause of its own: an index carrying NaT is
    # never monotonic, because every comparison against NaT is false.
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None
            or str(index.tz) != "UTC"
            or not index.is_monotonic_increasing or not index.is_unique):
        raise _refuse(
            "a frame is indexed by unique, increasing, UTC timestamps", "index",
            symbol=symbol, timeframe=timeframe,
        )
    if list(frame.columns) != BAR_COLUMNS:
        raise _refuse(
            f"a frame carries exactly {', '.join(BAR_COLUMNS)}", "columns",
            symbol=symbol, timeframe=timeframe,
            columns=tuple(str(column) for column in frame.columns),
        )
    for column in BAR_COLUMNS:
        if frame[column].dtype != np.float64:
            raise _refuse(
                "every bar column is binary64", "dtype",
                symbol=symbol, timeframe=timeframe, column=column,
            )
    values = {column: frame[column].to_numpy() for column in BAR_COLUMNS}
    for column, series in values.items():
        if not np.isfinite(series).all():
            raise _refuse(
                "every bar value is finite", column,
                symbol=symbol, timeframe=timeframe,
            )
    high, low = values["high"], values["low"]
    if not (high >= np.maximum.reduce([values["open"], values["close"], low])).all():
        raise _refuse(
            "a bar high is its own highest price", "high",
            symbol=symbol, timeframe=timeframe,
        )
    if not (low <= np.minimum.reduce([values["open"], values["close"], high])).all():
        raise _refuse(
            "a bar low is its own lowest price", "low",
            symbol=symbol, timeframe=timeframe,
        )
    if not (values["volume"] >= 0.0).all():
        raise _refuse(
            "bar volume is never negative", "volume",
            symbol=symbol, timeframe=timeframe,
        )


def _require_scheduled_labels(
    frame: pd.DataFrame, key: tuple[str, str], labels: tuple[pd.Timestamp, ...],
) -> None:
    """The index is exactly the schedule's labels, in order, and nothing else."""
    expected = pd.DatetimeIndex(list(labels), tz="UTC")
    if frame.index.equals(expected):
        return
    absent = expected.difference(frame.index)
    surplus = frame.index.difference(expected)
    raise _refuse(
        "a frame's labels are exactly the scheduled labels it must cover", "labels",
        symbol=key[0], timeframe=key[1], expected=len(expected), actual=len(frame.index),
        first_absent=absent[0].isoformat() if len(absent) else None,
        first_surplus=surplus[0].isoformat() if len(surplus) else None,
    )
