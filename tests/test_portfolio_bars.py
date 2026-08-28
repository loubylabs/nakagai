"""The bar boundary: a read-only mapping, engine-owned copies, strict preflight.

`PortfolioBars` is the only door bars come through, and everything past it is
defensive. The caller keeps its frames, core keeps its own deep copies, and
neither can reach the other's. Preparation then refuses the whole replay on
any missing, malformed, mislabeled, or surplus frame, before a strategy could
have been constructed.
"""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nakagai.engine.bars import (
    PortfolioBars,
    ReplayDependencies,
    prepare_portfolio_bars,
)
from nakagai.engine.context import build_scheduled_context
from nakagai.engine.execution import _PortfolioRuntime
from nakagai.engine.portfolio import _Ledger
from nakagai.engine.portfolio_types import (
    BenchmarkSpec,
    ReplayInputError,
    ReplayWindow,
)
from nakagai.engine.schedule import validate_schedule
from nakagai.strategies.rules.vocabulary import core_vocabulary
from tests.portfolio_fixtures import (
    bar_frame,
    base_context_bars,
    base_dependencies,
    base_frames,
    base_request,
    base_schedule,
    base_validated_schedule,
    frames_for,
    schedule_with,
    scheduled_labels,
    strategy_registry,
    ts,
    without_pair,
)

SPY_15M = ("SPY", "15m")
QQQ_15M = ("QQQ", "15m")
SCHEDULED_INTERVALS = 40


def prepared_base():
    return prepare_portfolio_bars(
        base_request(), PortfolioBars(base_frames()), base_validated_schedule(),
        base_dependencies(),
    )


def refuse(frames, request=None, dependencies=None, schedule=None):
    """Prepare a mutated frame set and return the refusal it raised."""
    request = base_request() if request is None else request
    schedule = validate_schedule(request, base_schedule()) if schedule is None else schedule
    with pytest.raises(ReplayInputError) as raised:
        prepare_portfolio_bars(
            request, PortfolioBars(frames), schedule,
            base_dependencies() if dependencies is None else dependencies,
        )
    return raised.value


def tail_request(**overrides):
    """A request whose IC tail runs one hour past the test range."""
    return base_request(
        window=ReplayWindow(
            train_start=ts("2026-11-25T14:30:00Z"),
            train_end=ts("2026-11-27T14:30:00Z"),
            test_start=ts("2026-11-27T14:30:00Z"),
            test_end=ts("2026-11-27T17:00:00Z"),
        ),
        ic_tail_end=ts("2026-11-27T18:00:00Z"),
        **overrides,
    )


def replaced_column(frames, key, column, values):
    frame = frames[key].copy(deep=True)
    frame[column] = values
    return {**frames, key: frame}


# --------------------------------------------------------- the read-only map


def test_portfolio_bars_owns_frames_and_never_returns_its_copy():
    source = base_frames()
    bars = PortfolioBars(source)
    source[SPY_15M].iloc[0, 0] = -1.0
    first = bars[SPY_15M]
    first.iloc[0, 0] = -2.0
    assert bars[SPY_15M].iloc[0, 0] == 100.0
    with pytest.raises(TypeError):
        bars[QQQ_15M] = first


def test_portfolio_bars_normalizes_its_keys():
    frames = base_frames()
    bars = PortfolioBars({("spy", "15M"): frames[SPY_15M]})
    assert list(bars) == [SPY_15M]
    assert len(bars) == 1
    assert SPY_15M in bars
    assert len(bars[("spy", "15m")]) == SCHEDULED_INTERVALS


def test_portfolio_bars_refuse_two_keys_that_normalize_to_one():
    frame = base_frames()[SPY_15M]
    with pytest.raises(ReplayInputError) as raised:
        PortfolioBars({("spy", "15m"): frame, ("SPY", "15M"): frame})
    assert raised.value.code == "duplicate_value"


@pytest.mark.parametrize(
    ("key", "code"),
    [
        ("SPY", "invalid_type"),
        (("SPY",), "invalid_value"),
        (("SPY", "15m", "extra"), "invalid_value"),
        ((None, "15m"), "invalid_type"),
        (("SPY", "5m"), "invalid_value"),
        (("SPY", " 15m"), "invalid_value"),
    ],
)
def test_portfolio_bars_refuse_a_key_outside_the_contract(key, code):
    frame = base_frames()[SPY_15M]
    with pytest.raises(ReplayInputError) as raised:
        PortfolioBars({key: frame})
    assert raised.value.code == code


@pytest.mark.parametrize("value", [None, "frame", base_frames()[SPY_15M].to_numpy()])
def test_portfolio_bars_refuse_a_value_that_is_not_a_frame(value):
    with pytest.raises(ReplayInputError) as raised:
        PortfolioBars({SPY_15M: value})
    assert raised.value.code == "invalid_type"


def test_an_absent_pair_raises_a_key_error():
    with pytest.raises(KeyError):
        PortfolioBars(base_frames())[("IWM", "15m")]


def test_membership_of_a_key_outside_the_contract_refuses_rather_than_denies():
    # A malformed key is a caller defect, and answering "no" to it would let
    # a typo read as an absent frame.
    bars = PortfolioBars(base_frames())
    with pytest.raises(ReplayInputError) as raised:
        _ = ("SPY", "5m") in bars
    assert raised.value.code == "invalid_value"


def test_a_prepared_pair_nobody_declared_raises_a_key_error():
    with pytest.raises(KeyError):
        prepared_base().frame("IWM", "15m")


# ------------------------------------------------------------- dependencies


def test_dependencies_normalize_into_the_fixed_order():
    dependencies = ReplayDependencies(
        timeframes=("1d", "15m", "1h", "1d"),
        reference_pairs=(
            ("spy", "15m"), ("aapl", "1d"), ("SPY", "15m"),
        ),
    )
    assert dependencies.timeframes == ("15m", "1h", "1d")
    assert dependencies.reference_pairs == (("AAPL", "1d"), ("SPY", "15m"))


@pytest.mark.parametrize(
    ("timeframes", "reference_pairs", "code"),
    [
        ((), (), "invalid_value"),
        (("1h",), (), "invalid_value"),
        (("15m", "5m"), (), "invalid_value"),
        ("15m", (), "invalid_type"),
        (("15m",), (("SP Y", "15m"),), "invalid_value"),
        (("15m",), (None,), "invalid_type"),
    ],
)
def test_dependencies_refuse_anything_outside_the_contract(
        timeframes, reference_pairs, code):
    with pytest.raises(ReplayInputError) as raised:
        ReplayDependencies(timeframes=timeframes, reference_pairs=reference_pairs)
    assert raised.value.code == code


# ----------------------------------------------------------------- preflight


def test_missing_input_refuses_during_preparation():
    bars = PortfolioBars(without_pair(base_frames(), "QQQ", "15m"))
    with pytest.raises(ReplayInputError) as exc:
        prepare_portfolio_bars(
            base_request(), bars, base_validated_schedule(), base_dependencies(),
        )
    assert exc.value.code == "missing_required_bar"
    assert exc.value.details["symbol"] == "QQQ"
    assert exc.value.details["timeframe"] == "15m"


def test_every_declared_pair_is_prepared_on_the_schedule_labels():
    prepared = prepared_base()
    assert prepared.pairs == (
        ("QQQ", "15m"), ("QQQ", "1h"), ("QQQ", "4h"), ("QQQ", "1d"),
        ("SPY", "15m"), ("SPY", "1h"), ("SPY", "4h"), ("SPY", "1d"),
    )
    assert len(prepared.frame("SPY", "15m")) == SCHEDULED_INTERVALS
    assert [len(prepared.frame("SPY", timeframe)) for timeframe in ("1h", "4h", "1d")] \
        == [2, 1, 1]
    assert prepared.frame("SPY", "1d").index[0] == ts("2026-11-25T05:00:00Z")


def test_preparation_takes_its_own_copy_of_every_frame():
    frames = base_frames()
    bars = PortfolioBars(frames)
    prepared = prepare_portfolio_bars(
        base_request(), bars, base_validated_schedule(), base_dependencies(),
    )
    owned = prepared.frame("SPY", "15m")
    assert owned is not bars[SPY_15M]
    owned.iloc[0, 0] = -5.0
    assert prepared.frame("SPY", "15m").iloc[0, 0] == -5.0
    assert bars[SPY_15M].iloc[0, 0] == 100.0
    assert frames[SPY_15M].iloc[0, 0] == 100.0


def test_a_surplus_frame_refuses():
    frames = base_frames()
    frames[("IWM", "15m")] = frames[SPY_15M].copy(deep=True)
    error = refuse(frames)
    assert error.code == "missing_required_bar"
    assert error.details["field"] == "unexpected_frame"


def test_a_missing_external_dependency_refuses():
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    request = base_request()
    frames = frames_for(request, base_schedule(), dependencies)
    error = refuse(without_pair(frames, "IWM", "15m"), dependencies=dependencies)
    assert (error.code, error.details["symbol"]) == ("missing_required_bar", "IWM")


def test_a_missing_benchmark_dependency_refuses():
    request = base_request(benchmark=BenchmarkSpec(
        kind="single_symbol", symbol="IWM", weighting="equal", rebalance="never",
    ))
    frames = frames_for(request, base_schedule(), base_dependencies())
    assert ("IWM", "15m") in frames
    error = refuse(without_pair(frames, "IWM", "15m"), request=request)
    assert (error.code, error.details["symbol"]) == ("missing_required_bar", "IWM")


def test_an_explicit_benchmark_symbol_needs_only_the_base_timeframe():
    request = base_request(benchmark=BenchmarkSpec(
        kind="single_symbol", symbol="IWM", weighting="equal", rebalance="never",
    ))
    frames = frames_for(request, base_schedule(), base_dependencies())
    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, base_schedule()),
        base_dependencies(),
    )
    assert ("IWM", "15m") in prepared.pairs
    assert ("IWM", "1h") not in prepared.pairs


def test_exact_reference_closure_never_builds_a_symbol_timeframe_cross_product():
    request = base_request(symbols=("SPY",))
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("QQQ", "1d"),),
    )
    frames = frames_for(request, schedule, dependencies)

    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, schedule),
        dependencies,
    )

    assert prepared.pairs == (("QQQ", "1d"), ("SPY", "15m"))
    assert ("SPY", "1d") not in prepared.pairs
    assert ("QQQ", "15m") not in prepared.pairs


def test_a_trading_symbol_covers_the_tail_and_a_context_symbol_stops_at_test_end():
    request = tail_request()
    dependencies = ReplayDependencies(
        timeframes=("15m", "1h"),
        reference_pairs=(("IWM", "15m"), ("IWM", "1h")),
    )
    schedule = validate_schedule(request, base_schedule())
    frames = frames_for(request, base_schedule(), dependencies)
    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), schedule, dependencies,
    )
    # The traded symbols carry every scheduled interval through `ic_tail_end`;
    # a context-only symbol stops at the last interval opening before
    # `test_end`, four intervals earlier.
    assert len(prepared.frame("SPY", "15m")) == SCHEDULED_INTERVALS
    assert len(prepared.frame("IWM", "15m")) == SCHEDULED_INTERVALS - 4
    assert prepared.frame("IWM", "15m").index[-1] == ts("2026-11-27T16:45:00Z")


def test_a_declared_timeframe_the_schedule_never_materialized_refuses():
    # The exactness of the label check is only as strong as the schedule
    # behind it. A play declaring 4h against a schedule with no 4h bars would
    # otherwise be handed an empty frame for the whole replay and emit
    # nothing, refusing nothing.
    schedule = schedule_with(context_bars=tuple(
        row for row in base_context_bars() if row.timeframe != "4h"))
    request = base_request(schedule_identity=schedule.identity)
    dependencies = ReplayDependencies(
        timeframes=("15m", "4h"), reference_pairs=(),
    )
    frames = frames_for(request, schedule, dependencies)
    with pytest.raises(ReplayInputError) as raised:
        prepare_portfolio_bars(
            request, PortfolioBars(frames), validate_schedule(request, schedule),
            dependencies,
        )
    assert (raised.value.code, raised.value.details["field"]) == (
        "missing_required_bar", "timeframe")
    assert raised.value.details["timeframe"] == "4h"


def test_a_context_bar_that_could_never_become_available_is_not_required():
    # The only scheduled four-hour bucket is labeled at `test_end`, so a
    # context-only symbol can never read it. Its frame is present and empty
    # rather than carrying data no context could ever show.
    request = tail_request()
    dependencies = ReplayDependencies(
        timeframes=("15m", "4h"),
        reference_pairs=(("IWM", "15m"), ("IWM", "4h")),
    )
    frames = frames_for(request, base_schedule(), dependencies)
    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, base_schedule()),
        dependencies,
    )
    assert len(prepared.frame("IWM", "4h")) == 0
    assert len(prepared.frame("SPY", "4h")) == 1


def test_a_context_symbol_refuses_bars_past_its_own_boundary():
    request = tail_request()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    frames = frames_for(request, base_schedule(), dependencies)
    frames[("IWM", "15m")] = bar_frame(
        scheduled_labels(base_schedule(), "15m", request.ic_tail_end))
    error = refuse(frames, request=request, dependencies=dependencies)
    assert (error.code, error.details["field"]) == ("missing_required_bar", "labels")


def test_sparse_external_history_is_reindexed_without_forward_fill():
    request = base_request()
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    frames = frames_for(request, schedule, dependencies)
    expected = list(frames[("IWM", "15m")].index)
    missing = expected[5]
    supplied = frames[("IWM", "15m")].drop(index=missing)
    frames[("IWM", "15m")] = supplied

    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, schedule),
        dependencies,
    )

    internal = prepared.frame("IWM", "15m")
    assert list(internal.index) == expected
    assert internal.loc[missing].isna().all()
    assert internal.iloc[4]["close"] == supplied.iloc[4]["close"]
    assert internal.iloc[6]["close"] == supplied.iloc[5]["close"]
    assert missing not in supplied.index


def test_empty_external_history_becomes_an_all_null_internal_frame():
    request = base_request()
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    frames = frames_for(request, schedule, dependencies)
    frames[("IWM", "15m")] = frames[("IWM", "15m")].iloc[:0]

    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, schedule),
        dependencies,
    )

    internal = prepared.frame("IWM", "15m")
    assert len(internal) == len(scheduled_labels(
        schedule, "15m", request.window.test_end))
    assert internal.isna().all().all()


def test_sparse_external_history_refuses_a_surplus_scheduled_label():
    request = base_request()
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    frames = frames_for(request, schedule, dependencies)
    external = frames[("IWM", "15m")]
    frames[("IWM", "15m")] = pd.concat([
        external,
        bar_frame((ts("2026-11-27T18:00:00Z"),), base=200.0),
    ])

    error = refuse(frames, request=request, dependencies=dependencies)
    assert (error.code, error.details["field"]) == (
        "missing_required_bar", "labels")


def test_pair_role_precedence_is_not_symbol_wide():
    request = base_request(symbols=("AAPL",))
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("AAPL", "1h"),),
    )
    frames = frames_for(request, schedule, dependencies)
    hourly_missing = frames[("AAPL", "1h")].index[0]
    frames[("AAPL", "1h")] = frames[("AAPL", "1h")].drop(
        index=hourly_missing)

    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames), validate_schedule(request, schedule),
        dependencies,
    )
    assert prepared.frame("AAPL", "1h").loc[hourly_missing].isna().all()

    missing_15m = frames[("AAPL", "15m")].index[0]
    frames[("AAPL", "15m")] = frames[("AAPL", "15m")].drop(index=missing_15m)
    error = refuse(frames, request=request, dependencies=dependencies)
    assert (error.code, error.details["symbol"], error.details["timeframe"]) == (
        "missing_required_bar", "AAPL", "15m")


def test_same_exact_pair_traded_role_precedes_external_role():
    request = base_request(symbols=("AAPL",))
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("AAPL", "15m"),),
    )
    frames = frames_for(request, schedule, dependencies)
    assert list(frames) == [("AAPL", "15m")]
    missing = frames[("AAPL", "15m")].index[0]
    frames[("AAPL", "15m")] = frames[("AAPL", "15m")].drop(index=missing)

    error = refuse(frames, request=request, dependencies=dependencies)
    assert (error.code, error.details["symbol"], error.details["timeframe"]) == (
        "missing_required_bar", "AAPL", "15m")


def test_same_exact_pair_benchmark_role_precedes_external_role():
    request = base_request(benchmark=BenchmarkSpec(
        kind="single_symbol", symbol="IWM", weighting="equal", rebalance="never",
    ))
    schedule = base_schedule()
    dependencies = ReplayDependencies(
        timeframes=("15m",), reference_pairs=(("IWM", "15m"),),
    )
    frames = frames_for(request, schedule, dependencies)
    assert list(key for key in frames if key == ("IWM", "15m")) == [
        ("IWM", "15m")]
    missing = frames[("IWM", "15m")].index[0]
    frames[("IWM", "15m")] = frames[("IWM", "15m")].drop(index=missing)

    error = refuse(frames, request=request, dependencies=dependencies)
    assert (error.code, error.details["symbol"], error.details["timeframe"]) == (
        "missing_required_bar", "IWM", "15m")


# ----------------------------------------------------------- frame integrity


@pytest.mark.parametrize(
    ("labels", "field"),
    [
        # One scheduled interval absent from the middle of the frame.
        ("missing", "labels"),
        # One interval that the schedule does not contain, past the cutoff.
        ("surplus", "labels"),
        # A label shifted off the scheduled open.
        ("shifted", "labels"),
    ],
)
def test_a_frame_whose_labels_leave_the_schedule_refuses(labels, field):
    frames = base_frames()
    scheduled = list(frames[SPY_15M].index)
    if labels == "missing":
        replacement = scheduled[:5] + scheduled[6:]
    elif labels == "surplus":
        replacement = scheduled + [ts("2026-11-27T18:00:00Z")]
    else:
        replacement = scheduled[:-1] + [ts("2026-11-27T17:50:00Z")]
    frames[SPY_15M] = bar_frame(replacement)
    error = refuse(frames)
    assert (error.code, error.details["field"]) == ("missing_required_bar", field)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param("naive", id="naive index"),
        pytest.param("exchange", id="non-UTC index"),
        pytest.param("duplicate", id="duplicate labels"),
        pytest.param("unsorted", id="unsorted labels"),
        pytest.param("missing label", id="a missing label"),
        pytest.param("range", id="not a datetime index"),
    ],
)
def test_a_frame_with_a_broken_index_refuses(index):
    frames = base_frames()
    frame = frames[SPY_15M].copy(deep=True)
    if index == "naive":
        frame.index = frame.index.tz_localize(None)
    elif index == "exchange":
        frame.index = frame.index.tz_convert("America/New_York")
    elif index == "duplicate":
        labels = list(frame.index)
        labels[1] = labels[0]
        frame.index = pd.DatetimeIndex(labels, tz="UTC")
    elif index == "unsorted":
        frame = frame.iloc[::-1]
    elif index == "missing label":
        labels = list(frame.index)
        labels[1] = pd.NaT
        frame.index = pd.DatetimeIndex(labels, tz="UTC")
    else:
        frame.index = pd.RangeIndex(len(frame))
    error = refuse({**frames, SPY_15M: frame})
    assert (error.code, error.details["field"]) == ("missing_required_bar", "index")


@pytest.mark.parametrize(
    ("column", "values", "field"),
    [
        pytest.param("open", None, "columns", id="missing column"),
        pytest.param("close", np.nan, "close", id="not a number"),
        pytest.param("close", np.inf, "close", id="infinite"),
        pytest.param("volume", -1.0, "volume", id="negative volume"),
    ],
)
def test_a_frame_with_malformed_columns_refuses(column, values, field):
    frames = base_frames()
    if values is None:
        frames[SPY_15M] = frames[SPY_15M].drop(columns=[column])
    else:
        frames = replaced_column(frames, SPY_15M, column, values)
    error = refuse(frames)
    assert (error.code, error.details["field"]) == ("missing_required_bar", field)


def test_a_frame_naming_one_column_twice_refuses():
    frames = base_frames()
    frame = frames[SPY_15M]
    doubled = pd.concat([frame, frame[["open"]]], axis=1)
    error = refuse({**frames, SPY_15M: doubled})
    assert (error.code, error.details["field"]) == ("missing_required_bar", "columns")


@pytest.mark.parametrize("dtype", ["bool", "object", "str"])
def test_a_frame_column_that_is_not_a_number_refuses(dtype):
    frames = base_frames()
    frame = frames[SPY_15M].copy(deep=True)
    frame["volume"] = frame["volume"].astype(dtype)
    error = refuse({**frames, SPY_15M: frame})
    assert (error.code, error.details["field"]) == ("missing_required_bar", "dtype")


def test_a_provider_frame_prepares_as_binary64_ohlcv():
    # What an Alpaca-written parquet actually looks like: the five columns in
    # provider order, an integer volume, and two columns core does not read.
    frames = base_frames()
    frame = frames[SPY_15M]
    provider = pd.DataFrame(
        {
            "volume": frame["volume"].astype("int64"),
            "trade_count": np.arange(len(frame), dtype="int64"),
            "close": frame["close"],
            "high": frame["high"],
            "low": frame["low"],
            "open": frame["open"],
            "vwap": frame["close"],
        },
        index=frame.index,
    )
    prepared = prepare_portfolio_bars(
        base_request(), PortfolioBars({**frames, SPY_15M: provider}),
        base_validated_schedule(), base_dependencies(),
    )
    owned = prepared.frame("SPY", "15m")
    assert list(owned.columns) == ["open", "high", "low", "close", "volume"]
    assert set(owned.dtypes) == {np.dtype("float64")}
    assert owned["open"].iloc[0] == 100.0
    assert owned["volume"].iloc[0] == 1000.0


@pytest.mark.parametrize(
    ("column", "value", "field"),
    [
        ("high", 99.0, "high"),
        # Above the open but still under the high, so only the low rule can
        # catch it.
        ("low", 100.15, "low"),
    ],
)
def test_a_frame_with_impossible_geometry_refuses(column, value, field):
    frames = base_frames()
    frame = frames[SPY_15M].copy(deep=True)
    frame.iloc[0, frame.columns.get_loc(column)] = value
    error = refuse({**frames, SPY_15M: frame})
    assert (error.code, error.details["field"]) == ("missing_required_bar", field)


def test_preparation_refuses_a_schedule_built_for_another_request():
    """`mismatched_schedule`, by name, and not a generic structural refusal.

    Three doors test this one condition: this one, `_Ledger`, and
    `_PortfolioRuntime`. The bar preflight runs FIRST inside `run_portfolio`,
    so it is the only one a caller ever reaches, and while it answered
    `invalid_value` the named code was published and unreachable and a
    mismatched schedule arrived indistinguishable from every other structural
    refusal in the contract. The code is what a caller matches on, so it is
    what this asserts.
    """
    with pytest.raises(ReplayInputError) as raised:
        prepare_portfolio_bars(
            tail_request(), PortfolioBars(base_frames()), base_validated_schedule(),
            base_dependencies(),
        )
    assert raised.value.code == "mismatched_schedule"
    assert raised.value.details["field"] == "schedule"


def test_every_door_that_checks_the_schedule_refuses_under_one_code():
    """The bar door, the ledger, and the runtime answer the same condition.

    Asserted together rather than three times apart, because the value of a
    named code is that it means one thing wherever it comes from: a caller
    branching on `mismatched_schedule` must not have to know which door
    happened to run first.
    """
    request, schedule = tail_request(), base_validated_schedule()
    prepared = prepared_base()
    doors = {
        "bars": lambda: prepare_portfolio_bars(
            request, PortfolioBars(base_frames()), schedule, base_dependencies()),
        "ledger": lambda: _Ledger(request, schedule),
        "runtime": lambda: _PortfolioRuntime(
            request, schedule, strategy_registry(), prepared, base_dependencies()),
    }
    codes = {}
    for name, call in doors.items():
        with pytest.raises(ReplayInputError) as raised:
            call()
        codes[name] = raised.value.code

    assert codes == {name: "mismatched_schedule" for name in doors}


# ---------------------------------------------------------- scheduled context


def test_a_context_shows_only_what_the_schedule_has_made_available():
    prepared = prepared_base()
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), base_validated_schedule(),
        base_dependencies(), vocabulary=core_vocabulary(),
    )
    assert context.symbol == "SPY"
    assert context.now == ts("2026-11-27T15:00:00Z")
    # 26 warmup intervals plus the two of the test session that have closed.
    assert len(context.bars["15m"]) == 28
    assert context.bars["15m"].index[-1] == ts("2026-11-27T14:45:00Z")
    # Both hourly bars and the prior session's daily bar are available; the
    # four-hour bucket does not become available until after the schedule ends.
    assert len(context.bars["1h"]) == 2
    assert len(context.bars["1d"]) == 1
    assert len(context.bars["4h"]) == 0
    assert context.cursor == {"15m": 27, "1h": 1, "4h": -1, "1d": 0}


def test_a_context_at_the_first_test_close_sees_no_test_session_daily_bar():
    prepared = prepared_base()
    schedule = base_validated_schedule()
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T14:45:00Z"), schedule,
        base_dependencies(), vocabulary=core_vocabulary(),
    )
    assert len(context.bars["15m"]) == 27
    assert len(context.bars["1h"]) == 1
    assert list(context.bars["1d"].index) == [ts("2026-11-25T05:00:00Z")]


def test_a_context_only_carries_the_declared_timeframes():
    prepared = prepare_portfolio_bars(
        base_request(), PortfolioBars(base_frames()), base_validated_schedule(),
        base_dependencies(),
    )
    dependencies = ReplayDependencies(timeframes=("15m", "1d"), reference_pairs=())
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), base_validated_schedule(),
        dependencies, vocabulary=core_vocabulary(),
    )
    assert set(context.bars) == {"15m", "1d"}
    assert context.tfs.all == ("15m", "1d")
    assert context.driving_bars is context.bars["15m"]


def test_a_context_refuses_anything_that_is_not_a_grammar():
    """The grammar is an argument now, so it is a checked argument.

    A vocabulary factory is caller code, and one that hands back a mapping or a
    None would otherwise surface as an `AttributeError` from inside `FrameEval`
    on the first node it evaluated, outside the closed refusal taxonomy and
    naming no argument.
    """
    for bad in (None, {}, core_vocabulary().indicators):
        with pytest.raises(ReplayInputError) as raised:
            build_scheduled_context(
                prepared_base(), "SPY", ts("2026-11-27T15:00:00Z"),
                base_validated_schedule(), base_dependencies(), vocabulary=bad,
            )
        assert raised.value.code == "invalid_type"
        assert raised.value.details["field"] == "vocabulary"


def test_a_context_cannot_be_built_off_a_scheduled_close():
    with pytest.raises(ReplayInputError) as raised:
        build_scheduled_context(
            prepared_base(), "SPY", ts("2026-11-27T15:07:00Z"),
            base_validated_schedule(), base_dependencies(),
            vocabulary=core_vocabulary(),
        )
    assert raised.value.code == "invalid_context_time"


def test_a_context_cannot_be_built_inside_the_ic_tail():
    # The tail closes are scheduled base closes too, so being a real close is
    # not enough: tail bars belong to the IC lens after the replay, and no
    # strategy may see them.
    request = tail_request()
    schedule = validate_schedule(request, base_schedule())
    dependencies = ReplayDependencies(timeframes=("15m",), reference_pairs=())
    prepared = prepare_portfolio_bars(
        request, PortfolioBars(frames_for(request, base_schedule(), dependencies)),
        schedule, dependencies,
    )
    last_test_close = build_scheduled_context(
        prepared, "SPY", request.window.test_end, schedule, dependencies,
        vocabulary=core_vocabulary())
    assert len(last_test_close.bars["15m"]) == SCHEDULED_INTERVALS - 4
    with pytest.raises(ReplayInputError) as raised:
        build_scheduled_context(
            prepared, "SPY", ts("2026-11-27T18:00:00Z"), schedule, dependencies,
            vocabulary=core_vocabulary())
    assert raised.value.code == "invalid_context_time"


def test_mutating_a_context_frame_cannot_reach_the_prepared_frame():
    prepared = prepared_base()
    context = build_scheduled_context(
        prepared, "SPY", ts("2026-11-27T15:00:00Z"), base_validated_schedule(),
        base_dependencies(), vocabulary=core_vocabulary(),
    )
    context.bars["15m"].iloc[0, 0] = -3.0
    context.bars["1d"].iloc[0, 0] = -4.0
    context.bars["15m"]["close"] = 0.0
    assert prepared.frame("SPY", "15m").iloc[0, 0] == 100.0
    assert prepared.frame("SPY", "15m")["close"].iloc[0] == 100.05
    assert prepared.frame("SPY", "1d").iloc[0, 0] == 100.0


def test_the_declared_pandas_floor_is_what_keeps_the_context_views_safe():
    # The two mutation regressions above pass because copy-on-write is
    # unconditional from pandas 3.0. It is opt-in in 2.x, and a lock file does
    # not travel to a consumer that pins this repo by revision, so the
    # DECLARED floor is the whole guarantee: under an installed 2.x a strategy
    # writing into ctx.bars[tf] would write through into the replay's prices.
    manifest = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_bytes().decode())
    declared = [name for name in manifest["project"]["dependencies"]
                if name.startswith("pandas")]
    assert declared == ["pandas>=3"]
    assert int(pd.__version__.split(".")[0]) >= 3


def test_a_context_frame_hands_out_no_writable_array():
    # The last way through: a strategy reaching past pandas into the buffer.
    # Copy-on-write hands out read-only arrays, which is what makes the
    # zero-copy prefix safe to expose at all.
    context = build_scheduled_context(
        prepared_base(), "SPY", ts("2026-11-27T15:00:00Z"),
        base_validated_schedule(), base_dependencies(),
        vocabulary=core_vocabulary(),
    )
    values = context.bars["15m"]["open"].to_numpy()
    assert not values.flags.writeable
    with pytest.raises(ValueError):
        values[0] = -9.0


def test_a_context_for_an_undeclared_symbol_raises():
    with pytest.raises(KeyError):
        build_scheduled_context(
            prepared_base(), "IWM", ts("2026-11-27T15:00:00Z"),
            base_validated_schedule(), base_dependencies(),
            vocabulary=core_vocabulary(),
        )
