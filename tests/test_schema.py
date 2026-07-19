import pandas as pd
import pytest

from nakagai.data.schema import (
    BAR_COLUMNS,
    DEFAULT_TIMEFRAMES,
    TIMEFRAMES,
    TimeframeSet,
    validate_bars,
)


def test_validate_passes_and_sorts(make_bars):
    df = make_bars(5)
    shuffled = df.sample(frac=1, random_state=1)
    out = validate_bars(shuffled)
    assert list(out.columns) == BAR_COLUMNS
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "ts"


def test_validate_rejects_naive_index(make_bars):
    df = make_bars(3)
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware UTC"):
        validate_bars(df)


def test_validate_rejects_missing_columns(make_bars):
    df = make_bars(3).drop(columns=["volume"])
    with pytest.raises(ValueError, match="volume"):
        validate_bars(df)


def test_constants():
    assert TIMEFRAMES == ("15m", "1h", "1d")


def test_default_timeframes_match_legacy_axis():
    assert DEFAULT_TIMEFRAMES.all == ("15m", "1h", "1d")
    assert DEFAULT_TIMEFRAMES.step == pd.Timedelta(minutes=15)
    assert DEFAULT_TIMEFRAMES.deltas["1h"] == pd.Timedelta(hours=1)
    assert DEFAULT_TIMEFRAMES.session_aligned == frozenset({"1d"})
    assert TIMEFRAMES == DEFAULT_TIMEFRAMES.all


def test_timeframe_set_validation():
    with pytest.raises(ValueError):
        TimeframeSet(driving="1h")  # driving needs a delta
    with pytest.raises(ValueError):
        TimeframeSet(driving="1d", session_aligned=frozenset({"1d"}))  # driving cannot be session-aligned
    with pytest.raises(ValueError):
        TimeframeSet(driving="15m", higher=("4h",),
                     deltas={"15m": pd.Timedelta(minutes=15)})  # 4h has no delta and is not session-aligned


def test_timeframe_set_custom_axis():
    tfs = TimeframeSet(driving="1h", higher=("1d",),
                       deltas={"1h": pd.Timedelta(hours=1)},
                       session_aligned=frozenset({"1d"}))
    assert tfs.all == ("1h", "1d")
    assert tfs.step == pd.Timedelta(hours=1)
