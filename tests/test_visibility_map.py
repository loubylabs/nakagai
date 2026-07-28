"""The vectorized visibility map equals closed_before at every row."""

import numpy as np
import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES as TFS
from nakagai.engine.context import closed_before, visible_counts


def _frame(n, freq, start="2026-01-05 14:30"):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    c = pd.Series(np.arange(n, dtype=float), index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": 1000.0}, index=idx)


def _daily(n, start="2026-01-05"):
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC")
    c = pd.Series(np.arange(n, dtype=float), index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c,
                         "volume": 1000.0}, index=idx)


def test_map_matches_closed_before_for_intraday_source():
    driving = _frame(200, "15min")
    src = _frame(60, "1h")
    closes = driving.index + TFS.step
    got = visible_counts(src.index, closes, "1h", TFS)
    want = [len(closed_before(src, "1h", now, TFS)) for now in closes]
    assert list(got) == want


def test_map_matches_closed_before_for_session_aligned_source():
    driving = _frame(400, "15min")
    src = _daily(30)
    closes = driving.index + TFS.step
    got = visible_counts(src.index, closes, "1d", TFS)
    want = [len(closed_before(src, "1d", now, TFS)) for now in closes]
    assert list(got) == want


def test_map_matches_closed_before_for_the_driving_frame_itself():
    driving = _frame(200, "15min")
    closes = driving.index + TFS.step
    got = visible_counts(driving.index, closes, "15m", TFS)
    want = [len(closed_before(driving, "15m", now, TFS)) for now in closes]
    assert list(got) == want


def test_empty_source_is_all_zero():
    driving = _frame(10, "15min")
    empty = pd.DatetimeIndex([], tz="UTC")
    got = visible_counts(empty, driving.index + TFS.step, "1h", TFS)
    assert list(got) == [0] * len(driving)
