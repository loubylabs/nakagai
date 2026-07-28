"""Row-wise end-anchored primitives equal the scalar function on each prefix."""

import numpy as np
import pandas as pd
import pytest

from nakagai.strategies.rules.primitives import (END_ANCHORED, end_anchored_series,
                                                 fvg_nearest, order_block)


def _bars(n=120, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    return pd.DataFrame({"open": close - rng.normal(0, 0.1, n),
                         "high": close + np.abs(rng.normal(0, 0.5, n)),
                         "low": close - np.abs(rng.normal(0, 0.5, n)),
                         "close": close, "volume": 1000.0}, index=idx)


@pytest.mark.parametrize("name,fn,args", [
    ("order_block", order_block, {"direction": "long", "field": "top",
                                  "body_atr": 1.5, "lookback": 40}),
    ("fvg_nearest", fvg_nearest, {"direction": "long", "field": "top",
                                  "state": "open", "min_size_atr": 0.25,
                                  "lookback": 40}),
])
def test_series_equals_scalar_on_every_prefix(name, fn, args):
    bars = _bars()
    lo, hi = 60, len(bars)
    got = end_anchored_series(name, None, bars, lo, hi, **args)
    assert list(got.index) == list(bars.index[lo:hi])
    for i in range(lo, hi):
        want = fn(None, bars.iloc[: i + 1], **args)
        a, b = got.iloc[i - lo], want
        assert (pd.isna(a) and pd.isna(b)) or a == b, f"row {i}: {a} != {b}"


def test_registry_names_match_the_functions_that_are_end_anchored():
    assert END_ANCHORED == {"fvg_nearest", "order_block"}
