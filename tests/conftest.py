import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CATALOG_SPECS = (Path(__file__).resolve().parents[1]
                 / "nakagai" / "strategies" / "catalog" / "specs")


@pytest.fixture
def make_bars():
    """Factory: n bars of the given timeframe starting at start (UTC), gently rising."""

    def _make(n=10, timeframe="15m", start="2026-06-01 13:30", base=100.0):
        freq = {"15m": "15min", "1h": "1h", "1d": "1D"}[timeframe]
        idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC", name="ts")
        close = base + np.arange(n) * 0.1
        df = pd.DataFrame(
            {
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.15,
                "close": close,
                "volume": 1_000.0,
            },
            index=idx,
        )
        return df

    return _make


@pytest.fixture
def load_spec():
    """Factory: one shipped catalog spec by name, read fresh from its JSON.

    Read rather than taken from catalog.load_entries because that loader is
    @cache'd and hands every caller the same dict; a test that mutated it would
    poison the rest of the session.
    """

    def _load(name: str) -> dict:
        return json.loads((CATALOG_SPECS / f"{name}.json").read_text())["spec"]

    return _load
