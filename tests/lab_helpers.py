"""Shared fixtures for the lab tests.

Bars are generated at 15m over RTH sessions and the 1h and 1d views are
resampled up from that same series, twelve months by default: long enough for
a walk-forward split, small enough that a calibration replicate is seconds
rather than minutes.

Bars are built at 15m, not 1h, because the engine's default timeframe set
(`nakagai.data.schema.DEFAULT_TIMEFRAMES`) drives off `15m`, and
`MemoryBars.load` returns an empty frame for a missing key instead of
raising. Generating only `1h` and `1d` starves that driving timeframe and the
engine silently steps zero bars.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import validate_bars
from nakagai.engine.windows import walk_forward
from nakagai.strategies.catalog import load_catalog
from nakagai.strategies.composite import CompositeStrategy
from nakagai.strategies.rules import core_vocabulary
from nakagai.strategies.rules.strategy import RuleStrategy

SPECS_DIR = Path(__file__).resolve().parents[1] / "nakagai/strategies/catalog/specs"
BARS_PER_SESSION = 28          # RTH 14:00-21:00 UTC at 15m
SESSIONS_PER_MONTH = 21
MONTHS = 12                    # default span

AGG = {"open": "first", "high": "max", "low": "min",
       "close": "last", "volume": "sum"}

BASE_SPEC = {
    "version": 2,
    "name": "rsi_reversion",
    "timeframe": "1h",
    "long": {"all": [
        {"lhs": {"ind": "rsi", "n": 14}, "op": "crosses_above", "rhs": 45},
    ]},
    "risk": {"stop": {"kind": "atr", "n": 14, "mult": 2.0},
             "target": {"kind": "rr", "rr": 1.5}},
}


def _session_index(months: int) -> pd.DatetimeIndex:
    sessions = pd.bdate_range("2024-01-02", periods=months * SESSIONS_PER_MONTH,
                              tz="UTC")
    return pd.DatetimeIndex(
        [d + pd.Timedelta(hours=14) + pd.Timedelta(minutes=15 * i)
         for d in sessions for i in range(BARS_PER_SESSION)], name="ts")


def _frames_from_close(idx, close, symbol: str) -> dict:
    """15m bars, plus the 1h and 1d views resampled up from them.

    Derived rather than generated independently: the engine steps 15m bars
    while a v2 spec's rules read its own timeframe, so the two must describe
    the same price path or a signal and its fill refer to different
    histories.
    """
    prev = close.shift(1).fillna(close.iloc[0])
    m15 = validate_bars(pd.DataFrame({
        "open": prev,
        "high": np.maximum(close, prev) * 1.002,
        "low": np.minimum(close, prev) * 0.998,
        "close": close,
        "volume": 1_000_000.0,
    }, index=idx))
    return {
        (symbol, "15m"): m15,
        (symbol, "1h"): validate_bars(m15.resample("1h").agg(AGG).dropna()),
        (symbol, "1d"): validate_bars(m15.resample("1D").agg(AGG).dropna()),
    }


def random_walk_frames(symbol: str = "TEST", seed: int = 0,
                       months: int = MONTHS) -> dict:
    """Driftless geometric random walk: no exploitable structure whatsoever.

    This is the calibration substrate. Any strategy's true edge here is zero,
    so a survivor is a false positive by construction.
    """
    idx = _session_index(months)
    rng = np.random.default_rng(seed)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(idx)))), index=idx)
    return _frames_from_close(idx, close, symbol)


def oscillating_frames(symbol: str = "TEST", seed: int = 0,
                       months: int = MONTHS) -> dict:
    """A strong mean-reverting cycle plus noise: a real, capturable effect.

    The positive control. Permuting the bars destroys the cycle while leaving
    the return marginals intact, so a working null must find this significant.
    """
    idx = _session_index(months)
    rng = np.random.default_rng(seed)
    t = np.arange(len(idx))
    cycle = 0.06 * np.sin(2 * np.pi * t / 160.0)     # 160 bars at 15m
    noise = rng.normal(0.0, 0.0005, len(idx))
    close = pd.Series(100.0 * np.exp(cycle + np.cumsum(noise)), index=idx)
    return _frames_from_close(idx, close, symbol)


def lab_registry():
    """A zero-arg registry callable of the shape run_one requires: the catalog
    plays, the raw `rules` escape hatch, and `composite` bound to those members
    so a composite block can resolve its legs."""
    catalog = load_catalog(SPECS_DIR, core_vocabulary)
    members = {**catalog, "rules": RuleStrategy}

    def registry():
        return {**members, "composite": CompositeStrategy.bound(members)}

    return registry


def short_windows(frames: dict, symbol: str, count: int = 3) -> list:
    """The first `count` walk-forward windows over the cached span.

    Three two-month windows, not two one-month ones: the trial has to place
    enough trades that its profit factor is a statistic rather than an
    accident.
    """
    bars = frames[(symbol, "15m")]
    windows = walk_forward(bars.index[0], bars.index[-1],
                           train_months=2, test_months=2, step_months=2)
    if len(windows) < count:
        raise ValueError(f"only {len(windows)} windows in this span, need {count}")
    return windows[:count]


def memory_cache(frames: dict) -> MemoryBars:
    return MemoryBars(frames)
