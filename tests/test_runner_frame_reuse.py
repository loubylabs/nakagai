"""run_grid loads each (symbol, timeframe) once, not once per window.

The frames now travel inside the job tuples, so they have to survive the
ProcessPoolExecutor path as well as the in-process one.
"""

from pathlib import Path

import pandas as pd

from nakagai.data.cache import BarCache
from nakagai.data.schema import DEFAULT_TIMEFRAMES
from nakagai.engine.runner import run_grid
from nakagai.engine.windows import walk_forward
from nakagai.strategies.catalog import load_catalog

SPECS = Path(__file__).resolve().parents[1] / "nakagai" / "strategies" / "catalog" / "specs"


def _registry():
    return load_catalog(SPECS)


def test_frames_load_once_per_symbol_not_once_per_window(tmp_path, make_bars, monkeypatch):
    cache = BarCache(tmp_path / "cache")
    for tf in ("15m", "1h", "1d"):
        cache.upsert("SPY", tf, make_bars(n=900, timeframe=tf,
                                          start="2024-01-02 14:30"))
    windows = walk_forward(pd.Timestamp("2024-01-02", tz="UTC"),
                           pd.Timestamp("2026-01-02", tz="UTC"), 4, 1)
    calls = []
    real = BarCache.load
    monkeypatch.setattr(BarCache, "load",
                        lambda self, s, tf: (calls.append((s, tf)), real(self, s, tf))[1])
    run_grid(str(tmp_path / "cache"), ["rsi_reversion"], ["SPY"], windows[:6],
             workers=1, out=str(tmp_path / "runs.parquet"), registry=_registry)
    want = len(DEFAULT_TIMEFRAMES.all)
    assert len(calls) == want, f"expected {want} loads (one per timeframe), got {len(calls)}"


# run_id and ts_run are stamped per row and are deliberately non-deterministic.
_NONDET = ["run_id", "ts_run"]


def _grid(tmp_path, workers: int, windows) -> pd.DataFrame:
    rows = run_grid(str(tmp_path / "cache"), ["rsi_reversion"], ["SPY", "QQQ"],
                    windows, workers=workers,
                    out=str(tmp_path / f"runs_{workers}.parquet"),
                    registry=_registry)
    return rows.drop(columns=_NONDET)


def test_pooled_workers_match_the_in_process_path(tmp_path, make_bars):
    """A pooled grid must pickle the frames and return the same rows.

    The jobs used to carry a path string; they now carry a MemoryBars, so this
    is the test that the pool path still works at all, not only that it agrees.
    """
    cache = BarCache(tmp_path / "cache")
    for sym, base in (("SPY", 100.0), ("QQQ", 350.0)):
        for tf in ("15m", "1h", "1d"):
            cache.upsert(sym, tf, make_bars(n=900, timeframe=tf, base=base,
                                            start="2024-01-02 14:30"))
    windows = walk_forward(pd.Timestamp("2024-01-02", tz="UTC"),
                           pd.Timestamp("2026-01-02", tz="UTC"), 4, 1)[:4]
    serial = _grid(tmp_path, 1, windows)
    pooled = _grid(tmp_path, 3, windows)
    assert len(pooled) == len(serial) == 8
    pd.testing.assert_frame_equal(pooled, serial)
