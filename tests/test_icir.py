"""Rank-IC / IR: Spearman IC of spec margins vs forward returns, per window."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import TimeframeSet
from nakagai.engine.windows import Window
from nakagai.icir import icir_fields, rank_ic, window_icir

TFS = TimeframeSet(driving="15m", deltas={"15m": pd.Timedelta(minutes=15)})


def _frame(closes, start="2026-01-05 14:30"):
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1000.0})


def test_rank_ic_is_spearman():
    idx = pd.date_range("2026-01-05", periods=50, freq="15min", tz="UTC")
    f = pd.Series(np.arange(50, dtype=float), index=idx)
    ic, n = rank_ic(f, f ** 3)          # monotone nonlinear: rank corr is 1
    assert ic == 1.0 and n == 50
    ic, _ = rank_ic(f, -f)
    assert ic == -1.0
    rng = np.random.default_rng(0)
    ic, _ = rank_ic(f, pd.Series(rng.normal(size=50), index=idx))
    assert abs(ic) < 0.35


def test_rank_ic_guards():
    idx = pd.date_range("2026-01-05", periods=50, freq="15min", tz="UTC")
    f = pd.Series(np.arange(50, dtype=float), index=idx)
    assert rank_ic(f.iloc[:5], f.iloc[:5]) == (None, 5)          # < MIN_IC_OBS
    const = pd.Series(1.0, index=idx)
    assert rank_ic(const, f) == (None, 50)                       # no variance
    assert rank_ic(f, pd.Series(np.nan, index=idx)) == (None, 0)


def test_window_icir_positive_for_momentum_factor():
    closes = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, 400))
    frame = _frame(closes)
    cache = MemoryBars({("SPY", "15m"): frame})
    spec = {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": {"ind": "sma", "n": 5}}]}}
    start, end = frame.index[200], frame.index[-1] + pd.Timedelta(minutes=15)
    w = Window(frame.index[0], start, start, end)
    out = window_icir(spec, cache, "SPY", w, tfs=TFS)
    assert out["ic_1"] > 0.5             # above-sma predicts the next move up
    assert out["ic_n_1"] == 199 and out["ic_n_20"] == 180
    assert set(out) == {"ic_1", "ic_5", "ic_20", "ic_n_1", "ic_n_5", "ic_n_20"}


def test_window_icir_constant_prices_yield_none():
    frame = _frame([100.0] * 120)
    cache = MemoryBars({("SPY", "15m"): frame})
    spec = {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": {"ind": "sma", "n": 5}}]}}
    start, end = frame.index[60], frame.index[-1] + pd.Timedelta(minutes=15)
    out = window_icir(spec, cache, "SPY", Window(frame.index[0], start, start, end),
                      tfs=TFS)
    assert out["ic_1"] is None and out["ic_n_1"] > 0


def test_icir_fields_aggregates_windows():
    runs = pd.DataFrame({"ic_1": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                         "ic_n_1": [100] * 6,
                         "ic_5": [None] * 6, "ic_n_5": [0] * 6,
                         "ic_20": [None] * 6, "ic_n_20": [0] * 6})
    out = icir_fields(runs)
    h1 = out["h1"]
    assert h1["ic_mean"] == 0.15 and h1["ic_positive_rate"] == 1.0
    assert h1["ic_ir"] == pytest.approx(2.739, abs=1e-3)
    assert h1["n_windows"] == 6 and h1["n_obs"] == 600
    assert out["h5"] == {"ic_mean": None, "ic_std": None, "ic_ir": None,
                         "ic_positive_rate": None, "n_windows": 0, "n_obs": 0}


def test_icir_fields_single_window_has_no_ir():
    runs = pd.DataFrame({"ic_1": [0.1], "ic_n_1": [50]})
    h1 = icir_fields(runs)["h1"]
    assert h1["ic_mean"] == 0.1 and h1["ic_std"] is None and h1["ic_ir"] is None


def test_icir_fields_missing_columns():
    h1 = icir_fields(pd.DataFrame({"strategy": ["x"]}))["h1"]
    assert h1["n_windows"] == 0 and h1["ic_mean"] is None
