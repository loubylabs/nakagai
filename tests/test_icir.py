"""Rank-IC / IR: Spearman IC of spec margins vs forward returns, per window."""

import numpy as np
import pandas as pd
import pytest

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import TimeframeSet
from nakagai.engine.windows import Window
from nakagai.icir import icir_fields, rank_ic, window_icir
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.margins import condition_margin

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


def _order_block_frame(n=400):
    # Real (non-flat) OHLC, unlike _frame's open==close bars: order_block needs
    # actual candle bodies to find a displacement (body >= 1.5x ATR) with an
    # opposing candle before it, inside its 40-bar lookback. One such pattern
    # near the tail would leave every earlier row NaN and there would be no
    # factor to score, so the pattern repeats every 25 bars and the level moves
    # through the window the way a point-in-time level does.
    idx = pd.date_range("2026-01-05 14:30", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    step = rng.normal(0, 0.05, n)
    step[10::25] = -1.0                   # opposing (down) candle
    step[12::25] = 3.0                    # displacement (big up) candle
    close = 100 + np.cumsum(step)
    open_ = np.r_[close[0], close[:-1]]   # body of bar i is exactly step[i]
    high = np.maximum(open_, close) + 0.2
    low = np.minimum(open_, close) - 0.2
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": 1000.0}, index=idx)


def test_window_icir_covers_end_anchored_primitives():
    # The lens used to refuse these specs outright, because the margin walker
    # broadcast order_block's one end-of-window float across every row. The
    # primitive is row-wise now and the window declares its span, so the factor
    # is point-in-time and the lens has an answer instead of an abstention.
    frame = _order_block_frame()
    cache = MemoryBars({("SPY", "15m"): frame})
    spec = {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": {"prim": "order_block", "direction": "long",
                                      "field": "top"}}]}}
    start, end = frame.index[200], frame.index[-1] + pd.Timedelta(minutes=15)
    w = Window(frame.index[0], start, start, end)
    out = window_icir(spec, cache, "SPY", w, tfs=TFS)
    assert all(out[f"ic_n_{k}"] > 0 for k in (1, 5, 20))


def test_window_icir_end_anchored_factor_is_not_broadcast():
    # The point-in-time proof: the margin the lens scores must be the one the
    # window's own rows would have seen, row by row, not one level from the end
    # of the window repeated. A broadcast level would make close - level a
    # simple monotone function of close, so its rank would equal close's rank.
    frame = _order_block_frame()
    node = {"prim": "order_block", "direction": "long", "field": "top"}
    fe = FrameEval({"15m": frame}, TFS, "SPY")
    in_win = frame.index[200:]
    fe.set_span("15m", 200, len(frame))
    level = fe.series(node, "15m").loc[in_win]
    assert level.nunique() > 1, "the level must move within the window"
    margin = condition_margin({"lhs": {"src": "close"}, "op": ">", "rhs": node},
                              fe, "15m").loc[in_win]
    assert not margin.rank(pct=True).equals(frame["close"].loc[in_win].rank(pct=True))


def test_window_icir_forward_returns_extend_past_test_end():
    # The factor must be point-in-time, but realized forward returns may look
    # past test_end: a cache that extends 30 bars beyond the window shouldn't
    # structurally lose ic_20 just because the window itself is short.
    closes = 100 + 10 * np.sin(np.linspace(0, 6 * np.pi, 430))
    frame = _frame(closes)
    cache = MemoryBars({("SPY", "15m"): frame})
    spec = {"version": 2, "name": "t", "timeframe": "15m",
            "long": {"all": [{"lhs": {"src": "close"}, "op": ">",
                              "rhs": {"ind": "sma", "n": 5}}]}}
    start, end = frame.index[200], frame.index[400]
    w = Window(frame.index[0], start, start, end)
    out = window_icir(spec, cache, "SPY", w, tfs=TFS)
    assert out["ic_n_1"] == 200 and out["ic_n_20"] == 200


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
