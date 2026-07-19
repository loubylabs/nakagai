"""Bar permutation: multiset-preserving, order-destroying, deterministic."""

import numpy as np
import pandas as pd
import pytest

from nakagai.engine.permutation import permutation_seed, permute_bars


def _bars(n=300, seed=42):
    """Random-walk OHLCV bars with real gaps and intrabar range."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=n)))
    gap = np.exp(rng.normal(0, 0.002, size=n))
    open_ = np.concatenate([[100.0], close[:-1]]) * gap
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.003, size=n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.003, size=n)))
    idx = pd.date_range("2026-01-05", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close,
                         "volume": rng.uniform(500, 5000, size=n)}, index=idx)


def _gaps_and_shapes(df):
    o, h, l, c = (np.log(df[k].to_numpy()) for k in ("open", "high", "low", "close"))
    gaps = o[1:] - c[:-1]
    shapes = np.column_stack([h[1:] - o[1:], l[1:] - o[1:], c[1:] - o[1:]])
    return gaps, shapes


def test_permute_preserves_gap_and_shape_multisets():
    orig = _bars()
    perm = permute_bars(orig, np.random.default_rng(1))
    g0, s0 = _gaps_and_shapes(orig)
    g1, s1 = _gaps_and_shapes(perm)
    np.testing.assert_allclose(np.sort(g0), np.sort(g1), rtol=1e-9)
    for col in range(3):
        np.testing.assert_allclose(np.sort(s0[:, col]), np.sort(s1[:, col]), rtol=1e-9)
    np.testing.assert_allclose(np.sort(orig["volume"].to_numpy()),
                               np.sort(perm["volume"].to_numpy()), rtol=1e-9)


def test_permute_destroys_ordering():
    orig = _bars()
    perm = permute_bars(orig, np.random.default_rng(1))
    assert not np.allclose(orig["close"].to_numpy(), perm["close"].to_numpy())


def test_permute_keeps_ohlc_invariants():
    perm = permute_bars(_bars(), np.random.default_rng(2))
    assert (perm["high"] >= perm["open"] - 1e-9).all()
    assert (perm["high"] >= perm["close"] - 1e-9).all()
    assert (perm["low"] <= perm["open"] + 1e-9).all()
    assert (perm["low"] <= perm["close"] + 1e-9).all()


def test_permute_keeps_index_and_anchor_bar():
    orig = _bars()
    perm = permute_bars(orig, np.random.default_rng(3))
    assert perm.index.equals(orig.index)
    np.testing.assert_allclose(perm.iloc[0].to_numpy(), orig.iloc[0].to_numpy())


def test_permute_is_deterministic_under_seed():
    orig = _bars()
    seed = permutation_seed("SPY", "15m", "2026-06-01T00:00:00+00:00", 7)
    a = permute_bars(orig, np.random.default_rng(seed))
    b = permute_bars(orig, np.random.default_rng(seed))
    pd.testing.assert_frame_equal(a, b)
    c = permute_bars(orig, np.random.default_rng(
        permutation_seed("SPY", "15m", "2026-06-01T00:00:00+00:00", 8)))
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_permutation_seed_is_stable():
    s = permutation_seed("SPY", "1h", "epoch", 0)
    assert s == permutation_seed("SPY", "1h", "epoch", 0)
    assert s != permutation_seed("QQQ", "1h", "epoch", 0)
    assert s != permutation_seed("SPY", "15m", "epoch", 0)
    assert 0 <= s < 2 ** 64


def test_permute_tiny_series_returns_copy():
    tiny = _bars(n=2)
    perm = permute_bars(tiny, np.random.default_rng(0))
    # validate_bars renames the index to "ts" and may drop freq; values are
    # what matters here
    pd.testing.assert_frame_equal(perm, tiny[["open", "high", "low", "close", "volume"]],
                                  check_names=False, check_freq=False)


def test_permute_rejects_invalid_bars():
    bad = _bars().drop(columns=["volume"])
    with pytest.raises(ValueError):
        permute_bars(bad, np.random.default_rng(0))
