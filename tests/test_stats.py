"""Statistical core: pooled PF, permutation p-values, bootstrap CIs, gates."""

import numpy as np
import pandas as pd

from nakagai import stats as st
from nakagai.data.schema import TimeframeSet
from nakagai.engine.windows import Window
from nakagai.stats import (PF_CLAMP, bootstrap_cis, permutation_pvalue,
                           pf_from_trades, statistical_fields)


def _trades(pnls, rs=None):
    rs = rs if rs is not None else [p / 10 for p in pnls]
    return pd.DataFrame({"run_id": "r", "pnl": pnls, "r_multiple": rs})


def test_pf_from_trades():
    assert pf_from_trades(_trades([10.0, -5.0])) == 2.0
    assert pf_from_trades(_trades([])) is None
    assert pf_from_trades(None) is None
    assert pf_from_trades(_trades([10.0, 5.0])) == PF_CLAMP   # no losers
    assert pf_from_trades(_trades([-10.0, -5.0])) == 0.0      # no winners


def test_permutation_pvalue():
    nulls = [0.5, 0.8, 1.0, 1.2, 2.0]
    assert permutation_pvalue(1.5, nulls) == (1 + 1) / (5 + 1)
    assert permutation_pvalue(3.0, nulls) == (1 + 0) / (5 + 1)
    assert permutation_pvalue(0.1, nulls) == (1 + 5) / (5 + 1)
    assert permutation_pvalue(None, nulls) is None
    assert permutation_pvalue(1.5, []) is None


def test_bootstrap_cis_on_a_clear_winner():
    rng = np.random.default_rng(5)
    rs = list(rng.normal(0.5, 0.3, size=200))    # strongly positive ledger
    out = bootstrap_cis(rs, resamples=500, seed=1)
    assert out["pf_ci_low"] > 1.0
    assert out["pf_ci_high"] >= out["pf_ci_low"]
    assert out["expectancy_ci_low"] > 0.0


def test_bootstrap_cis_zero_trades_is_all_none():
    assert bootstrap_cis([], resamples=100) == {
        "pf_ci_low": None, "pf_ci_high": None, "expectancy_ci_low": None}


def test_bootstrap_cis_all_winners_clamps():
    out = bootstrap_cis([1.0, 2.0, 0.5], resamples=100, seed=2)
    assert out["pf_ci_low"] == PF_CLAMP == out["pf_ci_high"]


def test_bootstrap_is_deterministic_under_seed():
    rs = [1.0, -0.5, 2.0, -1.0, 0.3]
    assert bootstrap_cis(rs, resamples=200, seed=3) == bootstrap_cis(rs, resamples=200, seed=3)


def test_statistical_fields_gates():
    strong = _trades(list(np.random.default_rng(6).normal(0.5, 0.3, size=200)))
    rob = {"p_value": 0.01, "n_permutations": 200}
    f = statistical_fields(strong, rob, 500, 1.0, 0.05)
    assert f["robust"] is True and f["p_value"] == 0.01 and f["n_permutations"] == 200

    f = statistical_fields(strong, {"p_value": 0.2, "n_permutations": 200}, 500, 1.0, 0.05)
    assert f["robust"] is False                     # p too high

    f = statistical_fields(strong, None, 500, 1.0, 0.05)
    assert f["robust"] is False and f["p_value"] is None and f["n_permutations"] == 0

    f = statistical_fields(strong, {"p_value": float("nan"), "n_permutations": 200}, 500, 1.0, 0.05)
    assert f["robust"] is False and f["p_value"] is None   # parquet NaN reads as missing

    f = statistical_fields(strong, {"p_value": 0.01, "n_permutations": float("nan")}, 500, 1.0, 0.05)
    assert f["n_permutations"] == 0

    f = statistical_fields(None, rob, 500, 1.0, 0.05)
    assert f["robust"] is False and f["pf_ci_low"] is None  # no trades, fail closed


def test_collect_nulls_threads_custom_tfs_to_run_one(monkeypatch):
    """A custom tfs axis must reach run_one unchanged. Before the fix,
    null_batch dropped tfs on the floor: frames were keyed by the custom
    timeframe but run_one replayed on DEFAULT_TIMEFRAMES, so MemoryBars'
    missing-key contract handed the engine an empty frame and every null
    PF silently went degenerate."""
    custom_tfs = TimeframeSet(driving="1h", deltas={"1h": pd.Timedelta(hours=1)})
    idx = pd.date_range("2025-06-02", periods=5, freq="1h", tz="UTC")
    c = pd.Series(np.linspace(100.0, 110.0, 5), index=idx)
    bars = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": 1000.0}, index=idx)

    class _FakeCache:
        def load(self, sym, tf):
            return bars if tf == "1h" else pd.DataFrame()

    seen = {}

    def fake_run_one(cache, strategy_name, params, symbol, window, equity0=10_000.0,
                     risk_pct=0.01, config="", batch_id="", tfs=None, registry=None):
        seen["tfs"] = tfs
        seen["has_custom_key"] = len(cache.load(symbol, "1h")) > 0
        seen["has_default_key"] = len(cache.load(symbol, "15m")) > 0
        return {"trades": [{"run_id": batch_id, "pnl": 1.0, "r_multiple": 1.0}]}

    monkeypatch.setattr(st, "run_one", fake_run_one)

    window = Window(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC"),
                    pd.Timestamp("2025-02-01", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC"))

    st.collect_nulls("play", "SYM", [window], _FakeCache(), "epoch", 1, 0.05, 1,
                     observed=100.0, registry=lambda: {}, tfs=custom_tfs)

    assert seen["tfs"] == custom_tfs
    assert seen["has_custom_key"] is True    # frames carried the custom key
    assert seen["has_default_key"] is False  # never fell back to DEFAULT_TIMEFRAMES
