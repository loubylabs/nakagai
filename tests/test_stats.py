"""Statistical core: pooled PF and permutation p-values."""

import pandas as pd

from nakagai.stats import PF_CLAMP, permutation_pvalue, pf_from_trades


def _trades(pnls):
    return pd.DataFrame({"run_id": "r", "pnl": pnls})


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
