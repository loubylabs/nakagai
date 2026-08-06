"""Statistical core: pooled profit factor."""

import pandas as pd

from nakagai.stats import PF_CLAMP, pf_from_trades


def _trades(pnls):
    return pd.DataFrame({"run_id": "r", "pnl": pnls})


def test_pf_from_trades():
    assert pf_from_trades(_trades([10.0, -5.0])) == 2.0
    assert pf_from_trades(_trades([])) is None
    assert pf_from_trades(None) is None
    assert pf_from_trades(_trades([10.0, 5.0])) == PF_CLAMP   # no losers
    assert pf_from_trades(_trades([-10.0, -5.0])) == 0.0      # no winners
