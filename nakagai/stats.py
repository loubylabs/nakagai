"""Pure permutation-test math for backtest results.

No evidence store, no workspace, no config: everything is parameterized.
The bar-permutation primitive is nakagai/engine/permutation.py. The study
subsystem's orchestration (which trials to score, where results live) is
nakagai/lab/.
"""

import pandas as pd

PF_CLAMP = 1000.0   # a ledger or resample with no losers: "infinite" PF


def pf_from_trades(trades: pd.DataFrame | None) -> float | None:
    """Pooled profit factor over a trade ledger; None when it has no trades."""
    if trades is None or len(trades) == 0:
        return None
    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl <= 0].sum()))
    if losses == 0:
        return PF_CLAMP if wins > 0 else 0.0
    return wins / losses


def permutation_pvalue(observed: float | None, nulls: list[float]) -> float | None:
    """Bias-corrected permutation estimate: (1 + #{null >= obs}) / (N + 1)."""
    if observed is None or not nulls:
        return None
    ge = sum(1 for x in nulls if x >= observed)
    return (1 + ge) / (len(nulls) + 1)
