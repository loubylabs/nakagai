"""Statistical math for backtest results.

No evidence store, no workspace, no config: everything is parameterized.
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
