"""Run-level metrics. The honesty check: always reported next to buy-and-hold."""

import math

import numpy as np
import pandas as pd

from nakagai.engine.engine import BacktestResult, Trade
from nakagai.strategies.base import Direction


def _trade_stats(trades: list[Trade]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0,
                "gross_profit": 0.0, "gross_loss": 0.0}
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))
    pf = (gross_profit / gross_loss) if gross_loss else (math.inf if wins else 0.0)
    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "profit_factor": pf,
        "expectancy_r": float(np.mean([t.r_multiple for t in trades])),
        # Gross sums travel with the ratio so aggregations can recompute PF
        # over many windows instead of averaging per-window ratios. Four
        # sites do that today: GET /api/backtests, api/baselines.py,
        # scan/evidence.py, and web strategies/aggregate.ts.
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    return float(((peak - curve) / peak).max())


# Below this many daily returns, an annualized Sharpe is not a statistic, it is
# a rounding of noise. The house protocol's test window is ONE MONTH, which
# yields about 21 daily points and 20 returns, and _sharpe multiplied that by
# sqrt(252) and reported the result as a number. Sixty returns is roughly a
# quarter: still small, but far enough from the cliff that the figure carries
# some information.
#
# The consequence is deliberate: under the standard 13/4/1 protocol every
# per-window `sharpe` is now None. That is the honest reading, not a
# regression. Windows long enough to support the statistic (custom Builder
# ranges, say) still get one, which is why the field survives at all.
MIN_SHARPE_OBSERVATIONS = 60


def _sharpe(curve: pd.Series) -> float | None:
    """Annualized Sharpe, or None when the sample is too small to mean anything.

    None rather than 0.0 because they say different things and the old code
    conflated them: 0.0 is "this strategy had no risk-adjusted edge", None is
    "do not read a number here". Every consumer already treats null as
    insufficient data (see the platform's scan/evidence.py), so None travels
    correctly through parquet, Postgres and signal JSON alike.
    """
    if curve.empty:
        return None
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    if len(daily) < MIN_SHARPE_OBSERVATIONS or daily.std() == 0:
        return None
    return float(daily.mean() / daily.std() * np.sqrt(252))


def buy_and_hold_return(bars: pd.DataFrame, start, end) -> float:
    w = bars[(bars.index >= start) & (bars.index < end)]
    if w.empty:
        return 0.0
    return float(w["close"].iloc[-1] / w["open"].iloc[0] - 1)


def summarize(result: BacktestResult, bh_return: float) -> dict:
    curve = result.equity_curve
    out = {
        **_trade_stats(result.trades),
        "max_drawdown": _max_drawdown(curve),
        "sharpe": _sharpe(curve),
        "total_return": float(curve.iloc[-1] / result.starting_equity - 1) if len(curve) else 0.0,
        "bh_return": float(bh_return),
        "rejected_unsettled": result.rejected_unsettled,
    }
    for direction, prefix in ((Direction.LONG, "long_"), (Direction.SHORT, "short_")):
        stats = _trade_stats([t for t in result.trades if t.direction == direction])
        out.update({f"{prefix}{k}": v for k, v in stats.items()})
    return out
