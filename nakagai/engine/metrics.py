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
        # over many windows instead of averaging per-window ratios.
        #
        # The division happens in exactly ONE place, the platform's
        # web/lib/metrics.ts (profitFactor). Its Python side sums the gross
        # columns and derives nothing, deliberately: a never-lost aggregate has
        # an infinite PF, inf is not valid strict JSON, and an agent reads the
        # null a server would have to send as "insufficient data" rather than
        # as "never lost". Only the browser can render that honestly.
        #
        # This comment named four sites (GET /api/backtests, api/baselines.py,
        # scan/evidence.py, web strategies/aggregate.ts) until 2026-08-02.
        # Three of them no longer exist; the platform corrected the same stale
        # list in its own tree in PR #210 and could not reach this copy across
        # the repo split. Do not re-derive the site list from an older doc.
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
    }


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    return float(((peak - curve) / peak).max())


def _ulcer_index(curve: pd.Series) -> float:
    """Root-mean-square drawdown over the whole curve.

    Max drawdown reports one moment; this reports how much of the window was
    spent underwater and how deep. Two curves with the same max drawdown, one
    that recovered in a day and one that sat at the bottom for a month, are
    indistinguishable to max_drawdown and far apart here.

    Well-defined at 21 points, unlike anything that annualizes a volatility
    estimate, which is why this is the metric that actually earns its place
    under the house protocol."""
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    dd = (peak - curve) / peak
    return float(np.sqrt((dd ** 2).mean()))


def _daily_returns(curve: pd.Series) -> pd.Series:
    if curve.empty:
        return pd.Series(dtype="float64")
    return curve.resample("1D").last().dropna().pct_change().dropna()


def _daily_stats(daily: pd.Series) -> dict:
    """Sufficient statistics for recomputing Sharpe and Sortino over POOLED
    windows rather than averaging per-window ratios.

    This is the answer to the small-sample problem, decided 2026-08-02. The
    house protocol's test window yields ~20 returns, which is why every
    per-window sharpe is None; thirteen such windows pooled yield ~260, which
    is a statistic. Ratios cannot be averaged back into that, but these four
    sums add, exactly as gross_profit and gross_loss already do for profit
    factor (see _trade_stats).

    Same division-of-labour as the gross sums above: this side emits the sums
    and derives nothing, so where the pooled ratio is computed stays one
    decision made in one place rather than drifting per consumer.

    An aggregator recovers the pooled figures as:
        mean          = daily_sum / daily_n
        variance      = daily_sum_sq / daily_n - mean**2
        downside_dev  = sqrt(daily_sum_sq_down / daily_n)
        sharpe        = mean / sqrt(variance)   * sqrt(252)
        sortino       = mean / downside_dev     * sqrt(252)

    daily_sum_sq_down uses a minimum acceptable return of zero and divides by
    the FULL count, not the count of losing days. That is the standard Sortino
    convention, and it is what makes the field poolable: a denominator that
    varied with the window would not add."""
    down = daily[daily < 0]
    return {
        "daily_n": int(len(daily)),
        "daily_sum": float(daily.sum()),
        "daily_sum_sq": float((daily ** 2).sum()),
        "daily_sum_sq_down": float((down ** 2).sum()),
    }


def _sortino(daily: pd.Series) -> float | None:
    """Annualized Sortino, or None on the same terms as _sharpe.

    Same threshold, and deliberately so: Sortino replaces the standard
    deviation with a downside deviation estimated from FEWER points, so if 20
    returns cannot support a Sharpe they certainly cannot support this. The
    non-None figure under the house protocol comes from pooling, not from
    relaxing the floor here.

    A downside deviation of exactly zero returns None rather than inf. inf is
    not a measurement, and it sorts to the top of any ranking on this field."""
    if len(daily) < MIN_SHARPE_OBSERVATIONS:
        return None
    downside = daily[daily < 0]
    dd = math.sqrt(float((downside ** 2).sum()) / len(daily))
    if dd == 0:
        return None
    return float(daily.mean() / dd * np.sqrt(252))


def _cagr(curve: pd.Series, starting_equity: float) -> float:
    """Compound annual growth rate over the span the curve actually covers.

    Annualizes whatever window it is given, so a one-month window is being
    extrapolated twelvefold and inherits that window's luck. It is reported
    because Calmar needs a return in the numerator and because the extrapolation
    is at least transparent, not because a one-month CAGR is a forecast."""
    if len(curve) < 2 or starting_equity <= 0:
        return 0.0
    years = (curve.index[-1] - curve.index[0]).total_seconds() / (365.25 * 86_400)
    if years <= 0:
        return 0.0
    total = float(curve.iloc[-1]) / starting_equity
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1)


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


def _exposure_and_holding(trades: list[Trade], curve: pd.Series) -> dict:
    """Share of the window holding a position, and the mean time in one.

    Both read off the trades rather than the bar loop, so neither costs
    anything at replay time. Exposure sums holding time and divides by the span
    the curve covers: with a concurrent position cap of one it cannot exceed
    1.0, and a portfolio replay (P1) that lifts that cap will legitimately push
    it past 1.0 rather than being clamped into a lie."""
    if not trades:
        return {"exposure_pct": 0.0, "avg_holding_hours": 0.0}
    held = sum((t.exit_ts - t.entry_ts).total_seconds() for t in trades)
    span = ((curve.index[-1] - curve.index[0]).total_seconds()
            if len(curve) > 1 else 0.0)
    return {
        "exposure_pct": float(held / span) if span > 0 else 0.0,
        "avg_holding_hours": float(held / len(trades) / 3_600),
    }


def summarize(result: BacktestResult, bh_return: float) -> dict:
    curve = result.equity_curve
    daily = _daily_returns(curve)
    max_dd = _max_drawdown(curve)
    cagr = _cagr(curve, result.starting_equity)
    out = {
        **_trade_stats(result.trades),
        "max_drawdown": max_dd,
        "sharpe": _sharpe(curve),
        "sortino": _sortino(daily),
        "ulcer_index": _ulcer_index(curve),
        "cagr": cagr,
        # Calmar divides an annualized return by the worst peak-to-trough. A
        # window that never drew down has no denominator, and 0.0 would read as
        # "no edge" for what is actually the best possible drawdown, so it is
        # None on the same None-versus-0.0 rule every other field here follows.
        "calmar": float(cagr / max_dd) if max_dd > 0 else None,
        **_daily_stats(daily),
        **_exposure_and_holding(result.trades, curve),
        "total_return": float(curve.iloc[-1] / result.starting_equity - 1) if len(curve) else 0.0,
        "bh_return": float(bh_return),
        "rejected_unsettled": result.rejected_unsettled,
    }
    # Only the trade stats split by direction. The curve-derived metrics above
    # are properties of ONE equity curve, and a per-direction copy would mean
    # synthesizing a long-only and a short-only curve, which is a modelling
    # decision and not a reporting one. Decided 2026-08-02; doing it by reflex
    # would have doubled the schema for numbers nobody could interpret.
    for direction, prefix in ((Direction.LONG, "long_"), (Direction.SHORT, "short_")):
        stats = _trade_stats([t for t in result.trades if t.direction == direction])
        out.update({f"{prefix}{k}": v for k, v in stats.items()})
    return out
