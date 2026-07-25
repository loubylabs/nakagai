import math

import pandas as pd
import pytest

from nakagai.engine.engine import BacktestResult, Trade
from nakagai.engine.metrics import MIN_SHARPE_OBSERVATIONS, buy_and_hold_return, summarize
from nakagai.strategies.base import Direction


def trade(direction, pnl, r):
    ts = pd.Timestamp("2026-06-01 14:00", tz="UTC")
    return Trade("SPY", direction, 10, ts, 100.0, ts, 100.0 + pnl / 10, 99.0, 105.0, pnl, r, ("t",), "target")


def result(trades, curve=None):
    if curve is None:
        idx = pd.date_range("2026-06-01", periods=5, freq="1D", tz="UTC")
        curve = pd.Series([10_000, 10_100, 9_900, 10_200, 10_300], index=idx, dtype="float64")
    return BacktestResult(trades, curve, rejected_unsettled=2, starting_equity=10_000.0)


def test_summarize_basic():
    trades = [trade(Direction.LONG, 100, 1.0), trade(Direction.LONG, -50, -0.5), trade(Direction.SHORT, 200, 2.0)]
    m = summarize(result(trades), bh_return=0.05)
    assert m["n_trades"] == 3
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(300 / 50)
    assert m["gross_profit"] == pytest.approx(300.0)
    assert m["gross_loss"] == pytest.approx(50.0)
    assert m["expectancy_r"] == pytest.approx((1.0 - 0.5 + 2.0) / 3)
    assert m["total_return"] == pytest.approx(0.03)
    assert m["bh_return"] == 0.05
    assert m["rejected_unsettled"] == 2
    assert m["long_n_trades"] == 2 and m["short_n_trades"] == 1
    assert m["short_win_rate"] == 1.0


def test_max_drawdown():
    m = summarize(result([]), bh_return=0.0)
    assert m["max_drawdown"] == pytest.approx((10_100 - 9_900) / 10_100)


def test_empty_trades_safe():
    m = summarize(result([]), bh_return=0.0)
    assert m["n_trades"] == 0 and m["win_rate"] == 0.0 and m["profit_factor"] == 0.0
    assert m["gross_profit"] == 0.0 and m["gross_loss"] == 0.0


def test_summarize_key_set():
    trades = [trade(Direction.LONG, 100, 1.0), trade(Direction.SHORT, -50, -0.5)]
    m = summarize(result(trades), bh_return=0.0)
    expected = {
        "n_trades", "win_rate", "profit_factor", "expectancy_r",
        "gross_profit", "gross_loss",
        "max_drawdown", "sharpe", "total_return", "bh_return", "rejected_unsettled",
    } | {f"{p}{k}" for p in ("long_", "short_")
         for k in ("n_trades", "win_rate", "profit_factor", "expectancy_r",
                   "gross_profit", "gross_loss")}
    assert set(m.keys()) == expected


def test_profit_factor_inf_when_no_losses():
    trades = [trade(Direction.LONG, 100, 1.0), trade(Direction.LONG, 50, 0.5)]
    m = summarize(result(trades), bh_return=0.0)
    assert m["profit_factor"] == math.inf
    assert m["win_rate"] == 1.0
    assert m["gross_profit"] == pytest.approx(150.0)
    assert m["gross_loss"] == 0.0


def test_buy_and_hold(make_bars):
    df = make_bars(10, "1d", start="2026-06-01")
    ret = buy_and_hold_return(df, df.index[0], df.index[-1] + pd.Timedelta(days=1))
    assert ret == pytest.approx(df["close"].iloc[-1] / df["open"].iloc[0] - 1)


# ---------------------------------------------------------------------------
# Sharpe suppression (Pillar 5 / joint J6). A one-month test window resampled
# to daily yields about 20 returns, and _sharpe annualized that by sqrt(252)
# and reported it as a number.

def _curve(n_days: int, step: float = 3.0) -> pd.Series:
    return pd.Series(
        [10_000.0 + i * step for i in range(n_days)],
        index=pd.date_range("2025-03-03", periods=n_days, freq="1D", tz="UTC"))


def _summary_for(curve: pd.Series) -> dict:
    res = BacktestResult(trades=[], equity_curve=curve, rejected_unsettled=0,
                         starting_equity=10_000.0)
    return summarize(res, bh_return=0.0)


def test_sharpe_is_none_for_a_one_month_window():
    """The house protocol's actual window size. This is the case that made the
    old behavior wrong, and after this change every per-window sharpe under the
    standard 13/4/1 protocol is None. That is the honest reading."""
    assert _summary_for(_curve(21))["sharpe"] is None


def test_sharpe_is_none_just_below_the_threshold():
    """MIN_SHARPE_OBSERVATIONS counts daily *returns*, which is one fewer than
    the number of daily points."""
    assert _summary_for(_curve(MIN_SHARPE_OBSERVATIONS))["sharpe"] is None


def test_sharpe_is_reported_once_the_sample_is_large_enough():
    """Windows long enough to support the statistic still get one, which is why
    the field survives at all rather than being deleted."""
    value = _summary_for(_curve(MIN_SHARPE_OBSERVATIONS + 2))["sharpe"]
    assert value is not None and value > 0


def test_sharpe_is_none_not_zero_when_uncomputable():
    """None and 0.0 say different things and the old code conflated them: 0.0
    is "no risk-adjusted edge", None is "do not read a number here"."""
    assert _summary_for(pd.Series(dtype="float64"))["sharpe"] is None
    flat = pd.Series([10_000.0] * (MIN_SHARPE_OBSERVATIONS + 2),
                     index=pd.date_range("2025-03-03",
                                         periods=MIN_SHARPE_OBSERVATIONS + 2,
                                         freq="1D", tz="UTC"))
    assert _summary_for(flat)["sharpe"] is None   # zero variance
