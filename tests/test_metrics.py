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
        # P0. Note what is NOT here: no long_/short_ copies of any of these.
        # They describe one equity curve, and splitting them would mean
        # inventing a per-direction curve to describe.
        "sortino", "ulcer_index", "cagr", "calmar",
        "exposure_pct", "avg_holding_hours",
        "daily_n", "daily_sum", "daily_sum_sq", "daily_sum_sq_down",
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


# ---------------------------------------------------------------------------
# P0, metric breadth. The house protocol yielded exactly one risk metric: every
# per-window sharpe is None by design, leaving max_drawdown alone to judge a
# portfolio replay against.

def _wobbly(n_days: int) -> pd.Series:
    """A curve that goes up and down, which _curve deliberately does not.

    _curve is monotonic, and a monotonic curve is degenerate for every metric
    that needs a drawdown or a losing day: it has no Calmar denominator and no
    downside deviation. Real equity curves are not monotonic, so the metrics
    that only exist under the house protocol get tested against one that is
    not."""
    values = [10_000.0]
    for i in range(1, n_days):
        values.append(values[-1] * (1.01 if i % 3 else 0.98))
    return pd.Series(values,
                     index=pd.date_range("2025-03-03", periods=n_days,
                                         freq="1D", tz="UTC"))


def test_the_protocol_window_now_carries_a_risk_adjusted_number():
    """The whole point of P0. Twenty-one points is what the standard 13/4/1
    protocol actually produces, and before this it yielded max_drawdown and
    nothing else that accounted for risk."""
    m = _summary_for(_wobbly(21))
    assert m["sharpe"] is None, "still refused, and that has not changed"
    assert m["ulcer_index"] > 0
    assert m["calmar"] is not None


def test_ulcer_index_is_zero_for_a_curve_that_never_draws_down():
    """0.0 here is a measurement, not a refusal: this curve genuinely had no
    drawdown. The None/0.0 distinction has to survive every metric added."""
    assert _summary_for(_curve(21))["ulcer_index"] == pytest.approx(0.0)


def test_ulcer_index_punishes_a_deep_drawdown_more_than_a_shallow_one():
    idx = pd.date_range("2025-03-03", periods=5, freq="1D", tz="UTC")
    shallow = pd.Series([10_000.0, 9_900, 9_900, 9_900, 10_000], index=idx)
    deep = pd.Series([10_000.0, 5_000, 5_000, 5_000, 10_000], index=idx)
    assert (_summary_for(deep)["ulcer_index"]
            > _summary_for(shallow)["ulcer_index"] > 0)


def test_sortino_holds_the_same_line_sharpe_does():
    """Sortino has the same small-sample problem, so it takes the same floor.
    Lowering the threshold to make a number appear is the failure mode
    MIN_SHARPE_OBSERVATIONS exists to prevent."""
    assert _summary_for(_wobbly(21))["sortino"] is None
    assert _summary_for(_wobbly(MIN_SHARPE_OBSERVATIONS + 2))["sortino"] is not None


def test_a_curve_that_only_rises_has_no_downside_deviation():
    """A monotonically rising curve divides by a downside deviation of exactly
    0.0. None rather than inf: there is no risk-adjusted number to read off a
    sample with no downside at all, and inf would sort straight to the top of
    every ranking that orders on this field."""
    rising = _curve(MIN_SHARPE_OBSERVATIONS + 2, step=3.0)
    assert _summary_for(rising)["daily_sum_sq_down"] == 0.0
    assert _summary_for(rising)["sortino"] is None

    assert _summary_for(_wobbly(MIN_SHARPE_OBSERVATIONS + 2))["sortino"] is not None


# -- the pooled headline ----------------------------------------------------
# Decided 2026-08-02: pool across the 13 windows rather than lengthen the test
# window or average per-window ratios. summarize carries the sufficient
# statistics so the platform can recompute over ~260 returns, exactly as
# gross_profit/gross_loss already let it recompute profit factor.

def test_the_daily_sufficient_statistics_describe_the_returns():
    m = _summary_for(_curve(21))
    curve = _curve(21)
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    assert m["daily_n"] == len(daily)
    assert m["daily_sum"] == pytest.approx(float(daily.sum()))
    assert m["daily_sum_sq"] == pytest.approx(float((daily ** 2).sum()))


def test_the_downside_statistic_counts_only_losing_days():
    idx = pd.date_range("2025-03-03", periods=4, freq="1D", tz="UTC")
    curve = pd.Series([10_000.0, 11_000, 9_900, 10_890], index=idx)
    daily = curve.pct_change().dropna()
    down = daily[daily < 0]
    m = _summary_for(curve)
    assert m["daily_sum_sq_down"] == pytest.approx(float((down ** 2).sum()))
    assert m["daily_sum_sq_down"] < m["daily_sum_sq"], "gains must not count"


def test_pooling_the_statistics_reproduces_the_single_window_sortino():
    """The contract the platform aggregation depends on: summing these fields
    over N windows and recomputing must give the same answer as computing over
    the concatenated returns. Proven here on one long window, where summarize
    reports sortino directly, so the pooled arithmetic has something to agree
    with."""
    import numpy as np

    curve = _wobbly(MIN_SHARPE_OBSERVATIONS + 2)
    m = _summary_for(curve)
    mean = m["daily_sum"] / m["daily_n"]
    downside_dev = math.sqrt(m["daily_sum_sq_down"] / m["daily_n"])
    pooled = mean / downside_dev * np.sqrt(252)
    assert pooled == pytest.approx(m["sortino"])


# -- the deterministic descriptives ------------------------------------------

def test_exposure_is_the_share_of_the_window_holding_a_position():
    idx = pd.date_range("2026-06-01", periods=5, freq="1D", tz="UTC")
    curve = pd.Series([10_000.0, 10_100, 9_900, 10_200, 10_300], index=idx)
    # One trade held for exactly one of the four days the curve spans.
    t = trade(Direction.LONG, 100, 1.0)
    t = t.__class__(**{**t.__dict__,
                       "entry_ts": idx[0], "exit_ts": idx[1]})
    res = BacktestResult([t], curve, rejected_unsettled=0, starting_equity=10_000.0)
    assert summarize(res, bh_return=0.0)["exposure_pct"] == pytest.approx(0.25)


def test_average_holding_period_is_reported_in_hours():
    idx = pd.date_range("2026-06-01", periods=5, freq="1D", tz="UTC")
    curve = pd.Series([10_000.0] * 5, index=idx)
    t = trade(Direction.LONG, 100, 1.0)
    held = t.__class__(**{**t.__dict__, "entry_ts": idx[0], "exit_ts": idx[2]})
    res = BacktestResult([held], curve, rejected_unsettled=0,
                         starting_equity=10_000.0)
    assert summarize(res, bh_return=0.0)["avg_holding_hours"] == pytest.approx(48.0)


def test_a_window_with_no_trades_is_flat_not_undefined():
    m = _summary_for(_curve(21))
    assert m["exposure_pct"] == 0.0
    assert m["avg_holding_hours"] == 0.0
