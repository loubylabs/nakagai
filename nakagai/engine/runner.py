"""Grid runner: strategy x symbol x window jobs across processes -> results/runs.parquet."""

import json
import math
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from nakagai.data.cache import BarCache, MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.engine import BacktestResult, Engine, Trade
from nakagai.filelock import append_parquet
from nakagai.engine.windows import Window
from nakagai.icir import empty_ic_fields, window_icir
from nakagai.strategies.base import Direction, Strategy
from nakagai.strategies.rules.strategy import RuleStrategy
from nakagai.strategies.rules.vocabulary import VocabularyFactory

# A zero-arg callable returning {name: Strategy class}. Passed as a callable,
# not a dict: catalog classes are created at runtime and cannot pickle by
# reference, so worker processes rebuild the mapping by importing the
# callable's module and calling it there.
Registry = Callable[[], Mapping[str, type[Strategy]]]


# ---------------------------------------------------------------------------
# The retired singleton engine's run summary, moved here unchanged from
# engine/metrics.py, which now holds the portfolio replay's arithmetic-v2
# metrics instead. This is the only production caller, and the two could not
# share a module: they define the same statistics under the same names against
# different inputs, so one would have silently shadowed the other. Task C11
# deletes this runner, `summarize`, and `buy_and_hold_return` together with the
# engine they belong to.
# ---------------------------------------------------------------------------


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
    """Sufficient statistics for recomputing Sharpe, Sortino, and the
    deflated-Sharpe family (skew, kurtosis, PSR, DSR) over POOLED windows
    rather than averaging per-window ratios.

    This is the answer to the small-sample problem, decided 2026-08-02. The
    house protocol's test window yields ~20 returns, which is why every
    per-window sharpe is None; thirteen such windows pooled yield ~260, which
    is a statistic. Ratios cannot be averaged back into that, but these six
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
        skew, kurtosis = nakagai.stats.pooled_moments(...)   # bias-corrected

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
        # Third and fourth raw power sums, on the same poolable footing as the
        # three above and for the same reason. The deflated Sharpe ratio needs
        # the skew and the kurtosis of the return distribution, and neither is
        # recoverable from the first two moments. nakagai.stats.pooled_moments
        # is the one place they are derived back out.
        "daily_sum_cube": float((daily ** 3).sum()),
        "daily_sum_fourth": float((daily ** 4).sum()),
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


def _sharpe(daily: pd.Series) -> float | None:
    """Annualized Sharpe, or None when the sample is too small to mean anything.

    Takes the daily returns rather than the curve, so summarize resamples once
    and hands the same series to both this and _sortino. Two functions deriving
    the same returns from the same curve is not only wasted work, it is a
    second place for the resampling rule to drift.

    None rather than 0.0 because they say different things and the old code
    conflated them: 0.0 is "this strategy had no risk-adjusted edge", None is
    "do not read a number here". Every consumer already treats null as
    insufficient data, so None travels correctly through parquet, Postgres and
    signal JSON alike.
    """
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
        "sharpe": _sharpe(daily),
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


def _trade_rows(trades, run_id: str) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "symbol": t.symbol,
            "direction": t.direction.value,
            "qty": t.qty,
            "entry_ts": t.entry_ts.isoformat(),
            "entry": t.entry,
            "exit_ts": t.exit_ts.isoformat(),
            "exit": t.exit,
            "stop": t.stop,
            "target": t.target,
            "pnl": t.pnl,
            "r_multiple": t.r_multiple,
            "setup_tags": "+".join(t.setup_tags),
            "exit_reason": t.exit_reason,
            # fees was carried on Trade so a fee change would be re-provable
            # rather than archaeological, and then dropped here, which is the
            # one place that claim had to hold. Persisted now.
            "fees": t.fees,
            # The excursion, in R. Stopping at the dataclass would make it
            # write-only: "where should the stop have been" is a question asked
            # of a catalog of stored trades, not of one in-memory replay.
            "mae": t.mae,
            "mfe": t.mfe,
        }
        for t in trades
    ]


def run_one(cache_root, strategy_name: str, params: dict, symbol: str,
            window: Window, equity0: float = 10_000.0, risk_pct: float = 0.01,
            config: str = "", batch_id: str = "",
            tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
            registry: Registry | None = None,
            vocabulary_factory: VocabularyFactory | None = None) -> dict:
    # cache_root is a path string, or an already-loaded BarCache-shaped object.
    # Both pickle, so either crosses into a pool worker. run_grid hands over
    # MemoryBars to skip repeated parquet reads; single-run callers still pass
    # a path.
    cache = cache_root if hasattr(cache_root, "load") else BarCache(Path(cache_root))
    if registry is None:
        raise ValueError(
            "run_one requires a strategies registry: pass a zero-arg callable "
            "returning {name: Strategy class}")
    strategy_cls = registry()[strategy_name]
    if issubclass(strategy_cls, RuleStrategy):
        # A vocabulary, not a bound subclass. RuleStrategy.__init__ already
        # takes one, and bound() mints a throwaway class per call, which across
        # a grid is thousands of classes that differ in nothing but a factory
        # reference. bound() belongs to load_catalog, where the class IS the
        # product and is built once. Read the factory off the CLASS: a plain
        # function stored as a class attribute is a descriptor, so instance
        # access would bind the strategy as its first argument.
        factory = vocabulary_factory or strategy_cls.VOCABULARY_FACTORY
        strategy = strategy_cls(params, vocabulary=factory())
    elif vocabulary_factory is not None:
        # Refuse rather than drop it, though not for the old reason. A
        # composite carries its own vocabulary now (CompositeStrategy.__init__
        # in nakagai/strategies/composite/strategy.py takes it from its bound
        # members), so it is no longer true that a non-rule strategy has
        # nothing to inject into. What is still true: run_one has no way to
        # reach it here. CompositeStrategy.__init__ takes no vocabulary
        # argument, and its members were already built from whatever factory
        # the registry callable bound them to (typically RuleStrategy.bound()
        # via load_catalog) when the registry was constructed, not from this
        # call's factory. Accepting the factory here and ignoring it would let
        # a caller believe their override reached the composite's members when
        # it never could, which is exactly the silent no-op this seam exists
        # to prevent. A non-rule, non-composite strategy has no vocabulary
        # concept at all, so the same refusal covers it for the simpler
        # reason. Either way: rebuild the registry with members already bound
        # to the vocabulary you need, or split the grid.
        raise ValueError(
            f"vocabulary_factory was passed for strategy {strategy_name!r}, "
            "which is not a RuleStrategy; run_one has no per-call way to "
            "inject a vocabulary into it, rebuild the registry with the "
            "vocabulary already bound, or run non-rule strategies in a "
            "separate call")
    else:
        strategy = strategy_cls(params)
    # `params` are the caller's overrides on top of the spec's defaults, and
    # they are the SAME on every window: nothing is fit on window.train_start
    # .. window.train_end. This is fixed-parameter rolling out-of-sample
    # evaluation, not walk-forward optimization. See engine/windows.py.
    engine = Engine(strategy, cache, symbol, window.test_start, window.test_end,
                    equity0=equity0, risk_pct=risk_pct, tfs=tfs)
    result = engine.run()
    bh = buy_and_hold_return(cache.load(symbol, tfs.driving), window.test_start, window.test_end)
    run_id = uuid.uuid4().hex
    # ICIR lens: per-window rank-IC of the spec's margin vs forward returns.
    # Rule specs only, and it abstains to empty fields for anything else.
    ic_fields = empty_ic_fields()
    if isinstance(strategy, RuleStrategy) and strategy.spec:
        try:
            ic_fields = window_icir(strategy.spec, cache, symbol, window, tfs=tfs,
                                    vocabulary=strategy.vocabulary)
        except Exception:
            # The lens is informational; it must never kill a production row.
            ic_fields = empty_ic_fields()
    return {
        "run_id": run_id,
        "ts_run": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy": strategy_name,
        "symbol": symbol,
        "params_json": json.dumps(strategy.params, sort_keys=True),
        # Which saved strategy config launched this row, and which launch (API
        # job id). Empty for drafts, templates, and the nightly CLI; the
        # Backtests page shows labeled rows only.
        "config": config,
        "batch_id": batch_id,
        "arithmetic_version": engine.arithmetic_version,
        "fill_mode": engine.fill_mode,
        "equity0": float(equity0),
        "risk_pct": float(risk_pct),
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "test_start": window.test_start.isoformat(),
        "test_end": window.test_end.isoformat(),
        **summarize(result, bh),
        **ic_fields,
        "trades": _trade_rows(result.trades, run_id),
    }


def run_grid(cache_root: str, strategy_names: list[str], symbols: list[str], windows: list[Window],
             workers: int = 2, equity0: float = 10_000.0, risk_pct: float = 0.01,
             out: str = "results/runs.parquet",
             trades_out: str | None = None, params_by_strategy: dict[str, dict] | None = None,
             on_progress: Callable[[int, int], None] | None = None,
             config: str = "", batch_id: str = "",
             tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
             registry: Registry | None = None,
             vocabulary_factory: VocabularyFactory | None = None) -> pd.DataFrame:
    if registry is None:
        raise ValueError(
            "run_grid requires a strategies registry: pass a zero-arg callable "
            "returning {name: Strategy class}")
    overrides = params_by_strategy or {}
    # One parquet read per (symbol, timeframe) for the whole grid. BarCache.load
    # re-reads the file and recomputes pd.infer_freq on every call, and Engine.run
    # calls it per timeframe per window, so the naive grid re-reads identical bars
    # thousands of times. MemoryBars is BarCache-shaped, returns exactly the frames
    # BarCache.load returned, and pickles, so it crosses into pool workers too.
    disk = BarCache(Path(cache_root))
    caches = {sym: MemoryBars({(sym, tf): disk.load(sym, tf) for tf in tfs.all})
              for sym in symbols}
    jobs = [(caches[sym], s, overrides.get(s, {}), sym, w, equity0, risk_pct,
             config, batch_id, tfs, registry, vocabulary_factory)
            for s in strategy_names for sym in symbols for w in windows]

    def _tick(done: int) -> None:
        if on_progress is not None:
            on_progress(done, len(jobs))

    if workers <= 1:
        rows = []
        for i, j in enumerate(jobs, 1):
            rows.append(run_one(*j))
            _tick(i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = []
            # pool.map yields in submission order as results land, close
            # enough for a progress bar without giving up result ordering.
            for i, r in enumerate(pool.map(run_one, *zip(*jobs)), 1):
                rows.append(r)
                _tick(i)
    trade_rows = [tr for r in rows for tr in r.pop("trades")]
    new = pd.DataFrame(rows)
    # Locked: API job threads, the MCP run_backtest subprocess, and the nightly
    # CLI all append here concurrently.
    append_parquet(Path(out), new)
    if trade_rows:
        trades_path = Path(trades_out) if trades_out else Path(out).with_name("trades.parquet")
        append_parquet(trades_path, pd.DataFrame(trade_rows))
    return new
