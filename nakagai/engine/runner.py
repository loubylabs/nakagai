"""Grid runner: strategy x symbol x window jobs across processes -> results/runs.parquet."""

import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from nakagai.data.cache import BarCache, MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.engine import Engine
from nakagai.filelock import append_parquet
from nakagai.engine.metrics import buy_and_hold_return, summarize
from nakagai.engine.windows import Window
from nakagai.icir import empty_ic_fields, window_icir
from nakagai.strategies.base import Strategy
from nakagai.strategies.rules.strategy import RuleStrategy

# A zero-arg callable returning {name: Strategy class}. Passed as a callable,
# not a dict: catalog classes are created at runtime and cannot pickle by
# reference, so worker processes rebuild the mapping by importing the
# callable's module and calling it there.
Registry = Callable[[], Mapping[str, type[Strategy]]]


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
            registry: Registry | None = None, icir: bool = True) -> dict:
    # cache_root is a path string, or an already-loaded BarCache-shaped object.
    # Both pickle, so either crosses into a pool worker. run_grid and the
    # permutation harness hand over MemoryBars to skip repeated parquet reads;
    # single-run callers still pass a path.
    cache = cache_root if hasattr(cache_root, "load") else BarCache(Path(cache_root))
    if registry is None:
        raise ValueError(
            "run_one requires a strategies registry: pass a zero-arg callable "
            "returning {name: Strategy class}")
    strategy = registry()[strategy_name](params)
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
    # Rule specs only; permutation replays pass icir=False (an IC of shuffled
    # bars is meaningless and the nulls run thousands of times).
    ic_fields = empty_ic_fields()
    if icir and isinstance(strategy, RuleStrategy) and strategy.spec:
        try:
            ic_fields = window_icir(strategy.spec, cache, symbol, window, tfs=tfs)
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
             registry: Registry | None = None) -> pd.DataFrame:
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
    jobs = [(caches[sym], s, overrides.get(s, {}), sym, w, equity0, risk_pct, config, batch_id, tfs, registry)
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
