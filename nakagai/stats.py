"""Pure permutation-test and bootstrap math for backtest results.

No evidence store, no workspace, no config: everything is parameterized.
The bar-permutation primitive is nakagai/engine/permutation.py; the proving
pipeline's orchestration (which pairs to test, where results live) stays in
nakagai/robustness.py."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.permutation import permutation_seed, permute_bars
from nakagai.engine.runner import Registry, run_one

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


def bootstrap_cis(r_multiples, resamples: int = 2000, seed: int = 0) -> dict:
    """Resample a pair's trade R-multiples with replacement; 5th/95th
    percentiles of the resampled PF and the 5th of expectancy."""
    out = {"pf_ci_low": None, "pf_ci_high": None, "expectancy_ci_low": None}
    r = np.asarray(list(r_multiples), dtype=float)
    if len(r) == 0:
        return out
    rng = np.random.default_rng(seed)
    draws = r[rng.integers(0, len(r), size=(resamples, len(r)))]
    wins = np.where(draws > 0, draws, 0.0).sum(axis=1)
    losses = np.abs(np.where(draws <= 0, draws, 0.0).sum(axis=1))
    pf = np.where(losses > 0, wins / np.maximum(losses, 1e-12),
                  np.where(wins > 0, PF_CLAMP, 0.0))
    out["pf_ci_low"] = round(float(np.percentile(pf, 5)), 3)
    out["pf_ci_high"] = round(float(np.percentile(pf, 95)), 3)
    out["expectancy_ci_low"] = round(float(np.percentile(draws.mean(axis=1), 5)), 3)
    return out


def statistical_fields(trades: pd.DataFrame | None, rob: dict | None,
                       resamples: int, min_pf_ci_low: float,
                       max_p_value: float) -> dict:
    """The per-pair stats block symbol_stats carries, plus the robust gate.
    Fail closed: no permutation row or no trades means not robust."""
    r = trades["r_multiple"].tolist() if trades is not None and len(trades) else []
    fields = bootstrap_cis(r, resamples=resamples)
    p = None if rob is None else rob.get("p_value")
    fields["p_value"] = None if p is None or pd.isna(p) else float(p)
    n = rob.get("n_permutations") if rob else None
    fields["n_permutations"] = 0 if n is None or pd.isna(n) else int(n)
    fields["robust"] = bool(
        fields["pf_ci_low"] is not None and fields["pf_ci_low"] >= min_pf_ci_low
        and fields["p_value"] is not None and fields["p_value"] <= max_p_value)
    return fields


_CHUNK = 10   # permutations per pool task: amortizes pickling, keeps early-stop responsive


def null_batch(play: str, sym: str, windows, frames: dict, epoch: str,
               indices: list[int], registry: Registry,
               tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> list[float]:
    """A batch of permutations, fully in memory: permute every cached
    timeframe, replay the play over the same windows via run_one, pool each
    copy's trades into one null PF. No parquet, no per-permutation pools."""
    out = []
    for i in indices:
        null_cache = MemoryBars({
            (sym, tf): permute_bars(bars, np.random.default_rng(
                permutation_seed(sym, tf, epoch, i)))
            for tf, bars in frames.items()})
        rows = [run_one(null_cache, play, {}, sym, w, batch_id=f"perm-{i}",
                        tfs=tfs, registry=registry)
                for w in windows]
        t = pd.DataFrame([tr for r in rows for tr in r["trades"]])
        # a null run with no trades cannot beat any positive observed PF
        out.append(pf_from_trades(t) or 0.0)
    return out


def _say(msg: str) -> None:
    # progress goes to stderr so the farm's `| tee summary` only captures the
    # JSON result on stdout while the Actions log still shows live movement
    print(msg, file=sys.stderr, flush=True)


def collect_nulls(play: str, sym: str, windows, cache, epoch: str,
                  n_perm: int, p_ceiling: float, workers: int,
                  observed: float, registry: Registry,
                  tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> list[float]:
    """Run permutations until n_perm or until significance is impossible.

    Early stop is decision-exact for the p <= p_ceiling gate: once
    (1 + #{null >= observed}) exceeds p_ceiling * (n_perm + 1), no outcome of
    the remaining permutations can pull p back under the ceiling, so the
    remaining compute cannot change the verdict. The recorded n_permutations
    is the count actually run."""
    frames = {tf: cache.load(sym, tf) for tf in tfs.all}
    frames = {tf: df for tf, df in frames.items() if len(df)}
    chunks = [list(range(i, min(i + _CHUNK, n_perm)))
              for i in range(0, n_perm, _CHUNK)]
    t0 = time.monotonic()
    nulls: list[float] = []
    ge = 0

    def note(batch: list[float]) -> bool:
        nonlocal ge
        nulls.extend(batch)
        ge += sum(1 for x in batch if x >= observed)
        _say(f"[robustness] {play}/{sym}: {len(nulls)}/{n_perm} permutations, "
             f"{ge} null >= observed, {time.monotonic() - t0:.0f}s elapsed")
        return (1 + ge) > p_ceiling * (n_perm + 1) + 1e-9

    _say(f"[robustness] {play}/{sym}: observed pooled PF {observed:.3f}, "
         f"up to {n_perm} permutations x {len(windows)} windows")
    stopped = False
    if workers <= 1:
        for ch in chunks:
            if note(null_batch(play, sym, windows, frames, epoch, ch, registry, tfs=tfs)):
                stopped = True
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(null_batch, play, sym, windows, frames,
                                   epoch, ch, registry, tfs=tfs) for ch in chunks]
            for f in as_completed(futures):
                if note(f.result()):
                    stopped = True
                    for other in futures:
                        other.cancel()
                    break
    if stopped and len(nulls) < n_perm:
        _say(f"[robustness] {play}/{sym}: stopped at {len(nulls)}/{n_perm}: "
             f"p can no longer reach <= {p_ceiling}, verdict already decided")
    return nulls
