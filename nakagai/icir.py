"""Rank-IC / IR: the ICIR lens on rule-spec signals.

Informational evaluation only: Spearman correlation between a spec's graded
margin (strategies/rules/margins.py) and forward close-to-close returns at
1/5/20 bars of the spec's own timeframe, one IC per walk-forward test window.
IR = mean/std of window ICs, aggregated by icir_fields at report time.
Forward returns are realized returns and may extend past window.test_end;
only the factor (the margin) is point-in-time. Nothing here feeds the signal
path. Overlapping k-bar forward returns across rows are not independent
observations, so n_obs inflates relative to the effective sample size.

Import discipline: the engine runner imports this module, so it must never
import nakagai.engine.runner. nakagai.stats used to be banned here for the
same reason, since it pulled in the runner itself; it is a leaf on pandas
now, so importing it would no longer close a cycle.
"""

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.context import closed_before
from nakagai.engine.windows import Window
from nakagai.strategies.rules.frame_eval import FrameEval
from nakagai.strategies.rules.margins import spec_margin
from nakagai.strategies.rules.vocabulary import Vocabulary

IC_HORIZONS = (1, 5, 20)
MIN_IC_OBS = 10


def empty_ic_fields() -> dict:
    """The None/0 shape of the six run-row IC fields."""
    out: dict = {f"ic_{k}": None for k in IC_HORIZONS}
    out.update({f"ic_n_{k}": 0 for k in IC_HORIZONS})
    return out


def rank_ic(factor: pd.Series, fwd_returns: pd.Series) -> tuple[float | None, int]:
    """Spearman correlation on the rows where both sides exist. Returns
    (ic, n_obs); ic is None below MIN_IC_OBS pairs or without variance."""
    pair = pd.concat([factor, fwd_returns], axis=1, keys=["f", "r"]).dropna()
    n = int(len(pair))
    if n < MIN_IC_OBS or pair["f"].nunique() < 2 or pair["r"].nunique() < 2:
        return None, n
    f_ranks = pair["f"].rank()
    r_ranks = pair["r"].rank()
    ic = f_ranks.corr(r_ranks)
    return (None if pd.isna(ic) else round(float(ic), 4)), n


def window_icir(spec: dict, cache, symbol: str, window: Window,
                tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
                vocabulary: Vocabulary | None = None) -> dict:
    """Per-window rank-IC of one spec on one symbol: the six run-row fields.
    The factor (margin) is built from frames cut at window.test_end so it
    stays point-in-time. Forward returns, by contrast, are realized returns
    and are computed from the uncut cache, so they may extend past test_end;
    a short window on a daily spec can still have ic_20 as long as the cache
    itself has 20 more bars, even if they fall outside this window."""
    out = empty_ic_fields()
    tf = spec.get("timeframe", "1h")
    frames = {t: closed_before(cache.load(symbol, t), t, window.test_end, tfs)
              for t in tfs.all}
    bars = frames.get(tf)
    if bars is None or bars.empty:
        return out
    fe = FrameEval(frames, tfs, vocabulary)
    in_win = bars.index[(bars.index >= window.test_start)
                        & (bars.index < window.test_end)]
    if not len(in_win):
        return out
    # The end-anchored primitives are row-wise, but only over a declared span:
    # tell FrameEval the window it must cover so their per-row values are the
    # ones the window's own rows would have seen.
    fe.set_span(tf, int(bars.index.searchsorted(in_win[0], side="left")),
                int(bars.index.searchsorted(in_win[-1], side="right")))
    margin = spec_margin(spec, fe, in_win)
    if margin.empty:
        return out
    full_close = cache.load(symbol, tf)["close"]
    for k in IC_HORIZONS:
        fwd = (full_close.shift(-k) / full_close - 1.0).loc[in_win]
        ic, n = rank_ic(margin, fwd)
        out[f"ic_{k}"] = ic
        out[f"ic_n_{k}"] = n
    return out


def icir_fields(runs: pd.DataFrame) -> dict:
    """Aggregate one (play, symbol) slice of run rows into the ICIR block:
    {"h1": {...}, "h5": {...}, "h20": {...}}. Tolerates rows written before
    the IC columns existed (fields None, counts 0)."""
    out: dict = {}
    for k in IC_HORIZONS:
        ics = (pd.to_numeric(runs[f"ic_{k}"], errors="coerce").dropna()
               if f"ic_{k}" in runs else pd.Series(dtype=float))
        n_obs = (int(pd.to_numeric(runs[f"ic_n_{k}"], errors="coerce")
                     .fillna(0).sum()) if f"ic_n_{k}" in runs else 0)
        h = {"ic_mean": None, "ic_std": None, "ic_ir": None,
             "ic_positive_rate": None, "n_windows": int(len(ics)),
             "n_obs": n_obs}
        if len(ics):
            h["ic_mean"] = round(float(ics.mean()), 4)
            h["ic_positive_rate"] = round(float((ics > 0).mean()), 3)
        if len(ics) >= 2:
            std = float(ics.std(ddof=1))
            if std > 1e-12:
                h["ic_std"] = round(std, 4)
                h["ic_ir"] = round(float(ics.mean()) / std, 3)
        out[f"h{k}"] = h
    return out
