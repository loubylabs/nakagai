"""Running a frozen trial set, and the one statistic every part of the lab
scores with.

trial_pf is deliberately the only place a number is produced. The observed
best and every null replay come out of the same function over the same
windows, because a p-value computed from two different procedures is not a
p-value at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.runner import run_one
from nakagai.engine.windows import Window
from nakagai.lab.mutate import Trial
from nakagai.stats import pf_from_trades


def trial_pf(cache, trial: Trial, symbols: Sequence[str],
             windows: Sequence[Window], registry,
             tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> tuple[float, int]:
    """Pooled profit factor for one trial across every symbol and window.

    Pooled, not averaged: one ledger built from every replay, so a trial that
    trades heavily on one symbol and never on another is scored on what it
    actually did rather than on a mean of incomparable per-symbol numbers.

    Returns 0.0 for an empty ledger. pf_from_trades returns None there, and
    None cannot be compared against an observed PF when the nulls are ranked.
    """
    rows = []
    for symbol in symbols:
        for window in windows:
            rows.append(run_one(cache, trial.strategy, {"spec": trial.spec},
                                symbol, window, tfs=tfs, registry=registry,
                                icir=False))
    trades = pd.DataFrame([t for r in rows for t in r["trades"]])
    return float(pf_from_trades(trades) or 0.0), int(len(trades))


@dataclass(frozen=True)
class StudySpec:
    """A study's complete, immutable definition.

    `trials` is a tuple because N is frozen at commission: a study that could
    grow its own trial set has defeated its own null, since the best-of-N
    distribution is computed for exactly this N.
    """
    trials: tuple[Trial, ...]
    symbols: tuple[str, ...]
    windows: tuple[Window, ...]
    seed: int


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    spec_hash: str
    pf: float
    n_trades: int


@dataclass(frozen=True)
class StudyResult:
    results: tuple[TrialResult, ...]
    best: TrialResult | None
    n_trials: int


def run_study(cache, study: StudySpec, registry,
              tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> StudyResult:
    """Score every trial in the frozen set and report the best.

    Results are returned in the trial set's own order, so a caller comparing
    two runs compares like with like without sorting first.
    """
    if not study.trials:
        raise ValueError("a study needs at least one trial")
    results = []
    for trial in study.trials:
        pf, n_trades = trial_pf(cache, trial, study.symbols, study.windows,
                                registry, tfs=tfs)
        results.append(TrialResult(trial_id=trial.id, spec_hash=trial.spec_hash,
                                   pf=pf, n_trades=n_trades))
    best = max(results, key=lambda r: r.pf) if results else None
    return StudyResult(results=tuple(results), best=best, n_trials=len(results))
