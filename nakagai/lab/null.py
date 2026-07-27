"""The best-of-N permutation null: what the luckiest of N trials looks like
when there is nothing there to find.

nakagai/stats.py::null_batch replays ONE play per permutation and yields that
play's null PF, which is the right null for a single hypothesis. A search is
not a single hypothesis. Here every trial in the frozen set is replayed on the
SAME permuted copy of the bars and only the maximum is kept, which is exactly
the distribution of "best of N under the null".

That distinction is the entire statistical content of the lab. Comparing an
observed best-of-60 against a single-hypothesis null would clear roughly one
study in three on pure noise.
"""

import numpy as np

from nakagai.data.cache import MemoryBars
from nakagai.data.schema import DEFAULT_TIMEFRAMES, TimeframeSet
from nakagai.engine.permutation import permutation_seed, permute_bars
from nakagai.lab.study import StudySpec, trial_pf
from nakagai.stats import permutation_pvalue


def best_of_n_null(frames: dict, study: StudySpec, registry,
                   n_permutations: int, *, epoch: str = "",
                   tfs: TimeframeSet = DEFAULT_TIMEFRAMES,
                   on_progress=None, should_cancel=None) -> list[float]:
    """One maximum-across-trials PF per permutation.

    The `frames` given here and the `cache` given to `run_study` for the
    observed statistic must cover the SAME bars, i.e. `cache` should be
    `MemoryBars(frames)`. Nothing enforces this: the single-`trial_pf`
    contract ties the observed and the null to the same function, not to the
    same bars, so a `cache` spanning a different range than `frames` produces
    a p-value that compares two different histories and means nothing,
    silently.

    `permutation_seed(symbol, tf, epoch, i)` keys on the timeframe as well as
    the symbol, so within one permutation index i, `15m`, `1h`, and `1d` each
    get their OWN independent shuffle rather than one shuffle shared across
    the three. On the observed bars the three timeframes are mutually
    consistent because `1h` and `1d` are resampled up from `15m`; on a
    permuted copy they are not, since each was permuted on its own. This
    matches the shipped `nakagai/stats.py::null_batch` exactly and is
    deliberate existing convention, not something introduced here.

    Every trial in the frozen set is scored with `trial_pf`, the same
    function `run_study` uses for the observed statistic, so the null and the
    observed are always two evaluations of one procedure. Only the maximum
    across trials is kept per permutation: that maximum, not any single
    trial's null, is the distribution a best-of-N observed statistic must be
    compared against.

    When `epoch` is not given, it defaults to `f"study-{study.seed}"`, which
    depends only on the seed. Two different studies that happen to share a
    seed therefore draw the identical set of permuted alternate histories:
    each study's own p-value stays valid on its own, but a researcher
    sweeping many base specs at one fixed seed is reusing that one set of
    histories across every sweep member, which correlates the resulting
    p-values rather than drawing each independently.

    `should_cancel`, when given, is consulted before each permutation and
    `on_progress` is called as on_progress(done, total) after each one. A
    caller whose should_cancel returns True gets the permutations completed so
    far, so a list shorter than n_permutations means cancelled.

    Both live here rather than in a caller's own copy of this loop because a
    permutation count is not resumable: this function always iterates range(n)
    from zero, so a caller driving it one permutation at a time would replay
    index 0 every time and collect n copies of one number, which is not a null
    distribution at all. Neither hook can move a number, since each is
    consulted outside the statistic's own computation.
    """
    if not study.trials:
        raise ValueError("a study needs at least one trial")
    epoch = epoch or f"study-{study.seed}"
    nulls: list[float] = []
    for i in range(int(n_permutations)):
        if should_cancel is not None and should_cancel():
            break
        permuted = {
            (symbol, tf): permute_bars(
                bars, np.random.default_rng(permutation_seed(symbol, tf, epoch, i)))
            for (symbol, tf), bars in frames.items()
        }
        cache = MemoryBars(permuted)
        best = 0.0
        for trial in study.trials:
            pf, _ = trial_pf(cache, trial, study.symbols, study.windows,
                             registry, tfs=tfs)
            best = max(best, pf)
        nulls.append(float(best))
        if on_progress is not None:
            on_progress(i + 1, int(n_permutations))
    return nulls


# Mirrors DEFAULT_PROVING["verdict"]["min_trades"] in the platform's
# proving.py. The lab never imports the platform, so the floor is a parameter
# carrying the same default rather than a shared constant.
DEFAULT_MIN_TRADES = 20


def study_verdict(observed_pf: float | None, nulls: list[float], *,
                  alpha: float = 0.05, min_trades: int = DEFAULT_MIN_TRADES,
                  n_trades: int = 0) -> dict:
    """Score the observed best against the best-of-N null.

    Fails closed in every direction: no nulls, no observed value, or a ledger
    thinner than the floor all refute. A study that could not be scored is not
    a study that passed.
    """
    out = {"p_value": None, "survived": False, "n_permutations": len(nulls),
           "observed_pf": observed_pf, "reason": ""}
    if observed_pf is None or n_trades < min_trades:
        out["reason"] = "too few trades"
        return out
    if not nulls:
        out["reason"] = "no null distribution"
        return out
    p = permutation_pvalue(observed_pf, nulls)
    out["p_value"] = p
    if p is not None and p <= alpha:
        out["survived"] = True
        out["reason"] = "survived"
    else:
        out["reason"] = "p_value above alpha"
    return out
