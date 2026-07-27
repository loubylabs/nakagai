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


def best_of_n_null(frames: dict, study: StudySpec, registry,
                   n_permutations: int, *, epoch: str = "",
                   tfs: TimeframeSet = DEFAULT_TIMEFRAMES) -> list[float]:
    """One maximum-across-trials PF per permutation.

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
    """
    if not study.trials:
        raise ValueError("a study needs at least one trial")
    epoch = epoch or f"study-{study.seed}"
    nulls: list[float] = []
    for i in range(int(n_permutations)):
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
    return nulls
