"""The lab: strategy-spec search, scored against a best-of-N permutation null.

Deterministic and offline, like the rest of the core. Given a seed, the same
trials are generated and the same numbers come out, in this process or any
other. Nothing here reaches a network, holds a credential, or knows what a
platform is: the edge and the platform both consume this package, so it may
depend on neither.

Usage, in the order the pieces are meant to be called. `cache` must be built
over the same bars as `frames`, i.e. `cache = MemoryBars(frames)`, or the
observed statistic and the null are scored on different histories and the
resulting p-value means nothing:

    trials = literal_trials(base_spec, n=60, seed=7)
    study = StudySpec(trials=tuple(trials), symbols=("SPY",),
                      windows=tuple(windows), seed=7)
    cache = MemoryBars(frames)
    observed = run_study(cache, study, registry)
    nulls = best_of_n_null(frames, study, registry, n_permutations=200)
    verdict = study_verdict(observed.best.pf, nulls,
                            n_trades=sum(r.n_trades for r in observed.results))
"""

from nakagai.lab.mutate import (Site, Trial, composite_trials, literal_trials,
                                mutable_sites, spec_hash)
from nakagai.lab.null import best_of_n_null, study_verdict
from nakagai.lab.study import (StudyResult, StudySpec, TrialResult, run_study,
                               trial_pf)

__all__ = [
    "Site", "Trial", "composite_trials", "literal_trials", "mutable_sites",
    "spec_hash", "best_of_n_null", "study_verdict", "StudyResult",
    "StudySpec", "TrialResult", "run_study", "trial_pf",
]
