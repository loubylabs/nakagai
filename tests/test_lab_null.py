"""The best-of-N permutation null."""

import pytest

from nakagai.lab.mutate import literal_trials
from nakagai.lab.null import best_of_n_null
from nakagai.lab.study import StudySpec
from tests.lab_helpers import (BASE_SPEC, lab_registry, random_walk_frames,
                               short_windows)


def _study(frames, n=3, seed=1):
    return StudySpec(trials=tuple(literal_trials(BASE_SPEC, n=n, seed=seed)),
                     symbols=("TEST",),
                     windows=tuple(short_windows(frames, "TEST")),
                     seed=seed)


def test_null_returns_one_value_per_permutation():
    frames = random_walk_frames("TEST", seed=3)
    nulls = best_of_n_null(frames, _study(frames), lab_registry(),
                           n_permutations=5)
    assert len(nulls) == 5
    assert all(isinstance(x, float) and x >= 0.0 for x in nulls)


def test_null_is_deterministic_for_an_epoch():
    frames = random_walk_frames("TEST", seed=3)
    study = _study(frames)
    a = best_of_n_null(frames, study, lab_registry(), n_permutations=4,
                       epoch="e1")
    b = best_of_n_null(frames, study, lab_registry(), n_permutations=4,
                       epoch="e1")
    assert a == b


def test_null_differs_across_epochs():
    frames = random_walk_frames("TEST", seed=3)
    study = _study(frames)
    a = best_of_n_null(frames, study, lab_registry(), n_permutations=4,
                       epoch="e1")
    b = best_of_n_null(frames, study, lab_registry(), n_permutations=4,
                       epoch="e2")
    assert a != b


@pytest.mark.slow
def test_null_takes_the_max_across_trials_not_the_first():
    # With more trials in the set, the best-of-N maximum can only rise. This is
    # the whole reason the null is computed this way: a bigger search has a
    # higher bar to clear.
    #
    # Marked slow: this one replays 8 trials plus 2 trials across 6
    # permutations and 3 windows, which is the "replay hundreds of backtests"
    # case the marker exists for. Its neighbours in this file replay far
    # fewer and stay unmarked on purpose, since they are the only default-run
    # coverage best_of_n_null has; moving all of them behind the marker would
    # leave the fast suite exercising the null not at all.
    frames = random_walk_frames("TEST", seed=3)
    small = best_of_n_null(frames, _study(frames, n=2), lab_registry(),
                           n_permutations=6, epoch="e")
    big = best_of_n_null(frames, _study(frames, n=8), lab_registry(),
                         n_permutations=6, epoch="e")
    assert sum(big) >= sum(small)


def test_zero_permutations_is_an_empty_null():
    frames = random_walk_frames("TEST", seed=3)
    assert best_of_n_null(frames, _study(frames), lab_registry(),
                          n_permutations=0) == []
