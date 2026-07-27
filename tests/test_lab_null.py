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
    #
    # literal_trials is prefix-stable: the n=2 trial set is exactly the first
    # two trials of the n=8 set, and both calls share epoch="e", so both see
    # identical permutations. That makes the comparison exact rather than
    # merely probable: under the correct implementation big[i] >= small[i]
    # holds elementwise and deterministically, and a sum comparison alone is
    # too weak to tell that from an implementation that only ever looks at
    # the first trial (small and big would then be the same list, sum equal
    # by construction, "sum(big) >= sum(small)" passing on exact equality).
    frames = random_walk_frames("TEST", seed=3)
    small = best_of_n_null(frames, _study(frames, n=2), lab_registry(),
                           n_permutations=6, epoch="e")
    big = best_of_n_null(frames, _study(frames, n=8), lab_registry(),
                         n_permutations=6, epoch="e")
    assert all(b >= s for b, s in zip(big, small))
    assert big != small


def test_zero_permutations_is_an_empty_null():
    frames = random_walk_frames("TEST", seed=3)
    assert best_of_n_null(frames, _study(frames), lab_registry(),
                          n_permutations=0) == []


def test_null_scores_every_trial_against_the_same_permuted_copy():
    # The module's defining property: within one permutation index, every
    # trial in the set is replayed on the SAME permuted copy of the bars, and
    # only the maximum across trials is kept. An implementation that instead
    # gave each trial its own independent permutation would still return a
    # max-shaped list of the right length, deterministic within an epoch and
    # rising with more trials, so none of the tests above can tell it apart
    # from the real thing. Decomposing the max is what catches it: if trial a
    # and trial b are each scored alone on one epoch, and then scored
    # together on that same epoch, the combined null must equal the
    # elementwise max of the two singles, permutation by permutation, because
    # permutation index i is one shared alternate history for every trial
    # sharing that run.
    frames = random_walk_frames("TEST", seed=3)
    trial_a, trial_b = literal_trials(BASE_SPEC, n=2, seed=1)
    windows = tuple(short_windows(frames, "TEST"))
    registry = lab_registry()

    def study_of(*trials):
        return StudySpec(trials=trials, symbols=("TEST",), windows=windows, seed=1)

    a = best_of_n_null(frames, study_of(trial_a), registry, n_permutations=4,
                       epoch="e")
    b = best_of_n_null(frames, study_of(trial_b), registry, n_permutations=4,
                       epoch="e")
    both = best_of_n_null(frames, study_of(trial_a, trial_b), registry,
                          n_permutations=4, epoch="e")
    assert both == [max(x, y) for x, y in zip(a, b)]


def test_null_does_not_carry_a_running_max_across_permutations():
    # Each permutation index is its own independent alternate history, so its
    # null value must be that permutation's own maximum across trials, not a
    # maximum running forward from earlier permutations. Hoisting `best =
    # 0.0` above the permutation loop instead of resetting it inside is a
    # one-line slip that produces a monotone non-decreasing sequence: since
    # max is associative, running-max commutes with the elementwise max used
    # in the decomposition test above, so that test cannot catch this. A
    # monotone sequence is not a null distribution; every p-value drawn from
    # it would be biased toward significance as more permutations pile on.
    # On this fixture the correct output is not sorted in ascending order,
    # which a running-max implementation can never produce, so asserting
    # that catches the slip deterministically without hardcoding the values.
    frames = random_walk_frames("TEST", seed=3)
    nulls = best_of_n_null(frames, _study(frames), lab_registry(),
                           n_permutations=4, epoch="e1")
    assert nulls != sorted(nulls)


def test_null_rejects_an_empty_trial_set():
    # A best-of-zero null is all zeros, which makes any observed statistic
    # look significant. run_study already refuses an empty trial set with
    # this same message; best_of_n_null must fail the same way rather than
    # silently returning a null that would rubber-stamp anything.
    frames = random_walk_frames("TEST", seed=3)
    empty = StudySpec(trials=(), symbols=("TEST",),
                      windows=tuple(short_windows(frames, "TEST")), seed=1)
    with pytest.raises(ValueError, match="at least one trial"):
        best_of_n_null(frames, empty, lab_registry(), n_permutations=3)
