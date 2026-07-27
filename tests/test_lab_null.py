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


from nakagai.lab.null import study_verdict


def test_verdict_survives_a_clear_winner():
    # 24 nulls, mirroring the PERMUTATIONS count Task 11 uses for the same
    # reason: with N nulls the smallest reachable p-value is 1/(N+1), and
    # 1/25 = 0.04 sits comfortably inside the default alpha of 0.05 rather
    # than exactly on the boundary where float equality would decide it.
    nulls = [1.0] * 24
    out = study_verdict(2.5, nulls, n_trades=40)
    assert out["survived"] is True
    assert out["p_value"] == (1 + 0) / (24 + 1)
    assert out["reason"] == "survived"


def test_verdict_cannot_survive_with_too_few_permutations():
    # The smallest p-value a permutation test can report is 1/(N+1), so N
    # must be at least 19 before alpha=0.05 is reachable at all. A study that
    # runs too few permutations cannot produce a survivor no matter how large
    # its edge, even a decisive one like an observed PF of 2.5 against a
    # null clustered at 1.0. This is the exact shape of test data that once
    # sat in test_verdict_survives_a_clear_winner: 5 nulls give a floor of
    # 1/6, which can never clear alpha=0.05, so that test's expectation of
    # survival was self-contradictory rather than merely unlucky.
    out = study_verdict(2.5, [1.0, 1.1, 0.9, 1.2, 1.05], n_trades=40)
    assert out["survived"] is False
    assert out["reason"] == "p_value above alpha"
    assert out["p_value"] == (1 + 0) / (5 + 1)


def test_verdict_refutes_an_ordinary_result():
    out = study_verdict(1.05, [1.0, 1.1, 0.9, 1.2, 1.5], n_trades=40)
    assert out["survived"] is False
    assert out["reason"] == "p_value above alpha"


def test_alpha_actually_gates_survival():
    # The p-value does not change with alpha, only how it is judged. Score
    # the SAME observed/null pair twice, differing only in alpha, so the
    # only thing that can explain a different verdict is alpha itself. An
    # implementation that hardcoded 0.05 and ignored the parameter would
    # refuse both calls.
    observed, nulls = 1.05, [1.0, 1.1, 0.9, 1.2, 1.5]
    refuted = study_verdict(observed, nulls, n_trades=40, alpha=0.05)
    survived = study_verdict(observed, nulls, n_trades=40, alpha=0.7)
    assert refuted["p_value"] == survived["p_value"]
    assert refuted["survived"] is False
    assert survived["survived"] is True
    assert survived["reason"] == "survived"


def test_verdict_survives_at_the_alpha_boundary():
    # p == alpha is deliberately inclusive: the check is p <= alpha, not
    # p < alpha. Pinned on purpose so a future reader does not "tidy" this
    # into a strict comparison, which would silently raise the bar and
    # refuse every study that lands exactly at alpha.
    out = study_verdict(9.0, [1.0] * 19, n_trades=40)
    assert out["p_value"] == 0.05
    assert out["survived"] is True
    assert out["reason"] == "survived"


def test_verdict_trade_floor_boundary_is_inclusive():
    # n_trades == min_trades is sufficient, matching "twenty trades is the
    # floor" on the thin-ledger test below: the floor itself counts, it is
    # not the first refused value. Bracket the boundary from both sides on
    # the same observed/null pair so only n_trades differs between calls.
    observed, nulls = 5.0, [1.0] * 24
    just_under = study_verdict(observed, nulls, n_trades=19)
    at_floor = study_verdict(observed, nulls, n_trades=20)
    assert just_under["survived"] is False
    assert just_under["reason"] == "too few trades"
    assert at_floor["survived"] is True
    assert at_floor["reason"] == "survived"


def test_verdict_refutes_a_thin_ledger_however_significant():
    # Twenty trades is the floor the proving verdict already uses. A PF built
    # on three trades is not evidence, no matter where it sits in the null.
    out = study_verdict(5.0, [1.0] * 20, n_trades=3)
    assert out["survived"] is False
    assert out["reason"] == "too few trades"


def test_verdict_with_no_nulls_cannot_survive():
    out = study_verdict(2.0, [], n_trades=40)
    assert out["survived"] is False
    assert out["p_value"] is None
    assert out["reason"] == "no null distribution"


def test_verdict_with_no_observed_cannot_survive():
    out = study_verdict(None, [1.0, 1.1], n_trades=0)
    assert out["survived"] is False
    assert out["reason"] == "too few trades"


def test_verdict_reports_the_permutation_count():
    assert study_verdict(2.0, [1.0] * 7, n_trades=40)["n_permutations"] == 7
