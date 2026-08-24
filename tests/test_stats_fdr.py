"""Benjamini-Hochberg, the false-discovery control a search is judged by.

The two worked examples are the ones that separate a step-up procedure from a
step-down one. A test carrying only the first would pass against either, and
step-down is the natural thing to write by accident: scan from the smallest
p-value and stop at the first that fails its threshold.

Order invariance is asserted rather than assumed, because it is what makes
this procedure usable on a candidate set at all. A candidate set has no order,
and the autocorrelation-based trial count retired in this same change returned
a different answer for the same Sharpes in a different arrangement.
"""

from nakagai.stats import benjamini_hochberg


def test_matches_the_worked_example():
    # Sorted [0.01, 0.03, 0.04, 0.20] against (k/4)*0.05 = 0.0125, 0.025,
    # 0.0375, 0.05. Scanning down from k=4: 0.20 <= 0.05 fails, 0.04 <= 0.0375
    # fails, 0.03 <= 0.025 fails, 0.01 <= 0.0125 holds. k=1.
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.20], 0.05) == \
        [True, False, False, False]


def test_rejects_the_whole_cascade_not_only_the_smallest():
    # Every one of the first three clears its own rank threshold, so k=3.
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.20], 0.05) == \
        [True, True, True, False]


def test_steps_up_past_a_rank_that_fails_its_own_threshold():
    """The case that separates step-up from step-down, and the only one that does.

    Sorted [0.001, 0.03, 0.035, 0.04] against (k/4)*0.05 = 0.0125, 0.025,
    0.0375, 0.05. Rank 2 FAILS (0.03 > 0.025) while ranks 3 and 4 both pass.
    Step-up takes the largest passing rank, k=4, and rejects everything.
    Step-down stops at the first failure and rejects only the smallest.

    Neither worked example above catches this: in both of them the first
    failing rank is also the last, so the two procedures agree. Measured, a
    step-down mutant passed all five of the other tests in this file.
    """
    assert benjamini_hochberg([0.001, 0.03, 0.035, 0.04], 0.05) == \
        [True, True, True, True]


def test_degenerate_all_fail_and_all_pass():
    assert benjamini_hochberg([1.0] * 10, 0.05) == [False] * 10
    assert benjamini_hochberg([0.001] * 10, 0.05) == [True] * 10


def test_empty_input_is_total_rather_than_a_raise():
    # Unreachable rather than undecided: the search verdict is gated behind
    # more than one candidate, so this is never called with fewer than two.
    # It is total anyway, because a raise here would turn an upstream defect
    # into a failed read rather than an empty answer.
    assert benjamini_hochberg([], 0.05) == []


def test_is_order_invariant():
    """The property that carries the verdict.

    Fixed permutations rather than random ones, so a failure is reproducible
    from the test alone.
    """
    p = [0.01, 0.20, 0.03, 0.04, 0.90]
    base = benjamini_hochberg(p, 0.05)
    permutations = [
        [4, 3, 2, 1, 0],
        [1, 0, 3, 2, 4],
        [2, 4, 0, 1, 3],
        [0, 2, 4, 1, 3],
        [3, 1, 4, 0, 2],
    ]
    for order in permutations:
        permuted = [p[i] for i in order]
        assert benjamini_hochberg(permuted, 0.05) == [base[i] for i in order], \
            f"order {order} changed the decision"


def test_the_rank_threshold_is_inclusive():
    """`<=`, not `<`, and the boundary is the only case that can tell.

    With m=4 and alpha=0.05 the rank-1 threshold is exactly 0.0125. A p-value
    sitting exactly on it must be rejected. Measured: mutating `<=` to `<` left
    all ten of this node's other tests green, because none of them put a
    p-value on a threshold.
    """
    assert benjamini_hochberg([0.0125, 0.9, 0.9, 0.9], 0.05) == \
        [True, False, False, False]


def test_refuses_a_p_value_that_is_not_a_probability():
    """A meaningless p-value is refused, never coerced.

    NaN compares False against every threshold, so it looks safe, but it also
    sorts unpredictably and drags the rank cut with it. Measured before this
    guard existed, [0.01, nan, 0.02] returned [True, True, True]: the
    procedure certified the NaN as a discovery.
    """
    import pytest

    for bad in (float("nan"), float("inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="p-value out of range"):
            benjamini_hochberg([0.01, bad, 0.02], 0.05)
