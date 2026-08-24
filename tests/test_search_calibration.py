"""The synthetic control: does the search verdict actually reject noise?

A search space large enough to be interesting manufactures spectacular
backtests by chance alone. This file is the evidence that the verdict shipped
against that space controls its false-positive rate, measured rather than
asserted, and that the criterion can go red.

THE VERDICT, FROZEN. Given a batch's variants:

  p_i      = 1 - probabilistic_sharpe_ratio(moments_i, 0.0), or 1.0 where the
             moments are None (no evidence against the null is not evidence
             for it)
  m        = the raw candidate count, INCLUDING the p = 1 ones
  verdict  = any(benjamini_hochberg(p, ALPHA))
  leader   = the smallest (p, variant_id) pair, or None if every p is 1.0;
             never argmax(sharpe); see the unequal-n test below
  deflated = deflated_sharpe_ratio(leader's moments, m, var_sharpe), where
             var_sharpe is the population variance of the PER-OBSERVATION
             Sharpes of the variants that have moments

Benjamini-Hochberg decides. The deflated Sharpe is computed and reported
beside it and never gates it: the two correct for the same selection effect,
so ANDing them charges one multiplicity twice and produces a rule that says
"not significant" to a genuinely good strategy almost always.

WHY THIS LIVES IN CORE. It needs nothing but this module. The platform will
compute the same verdict over real rows; if it computes it any differently,
this file certifies a procedure the product does not run. The platform half
owes an assertion that its verdict agrees with `_verdict` here on the same
fixture.
"""

import numpy as np
import pytest

from nakagai.stats import (benjamini_hochberg, deflated_sharpe_ratio,
                           pooled_moments, probabilistic_sharpe_ratio)

ALPHA = 0.05
SEED = 20260810
REPS = 8000
CANDIDATES = 100
N_OBS = 260
INTERVAL = (0.03, 0.07)


def _draw(rng, count, n_obs):
    """One batch of pure-noise candidates, as a (count, n_obs) array.

    Mean zero, so the true Sharpe is exactly zero by construction. The
    t-distributed term supplies the non-Gaussian skew and kurtosis the PSR
    formula exists to correct for; it is the same construction the pooled
    moments fixture in test_stats_deflated.py uses.

    THE DRAW ORDER IS PART OF THE FIXTURE. Taking one (count, n_obs) matrix
    per batch consumes the generator differently from taking `count` separate
    (n_obs,) vectors, and the measured false-positive rate moves by about half
    a point between the two. The interval below absorbs that; a golden on the
    rate would not, which is why this asserts a range and not a point.
    """
    return (rng.normal(0.0, 0.011, (count, n_obs))
            + rng.standard_t(5, (count, n_obs)) * 0.0015)


def _moments(series):
    n = len(series)
    return pooled_moments(n, float(series.sum()), float((series ** 2).sum()),
                          float((series ** 3).sum()), float((series ** 4).sum()))


def _verdict(candidates):
    """The frozen verdict over one batch. See the module docstring."""
    m = len(candidates)
    p_values = []
    for _, mom in candidates:
        psr = probabilistic_sharpe_ratio(mom, 0.0)
        p_values.append(1.0 if psr is None else 1.0 - psr)
    significant = any(benjamini_hochberg(p_values, ALPHA))
    if all(p_value == 1.0 for p_value in p_values):
        return {"significant": significant, "leader": None,
                "p_values": p_values, "deflated": None,
                "psr": None, "n_candidates": m}
    leader_index = min(range(m), key=lambda i: (p_values[i], candidates[i][0]))
    leader, leader_moments = candidates[leader_index]
    sharpes = [mom.sharpe for _, mom in candidates if mom is not None]
    var_sharpe = float(np.var(np.asarray(sharpes), ddof=0)) if sharpes else 0.0
    deflated = deflated_sharpe_ratio(leader_moments, m, var_sharpe)
    return {"significant": significant, "leader": leader,
            "p_values": p_values, "deflated": deflated,
            "psr": 1.0 - p_values[leader_index], "n_candidates": m}


def _noise_batch(seed, count=CANDIDATES, n_obs=N_OBS):
    rng = np.random.default_rng(seed)
    return [(f"noise-{index}", _moments(row))
            for index, row in enumerate(_draw(rng, count, n_obs))]


def test_the_verdict_controls_its_false_positive_rate_over_pure_noise():
    """The test that matters. Everything else in this node is downstream of it.

    Pure noise, so every rejection is a false one. The rate must land near the
    nominal alpha, and the interval is eight empirical standard errors wide at
    this sample size: tight enough to catch an inverted procedure or a wrong
    m, wide enough to absorb sampling noise.

    The same run measures the comparator the program's own acceptance bullet
    names: the single-strategy statistic the product already serves, applied
    to the batch's best candidate. That is the figure this node has to beat.
    """
    rng = np.random.default_rng(SEED)
    hits = 0
    per_strategy_hits = 0
    for _ in range(REPS):
        candidates = [(f"noise-{index}", _moments(row))
                      for index, row in enumerate(_draw(rng, CANDIDATES, N_OBS))]
        result = _verdict(candidates)
        hits += result["significant"]
        # The per-strategy lens: read the best candidate through the
        # undeflated PSR the product already ships, with its "usual bar".
        per_strategy_hits += (1.0 - min(result["p_values"])) >= 0.95

    rate = hits / REPS
    per_strategy_rate = per_strategy_hits / REPS
    assert INTERVAL[0] <= rate <= INTERVAL[1], (
        f"false-positive rate {rate:.5f} outside {INTERVAL}")
    # Strictly more conservative than the per-strategy figure, on the identical
    # batches. Measured at roughly 0.048 against 0.994, a factor of about 20.
    assert rate < per_strategy_rate
    assert per_strategy_rate > 0.9, (
        "the comparator collapsed, so the conservatism claim proves nothing")


def test_the_deflation_is_pinned_to_a_golden_not_to_an_inequality():
    """The deflation's own criterion, and why it is a golden.

    The obvious guard, `0 < deflated < psr`, CANNOT FAIL under the mutation it
    exists to catch. Feeding an annualized var_sharpe (the units mismatch: the
    display Sharpe is annualized, the moments' is per-observation, they differ
    by sqrt(252), and both are floats named `sharpe`) drives the result to
    7.0e-305. That is a positive float, so `0 < deflated` holds and the guard
    stays green. The golden reddens on it.
    """
    result = _verdict(_noise_batch(SEED))
    assert result["psr"] == pytest.approx(0.9988490717803324, rel=1e-9)
    assert result["deflated"] == pytest.approx(0.6936677052012623, rel=1e-9)


def test_tied_leaders_use_the_smallest_stable_variant_id():
    """An equal-p tie must not let input position choose the leader."""
    beta = _variant(1.1, 200, 1)
    alpha = _variant(0.2, 10000, 2)
    forward = [("beta", beta), ("alpha", alpha)]
    reverse = list(reversed(forward))

    forward_result = _verdict(forward)
    reverse_result = _verdict(reverse)

    assert forward_result["leader"] == "alpha"
    assert reverse_result["leader"] == "alpha"
    assert forward_result["psr"] == reverse_result["psr"]
    assert forward_result["deflated"] == reverse_result["deflated"]


def test_all_insufficient_candidates_have_no_leader():
    """A batch with no evidence has no variant whose figures can be reported."""
    alpha = _variant(-1.1, 200, 1)
    beta = _variant(-0.2, 10000, 2)
    result = _verdict([("alpha", alpha), ("beta", beta)])

    assert result["significant"] is False
    assert result["leader"] is None
    assert result["psr"] is None
    assert result["deflated"] is None
    assert result["p_values"] == [1.0, 1.0]
    assert result["n_candidates"] == 2


def test_none_moments_contribute_a_one_p_value():
    """A missing moments record carries no evidence against the null."""
    result = _verdict([("alpha", None), ("beta", None)])

    assert result["significant"] is False
    assert result["leader"] is None
    assert result["psr"] is None
    assert result["deflated"] is None
    assert result["p_values"] == [1.0, 1.0]
    assert result["n_candidates"] == 2


def test_the_verdict_is_invariant_under_permutation_of_the_candidate_set():
    """A candidate set has no order, so the verdict must not have one either.

    This is the property that decided the trial count. The autocorrelation
    based count retired in this same change derived its answer from the
    SEQUENTIAL LAGS of the list it was handed, so the same Sharpes in a
    different arrangement gave a different count and, measured, flipped the
    verdict on 2.6% of pure-noise candidate sets. Benjamini-Hochberg sorts
    before deciding, so it cannot.
    """
    rng = np.random.default_rng(4242)
    for offset in range(100):
        candidates = _noise_batch(1000 + offset, count=10)
        base = _verdict(candidates)
        for _ in range(25):
            order = rng.permutation(len(candidates))
            permuted = [candidates[i] for i in order]
            shuffled = _verdict(permuted)
            assert shuffled["significant"] == base["significant"], (
                f"set {offset} verdict flipped under permutation {list(order)}")
            # The DEFLATION is asserted too, and that is not belt-and-braces.
            # An order-dependent count reintroduced into the deflation alone
            # moves this number while leaving the verdict untouched, so a
            # guard that watched only `significant` would not see it, which is
            # exactly the reintroduction Acceptance 3 exists to catch.
            assert shuffled["deflated"] == pytest.approx(base["deflated"], rel=1e-12), (
                f"set {offset} deflation moved under permutation {list(order)}")


def _variant(target_sharpe, n, seed):
    """Pooled moments with an EXACT per-observation Sharpe.

    Standardize first, then scale. Adding a target mean onto an unscaled
    spread changes the standard deviation, so the realized Sharpe misses the
    target, and for this fixture that destroys the discrimination entirely:
    both variants end up rejected and the two selection rules agree.
    """
    rng = np.random.default_rng(seed)
    series = rng.normal(0.0, 1.0, n)
    series = (series - series.mean()) / series.std(ddof=0)
    return _moments(series * 0.011 + target_sharpe * 0.011)


def test_the_leader_is_the_minimum_p_variant_not_the_maximum_sharpe_one():
    """Constructed, because the synthetic control is blind to this by design.

    Every candidate in the control has the same n, so the two selection rules
    almost never part company there. They part company on unequal pooled
    observation counts, which are ordinary rather than contrived: a variant's
    rows with null daily sums are dropped before counting, so two variants of
    one sweep routinely pool different numbers of observations.

    `p = 1 - PSR` charges skew, kurtosis and the observation count, so it is
    not monotone in the Sharpe alone. A verdict that consulted the largest
    Sharpe would read NOT SIGNIFICANT on a search that did produce a rejection.
    """
    a = _variant(0.27531, 60, 11)    # the batch's largest Sharpe, few observations
    b = _variant(0.19017, 520, 12)   # a smaller Sharpe, many observations
    assert a.sharpe > b.sharpe

    batch = ([("a", a), ("b", b)]
             + [(f"null-{index}", None) for index in range(98)])
    result = _verdict(batch)

    assert result["n_candidates"] == 100
    assert result["significant"] is True
    assert result["leader"] == "b", "the leader must be B, the minimum-p variant"
    assert result["deflated"] == pytest.approx(0.9666696522438584, rel=1e-12)

    # And the rule the previous draft specified reads the opposite verdict on
    # the identical fixture, which is what makes this test discriminate.
    rejected = benjamini_hochberg(result["p_values"], ALPHA)
    max_sharpe_index = 0
    assert rejected[max_sharpe_index] is False
    assert rejected[1] is True
