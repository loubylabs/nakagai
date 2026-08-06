"""PSR, DSR, MTRL and effective_n_trials against the reference implementation.

Goldens were produced by purgedcv 0.1.3 (MIT) from a returns ARRAY; we feed
the same series' pooled SUMS. Agreement is therefore evidence for both the
vendored maths and the pooling identity at once.
"""
import json
import pathlib

import numpy as np
import pytest

from nakagai.stats import (deflated_sharpe_ratio, effective_n_trials,
                           min_track_record_length, pooled_moments,
                           probabilistic_sharpe_ratio)

GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "dsr_goldens.json").read_text())


@pytest.fixture
def moments():
    rng = np.random.default_rng(20260806)
    r = np.round(
        rng.normal(0.0008, 0.011, 260) + rng.standard_t(5, 260) * 0.0015, 12)
    return pooled_moments(int(r.size), float(r.sum()), float((r ** 2).sum()),
                          float((r ** 3).sum()), float((r ** 4).sum()))


def test_psr_matches_reference(moments):
    assert probabilistic_sharpe_ratio(moments, 0.0) == pytest.approx(
        GOLDENS["psr_vs_0"], rel=1e-9)
    assert probabilistic_sharpe_ratio(moments, 0.05) == pytest.approx(
        GOLDENS["psr_vs_0p05"], rel=1e-9)


def test_dsr_matches_reference(moments):
    for key, n in (("dsr_n1_var0p5", 1), ("dsr_n10_var0p5", 10),
                   ("dsr_n200_var0p5", 200)):
        assert deflated_sharpe_ratio(moments, n, 0.5) == pytest.approx(
            GOLDENS[key], rel=1e-7, abs=1e-300)
    assert deflated_sharpe_ratio(moments, 10, 0.001) == pytest.approx(
        GOLDENS["dsr_n10_var0p001"], rel=1e-9)


def test_one_trial_is_psr_against_zero(moments):
    """No search means no multiple-comparison correction, so DSR reduces."""
    assert deflated_sharpe_ratio(moments, 1, 0.5) == pytest.approx(
        probabilistic_sharpe_ratio(moments, 0.0), rel=1e-12)


def test_more_trials_never_raises_the_verdict(moments):
    """The direction is the whole point: a deflation that could go either way
    is not a deflation."""
    values = [deflated_sharpe_ratio(moments, n, 0.5) for n in (1, 5, 25, 200)]
    assert values == sorted(values, reverse=True)


def test_mtrl_matches_reference(moments):
    assert min_track_record_length(
        moments.sharpe, 0.0, 0.05, moments.skew, moments.kurtosis
    ) == pytest.approx(GOLDENS["mtrl_ddof0_alpha05"], rel=1e-9)


def test_effective_n_trials_matches_reference():
    assert effective_n_trials(GOLDENS["effective_n_trials_input"]) == \
        GOLDENS["effective_n_trials_autocorr"]


def test_effective_n_trials_is_bounded():
    n = len(GOLDENS["effective_n_trials_input"])
    assert 1 <= effective_n_trials(GOLDENS["effective_n_trials_input"]) <= n
    assert effective_n_trials([0.5]) == 1
    assert effective_n_trials([]) == 1


def test_refuses_rather_than_returning_a_number(moments):
    assert probabilistic_sharpe_ratio(None) is None
    assert deflated_sharpe_ratio(None, 10, 0.5) is None
    assert deflated_sharpe_ratio(moments, 10, -1.0) is None   # negative variance
    assert min_track_record_length(0.05, 0.05, 0.05, 0.0, 3.0) is None  # no gap
