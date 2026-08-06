"""The pooling identity: raw moment sums recover the same moments as the array.

This is the test that would catch a silent divergence. It checks the pooled
path two independent ways, because either alone is weak: against moments
computed directly from the concatenated array (catches pooling algebra), and
against golden values generated from scipy.stats outside this project
(catches the test and the implementation sharing a wrong understanding of the
bias correction).
"""
import json
import pathlib

import numpy as np
import pytest

from nakagai.stats import pooled_moments

GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "dsr_goldens.json").read_text())


def _returns() -> np.ndarray:
    """Rebuild the exact series the goldens were generated from."""
    rng = np.random.default_rng(20260806)
    return np.round(
        rng.normal(0.0008, 0.011, 260) + rng.standard_t(5, 260) * 0.0015, 12)


def test_fixture_series_reproduces():
    r = _returns()
    assert r.size == GOLDENS["n"]
    assert list(r[:5]) == pytest.approx(GOLDENS["first5"], abs=1e-12)
    assert float(r[-1]) == pytest.approx(GOLDENS["last"], abs=1e-12)


def test_sums_match_the_goldens():
    r = _returns()
    assert float(r.sum()) == pytest.approx(GOLDENS["sum"], rel=1e-12)
    assert float((r ** 2).sum()) == pytest.approx(GOLDENS["sum_sq"], rel=1e-12)
    assert float((r ** 3).sum()) == pytest.approx(GOLDENS["sum_cube"], rel=1e-12)
    assert float((r ** 4).sum()) == pytest.approx(GOLDENS["sum_fourth"], rel=1e-12)


def test_pooled_moments_match_the_goldens():
    r = _returns()
    m = pooled_moments(int(r.size), float(r.sum()), float((r ** 2).sum()),
                       float((r ** 3).sum()), float((r ** 4).sum()))
    assert m is not None
    assert m.sharpe == pytest.approx(GOLDENS["sharpe_ddof0"], rel=1e-10)
    assert m.skew == pytest.approx(GOLDENS["skew_bias_corrected"], rel=1e-9)
    assert m.kurtosis == pytest.approx(
        GOLDENS["kurtosis_bias_corrected_nonexcess"], rel=1e-9)


def test_pooling_two_windows_equals_one_long_window():
    """Sums add, which is the whole reason the engine emits them."""
    r = _returns()
    a, b = r[:130], r[130:]
    parts = [(x.size, x.sum(), (x ** 2).sum(), (x ** 3).sum(), (x ** 4).sum())
             for x in (a, b)]
    combined = pooled_moments(*(int(sum(p[0] for p in parts)),
                                *(float(sum(p[i] for p in parts))
                                  for i in (1, 2, 3, 4))))
    whole = pooled_moments(int(r.size), float(r.sum()), float((r ** 2).sum()),
                           float((r ** 3).sum()), float((r ** 4).sum()))
    assert combined.skew == pytest.approx(whole.skew, rel=1e-9)
    assert combined.kurtosis == pytest.approx(whole.kurtosis, rel=1e-9)


def test_refuses_when_it_cannot_answer():
    assert pooled_moments(3, 0.1, 0.1, 0.1, 0.1) is None      # n < 4
    assert pooled_moments(100, 0.0, 0.0, 0.0, 0.0) is None    # zero variance
