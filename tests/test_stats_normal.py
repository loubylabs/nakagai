"""Accuracy pins for the normal distribution helpers.

These exist because core carries no scipy and the deflated-Sharpe maths needs
Phi and Phi-inverse. Expected values come from tests/fixtures/dsr_goldens.json,
generated once from scipy.stats; see that file's `_source` key.
"""
import json
import pathlib

import pytest

from nakagai.stats import _norm_cdf, _norm_ppf

GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "dsr_goldens.json").read_text())


@pytest.mark.parametrize("z,expected", sorted(GOLDENS["norm_cdf"].items()))
def test_norm_cdf_matches_scipy(z, expected):
    assert _norm_cdf(float(z)) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("p,expected", sorted(GOLDENS["norm_ppf"].items()))
def test_norm_ppf_matches_scipy(p, expected):
    assert _norm_ppf(float(p)) == pytest.approx(expected, abs=1e-9)


def test_ppf_inverts_cdf():
    for z in (-2.5, -0.3, 0.0, 0.7, 1.9):
        assert _norm_ppf(_norm_cdf(z)) == pytest.approx(z, abs=1e-8)


def test_ppf_rejects_out_of_range():
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            _norm_ppf(bad)
