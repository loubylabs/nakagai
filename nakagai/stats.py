"""Statistical math for backtest results.

No evidence store, no workspace, no config: everything is parameterized.
"""

import math
from dataclasses import dataclass

import pandas as pd

PF_CLAMP = 1000.0   # a ledger or resample with no losers: "infinite" PF


def _norm_cdf(z: float) -> float:
    """Standard normal CDF, from math.erf and math.erfc.

    Core carries four dependencies and scipy is not among them. Phi is a
    two-line identity on erf, which the standard library has, so importing a
    numerical stack for it would be a poor trade.

    Routed through erfc for z < 0 rather than 1 + erf(z/sqrt2) uniformly:
    erf saturates to exactly -1.0 well before the true left-tail probability
    underflows (deflated_sharpe_ratio reaches z below -15 at realistic trial
    counts), and 1 + (-1.0) cancels to exactly 0.0 where the true answer is
    still representable, e.g. 1e-65. erfc has no such cancellation."""
    if z < 0:
        return 0.5 * math.erfc(-z / math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Wichura's AS241 rational approximation, accurate to about 1e-16 across the
# whole open interval. The long coefficient tables are the algorithm; they are
# not tunable and must not be "cleaned up".
_A = (3.3871328727963666080e0, 1.3314166789178437745e2, 1.9715909503065514427e3,
      1.3731693765509461125e4, 4.5921953931549871457e4, 6.7265770927008700853e4,
      3.3430575583588128105e4, 2.5090809287301226727e3)
_B = (1.0, 4.2313330701600911252e1, 6.8718700749205790830e2,
      5.3941960214247511077e3, 2.1213794301586595867e4, 3.9307895800092710610e4,
      2.8729085735721942674e4, 5.2264952788528545610e3)
_C = (1.42343711074968357734e0, 4.63033784615654529590e0,
      5.76949722146069140550e0, 3.64784832476320460504e0,
      1.27045825245236838258e0, 2.41780725177450611770e-1,
      2.27238449892691845833e-2, 7.74545014278341407640e-4)
_D = (1.0, 2.05319162663775882187e0, 1.67638483018380384940e0,
      6.89767334985100004550e-1, 1.48103976427480074590e-1,
      1.51986665636164571966e-2, 5.47593808499534494600e-4,
      1.05075007164441684324e-9)
_E = (6.65790464350110377720e0, 5.46378491116411436990e0,
      1.78482653991729133580e0, 2.96560571828504891230e-1,
      2.65321895265761230930e-2, 1.24266094738807843860e-3,
      2.71155556874348757815e-5, 2.01033439929228813265e-7)
_F = (1.0, 5.99832206555887937690e-1, 1.36929880922735805310e-1,
      1.48753612908506148525e-2, 7.86869131145613259100e-4,
      1.84631831751005468180e-5, 1.42151175831644588870e-7,
      2.04426310338993978564e-15)


def _poly(coeffs: tuple[float, ...], x: float) -> float:
    out = 0.0
    for c in reversed(coeffs):
        out = out * x + c
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Wichura AS241).

    The one piece of this module with no standard-library equivalent. Raises
    on p outside the OPEN interval (0, 1): Phi-inverse diverges at both ends,
    and returning an infinity would travel silently into a Sharpe deflation."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in the open interval (0, 1), got {p!r}")
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _poly(_A, r) / _poly(_B, r)
    r = p if q < 0 else 1.0 - p
    r = math.sqrt(-math.log(r))
    if r <= 5.0:
        r -= 1.6
        value = _poly(_C, r) / _poly(_D, r)
    else:
        r -= 5.0
        value = _poly(_E, r) / _poly(_F, r)
    return -value if q < 0 else value


def pf_from_trades(trades: pd.DataFrame | None) -> float | None:
    """Pooled profit factor over a trade ledger; None when it has no trades."""
    if trades is None or len(trades) == 0:
        return None
    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl <= 0].sum()))
    if losses == 0:
        return PF_CLAMP if wins > 0 else 0.0
    return wins / losses


@dataclass(frozen=True)
class PooledMoments:
    """The first four moments of a return series, recovered from raw sums.

    `sharpe` is PER OBSERVATION and uses the population standard deviation
    (ddof=0), matching the convention the deflated-Sharpe maths below is
    written against. Annualizing is the caller's job and deliberately not
    done here: this module knows nothing about trading calendars.

    `skew` and `kurtosis` are BIAS-CORRECTED sample estimates, and `kurtosis`
    is NON-excess (a normal distribution scores 3.0, not 0.0). Both match
    scipy.stats' `bias=False` / `fisher=False` forms, which is what the
    published PSR and DSR formulae expect.
    """

    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float


def pooled_moments(n: int, total: float, total_sq: float,
                   total_cube: float, total_fourth: float) -> PooledMoments | None:
    """Recover the first four moments from raw power sums, or None.

    Sums add across windows where ratios cannot, which is why
    engine/metrics.py emits them per window instead of emitting the moments.
    Thirteen one-month windows pooled here is a statistic; each one alone is
    not.

    ONE derivation with two repos calling it. The platform's pooled_risk
    imports this rather than repeating the algebra, because a second copy is
    how the bias correction drifts in exactly one of them.

    None rather than a number when n < 4 (the kurtosis bias correction has
    (n-2)(n-3) in a denominator) or the variance is not positive. Zero
    variance means every day returned the same amount, and a Sharpe there is
    a division by zero, not an infinitely good result.
    """
    if n < 4:
        return None
    mean = total / n
    # Central moments from raw power sums (the binomial expansions).
    m2 = total_sq / n - mean ** 2
    if not (m2 > 0):
        return None
    m3 = total_cube / n - 3 * mean * total_sq / n + 2 * mean ** 3
    m4 = (total_fourth / n - 4 * mean * total_cube / n
          + 6 * mean ** 2 * total_sq / n - 3 * mean ** 4)
    std = math.sqrt(m2)
    g1 = m3 / std ** 3          # biased skew
    g2 = m4 / m2 ** 2           # biased NON-excess kurtosis
    # scipy's bias=False corrections. Written out rather than folded together
    # so each one can be checked against its own reference.
    skew = g1 * math.sqrt(n * (n - 1)) / (n - 2)
    excess = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (g2 - 3) + 6)
    return PooledMoments(n=n, mean=mean, std=std, sharpe=mean / std,
                         skew=skew, kurtosis=excess + 3.0)


# ---------------------------------------------------------------------------
# Search-aware Sharpe statistics.
#
# Vendored from github.com/eslazarev/purged-cross-validation (purgedcv 0.1.3),
# MIT, Copyright (c) 2026 Evgenii Lazarev. The maths is unchanged; the entry
# points take PooledMoments instead of a returns array, because the platform
# holds thirteen windows of sums and never the whole series at once.
#
# Formulae: Bailey & Lopez de Prado, "The Sharpe Ratio Efficient Frontier"
# (2012) for PSR and the minimum track record length, and "The Deflated Sharpe
# Ratio" (2014) for DSR.
# ---------------------------------------------------------------------------

_GAMMA_EM = 0.5772156649015329   # Euler-Mascheroni


def probabilistic_sharpe_ratio(m: PooledMoments | None,
                               benchmark: float = 0.0) -> float | None:
    """Probability the true Sharpe exceeds `benchmark`, given skew and kurtosis.

    The answer to a problem this codebase already had: an annualized Sharpe on
    twenty returns is a rounding of noise, which is why `_sharpe` returns None
    below sixty. This reports a PROBABILITY instead of a point estimate, so it
    stays meaningful at small n where the ratio does not.
    """
    if m is None:
        return None
    denom_sq = (1 - m.skew * m.sharpe
                + (m.kurtosis - 1) / 4 * m.sharpe ** 2)
    if not (denom_sq > 0) or not math.isfinite(denom_sq):
        # The higher moments are too extreme for the Gaussian approximation.
        # None rather than a number computed from a formula outside its domain.
        return None
    z = (m.sharpe - benchmark) * math.sqrt(m.n - 1) / math.sqrt(denom_sq)
    return _norm_cdf(z)


def _expected_max_z(n_trials: int) -> float:
    """Standardized expected maximum of n_trials independent N(0,1) draws.

    One trial has no maximum to correct for, and the formula diverges there
    (Phi-inverse of 0), so it is 0.0 by definition rather than by accident.
    """
    if n_trials <= 1:
        return 0.0
    return ((1 - _GAMMA_EM) * _norm_ppf(1 - 1 / n_trials)
            + _GAMMA_EM * _norm_ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe_ratio(m: PooledMoments | None, n_trials: int,
                          var_sharpe: float) -> float | None:
    """PSR against a benchmark raised to what the best of `n_trials` would
    show by luck alone.

    `var_sharpe` is the variance of the per-observation Sharpes ACROSS the
    search, so it is a property of the search and not of one strategy. A
    single backtest that was never searched over passes n_trials=1, where the
    deflation is zero and this reduces exactly to PSR against zero.

    Feed `effective_n_trials`, not a raw candidate count, whenever the
    candidates came from a grammar: specs differing in one threshold are
    nearly the same strategy, and a raw count over-deflates.
    """
    if m is None:
        return None
    if n_trials < 1 or not math.isfinite(var_sharpe) or var_sharpe < 0:
        return None
    sr_star = math.sqrt(var_sharpe) * _expected_max_z(n_trials)
    return probabilistic_sharpe_ratio(m, sr_star)


def min_track_record_length(sharpe: float, target: float, alpha: float,
                            skew: float, kurtosis: float) -> float | None:
    """Observations needed before this Sharpe could be believed at 1 - alpha.

    Inverts PSR for n. The product answer to a blank field: "too short to
    say, and here is how much longer it would have to be" beats an empty cell.
    """
    gap = sharpe - target
    if gap <= 0 or not 0.0 < alpha < 1.0:
        return None
    denom_sq = 1 - skew * sharpe + (kurtosis - 1) / 4 * sharpe ** 2
    if not (denom_sq > 0):
        return None
    return 1.0 + math.ceil((_norm_ppf(1 - alpha) ** 2) * denom_sq / gap ** 2)


def effective_n_trials(trial_sharpes) -> int:
    """Independent-equivalent trial count, from the autocorrelation of the
    trial Sharpes.

    The house answer to "how many things did we really try". Grammar-generated
    candidates are correlated by construction, so the raw count is an
    overstatement and feeding it to `deflated_sharpe_ratio` would reject real
    edges. n / (1 + 2 * sum of positive autocorrelations), truncated at the
    first non-positive lag.

    Never above the raw count and never below 1: both are guarded rather than
    assumed, because either violation would silently corrupt a deflation.
    """
    values = [float(x) for x in trial_sharpes if math.isfinite(float(x))]
    n = len(values)
    if n < 2:
        return 1
    mean = sum(values) / n
    dev = [v - mean for v in values]
    var = sum(d * d for d in dev) / n
    if var <= 0:
        return 1
    total = 0.0
    for lag in range(1, n):
        rho = sum(dev[i] * dev[i + lag] for i in range(n - lag)) / (n * var)
        if rho <= 0:
            break
        total += rho
    n_eff = n / (1.0 + 2.0 * total)
    return max(1, min(n, int(math.ceil(n_eff))))
