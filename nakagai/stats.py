"""Statistical math for backtest results.

No evidence store, no workspace, no config: everything is parameterized.
"""

import math

import pandas as pd

PF_CLAMP = 1000.0   # a ledger or resample with no losers: "infinite" PF


def _norm_cdf(z: float) -> float:
    """Standard normal CDF, from math.erf.

    Core carries four dependencies and scipy is not among them. Phi is a
    two-line identity on erf, which the standard library has, so importing a
    numerical stack for it would be a poor trade."""
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
