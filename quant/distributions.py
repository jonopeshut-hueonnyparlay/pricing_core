"""Discrete and continuous distribution PMFs/CDFs used for prop win probabilities.

Pure functions — stdlib ``math`` only. Extracted verbatim from run_picks.py (the
canonical implementations); sgp_builder.py and mlb_sgp_builder.py previously kept
private copies of the same math, now consolidated here.
"""
import math


def poisson_pmf(k, lam):
    """Poisson PMF: P(X = k) given lambda."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k, lam):
    """Poisson CDF: P(X <= k)."""
    if lam <= 0:
        return 1.0
    total = 0.0
    for i in range(int(k) + 1):
        total += poisson_pmf(i, lam)
    return min(total, 1.0)


def normal_cdf(x, mu, sigma):
    """Normal CDF using math.erf."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def negbinom_pmf(k, mu, r):
    """Negative binomial PMF: P(X = k) with mean=mu, dispersion=r.

    Parameterisation: p = r/(r+mu), n = r (number of successes).
    P(X=k) = C(k+r-1, k) * p^r * (1-p)^k.
    Uses log-space arithmetic to avoid overflow at large k.
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if r <= 0:
        raise ValueError("negbinom_pmf: r must be > 0")
    k = int(k)
    if k < 0:
        return 0.0
    p = r / (r + mu)
    # log PMF = lgamma(k+r) - lgamma(r) - lgamma(k+1) + r*log(p) + k*log(1-p)
    log_pmf = (
        math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
        + r * math.log(p)
        + k * math.log(1.0 - p)
    )
    return math.exp(log_pmf)


def negbinom_cdf(k, mu, r):
    """Negative binomial CDF: P(X <= k) with mean=mu, dispersion=r."""
    if mu <= 0:
        return 1.0
    total = 0.0
    for i in range(int(k) + 1):
        total += negbinom_pmf(i, mu, r)
    return min(total, 1.0)
