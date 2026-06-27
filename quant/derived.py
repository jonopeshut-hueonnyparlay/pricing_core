"""Derived probability calculations composed from the distribution + odds primitives.

Pure functions. Extracted verbatim from run_picks.py:
  - mlb_ml_from_nb : moneyline win prob via discrete NB run-total convolution
  - calc_tb_prob   : total-bases over/under via exact Poisson convolution
  - calc_edge      : no-vig edge for both sides of a two-way market
"""
import math

from .distributions import negbinom_pmf, negbinom_cdf, poisson_pmf
from .odds import implied_prob, no_vig


def mlb_ml_from_nb(mu_home, mu_away, r):
    """P(home wins) via direct NB probability sum over discrete run totals.

    More accurate than Normal(margin, sigma) for MLB because run-scoring is
    discrete and overdispersed (var/mu~2.26). Ties (extra innings) treated
    as 50/50 split. Sum to 30 runs per team covers >99.99% of probability mass.
    """
    if mu_home <= 0 or mu_away <= 0:
        return 0.5
    home_wp = 0.0
    for k in range(31):
        ph = negbinom_pmf(k, mu_home, r)
        pa_lt = negbinom_cdf(k - 1, mu_away, r) if k > 0 else 0.0
        pa_eq = negbinom_pmf(k, mu_away, r)
        home_wp += ph * (pa_lt + 0.5 * pa_eq)
    return min(max(home_wp, 0.0), 1.0)


def calc_tb_prob(singles: float, doubles: float, triples: float, hr: float, line: float):
    """Discrete total-bases probability via Poisson convolution.

    Models each hit type as an independent Poisson process (lambda = projected
    count per game) and computes P(TB > line) by convolving the distributions
    exactly. Replaces Normal(mean_TB, sigma) which misrepresents the discrete,
    zero-inflated, right-skewed nature of total bases — the Normal model was
    predicting ~56% for O1.5 when the empirical rate is 35-38%.

    SaberSim provides 1B/2B/3B/HR separately; this function uses all four.
    """
    max_tb = 16  # ceiling: 4 HR = 16 TB
    threshold = int(math.floor(line)) + 1  # P(TB >= threshold) for half-integer lines

    dist = [0.0] * (max_tb + 1)
    dist[0] = 1.0  # start with P(0 TB) = 1

    for lam, weight in ((singles, 1), (doubles, 2), (triples, 3), (hr, 4)):
        if lam <= 0:
            continue
        new_dist = [0.0] * (max_tb + 1)
        max_count = max(8, int(lam * 5))
        for count in range(max_count + 1):
            pmf = poisson_pmf(count, lam)
            if pmf < 1e-9:
                continue
            added = count * weight
            for tb in range(max_tb + 1):
                target = min(tb + added, max_tb)
                new_dist[target] += dist[tb] * pmf
        dist = new_dist

    over_p = sum(dist[threshold:])
    under_p = 1.0 - over_p
    return over_p, under_p


def calc_edge(model_prob, over_odds, under_odds):
    """Calculate no-vig edge for both sides. Returns (over_edge, under_edge)."""
    imp_over = implied_prob(over_odds)
    imp_under = implied_prob(under_odds)
    nv_over, nv_under = no_vig(imp_over, imp_under)
    # Convention: model_prob is interpreted as the probability of the over.
    over_edge = model_prob - nv_over
    under_edge = (1.0 - model_prob) - nv_under
    return over_edge, under_edge, nv_over, nv_under
