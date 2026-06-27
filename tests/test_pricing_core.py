"""Standalone tests for pricing_core -- verify the shared library on its own,
independent of either consuming repo.

Covers the distribution primitives (quant), the game-line engine
(game_line_pricing) including the market-anchoring `trust` contract, and the
package version. Pure/offline.

Run from C:\\Dev\\pricing_core:
    python -m pytest -q
"""
import math

import pricing_core
from quant.distributions import normal_cdf, negbinom_pmf, negbinom_cdf, poisson_pmf, poisson_cdf
from quant.derived import mlb_ml_from_nb
from game_line_pricing import (
    blend, team_total_mlb_nb,
    prob_total_over, prob_spread_cover, prob_ml_normal, prob_ml_mlb_nb, prob_team_total_normal,
)

_ALPHA = 0.25  # the consumers' BLEND_ALPHA


def test_version_exposed():
    assert isinstance(pricing_core.__version__, str) and pricing_core.__version__


# ---- distributions ---------------------------------------------------------
def test_normal_cdf_known_and_degenerate():
    assert normal_cdf(0.0, 0.0, 1.0) == 0.5
    assert math.isclose(normal_cdf(1.0, 0.0, 1.0), 0.8413447460685429, rel_tol=1e-12)
    assert normal_cdf(5.0, 0.0, 0.0) == 1.0   # sigma<=0 degenerate
    assert normal_cdf(-5.0, 0.0, 0.0) == 0.0


def test_negbinom_pmf_sums_to_cdf():
    mu, r = 4.4, 3.548
    assert math.isclose(negbinom_cdf(10, mu, r), sum(negbinom_pmf(k, mu, r) for k in range(11)), rel_tol=1e-12)
    assert negbinom_pmf(0, 0.0, r) == 1.0     # mu<=0 -> all mass at 0


def test_poisson_pmf_cdf_consistent():
    lam = 3.0
    assert math.isclose(poisson_cdf(8, lam), sum(poisson_pmf(k, lam) for k in range(9)), rel_tol=1e-12)


def test_mlb_ml_from_nb_symmetry_and_bounds():
    r = 3.548
    # Equal projections -> ~coin flip (not exactly 0.5: the convolution truncates
    # at 30 runs/team, leaving ~4e-6 of tail mass uncounted).
    assert math.isclose(mlb_ml_from_nb(4.5, 4.5, r), 0.5, abs_tol=1e-4)
    p = mlb_ml_from_nb(5.0, 4.0, r)
    assert 0.5 < p < 1.0                                  # stronger home favored
    assert math.isclose(p + mlb_ml_from_nb(4.0, 5.0, r), 1.0, abs_tol=1e-4)


# ---- game-line engine: the anchoring `trust` contract ----------------------
def test_blend_trust_one_is_exact_projection():
    for proj, anchor in [(4.4, 8.0), (9.0, 8.5), (-1.5, 0.0)]:
        assert blend(proj, anchor, 1.0) == proj          # bit-exact, no anchor drift
        assert blend(proj, anchor, _ALPHA) == anchor + _ALPHA * (proj - anchor)


def test_prob_total_over_trust_endpoints():
    proj, line, sigma = 9.0, 8.5, 4.6
    assert prob_total_over(proj, line, sigma, trust=1.0) == 1.0 - normal_cdf(line, proj, sigma)
    blended = line + _ALPHA * (proj - line)
    assert prob_total_over(proj, line, sigma, trust=_ALPHA) == 1.0 - normal_cdf(line, blended, sigma)


def test_prob_ml_mlb_nb_anchors_to_novig():
    r, nv = 3.548, 0.52
    raw = mlb_ml_from_nb(4.4, 4.6, r)
    assert prob_ml_mlb_nb(4.4, 4.6, r, nv, is_home=True, trust=1.0) == raw
    assert prob_ml_mlb_nb(4.4, 4.6, r, nv, is_home=True, trust=_ALPHA) == nv + _ALPHA * (raw - nv)


def test_prob_spread_and_ml_normal_trust_one():
    assert prob_spread_cover(-0.2, 1.5, -1.5, 4.2, is_home=True, trust=1.0) == 1.0 - normal_cdf(1.5, -0.2, 4.2)
    assert prob_ml_normal(4.0, -3.5, 12.5, is_home=True, trust=1.0) == 1.0 - normal_cdf(0.0, 4.0, 12.5)
    assert prob_team_total_normal(110.0, 105.5, 11.0, trust=1.0) == 1.0 - normal_cdf(105.5, 110.0, 11.0)


def test_team_total_mlb_nb_push_adjusted_and_halfline():
    # half-line: no push -> over + under == 1 exactly
    over_p, under_p = team_total_mlb_nb(4.4, 4.5, 3.548)
    assert over_p + under_p == 1.0
    # integer line: push-adjusted, both sides in [0,1]
    o, u = team_total_mlb_nb(4.15, 4.0, 3.258)
    assert 0.0 <= o <= 1.0 and 0.0 <= u <= 1.0
    # degenerate all-push -> (0.5, 0.5)
    assert team_total_mlb_nb(0.0, 0.0, 3.548) == (0.5, 0.5)
