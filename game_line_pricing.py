"""Canonical game-line pricing engine.

ONE home for the per-market probability math shared by the live game-line
analyzer (analyze_game_lines.py) and the prop-correlation path
(engine/evaluators.py). Built incrementally during the Stage 1 pricer collapse.

Leaf distribution math comes from quant.* (already canonical). This module owns
the COMPOSED game-line pricing. First in: the push-adjusted MLB team-total NB,
which was previously duplicated verbatim across both callers.

Pure functions — no I/O. Market anchoring (the BLEND_ALPHA blend toward the
market) is NOT applied here yet; callers still compute the projection they want
priced. Anchoring becomes a mandatory parameter when the Normal-market pricing
moves here in Stage 1 Commit 2 (the behavioral, approval-gated change).
"""
import math

from quant.distributions import normal_cdf, negbinom_pmf, negbinom_cdf
from quant.derived import mlb_ml_from_nb


def blend(projection, market_anchor, trust):
    """Blend a projection toward a market anchor.

    ``trust`` is the weight on the projection vs the market anchor:
      * trust == 1.0  -> the raw projection, returned EXACTLY (no float drift, so
        callers that price off the raw projection stay byte-identical);
      * trust  < 1.0  -> ``anchor + trust*(projection - anchor)`` (pull toward the
        market). evaluators uses trust = BLEND_ALPHA = 0.25.

    This is the single, mandatory market-anchoring step for game lines. Routing the
    raw analyzer through here at trust=1.0 is a no-op; lowering it to 0.25 is the
    behavioural CLV fix (approval-gated).
    """
    if trust == 1.0:
        return projection
    return market_anchor + trust * (projection - market_anchor)


def prob_total_over(total_proj, line, sigma, *, trust):
    """P(total > line) for a Normal market. Anchor = the market line."""
    mu = blend(total_proj, line, trust)
    return 1.0 - normal_cdf(line, mu, sigma)


def prob_spread_cover(raw_margin, market_margin, sp_line, sigma, *, is_home, trust):
    """P(team covers ``sp_line``). Anchor = market-implied margin (home perspective)."""
    mu = blend(raw_margin, market_margin, trust)
    team_margin = mu if is_home else -mu
    return 1.0 - normal_cdf(-sp_line, team_margin, sigma)


def prob_ml_normal(raw_margin, market_margin, sigma, *, is_home, trust):
    """P(team wins outright) for a variable-spread sport. Anchor = market-implied margin."""
    mu = blend(raw_margin, market_margin, trust)
    team_margin = mu if is_home else -mu
    return 1.0 - normal_cdf(0.0, team_margin, sigma)


def prob_ml_mlb_nb(mu_home, mu_away, r, nv_this, *, is_home, trust):
    """P(team wins) for MLB via NB run-total convolution. Anchor = no-vig market prob.

    Unlike the Normal MLs, the anchor here is in PROBABILITY space (the no-vig price),
    matching evaluators: win_prob = nv + trust*(raw_team_wp - nv).
    """
    raw_home_wp = mlb_ml_from_nb(mu_home, mu_away, r)
    raw_team_wp = raw_home_wp if is_home else (1.0 - raw_home_wp)
    return blend(raw_team_wp, nv_this, trust)


def prob_team_total_normal(proj, line, sigma, *, trust):
    """P(team total > line) for a Normal (non-MLB) team total. Anchor = the line."""
    mu = blend(proj, line, trust)
    return 1.0 - normal_cdf(line, mu, sigma)


def team_total_mlb_nb(mu, line, r):
    """P(over), P(under) for an MLB team total via Negative Binomial.

    Push-adjusted on integer lines (the probability mass on exactly ``line`` is
    removed from both sides); a half-line is a plain split. Byte-identical to the
    formula previously duplicated in ``analyze_game_lines.mlb_tt_prob`` and inline
    in ``evaluators.evaluate_game_lines``.

    ``r`` is the NB dispersion — pass the per-team value where one exists, with the
    global league ``r`` only as a fallback (see feedback: prefer per-entity params).
    Returns ``(over_p, under_p)``; both 0.5 on a degenerate all-push line.
    """
    k_floor = int(math.floor(line))
    if line == k_floor:  # integer line — push-adjusted
        push = negbinom_pmf(k_floor, mu, r)
        non_push = 1.0 - push
        if non_push <= 0:
            return 0.5, 0.5
        over_p = (1.0 - negbinom_cdf(k_floor, mu, r)) / non_push
        under_p = negbinom_cdf(k_floor - 1, mu, r) / non_push
    else:  # half-line
        over_p = 1.0 - negbinom_cdf(k_floor, mu, r)
        under_p = negbinom_cdf(k_floor, mu, r)
    return over_p, under_p
