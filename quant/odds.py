"""American-odds / probability conversions and vig removal.

Pure functions (no stdlib deps beyond arithmetic). Canonical home for the odds
math that was previously duplicated across run_picks.py and the SGP builders.

The vig math lives ONLY here. capture_clv.py and clv_report.py consume the
None-guarding wrapper implied_prob_or_none() (audit P0.6) instead of their own
copies, so a change to the formula propagates to the CLV ledger automatically.
"""
import math


def implied_prob(odds):
    """American odds → implied probability. Assumes a valid numeric `odds`
    (returns 0.0 for 0). For untrusted input (CSV/API strings, NaN) use
    implied_prob_or_none()."""
    if odds == 0:
        return 0.0
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def implied_prob_or_none(american_odds):
    """American odds → implied probability, or None for unusable input.

    The hardened variant used by the CLV path (capture_clv.py, clv_report.py):
    coerces to float and returns None for 0, NaN, inf, or non-numeric values so
    callers skip the row instead of writing corrupted CLV (audit C6). The actual
    formula is delegated to implied_prob() — single source of truth (audit P0.6).
    """
    try:
        o = float(american_odds)
    except (TypeError, ValueError):
        return None
    if not o or not math.isfinite(o):
        return None
    return implied_prob(o)


def no_vig(imp1, imp2):
    """Remove vig from two-sided implied probs."""
    total = imp1 + imp2
    if total == 0:
        return 0.5, 0.5
    return imp1 / total, imp2 / total


def is_decimal_leak(odds):
    """Check if odds look like decimal format leaked through.
    Range 1.0 < odds < 2.5 catches decimal (e.g. 1.91 for -110).
    Upper bound is 2.5 (not 3.0) to avoid rejecting valid +100 to +149 American odds,
    whose decimal equivalents are 2.0–2.49.
    """
    return 1.0 < odds < 2.5


def prob_to_american(prob):
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return -(prob / (1.0 - prob)) * 100
    else:
        return ((1.0 - prob) / prob) * 100


def american_to_decimal(odds):
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def decimal_to_american(dec):
    if dec >= 2.0:
        return int(round((dec - 1) * 100))
    else:
        return int(round(-100 / (dec - 1)))
