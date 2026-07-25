"""Direct unit tests for quant/odds.py -- American-odds/probability conversion
and vig removal. Backs essentially every JonnyParlay pick and SGP leg, but had
zero direct test coverage before this (indirect only, via JonnyParlay's own
test_quant_properties.py). Pure functions, no I/O -- tests are deterministic.

Run from C:\\Dev\\pricing_core:
    python -m pytest tests/test_quant_odds.py -q
"""
import math

import pytest

from quant.odds import (
    american_to_decimal,
    decimal_to_american,
    implied_prob,
    implied_prob_or_none,
    is_decimal_leak,
    no_vig,
    prob_to_american,
)


# ---- implied_prob -----------------------------------------------------------

def test_implied_prob_known_values():
    assert math.isclose(implied_prob(-110), 110 / 210, rel_tol=1e-12)
    assert math.isclose(implied_prob(150), 100 / 250, rel_tol=1e-12)
    assert math.isclose(implied_prob(-200), 200 / 300, rel_tol=1e-12)


def test_implied_prob_symmetric_at_100():
    # -100 and +100 are both "even money" -- both must give exactly 0.5.
    assert implied_prob(-100) == 0.5
    assert implied_prob(100) == 0.5


def test_implied_prob_zero_returns_zero():
    assert implied_prob(0) == 0.0


def test_implied_prob_extreme_favorite_and_underdog_stay_in_bounds():
    huge_favorite = implied_prob(-10000)
    huge_dog = implied_prob(10000)
    assert math.isclose(huge_favorite, 10000 / 10100, rel_tol=1e-12)
    assert math.isclose(huge_dog, 100 / 10100, rel_tol=1e-12)
    assert 0.0 < huge_dog < huge_favorite < 1.0


# ---- implied_prob_or_none ----------------------------------------------------

def test_implied_prob_or_none_delegates_exactly_to_implied_prob():
    # Docstring's own claim (audit P0.6): single source of truth for the formula.
    for odds in (-110, 150, -100, 100, -10000, 10000):
        assert implied_prob_or_none(odds) == implied_prob(odds)
        assert implied_prob_or_none(float(odds)) == implied_prob(odds)
        assert implied_prob_or_none(str(odds)) == implied_prob(odds)


@pytest.mark.parametrize("bad", [None, "abc", "", object(), [1, 2]])
def test_implied_prob_or_none_unusable_type_returns_none(bad):
    assert implied_prob_or_none(bad) is None


@pytest.mark.parametrize("bad", [0, 0.0, "0", float("nan"), float("inf"), float("-inf")])
def test_implied_prob_or_none_unusable_numeric_returns_none(bad):
    assert implied_prob_or_none(bad) is None


# ---- no_vig -------------------------------------------------------------------

def test_no_vig_normalizes_to_sum_one():
    p1, p2 = no_vig(0.55, 0.55)
    assert math.isclose(p1 + p2, 1.0, rel_tol=1e-12)
    assert p1 == p2 == 0.5


def test_no_vig_preserves_ratio():
    imp1, imp2 = 0.6, 0.3
    p1, p2 = no_vig(imp1, imp2)
    assert math.isclose(p1 + p2, 1.0, rel_tol=1e-12)
    assert math.isclose(p1 / p2, imp1 / imp2, rel_tol=1e-9)


def test_no_vig_already_fair_is_unchanged():
    p1, p2 = no_vig(0.4, 0.6)
    assert math.isclose(p1, 0.4, rel_tol=1e-12)
    assert math.isclose(p2, 0.6, rel_tol=1e-12)


def test_no_vig_degenerate_zero_total_returns_half_half():
    assert no_vig(0.0, 0.0) == (0.5, 0.5)


# ---- is_decimal_leak ----------------------------------------------------------

@pytest.mark.parametrize("decimal_odds", [1.91, 1.5, 2.0, 2.49])
def test_is_decimal_leak_detects_decimal_range(decimal_odds):
    assert is_decimal_leak(decimal_odds) is True


@pytest.mark.parametrize("boundary", [1.0, 2.5])
def test_is_decimal_leak_boundaries_are_exclusive(boundary):
    assert is_decimal_leak(boundary) is False


@pytest.mark.parametrize("american_odds", [-110, 150, -100, 100, 3.0])
def test_is_decimal_leak_rejects_real_american_odds(american_odds):
    assert is_decimal_leak(american_odds) is False


# ---- prob_to_american -----------------------------------------------------

def test_prob_to_american_evens_at_half():
    assert prob_to_american(0.5) == -100.0


def test_prob_to_american_favorite_is_negative():
    # p=0.6 -> favorite -> negative American odds
    assert math.isclose(prob_to_american(0.6), -150.0, rel_tol=1e-9)


def test_prob_to_american_underdog_is_positive():
    # p=0.4 -> underdog -> positive American odds
    assert math.isclose(prob_to_american(0.4), 150.0, rel_tol=1e-9)


@pytest.mark.parametrize("out_of_domain", [0.0, -0.1, 1.0, 1.5])
def test_prob_to_american_out_of_domain_returns_zero(out_of_domain):
    assert prob_to_american(out_of_domain) == 0


def test_prob_to_american_round_trips_implied_prob():
    for odds in (-150, -110, -101, 120, 150, 300):
        p = implied_prob(odds)
        assert math.isclose(prob_to_american(p), float(odds), rel_tol=1e-6)


# ---- american_to_decimal / decimal_to_american -----------------------------

def test_american_to_decimal_known_values():
    assert math.isclose(american_to_decimal(150), 2.5, rel_tol=1e-12)
    assert math.isclose(american_to_decimal(-150), 1 + 100 / 150, rel_tol=1e-12)


def test_american_to_decimal_symmetric_at_100():
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0


def test_american_to_decimal_zero_raises_zero_division():
    # 0 is not a valid American-odds value (valid odds are always <=-100 or
    # >=+100), so this documents current fail-fast behavior on out-of-domain
    # input rather than a bug to fix -- the else branch divides by abs(0).
    with pytest.raises(ZeroDivisionError):
        american_to_decimal(0)


def test_decimal_to_american_known_values():
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.6667) == -150  # round(-100/0.6667) == -150


def test_decimal_to_american_boundary_at_two():
    assert decimal_to_american(2.0) == 100


def test_decimal_to_american_one_raises_zero_division():
    # decimal odds of 1.0 (100% implied probability) hits the <2.0 branch's
    # division by (dec - 1) == 0 -- documents current behavior, not a fix.
    with pytest.raises(ZeroDivisionError):
        decimal_to_american(1.0)


def test_american_to_decimal_and_back_round_trip():
    for odds in (-300, -150, -110, -101, 101, 110, 150, 300):
        dec = american_to_decimal(odds)
        assert decimal_to_american(dec) == odds
