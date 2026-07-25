"""Direct unit tests for quant/copula.py -- t-copula joint-probability math for
same-game parlays. Backs every JonnyParlay SGP price, but had zero direct test
coverage before this (indirect only, via JonnyParlay's own
test_quant_properties.py). math.erfinv is unavailable on this Python (3.13.14
here), so probit() always exercises its Beasley-Springer-Moro fallback path in
this environment -- verified empirically, not assumed.

Run from C:\\Dev\\pricing_core:
    python -m pytest tests/test_quant_copula.py -q
"""
import math

import pytest
from scipy.stats import norm

from quant.copula import (
    cholesky,
    copula_joint_approx,
    copula_joint_prob,
    probit,
    validate_corr_matrix,
)


# ---- probit -------------------------------------------------------------------

def test_probit_median_is_zero():
    assert probit(0.5) == 0.0


def test_probit_matches_scipy_normal_ppf_within_bsm_tolerance():
    # Docstring's own stated max error for the BSM fallback: ~4.5e-4.
    for p in (0.025, 0.1587, 0.3, 0.7, 0.8413, 0.975, 0.99):
        assert math.isclose(probit(p), norm.ppf(p), abs_tol=5e-4)


def test_probit_is_antisymmetric_around_half():
    for p in (0.1, 0.25, 0.4, 0.6, 0.75, 0.9):
        assert math.isclose(probit(p), -probit(1.0 - p), abs_tol=1e-9)


def test_probit_clamps_extreme_inputs_to_finite():
    # p<=0 and p>=1 are clamped to (1e-9, 1-1e-9) internally rather than
    # producing +-inf or raising on log(0)/erfinv(+-1).
    lo = probit(0.0)
    hi = probit(1.0)
    below = probit(-5.0)
    above = probit(5.0)
    assert math.isfinite(lo) and math.isfinite(hi)
    assert math.isfinite(below) and math.isfinite(above)
    assert lo == below  # both clamp to the same floor
    assert hi == above  # both clamp to the same ceiling
    assert lo < -5 and hi > 5  # still meaningfully extreme, not zeroed out


# ---- validate_corr_matrix -----------------------------------------------------

def test_validate_corr_matrix_identity_is_ok():
    assert validate_corr_matrix([[1.0]]) == (True, "ok")
    assert validate_corr_matrix([[1.0, 0.0], [0.0, 1.0]]) == (True, "ok")


def test_validate_corr_matrix_realistic_matrix_is_ok():
    ok, reason = validate_corr_matrix([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])
    assert ok is True and reason == "ok"


def test_validate_corr_matrix_empty_is_rejected():
    assert validate_corr_matrix([]) == (False, "empty matrix")


def test_validate_corr_matrix_non_square_is_rejected():
    ok, reason = validate_corr_matrix([[1.0, 0.5], [0.5, 1.0, 0.1]])
    assert ok is False
    assert "not square" in reason


def test_validate_corr_matrix_non_unit_diagonal_is_rejected():
    ok, reason = validate_corr_matrix([[0.9, 0.0], [0.0, 1.0]])
    assert ok is False
    assert "diagonal" in reason


def test_validate_corr_matrix_out_of_range_entry_is_rejected():
    ok, reason = validate_corr_matrix([[1.0, 1.5], [1.5, 1.0]])
    assert ok is False
    assert "outside [-1, 1]" in reason


def test_validate_corr_matrix_asymmetric_is_rejected():
    ok, reason = validate_corr_matrix([[1.0, 0.3], [0.5, 1.0]])
    assert ok is False
    assert "asymmetric" in reason


def test_validate_corr_matrix_non_psd_is_rejected():
    # Classic non-PSD example: three pairwise correlations that are each
    # individually in [-1, 1] but cannot jointly hold (triangle inequality
    # violation on correlations).
    mat = [
        [1.0, 0.9, -0.9],
        [0.9, 1.0, 0.9],
        [-0.9, 0.9, 1.0],
    ]
    ok, reason = validate_corr_matrix(mat)
    assert ok is False
    assert "not positive semi-definite" in reason


def test_validate_corr_matrix_boundary_entries_at_plus_minus_one_are_allowed():
    # Perfect correlation / anti-correlation are valid boundary values, not
    # "out of range" -- distinct from the non-PSD 3x3 case above.
    ok, reason = validate_corr_matrix([[1.0, 1.0], [1.0, 1.0]])
    assert ok is True and reason == "ok"
    ok, reason = validate_corr_matrix([[1.0, -1.0], [-1.0, 1.0]])
    assert ok is True and reason == "ok"


# ---- cholesky -----------------------------------------------------------------

def test_cholesky_identity_is_identity():
    L = cholesky([[1.0, 0.0], [0.0, 1.0]])
    assert L == [[1.0, 0.0], [0.0, 1.0]]


def test_cholesky_reconstructs_original_matrix():
    mat = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]]
    L = cholesky(mat)
    n = len(mat)
    for i in range(n):
        for j in range(n):
            reconstructed = sum(L[i][k] * L[j][k] for k in range(n))
            assert math.isclose(reconstructed, mat[i][j], abs_tol=1e-9)


def test_cholesky_near_singular_does_not_raise():
    # Perfectly correlated (rank-deficient) matrix -- the second pivot would
    # be exactly zero without the max(d, 0.0) clip.
    L = cholesky([[1.0, 1.0], [1.0, 1.0]])
    assert math.isfinite(L[1][1])
    assert L[1][1] >= 0.0


# ---- copula_joint_prob ---------------------------------------------------------

def test_copula_joint_prob_empty_returns_zero():
    assert copula_joint_prob([], [[1.0]]) == 0.0


def test_copula_joint_prob_single_leg_returns_prob_directly_bypassing_corr():
    # n==1 short-circuits before touching corr_mat at all.
    assert copula_joint_prob([0.73], [[999.0]]) == 0.73


def test_copula_joint_prob_deterministic_given_fixed_seed():
    probs = [0.65, 0.70, 0.60]
    corr = [[1.0, 0.2, 0.1], [0.2, 1.0, 0.15], [0.1, 0.15, 1.0]]
    r1 = copula_joint_prob(probs, corr, n_samples=2000, seed=42)
    r2 = copula_joint_prob(probs, corr, n_samples=2000, seed=42)
    assert r1 == r2  # bit-identical per the module's own documented contract


def test_copula_joint_prob_zero_correlation_approximates_independence():
    probs = [0.7, 0.6, 0.65]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    result = copula_joint_prob(probs, identity, n_samples=10_000, seed=42)
    independent = probs[0] * probs[1] * probs[2]
    # SE ~0.3% at 10k samples per the module's own docstring; allow generous
    # slack for t-copula (df=6) tail dependence still present even at rho=0.
    assert math.isclose(result, independent, abs_tol=0.03)


def test_copula_joint_prob_cholesky_failure_falls_back_to_independence():
    # The try/except only wraps the cholesky(corr_mat) call itself. cholesky()
    # has no explicit raise and clips negative/degenerate diagonals, so
    # neither NaN entries nor plain dimension mismatches make it raise
    # (verified empirically) -- only genuinely non-numeric entries do (a
    # TypeError from the subtraction/sqrt arithmetic). This exercises the
    # documented fallback path for the failure mode it actually guards.
    probs = [0.5, 0.6, 0.7]
    non_numeric_corr = [[1.0, "bad", 0.2], ["bad", 1.0, 0.1], [0.2, 0.1, 1.0]]
    result = copula_joint_prob(probs, non_numeric_corr, n_samples=100, seed=1)
    expected = 0.5 * 0.6 * 0.7
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_copula_joint_prob_nan_corr_entries_do_not_trigger_fallback():
    # Documents a real numerical-stability gap discovered while testing:
    # NaN entries pass silently through cholesky() (NaN comparisons are
    # always False, so the `d < -tol` guard never fires) and propagate into
    # the sampling loop instead of triggering the except-block fallback.
    # Not fixed here -- out of scope for this stage (see final report).
    probs = [0.5, 0.6, 0.7]
    nan_corr = [[1.0, float("nan"), 0.2], [float("nan"), 1.0, 0.1], [0.2, 0.1, 1.0]]
    result = copula_joint_prob(probs, nan_corr, n_samples=100, seed=1)
    independent = 0.5 * 0.6 * 0.7
    # Not close to the independence fallback -- confirms this silently takes
    # a different, NaN-poisoned code path instead.
    assert not math.isclose(result, independent, rel_tol=1e-9)


def test_copula_joint_prob_dimension_mismatch_raises_index_error():
    # NOT part of the documented cholesky-failure fallback: a corr_mat
    # smaller than probs passes cholesky() fine (it only looks at corr_mat's
    # own shape), then raises IndexError later in the sampling loop when it
    # indexes L using range(len(probs)). Discovered while writing this test
    # -- documented here as current behavior, not fixed (out of scope for
    # this stage: it's a caller-contract gap between probs and corr_mat
    # dimensions, not a math-correctness bug in this module).
    probs = [0.5, 0.6, 0.7]
    undersized_corr = [[1.0, 0.2], [0.2, 1.0]]
    with pytest.raises(IndexError):
        copula_joint_prob(probs, undersized_corr, n_samples=100, seed=1)


def test_copula_joint_prob_boundary_probs_near_zero_and_one():
    identity = [[1.0, 0.0], [0.0, 1.0]]
    near_zero = copula_joint_prob([0.0, 0.0], identity, n_samples=2000, seed=42)
    near_one = copula_joint_prob([1.0, 1.0], identity, n_samples=2000, seed=42)
    assert near_zero < 0.01
    assert near_one > 0.99


# ---- copula_joint_approx -------------------------------------------------------

def test_copula_joint_approx_zero_rho_is_independence_deflated():
    probs = [0.7, 0.6, 0.8]
    p_indep = probs[0] * probs[1] * probs[2]
    assert math.isclose(copula_joint_approx(probs, 0.0), p_indep * 0.87, rel_tol=1e-12)


def test_copula_joint_approx_rho_one_is_min_prob_deflated():
    probs = [0.7, 0.6, 0.8]
    assert math.isclose(copula_joint_approx(probs, 1.0), min(probs) * 0.87, rel_tol=1e-12)


def test_copula_joint_approx_midpoint_matches_linear_interpolation_formula():
    probs = [0.7, 0.5, 0.9]
    avg_rho = 0.3
    p_indep = probs[0] * probs[1] * probs[2]
    p_min = min(probs)
    expected = (p_indep + avg_rho * (p_min - p_indep)) * 0.87
    assert math.isclose(copula_joint_approx(probs, avg_rho), expected, rel_tol=1e-12)


def test_copula_joint_approx_single_leg_is_prob_deflated_regardless_of_rho():
    for avg_rho in (0.0, 0.5, 1.0):
        assert math.isclose(copula_joint_approx([0.65], avg_rho), 0.65 * 0.87, rel_tol=1e-12)


def test_copula_joint_approx_empty_probs_raises_value_error():
    # min([]) raises -- documents current behavior on an empty-legs input
    # (which no real SGP call site produces), not a fix.
    with pytest.raises(ValueError):
        copula_joint_approx([], 0.5)
