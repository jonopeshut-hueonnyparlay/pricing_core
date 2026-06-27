"""t-copula joint-probability math for same-game parlays.

stdlib ``math`` + ``random`` for everything except ``copula_joint_prob``, which
also uses ``scipy.stats.t`` for the inverse-t-CDF (t-copula marginals). Extracted
from sgp_builder.py (the canonical implementation). The Monte-Carlo sampler builds
its RNG inside the function from the ``seed`` argument (default 42), so output is
bit-identical across runs and machines for a given (probs, corr_mat, n_samples, seed).

The domain-specific correlation lookup (``_pairwise_rho``) and matrix builder
(``_build_corr_matrix``) deliberately stay in sgp_builder.py — they encode NBA
business logic, not generic math.
"""
import math
import random

from scipy.stats import t as _scipy_t

# t-copula degrees of freedom. ν=6 per Demarta & McNeil 2005 (midpoint of the
# research-backed 4–8 range). Lower ν = heavier tails = more joint-extreme
# probability. June 2026: replaces the Gaussian copula (zero tail dependence,
# λ_U=λ_L=0 for any ρ<1).
COPULA_DF = 6


def probit(p):
    """Standard normal quantile function Φ^{-1}(p).

    Uses math.erfinv when available (Python ≥ 3.12); otherwise falls back to
    the Beasley-Springer-Moro rational approximation (max error ≈ 4.5e-4).
    """
    p = max(1e-9, min(1.0 - 1e-9, p))
    try:
        return math.sqrt(2.0) * math.erfinv(2.0 * p - 1.0)
    except AttributeError:
        # BSM coefficients
        _a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
        _b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
        _c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
              0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
              0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
        y = p - 0.5
        if abs(y) < 0.42:
            r = y * y
            return y * ((((_a[3]*r + _a[2])*r + _a[1])*r + _a[0])
                        / ((((_b[3]*r + _b[2])*r + _b[1])*r + _b[0])*r + 1.0))
        r = p if y < 0 else 1.0 - p
        r = math.log(-math.log(r))
        x = _c[0] + r*(_c[1] + r*(_c[2] + r*(_c[3] + r*(_c[4]
              + r*(_c[5] + r*(_c[6] + r*(_c[7] + r*_c[8])))))))
        return -x if y < 0 else x


def validate_corr_matrix(mat, tol=1e-9):
    """Validate a Gaussian-copula correlation matrix; return ``(ok, reason)``.

    Checks, in order: non-empty and square, symmetric (within ``tol``), unit
    diagonal, every entry in ``[-1, 1]``, and positive semi-definite (no Cholesky
    pivot below ``-tol``).

    Hand-assigned pairwise ρ (``_pairwise_rho`` in the SGP builders) can combine
    into a non-PSD matrix even when every individual entry is in range. A non-PSD
    matrix makes the copula sampler produce nonsense — and ``copula_joint_prob``
    silently falls back to the independence product on a Cholesky failure — so a
    malformed ρ table degrades pricing with no error. This is the explicit,
    testable guard used by the SGP builders' load-time invariants.
    """
    n = len(mat)
    if n == 0:
        return False, "empty matrix"
    for i, row in enumerate(mat):
        if len(row) != n:
            return False, f"row {i} length {len(row)} != {n} (not square)"
    for i in range(n):
        if abs(mat[i][i] - 1.0) > tol:
            return False, f"diagonal[{i}]={mat[i][i]} != 1.0 (not a correlation matrix)"
        for j in range(n):
            if not (-1.0 - tol <= mat[i][j] <= 1.0 + tol):
                return False, f"entry[{i}][{j}]={mat[i][j]} outside [-1, 1]"
            if abs(mat[i][j] - mat[j][i]) > tol:
                return False, f"asymmetric: [{i}][{j}]={mat[i][j]} vs [{j}][{i}]={mat[j][i]}"
    # Positive semi-definite via strict Cholesky: any pivot < -tol → not PSD.
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = mat[i][i] - s
                if d < -tol:
                    return False, f"not positive semi-definite (pivot {d:.3e} at index {i})"
                L[i][j] = math.sqrt(max(d, 0.0))
            else:
                L[i][j] = (mat[i][j] - s) / L[j][j] if L[j][j] > tol else 0.0
    return True, "ok"


def cholesky(mat):
    """Lower triangular Cholesky L such that mat = L @ L^T (n ≤ 4).

    Clips near-zero diagonal to avoid sqrt of negative due to floating-point
    rounding on near-singular matrices.
    """
    n = len(mat)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(mat[i][i] - s, 1e-12))
            else:
                L[i][j] = (mat[i][j] - s) / L[j][j] if L[j][j] > 1e-12 else 0.0
    return L


def copula_joint_prob(probs, corr_mat, n_samples=10_000, seed=42,
                      df=COPULA_DF):
    """t-Copula joint probability via Monte Carlo.

    P(all legs hit) accounting for inter-leg correlations.

    Algorithm:
      1. Factorise R = L L^T  (Cholesky — same as Gaussian version)
      2. For each simulation:
         a. Sample chi2(df) = sum of df squared standard normals
         b. Sample z ~ N(0, I_n), then x = L z  (correlated Gaussian)
         c. t_i = x_i / sqrt(chi2/df)  →  t-distributed with df df
         d. Joint hit = all t_i ≤ Φ_t^{-1}(p_i)
      3. Return hit_count / n_samples

    Replaces Gaussian copula (L8, May 2026). Gaussian copula has
    λ_U = λ_L = 0 for any ρ < 1 (zero tail dependence), systematically
    underestimating joint probability when extreme game outcomes make
    legs jointly more likely to hit. t-Copula with ν=6 adds nonzero
    symmetric tail dependence via degrees-of-freedom parameter.
    (Demarta & McNeil 2005; ArbitrageLab Copula Reference; June 2026
    architecture research.)

    n_samples=10,000: SE ≈ 0.3% for joint≈0.10.
    Runtime: ~5-8 ms for 4-leg at 10,000 samples (called once per
    final SGP). Fixed seed gives reproducible output.
    """
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        return probs[0]
    try:
        L = cholesky(corr_mat)
    except Exception:
        result = 1.0
        for p in probs:
            result *= p
        return result

    # Precompute t-distribution thresholds (inverse t-CDF at each leg prob).
    # Clamp to (0, 1) — consistent with probit() — so extreme probs never yield
    # ±inf thresholds. Leg probs are ≥0.62 in practice; this is defensive only.
    thresholds = [float(_scipy_t.ppf(min(1.0 - 1e-9, max(1e-9, p)), df))
                  for p in probs]

    rng = random.Random(seed)
    gauss = rng.gauss
    hits = 0
    for _ in range(n_samples):
        # Sample chi2(df) = sum of df squared standard normals
        chi2 = sum(gauss(0.0, 1.0) ** 2 for _ in range(df))
        w = math.sqrt(chi2 / df)

        # Sample correlated Gaussian z ~ N(0, R) via Cholesky
        eps = [gauss(0.0, 1.0) for _ in range(n)]

        ok = True
        for i in range(n):
            zi = sum(L[i][k] * eps[k] for k in range(i + 1))
            ti = zi / w          # t-distributed marginal
            if ti > thresholds[i]:
                ok = False
                break
        if ok:
            hits += 1
    return hits / n_samples


def copula_joint_approx(probs, avg_rho):
    """Fast equicorrelation Gaussian copula approximation for combo scoring.

    Linearly interpolates between independence (ρ=0) and perfect correlation
    (ρ=1, joint = min(p_i)).  Error ~15-20% for ρ ∈ [0.20, 0.35] — accurate
    enough to rank 91k combos; full MC is reserved for the final chosen SGP.
    """
    p_indep = 1.0
    for p in probs:
        p_indep *= p
    p_min = min(probs)
    # Plan 10 §Z: linear interp is optimistically biased +8% (3-leg) to +29% (4-leg, low-p)
    # vs full Gaussian copula MC; deflate by 0.87 (midpoint of recommended 0.85-0.90).
    return (p_indep + avg_rho * (p_min - p_indep)) * 0.87
