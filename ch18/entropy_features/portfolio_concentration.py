"""
Chapter 18: Entropy Features -- Portfolio Concentration (Section 18.8.3)
==========================================================================

Book status: FORMULA-ONLY. Section 18.8.3 has no printed code snippet --
just the four-step derivation reproduced in the docstrings below. As with
Chapter 17's CUSUM/Chow-DF sections, this is implemented directly from the
book's math and verified with hand-traced known values rather than diffed
against a snippet.

Why this was deferred, and why it's being built now
-----------------------------------------------------
When Chapter 18 was originally scoped, 18.8.3 was flagged "conceptual only,
not pipeline-necessary" because it needs an NxN covariance matrix and an
allocation vector omega as real inputs -- neither existed yet in this
project (this pipeline is single-asset BTC/TUSD). Chapter 16 (HRP) later
built exactly that: a real 6-commodity covariance matrix, plus two real
allocation vectors (HRP and IVP weights). Per this project's standing
"revisit once a prerequisite exists" rule (2026-08-04), that makes 18.8.3
implementable against real data. This module takes V (a covariance matrix)
and omega (an allocation vector, sums to 1) as generic inputs -- callers
wire in Chapter 16's real cov/HRP/IVP artifacts; the module itself has no
dependency on Chapter 16's code.

The four steps (book's own notation)
-------------------------------------
1. Eigendecomposition of the covariance matrix: V W = W Lambda.
2. Factor loadings vector: f_omega = W' omega.
3. Risk contribution of each principal component:
       theta_i = ([f_omega]_i^2 * Lambda_i,i) / sum_n([f_omega]_n^2 * Lambda_n,n)
   with sum_i theta_i = 1 and theta_i in [0, 1].
4. Meucci [2009]'s entropy-inspired concentration measure:
       H = 1 - (1/N) * exp(-sum_i theta_i * log[theta_i])

Note on theta_i and "not a probability": the book flags that theta_i isn't
literally a probability, yet H still behaves like an entropy measure. That
connection runs through the generalized mean (Section 18.7, same chapter --
not an erratum despite Ch18 self-referencing Ch18): H is 1 minus the
"effective number of components" (exp of theta's Shannon entropy in nats)
divided by N. Uniform theta (fully diversified risk) -> effective number
== N -> H == 0. All risk in one component (fully concentrated) -> effective
number == 1 -> H == 1 - 1/N (approaches 1 as N grows). 18.7 itself has no
code snippet either and stays conceptual-only here (no function to test
against) -- this docstring is the pipeline's record of the connection.

Convention: natural log (ln), not log2, per the book's use of e^{...} in
both this section and 18.7's geometric-mean special case
(lim_{q->0} Mq[x,p] = e^{sum p_i log[x_i]}) -- log paired with e is natural
log throughout, unlike 18.1-18.4's entropy-rate estimators, which are
explicitly log2 (bits).
"""
import numpy as np


# =============================================================================
# 18.8.3, step 1: eigendecomposition
# =============================================================================
def eigen_decomposition(V):
    """Eigendecompose a covariance matrix: V W = W Lambda.

    Uses np.linalg.eigh (not the general eig), because a covariance matrix
    is symmetric by construction -- eigh is both faster and numerically
    more stable for symmetric matrices, and guarantees real eigenvalues
    (a general eig on a symmetric matrix can return spurious tiny complex
    components from floating-point noise).

    Parameters
    ----------
    V : array-like, shape (N, N)
        Covariance matrix computed on returns (or price changes, per the
        book's footnote 1 -- the caller decides which, this function is
        agnostic).

    Returns
    -------
    W : ndarray, shape (N, N)
        Eigenvectors as columns, satisfying V @ W == W @ diag(eigenvalues).
    eigenvalues : ndarray, shape (N,)
        Eigenvalues (Lambda's diagonal), ascending, matched to W's columns
        by position (eigh's convention -- the book's step 3 only ever uses
        Lambda_i,i paired with column i of W, so ordering/sign convention
        doesn't affect the final theta or H).
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"V must be a square covariance matrix, got shape {V.shape}")
    if not np.allclose(V, V.T, atol=1e-8):
        raise ValueError("V must be symmetric (a covariance matrix)")
    eigenvalues, W = np.linalg.eigh(V)
    return W, eigenvalues


# =============================================================================
# 18.8.3, step 2: factor loadings
# =============================================================================
def factor_loadings(W, omega):
    """f_omega = W' omega.

    Parameters
    ----------
    W : array-like, shape (N, N)
        Eigenvectors as columns, from eigen_decomposition.
    omega : array-like, shape (N,)
        Vector of allocations (portfolio weights), sum to 1.

    Returns
    -------
    ndarray, shape (N,)
    """
    W = np.asarray(W, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if not np.isclose(omega.sum(), 1.0, atol=1e-6):
        raise ValueError(f"omega must sum to 1, got sum={omega.sum()}")
    return W.T @ omega


# =============================================================================
# 18.8.3, step 3: risk contribution per principal component
# =============================================================================
def risk_contribution(f_omega, eigenvalues):
    """theta_i = [f_omega]_i^2 * Lambda_i,i / sum_n([f_omega]_n^2 * Lambda_n,n).

    Parameters
    ----------
    f_omega : array-like, shape (N,)
        Factor loadings, from factor_loadings.
    eigenvalues : array-like, shape (N,)
        Eigenvalues, matched by position to f_omega (i.e. from the same
        eigen_decomposition call that produced the W used for f_omega).

    Returns
    -------
    ndarray, shape (N,)
        theta, with sum(theta) == 1 and each theta_i in [0, 1].
    """
    f_omega = np.asarray(f_omega, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    numer = f_omega ** 2 * eigenvalues
    denom = numer.sum()
    if np.isclose(denom, 0.0):
        raise ValueError(
            "sum of f_omega^2 * eigenvalues is ~0 -- cannot normalize theta "
            "(degenerate covariance matrix or all-zero allocation)"
        )
    return numer / denom


# =============================================================================
# 18.8.3, step 4: Meucci's entropy-inspired concentration measure
# =============================================================================
def portfolio_concentration(theta):
    """H = 1 - (1/N) * exp(-sum_i theta_i * log[theta_i]).

    theta_i * log(theta_i) uses the standard entropy convention
    0 * log(0) := 0 (the limit as theta_i -> 0+), so zero-risk-contribution
    components don't raise or produce NaN.

    Parameters
    ----------
    theta : array-like, shape (N,)
        Risk contributions, from risk_contribution. Must sum to 1.

    Returns
    -------
    float
        H in [0, 1 - 1/N]. H == 0 <=> theta is uniform (risk is spread
        evenly across all N components -- maximally diversified). H
        approaches 1 - 1/N <=> theta is degenerate (all risk in one
        component -- maximally concentrated); H -> 1 as N -> infinity in
        that degenerate case.
    """
    theta = np.asarray(theta, dtype=float)
    if not np.isclose(theta.sum(), 1.0, atol=1e-6):
        raise ValueError(f"theta must sum to 1, got sum={theta.sum()}")
    if np.any(theta < -1e-12):
        raise ValueError("theta values must be non-negative")
    n = len(theta)
    nonzero = theta > 0
    entropy_term = -np.sum(theta[nonzero] * np.log(theta[nonzero]))
    return 1 - (1.0 / n) * np.exp(entropy_term)


# =============================================================================
# Convenience wrapper -- chains all four steps
# =============================================================================
def compute_portfolio_concentration(V, omega):
    """Run all four steps of Section 18.8.3 in sequence.

    Parameters
    ----------
    V : array-like, shape (N, N)
        Covariance matrix.
    omega : array-like, shape (N,)
        Allocation vector, sums to 1 (e.g. Chapter 16's real HRP or IVP
        weights).

    Returns
    -------
    H : float
        Meucci's portfolio concentration measure.
    theta : ndarray, shape (N,)
        Per-component risk contributions (returned alongside H so callers
        can inspect/plot which principal components dominate risk).
    """
    W, eigenvalues = eigen_decomposition(V)
    f_omega = factor_loadings(W, omega)
    theta = risk_contribution(f_omega, eigenvalues)
    H = portfolio_concentration(theta)
    return H, theta


# =============================================================================
# Sandbox pytest results (Claude's sandbox, Python 3.12.3) -- 2026-08-06
# Real-machine confirmation (mlfinlab, Python 3.10.20) still pending.
# =============================================================================
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 21 items
#
# test_portfolio_concentration.py::TestEigenDecomposition::test_diagonal_matrix_hand_traced PASSED
# test_portfolio_concentration.py::TestEigenDecomposition::test_reconstruction_property_nondiagonal PASSED
# test_portfolio_concentration.py::TestEigenDecomposition::test_non_square_raises PASSED
# test_portfolio_concentration.py::TestEigenDecomposition::test_non_symmetric_raises PASSED
# test_portfolio_concentration.py::TestFactorLoadings::test_identity_W_hand_traced PASSED
# test_portfolio_concentration.py::TestFactorLoadings::test_omega_not_summing_to_one_raises PASSED
# test_portfolio_concentration.py::TestRiskContribution::test_hand_traced_diag_1_4 PASSED
# test_portfolio_concentration.py::TestRiskContribution::test_theta_sums_to_one_general_case PASSED
# test_portfolio_concentration.py::TestRiskContribution::test_degenerate_zero_denominator_raises PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_uniform_theta_is_zero_concentration PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_degenerate_theta_hits_upper_bound PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_hand_traced_theta_0_2_0_8 PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_hand_traced_three_component_theta PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_theta_not_summing_to_one_raises PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_negative_theta_raises PASSED
# test_portfolio_concentration.py::TestPortfolioConcentration::test_zero_theta_component_does_not_nan PASSED
# test_portfolio_concentration.py::TestComputePortfolioConcentration::test_hand_traced_end_to_end_diag_1_4 PASSED
# test_portfolio_concentration.py::TestComputePortfolioConcentration::test_hand_traced_end_to_end_concentrated PASSED
# test_portfolio_concentration.py::TestComputePortfolioConcentration::test_hand_traced_end_to_end_three_assets PASSED
# test_portfolio_concentration.py::TestComputePortfolioConcentration::test_eigenvector_sign_invariance PASSED
# test_portfolio_concentration.py::TestComputePortfolioConcentration::test_result_is_permutation_invariant PASSED
#
# ====================== 21 passed in 0.11s (repo-root-style path) ================
# ====================== 21 passed in 0.10s (from inside entropy_features/) =======
