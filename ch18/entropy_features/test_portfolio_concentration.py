"""
Tests for Section 18.8.3 (Portfolio Concentration) -- formula-only section,
no book snippet to diff against, so every expected value here is hand-traced
by hand (and cross-checked with a plain numpy calculator, see module comments)
before being pinned in these tests.
"""
import numpy as np
import pytest

from portfolio_concentration import (
    eigen_decomposition,
    factor_loadings,
    risk_contribution,
    portfolio_concentration,
    compute_portfolio_concentration,
)


# =============================================================================
# Step 1: eigen_decomposition
# =============================================================================
class TestEigenDecomposition:
    def test_diagonal_matrix_hand_traced(self):
        # A diagonal covariance matrix is already its own eigendecomposition:
        # eigenvalues are the diagonal entries, eigenvectors are the standard
        # basis (up to sign/order, which eigh returns ascending).
        V = np.diag([1.0, 4.0])
        W, eigenvalues = eigen_decomposition(V)
        np.testing.assert_allclose(eigenvalues, [1.0, 4.0])
        # reconstruction property from the book's own equation: V W = W Lambda
        np.testing.assert_allclose(V @ W, W @ np.diag(eigenvalues), atol=1e-10)

    def test_reconstruction_property_nondiagonal(self):
        # A real (non-diagonal) symmetric covariance-like matrix -- the
        # book's defining equation V W = W Lambda must hold regardless of
        # whether V is diagonal.
        V = np.array([[2.0, 0.5, 0.1],
                      [0.5, 1.5, 0.2],
                      [0.1, 0.2, 1.0]])
        W, eigenvalues = eigen_decomposition(V)
        np.testing.assert_allclose(V @ W, W @ np.diag(eigenvalues), atol=1e-10)
        # eigenvectors must be orthonormal (W'W = I) -- required for step 2's
        # f_omega = W'omega to be a well-defined change of basis.
        np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-10)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            eigen_decomposition(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_non_symmetric_raises(self):
        with pytest.raises(ValueError, match="symmetric"):
            eigen_decomposition(np.array([[1.0, 2.0], [99.0, 1.0]]))


# =============================================================================
# Step 2: factor_loadings
# =============================================================================
class TestFactorLoadings:
    def test_identity_W_hand_traced(self):
        # W = identity -> f_omega == omega exactly (no basis change).
        W = np.eye(2)
        omega = np.array([0.5, 0.5])
        f_omega = factor_loadings(W, omega)
        np.testing.assert_allclose(f_omega, [0.5, 0.5])

    def test_omega_not_summing_to_one_raises(self):
        W = np.eye(2)
        with pytest.raises(ValueError, match="sum to 1"):
            factor_loadings(W, np.array([0.5, 0.6]))


# =============================================================================
# Step 3: risk_contribution
# =============================================================================
class TestRiskContribution:
    def test_hand_traced_diag_1_4(self):
        # f_omega = [0.5, 0.5], eigenvalues = [1, 4]
        # theta_1 = 0.5^2*1 / (0.5^2*1 + 0.5^2*4) = 0.25/1.25 = 0.2
        # theta_2 = 1.0/1.25 = 0.8
        theta = risk_contribution(np.array([0.5, 0.5]), np.array([1.0, 4.0]))
        np.testing.assert_allclose(theta, [0.2, 0.8], atol=1e-10)
        assert theta.sum() == pytest.approx(1.0)

    def test_theta_sums_to_one_general_case(self):
        # Property check on a less trivial input -- sum(theta) == 1 must
        # hold by construction (it's a normalization), not just in the
        # hand-picked hand-traced case above.
        theta = risk_contribution(
            np.array([0.3, -0.7, 1.2]), np.array([2.0, 5.0, 0.5])
        )
        assert theta.sum() == pytest.approx(1.0)
        assert np.all(theta >= 0)

    def test_degenerate_zero_denominator_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            risk_contribution(np.array([0.0, 0.0]), np.array([1.0, 4.0]))


# =============================================================================
# Step 4: portfolio_concentration (Meucci's H)
# =============================================================================
class TestPortfolioConcentration:
    def test_uniform_theta_is_zero_concentration(self):
        # Risk spread perfectly evenly across N components -> H == 0
        # (maximally diversified, the book's H lower bound).
        H = portfolio_concentration(np.array([0.5, 0.5]))
        assert H == pytest.approx(0.0, abs=1e-10)

        H3 = portfolio_concentration(np.array([1 / 3, 1 / 3, 1 / 3]))
        assert H3 == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_theta_hits_upper_bound(self):
        # All risk in one component -> H == 1 - 1/N (the book's H upper
        # bound for finite N; approaches 1 as N -> infinity).
        H = portfolio_concentration(np.array([1.0, 0.0]))
        assert H == pytest.approx(1 - 1 / 2, abs=1e-10)

    def test_hand_traced_theta_0_2_0_8(self):
        # theta = [0.2, 0.8] (from the risk_contribution hand-trace above).
        # entropy_term = -(0.2*ln(0.2) + 0.8*ln(0.8)) = 0.5004024235...
        # H = 1 - 0.5*exp(0.5004024235) = 1 - 0.5*1.649385... = 0.1753076
        H = portfolio_concentration(np.array([0.2, 0.8]))
        assert H == pytest.approx(0.17530755576694113, abs=1e-9)

    def test_hand_traced_three_component_theta(self):
        # theta = [1/6, 1/3, 1/2] (from diag([1,2,3]) with equal-weight omega)
        # H = 1 - (1/3)*exp(-(sum theta_i ln theta_i)) = 0.08351357...
        H = portfolio_concentration(np.array([1 / 6, 1 / 3, 1 / 2]))
        assert H == pytest.approx(0.08351357533426496, abs=1e-9)

    def test_theta_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            portfolio_concentration(np.array([0.5, 0.4]))

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            portfolio_concentration(np.array([1.5, -0.5]))

    def test_zero_theta_component_does_not_nan(self):
        # 0*log(0) convention -- must not raise or produce NaN.
        H = portfolio_concentration(np.array([1.0, 0.0, 0.0]))
        assert np.isfinite(H)


# =============================================================================
# Full pipeline: compute_portfolio_concentration
# =============================================================================
class TestComputePortfolioConcentration:
    def test_hand_traced_end_to_end_diag_1_4(self):
        # Full pipeline reproduces the step-by-step hand trace above:
        # V=diag(1,4), omega=[.5,.5] -> theta=[0.2,0.8] -> H=0.1753076
        H, theta = compute_portfolio_concentration(
            np.diag([1.0, 4.0]), np.array([0.5, 0.5])
        )
        np.testing.assert_allclose(theta, [0.2, 0.8], atol=1e-10)
        assert H == pytest.approx(0.17530755576694113, abs=1e-9)

    def test_hand_traced_end_to_end_concentrated(self):
        # V=diag(1,4), omega=[1,0] (all weight on the low-variance asset)
        # -> theta=[1,0] -> H = 1 - 1/2 = 0.5
        H, theta = compute_portfolio_concentration(
            np.diag([1.0, 4.0]), np.array([1.0, 0.0])
        )
        np.testing.assert_allclose(theta, [1.0, 0.0], atol=1e-10)
        assert H == pytest.approx(0.5, abs=1e-10)

    def test_hand_traced_end_to_end_three_assets(self):
        # V=diag(1,2,3), omega=[1/3,1/3,1/3] -> theta=[1/6,1/3,1/2]
        # -> H = 0.08351357...
        H, theta = compute_portfolio_concentration(
            np.diag([1.0, 2.0, 3.0]), np.array([1 / 3, 1 / 3, 1 / 3])
        )
        np.testing.assert_allclose(theta, [1 / 6, 1 / 3, 1 / 2], atol=1e-10)
        assert H == pytest.approx(0.08351357533426496, abs=1e-9)

    def test_eigenvector_sign_invariance(self):
        # eigh's sign convention for eigenvectors is arbitrary (either +v or
        # -v is a valid eigenvector) -- the book's step 3 squares f_omega,
        # so the final theta/H must be identical regardless of which sign
        # numpy happens to return. Cross-check by manually flipping W's
        # column signs and confirming theta is unchanged.
        V = np.array([[2.0, 0.5], [0.5, 1.5]])
        omega = np.array([0.3, 0.7])
        W, eigenvalues = eigen_decomposition(V)
        theta_original = risk_contribution(factor_loadings(W, omega), eigenvalues)

        W_flipped = W * np.array([1, -1])  # flip second eigenvector's sign
        theta_flipped = risk_contribution(
            factor_loadings(W_flipped, omega), eigenvalues
        )
        np.testing.assert_allclose(theta_original, theta_flipped, atol=1e-10)

    def test_result_is_permutation_invariant(self):
        # Reordering which asset is "first" in V/omega shouldn't change H
        # (H characterizes the portfolio's risk concentration, not asset
        # labels/order).
        V = np.diag([1.0, 4.0, 9.0])
        omega = np.array([0.2, 0.3, 0.5])
        H1, _ = compute_portfolio_concentration(V, omega)

        perm = [2, 0, 1]
        V_perm = V[np.ix_(perm, perm)]
        omega_perm = omega[perm]
        H2, _ = compute_portfolio_concentration(V_perm, omega_perm)

        assert H1 == pytest.approx(H2, abs=1e-10)
