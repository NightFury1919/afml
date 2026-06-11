import numpy as np

def pca_weights(cov, risk_dist=None, risk_target=1.0):
    # PCA Weights — AFML Chapter 2, Section 2.4.2, pages 35-36
    # (Snippet 2.1 in the book)
    #
    # Computes allocation weights ω for a portfolio of N instruments such that
    # risk is distributed across principal components according to a target
    # distribution R. Uses spectral decomposition of the covariance matrix V.
    #
    # --- What problem does this solve? ---
    # Suppose you have three highly correlated futures contracts (e.g. March,
    # June, September S&P 500 futures). A naive equal-weight portfolio
    # (1/3, 1/3, 1/3) does NOT give you equal risk exposure — because all three
    # contracts move almost in lockstep, nearly all your risk comes from one
    # "direction" (the broad market). You are not actually diversified.
    #
    # PCA Weights fix this by:
    # 1. Finding the PRINCIPAL COMPONENTS (the "directions" of variation in
    #    the covariance matrix that are mutually uncorrelated).
    # 2. Deciding how much RISK to allocate to each component (via risk_dist).
    # 3. Computing the instrument weights ω that achieve that risk distribution.
    #
    # --- What is a principal component? ---
    # The covariance matrix V has N eigenvectors (W) and N eigenvalues (Λ).
    # Each eigenvector is a "direction" in the N-dimensional price space.
    # The eigenvalue tells you how much variance (risk) lies in that direction.
    #
    # The eigenvector with the LARGEST eigenvalue is the "first principal component":
    # the direction in which the portfolio varies the most. For three correlated
    # futures, this is roughly (1/√3, 1/√3, 1/√3) — the "market" direction.
    #
    # The eigenvector with the SMALLEST eigenvalue is the "last principal component":
    # the direction of least variation. For correlated futures, this is an almost
    # flat spread (e.g. long March, short September) — a nearly risk-free position.
    #
    # --- What does risk_dist control? ---
    # risk_dist[k] = the fraction of total portfolio risk to allocate to
    # principal component k. It must sum to 1.
    #
    # risk_dist = [1, 0, 0]   → all risk in the largest component (market beta)
    # risk_dist = [0, 0, 1]   → all risk in the smallest component (minimum variance)
    # risk_dist = [1/3, 1/3, 1/3] → equal risk across all components
    #
    # By default (risk_dist=None), all risk goes into the smallest eigenvalue
    # component — this is the MINIMUM VARIANCE PORTFOLIO, which takes the
    # least-risky bet available in the data.
    #
    # --- Five-step derivation from pages 35-36 ---
    # Step 1 — Spectral decomposition: VW = WΛ
    #   V = covariance matrix (N×N)
    #   W = eigenvectors (columns are principal components, each of length N)
    #   Λ = diagonal matrix of eigenvalues (sorted descending)
    #
    # Step 2 — Portfolio risk: σ² = ω'Vω = β'Λβ
    #   β = W'ω  (projection of weights ω onto the eigenvector basis)
    #   In the eigenvector basis, the covariance matrix is diagonal (Λ),
    #   so risk contributions from each component are simply β²_n * Λ_{n,n}.
    #
    # Step 3 — Risk attribution per component:
    #   R_n = β²_n * Λ_{n,n} * σ^{-2}  (fraction of total risk from component n)
    #   Σ R_n = 1 by construction
    #
    # Step 4 — Allocation in eigenvector basis β:
    #   β_n = σ * sqrt(R_n / Λ_{n,n})
    #   Derived by solving R_n = β²_n * Λ_{n,n} / σ² for β_n.
    #   risk_target plays the role of σ (the desired total portfolio risk).
    #
    # Step 5 — Convert back to original instrument basis:
    #   ω = W * β
    #   Because W is orthogonal (W'W = I), this transformation is exact.

    # -----------------------------------------------------------------------
    # Step 1: Spectral decomposition VW = WΛ
    # -----------------------------------------------------------------------
    # np.linalg.eigh is used instead of eig because cov is symmetric (positive
    # semi-definite). eigh is faster and guaranteed to return real eigenvalues.
    #
    # eVal[k] = eigenvalue k  (a scalar — the "size" of that direction of risk)
    # eVec[:, k] = eigenvector k  (a vector of length N — the "direction")
    eVal, eVec = np.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in DESCENDING order (largest first).
    # This is conventional: PC1 = most variance, PC_N = least variance.
    indices = eVal.argsort()[::-1]     # [::-1] reverses the array (ascending → descending)
    eVal, eVec = eVal[indices], eVec[:, indices]

    # -----------------------------------------------------------------------
    # Default risk distribution: minimum variance portfolio
    # -----------------------------------------------------------------------
    # risk_dist[-1] = 1.0 means ALL risk goes into the LAST (smallest eigenvalue)
    # principal component. The last component is the direction in which the
    # portfolio varies the LEAST — this produces the minimum variance portfolio.
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0     # concentrate all risk in the smallest eigenvalue component

    # -----------------------------------------------------------------------
    # Step 4: Compute allocations β in the eigenvector (principal component) basis
    # -----------------------------------------------------------------------
    # β_n = risk_target * sqrt(risk_dist[n] / eVal[n])
    #
    # risk_target is the desired total portfolio volatility σ.
    # risk_dist[n] is the fraction of that σ² we want from component n.
    # Dividing by eVal[n] (= Λ_{n,n}) scales by the variance of component n:
    #   a component with large eigenvalue already has a lot of variance per unit β,
    #   so we need a SMALL β to contribute only risk_dist[n] of total risk.
    #   a component with small eigenvalue has little variance per unit β,
    #   so we need a LARGE β to contribute the same fraction of total risk.
    loads = risk_target * (risk_dist / eVal) ** 0.5
    # loads is a 1D array of β_n values, one per principal component

    # -----------------------------------------------------------------------
    # Step 5: Convert allocations back to original instrument space
    # -----------------------------------------------------------------------
    # ω = W * β
    # W is the matrix of eigenvectors (N×N), β (loads) is an N-vector.
    # The result ω is an N-vector of weights for the original instruments.
    #
    # np.reshape(loads, (-1, 1)) turns the 1D array into a column vector (N×1)
    # so that np.dot(eVec, ...) does a proper matrix-vector multiply (N×N) × (N×1) = (N×1).
    weights = np.dot(eVec, np.reshape(loads, (-1, 1)))
    # weights[i] is the allocation to instrument i in the original price space

    return weights
