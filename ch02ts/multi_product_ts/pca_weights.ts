// pca_weights.ts — mirror of ch02/multi_product/pca_weights.py
// AFML Chapter 2, Section 2.4.2, pages 35-36 (Snippet 2.1)
//
// 📁 C:\ws\AFML\
// └── ch02_ts\
//     └── multi_product_ts\
//         └── pca_weights.ts   ← goes here
//
// Dependency: mathjs  (npm install mathjs)
// mathjs is used solely for eigendecomposition of the covariance matrix —
// the one operation numpy.linalg.eigh provides that plain TypeScript cannot.

import { eigs } from 'mathjs';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** N×N covariance matrix — row-major: cov[i][j] = covariance of instruments i and j. */
export type CovMatrix = number[][];

/** N-vector of portfolio weights ω, one per instrument. */
export type WeightVector = number[];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Matrix-vector multiply: (N×N) × (N×1) → N-vector.
 * Replaces numpy's np.dot(eVec, loads).
 */
function matVecMul(mat: number[][], vec: number[]): number[] {
  const N = mat.length;
  const result = new Array<number>(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      result[i] += mat[i][j] * vec[j];
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// pcaWeights
// ---------------------------------------------------------------------------

/**
 * PCA Weights — AFML Chapter 2, Section 2.4.2, pages 35-36.
 *
 * Computes allocation weights ω for a portfolio of N instruments such that
 * risk is distributed across principal components according to risk_dist.
 *
 * Why?
 *   A naive equal-weight portfolio across correlated instruments is NOT
 *   equally diversified — nearly all risk comes from one "direction"
 *   (the common factor). PCA weights let you explicitly control how much
 *   risk goes into each independent direction of variation.
 *
 * Five-step derivation (pages 35-36):
 *   1. Spectral decomposition: V = W Λ W'
 *      W = eigenvectors (principal components), Λ = diagonal eigenvalues
 *   2. Portfolio risk: σ² = ω'Vω = β'Λβ  where β = W'ω
 *   3. Risk attribution: R_n = β²_n * Λ_n / σ²  (fraction from component n)
 *   4. Solve for β: β_n = σ * sqrt(R_n / Λ_n)
 *   5. Convert back: ω = W * β
 *
 * @param cov         - N×N covariance matrix (symmetric, positive semi-definite)
 * @param riskDist    - N-vector: fraction of total risk to put in each component.
 *                      Defaults to minimum variance portfolio (all risk in smallest PC).
 * @param riskTarget  - desired total portfolio volatility σ (default 1.0)
 * @returns N-vector of instrument weights ω
 */
export function pcaWeights(
  cov: CovMatrix,
  riskDist?: number[],
  riskTarget = 1.0
): WeightVector {
  const N = cov.length;

  // -----------------------------------------------------------------------
  // Step 1: Spectral decomposition VW = WΛ
  // -----------------------------------------------------------------------
  // mathjs.eigs() is the equivalent of numpy.linalg.eigh() for symmetric matrices.
  // It returns eigenvalues in ASCENDING order (smallest first) — same as numpy.eigh.
  //
  // result.values          → array of N eigenvalues (ascending)
  // result.eigenvectors[k] → { value: λ_k, vector: [w_0k, w_1k, ...] }
  //   where vector is the k-th eigenvector (principal component direction)
  const result = eigs(cov);

  // Extract eigenvalues (ascending) and eigenvectors as a column matrix.
  // eVec[i][k] = component i of eigenvector k  (N×N matrix, columns = eigenvectors)
  const eigenvaluesAsc: number[] = (result.values as number[]);
  const eigenvectorsAsc = result.eigenvectors as { value: number; vector: number[] }[];

  // Sort DESCENDING (largest eigenvalue first) — mirrors numpy's [::-1] reversal.
  // PC1 = most variance (market direction), PC_N = least variance (min-var direction).
  const indices = eigenvaluesAsc
    .map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)   // descending
    .map(x => x.i);

  const eVal: number[]   = indices.map(i => eigenvaluesAsc[i]);
  // Build eVec as N×N row-major matrix (eVec[row][col] = element of eigenvector col)
  const eVec: number[][] = Array.from({ length: N }, (_, row) =>
    indices.map(col => eigenvectorsAsc[col].vector[row])
  );

  // -----------------------------------------------------------------------
  // Default risk distribution: minimum variance portfolio
  // -----------------------------------------------------------------------
  // risk_dist[-1] = 1.0 → all risk in the LAST (smallest eigenvalue) component.
  // The smallest-eigenvalue PC is the direction of LEAST variance —
  // concentrating risk there gives the minimum variance portfolio.
  const rD: number[] = riskDist ?? (() => {
    const d = new Array<number>(N).fill(0);
    d[N - 1] = 1.0;  // last element = smallest eigenvalue component (after descending sort)
    return d;
  })();

  // -----------------------------------------------------------------------
  // Step 4: Compute allocations β in the eigenvector (PC) basis
  // -----------------------------------------------------------------------
  // β_n = riskTarget * sqrt(riskDist[n] / eVal[n])
  //
  // Why divide by eVal[n]?
  //   A component with large eigenvalue already has a lot of variance per unit β.
  //   To contribute only riskDist[n] of total risk, we need a SMALL β.
  //   A component with small eigenvalue has little variance per unit β,
  //   so we need a LARGE β to contribute the same fraction.
  const loads: number[] = eVal.map((lambda, n) =>
    riskTarget * Math.sqrt(rD[n] / lambda)
  );
  // loads[n] = β_n — allocation to principal component n

  // -----------------------------------------------------------------------
  // Step 5: Convert back to original instrument space
  // -----------------------------------------------------------------------
  // ω = W * β
  // W (eVec) is orthogonal (W'W = I), so this is an exact change of basis.
  // matVecMul does (N×N) × (N×1) → N-vector of instrument weights.
  const weights = matVecMul(eVec, loads);

  return weights;
}
