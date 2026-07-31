"""
Tests for hrp.py -- AFML Chapter 16 (Hierarchical Risk Parity).

Where possible, known values come straight from the book's own worked
example (Examples 16.1-16.6, Section 16.4.1) rather than being invented --
this catches drift from the book's actual arithmetic, not just shape/type
checks. Every hand-traced value below was independently re-derived by hand
(not just read off scipy's output) before being embedded here; see the
project handoff for the by-hand derivation.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch

sys.path.insert(0, os.path.dirname(__file__))
from hrp import (
    correlDist, getIVP, getClusterVar, getQuasiDiag, getRecBipart, getHRP,
    generateData,
)


# =============================================================================
# Book's own worked example (Examples 16.1-16.6) -- shared fixture
# =============================================================================
BOOK_RHO = np.array([
    [1.0, 0.7, 0.2],
    [0.7, 1.0, -0.2],
    [0.2, -0.2, 1.0],
])


class TestCorrelDist:
    def test_matches_book_example_16_1(self):
        # Book: rho -> d, d_{1,2}=.3873, d_{1,3}=.6325, d_{2,3}=.7746
        dist = correlDist(BOOK_RHO)
        expected = np.array([
            [0.0, 0.3873, 0.6325],
            [0.3873, 0.0, 0.7746],
            [0.6325, 0.7746, 0.0],
        ])
        np.testing.assert_allclose(dist, expected, atol=1e-4)

    def test_diagonal_is_zero(self):
        # d[X,X] = sqrt((1-1)/2) = 0, always, for any valid correlation matrix.
        dist = correlDist(BOOK_RHO)
        np.testing.assert_allclose(np.diag(dist), [0, 0, 0], atol=1e-12)

    def test_perfect_correlation_gives_zero_distance(self):
        rho = np.array([[1.0, 1.0], [1.0, 1.0]])
        dist = correlDist(rho)
        np.testing.assert_allclose(dist, np.zeros((2, 2)), atol=1e-12)

    def test_perfect_anticorrelation_gives_max_distance(self):
        # rho=-1 -> d = sqrt((1-(-1))/2) = sqrt(1) = 1, the metric's max value.
        rho = np.array([[1.0, -1.0], [-1.0, 1.0]])
        dist = correlDist(rho)
        assert dist[0, 1] == pytest.approx(1.0)


class TestBookLinkage:
    """Verifies scipy reproduces the book's own d-tilde and merge sequence
    (Examples 16.2-16.6) when fed the raw (uncondensed) distance matrix,
    per Snippet 16.1's sch.linkage(dist, 'single') call."""

    def test_d_tilde_matches_book_example_16_2(self):
        dist = correlDist(BOOK_RHO)
        from scipy.spatial.distance import pdist, squareform
        d_tilde = squareform(pdist(dist, metric='euclidean'))
        expected = np.array([
            [0.0, 0.5659, 0.9747],
            [0.5659, 0.0, 1.1225],
            [0.9747, 1.1225, 0.0],
        ])
        np.testing.assert_allclose(d_tilde, expected, atol=1e-4)

    def test_merge_sequence_matches_book_examples_16_3_to_16_6(self):
        dist = correlDist(BOOK_RHO)
        with pytest.warns(sch.ClusterWarning):
            link = sch.linkage(dist, 'single')
        # First merge: items 0,1 (book's items 1,2) at d~=.5659 -> u[1]
        assert set(link[0, :2].astype(int)) == {0, 1}
        assert link[0, 2] == pytest.approx(0.5659, abs=1e-4)
        assert link[0, 3] == 2
        # Second merge: item 2 (book's item 3) + cluster u[1] at d~=.9747
        assert set(link[1, :2].astype(int)) == {2, 3}
        assert link[1, 2] == pytest.approx(0.9747, abs=1e-4)
        assert link[1, 3] == 3


class TestGetIVP:
    def test_hand_computed_three_asset(self):
        # cov = diag([1,2,4]) -> ivp = 1/[1,2,4] = [1,.5,.25], normalized
        # by sum=1.75 -> [4/7, 2/7, 1/7]
        cov = np.diag([1.0, 2.0, 4.0])
        ivp = getIVP(cov)
        np.testing.assert_allclose(ivp, [4 / 7, 2 / 7, 1 / 7], atol=1e-9)

    def test_sums_to_one(self):
        cov = np.diag([3.0, 7.0, 1.5, 22.0])
        assert getIVP(cov).sum() == pytest.approx(1.0)

    def test_equal_variance_gives_equal_weight(self):
        cov = np.diag([2.0, 2.0, 2.0])
        np.testing.assert_allclose(getIVP(cov), [1 / 3, 1 / 3, 1 / 3])

    def test_ignores_off_diagonal_by_design(self):
        # getIVP only ever looks at np.diag(cov) -- off-diagonal entries
        # (covariances) shouldn't move the result at all. This IS the
        # documented tradeoff of IVP vs HRP (Section 16.6): it ignores
        # correlation structure entirely.
        cov_uncorrelated = np.array([[4.0, 0.0], [0.0, 1.0]])
        cov_correlated = np.array([[4.0, 3.9], [3.9, 1.0]])
        np.testing.assert_allclose(
            getIVP(cov_uncorrelated), getIVP(cov_correlated)
        )


class TestGetClusterVar:
    def test_hand_computed_uncorrelated_pair(self):
        # cov=[[4,0],[0,1]] -> ivp=[0.2,0.8] -> cVar = .2^2*4 + .8^2*1 = 0.8
        cov = pd.DataFrame([[4.0, 0.0], [0.0, 1.0]], index=[0, 1], columns=[0, 1])
        cVar = getClusterVar(cov, [0, 1])
        assert cVar == pytest.approx(0.8)

    def test_single_item_cluster_returns_its_own_variance(self):
        cov = pd.DataFrame([[4.0, 0.7], [0.7, 1.0]], index=[0, 1], columns=[0, 1])
        assert getClusterVar(cov, [0]) == pytest.approx(4.0)
        assert getClusterVar(cov, [1]) == pytest.approx(1.0)

    def test_matrix_slice_ignores_items_outside_cluster(self):
        # A 3rd, wildly-different asset shouldn't leak into a 2-item
        # cluster's variance computation -- getClusterVar must slice.
        cov = pd.DataFrame(
            [[4.0, 0.0, 999.0], [0.0, 1.0, 999.0], [999.0, 999.0, 999.0]],
            index=[0, 1, 2], columns=[0, 1, 2],
        )
        cVar = getClusterVar(cov, [0, 1])
        assert cVar == pytest.approx(0.8)


class TestGetQuasiDiag:
    def test_matches_book_example_hand_trace(self):
        # Independently hand-traced (see project handoff): for the book's
        # 3-item example, getQuasiDiag should return [2, 0, 1] (0-indexed).
        dist = correlDist(BOOK_RHO)
        with pytest.warns(sch.ClusterWarning):
            link = sch.linkage(dist, 'single')
        sortIx = getQuasiDiag(link)
        assert sortIx == [2, 0, 1]

    def test_returns_a_permutation_of_all_original_items(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 6))
        corr = pd.DataFrame(x).corr().values
        dist = correlDist(corr)
        with pytest.warns(sch.ClusterWarning):
            link = sch.linkage(dist, 'single')
        sortIx = getQuasiDiag(link)
        assert sorted(sortIx) == list(range(6))


class TestGetRecBipart:
    def test_matches_book_example_hand_trace(self):
        # Independently hand-traced (see project handoff): using the book's
        # correlation matrix AS an equal-unit-variance covariance matrix,
        # HRP on sortIx=[2,0,1] should give weights
        # {2: 9/17.7... } -- exact fractions: alpha=1-1/1.85=17/37 first
        # split, giving w2=17/37, and the (0,1) pair splits 50/50 of the
        # remaining 20/37, giving w0=w1=10/37.
        cov = pd.DataFrame(BOOK_RHO, index=[0, 1, 2], columns=[0, 1, 2])
        sortIx = [2, 0, 1]
        w = getRecBipart(cov, sortIx)
        assert w[2] == pytest.approx(17 / 37, abs=1e-6)
        assert w[0] == pytest.approx(10 / 37, abs=1e-6)
        assert w[1] == pytest.approx(10 / 37, abs=1e-6)
        assert w.sum() == pytest.approx(1.0)

    def test_weights_sum_to_one_and_are_non_negative(self):
        # Guarantee from the book: 0<=w_i<=1 and sum(w)=1 (Section 16.4.3).
        rng = np.random.default_rng(11)
        n = 8
        x = rng.normal(size=(500, n))
        cov = pd.DataFrame(x).cov()
        corr = pd.DataFrame(x).corr()
        dist = correlDist(corr)
        with pytest.warns(sch.ClusterWarning):
            link = sch.linkage(dist, 'single')
        sortIx = corr.index[getQuasiDiag(link)].tolist()
        w = getRecBipart(cov, sortIx)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
        assert (w <= 1).all()

    def test_single_asset_gets_full_weight(self):
        cov = pd.DataFrame([[1.0]], index=[0], columns=[0])
        w = getRecBipart(cov, [0])
        assert w[0] == pytest.approx(1.0)


class TestGetHRPEndToEnd:
    def test_full_pipeline_weights_sum_to_one(self):
        rng = np.random.default_rng(2026)
        x = pd.DataFrame(rng.normal(size=(300, 5)), columns=list('ABCDE'))
        cov, corr = x.cov(), x.corr()
        hrp = getHRP(cov, corr)
        assert hrp.sum() == pytest.approx(1.0)
        assert set(hrp.index) == set('ABCDE')
        assert (hrp >= 0).all()

    def test_returns_series_indexed_by_original_labels(self):
        rng = np.random.default_rng(99)
        x = pd.DataFrame(rng.normal(size=(300, 4)),
                          columns=['gold', 'crude_oil', 'corn', 'tbonds'])
        cov, corr = x.cov(), x.corr()
        hrp = getHRP(cov, corr)
        assert sorted(hrp.index) == ['corn', 'crude_oil', 'gold', 'tbonds']

    def test_perfectly_correlated_assets_split_evenly_within_cluster(self):
        # Two assets with identical variance and rho=1 are indistinguishable
        # to HRP -- they must end up with identical weight.
        rng = np.random.default_rng(3)
        base = rng.normal(size=300)
        x = pd.DataFrame({
            'A': base, 'B': base,           # perfectly correlated, identical
            'C': rng.normal(size=300),      # independent
        })
        cov, corr = x.cov(), x.corr()
        hrp = getHRP(cov, corr)
        assert hrp['A'] == pytest.approx(hrp['B'], abs=1e-9)


class TestGenerateData:
    def test_shape(self):
        rng = np.random.default_rng(1)
        x, cols = generateData(nObs=100, size0=5, size1=3, sigma1=0.25,
                                random_state=rng)
        assert x.shape == (100, 8)          # size0 + size1 columns
        assert len(cols) == 3

    def test_seeded_reproducibility(self):
        x1, cols1 = generateData(nObs=50, size0=4, size1=2, sigma1=0.25,
                                  random_state=np.random.default_rng(2026))
        x2, cols2 = generateData(nObs=50, size0=4, size1=2, sigma1=0.25,
                                  random_state=np.random.default_rng(2026))
        pd.testing.assert_frame_equal(x1, x2)
        np.testing.assert_array_equal(cols1, cols2)

    def test_perturbed_columns_correlate_with_their_source(self):
        # Columns size0+1..size0+size1 are built as (base column + noise),
        # so they should be meaningfully correlated with their source --
        # not just coincidentally, given a low-noise sigma1.
        rng = np.random.default_rng(42)
        x, cols = generateData(nObs=5000, size0=3, size1=3, sigma1=0.1,
                                random_state=rng)
        corr = x.corr()
        for i, source_col in enumerate(cols):
            perturbed_col = 3 + i + 1     # 1-indexed columns, offset by size0
            assert corr.loc[source_col + 1, perturbed_col] > 0.8
