"""
TDD suite for cusum.py (AFML Sec 17.3.1 BDE and 17.3.2 CSW, both
formula-only, no printed book code). Every test pins a known value: either
hand-computed OLS/regression algebra on a tiny fixed example, or a
regression-detection sanity check against a synthetically injected break.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from cusum import (
    get_bde_recursive_residuals, get_bde_cusum,
    get_csw_stat, get_csw_critical_value, get_csw_sup, get_csw_cusum,
)


# =============================================================================
# 17.3.1 -- BDE recursive residuals
# =============================================================================
class TestBdeRecursiveResiduals:
    def test_hand_computed_single_residual(self):
        # Near-exact linear fit (y ~ 1 + 2x) with small deliberate noise so
        # the residual variance isn't degenerate (a PERFECT noiseless fit
        # gives sigma2~0 and blows up the standardization -- confirmed by
        # direct test, see test_degenerate_perfect_fit_warns below).
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y = np.array([3.1, 4.9, 7.2, 8.8])
        times, omega = get_bde_recursive_residuals(X, y, min_sample=3)
        assert times.tolist() == [3]
        assert omega[0] == pytest.approx(-0.9838699100998953, abs=1e-10)

    def test_raises_when_min_sample_too_small(self):
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.array([3.0, 5.0, 7.0])
        with pytest.raises(ValueError, match="must exceed the number of regressors"):
            get_bde_recursive_residuals(X, y, min_sample=2)  # p=2, needs >2

    def test_degenerate_perfect_fit_gives_near_zero_residual_variance(self):
        # Documents real behavior on degenerate (noiseless) data: with an
        # EXACTLY linear y=1+2x and no noise, the pre-forecast fit has
        # residual variance ~0 (floating-point noise only), which is the
        # root cause of instability in the standardized ratio downstream.
        # NOTE: the exact downstream omega value here is platform/BLAS-
        # rounding-dependent (can come out as nan, a huge finite ratio, or
        # an unstable-but-finite O(1) value depending on exact floating-
        # point cancellation) -- not asserted directly, since it isn't
        # deterministic. What IS deterministic and worth pinning: the
        # residual variance genuinely collapses to numerical noise.
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])   # EXACT y=1+2x, zero residual
        X_prev, y_prev = X[:3], y[:3]
        beta = np.linalg.inv(X_prev.T @ X_prev) @ (X_prev.T @ y_prev)
        resid = y_prev - X_prev @ beta
        sigma2 = np.sum(resid ** 2) / (3 - 2)
        assert sigma2 < 1e-20   # numerical noise, not a real residual variance
        # And the function itself must not raise -- it should still return
        # SOME value (finite or not), not crash outright.
        times, omega = get_bde_recursive_residuals(X, y, min_sample=3)
        assert len(omega) == 1

    def test_recursive_beta_uses_only_past_data(self):
        # The whole point of "recursive" residuals is that beta_hat_{t-1}
        # must NOT see y_t. Verify by checking that a large outlier placed
        # ONLY at the final point doesn't change the residual computed for
        # an earlier point.
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(10), rng.normal(0, 1, 10)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 0.1, 10)
        _, omega_clean = get_bde_recursive_residuals(X, y, min_sample=4)

        y_outlier = y.copy()
        y_outlier[-1] = 1000.0   # huge outlier only at the very last point
        _, omega_dirty = get_bde_recursive_residuals(X, y_outlier, min_sample=4)

        # every residual EXCEPT the one that forecasts the outlier itself
        # should be identical -- past residuals can't see a future outlier.
        assert omega_clean[:-1] == pytest.approx(omega_dirty[:-1])
        assert omega_clean[-1] != pytest.approx(omega_dirty[-1])


class TestBdeCusum:
    def test_returns_expected_columns_and_index(self):
        rng = np.random.default_rng(1)
        T = 20
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 1, T)
        dates = pd.bdate_range('2020-01-01', periods=T)
        out = get_bde_cusum(X, y, min_sample=5, index=dates)
        assert list(out.columns) == ['omega', 'S', 'band_95']
        assert out.index[0] == dates[5]

    def test_S_matches_raw_cumsum_over_sigma_omega(self):
        # Book's literal formula: S_t sums RAW (non-demeaned) omega_j,
        # dividing once by sigma_hat_omega (which IS computed from demeaned
        # variance) -- confirmed this is what the book actually says, not
        # a bug to "fix" toward a fully-demeaned cumulative sum.
        rng = np.random.default_rng(2)
        T = 20
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 1, T)
        out = get_bde_cusum(X, y, min_sample=5)

        sigma_omega = np.std(out['omega'].values - out['omega'].values.mean())
        manual_S = np.cumsum(out['omega'].values) / sigma_omega
        assert out['S'].values == pytest.approx(manual_S)

    def test_injected_break_produces_visible_jump_in_S(self):
        # A real, sudden coefficient break should show up as a jump in S
        # noticeably larger than the typical step size elsewhere in the
        # series (see cusum.py's own docstring / this chapter's README for
        # why "crosses the 95% band" isn't always achievable on a short
        # series -- the jump itself is the honest signal here).
        rng = np.random.default_rng(1)
        T = 40
        X = np.column_stack([np.ones(T), rng.normal(0, 1, T)])
        y = X @ np.array([1.0, 2.0]) + rng.normal(0, 1, T)
        break_t = 25
        y[break_t:] = X[break_t:] @ np.array([1.0, -3.0]) + rng.normal(0, 1, T - break_t)

        out = get_bde_cusum(X, y, min_sample=5)
        S = out['S'].values
        step_changes = np.abs(np.diff(S))
        break_row = np.where(out.index == break_t)[0]
        assert len(break_row) == 1
        jump_at_break = step_changes[break_row[0] - 1]
        typical_step = np.median(step_changes)
        assert jump_at_break > 3 * typical_step

    def test_raises_on_too_short_series(self):
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
        y = np.array([3.1, 4.9, 7.2, 8.8])
        with pytest.raises(ValueError, match="Fewer than 2"):
            get_bde_cusum(X, y, min_sample=3)  # only 1 residual possible


# =============================================================================
# 17.3.2 -- Chu-Stinchcombe-White CUSUM
# =============================================================================
class TestCswStat:
    def test_hand_computed(self):
        # Tiny fixed price series, hand-computed sigma_t and S_{n,t}.
        prices = np.array([10.0, 10.5, 11.2, 10.8, 12.0, 13.5])
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=6))
        n_idx, t_idx = 1, 5
        S = get_csw_stat(logp, n_idx, t_idx)

        values = logp.values
        diffs = np.diff(values[:t_idx + 1])   # Delta y_2..Delta y_5 (5 diffs, 0-idx: values[0:6])
        sigma_t2 = np.sum(diffs ** 2) / (len(diffs) - 1)
        sigma_t = np.sqrt(sigma_t2)
        S_manual = (values[t_idx] - values[n_idx]) / (sigma_t * np.sqrt(t_idx - n_idx))
        assert S == pytest.approx(S_manual, abs=1e-12)

    def test_accepts_dataframe_input(self):
        prices = np.array([10.0, 10.5, 11.2, 10.8, 12.0, 13.5])
        idx = pd.bdate_range('2020-01-01', periods=6)
        logp_series = pd.Series(np.log(prices), index=idx)
        logp_df = logp_series.to_frame()
        assert get_csw_stat(logp_series, 1, 5) == pytest.approx(get_csw_stat(logp_df, 1, 5))

    def test_raises_when_t_not_after_n(self):
        prices = np.array([10.0, 10.5, 11.2])
        logp = pd.Series(np.log(prices), index=pd.bdate_range('2020-01-01', periods=3))
        with pytest.raises(ValueError, match="must exceed"):
            get_csw_stat(logp, n_idx=2, t_idx=1)


class TestCswCriticalValue:
    def test_matches_book_formula(self):
        # c_alpha[n,t] = sqrt(b_alpha + log(t-n)), b_0.05=4.6 (book's own
        # Monte-Carlo-derived constant).
        cv = get_csw_critical_value(n_idx=5, t_idx=15, b_alpha=4.6)
        assert cv == pytest.approx(np.sqrt(4.6 + np.log(10)), abs=1e-12)


class TestCswSup:
    def test_matches_manual_max_over_n(self):
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 20))
        logp = pd.Series(np.log(prices), index=pd.bdate_range('2020-01-01', periods=20))
        t_idx = 19
        out = get_csw_sup(logp, t_idx)

        manual_best_S, manual_best_n = -np.inf, None
        for n_idx in range(0, t_idx):
            S = get_csw_stat(logp, n_idx, t_idx)
            if np.isfinite(S) and S > manual_best_S:
                manual_best_S, manual_best_n = S, n_idx
        assert out['S'] == pytest.approx(manual_best_S)
        assert out['n_star'] == manual_best_n

    def test_explosive_jump_gives_higher_sup_than_random_walk(self):
        rng = np.random.default_rng(0)
        rw = 100 + np.cumsum(rng.normal(0, 1, 20))
        idx = pd.bdate_range('2020-01-01', periods=23)

        logp_rw = pd.Series(np.log(np.concatenate([rw, rw[-1] + rng.normal(0, 1, 3)])),
                             index=idx)
        jump_prices = np.concatenate([rw, [rw[-1] * 1.5, rw[-1] * 1.8, rw[-1] * 2.3]])
        logp_jump = pd.Series(np.log(jump_prices), index=idx)

        S_rw = get_csw_sup(logp_rw, 22)['S']
        S_jump = get_csw_sup(logp_jump, 22)['S']
        assert S_jump > S_rw


class TestCswCusum:
    def test_returns_expected_shape(self):
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logp = pd.Series(np.log(prices), index=pd.bdate_range('2020-01-01', periods=15))
        out = get_csw_cusum(logp, min_sample=3)
        assert list(out.columns) == ['S', 'n_star', 'critical_value_95']
        assert len(out) == 15 - 3

    def test_last_row_matches_direct_get_csw_sup_call(self):
        rng = np.random.default_rng(3)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logp = pd.Series(np.log(prices), index=pd.bdate_range('2020-01-01', periods=15))
        out = get_csw_cusum(logp, min_sample=3)
        direct = get_csw_sup(logp, 14)
        assert out['S'].iloc[-1] == pytest.approx(direct['S'])
        assert out['n_star'].iloc[-1] == direct['n_star']

    def test_raises_on_too_short_series(self):
        prices = np.array([10.0, 10.5, 11.2])
        logp = pd.Series(np.log(prices), index=pd.bdate_range('2020-01-01', periods=3))
        with pytest.raises(ValueError, match="needs more than"):
            get_csw_cusum(logp, min_sample=5)
