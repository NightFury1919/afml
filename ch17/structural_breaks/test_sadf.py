"""
TDD suite for sadf.py (AFML Snippets 17.1-17.4).

Every test pins a KNOWN value: either hand-computed OLS algebra on a tiny
fixed array, an independent cross-check against statsmodels' adfuller
(computed once in Claude's sandbox, pinned here as a literal so mlfinlab
doesn't need statsmodels as a real dependency -- see CLAUDE.md's "don't
add new dependencies without confirming compatibility"), or a
regression-proving test that the None-bug fix actually changes behavior
(a literal transcription of Snippet 17.1 crashes outright).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from sadf import lagDF, getYX, getBetas, get_bsadf, get_sadf


# =============================================================================
# lagDF
# =============================================================================
class TestLagDF:
    def test_int_lags_includes_lag_zero(self):
        # isinstance(lags, int) -> lags = range(lags+1), i.e. lag=2 means
        # lags [0, 1, 2] -- THREE columns, not two. Easy off-by-one to miss.
        df0 = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = lagDF(df0, lags=2)
        assert list(out.columns) == ['x_0', 'x_1', 'x_2']

    def test_shift_values_hand_traced(self):
        df0 = pd.DataFrame({'x': [10.0, 20.0, 30.0, 40.0]})
        out = lagDF(df0, lags=1)
        # x_0 is unlagged: [10,20,30,40]. x_1 is shifted by 1: [NaN,10,20,30].
        assert out['x_0'].tolist() == [10.0, 20.0, 30.0, 40.0]
        assert np.isnan(out['x_1'].iloc[0])
        assert out['x_1'].iloc[1:].tolist() == [10.0, 20.0, 30.0]

    def test_explicit_lag_list(self):
        df0 = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = lagDF(df0, lags=[0, 3])
        assert list(out.columns) == ['x_0', 'x_3']


# =============================================================================
# getYX
# =============================================================================
class TestGetYX:
    def test_series_auto_converted_to_dataframe(self):
        # Book's docstring says "a pandas series"; the printed code's 2D
        # indexing only works on a single-column DataFrame. Confirmed by
        # direct test that a bare Series crashes inside lagDF with an
        # opaque AttributeError -- getYX must auto-convert, not crash.
        dates = pd.bdate_range('2020-01-01', periods=10)
        s = pd.Series(np.log(100 + np.arange(10, dtype=float)), index=dates,
                       name='close')
        y, x = getYX(s, constant='nc', lags=1)
        assert y.shape[0] > 0 and x.shape[0] > 0

    def test_dataframe_input_shapes(self):
        dates = pd.bdate_range('2020-01-01', periods=15)
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        y, x = getYX(logP, constant='nc', lags=1)
        # 15 rows -> diff loses 1 -> lagDF(lags=1) loses another 1 to NaN
        # -> 13 usable rows. x has 2 columns for lags=1,'nc': [level, diff_lag1].
        assert y.shape == (13, 1)
        assert x.shape == (13, 2)

    def test_constant_nc_adds_no_columns(self):
        dates = pd.bdate_range('2020-01-01', periods=15)
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        _, x = getYX(logP, constant='nc', lags=1)
        assert x.shape[1] == 2  # level + 1 lagged diff, nothing else

    def test_constant_ct_adds_two_columns(self):
        # 'ct' appends BOTH a ones (constant) column AND a linear-trend
        # column (constant[:2]=='ct' fires the trend branch too) -- verified
        # directly (x.shape[1] goes 2 -> 4, not 2 -> 3), not assumed.
        dates = pd.bdate_range('2020-01-01', periods=15)
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        _, x_nc = getYX(logP, constant='nc', lags=1)
        _, x_ct = getYX(logP, constant='ct', lags=1)
        assert x_ct.shape[1] == x_nc.shape[1] + 2

    def test_constant_ctt_adds_two_columns_vs_ct(self):
        dates = pd.bdate_range('2020-01-01', periods=15)
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 15))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        _, x_ct = getYX(logP, constant='ct', lags=1)
        _, x_ctt = getYX(logP, constant='ctt', lags=1)
        # 'ctt' = 'ct' + one more column (trend**2)
        assert x_ctt.shape[1] == x_ct.shape[1] + 1

    def test_column_zero_overwritten_with_level_not_diff(self):
        # Hand-traced: for a 4-row price series, column 0 of x (post-overwrite)
        # should equal the LEVEL y_{t-1}, not the lag-0 diff lagDF originally
        # put there.
        dates = pd.bdate_range('2020-01-01', periods=6)
        prices = np.array([10.0, 12.0, 15.0, 19.0, 24.0, 30.0])
        logP = pd.DataFrame({'close': prices}, index=dates)  # not log-transformed,
        # doesn't matter for this structural check
        y, x = getYX(logP, constant='nc', lags=1)
        # diffs: [2,3,4,5,6] (5 values, index 1..5)
        # lagDF(lags=1) on diffs -> lag0=[2,3,4,5,6], lag1=[nan,2,3,4,5] -> dropna
        # keeps rows where lag1 exists: 4 rows (diffs at index 2,3,4,5)
        # x.iloc[:,0] overwritten with series.values[-x.shape[0]-1:-1,0]
        #   = prices[-5:-1] = prices[1:5] = [12,15,19,24] (levels y_{t-1})
        assert x[:, 0].tolist() == [12.0, 15.0, 19.0, 24.0]


# =============================================================================
# getBetas -- hand-computed OLS on a tiny fixed example
# =============================================================================
class TestGetBetas:
    def test_hand_computed_simple_regression(self):
        # y = 2*x (no intercept), plus tiny noise -> beta should recover ~2
        # exactly for the noiseless case.
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([[2.0], [4.0], [6.0], [8.0]])
        bMean, bVar = getBetas(y, x)
        assert bMean[0, 0] == pytest.approx(2.0, abs=1e-10)
        # Perfect fit -> zero residual variance
        assert bVar[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_two_regressor_hand_computed(self):
        # y = 1 + 2*x, exact fit, x = [const=1, x]
        xs = np.array([1.0, 2.0, 3.0, 4.0])
        ys = 1.0 + 2.0 * xs
        x = np.column_stack([xs, np.ones_like(xs)])
        y = ys.reshape(-1, 1)
        bMean, bVar = getBetas(y, x)
        assert bMean[0, 0] == pytest.approx(2.0, abs=1e-10)  # slope
        assert bMean[1, 0] == pytest.approx(1.0, abs=1e-10)  # intercept


# =============================================================================
# get_bsadf -- the None-bug fix, and behavior checks
# =============================================================================
class TestGetBsadf:
    def _make_logp(self, seed, n=30, explosive=False):
        dates = pd.bdate_range('2020-01-01', periods=n)
        rng = np.random.default_rng(seed)
        if explosive:
            prices = 100 * np.exp(0.02 * np.arange(n)) + rng.normal(0, 0.5, n)
        else:
            prices = 100 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({'close': np.log(prices)}, index=dates)

    def test_literal_book_transcription_crashes_on_none(self):
        """REGRESSION TEST proving the fix matters, not a cosmetic rewrite:
        a literal transcription of Snippet 17.1 (bsadf=None) raises
        TypeError on the very first loop iteration under Python 3."""
        def get_bsadf_as_literally_printed(logP, minSL, constant, lags):
            y, x = getYX(logP, constant=constant, lags=lags)
            startPoints, bsadf, allADF = (
                range(0, y.shape[0] + lags - minSL + 1), None, [])
            for start in startPoints:
                y_, x_ = y[start:], x[start:]
                bMean_, bStd_ = getBetas(y_, x_)
                bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** .5
                allADF.append(bMean_ / bStd_)
                if allADF[-1] > bsadf:
                    bsadf = allADF[-1]
            return {'Time': logP.index[-1], 'gsadf': bsadf}

        logP = self._make_logp(seed=0)
        with pytest.raises(TypeError, match="not supported between"):
            get_bsadf_as_literally_printed(logP, minSL=10, constant='nc', lags=1)

    def test_fixed_version_does_not_crash(self):
        logP = self._make_logp(seed=0)
        out = get_bsadf(logP, minSL=10, constant='nc', lags=1)
        assert np.isfinite(out['gsadf'])

    def test_explosive_series_gives_higher_bsadf_than_random_walk(self):
        # Not a hand-traced exact value (the whole point of ADF is that it's
        # a statistical test, not a deterministic formula on real-looking
        # data) -- but the qualitative behavior the book claims (explosive
        # series should score higher than a random walk) must hold, and
        # does, robustly, across multiple seeds.
        for seed in range(5):
            logp_explosive = self._make_logp(seed, n=30, explosive=True)
            logp_rw = self._make_logp(seed, n=30, explosive=False)
            out_explosive = get_bsadf(logp_explosive, minSL=10, constant='nc', lags=1)
            out_rw = get_bsadf(logp_rw, minSL=10, constant='nc', lags=1)
            assert out_explosive['gsadf'] > out_rw['gsadf']

    def test_matches_standard_adf_when_minSL_forces_single_start_point(self):
        """Book's own claim (Sec 17.4.2): 'The standard ADF test is a
        special case of SADFt, where tau = t-1.' Cross-checked once against
        statsmodels.tsa.stattools.adfuller(regression='n', maxlag=1,
        autolag=None) on this exact seeded series in Claude's sandbox;
        pinned here as a literal so mlfinlab doesn't need statsmodels
        installed as a real dependency."""
        logP = self._make_logp(seed=3, n=40, explosive=False)
        T = len(logP)
        out = get_bsadf(logP, minSL=T - 1, constant='nc', lags=1)
        # Cross-checked against statsmodels adfuller -- exact match to
        # float precision (not approximate).
        assert out['gsadf'] == pytest.approx(-0.20429760156463905, abs=1e-9)


# =============================================================================
# get_sadf -- the outer loop (our own driver, per book's own description)
# =============================================================================
class TestGetSadf:
    def test_returns_one_value_per_valid_end_point(self):
        dates = pd.bdate_range('2020-01-01', periods=20)
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 20))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        sadf = get_sadf(logP, minSL=8, constant='nc', lags=1)
        assert len(sadf) == 20 - (8 + 1)
        assert sadf.index.is_monotonic_increasing

    def test_last_value_matches_direct_get_bsadf_call(self):
        # get_sadf's last entry should be identical to calling get_bsadf
        # directly on the full series -- same end point, same computation.
        dates = pd.bdate_range('2020-01-01', periods=20)
        rng = np.random.default_rng(2)
        prices = 100 + np.cumsum(rng.normal(0, 1, 20))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)
        sadf = get_sadf(logP, minSL=8, constant='nc', lags=1)
        direct = get_bsadf(logP, minSL=8, constant='nc', lags=1)
        assert sadf.iloc[-1] == pytest.approx(direct['gsadf'])

    def test_raises_on_too_short_series(self):
        dates = pd.bdate_range('2020-01-01', periods=5)
        logP = pd.DataFrame({'close': np.log(100 + np.arange(5.0))}, index=dates)
        with pytest.raises(ValueError, match="requires at least"):
            get_sadf(logP, minSL=10, constant='nc', lags=1)

    def test_explosive_series_sadf_rises_then_the_bubble_is_visible(self):
        # Build a series that's a random walk for the first half, then
        # switches to genuine exponential growth -- SADF should be
        # noticeably higher in the explosive second half than the first,
        # mirroring the book's own Figure 17.1 (SADF spikes during
        # bubble-like behavior).
        rng = np.random.default_rng(7)
        n_rw, n_explosive = 20, 20
        rw_part = 100 + np.cumsum(rng.normal(0, 1, n_rw))
        explosive_part = rw_part[-1] * np.exp(0.03 * np.arange(1, n_explosive + 1))
        prices = np.concatenate([rw_part, explosive_part])
        dates = pd.bdate_range('2020-01-01', periods=len(prices))
        logP = pd.DataFrame({'close': np.log(prices)}, index=dates)

        sadf = get_sadf(logP, minSL=10, constant='nc', lags=1)
        first_half_mean = sadf.iloc[:len(sadf) // 2].mean()
        second_half_mean = sadf.iloc[len(sadf) // 2:].mean()
        assert second_half_mean > first_half_mean
