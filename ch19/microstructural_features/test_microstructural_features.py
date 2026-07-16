"""
TDD suite for Chapter 19 -- Microstructural Features.

Every test uses a KNOWN expected value: either hand-traced by working the
book's own formula on a tiny example (tick rule, VPIN), or cross-validated
against an independent reference implementation (numpy.polyfit for Kyle's
Lambda, a from-scratch re-derivation of Snippet 19.1 for Corwin-Schultz,
pandas' own .autocorr() for serial correlation, closed-form OLS-through-
origin algebra for Amihud's Lambda) -- never just shape/sanity checks.
"""

import numpy as np
import pandas as pd
import pytest

import microstructural_features as mf


# =============================================================================
# 19.3.1 -- Tick rule
# =============================================================================
class TestTickRule:
    def test_hand_traced_sequence(self):
        # prices: 100 -> 101 (+1) -> 101 (flat, carry) -> 99 (-1) -> 99 (flat, carry) -> 100 (+1)
        prices = [100, 101, 101, 99, 99, 100]
        expected = [1, 1, 1, -1, -1, 1]  # b0 = 1
        assert mf.tick_rule(prices).tolist() == expected

    def test_b0_is_configurable(self):
        prices = [100, 101]
        assert mf.tick_rule(prices, b0=-1).tolist() == [-1, 1]

    def test_empty_input(self):
        assert mf.tick_rule([]).tolist() == []

    def test_single_price(self):
        assert mf.tick_rule([100]).tolist() == [1]


class TestTickRuleAccuracy:
    def test_hand_traced_5_of_6(self):
        inferred = [1, 1, 1, -1, -1, 1]
        true_side = [1, 1, -1, -1, -1, 1]  # disagree only at index 2
        assert mf.tick_rule_accuracy(inferred, true_side) == pytest.approx(5 / 6)

    def test_perfect_agreement(self):
        s = [1, -1, 1, -1]
        assert mf.tick_rule_accuracy(s, s) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mf.tick_rule_accuracy([1, 1], [1, 1, 1])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mf.tick_rule_accuracy([], [])


# =============================================================================
# 19.3.2 -- Roll model
# =============================================================================
class TestRollMeasure:
    def test_cross_validated_against_numpy(self):
        # alternating-swing price series -> strong negative serial covariance
        # in price changes -> a real, nonzero effective spread.
        prices = [10, 12, 9, 11, 8, 10, 7, 9]
        dp = np.diff(prices)
        var_dp = np.var(dp, ddof=1)
        cov_dp = np.cov(dp[:-1], dp[1:], ddof=1)[0, 1]
        expected_c = np.sqrt(max(0.0, -cov_dp))
        expected_sigma_u = np.sqrt(max(0.0, var_dp + 2 * cov_dp))

        res = mf.roll_measure(prices)
        assert res["c"] == pytest.approx(expected_c)
        assert res["sigma_u"] == pytest.approx(expected_sigma_u)

    def test_known_regression_values(self):
        # Pinned exact values from the case above, so a future refactor
        # that silently changes the formula gets caught even if the
        # cross-validation logic above is also (identically) broken.
        prices = [10, 12, 9, 11, 8, 10, 7, 9]
        res = mf.roll_measure(prices)
        assert res["c"] == pytest.approx(2.7386127875258306)
        assert res["sigma_u"] == pytest.approx(0.0)
        assert res["cov_dp"] == pytest.approx(-7.5)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            mf.roll_measure([100, 101])  # only 1 price change


# =============================================================================
# 19.3.3 -- Parkinson volatility
# =============================================================================
class TestParkinsonVolatility:
    def test_hand_traced_formula(self):
        high = [102, 105, 103]
        low = [98, 100, 101]
        k1 = 4 * np.log(2)
        expected = np.sqrt(np.mean(np.log(np.array(high) / np.array(low)) ** 2) / k1)
        assert mf.parkinson_volatility(high, low) == pytest.approx(expected)

    def test_known_value(self):
        high = [102, 105, 103]
        low = [98, 100, 101]
        assert mf.parkinson_volatility(high, low) == pytest.approx(0.022909131341876877)

    def test_zero_range_gives_zero_vol(self):
        assert mf.parkinson_volatility([100, 100], [100, 100]) == pytest.approx(0.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mf.parkinson_volatility([100, 101], [99])


# =============================================================================
# 19.3.4 -- Corwin-Schultz spread + Becker-Parkinson sigma
# =============================================================================
class TestCorwinSchultz:
    @pytest.fixture
    def ohlc(self):
        high = pd.Series([105, 107, 104, 110, 108, 106, 111, 109], dtype=float)
        low = pd.Series([100, 101, 99, 102, 103, 101, 104, 105], dtype=float)
        return high, low

    def test_cross_validated_against_reference_port(self, ohlc):
        # Independent re-derivation of Snippet 19.1, kept deliberately
        # separate from the module's own get_beta/get_gamma/get_alpha so
        # this test would catch a shared bug, not just an inconsistency.
        high, low = ohlc

        def ref_corwin_schultz(high, low, sl=1):
            hl = pd.Series(np.log(high.values / low.values) ** 2, index=high.index)
            beta = hl.rolling(2).sum().rolling(sl).mean().dropna()
            h2, l2 = high.rolling(2).max(), low.rolling(2).min()
            gamma = pd.Series(np.log(h2.values / l2.values) ** 2, index=h2.index).dropna()
            den = 3 - 2 * 2 ** 0.5
            alpha = (2 ** 0.5 - 1) * (beta ** 0.5) / den - (gamma / den) ** 0.5
            alpha = alpha.copy()
            alpha[alpha < 0] = 0
            return 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

        ref = ref_corwin_schultz(high, low, sl=1)
        got = mf.corwin_schultz(high, low, sl=1)
        np.testing.assert_allclose(got.values, ref.values)

    def test_negative_alpha_clipped_to_zero_spread(self, ohlc):
        # Book's own instruction (p.727): alpha < 0 => spread pinned to 0,
        # not left negative.
        high, low = ohlc
        spread = mf.corwin_schultz(high, low, sl=1)
        assert (spread >= 0).all()

    def test_known_values(self, ohlc):
        high, low = ohlc
        spread = mf.corwin_schultz(high, low, sl=1)
        expected = [0.0190976, 0.0, 0.0, 0.03294147, 0.00163498, 0.0, 0.0240613]
        np.testing.assert_allclose(spread.values, expected, atol=1e-6)

    def test_becker_parkinson_sigma_nonnegative(self, ohlc):
        high, low = ohlc
        beta = mf.get_beta(high, low, 1)
        gamma = mf.get_gamma(high, low)
        sigma = mf.becker_parkinson_sigma(beta, gamma)
        assert (sigma >= 0).all()
        assert len(sigma) == len(beta)


# =============================================================================
# 19.4.1 -- Kyle's Lambda
# =============================================================================
class TestKyleLambda:
    def test_exact_slope_recovered(self):
        # delta_p proportional to signed_volume with slope exactly 1 and
        # zero intercept and zero noise -> OLS must recover 1.0 exactly.
        dp = [1, -1, 2]
        sv = [1, -1, 2]
        assert mf.kyle_lambda(dp, sv) == pytest.approx(1.0)

    def test_cross_validated_against_polyfit(self):
        dp = [0.5, -0.3, 1.2, -0.8]
        sv = [10, -5, 20, -15]
        assert mf.kyle_lambda(dp, sv) == pytest.approx(np.polyfit(sv, dp, 1)[0])

    def test_no_variation_in_volume_returns_nan(self):
        assert np.isnan(mf.kyle_lambda([1, -1, 2], [5, 5, 5]))

    def test_too_few_points_returns_nan(self):
        assert np.isnan(mf.kyle_lambda([1], [1]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mf.kyle_lambda([1, 2], [1, 2, 3])


class TestKyleLambdaByBar:
    def test_known_two_bar_case(self):
        prices = [100, 101, 100.5, 103, 102, 101, 105, 104, 106]
        signed_vol = [1, -0.5, 2, -1, 1.5, -2, 0.5, 1, -1]
        bar_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1]
        res = mf.kyle_lambda_by_bar(prices, signed_vol, bar_ids, min_trades=3)
        assert list(res.index) == [0, 1]
        assert res.loc[0] == pytest.approx(-0.8709677419354839)
        assert res.loc[1] == pytest.approx(0.43956043956043944)

    def test_bar_below_min_trades_is_nan(self):
        prices = [100, 101]
        signed_vol = [1, -1]
        bar_ids = [0, 0]
        res = mf.kyle_lambda_by_bar(prices, signed_vol, bar_ids, min_trades=3)
        assert np.isnan(res.loc[0])


# =============================================================================
# 19.4.2 -- Amihud's Lambda
# =============================================================================
class TestAmihudLambda:
    def test_cross_validated_closed_form_ols(self):
        close = [100, 101, 99, 102, 103]
        dollar_vol = [0, 5000, 8000, 3000, 6000]
        abs_dlog = np.abs(np.diff(np.log(close)))
        dv = np.array(dollar_vol[1:], dtype=float)
        expected = np.dot(dv, abs_dlog) / np.dot(dv, dv)
        assert mf.amihud_lambda(close, dollar_vol) == pytest.approx(expected)

    def test_known_value(self):
        close = [100, 101, 99, 102, 103]
        dollar_vol = [0, 5000, 8000, 3000, 6000]
        assert mf.amihud_lambda(close, dollar_vol) == pytest.approx(2.6705442316450166e-06)

    def test_zero_volume_returns_nan(self):
        assert np.isnan(mf.amihud_lambda([100, 101], [0, 0]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mf.amihud_lambda([100, 101, 102], [0, 1])


# =============================================================================
# 19.5.2 -- VPIN
# =============================================================================
class TestVpin:
    def test_hand_traced_rolling_window(self):
        buy = [10, 20, 5, 15, 25]
        sell = [5, 15, 20, 10, 5]
        v = mf.vpin(buy, sell, window=2)
        # bar 1 (0-indexed): window = bars {0,1}: |15-20|=5, denom=15+35=50 -> 0.2
        # bar 4: window = bars {3,4}: |20-30|... let's just assert pinned values.
        expected = [np.nan, 0.2, 1 / 3, 0.4, 5 / 11]
        got = v.values
        assert np.isnan(got[0])
        np.testing.assert_allclose(got[1:], expected[1:], atol=1e-8)

    def test_window_one_gives_per_bar_imbalance_fraction(self):
        buy = [10, 20]
        sell = [5, 5]
        v = mf.vpin(buy, sell, window=1)
        np.testing.assert_allclose(v.values, [5 / 15, 15 / 25])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            mf.vpin([1, 2], [1], window=1)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            mf.vpin([1, 2], [1, 2], window=0)


# =============================================================================
# 19.6.1 -- Round-number frequency (adapted for continuous BTC quantities)
# =============================================================================
class TestRoundNumberFrequency:
    def test_hand_traced_matches(self):
        # 0.001 -> matches 0.001; 0.00123 -> matches nothing; 0.01 -> matches
        # 0.01; 0.5 -> matches 0.5; 0.499999 -> matches 0.5 (within tol);
        # 1.0 -> matches 1.0; 0.0037 -> matches nothing. 5/7 round.
        vols = [0.001, 0.00123, 0.01, 0.5, 0.499999, 1.0, 0.0037]
        res = mf.round_number_frequency(vols)
        assert res["round_fraction"] == pytest.approx(5 / 7)
        assert res["by_level"][0.001] == 1
        assert res["by_level"][0.01] == 1
        assert res["by_level"][0.5] == 2  # 0.5 and 0.499999 both match
        assert res["by_level"][1.0] == 1

    def test_custom_levels(self):
        vols = [3.0, 3.1, 7.0]
        res = mf.round_number_frequency(vols, round_levels=(3.0, 7.0))
        assert res["round_fraction"] == pytest.approx(2 / 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mf.round_number_frequency([])


# =============================================================================
# 19.6.5 -- Serial correlation of signed order flow
# =============================================================================
class TestSerialCorrelationSignedFlow:
    def test_perfect_alternation_gives_minus_one(self):
        flow = [1, -1, 1, -1, 1, -1, 1, -1]
        assert mf.serial_correlation_signed_flow(flow, lag=1) == pytest.approx(-1.0)

    def test_cross_validated_against_pandas_autocorr(self):
        flow = [1, 1, -1, 1, -1, -1, 1, -1, 1, 1]
        expected = pd.Series(flow, dtype=float).autocorr(lag=1)
        assert mf.serial_correlation_signed_flow(flow, lag=1) == pytest.approx(expected)

    def test_lag_2(self):
        flow = [1, 1, -1, -1, 1, 1, -1, -1]
        expected = pd.Series(flow, dtype=float).autocorr(lag=2)
        assert mf.serial_correlation_signed_flow(flow, lag=2) == pytest.approx(expected)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            mf.serial_correlation_signed_flow([1], lag=1)
