"""
AFML Chapter 15, Section 15.4 -- The Probability of Strategy Failure.

Strategy risk != portfolio risk. Given a real (or simulated) series of bet
outcomes, this estimates P[p < p_theta*] -- the probability that the
strategy's true precision falls below the level needed to sustain a
target Sharpe ratio tSR. Sec 15.4.1 describes a full bootstrap procedure
(draw nk-sample blocks with replacement, refit p per draw, KDE the
resulting distribution) and then gives the large-k Gaussian approximation
to it: f[p] ~ N[p_bar, p_bar(1-p_bar)]. Snippet 15.5's probFailure()
implements exactly that closed-form approximation (not the literal
bootstrap loop) -- this module ports Snippet 15.5 only, matching what was
actually printed as code.

Function names (mixGaussians, probFailure) kept exactly as printed, per
project convention (see asymmetric.py's module docstring).
"""

import numpy as np
import scipy.stats as ss

from asymmetric import binHR


def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs, rng=None):
    """Snippet 15.5 -- mixGaussians.

    Random draws of bet outcomes from a mixture of two Gaussians: a
    "winning-bet" component (mu1, sigma1) drawn with weight prob1, and a
    "losing-bet" component (mu2, sigma2) drawn with weight 1-prob1. Used
    to synthesize a realistic-looking series of bet returns for testing
    probFailure against a known mixture, mirroring the book's own
    Sec 15.4.1 remark that {pi_-, pi_+} could alternatively be derived by
    fitting a two-Gaussian mixture (EF3M algorithm) instead of the simple
    conditional-mean estimator probFailure uses internally.

    Inputs
    ------
    mu1, sigma1 : mean/std of the first (winning-bet) component
    mu2, sigma2 : mean/std of the second (losing-bet) component
    prob1       : mixture weight of the first component, 0 < prob1 < 1
    nObs        : total number of draws
    rng         : numpy.random.Generator; a fresh default_rng() is
                  created if not provided. Book used the legacy global
                  np.random state -- ported to a Generator per project's
                  random_state convention (a shared Generator should be
                  threaded through repeated calls for reproducibility,
                  not reset per call).

    Output
    ------
    1-D numpy array of nObs simulated bet returns, shuffled
    """
    if rng is None:
        rng = np.random.default_rng()
    ret1 = rng.normal(mu1, sigma1, size=int(nObs * prob1))
    ret2 = rng.normal(mu2, sigma2, size=int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis=0)
    rng.shuffle(ret)
    return ret


def probFailure(ret, freq, tSR):
    """Snippet 15.5 -- probFailure, Sec 15.4/15.4.1's closed-form
    approximation. Estimates P[p < p_theta*] for a given series of bet
    returns.

    Estimates pi_- = E[{ret_t | ret_t <= 0}], pi_+ = E[{ret_t | ret_t>0}],
    and the empirical precision p = fraction of positive returns; derives
    p_theta* via binHR; and returns the Gaussian-approximation CDF of p
    evaluated at p_theta* -- the approximate probability that the true
    precision falls short of what's needed to sustain tSR.

    Inputs
    ------
    ret  : 1-D array-like of bet returns (real or simulated)
    freq : annualized number of bets, n = T/y
    tSR  : target annual Sharpe ratio

    Output
    ------
    Approximate P[p < p_theta*] (the strategy's probability of failure)

    LOAD-BEARING (confirmed real book bug, Ethan sign-off 2026-07-23): the
    book's printed line is
        risk=ss.norm.cdf(thresP,p,p*(1-p))
    scipy.stats.norm.cdf(x, loc, scale) documents `scale` as a STANDARD
    DEVIATION. Sec 15.4.1's own approximation is f[p] ~ N[p_bar,
    p_bar(1-p_bar)] -- p_bar(1-p_bar) is a VARIANCE (it's literally the
    Bernoulli variance formula), not a std. As printed, the book passes
    variance where scipy expects std. Fixed here to pass
    sqrt(p*(1-p)) instead. Same category of real printed-snippet bug as
    Ch5's tuple-assignment issue and Ch9's bagging-tuple-order bug --
    printed AFML code isn't assumed bug-free (CLAUDE.md gotcha).
    """
    ret = np.asarray(ret)
    rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
    p = ret[ret > 0].shape[0] / float(ret.shape[0])
    thresP = binHR(rNeg, rPos, freq, tSR)
    risk = ss.norm.cdf(thresP, p, (p * (1 - p)) ** 0.5)  # FIXED: sqrt (was p*(1-p))
    return risk


# =============================================================================
# TDD results -- embedded per project convention, after tests passed
# =============================================================================
# REAL-MACHINE CONFIRMED (Python 3.10.20, pytest 9.0.3, mlfinlab env,
# 2026-07-24). Suite covers all three ch15 modules (symmetric.py,
# asymmetric.py, algorithm.py) since they're tightly interdependent
# (asymmetric.binSR underlies symmetric's cross-check test;
# algorithm.probFailure calls asymmetric.binHR). Identical pass count,
# identical warnings, and identical numeric behavior to the sandbox run --
# nothing here was environment-sensitive.
#
# ============================= test session starts ==============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML\ch15\strategy_risk
# collected 34 items
#
# test_algorithm.py::TestProbFailureHandTraced::test_hand_traced_every_step PASSED [  2%]
# test_algorithm.py::TestProbFailureHandTraced::test_all_positive_returns PASSED [  5%]
# test_algorithm.py::TestProbFailureSeededRegression::test_book_parameters_seeded_regression PASSED [  8%]
# test_algorithm.py::TestProbFailureSeededRegression::test_fix_actually_changes_result_vs_literal_book_code PASSED [ 11%]
# test_algorithm.py::TestMixGaussians::test_output_length_matches_nObs PASSED [ 14%]
# test_algorithm.py::TestMixGaussians::test_reproducible_with_seeded_generator PASSED [ 17%]
# test_algorithm.py::TestMixGaussians::test_prob1_one_gives_pure_first_component PASSED [ 20%]
# test_asymmetric.py::TestBinSR::test_book_worked_example PASSED           [ 23%]
# test_asymmetric.py::TestBinSR::test_reduces_to_symmetric_case PASSED     [ 26%]
# test_asymmetric.py::TestBinSR::test_precision_half_gives_expected_value_only_pull PASSED [ 29%]
# test_asymmetric.py::TestBinHR::test_book_worked_example_theta_2 PASSED   [ 32%]
# test_asymmetric.py::TestBinHR::test_p_theta_star_zero_special_case PASSED [ 35%]
# test_asymmetric.py::TestBinHR::test_roundtrip_with_binsr PASSED          [ 38%]
# test_asymmetric.py::TestBinHR::test_negative_discriminant_raises PASSED  [ 41%]
# test_asymmetric.py::TestBinFreq::test_roundtrip_recovers_book_frequency PASSED [ 44%]
# test_asymmetric.py::TestBinFreq::test_roundtrip_with_binsr_general PASSED [ 47%]
# test_asymmetric.py::TestBinFreq::test_higher_precision_needs_fewer_bets PASSED [ 50%]
# test_asymmetric.py::TestBinFreq::test_extraneous_below_breakeven_returns_none PASSED [ 52%]
# test_asymmetric.py::TestBinFreq::test_at_or_above_breakeven_precision_has_valid_solution PASSED [ 55%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_book_worked_example_p55 PASSED [ 58%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_book_worked_example_396_bets PASSED [ 61%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_p_half_gives_zero_sharpe PASSED [ 64%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_symmetric_around_half PASSED [ 67%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_n_zero_gives_zero_sharpe PASSED [ 70%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_p_out_of_range_raises PASSED [ 73%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_negative_n_raises PASSED [ 76%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_book_worked_example_weekly_bets PASSED [ 79%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_roundtrip_against_sharpe_ratio_symmetric PASSED [ 82%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_higher_n_needs_lower_precision PASSED [ 85%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_zero_frequency_zero_target_raises PASSED [ 88%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_negative_n_raises PASSED [ 91%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_cross_validated_against_closed_form PASSED [ 94%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_reproducible_with_seeded_generator PASSED [ 97%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_p_one_gives_std_zero_and_nan_sharpe PASSED [100%]
#
# =============================== warnings summary ===============================
# test_algorithm.py::TestProbFailureHandTraced::test_all_positive_returns
#   C:\ws\AFML\ch15\strategy_risk\algorithm.py:97: RuntimeWarning: Mean of empty slice.
#
# test_algorithm.py::TestProbFailureHandTraced::test_all_positive_returns
#   RuntimeWarning: invalid value encountered in double_scalars
#
# -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
# ======================== 34 passed, 2 warnings in 1.99s ========================
