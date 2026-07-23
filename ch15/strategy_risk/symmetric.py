"""
AFML Chapter 15, Section 15.2 -- Symmetric Payouts.

A strategy makes n IID bets/year. Each bet pays +pi with probability p
(precision) or -pi with probability 1-p. Because the payout is symmetric
(+pi vs -pi), pi cancels out of the Sharpe ratio algebraically -- the
annualized Sharpe depends only on precision (p) and betting frequency (n).
This is the economic basis for high-frequency trading: even a precision
barely above .5 can reach a high Sharpe ratio if n is large enough.
"""

import numpy as np


def sharpe_ratio_symmetric(p, n):
    """Book Sec 15.2, unnumbered equation just above the "t-value of p"
    brace: theta[p,n] = (2p-1) / (2*sqrt(p*(1-p))) * sqrt(n).

    theta[p,n] can be read as a re-scaled t-value of p under H0: p=1/2.

    Inputs
    ------
    p : precision (probability of a winning bet), 0 < p < 1
    n : number of IID bets per year, n >= 0

    Output
    ------
    Annualized Sharpe ratio (theta)
    """
    if not (0 < p < 1):
        raise ValueError('p must be strictly between 0 and 1')
    if n < 0:
        raise ValueError('n must be non-negative')
    return (2 * p - 1) / (2 * np.sqrt(p * (1 - p))) * np.sqrt(n)


def implied_precision_symmetric(n, tSR):
    """Book Sec 15.2: solving theta[p,n]=tSR for 0<=p<=1 gives
    -4p^2 + 4p - n/(tSR^2+n) = 0, with solution
    p = 1/2 * (1 + sqrt(1 - n/(tSR^2+n))).

    This is the precision needed, at a given betting frequency n, to hit
    a target annualized Sharpe ratio tSR.

    Inputs
    ------
    n   : number of IID bets per year, n >= 0
    tSR : target annualized Sharpe ratio

    Output
    ------
    p : the precision required to achieve tSR at frequency n
    """
    if n < 0:
        raise ValueError('n must be non-negative')
    denom = tSR ** 2 + n
    if denom == 0:
        # n=0 and tSR=0 simultaneously: theta is 0/0-undefined, no
        # meaningful "precision required" exists.
        raise ValueError('tSR and n cannot both be zero')
    return 0.5 * (1 + np.sqrt(1 - n / denom))


def simulate_symmetric_sharpe(p, n_draws=1_000_000, rng=None):
    """Snippet 15.1, ported to Python 3 and a reproducible
    numpy.random.Generator (book used the legacy global np.random state
    and Python 2's xrange/print).

    Empirically verifies sharpe_ratio_symmetric(p, n_draws) by directly
    simulating n_draws +-1 outcomes -- using +-1 rather than +-pi loses no
    generality since pi cancels out of the symmetric formula.

    Inputs
    ------
    p        : precision (probability of a +1 outcome)
    n_draws  : number of simulated bets (book uses 1,000,000)
    rng      : numpy.random.Generator; a fresh default_rng() is created if
               not provided (per project convention: pass a shared
               Generator when calling this repeatedly so draws don't
               silently reseed from the OS clock each call)

    Output
    ------
    (mean, std, sharpe) of the simulated +-1 outcome series
    """
    if rng is None:
        rng = np.random.default_rng()
    draws = rng.binomial(n=1, p=p, size=n_draws)
    out = np.where(draws == 1, 1, -1)
    mean, std = out.mean(), out.std()
    sharpe = mean / std if std > 0 else np.nan
    return mean, std, sharpe


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
