"""
AFML Chapter 15, Section 15.3 -- Asymmetric Payouts.

Generalizes Sec 15.2: a bet now pays pi_+ with probability p, or pi_-
(pi_- < pi_+) with probability 1-p -- the profit-taking and stop-loss
levels no longer need to be mirror images of each other. Function names
(binHR, binFreq, binSR) are kept exactly as printed in the book (Snippets
15.3-15.4), per this project's convention of preserving book naming even
where it departs from PEP8 (matches clfHyperFit, mixGaussians, etc.
elsewhere in the repo).

File order matches the book's own printed order: binHR (Snippet 15.3),
then binFreq followed by binSR (both in Snippet 15.4) -- binFreq calls
binSR before binSR is defined further down the file, which is fine in
Python since both are resolved at call time, not at binFreq's def time.
"""

import numpy as np


def binHR(sl, pt, freq, tSR):
    """Snippet 15.3 -- implied precision.

    Given a trading rule characterized by {sl, pt, freq}, what's the min
    precision p required to achieve a Sharpe ratio tSR?

    Inputs
    ------
    sl   : stop loss threshold (pi_-, typically <= 0)
    pt   : profit taking threshold (pi_+, typically > 0, pt > sl)
    freq : number of bets per year
    tSR  : target annual Sharpe ratio

    Output
    ------
    p : the min precision rate p required to achieve tSR

    LOAD-BEARING: the book's own p=(-b+sqrt(b^2-4ac))/(2a) takes only the
    "+" root. If b^2-4ac < 0, Python's `(-1)**.5` silently returns a
    COMPLEX number rather than raising -- a real language-semantics trap
    (same category as the Ch5 tuple-assignment gotcha in CLAUDE.md). Added
    an explicit guard so this fails loudly instead of returning a
    nonsensical complex "precision." Symbolically (verified via sympy),
    disc = tSR^2*(pt-sl)^2*[tSR^2*(pt-sl)^2 - 4*freq*pt*sl] -- for the
    book's normal usage (sl<0<pt, so pt*sl<0) the bracket can never go
    negative and this guard is inert; it only fires in the pathological
    case where sl and pt share a sign. Confirmed inert on every worked
    example in this chapter and on the real-data usage in
    chapter_15_strategy_risk.py.
    """
    a = (freq + tSR ** 2) * (pt - sl) ** 2
    b = (2 * freq * sl - tSR ** 2 * (pt - sl)) * (pt - sl)
    c = freq * sl ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError(
            f'No real precision (0<=p<=1) achieves tSR={tSR} with '
            f'sl={sl}, pt={pt}, freq={freq} (negative discriminant).'
        )
    p = (-b + discriminant ** .5) / (2. * a)
    return p


def binFreq(sl, pt, p, tSR):
    """Snippet 15.4 (first half) -- implied betting frequency.

    Given a trading rule characterized by {sl, pt} and a precision rate p,
    what's the number of bets/year needed to achieve a Sharpe ratio tSR?

    Note (book's own): equation with radicals -- check for an extraneous
    solution. Returns None if the candidate freq, fed back through binSR,
    doesn't actually reproduce tSR (mirrors the book's bare `return` with
    no value, which is Python's implicit None).

    Concretely (found while writing this chapter's tests): squaring the
    equation to isolate freq loses the SIGN of expected profit. Below the
    break-even precision (p = -sl/(pt-sl), the same p_theta*=0 special
    case from binHR), expected profit is negative, so the candidate freq
    actually solves for -tSR rather than +tSR, and this function correctly
    returns None for any positive tSR request in that regime.

    Inputs
    ------
    sl   : stop loss threshold
    pt   : profit taking threshold
    p    : precision rate
    tSR  : target annual Sharpe ratio

    Output
    ------
    freq : number of bets per year needed, or None if the solution is
           extraneous
    """
    freq = (tSR * (pt - sl)) ** 2 * p * (1 - p) / ((pt - sl) * p + sl) ** 2
    if not np.isclose(binSR(sl, pt, freq, p), tSR):
        return None
    return freq


def binSR(sl, pt, freq, p):
    """Snippet 15.4 (second half) -- theta[p,n,pi_-,pi_+], Sec 15.3's main
    equation. Given a trading rule characterized by {sl, pt, freq} and a
    precision rate p, what's the resulting annualized Sharpe ratio?

    Inputs
    ------
    sl   : stop-loss threshold (pi_-)
    pt   : profit-taking threshold (pi_+)
    freq : number of bets per year (n)
    p    : precision rate

    Output
    ------
    Annualized Sharpe ratio (theta)
    """
    return ((pt - sl) * p + sl) / ((pt - sl) * (p * (1 - p)) ** .5) * freq ** .5


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
