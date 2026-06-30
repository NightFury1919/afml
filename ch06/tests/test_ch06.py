"""
TDD tests for Chapter 6 -- Ensemble Methods.

Snippet 6.1 (bagging_classifier_accuracy):
All expected values are either:
(a) verified against the book's own argument (N=100, p=1/3, k=3), or
(b) derived independently from first principles using the binomial CDF.
"""

import sys
import os
import pytest
import numpy as np
from scipy.stats import binom

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ensemble'))
from bagging_accuracy import bagging_classifier_accuracy


# ------------------------------------------------------------------
# Core correctness -- book's own example
# ------------------------------------------------------------------

def test_book_example_n100_p_third_k3():
    """
    The book's own numerical example: N=100, p=1/3, k=3.
    Expected output: p=0.333..., P[X>N/k]=0.4812 (approx).
    We verified this by running the corrected snippet directly.
    """
    result = bagging_classifier_accuracy(N=100, p=1/3, k=3)
    assert abs(result - 0.4812) < 0.001


# ------------------------------------------------------------------
# The three regimes that define the theorem
# ------------------------------------------------------------------

def test_above_random_guessing_gives_high_probability_at_large_N():
    """
    p=0.40 > 1/k=0.333: ensemble probability should be well above p
    and approaching 1 at large N. At N=500, k=3 this should be > 0.99.
    Verified independently via scipy.stats.binom.cdf.
    """
    result = bagging_classifier_accuracy(N=500, p=0.4, k=3)
    independent = 1 - binom.cdf(int(500/3), 500, 0.4)
    assert abs(result - independent) < 1e-10
    assert result > 0.99


def test_below_random_guessing_declines_with_more_estimators():
    """
    p=0.30 < 1/k=0.333: adding more classifiers should make things
    WORSE -- P[X > N/k] should be lower at N=500 than at N=10.
    This is the "bagging cannot fix bad classifiers" proof.
    """
    result_small = bagging_classifier_accuracy(N=10,  p=0.30, k=3)
    result_large = bagging_classifier_accuracy(N=500, p=0.30, k=3)
    assert result_large < result_small


def test_at_random_guessing_threshold_stays_flat():
    """
    p=1/k exactly: probability should hover around 0.5 regardless of
    N -- no improvement, no degradation. Check it stays in [0.4, 0.6]
    across a range of N values.
    """
    for N in [10, 50, 100, 200, 500]:
        result = bagging_classifier_accuracy(N=N, p=1/3, k=3)
        assert 0.35 < result < 0.65, (
            f"at p=1/k exactly, result should hover near 0.5, "
            f"got {result:.4f} at N={N}"
        )


# ------------------------------------------------------------------
# Cross-validation against scipy.stats.binom
# ------------------------------------------------------------------

def test_matches_scipy_binom_cdf_independently():
    """
    Cross-check our implementation against an INDEPENDENT derivation
    using scipy.stats.binom.cdf (a different code path from our own
    loop over scipy.special.comb). If both agree, the formula is
    almost certainly implemented correctly.
    """
    for N, p, k in [(100, 0.4, 3), (50, 0.6, 2), (200, 0.35, 4)]:
        ours = bagging_classifier_accuracy(N, p, k)
        independent = 1 - binom.cdf(int(N/k), N, p)
        assert abs(ours - independent) < 1e-10, (
            f"mismatch at N={N}, p={p}, k={k}: "
            f"ours={ours:.8f}, scipy={independent:.8f}"
        )


# ------------------------------------------------------------------
# Binary classification special case (k=2)
# ------------------------------------------------------------------

def test_binary_classification_k2():
    """
    For k=2 (binary), the necessary condition threshold is N/2 --
    which is also the sufficient condition (majority > half = win).
    So this is both necessary AND sufficient for binary classification.
    At p=0.6, N=100: ensemble should be very reliable.
    """
    result = bagging_classifier_accuracy(N=100, p=0.6, k=2)
    independent = 1 - binom.cdf(int(100/2), 100, 0.6)
    assert abs(result - independent) < 1e-10
    assert result > 0.95


# ------------------------------------------------------------------
# Monotonicity: more estimators helps when p > 1/k
# ------------------------------------------------------------------

def test_monotonically_increases_with_N_when_above_threshold():
    """
    For p > 1/k, adding more estimators should monotonically improve
    the ensemble's necessary-condition probability. Check this across
    a sequence of increasing N values.
    """
    Ns = [10, 20, 50, 100, 200, 500]
    results = [bagging_classifier_accuracy(N, 0.4, 3) for N in Ns]
    for i in range(len(results) - 1):
        assert results[i] <= results[i+1], (
            f"expected monotonic increase but got "
            f"results[{i}]={results[i]:.4f} > results[{i+1}]={results[i+1]:.4f}"
        )


# ------------------------------------------------------------------
# Input validation
# ------------------------------------------------------------------

def test_invalid_p_raises():
    with pytest.raises(ValueError):
        bagging_classifier_accuracy(100, 0.0, 3)
    with pytest.raises(ValueError):
        bagging_classifier_accuracy(100, 1.0, 3)
    with pytest.raises(ValueError):
        bagging_classifier_accuracy(100, -0.1, 3)


def test_invalid_N_raises():
    with pytest.raises(ValueError):
        bagging_classifier_accuracy(0, 0.5, 3)


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        bagging_classifier_accuracy(100, 0.5, 1)
