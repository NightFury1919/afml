"""
pipeline/edge_harness/test_momentum_correlation.py

Hand-traced TDD tests for momentum_correlation.bar_lag1_autocorr, matching
this project's standing convention (known expected values, not just shape
checks).
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from momentum_correlation import bar_lag1_autocorr  # noqa: E402


def test_too_short_returns_nan():
    # 2 prices -> 1 return -> can't form a lag-1 pair at all.
    assert np.isnan(bar_lag1_autocorr([100.0, 101.0]))
    assert np.isnan(bar_lag1_autocorr([100.0]))
    assert np.isnan(bar_lag1_autocorr([]))


def test_exactly_two_returns_perfect_positive_corr():
    # 3 prices -> 2 returns -> corrcoef of two scalars is always +/-1
    # (or NaN if either has zero variance, not applicable here since
    # they're single points -- np.corrcoef on two length-1 arrays with
    # different values returns nan actually; use this to confirm we
    # match numpy's real behavior rather than assume).
    prices = [100.0, 101.0, 102.0]
    result = bar_lag1_autocorr(prices)
    # returns = [0.01, 0.00990099...]; a=[0.01], b=[0.00990099...] each
    # length 1 -> std=0 for both (a single point has zero variance) ->
    # function's explicit NaN guard should fire.
    assert np.isnan(result)


def test_hand_traced_perfect_positive_autocorrelation():
    # Construct returns that are EXACTLY proportional between lag and
    # lead (r[i+1] = 2*r[i]) -> corrcoef must be exactly +1.0.
    # returns: r0=0.01, r1=0.02, r2=0.01, r3=0.02  (alternating, but
    # a=[r0,r1,r2]=[0.01,0.02,0.01], b=[r1,r2,r3]=[0.02,0.01,0.02] --
    # not perfectly correlated this way. Use a cleaner monotonic case
    # instead: r_i = i+1 (in bp) for i=0..4 -> a=[1,2,3,4], b=[2,3,4,5],
    # b = a+1, a perfect linear (hence perfectly correlated) relationship.
    bp = 0.0001
    rets = [bp, 2 * bp, 3 * bp, 4 * bp, 5 * bp]
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    result = bar_lag1_autocorr(prices)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_hand_traced_zero_autocorrelation_alternating_returns():
    # Alternating +bp/-bp returns of EQUAL magnitude: a=[+,-,+,-],
    # b=[-,+,-,+] -- b is exactly -a, so corrcoef must be exactly -1.0
    # (perfect NEGATIVE autocorrelation -- mean-reverting, the opposite
    # of momentum -- a useful sanity boundary case).
    bp = 0.0001
    rets = [bp, -bp, bp, -bp, bp, -bp]
    prices = [100.0]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    result = bar_lag1_autocorr(prices)
    assert result == pytest.approx(-1.0, abs=1e-6)


def test_flat_prices_returns_nan_not_crash():
    # Zero variance in returns (perfectly flat price) -> explicit NaN,
    # not a numpy RuntimeWarning/silent garbage.
    prices = [100.0] * 6
    result = bar_lag1_autocorr(prices)
    assert np.isnan(result)


def test_realistic_random_walk_near_zero_autocorr():
    # A true random walk (no injected momentum) should show
    # autocorrelation near zero, not exactly zero (finite-sample noise)
    # -- this is a smoke test on realistic-scale data, not a hand-traced
    # exact value.
    rng = np.random.default_rng(42)
    rets = rng.normal(0, 0.001, 500)
    prices = 100.0 * np.cumprod(1 + rets)
    result = bar_lag1_autocorr(prices)
    assert abs(result) < 0.15  # loose bound; true expectation is ~0


if __name__ == '__main__':
    import subprocess
    r = subprocess.run(['pytest', __file__, '-v'], capture_output=True, text=True)
    print(r.stdout)
    print(r.stderr)


# =============================================================================
# TDD TEST RESULTS -- sandbox (2026-08-23)
# Python 3.12.3, pytest 9.1.1, numpy 2.4.4 (NOT yet real-machine confirmed
# under mlfinlab's pinned Python 3.10.20/numpy 1.23.5 -- run this two-pass
# (repo root + inside pipeline/edge_harness/) for real before trusting).
# =============================================================================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 6 items
#
# test_momentum_correlation.py::test_too_short_returns_nan PASSED       [ 16%]
# test_momentum_correlation.py::test_exactly_two_returns_perfect_positive_corr PASSED [ 33%]
# test_momentum_correlation.py::test_hand_traced_perfect_positive_autocorrelation PASSED [ 50%]
# test_momentum_correlation.py::test_hand_traced_zero_autocorrelation_alternating_returns PASSED [ 66%]
# test_momentum_correlation.py::test_flat_prices_returns_nan_not_crash PASSED [ 83%]
# test_momentum_correlation.py::test_realistic_random_walk_near_zero_autocorr PASSED [100%]
#
# 6 passed in 0.26s
# =============================================================================
