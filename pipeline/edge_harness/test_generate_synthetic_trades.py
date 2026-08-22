"""
test_generate_synthetic_trades.py

TDD tests for generate_synthetic_trades.py, per the project's TDD
convention (hand-traced known values, not just shape checks) and per
this harness's OWN explicit TDD requirement stated in the 2026-08-21
handoff: "verify it actually produces the intended imbalance/drift
correlation before trusting results from it."

All hand-traced values below were computed by Claude directly (numpy
2.4.4 / pandas 3.0.2, sandboxed) before this file was written, using
np.random.default_rng's PCG64 bit generator -- a stable algorithm across
numpy versions, so these should reproduce exactly on the real mlfinlab
environment (numpy 1.23.5). If any exact-value assertion fails on the
real machine, that is itself important information (either a numpy
version discrepancy in default_rng behavior worth knowing about, or a
real bug) -- paste the failure, don't just loosen the tolerance.

BUG FOUND DURING TDD (2026-08-22, see LOAD-BEARING comment in
generate_synthetic_trades.py): an earlier version of the generator tied
price drift to REALIZED imbalance deviation rather than to
edge_strength * z directly, which meant the null case (edge_strength=0)
mechanically injected a real drift response to pure sampling noise --
confirmed via a 200-seed hand-trace showing null-case correlation
averaging 0.0405 +/- SE 0.0032 (12.6 SEs from zero, NOT a true null).
Fixed by tying drift to edge_strength * z[i-1] directly. Test
`test_null_case_no_systematic_bias` below is the regression test for
this exact bug -- do not weaken it.
"""

import numpy as np
import pandas as pd
import pytest

from generate_synthetic_trades import generate_synthetic_trades


# ---------------------------------------------------------------------
# Schema / basic structure -- exact hand-traced values, seed=42
# ---------------------------------------------------------------------

def test_schema_and_trade_count():
    df = generate_synthetic_trades(
        n_windows=3, trades_per_window=5, edge_strength=0.0, seed=42
    )
    assert len(df) == 15
    assert list(df.columns) == ['Price', 'Volume', 'Timestamp', 'IsBuyerMaker']
    assert df['Price'].dtype == np.float64
    assert df['Volume'].dtype == np.float64
    assert df['IsBuyerMaker'].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(df['Timestamp'])
    assert df['Timestamp'].is_monotonic_increasing
    assert (df['Volume'] > 0).all()


def test_hand_traced_exact_values_seed42():
    """
    Exact values hand-traced 2026-08-22 for n_windows=3, trades_per_window=5,
    edge_strength=0.0, seed=42. Reproducing these exactly on the real
    machine confirms default_rng(42) behaves identically across the
    sandbox's numpy 2.4.4 and the real mlfinlab environment's numpy
    1.23.5 -- PCG64 is a stable algorithm, so this SHOULD match exactly,
    but confirm rather than assume.
    """
    df = generate_synthetic_trades(
        n_windows=3, trades_per_window=5, edge_strength=0.0, seed=42
    )
    expected_prices = [
        67011.99482007876, 66921.62436592848, 66990.02963220625,
        67115.40338248656, 67102.7493783155, 67009.44897668631,
        66917.88933194272, 66990.13845465443, 67072.67773702306,
        67132.99568108356, 67059.09001626901, 67084.8718143969,
        67097.82991584083, 67122.11555127426, 67218.88878007533,
    ]
    expected_is_buyer_maker = [
        True, False, False, False, True, False, False, False,
        True, True, True, True, False, False, False,
    ]
    np.testing.assert_allclose(df['Price'].values, expected_prices, rtol=1e-9)
    assert df['IsBuyerMaker'].tolist() == expected_is_buyer_maker
    # First timestamp should be exactly start_timestamp (elapsed_sec[0] = 0)
    assert df['Timestamp'].iloc[0] == pd.Timestamp('2026-03-01')


# ---------------------------------------------------------------------
# The core TDD requirement: verify the injected edge is real, and that
# the null case is genuinely null. This is the regression test for the
# real bug found and fixed 2026-08-22.
# ---------------------------------------------------------------------

def _lag1_corr(n_windows, trades_per_window, edge_strength, seed):
    """Helper: correlation between window i's realized imbalance and
    the price drift from window i to window i+1 (drift_next[i])."""
    df, diag = generate_synthetic_trades(
        n_windows=n_windows, trades_per_window=trades_per_window,
        edge_strength=edge_strength, seed=seed, return_diagnostics=True,
    )
    imb = diag['realized_imbalance']
    wmp = diag['window_mean_price']
    drift_next = np.diff(wmp)
    imb_lead = imb[:-1]
    return np.corrcoef(imb_lead, drift_next)[0, 1]


def test_null_case_no_systematic_bias():
    """
    REGRESSION TEST for the 2026-08-22 bug: at edge_strength=0.0, the
    lag-1 correlation between window imbalance and next-window drift
    must be statistically indistinguishable from zero, averaged over
    many seeds. Hand-traced 2026-08-22 (post-fix): mean=-0.00335,
    SE=0.00320 over 200 seeds (mean/SE = -1.05). Assert |mean/SE| < 3
    as a generous statistical-significance bound -- the pre-fix bug
    produced mean/SE = 12.59, so this bound cleanly separates "real bug"
    from "normal sampling noise."
    """
    n_windows, trades_per_window = 500, 50
    corrs = np.array([
        _lag1_corr(n_windows, trades_per_window, 0.0, seed=s)
        for s in range(200)
    ])
    mean_c = corrs.mean()
    se_c = corrs.std() / np.sqrt(len(corrs))
    assert abs(mean_c / se_c) < 3.0, (
        f"Null case shows a statistically significant correlation "
        f"(mean={mean_c:.5f}, SE={se_c:.5f}, mean/SE={mean_c/se_c:.2f}) "
        f"-- this is the exact bug pattern found and fixed 2026-08-22. "
        f"Do not weaken this test; find and fix the actual cause."
    )


def test_edge_strength_increases_correlation_fixed_seed():
    """
    Hand-traced exact values, seed=123, n_windows=500, trades_per_window=50
    -- same underlying z/noise draws reused across edge_strength values
    (edge_strength doesn't change RNG call order/count, only drift
    magnitude), isolating edge_strength's effect cleanly. Verified
    strictly increasing on 2026-08-22:
        edge_strength=0.0  -> corr=0.014875
        edge_strength=0.05 -> corr=0.038492
        edge_strength=0.1  -> corr=0.078436
        edge_strength=0.2  -> corr=0.118876
        edge_strength=0.4  -> corr=0.213538
    """
    seed = 123
    edge_strengths = [0.0, 0.05, 0.1, 0.2, 0.4]
    expected = [0.014875, 0.038492, 0.078436, 0.118876, 0.213538]
    observed = [
        _lag1_corr(500, 50, es, seed=seed) for es in edge_strengths
    ]
    np.testing.assert_allclose(observed, expected, atol=1e-4)
    assert all(
        observed[i] <= observed[i + 1] for i in range(len(observed) - 1)
    ), f"Correlation should be non-decreasing in edge_strength, got {observed}"


def test_positive_edge_case_detectable_on_average():
    """
    At a clearly nonzero edge_strength (0.2), averaged over 30 seeds,
    the injected imbalance-drift relationship should be robustly
    positive and well above the null case's noise floor. Hand-traced
    2026-08-22: mean=0.1234, std=0.0522 over 30 seeds.
    """
    corrs = np.array([
        _lag1_corr(500, 50, 0.2, seed=s) for s in range(30)
    ])
    assert corrs.mean() > 0.08, (
        f"Expected robustly positive correlation at edge_strength=0.2, "
        f"got mean={corrs.mean():.4f}"
    )


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def test_seed_reproducibility():
    df1 = generate_synthetic_trades(
        n_windows=10, trades_per_window=20, edge_strength=0.1, seed=7
    )
    df2 = generate_synthetic_trades(
        n_windows=10, trades_per_window=20, edge_strength=0.1, seed=7
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ():
    df1 = generate_synthetic_trades(
        n_windows=10, trades_per_window=20, edge_strength=0.1, seed=1
    )
    df2 = generate_synthetic_trades(
        n_windows=10, trades_per_window=20, edge_strength=0.1, seed=2
    )
    assert not df1['Price'].equals(df2['Price'])


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))


# =======================================================================
# REAL-MACHINE PYTEST CONFIRMATION (mlfinlab conda env)
# =======================================================================
# Pass 1 -- run from repo root (C:\ws\AFML), 2026-08-22:
#
#   platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
#   collected 7 items
#
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_schema_and_trade_count PASSED                          [ 14%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_hand_traced_exact_values_seed42 PASSED                 [ 28%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_null_case_no_systematic_bias PASSED                    [ 42%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_edge_strength_increases_correlation_fixed_seed PASSED  [ 57%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_positive_edge_case_detectable_on_average PASSED        [ 71%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_seed_reproducibility PASSED                            [ 85%]
#   pipeline/edge_harness/test_generate_synthetic_trades.py::test_different_seeds_differ PASSED                          [100%]
#
#   7 passed in 3.54s
#
# Notably: test_hand_traced_exact_values_seed42 PASSED on real hardware
# (numpy 1.23.5) using values hand-traced in a sandbox on numpy 2.4.4 --
# confirms np.random.default_rng's PCG64 bit generator is exactly
# reproducible across these two numpy versions for this call pattern.
#
# Pass 2 -- run from inside pipeline/edge_harness/, 2026-08-22:
#
#   platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
#   rootdir: C:\ws\AFML\pipeline\edge_harness
#   collected 7 items
#
#   test_schema_and_trade_count PASSED                          [ 14%]
#   test_hand_traced_exact_values_seed42 PASSED                 [ 28%]
#   test_null_case_no_systematic_bias PASSED                    [ 42%]
#   test_edge_strength_increases_correlation_fixed_seed PASSED  [ 57%]
#   test_positive_edge_case_detectable_on_average PASSED        [ 71%]
#   test_seed_reproducibility PASSED                            [ 85%]
#   test_different_seeds_differ PASSED                          [100%]
#
#   7 passed in 3.54s
#
# Two-pass real-machine confirmation complete: 7/7 both directions,
# identical results (repo-root pass and module-folder pass both green,
# no path-dependent import issues).
# =======================================================================
