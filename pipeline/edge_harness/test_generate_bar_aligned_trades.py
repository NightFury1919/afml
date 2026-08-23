"""
test_generate_bar_aligned_trades.py

TDD tests for generate_bar_aligned_trades.py. Per this project's TDD
convention (hand-traced known values) and this harness's own established
requirement (verify the injected edge is real and the null case is
genuinely null before trusting downstream results).

TESTING NOTE (2026-08-22): this test file uses LOCAL, test-only, VERBATIM
copies of rebuild.compute_dynamic_threshold() and features._retag_
trades_with_bar_id() (see conftest-style stub setup below), because the
real versions require ch02/ch03/ch04/ch19 chapter modules that this
sandbox doesn't have. On the real mlfinlab machine, this test file should
import the REAL rebuild.py/features.py directly (see the module-level
IMPORT SWITCH below) -- the stubs exist ONLY to let Claude verify this
module's own logic (window-finding, variable-window injection math,
null-case unbiasedness) before handing it off, not as a permanent
substitute for the real integration.

All hand-traced values below were computed 2026-08-22 using np.random.
default_rng (PCG64, stable across numpy versions -- see
test_generate_synthetic_trades.py's own note on this, confirmed to
reproduce exactly on the real machine for that generator).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------
# IMPORT SWITCH: on the real mlfinlab machine, generate_bar_aligned_
# trades.py already imports the REAL rebuild.compute_dynamic_threshold
# and features._retag_trades_with_bar_id (see that module's own imports,
# which point at pipeline/orchestration). This test file does not need
# to stub anything on the real machine -- it can just import
# generate_bar_aligned_trades directly. The sys.path manipulation below
# is ONLY needed in a sandbox without ch02/ch03/ch04/ch19 available.
# ---------------------------------------------------------------------
from generate_bar_aligned_trades import (
    generate_bar_aligned_synthetic_trades,
    _find_bar_windows,
)


# ---------------------------------------------------------------------
# Schema / structure -- exact hand-traced values, n_trades=2000,
# target_bars=15, edge_strength=0.15, seed=7
# ---------------------------------------------------------------------

def test_schema_and_dtypes():
    raw, diag = generate_bar_aligned_synthetic_trades(
        n_trades=2000, target_bars=15, edge_strength=0.15, seed=7,
        return_diagnostics=True,
    )
    assert list(raw.columns) == [
        'TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
        'IsBuyerMaker', 'IsBestMatch',
    ]
    assert raw['TradeID'].dtype == np.int64
    assert raw['Price'].dtype == np.float64
    assert raw['Timestamp'].dtype == np.int64
    assert raw['IsBuyerMaker'].dtype == bool
    assert (raw['TradeID'].diff().dropna() > 0).all()
    assert (raw['Timestamp'].diff().dropna() > 0).all()


def test_hand_traced_window_structure_seed7():
    """
    Hand-traced 2026-08-22: n_trades=2000, target_bars=15, edge_strength=
    0.15, seed=7 -- 14 complete bars found (target was 15; dynamic
    threshold rounding means this won't be exact), variable window sizes
    ranging from 117 to 158 trades (NOT constant, confirming the
    bar-aligned injection is genuinely using variable-length windows,
    unlike generate_synthetic_trades.py's fixed trades_per_window).
    """
    raw, diag = generate_bar_aligned_synthetic_trades(
        n_trades=2000, target_bars=15, edge_strength=0.15, seed=7,
        return_diagnostics=True,
    )
    assert diag['n_windows'] == 14
    assert diag['n_used_trades'] == 1871
    expected_window_sizes = [124, 117, 122, 138, 125, 141, 158, 141,
                              133, 142, 135, 145, 131, 119]
    assert diag['window_sizes'].tolist() == expected_window_sizes
    # Windows are genuinely variable-length, not all equal -- this is
    # the whole point of the bar-aligned redesign.
    assert diag['window_sizes'].min() != diag['window_sizes'].max()
    np.testing.assert_allclose(diag['threshold'], 38236.394364822336, rtol=1e-9)


def test_first_prices_hand_traced_seed7():
    raw, diag = generate_bar_aligned_synthetic_trades(
        n_trades=2000, target_bars=15, edge_strength=0.15, seed=7,
        return_diagnostics=True,
    )
    expected_first_5 = [
        67082.62759715386, 67086.08711911357, 67103.3730815882,
        67002.03934993097, 66975.30383475256,
    ]
    np.testing.assert_allclose(raw['Price'].head(5).values, expected_first_5, rtol=1e-9)


# ---------------------------------------------------------------------
# Core TDD requirement: null case genuinely null, edge scales correctly.
# Same category of regression coverage as generate_synthetic_trades.py's
# 2026-08-22 bug (a real bug was found and fixed THERE; this is new code
# with its own chance of the same mistake, so it needs its own from-
# scratch verification, not an assumption that "the same fix pattern
# was used so it must be fine").
# ---------------------------------------------------------------------

def _lag1_corr(edge_strength, seed, n_trades=20000, target_bars=150):
    raw, diag = generate_bar_aligned_synthetic_trades(
        n_trades=n_trades, target_bars=target_bars,
        edge_strength=edge_strength, seed=seed, return_diagnostics=True,
    )
    imb = diag['realized_imbalance']
    wmp = diag['window_mean_price']
    drift_next = np.diff(wmp)
    return np.corrcoef(imb[:-1], drift_next)[0, 1]


def test_null_case_no_systematic_bias():
    """
    Hand-traced 2026-08-22 (bar-aligned generator, 60 seeds): mean=
    0.00592, SE=0.00879, mean/SE=0.67 -- comfortably within normal
    sampling noise. Bound of 3.0 chosen the same way as the trade-count-
    window generator's equivalent test (its pre-fix bug produced
    mean/SE=12.59, so this threshold cleanly separates "real bug" from
    "normal noise").
    """
    corrs = np.array([_lag1_corr(0.0, s) for s in range(60)])
    mean_c = corrs.mean()
    se_c = corrs.std() / np.sqrt(len(corrs))
    assert abs(mean_c / se_c) < 3.0, (
        f"Null case shows a statistically significant correlation "
        f"(mean={mean_c:.5f}, SE={se_c:.5f}, mean/SE={mean_c/se_c:.2f}) "
        f"in the BAR-ALIGNED generator -- same bug category as the "
        f"2026-08-22 fix in generate_synthetic_trades.py. Investigate "
        f"the per-window drift injection loop."
    )


def test_edge_strength_increases_correlation():
    """
    Hand-traced 2026-08-22, 15 seeds per edge_strength: mean correlation
    increases monotonically with edge_strength (0.0043 -> 0.0470 ->
    0.1566 -> 0.2860 -> 0.3947 for edge_strength in
    [0.0, 0.1, 0.3, 0.6, 1.0]).
    """
    edge_strengths = [0.0, 0.1, 0.3, 0.6, 1.0]
    means = []
    for es in edge_strengths:
        corrs = np.array([_lag1_corr(es, s) for s in range(15)])
        means.append(corrs.mean())
    assert all(means[i] <= means[i + 1] for i in range(len(means) - 1)), (
        f"Expected non-decreasing mean correlation with edge_strength, "
        f"got {list(zip(edge_strengths, means))}"
    )
    assert means[-1] > 0.3, (
        f"Expected a strong, clearly-detectable correlation at "
        f"edge_strength=1.0, got {means[-1]:.4f}"
    )


def test_raises_on_too_few_bars():
    """
    A configuration producing fewer than 2 complete bars should raise a
    clear error rather than silently proceeding (can't inject a lag-1
    imbalance/drift relationship with fewer than 2 windows).

    LOAD-BEARING (2026-08-22): target_bars=1, NOT a large target_bars
    with few trades. compute_dynamic_threshold() divides total dollar
    volume by target_bars -- a LARGE target_bars with few trades
    produces a SMALL threshold, which means bars form FAST (many tiny
    bars from few trades), the opposite of "too few bars". An earlier
    version of this test used n_trades=50/target_bars=1000 expecting
    that to starve bar formation -- it didn't (confirmed: that config
    actually produces MANY bars). target_bars=1 reliably produces ZERO
    complete bars instead: the bar_id increment only fires AFTER the
    threshold is crossed, and with threshold=total dollar volume, that
    crossing happens only on the very last trade -- which is already
    tagged bar_id=0 before the increment, so bar_id never reaches 1 for
    any trade, and trades['bar_id'].max()==0 means the >0 completeness
    filter drops every trade. Verified directly against
    _find_bar_windows() (2026-08-22) before writing this assertion,
    not assumed.
    """
    with pytest.raises(ValueError, match='complete bar'):
        generate_bar_aligned_synthetic_trades(
            n_trades=50, target_bars=1, edge_strength=0.1, seed=1,
        )


def test_seed_reproducibility():
    raw1 = generate_bar_aligned_synthetic_trades(
        n_trades=5000, target_bars=40, edge_strength=0.1, seed=3,
    )
    raw2 = generate_bar_aligned_synthetic_trades(
        n_trades=5000, target_bars=40, edge_strength=0.1, seed=3,
    )
    pd.testing.assert_frame_equal(raw1, raw2)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# =======================================================================
# SANDBOX pytest confirmation, 2026-08-22 (NOT real-machine, NOT the
# real rebuild.py/features.py -- see module docstring's IMPORT SWITCH
# note. Uses verbatim test-only stubs of compute_dynamic_threshold() and
# _retag_trades_with_bar_id() because ch02/ch03/ch04/ch19 aren't
# available in this sandbox. This confirms generate_bar_aligned_trades.py's
# OWN logic -- window-finding, variable-window injection math, null-case
# unbiasedness -- but NOT that it correctly integrates with the actual
# real rebuild.py/features.py imports on the real machine.):
#
#   platform linux -- Python 3.12.3, pytest-9.1.1
#   collected 7 items
#
#   test_schema_and_dtypes PASSED                          [ 14%]
#   test_hand_traced_window_structure_seed7 PASSED         [ 28%]
#   test_first_prices_hand_traced_seed7 PASSED             [ 42%]
#   test_null_case_no_systematic_bias PASSED               [ 57%]
#   test_edge_strength_increases_correlation PASSED        [ 71%]
#   test_raises_on_too_few_bars PASSED                      [ 85%]
#   test_seed_reproducibility PASSED                        [100%]
#
#   7 passed in 4.63s
#
# REAL-MACHINE CONFIRMATION: [PENDING -- run on mlfinlab env with the
# real rebuild.py/features.py imports, not the sandbox stubs]
# =======================================================================
