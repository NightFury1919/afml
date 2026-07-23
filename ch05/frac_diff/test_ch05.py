"""
TDD tests for Snippet 5.1 (get_weights), Chapter 5 -- Fractionally
Differentiated Features.

All expected values below are computed BY HAND using the recursive
formula from the book (w_0=1, w_k = -w_{k-1}/k * (d-k+1)), not just
shape/type checks. See get_weights.py's docstring for the full trace.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'frac_diff'))
from get_weights import get_weights
from frac_diff import frac_diff
from get_weights_ffd import get_weights_ffd
from frac_diff_ffd import frac_diff_ffd
from find_min_ffd import find_min_ffd, find_minimum_d


def test_d_0_4_matches_hand_trace():
    """
    d=0.4, size=5. Hand-traced weights (forward order, w_0 first):
        w_0 = 1
        w_1 = -1     * 0.4          = -0.4
        w_2 = -(-0.4)* -0.3         = -0.12
        w_3 = -(-0.12)* -0.533333.. = -0.064
        w_4 = -(-0.064)* -0.65      = -0.0416
    Returned array is REVERSED (oldest -> newest), so we expect:
        [-0.0416, -0.064, -0.12, -0.4, 1.0]
    """
    w = get_weights(0.4, 5).flatten()
    expected = np.array([-0.0416, -0.064, -0.12, -0.4, 1.0])
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_d_1_0_kills_everything_past_lag_1():
    """
    d=1.0 is ordinary (integer) differencing. By the formula, the term
    (d - k + 1) hits exactly zero once k = d + 1 = 2, and every weight
    from k=2 onward is forced to exactly zero (each w_k is built by
    multiplying the previous w_k, so once one term is zero, all
    subsequent terms inherit that zero). Only w_0=1 and w_1=-1 survive.
    Forward order: [1, -1, 0, 0, 0] -> reversed: [0, 0, 0, -1, 1]
    """
    w = get_weights(1.0, 5).flatten()
    expected = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_d_0_is_identity_no_differencing():
    """
    d=0 means "do nothing." (d - k + 1) at k=1 is (0 - 1 + 1) = 0, so
    w_1 = 0 and everything after is also 0. Only the w_0=1 (today's own
    value, unchanged) survives. Forward: [1,0,0,0,0] -> reversed:
    [0,0,0,0,1]
    """
    w = get_weights(0.0, 5).flatten()
    expected = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(w, expected, atol=1e-10)


def test_last_weight_is_always_one():
    """
    Regardless of d, w_0 = 1 always (it's the base case before any
    recursion happens). After reversal, w_0 sits at the LAST index.
    This should hold for any d, including values > 1.
    """
    for d in [0.0, 0.3, 0.5, 0.9, 1.0, 1.5, 2.0]:
        w = get_weights(d, 10).flatten()
        assert w[-1] == pytest.approx(1.0)


def test_weights_decay_in_magnitude_for_fractional_d():
    """
    For 0 < d < 1, the book's whole point is that weights shrink in
    magnitude as you go further into the past, but never hit exactly
    zero (unlike integer d). Check |w_k| is monotonically non-increasing
    as we move away from "today" (i.e. reading the reversed array from
    the end backwards, magnitudes should not increase).
    """
    w = get_weights(0.4, 8).flatten()
    abs_w = np.abs(w)
    # abs_w is oldest->newest; reverse to walk newest->oldest (today first)
    abs_w_from_today = abs_w[::-1]
    diffs = np.diff(abs_w_from_today)
    assert np.all(diffs <= 1e-12), (
        "expected |weight| to shrink monotonically as lag increases"
    )


def test_size_one_returns_just_w0():
    """Edge case: size=1 should just return [[1.0]] -- no history yet."""
    w = get_weights(0.5, 1)
    assert w.shape == (1, 1)
    assert w[0, 0] == pytest.approx(1.0)


def test_invalid_size_raises():
    """size must be >= 1; size=0 or negative should raise, not silently
    return garbage."""
    with pytest.raises(ValueError):
        get_weights(0.5, 0)


def test_output_shape_is_column_vector():
    """Snippet 5.1 explicitly reshapes to (-1, 1) -- a column vector --
    because it's later used as a matrix-multiplication-style weight
    vector against a column of prices. Confirm shape always matches
    (size, 1)."""
    for size in [1, 5, 20]:
        w = get_weights(0.3, size)
        assert w.shape == (size, 1)


# ---------------------------------------------------------------------
# Snippet 5.2 -- frac_diff (expanding window)
# ---------------------------------------------------------------------

def test_frac_diff_matches_hand_trace_thres_1():
    """
    Series [10, 11, 9, 12, 13], d=0.4, thres=1 (Note 1: nothing skipped).
    Expected values hand-derived using w = [-0.0416, -0.064, -0.12,
    -0.4, 1.0] (oldest->newest), dotted against the growing window of
    available history at each point:
        iloc=0: 1.0*10                                       = 10.0
        iloc=1: -0.4*10 + 1.0*11                              = 7.0
        iloc=2: -0.12*10 -0.4*11 + 1.0*9                      = 3.4
        iloc=3: -0.064*10 -0.12*11 -0.4*9 + 1.0*12            = 6.44
        iloc=4: -0.0416*10 -0.064*11 -0.12*9 -0.4*12 + 1.0*13 = 6.0
    """
    s = pd.Series([10, 11, 9, 12, 13])
    result = frac_diff(s, d=0.4, thres=1)
    expected = pd.Series([10.0, 7.0, 3.4, 6.44, 6.0])
    np.testing.assert_allclose(result.values, expected.values, atol=1e-10)
    assert len(result) == 5, "thres=1 must skip nothing (Note 1 in the book)"


def test_frac_diff_handles_nan_gap_correctly():
    """
    Series [10, 11, NaN, 12, 13], d=0.4, thres=1. The NaN at position 2
    must be (a) forward-filled to 11 purely so later dot products don't
    break, but (b) excluded from the OUTPUT entirely, since the
    original series genuinely had no value there.
    Hand-traced expected: index0=10, index1=7, index2=ABSENT,
    index3=5.64, index4=5.76 (see frac_diff.py docstring for the trace).
    """
    s = pd.Series([10, 11, np.nan, 12, 13])
    result = frac_diff(s, d=0.4, thres=1)

    assert 2 not in result.index, (
        "position with a genuine NaN in the original series must not "
        "appear in the output, even though it was forward-filled for "
        "computation purposes"
    )
    assert len(result) == 4
    np.testing.assert_allclose(result.loc[0], 10.0, atol=1e-10)
    np.testing.assert_allclose(result.loc[1], 7.0, atol=1e-10)
    np.testing.assert_allclose(result.loc[3], 5.64, atol=1e-10)
    np.testing.assert_allclose(result.loc[4], 5.76, atol=1e-10)


def test_skip_count_matches_independent_derivation():
    """
    The 'skip' mechanism is computed internally from cumsum(abs(w))
    normalized by the total. This test recomputes that SAME quantity
    independently (a fresh derivation, not just re-calling frac_diff)
    and checks frac_diff's actual output length matches it exactly --
    guards against off-by-one errors or a wrong comparison operator
    inside frac_diff.

    Also a useful teaching note: for a SHORT series (20 points) with
    the default thres=0.01, the threshold check is strict enough that
    17 of the 20 points get skipped -- only 3 remain. This mirrors the
    Chapter 4 lesson about threshold sensitivity: short series or
    aggressive thresholds can quietly throw away most of your data.
    """
    np.random.seed(42)
    s = pd.Series(np.cumsum(np.random.randn(20)) + 100)

    d, thres, size = 0.4, 0.01, len(s)
    w_indep = get_weights(d, size)
    cum = np.cumsum(np.abs(w_indep))
    cum /= cum[-1]
    expected_skip = cum[cum > thres].shape[0]

    result = frac_diff(s, d=d, thres=thres)
    assert len(result) == size - expected_skip
    assert expected_skip == 17, (
        "sanity-check the specific number for this seed/config; if "
        "this fails it means the underlying weights changed, not "
        "necessarily that frac_diff broke"
    )


def test_frac_diff_accepts_dataframe_multi_column():
    """The book's snippet operates on a DataFrame with possibly several
    columns, processing each independently. Confirm two columns with
    DIFFERENT values produce independently-correct results (not, e.g.,
    accidentally sharing state or using the wrong column's weights)."""
    df = pd.DataFrame({
        'a': [10, 11, 9, 12, 13],
        'b': [100, 110, 90, 120, 130],  # exactly 10x column 'a'
    })
    result = frac_diff(df, d=0.4, thres=1)
    np.testing.assert_allclose(result['a'].values, result['b'].values / 10, atol=1e-10)


def test_frac_diff_series_in_series_out():
    """Convenience wrapper: passing a Series should return a Series,
    not force the caller to unwrap a 1-column DataFrame."""
    s = pd.Series([10, 11, 9, 12, 13], name='price')
    result = frac_diff(s, d=0.4, thres=1)
    assert isinstance(result, pd.Series)
    assert result.name == 'price'


def test_frac_diff_rejects_duplicate_index():
    """
    BUG FOUND ON REAL DATA (BTC tick data, 2026-03-01 to 2026-03-31):
    561 of 9205 trades shared a duplicate timestamp (multiple trades
    executing in the same microsecond). series.loc[loc, name] with a
    duplicate index label returns a Series instead of a scalar, which
    breaks np.isfinite() with a cryptic pandas internals error instead
    of a clear message. This must fail LOUDLY with an actionable
    message instead.
    """
    s = pd.Series([10, 11, 9], index=[0, 0, 1])  # duplicate label 0
    with pytest.raises(ValueError, match="UNIQUE index"):
        frac_diff(s, d=0.4, thres=1)


# ---------------------------------------------------------------------
# Snippet 5.3 -- get_weights_ffd (fixed-width window weights)
# ---------------------------------------------------------------------

def test_get_weights_ffd_matches_hand_trace():
    """
    d=0.4, thres=0.01. Uses the SAME recursive formula as get_weights,
    but stops once a term's magnitude drops below thres rather than at
    a fixed count. Hand-traced (see chat): the 11th computed term
    (k=10) is the first to drop below 0.01 in magnitude (~0.0096) and
    gets discarded; the 10 terms before it (w_0 through w_9) survive.
    Final array should have 11 entries total (w_0 plus 10 appended).
    """
    w = get_weights_ffd(0.4, 0.01)
    assert len(w) == 11
    # Newest-end values must match get_weights' d=0.4 trace exactly,
    # since both share the identical recursive formula.
    np.testing.assert_allclose(
        w.flatten()[-4:], [-0.064, -0.12, -0.4, 1.0],
        atol=1e-10,
    )


def test_get_weights_ffd_last_weight_always_one():
    """w_0 = 1 always, regardless of d or thres."""
    for d in [0.1, 0.4, 0.9, 1.5]:
        w = get_weights_ffd(d, 0.01)
        assert w[-1, 0] == pytest.approx(1.0)


def test_get_weights_ffd_smaller_thres_keeps_more_weights():
    """A smaller (stricter) magnitude cutoff should keep MORE weights
    (wider window), since it tolerates smaller terms before stopping."""
    w_strict = get_weights_ffd(0.4, 0.001)
    w_loose = get_weights_ffd(0.4, 0.1)
    assert len(w_strict) > len(w_loose)


def test_get_weights_ffd_cross_checks_against_get_weights():
    """
    Independent cross-validation: get_weights_ffd(d, thres) should
    produce exactly the weights from get_weights(d, N) (for N large
    enough to contain all significant terms) that have magnitude
    >= thres, in the same order. This checks the FFD stopping rule
    against the already-verified fixed-count formula, rather than
    re-deriving the recursion by hand a second time.
    """
    d, thres = 0.4, 0.01
    w_ffd = get_weights_ffd(d, thres).flatten()

    w_full = get_weights(d, 50).flatten()  # 50 is plenty for this d/thres
    w_full_significant = w_full[np.abs(w_full) >= thres]

    np.testing.assert_allclose(w_ffd, w_full_significant, atol=1e-10)


# ---------------------------------------------------------------------
# Snippet 5.3 -- frac_diff_ffd (fixed-width window application)
# ---------------------------------------------------------------------

def test_frac_diff_ffd_matches_hand_trace():
    """
    Series [10, 11, 9, 12, 13], d=0.4, thres=0.1 -> weights
    [-0.12, -0.4, 1.0], width=2 (a 3-point fixed window).
    Hand-traced:
        iloc1=2: -0.12*10 -0.4*11 + 1*9   = 3.4
        iloc1=3: -0.12*11 -0.4*9  + 1*12  = 7.08
        iloc1=4: -0.12*9  -0.4*12 + 1*13  = 7.12
    Only 3 points in the output (first `width`=2 points have no full
    window available and are absent, not NaN-filled).
    """
    s = pd.Series([10, 11, 9, 12, 13])
    result = frac_diff_ffd(s, d=0.4, thres=0.1)
    assert list(result.index) == [2, 3, 4]
    np.testing.assert_allclose(result.values, [3.4, 7.08, 7.12], atol=1e-10)


def test_frac_diff_ffd_handles_nan_gap_correctly():
    """
    Series [10, 11, NaN, 12, 13], same d/thres as above (width=2).
    The NaN at position 2 is forward-filled to 11 for computation, but:
      - the window ENDING at position 2 (iloc1=2) is skipped entirely
        (original value missing there)
      - later windows that merely CONTAIN the filled-in value (e.g.
        iloc1=3's window spans positions 1-3, including the filled
        position 2) still compute fine, using the ffilled value
    Hand-traced: iloc1=3 -> -0.12*11 -0.4*11 + 1*12 = 6.28
                 iloc1=4 -> -0.12*11 -0.4*12 + 1*13 = 6.88
    """
    s = pd.Series([10, 11, np.nan, 12, 13])
    result = frac_diff_ffd(s, d=0.4, thres=0.1)
    assert list(result.index) == [3, 4]
    np.testing.assert_allclose(result.values, [6.28, 6.88], atol=1e-10)


def test_frac_diff_ffd_uses_fixed_width_for_every_point():
    """
    Unlike frac_diff() (expanding window), every computed point in
    frac_diff_ffd's output should be based on the SAME number of
    underlying weights/history points. We can't directly inspect the
    window length from the output, but we CAN check that the number
    of valid output points equals exactly (series length - width),
    with no points sacrificed to a separate weight-loss threshold
    (that concept doesn't apply here -- the threshold was already
    baked into building w).
    """
    s = pd.Series(np.arange(20, dtype=float))
    w = get_weights_ffd(0.4, 0.01)
    width = len(w) - 1
    result = frac_diff_ffd(s, d=0.4, thres=0.01)
    assert len(result) == len(s) - width


def test_frac_diff_ffd_series_in_series_out():
    """Same convenience contract as frac_diff(): Series in, Series out."""
    s = pd.Series([10, 11, 9, 12, 13], name='price')
    result = frac_diff_ffd(s, d=0.4, thres=0.1)
    assert isinstance(result, pd.Series)
    assert result.name == 'price'


def test_frac_diff_ffd_rejects_duplicate_index():
    """Same real-data bug guard as frac_diff() -- see that test's
    docstring for the full story (561 duplicate timestamps found in
    real BTC tick data)."""
    s = pd.Series([10, 11, 9], index=[0, 0, 1])
    with pytest.raises(ValueError, match="UNIQUE index"):
        frac_diff_ffd(s, d=0.4, thres=0.1)


def test_frac_diff_ffd_accepts_dataframe_multi_column():
    """Same multi-column independence guarantee as frac_diff()."""
    df = pd.DataFrame({
        'a': [10, 11, 9, 12, 13],
        'b': [100, 110, 90, 120, 130],
    })
    result = frac_diff_ffd(df, d=0.4, thres=0.1)
    np.testing.assert_allclose(result['a'].values, result['b'].values / 10, atol=1e-10)


# ---------------------------------------------------------------------
# Snippet 5.4 -- find_min_ffd / find_minimum_d (ADF-based d search)
# ---------------------------------------------------------------------
#
# These tests use a KNOWN-GROUND-TRUTH synthetic series: a true
# unit-root random walk (built by cumulative-summing random
# increments). At d=0 this series is, by construction, exactly a
# unit-root process -- the ADF test MUST fail to reject the null. At
# d=1, fractional differencing recovers something close to the
# original white-noise increments -- the ADF test MUST pass easily.
# This isn't just "looks reasonable" -- it's checking against a
# series whose true stationarity properties we engineered ourselves.

def _make_random_walk_series(seed=7, n=1000):
    rng = np.random.RandomState(seed)
    increments = rng.randn(n) * 0.01
    log_price = np.cumsum(increments)
    price = 100 * np.exp(log_price)
    return pd.Series(price)


def test_find_min_ffd_d0_fails_adf_on_true_random_walk():
    """At d=0, no differencing has occurred at all -- the series IS
    the log price, which is a true unit-root process by construction.
    The ADF test must fail to reject the unit-root null (p > 0.05),
    and correlation with the original log price must be exactly 1.0
    (d=0 means identity, see frac_diff_ffd's behavior at d=0)."""
    s = _make_random_walk_series()
    results = find_min_ffd(s, d_values=[0.0])
    row = results.loc[0.0]
    assert row['p_value'] > 0.05, "a true unit-root series at d=0 must fail ADF"
    assert row['corr'] == pytest.approx(1.0, abs=1e-6)


def test_find_min_ffd_d1_passes_adf_on_true_random_walk():
    """At d=1, fractional differencing recovers something close to
    the original white-noise increments, which IS stationary by
    construction. ADF must pass easily (very small p-value), and
    memory correlation with the original log price should have
    dropped sharply (most memory destroyed, as expected at d=1)."""
    s = _make_random_walk_series()
    results = find_min_ffd(s, d_values=[1.0])
    row = results.loc[1.0]
    assert row['p_value'] < 0.001, "true white noise at d=1 must pass ADF easily"
    assert row['corr'] < 0.1, "d=1 should have destroyed almost all memory"


def test_find_min_ffd_p_values_decrease_monotonically_for_clean_random_walk():
    """For this seeded random walk, p-values should fall steadily as
    d increases from 0 to 1 -- more differencing, more stationarity,
    lower p-value. (Real, messier financial data won't always be this
    clean, but a textbook random walk should behave exactly this way.)"""
    s = _make_random_walk_series()
    results = find_min_ffd(s)
    p_values = results['p_value'].values
    assert np.all(np.diff(p_values) <= 1e-12), (
        "expected p-values to be non-increasing as d rises for a "
        "clean synthetic random walk"
    )


def test_find_minimum_d_returns_smallest_passing_d():
    """find_minimum_d should return the smallest d in the table whose
    p-value clears the 0.05 threshold. For this exact seed/series,
    that's d=0.4 (verified by inspecting the full table by hand)."""
    s = _make_random_walk_series()
    results = find_min_ffd(s)
    min_d = find_minimum_d(results)
    assert min_d == pytest.approx(0.4)

    # Cross-check: every d below the returned minimum should NOT pass,
    # and every d from the returned minimum onward SHOULD pass --
    # guards against find_minimum_d picking an arbitrary passing row
    # instead of genuinely the smallest one.
    for d in results.index:
        if d < min_d - 1e-9:
            assert results.loc[d, 'p_value'] >= 0.05
        else:
            assert results.loc[d, 'p_value'] < 0.05


def test_find_minimum_d_returns_none_when_nothing_passes():
    """If every d in the tested range fails the ADF test, the function
    should return None rather than raising or returning a misleading
    value. A negative threshold is impossible to clear (p-values are
    never negative), guaranteeing nothing passes regardless of how
    small any individual p-value underflows to."""
    s = _make_random_walk_series()
    results = find_min_ffd(s)
    result = find_minimum_d(results, p_value_threshold=-1.0)
    assert result is None



