"""
Snippet 5.2 -- Standard fractional differencing, EXPANDING WINDOW.

PLAIN-ENGLISH IDEA:
get_weights() (Snippet 5.1) gave us a recipe of weights. This function
actually APPLIES that recipe to a real price series. At every time t, it
takes a weighted sum of every value from the start of the series up to
and including t:

    X_t^(d) = w_0 * X_t + w_1 * X_{t-1} + w_2 * X_{t-2} + ... + w_t * X_0

The window of history used GROWS as t grows -- at the 5th data point you
use 5 weights, at the 100th you use 100 weights. That's the "expanding
window" in the snippet's name, and it's the slow/simple version (we'll
replace it with a fixed-width version in Snippets 5.3-5.4 for speed).

TWO SEPARATE SUBTLETIES THIS FUNCTION HANDLES:

(1) EARLY-POINT SKIPPING (the "skip" calculation):
    At the very start of the series there isn't enough history yet to
    use the FULL weight array -- you're forced to truncate it. If that
    truncation throws away more than `thres` (e.g. 1%) of the total
    weight mass, the resulting estimate is considered too unreliable
    and is skipped entirely (left out of the output) rather than
    reported as a biased number.
    Note: thres=1 means "skip nothing" -- every cumulative weight
    fraction is <= 1.0 by construction, so none can be STRICTLY
    greater than 1, and skip comes out to 0.

(2) FORWARD-FILL FOR COMPUTATION vs. EXCLUDING GAPS FROM OUTPUT:
    If the original series has a gap (NaN) in the middle, we still
    need *some* number there so later dot products don't break --
    so we forward-fill (carry the last known value forward) purely
    for computation. But we never claim a "real" fracDiff value at
    the position that was actually missing -- we check the ORIGINAL
    (non-filled) series and skip reporting output there, even though
    the filled value was used to keep later calculations going.
"""

import numpy as np
import pandas as pd
from get_weights import get_weights


def frac_diff(series, d: float, thres: float = 0.01):
    """
    Apply expanding-window fractional differencing to a series.

    Parameters
    ----------
    series : pd.Series or pd.DataFrame
        The price series (or several, as columns) to differentiate.
        A plain Series is accepted for convenience and an equivalent
        Series is returned; internally it's handled as a 1-column
        DataFrame to match the book's multi-column-capable snippet.
    d : float
        Order of fractional differencing (see get_weights docstring).
    thres : float, default 0.01
        Maximum acceptable weight-loss fraction for early points.
        thres=1 skips nothing (see Note 1 above).

    Returns
    -------
    Same type as the input (Series in -> Series out, DataFrame in ->
    DataFrame out). Early points that don't meet the weight-loss
    threshold, and any position where the ORIGINAL series had a gap,
    are simply absent from the result (not present as NaN rows -- the
    book's own pd.Series()/dict-assignment approach only ever writes
    entries it considers valid, so we follow that exactly).
    """
    input_was_series = isinstance(series, pd.Series)
    if input_was_series:
        series = series.to_frame(name=series.name if series.name is not None else 0)

    if series.index.duplicated().any():
        raise ValueError(
            "frac_diff requires a series with a UNIQUE index. Found "
            f"{series.index.duplicated().sum()} duplicate index labels "
            "(e.g. multiple ticks sharing the same timestamp). "
            "Deduplicate or reset to a positional index before calling "
            "-- e.g. series.reset_index(drop=True)."
        )

    # 1) Compute weights sized for the LONGEST possible window (the
    #    full series length). Early points will simply use a trailing
    #    SLICE of this same array -- see the -(iloc+1): slicing below.
    w = get_weights(d, series.shape[0])

    # 2) Determine how many initial points to skip based on how much
    #    weight-mass a truncated window would be missing.
    w_ = np.cumsum(np.abs(w))
    w_ /= w_[-1]
    skip = w_[w_ > thres].shape[0]

    # 3) Apply the weights to each column independently.
    out = {}
    for name in series.columns:
        # Forward-fill purely so the dot product below never hits a
        # gap (using modern .ffill() -- the book's fillna(method=
        # 'ffill') syntax is deprecated/removed in current pandas);
        # dropna() removes any UNFILLABLE leading NaNs (i.e. if the
        # series starts with NaN before any real value exists).
        series_filled = series[[name]].ffill().dropna()
        col_values = {}  # accumulate as a plain dict, then build the
                          # Series ONCE at the end -- growing a pd.Series
                          # via item assignment on an initially-EMPTY
                          # Series is fragile across pandas versions (it
                          # broke on pandas 3.0.1 with an IndexError,
                          # even though it worked fine on 3.0.2 -- not
                          # something worth depending on either way)

        for iloc in range(skip, series_filled.shape[0]):
            loc = series_filled.index[iloc]

            # Use the ORIGINAL (non-filled) series to decide whether
            # this position actually had real data -- a forward-filled
            # value is good enough to keep the math going, but not
            # good enough to report as a genuine fracDiff observation.
            if not np.isfinite(series.loc[loc, name]):
                continue

            # Use only the most recent (iloc+1) weights -- i.e. the
            # trailing slice of w, since w is ordered oldest->newest
            # and we only have iloc+1 historical points available.
            weights_window = w[-(iloc + 1):, :]
            history_window = series_filled.loc[:loc]
            col_values[loc] = np.dot(weights_window.T, history_window)[0, 0]

        out[name] = pd.Series(col_values, dtype=float)

    result = pd.concat(out, axis=1)

    if input_was_series:
        result = result.iloc[:, 0]
        result.name = series.columns[0]

    return result


# ---------------------------------------------------------------------
# TDD TEST RESULTS (frac_diff/test_ch05.py, frac_diff portion)
# Run 2026-06-26.
# ---------------------------------------------------------------------
# test_frac_diff_matches_hand_trace_thres_1            PASSED
# test_frac_diff_handles_nan_gap_correctly              PASSED
# test_skip_count_matches_independent_derivation        PASSED
# test_frac_diff_accepts_dataframe_multi_column         PASSED
# test_frac_diff_series_in_series_out                   PASSED
# test_frac_diff_rejects_duplicate_index                PASSED
#   (this last one was added after a REAL bug was found running this
#   function against real BTC tick data -- 561 duplicate timestamps
#   in 9205 trades broke series.loc[loc, name] before this guard was
#   added; see frac_diff.py's duplicate-index check above)
# ---------------------------------------------------------------------
