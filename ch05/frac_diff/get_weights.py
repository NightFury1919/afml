"""
Snippet 5.1 -- Weighting function for fractional differentiation.

PLAIN-ENGLISH IDEA (read this before the math):
Fractionally differencing a series means replacing each value with a
weighted combination of itself and ALL prior values:

    X_t^(d) = w_0 * X_t + w_1 * X_{t-1} + w_2 * X_{t-2} + ...

where the weights w_k come from expanding (1 - B)^d using the binomial
series (B is the "backshift" operator, B * X_t = X_{t-1}). This function
computes that list of weights for a chosen d and a chosen number of terms
("size"). It does NOT touch any price data -- it just builds the recipe
of weights that we'll later apply to a real series in frac_diff.py.

THE FORMULA (book page 79, Snippet 5.1):
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k        for k = 1, 2, ..., size-1

Each new weight is built from the previous one -- much cheaper than
recomputing a full binomial coefficient from scratch every time.

WORKED EXAMPLE (d = 0.4, size = 5), trace by hand:
    w_0 = 1
    w_1 = -w_0 * (0.4 - 1 + 1) / 1 = -1 * 0.4        = -0.4
    w_2 = -w_1 * (0.4 - 2 + 1) / 2 = -(-0.4) * -0.3  = -0.12
    w_3 = -w_2 * (0.4 - 3 + 1) / 3 = -(-0.12) * -0.5333... = -0.064
    w_4 = -w_3 * (0.4 - 4 + 1) / 4 = -(-0.064) * -0.65 = -0.0416

    Notice the weights shrink but never hit exactly zero -- that slow
    decay is "memory" being preserved. Compare to d = 1 (plain returns):
    w_2 and beyond become EXACTLY 0, because (d - k + 1) hits zero at
    k = d + 1 = 2. That's the textbook proof that ordinary differencing
    only looks at yesterday and throws everything else away.

WHY THE RESULT IS REVERSED BEFORE RETURNING:
The loop naturally builds [w_0, w_1, w_2, ...] -- biggest weight (today,
w_0 = 1) FIRST, decaying weights after. But a real price series is
ordered oldest -> newest. To let us later line up the weight array
directly against a slice of price history and just take a dot product
(no extra flipping logic needed downstream), we reverse the weight list
here so it ALSO reads oldest -> newest: the smallest, most-decayed
weight comes first, and today's weight (1.0) comes last.
"""

import numpy as np


def get_weights(d: float, size: int) -> np.ndarray:
    """
    Compute the fractional differentiation weights for a given d.

    Parameters
    ----------
    d : float
        The (possibly fractional) order of differencing.
        d=0   -> no differencing (original series, full memory)
        d=1   -> ordinary differencing (plain returns, memory destroyed
                 after lag 1)
        0<d<1 -> partial differencing (the whole point of this chapter)
    size : int
        How many weights to compute (i.e. how far back in history the
        weight sequence extends). Must be >= 1.

    Returns
    -------
    np.ndarray of shape (size, 1)
        Weights ordered OLDEST -> NEWEST (see docstring above).
        w[-1, 0] is always 1.0 (today's own weight, unchanged by d).
        w[0, 0] is the weight applied to the oldest lag in the window.
    """
    if size < 1:
        raise ValueError("size must be >= 1")

    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)

    # Reverse so the array reads oldest -> newest (see docstring).
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# ---------------------------------------------------------------------
# TDD TEST RESULTS (tests/test_ch05.py, get_weights portion)
# Run 2026-06-26. All hand-traced against the book's recursive formula.
# ---------------------------------------------------------------------
# test_d_0_4_matches_hand_trace                       PASSED
# test_d_1_0_kills_everything_past_lag_1               PASSED
# test_d_0_is_identity_no_differencing                 PASSED
# test_last_weight_is_always_one                       PASSED
# test_weights_decay_in_magnitude_for_fractional_d     PASSED
# test_size_one_returns_just_w0                        PASSED
# test_invalid_size_raises                             PASSED
# test_output_shape_is_column_vector                   PASSED
# ---------------------------------------------------------------------
