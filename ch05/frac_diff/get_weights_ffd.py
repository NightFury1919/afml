"""
Snippet 5.3 (part 1) -- Fixed-width window weighting function.

PLAIN-ENGLISH IDEA:
This uses the EXACT SAME recursive weight formula as Snippet 5.1
(get_weights):
    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

The difference is only in WHEN it stops. get_weights() stops after a
fixed COUNT ("size") you choose up front. get_weights_ffd() instead
keeps generating new weights until a newly computed weight's absolute
MAGNITUDE drops below `thres`, then stops. Since the weights decay
toward zero (for non-integer d > 0), this is guaranteed to terminate,
and it automatically finds "how many weights actually matter" rather
than you having to guess a window size.

IMPORTANT -- thres HERE IS NOT THE SAME KIND OF THING AS frac_diff's
thres (Snippet 5.2). There, thres was a FRACTION of total weight mass
(0 to 1, e.g. 0.01 = "1% of total weight"). HERE, thres is an ABSOLUTE
weight magnitude cutoff (e.g. 1e-5 = "stop once an individual weight's
size drops below this number"). Same parameter name, genuinely
different units and meaning -- easy to mix up, worth remembering.
"""

import numpy as np


def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    """
    Compute fractional differentiation weights, stopping once a newly
    generated weight's magnitude falls below `thres` (rather than
    stopping at a fixed count, as get_weights() does).

    Parameters
    ----------
    d : float
        Order of fractional differencing.
    thres : float
        Absolute weight-magnitude cutoff. Smaller thres -> more
        weights kept -> wider window -> more memory preserved, but
        slower. Larger thres -> fewer weights -> narrower window,
        faster, less memory.

    Returns
    -------
    np.ndarray of shape (M, 1)
        Weights ordered oldest -> newest (same convention as
        get_weights()), where M is however many weights were
        significant enough to keep. w[-1, 0] is always 1.0.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            # This term is too small to matter -- discard it (do NOT
            # append) and stop generating further (even smaller) terms.
            break
        w.append(w_)
        k += 1

    return np.array(w[::-1]).reshape(-1, 1)


# ---------------------------------------------------------------------
# TDD TEST RESULTS (tests/test_ch05.py, get_weights_ffd portion)
# Run 2026-06-26.
# ---------------------------------------------------------------------
# test_get_weights_ffd_matches_hand_trace               PASSED
# test_get_weights_ffd_last_weight_always_one            PASSED
# test_get_weights_ffd_smaller_thres_keeps_more_weights   PASSED
# test_get_weights_ffd_cross_checks_against_get_weights   PASSED
#   (cross-checked against the already-verified get_weights formula,
#   not just re-derived by hand a second time)
# ---------------------------------------------------------------------
