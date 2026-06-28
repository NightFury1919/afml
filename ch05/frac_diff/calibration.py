"""
calibrate_ffd_thres -- OUR OWN UTILITY, NOT A BOOK SNIPPET.

WHY THIS EXISTS:
frac_diff()'s thres (Snippet 5.2) is a RELATIVE weight-loss fraction
(e.g. 0.01 = "tolerate losing 1% of total weight mass"). frac_diff_ffd()'s
thres (Snippet 5.3) is an ABSOLUTE weight-magnitude cutoff (e.g. 0.01 =
"stop once an individual weight's size drops below 0.01"). These are
NOT the same kind of quantity, even though the book's own Snippet 5.4
demo passes the literal same number (.01) to both -- which on real data
(see our BTC tick data investigation) can produce very different
effective windows, especially for slowly-decaying low-d weights.

This helper finds the ABSOLUTE thres for frac_diff_ffd that retains
roughly the SAME total weight mass fraction as a given RELATIVE thres
would for frac_diff -- letting you make a genuinely fair, apples-to-
apples comparison between the two methods instead of an accidentally
apples-to-oranges one.
"""

import numpy as np
from get_weights import get_weights


def calibrate_ffd_thres(d: float, mass_retain: float = 0.99, max_size: int = 2000):
    """
    Find an absolute weight-magnitude threshold for get_weights_ffd
    that retains approximately `mass_retain` fraction of the total
    weight mass -- the same notion frac_diff's relative thres targets
    (mass_retain = 1 - frac_diff's thres).

    Parameters
    ----------
    d : float
        Order of fractional differencing.
    mass_retain : float, default 0.99
        Target fraction of total weight mass to keep (0.99 matches
        frac_diff's common default thres=0.01, since 1 - 0.01 = 0.99).
    max_size : int, default 2000
        How many weights to compute when establishing the "true" total
        mass. Should be generously larger than any width you expect
        get_weights_ffd to actually need for the d values you're using.

    Returns
    -------
    (thres, k) : (float, int)
        thres -- the absolute magnitude cutoff to pass to
                 get_weights_ffd(d, thres) to achieve approximately
                 the target retained mass.
        k     -- the number of weights this thres is expected to keep
                 (i.e. the resulting width + 1), for sanity-checking.
    """
    w_full = get_weights(d, max_size).flatten()
    abs_w = np.abs(w_full)
    total = abs_w.sum()

    # Mass captured by keeping only the k MOST RECENT terms (the ones
    # frac_diff_ffd would actually keep, since it works backward from
    # today rather than forward from the deep past).
    cum_from_recent = np.cumsum(abs_w[::-1])
    frac_from_recent = cum_from_recent / total

    k = int(np.argmax(frac_from_recent >= mass_retain)) + 1
    kept_min_abs = abs_w[-k]
    excluded_abs = abs_w[-(k + 1)] if k < max_size else 0.0

    # Midpoint between the smallest KEPT weight and the largest
    # EXCLUDED weight -- safely separates the two given the weights'
    # magnitudes are monotonic in this range.
    thres = (kept_min_abs + excluded_abs) / 2
    return thres, k
