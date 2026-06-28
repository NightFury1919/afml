"""
Chapter 5 -- Fractionally Differentiated Features.

Exposes the chapter's full toolkit:
    get_weights      -- Snippet 5.1, fixed-count weight generation
    frac_diff         -- Snippet 5.2, expanding-window application
    get_weights_ffd   -- Snippet 5.3 (part 1), threshold-stopped weights
    frac_diff_ffd     -- Snippet 5.3 (part 2), fixed-width application
                          (vectorized -- see frac_diff_ffd.py docstring)
    find_min_ffd      -- Snippet 5.4, ADF-based search across d values
    find_minimum_d    -- companion helper, extracts the smallest passing d
    calibrate_ffd_thres -- OUR OWN utility (not a book snippet), for
                          fair comparisons between frac_diff and
                          frac_diff_ffd -- see calibration.py
"""

from get_weights import get_weights
from frac_diff import frac_diff
from get_weights_ffd import get_weights_ffd
from frac_diff_ffd import frac_diff_ffd
from find_min_ffd import find_min_ffd, find_minimum_d
from calibration import calibrate_ffd_thres

__all__ = [
    'get_weights',
    'frac_diff',
    'get_weights_ffd',
    'frac_diff_ffd',
    'find_min_ffd',
    'find_minimum_d',
    'calibrate_ffd_thres',
]
