"""
Chapter 18, Section 18.5 -- Encoding schemes.

WHY (plain-English, before the math):
plugIn/konto (Snippets 18.1-18.4) operate on discrete strings -- but our
real data is continuous returns. Before any entropy estimator can run
on real BTC/TUSD data, every return has to be converted into a symbol
from some finite alphabet. This module implements the book's three
encoding schemes (Sec 18.5.1-18.5.3), each trading off differently
between how much information survives the discretization and how large
an alphabet is needed.

These sections are FORMULA-ONLY in the book -- no printed code snippet
exists to diff a fix against (unlike 18.1-18.4). Book-fidelity here
means implementing the printed math/prose carefully and verifying with
hand-computed test cases, same treatment as Ch17's CUSUM/Chow-DF.

DESIGN DECISIONS (confirmed with Ethan 2026-08-04, documented rather
than silently resolved):
  - Quantile encoding (18.5.2): the book describes setting quantile
    boundaries on an in-sample period and applying them out-of-sample.
    Our real dataset is ~87-88 events -- too small to split without
    leaving too little data on either side (same tension as Ch13's O-U
    calibration). DECISION: use the FULL sample to set quantile
    boundaries (no train/test split), documented as a limitation, not
    silently glossed over -- same treatment this project has given
    every other small-sample caveat (Ch13, Ch15, Ch17 Part C).
  - Default number of quantile letters: 4 (quartiles) unless the caller
    overrides.
  - Default sigma step for sigma encoding: std(returns)/4 unless the
    caller overrides -- a reasonable data-driven default, not a magic
    constant.
"""
import string

import numpy as np

# Alphabet for turning integer codes into single characters: digits
# first, then lowercase letters, so any alphabet size up to 36 renders
# as a single readable character per code (needed by konto/matchLength,
# which slice msg[i:j] expecting one code = one character).
_ALPHABET = string.digits + string.ascii_lowercase


def _code_to_char(code):
    if code >= len(_ALPHABET):
        raise ValueError(
            f"Alphabet size {code + 1} exceeds the {len(_ALPHABET)}-symbol "
            "single-character alphabet supported here."
        )
    return _ALPHABET[code]


# -----------------------------------------------------------------------
# 18.5.1 -- Binary encoding
# -----------------------------------------------------------------------
def binary_encode(returns):
    """
    Encode a stream of returns by sign: '1' for rt>0, '0' for rt<0,
    dropping rt==0 cases exactly as the book specifies (Sec 18.5.1).

    Parameters
    ----------
    returns : array-like of float

    Returns
    -------
    str : the encoded message, one character per non-zero return, in
        original order.
    """
    returns = np.asarray(returns, dtype=float)
    nonzero = returns[returns != 0]
    return ''.join('1' if r > 0 else '0' for r in nonzero)


# -----------------------------------------------------------------------
# 18.5.2 -- Quantile encoding
# -----------------------------------------------------------------------
def quantile_encode(returns, n_letters=4):
    """
    Encode each return according to which quantile bin it falls into
    (Sec 18.5.2). The book describes fitting quantile boundaries on an
    in-sample period and applying them out-of-sample; per the
    small-sample decision documented in this module's docstring, this
    implementation fits and applies boundaries on the SAME (full)
    sample -- caller should treat results as in-sample, not a genuine
    train/test-validated encoding.

    Parameters
    ----------
    returns : array-like of float
    n_letters : int, number of quantile bins (alphabet size). Default 4
        (quartiles).

    Returns
    -------
    str : the encoded message, one character per return.
    """
    returns = np.asarray(returns, dtype=float)
    quantile_edges = np.quantile(returns, np.linspace(0, 1, n_letters + 1))
    # Interior edges only (drop the global min/max) so np.digitize
    # assigns codes 0..n_letters-1 inclusive.
    interior_edges = quantile_edges[1:-1]
    codes = np.digitize(returns, interior_edges, right=False)
    return ''.join(_code_to_char(int(c)) for c in codes)


# -----------------------------------------------------------------------
# 18.5.3 -- Sigma encoding
# -----------------------------------------------------------------------
def sigma_encode(returns, sigma=None):
    """
    Encode each return according to a fixed discretization step sigma
    (Sec 18.5.3): code 0 covers [min(r), min(r)+sigma), code 1 covers
    [min(r)+sigma, min(r)+2*sigma), and so on, for
    ceil((max(r)-min(r))/sigma) total codes.

    Parameters
    ----------
    returns : array-like of float
    sigma : float, optional. Discretization step. Defaults to
        std(returns)/4 (a data-driven default, not a book-specified
        value -- documented explicitly since the book leaves sigma's
        choice to the practitioner).

    Returns
    -------
    str : the encoded message, one character per return.
    """
    returns = np.asarray(returns, dtype=float)
    if sigma is None:
        sigma = returns.std() / 4
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    r_min = returns.min()
    codes = np.floor((returns - r_min) / sigma).astype(int)
    # Guard the exact-maximum edge case: floor((max-min)/sigma) can
    # land exactly on the next-bin boundary when (max-min) is an exact
    # multiple of sigma; clip so the max value stays in the last valid
    # bin rather than opening a spurious extra one.
    n_codes = int(np.ceil((returns.max() - r_min) / sigma))
    codes = np.clip(codes, 0, max(n_codes - 1, 0))
    return ''.join(_code_to_char(int(c)) for c in codes)


# -----------------------------------------------------------------------
# pytest -v output (sandbox, Python 3.12.3) -- 16/16 passed (this file's
# tests only; 33/33 when run together with test_entropy_estimators.py).
# Real-machine confirmation under mlfinlab (Python 3.10.20) still
# pending as of this commit -- see chapter README for status.
# -----------------------------------------------------------------------
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
# rootdir: /home/claude/ch18/entropy_features
# collecting ... collected 16 items
#
# test_encoding_schemes.py::TestBinaryEncode::test_hand_traced_with_zeros_dropped PASSED [  6%]
# test_encoding_schemes.py::TestBinaryEncode::test_all_positive PASSED     [ 12%]
# test_encoding_schemes.py::TestBinaryEncode::test_all_negative PASSED     [ 18%]
# test_encoding_schemes.py::TestBinaryEncode::test_all_zero_returns_empty_string PASSED [ 25%]
# test_encoding_schemes.py::TestBinaryEncode::test_output_length_excludes_zeros PASSED [ 31%]
# test_encoding_schemes.py::TestQuantileEncode::test_hand_traced_quartiles PASSED [ 37%]
# test_encoding_schemes.py::TestQuantileEncode::test_alphabet_size_matches_n_letters PASSED [ 43%]
# test_encoding_schemes.py::TestQuantileEncode::test_roughly_equal_bin_counts_in_sample PASSED [ 50%]
# test_encoding_schemes.py::TestQuantileEncode::test_single_letter_degenerates_to_all_zero_code PASSED [ 56%]
# test_encoding_schemes.py::TestSigmaEncode::test_hand_traced_with_boundary_clip PASSED [ 62%]
# test_encoding_schemes.py::TestSigmaEncode::test_default_sigma_is_std_over_four PASSED [ 68%]
# test_encoding_schemes.py::TestSigmaEncode::test_number_of_distinct_codes_matches_formula PASSED [ 75%]
# test_encoding_schemes.py::TestSigmaEncode::test_rejects_non_positive_sigma PASSED [ 81%]
# test_encoding_schemes.py::TestSigmaEncode::test_constant_series_uses_single_code PASSED [ 87%]
# test_encoding_schemes.py::TestEncodingFeedsEstimators::test_binary_output_feeds_plugIn PASSED [ 93%]
# test_encoding_schemes.py::TestEncodingFeedsEstimators::test_quantile_output_feeds_konto PASSED [100%]
#
# ============================== 16 passed in 0.32s ===============================
