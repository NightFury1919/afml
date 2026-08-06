"""
Chapter 18 -- Entropy Features
Snippets 18.1-18.4: plug-in (maximum-likelihood) entropy estimator and
Lempel-Ziv-family entropy estimators (Kontoyiannis' method).

Function/variable names are kept exactly as printed in the book
(plugIn, pmf1, lempelZiv_lib, matchLength, konto) per this project's
book-fidelity convention -- not rewritten to snake_case.

WHY (plain-English, before the math):
Shannon's entropy H[X] = -sum(p[x] * log2(p[x])) tells you, on average,
how many bits per symbol you need to describe a message -- but it
requires KNOWING the true probabilities p[x]. On real price data we
never know the true distribution; we only have one observed sequence.
The estimators in this module are different ways of estimating entropy
FROM a single observed sequence, using progressively cleverer tricks:
  - plugIn: just count how often each length-w "word" actually appears
    in the sequence, and treat that empirical frequency as if it were
    the true probability (hence "plug-in" / maximum-likelihood).
  - lempelZiv_lib: parse the message into a dictionary of
    never-before-seen substrings. A message that's easy to predict
    (low entropy) needs few, long dictionary entries; a message that's
    hard to predict (high entropy) needs many, short entries.
  - matchLength / konto: for each position i, ask "what's the longest
    stretch starting at i that has already appeared somewhere in the
    last n symbols?" Kontoyiannis' theorem says the RECIPROCAL of the
    average of (this length / log2(window size)) converges to the true
    entropy rate as the window grows -- so entropy estimation reduces to
    measuring how far you have to look back to find a repeat.
"""
import numpy as np


# -----------------------------------------------------------------------
# Snippet 18.1 -- Plug-in (maximum-likelihood) entropy estimator
# -----------------------------------------------------------------------
def pmf1(msg, w):
    """
    Compute the probability mass function for a one-dim discrete rv.
    len(msg)-w occurrences of each length-w "word" in msg.

    LOAD-BEARING FIX (Py2->3, confirmed by direct test): the book's
    printed loop is `for i in xrange(w,len(msg))`. Python 3 has no
    xrange -- fixed to range(), semantics identical (xrange was just
    Python 2's lazy range()).
    """
    lib = {}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    for i in range(w, len(msg)):
        msg_ = msg[i - w:i]
        if msg_ not in lib:
            lib[msg_] = [i - w]
        else:
            lib[msg_] = lib[msg_] + [i - w]
    pmf = float(len(msg) - w)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def plugIn(msg, w):
    """
    Compute plug-in (ML) entropy rate: treat each length-w word's
    empirical frequency as its probability, then apply Shannon's
    entropy formula and divide by w to get a per-symbol (entropy RATE)
    estimate rather than a per-word one.
    """
    pmf = pmf1(msg, w)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w
    return out, pmf


# -----------------------------------------------------------------------
# Snippet 18.2 -- Lempel-Ziv dictionary
# -----------------------------------------------------------------------
def lempelZiv_lib(msg):
    """
    Build a library of non-redundant substrings using the LZ algorithm:
    starting from position i, grow the candidate substring one symbol
    at a time until it's NOT already in the dictionary, add it, then
    jump past it and repeat.

    LOAD-BEARING FIX (Py2->3, confirmed by direct test): `xrange` ->
    `range`, identical semantics otherwise.
    """
    i, lib = 1, [msg[0]]
    while i < len(msg):
        for j in range(i, len(msg)):
            msg_ = msg[i:j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return lib


# -----------------------------------------------------------------------
# Snippet 18.3 -- Longest match length
# -----------------------------------------------------------------------
def matchLength(msg, i, n):
    """
    Length of the longest substring starting at position i (up to
    length n) that has ALSO appeared somewhere in the n symbols
    immediately before i (with overlap allowed). Requires i>=n and
    len(msg)>=i+n.

    Returns (matched_length + 1, matched_substring) -- the "+1" is the
    book's own convention (Ln_i = 1 + max{l | match of length l
    exists}), not a bug.

    LOAD-BEARING FIX (Py2->3, confirmed by direct test): `xrange` ->
    `range` (both loops), identical semantics otherwise.
    """
    subS = ''
    for l in range(n):
        msg1 = msg[i:i + l + 1]
        for j in range(i - n, i):
            msg0 = msg[j:j + l + 1]
            if msg1 == msg0:
                subS = msg1
                break  # search for higher l.
    return len(subS) + 1, subS  # matched length + 1


# -----------------------------------------------------------------------
# Snippet 18.4 -- Kontoyiannis' LZ entropy estimate (2013, centered window)
# -----------------------------------------------------------------------
def konto(msg, window=None):
    """
    Kontoyiannis' LZ entropy estimate, 2013 version (centered window).
    Inverse of the avg length of the shortest non-redundant substring.
    If non-redundant substrings are short, the text is highly entropic.
    window==None for expanding window, in which case len(msg)%2==0.
    If the end of msg is more relevant, try konto(msg[::-1]).

    LOAD-BEARING FIXES (both confirmed by direct test):
    1. `xrange` -> `range` (Py2->3 syntax, both loops).
    2. `len(msg)/2` -> `len(msg)//2` (TWO places: the expanding-window
       `points` range, and the `window=min(window,...)` clamp). This is
       NOT just a Py2/3 syntax issue -- in Python 3, `/` is TRUE
       division, so `len(msg)/2` is a float. Passing a float to
       `range()` or using it as a `min()` bound against an int window
       raises TypeError / behaves inconsistently. `//` (floor division)
       restores the book's original intent (an integer half-length),
       same bug category as Ch16's getRecBipart len(i)/2 fix.
    """
    out = {'num': 0, 'sum': 0, 'subS': []}
    if not isinstance(msg, str):
        msg = ''.join(map(str, msg))
    if window is None:
        points = range(1, len(msg) // 2 + 1)
    else:
        window = min(window, len(msg) // 2)
        points = range(window, len(msg) - window + 1)
    for i in points:
        if window is None:
            l, msg_ = matchLength(msg, i, i)
            out['sum'] += np.log2(i + 1) / l  # to avoid Doeblin condition
        else:
            l, msg_ = matchLength(msg, i, window)
            out['sum'] += np.log2(window + 1) / l  # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num'] += 1
    out['h'] = out['sum'] / out['num']
    out['r'] = 1 - out['h'] / np.log2(len(msg))  # redundancy, 0<=r<=1
    return out


# -----------------------------------------------------------------------
# pytest -v output (sandbox, Python 3.12.3) -- 17/17 passed.
# Real-machine confirmation under mlfinlab (Python 3.10.20) still
# pending as of this commit -- see chapter README for status.
# -----------------------------------------------------------------------
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /usr/bin/python3
# cachedir: .pytest_cache
# rootdir: /home/claude/ch18/entropy_features
# collecting ... collected 17 items
#
# test_entropy_estimators.py::TestPmf1PlugIn::test_pmf1_hand_traced_w1 PASSED [  5%]
# test_entropy_estimators.py::TestPmf1PlugIn::test_plugIn_hand_traced_w1 PASSED [ 11%]
# test_entropy_estimators.py::TestPmf1PlugIn::test_plugIn_hand_traced_w2 PASSED [ 17%]
# test_entropy_estimators.py::TestPmf1PlugIn::test_plugIn_accepts_non_string_sequence PASSED [ 23%]
# test_entropy_estimators.py::TestPmf1PlugIn::test_pmf1_probabilities_sum_to_one PASSED [ 29%]
# test_entropy_estimators.py::TestLempelZivLib::test_hand_traced_101010 PASSED [ 35%]
# test_entropy_estimators.py::TestLempelZivLib::test_all_same_symbol_grows_by_one_each_time PASSED [ 41%]
# test_entropy_estimators.py::TestLempelZivLib::test_output_is_non_redundant PASSED [ 47%]
# test_entropy_estimators.py::TestMatchLength::test_hand_traced_perfect_repeat PASSED [ 52%]
# test_entropy_estimators.py::TestMatchLength::test_no_match_returns_length_one_empty_string PASSED [ 58%]
# test_entropy_estimators.py::TestMatchLength::test_requires_valid_index_bounds PASSED [ 64%]
# test_entropy_estimators.py::TestKonto::test_repetitive_message_has_lower_entropy_than_random PASSED [ 70%]
# test_entropy_estimators.py::TestKonto::test_redundancy_formula PASSED    [ 76%]
# test_entropy_estimators.py::TestKonto::test_fixed_window_matches_expected_point_count PASSED [ 82%]
# test_entropy_estimators.py::TestKonto::test_expanding_window_requires_even_length_by_construction PASSED [ 88%]
# test_entropy_estimators.py::TestPy2To3BugRegressions::test_konto_len_over_two_would_be_float_without_floor_division PASSED [ 94%]
# test_entropy_estimators.py::TestPy2To3BugRegressions::test_xrange_would_not_exist_under_python3 PASSED [100%]
#
# ============================== 17 passed in 0.11s ===============================
