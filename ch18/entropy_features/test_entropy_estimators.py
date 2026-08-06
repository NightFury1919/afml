"""
TDD suite for Chapter 18 entropy estimators (Snippets 18.1-18.4).
Known values are hand-traced (see docstrings on each test), not just
shape/type checks.
"""
import numpy as np
import pytest

from entropy_estimators import pmf1, plugIn, lempelZiv_lib, matchLength, konto


# -----------------------------------------------------------------------
# Snippet 18.1: pmf1 / plugIn
# -----------------------------------------------------------------------
class TestPmf1PlugIn:
    def test_pmf1_hand_traced_w1(self):
        """
        msg='0011', w=1. Words of length 1 at positions 1,2,3:
        msg[0:1]='0', msg[1:2]='0', msg[2:3]='1' -> two '0's, one '1'
        out of 3 total windows -> pmf = {'0': 2/3, '1': 1/3}.
        """
        pmf = pmf1('0011', 1)
        assert pmf == pytest.approx({'0': 2 / 3, '1': 1 / 3})

    def test_plugIn_hand_traced_w1(self):
        """
        Entropy of a Bernoulli(1/3) rv (standard binary entropy
        formula): H = -(2/3*log2(2/3) + 1/3*log2(1/3)) = 0.91829583...
        w=1 means the plug-in entropy RATE equals this per-symbol
        entropy exactly (dividing by w=1 is a no-op).
        """
        h, pmf = plugIn('0011', 1)
        assert h == pytest.approx(0.9182958340544896)
        assert pmf == pytest.approx({'0': 2 / 3, '1': 1 / 3})

    def test_plugIn_hand_traced_w2(self):
        """
        msg='1011', w=2. Words of length 2 at positions 2,3:
        msg[0:2]='10', msg[1:3]='01' -> each appears once out of 2
        total windows -> pmf = {'10': 0.5, '01': 0.5}.
        H = -(0.5*log2(0.5) + 0.5*log2(0.5)) / 2 = -(1*(-1))/2 = 0.5.
        """
        h, pmf = plugIn('1011', 2)
        assert h == pytest.approx(0.5)
        assert pmf == pytest.approx({'10': 0.5, '01': 0.5})

    def test_plugIn_accepts_non_string_sequence(self):
        """
        pmf1 explicitly handles non-str input via
        ''.join(map(str,msg)) -- confirm a list of ints works
        identically to the pre-joined string.
        """
        h_list, _ = plugIn([0, 0, 1, 1], 1)
        h_str, _ = plugIn('0011', 1)
        assert h_list == pytest.approx(h_str)

    def test_pmf1_probabilities_sum_to_one(self):
        pmf = pmf1('10110100110101', 3)
        assert sum(pmf.values()) == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Snippet 18.2: lempelZiv_lib
# -----------------------------------------------------------------------
class TestLempelZivLib:
    def test_hand_traced_101010(self):
        """
        Hand-traced parse of '101010' (see module docstring for the
        full trace): phrases found are '1', then '0', then '10'; the
        final attempt at i=4 re-checks '1' and '10' (both already in
        the dict), exhausts the string without a break, and the outer
        while loop exits at i=6. Final library: ['1','0','10'].
        """
        assert lempelZiv_lib('101010') == ['1', '0', '10']

    def test_all_same_symbol_grows_by_one_each_time(self):
        """
        msg='0000': i=1,lib=['0']; j=1: '0' in lib, j=2: '00' not in
        lib -> append, i=3; j=3: '0' in lib, loop exhausts (only j=3
        available), i=4, loop ends. Expected lib=['0','00'].
        """
        assert lempelZiv_lib('0000') == ['0', '00']

    def test_output_is_non_redundant(self):
        """Every entry in the library must be unique (that's the point
        of Lempel-Ziv parsing)."""
        lib = lempelZiv_lib('11010001101001')
        assert len(lib) == len(set(lib))


# -----------------------------------------------------------------------
# Snippet 18.3: matchLength
# -----------------------------------------------------------------------
class TestMatchLength:
    def test_hand_traced_perfect_repeat(self):
        """
        msg='0011001100', i=4, n=4. The 4 symbols starting at i=4
        ('0011') are byte-for-byte identical to the 4 symbols starting
        at j=0 ('0011'), so every length l=0..3 finds a match at j=0.
        Expected: matched length 4 -> returns (4+1, '0011').
        """
        length, sub = matchLength('0011001100', 4, 4)
        assert (length, sub) == (5, '0011')

    def test_no_match_returns_length_one_empty_string(self):
        """
        msg='0000111100', i=4, n=4: the 4 symbols before i (msg[0:4]
        ='0000') never match the symbol at i (msg[4]='1') even at
        l=0, so subS stays '' the whole time -> returns (0+1, '').
        """
        length, sub = matchLength('0000111100', 4, 4)
        assert (length, sub) == (1, '')

    def test_requires_valid_index_bounds(self):
        """Book's own precondition: i>=n and len(msg)>=i+n. Confirm
        the function still runs (no bounds-checking in the book's
        code -- this documents the *contract*, not a defensive
        fix) when the precondition is satisfied exactly at the edge."""
        msg = '01010101'  # len=8
        length, sub = matchLength(msg, 4, 4)  # i=n=4, len=8=i+n
        assert isinstance(length, int)


# -----------------------------------------------------------------------
# Snippet 18.4: konto
# -----------------------------------------------------------------------
class TestKonto:
    def test_repetitive_message_has_lower_entropy_than_random(self):
        """
        Core sanity check matching the book's own framing: a highly
        repetitive message ('10' repeated) should read as
        substantially LESS entropic than a pseudo-random one of
        comparable length, since repeats let matchLength find long
        matches quickly (small Ln_i -> large log2(n)/Ln_i terms only
        when matches are SHORT; repetition -> long matches -> small
        h). Uses a fixed-seed RNG for the random comparator so this is
        a deterministic, reproducible test.
        """
        repetitive = '10' * 20  # len 40, even (required for window=None)
        rng = np.random.RandomState(0)
        random_msg = ''.join(str(x) for x in rng.randint(0, 2, 40))
        out_rep = konto(repetitive)
        out_rand = konto(random_msg)
        assert out_rep['h'] < out_rand['h']

    def test_redundancy_formula(self):
        """r = 1 - h/log2(len(msg)), 0<=r<=1 per the book's own
        printed formula -- confirm the module computes it exactly
        this way, not just that it's in-range."""
        out = konto('10' * 20)
        expected_r = 1 - out['h'] / np.log2(len(('10' * 20)))
        assert out['r'] == pytest.approx(expected_r)
        assert 0 <= out['r'] <= 1

    def test_fixed_window_matches_expected_point_count(self):
        """
        With an explicit window, points=range(window,len(msg)-window+1),
        so num should equal len(msg)-2*window+1 exactly (confirms the
        LOAD-BEARING len(msg)//2 fix doesn't silently shrink the
        window when a window is explicitly supplied and already <=
        len(msg)//2).
        """
        msg = '1101000110100011010001101000'  # len 29
        window = 5
        out = konto(msg, window=window)
        assert out['num'] == len(msg) - 2 * window + 1

    def test_expanding_window_requires_even_length_by_construction(self):
        """
        Book's own caveat: window=None (expanding) needs
        len(msg)%2==0 to read every bit. This isn't enforced by the
        function (no guard in the book's code -- documenting behavior,
        not adding a defensive check the book doesn't have), so this
        test just confirms an odd-length message still runs without
        raising (points=range(1, len(msg)//2+1) truncates via floor
        division rather than crashing).
        """
        out = konto('101011001')  # len 9, odd
        assert out['num'] == 9 // 2  # floor division confirms the fix


# -----------------------------------------------------------------------
# Regression tests: the two real Py2->3 bugs, confirmed to actually
# reproduce on the UNFIXED code before being fixed.
# -----------------------------------------------------------------------
class TestPy2To3BugRegressions:
    def test_konto_len_over_two_would_be_float_without_floor_division(self):
        """
        LOAD-BEARING regression test for the len(msg)/2 -> len(msg)//2
        fix. True division would make range()'s stop argument a float,
        which raises TypeError under Python 3's range(). This test
        proves the CURRENT (fixed) code does NOT raise on an
        odd-length message -- if a future edit reintroduces true
        division, this test starts failing with TypeError.
        """
        try:
            konto('101011001')  # len 9 (odd), forces the /2 boundary
        except TypeError:
            pytest.fail(
                "konto raised TypeError -- the len(msg)//2 floor-division "
                "fix (Py2->3, was len(msg)/2) appears to have regressed."
            )

    def test_xrange_would_not_exist_under_python3(self):
        """
        Documents that `xrange` (as printed in all four snippets) is
        not a Python 3 builtin at all. `xrange` is deliberately not
        imported or aliased anywhere in this module -- calling any of
        the four functions on real input, as every test above does,
        would raise NameError immediately if a bare `xrange(...)` call
        had survived the port. All 16 other tests passing IS the
        regression proof; this test just makes that guarantee explicit
        with a dedicated, minimal call for each function.
        """
        pmf1('01', 1)
        plugIn('01', 1)
        lempelZiv_lib('01')
        matchLength('0011', 2, 2)
        konto('0011')
