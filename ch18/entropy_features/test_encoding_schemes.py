"""
TDD suite for Chapter 18, Section 18.5 -- encoding schemes.
Formula-only section (no printed book snippet) -- known values here are
hand-computed directly from the book's stated formulas, same treatment
as Ch17's CUSUM/Chow-DF tests.
"""
import numpy as np
import pytest

from encoding_schemes import binary_encode, quantile_encode, sigma_encode


# -----------------------------------------------------------------------
# 18.5.1 -- Binary encoding
# -----------------------------------------------------------------------
class TestBinaryEncode:
    def test_hand_traced_with_zeros_dropped(self):
        """
        returns = [0.01,-0.02,0,0.03,-0.01,0]. Book: 1 for rt>0, 0 for
        rt<0, DROP rt==0. Non-zero entries in order: +,-,+,- -> '1010'.
        The two zeros must vanish entirely, not become a third symbol.
        """
        assert binary_encode([0.01, -0.02, 0, 0.03, -0.01, 0]) == '1010'

    def test_all_positive(self):
        assert binary_encode([0.1, 0.2, 0.3]) == '111'

    def test_all_negative(self):
        assert binary_encode([-0.1, -0.2, -0.3]) == '000'

    def test_all_zero_returns_empty_string(self):
        assert binary_encode([0, 0, 0]) == ''

    def test_output_length_excludes_zeros(self):
        returns = [1, -1, 0, 2, 0, -2, 0]
        encoded = binary_encode(returns)
        n_zeros = sum(1 for r in returns if r == 0)
        assert len(encoded) == len(returns) - n_zeros


# -----------------------------------------------------------------------
# 18.5.2 -- Quantile encoding
# -----------------------------------------------------------------------
class TestQuantileEncode:
    def test_hand_traced_quartiles(self):
        """
        returns=[1..8], n_letters=4 (quartiles). np.quantile at
        [0,.25,.5,.75,1] on 1..8 gives edges
        [1, 2.75, 4.5, 6.25, 8]. Interior edges [2.75,4.5,6.25] split
        the 8 values into two clean pairs per bin:
        1,2 -> code 0 (both < 2.75)
        3,4 -> code 1 (2.75 <= x < 4.5)
        5,6 -> code 2 (4.5 <= x < 6.25)
        7,8 -> code 3 (x >= 6.25)
        Expected message: '00112233'.
        """
        assert quantile_encode(list(range(1, 9)), n_letters=4) == '00112233'

    def test_alphabet_size_matches_n_letters(self):
        """The number of DISTINCT symbols used must never exceed
        n_letters, regardless of the data's distribution."""
        rng = np.random.RandomState(0)
        returns = rng.normal(size=200)
        encoded = quantile_encode(returns, n_letters=5)
        assert len(set(encoded)) <= 5

    def test_roughly_equal_bin_counts_in_sample(self):
        """
        Book's own claim: quantile encoding gives the SAME number of
        observations per letter in-sample (since boundaries are fit
        on the very data being encoded). Confirm counts per symbol are
        exactly equal when n divides evenly by n_letters.
        """
        returns = list(range(1, 21))  # 20 values, divides evenly by 4
        encoded = quantile_encode(returns, n_letters=4)
        counts = [encoded.count(c) for c in '0123']
        assert counts == [5, 5, 5, 5]

    def test_single_letter_degenerates_to_all_zero_code(self):
        assert quantile_encode([1, 5, 9, 3], n_letters=1) == '0000'


# -----------------------------------------------------------------------
# 18.5.3 -- Sigma encoding
# -----------------------------------------------------------------------
class TestSigmaEncode:
    def test_hand_traced_with_boundary_clip(self):
        """
        returns=[1,1,1,5,5,9], sigma=2, min=1.
        Raw codes = floor((x-1)/2): [0,0,0,2,2,4].
        n_codes = ceil((9-1)/2) = 4 -> valid code range is 0..3.
        The raw code for x=9 (value 4) lands exactly ONE PAST the last
        valid bin (an exact-multiple-of-sigma edge case) and must clip
        down to 3, not open a spurious 5th bin.
        Expected: [0,0,0,2,2,3] -> '000223'.
        """
        assert sigma_encode([1, 1, 1, 5, 5, 9], sigma=2) == '000223'

    def test_default_sigma_is_std_over_four(self):
        returns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_sigma = np.std(returns) / 4
        default_encoded = sigma_encode(returns)
        explicit_encoded = sigma_encode(returns, sigma=expected_sigma)
        assert default_encoded == explicit_encoded

    def test_number_of_distinct_codes_matches_formula(self):
        """
        ceil[(max{r}-min{r})/sigma] total codes, per the book's own
        formula -- confirm the number of DISTINCT symbols actually
        used never exceeds this count.
        """
        returns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        sigma = 3.0
        expected_n_codes = int(np.ceil((max(returns) - min(returns)) / sigma))
        encoded = sigma_encode(returns, sigma=sigma)
        assert len(set(encoded)) <= expected_n_codes

    def test_rejects_non_positive_sigma(self):
        with pytest.raises(ValueError):
            sigma_encode([1, 2, 3], sigma=0)
        with pytest.raises(ValueError):
            sigma_encode([1, 2, 3], sigma=-1)

    def test_constant_series_uses_single_code(self):
        """When max==min (a degenerate all-identical series), every
        value must land in code 0, not divide-by-zero."""
        assert sigma_encode([5, 5, 5, 5], sigma=1) == '0000'


# -----------------------------------------------------------------------
# Cross-scheme consistency: all three schemes must round-trip cleanly
# into strings that plugIn/konto can consume directly (from Snippets
# 18.1-18.4).
# -----------------------------------------------------------------------
class TestEncodingFeedsEstimators:
    def test_binary_output_feeds_plugIn(self):
        from entropy_estimators import plugIn
        rng = np.random.RandomState(1)
        returns = rng.normal(size=50)
        encoded = binary_encode(returns)
        h, pmf = plugIn(encoded, 1)
        assert 0 <= h <= 1.001  # binary alphabet, entropy rate capped near 1 bit

    def test_quantile_output_feeds_konto(self):
        from entropy_estimators import konto
        rng = np.random.RandomState(2)
        returns = rng.normal(size=40)
        encoded = quantile_encode(returns, n_letters=4)
        out = konto(encoded)
        assert out['h'] > 0
        assert 0 <= out['r'] <= 1
