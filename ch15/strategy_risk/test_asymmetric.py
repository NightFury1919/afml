"""
TDD suite for Chapter 15's asymmetric-payout module (Sec 15.3,
Snippets 15.3-15.4).

Every test uses a KNOWN expected value: the book's own worked example
(n=260, pi_-=-.01, pi_+=.005, p=.7 -> theta=1.173; same params -> p=.72
for theta=2), the algebraically-derived pi_theta*=0 = 2/3 special case, or
a roundtrip cross-check between binHR/binFreq/binSR -- never just
shape/sanity checks.
"""

import numpy as np
import pytest

import asymmetric as asym


SL, PT, FREQ = -.01, .005, 260  # book's running example throughout Sec 15.3


# =============================================================================
# binSR -- theta[p,n,pi_-,pi_+]
# =============================================================================
class TestBinSR:
    def test_book_worked_example(self):
        # Book (Sec 15.3): "for n=260, pi_-=-.01, pi_+=.005, p=.7, we get
        # theta=1.173."
        theta = asym.binSR(SL, PT, FREQ, p=.7)
        assert theta == pytest.approx(1.173, abs=1e-3)

    def test_reduces_to_symmetric_case(self):
        # Book: "for pi_-=-pi_+ ... theta[p,n,-pi_+,pi_+] = theta[p,n]"
        # -- cross-checked against symmetric.py's independent formula.
        import symmetric as sym
        p, n, pi = .58, 150, .02
        asym_theta = asym.binSR(sl=-pi, pt=pi, freq=n, p=p)
        sym_theta = sym.sharpe_ratio_symmetric(p=p, n=n)
        assert asym_theta == pytest.approx(sym_theta)

    def test_precision_half_gives_expected_value_only_pull(self):
        # At p=.5 the numerator becomes (pt-sl)*.5+sl = (pt+sl)/2, i.e.
        # the outcome is a coin flip between pi_- and pi_+, no directional
        # edge from precision alone.
        theta = asym.binSR(sl=-.02, pt=.02, freq=100, p=.5)
        assert theta == pytest.approx(0.0)


# =============================================================================
# binHR -- implied precision
# =============================================================================
class TestBinHR:
    def test_book_worked_example_theta_2(self):
        # Book: "in order to get theta=2 we require a p=.72."
        p = asym.binHR(SL, PT, FREQ, tSR=2)
        assert p == pytest.approx(.72, abs=1e-2)

    def test_p_theta_star_zero_special_case(self):
        # Book (Sec 15.4): "for p_theta*=0 = 2/3, p < p_theta*=0 => theta<=0"
        # -- the precision below which the strategy has a NEGATIVE
        # expected Sharpe, algebraically p = -pi_-/(pi_+-pi_-).
        p = asym.binHR(SL, PT, FREQ, tSR=0)
        assert p == pytest.approx(2 / 3, abs=1e-6)

    def test_roundtrip_with_binsr(self):
        # The precision binHR returns, fed back through binSR, must
        # reproduce the target Sharpe -- an independent algebraic
        # cross-check, not just a pinned constant.
        tSR = 1.5
        p = asym.binHR(SL, PT, FREQ, tSR)
        assert asym.binSR(SL, PT, FREQ, p) == pytest.approx(tSR)

    def test_negative_discriminant_raises(self):
        # Symbolically, disc = tSR^2*(pt-sl)^2*[tSR^2*(pt-sl)^2-4*freq*pt*sl]
        # -- for the book's normal usage (sl<0<pt) the bracket can never
        # go negative (pt*sl<0 makes 4*freq*pt*sl<0, and the rest is a
        # square). It CAN go negative in the pathological case where sl
        # and pt share a sign (both thresholds positive here) combined
        # with high freq / tiny tSR -- confirmed numerically before
        # writing this test. Must raise, not silently return a complex
        # number (see module's LOAD-BEARING comment).
        with pytest.raises(ValueError):
            asym.binHR(sl=0.001, pt=0.002, freq=1000, tSR=0.01)


# =============================================================================
# binFreq -- implied betting frequency
# =============================================================================
class TestBinFreq:
    def test_roundtrip_recovers_book_frequency(self):
        # Feed the book's own theta=1.173 example back through binFreq at
        # p=.7 and confirm it recovers freq=260 (the book's own n).
        theta = asym.binSR(SL, PT, FREQ, p=.7)
        freq = asym.binFreq(SL, PT, p=.7, tSR=theta)
        assert freq == pytest.approx(FREQ, abs=1e-6)

    def test_roundtrip_with_binsr_general(self):
        # p must be above the break-even precision (2/3 for this sl/pt --
        # see test_extraneous_below_breakeven_returns_none below) or a
        # positive tSR has no valid solution.
        p = .75
        freq = asym.binFreq(SL, PT, p=p, tSR=1.2)
        assert asym.binSR(SL, PT, freq, p) == pytest.approx(1.2)

    def test_higher_precision_needs_fewer_bets(self):
        tSR = 1.5
        freq_low_p = asym.binFreq(SL, PT, p=.75, tSR=tSR)
        freq_high_p = asym.binFreq(SL, PT, p=.95, tSR=tSR)
        assert freq_high_p < freq_low_p

    def test_extraneous_below_breakeven_returns_none(self):
        # Below the break-even precision (p=2/3 for this sl/pt -- the same
        # p_theta*=0 special case verified in TestBinHR), expected profit
        # is negative, so squaring the equation to solve for freq produces
        # a freq whose ACTUAL Sharpe is -tSR, not +tSR -- an extraneous
        # root. Confirmed the book's own "check for extraneous solution"
        # caveat by hand: binSR(binFreq_raw, p=.6) == -1.0, not +1.0.
        assert asym.binFreq(SL, PT, p=.6, tSR=1.0) is None

    def test_at_or_above_breakeven_precision_has_valid_solution(self):
        # Just above 2/3, a positive tSR is achievable again.
        freq = asym.binFreq(SL, PT, p=.70, tSR=0.5)
        assert freq is not None
        assert asym.binSR(SL, PT, freq, p=.70) == pytest.approx(0.5)
