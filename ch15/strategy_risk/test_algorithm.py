"""
TDD suite for Chapter 15's strategy-risk algorithm module (Sec 15.4,
Snippet 15.5).

Every test uses a KNOWN expected value: a tiny 8-observation series
hand-traced through every intermediate step (rPos, rNeg, p, thresP, risk),
a seeded-Generator regression test pinning exact output from the book's
own mixGaussians(mu1,mu2,sigma1,sigma2,prob1,nObs=.05,-.1,.05,.1,.75,2600)
example, and a direct demonstration that the confirmed norm.cdf fix
actually changes the output vs. the literally-printed (buggy) call --
never just shape/sanity checks.
"""

import numpy as np
import pytest
import scipy.stats as ss

import algorithm as algo
import asymmetric as asym


# =============================================================================
# probFailure -- hand-traced small example
# =============================================================================
class TestProbFailureHandTraced:
    RET = np.array([0.02, 0.03, -0.01, 0.015, -0.02, 0.01, -0.005, 0.025])

    def test_hand_traced_every_step(self):
        ret = self.RET
        # Hand-traced: 5 positive (.02,.03,.015,.01,.025), 3 <=0 (-.01,-.02,-.005)
        expected_rPos = (0.02 + 0.03 + 0.015 + 0.01 + 0.025) / 5
        expected_rNeg = (-0.01 - 0.02 - 0.005) / 3
        expected_p = 5 / 8
        expected_thresP = asym.binHR(expected_rNeg, expected_rPos, freq=260, tSR=1.0)
        expected_risk = ss.norm.cdf(expected_thresP, expected_p, (expected_p * (1 - expected_p)) ** 0.5)

        risk = algo.probFailure(ret, freq=260, tSR=1.0)
        assert risk == pytest.approx(expected_risk)
        # pinned exact value, so a future refactor that silently changes
        # the formula gets caught even if the hand-traced logic above is
        # also (identically) broken
        assert risk == pytest.approx(0.3201562452932211)

    def test_all_positive_returns(self):
        # No losing bets at all -- rNeg is NaN (empty-slice mean), so
        # thresP/risk propagate NaN rather than crashing. Documents
        # actual behavior (matches the book's own unguarded printed code)
        # rather than silently papering over it.
        ret = np.array([0.01, 0.02, 0.03])
        risk = algo.probFailure(ret, freq=260, tSR=1.0)
        assert np.isnan(risk)


# =============================================================================
# probFailure -- seeded regression test, book's own mixGaussians example
# =============================================================================
class TestProbFailureSeededRegression:
    def test_book_parameters_seeded_regression(self):
        # Book's own main(): mu1,mu2,sigma1,sigma2,prob1,nObs=.05,-.1,.05,.1,.75,2600
        # tSR,freq=2.,260 -- book prints no concrete probF output to check
        # against, so this pins an exact value from a fixed seed instead
        # (a regression test, not a book-value cross-check).
        rng = np.random.default_rng(42)
        ret = algo.mixGaussians(.05, -.1, .05, .1, .75, 2600, rng=rng)
        risk = algo.probFailure(ret, freq=260, tSR=2.0)
        assert risk == pytest.approx(0.48705621989906417)

    def test_fix_actually_changes_result_vs_literal_book_code(self):
        # Demonstrates the confirmed norm.cdf fix is load-bearing: the
        # literally-printed book call (scale=p*(1-p), a variance) gives a
        # DIFFERENT answer than the fixed call (scale=sqrt(p*(1-p))) on
        # the identical seeded data -- not just a cosmetic rewrite.
        rng = np.random.default_rng(42)
        ret = algo.mixGaussians(.05, -.1, .05, .1, .75, 2600, rng=rng)
        fixed_risk = algo.probFailure(ret, freq=260, tSR=2.0)

        rPos, rNeg = ret[ret > 0].mean(), ret[ret <= 0].mean()
        p = ret[ret > 0].shape[0] / float(ret.shape[0])
        thresP = asym.binHR(rNeg, rPos, 260, 2.0)
        literal_book_risk = ss.norm.cdf(thresP, p, p * (1 - p))  # as literally printed

        assert fixed_risk != pytest.approx(literal_book_risk)
        assert fixed_risk == pytest.approx(0.48705621989906417)
        assert literal_book_risk == pytest.approx(0.47283700967471476)


# =============================================================================
# mixGaussians
# =============================================================================
class TestMixGaussians:
    def test_output_length_matches_nObs(self):
        rng = np.random.default_rng(0)
        ret = algo.mixGaussians(0.0, 0.0, 1.0, 1.0, 0.5, 500, rng=rng)
        assert len(ret) == 500

    def test_reproducible_with_seeded_generator(self):
        r1 = algo.mixGaussians(.05, -.1, .05, .1, .75, 100, rng=np.random.default_rng(9))
        r2 = algo.mixGaussians(.05, -.1, .05, .1, .75, 100, rng=np.random.default_rng(9))
        np.testing.assert_array_equal(r1, r2)

    def test_prob1_one_gives_pure_first_component(self):
        # prob1=1.0 -> every draw from (mu1, sigma1), none from the
        # second component.
        rng = np.random.default_rng(3)
        ret = algo.mixGaussians(mu1=5.0, mu2=-5.0, sigma1=0.001, sigma2=0.001,
                                 prob1=1.0, nObs=200, rng=rng)
        assert ret.mean() == pytest.approx(5.0, abs=0.01)
