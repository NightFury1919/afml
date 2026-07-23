"""
TDD suite for Chapter 15's symmetric-payout module (Sec 15.2).

Every test uses a KNOWN expected value: the book's own worked examples
(p=.55 -> theta coefficient 0.1005 and n=396 bets for theta=2; n=52 ->
implied precision 0.6336 for theta=2), hand-traced algebra, or a
cross-validation against direct Monte Carlo simulation of Snippet 15.1 --
never just shape/sanity checks.
"""

import numpy as np
import pytest

import symmetric as sym


# =============================================================================
# sharpe_ratio_symmetric
# =============================================================================
class TestSharpeRatioSymmetric:
    def test_book_worked_example_p55(self):
        # Book (Sec 15.2): "for p=.55, 2p-1/(2*sqrt(p(1-p))) = 0.1005"
        # -- this is the per-sqrt(n) coefficient, i.e. theta at n=1.
        assert sym.sharpe_ratio_symmetric(p=.55, n=1) == pytest.approx(0.1005, abs=1e-4)

    def test_book_worked_example_396_bets(self):
        # Book: achieving theta=2 at p=.55 requires 396 bets/year.
        theta = sym.sharpe_ratio_symmetric(p=.55, n=396)
        assert theta == pytest.approx(2.0, abs=1e-3)

    def test_p_half_gives_zero_sharpe(self):
        # No edge (p=1/2) -> theta=0 regardless of n.
        assert sym.sharpe_ratio_symmetric(p=.5, n=1000) == pytest.approx(0.0)

    def test_symmetric_around_half(self):
        # theta[1-p, n] should be the negative mirror of theta[p, n] --
        # flipping from an edge to an equal-and-opposite disadvantage.
        assert sym.sharpe_ratio_symmetric(p=.6, n=100) == pytest.approx(
            -sym.sharpe_ratio_symmetric(p=.4, n=100)
        )

    def test_n_zero_gives_zero_sharpe(self):
        assert sym.sharpe_ratio_symmetric(p=.7, n=0) == pytest.approx(0.0)

    def test_p_out_of_range_raises(self):
        with pytest.raises(ValueError):
            sym.sharpe_ratio_symmetric(p=0.0, n=100)
        with pytest.raises(ValueError):
            sym.sharpe_ratio_symmetric(p=1.0, n=100)
        with pytest.raises(ValueError):
            sym.sharpe_ratio_symmetric(p=1.5, n=100)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            sym.sharpe_ratio_symmetric(p=.6, n=-10)


# =============================================================================
# implied_precision_symmetric
# =============================================================================
class TestImpliedPrecisionSymmetric:
    def test_book_worked_example_weekly_bets(self):
        # Book (Sec 15.2): "a strategy that only produces weekly bets
        # (n=52) will need a fairly high precision of p=0.6336 to deliver
        # an annualized Sharpe of 2."
        p = sym.implied_precision_symmetric(n=52, tSR=2)
        assert p == pytest.approx(0.6336, abs=1e-4)

    def test_roundtrip_against_sharpe_ratio_symmetric(self):
        # The implied precision, fed back through the direct formula,
        # must reproduce the target Sharpe -- an independent algebraic
        # cross-check rather than a hand-picked constant.
        n, tSR = 200, 1.5
        p = sym.implied_precision_symmetric(n, tSR)
        assert sym.sharpe_ratio_symmetric(p, n) == pytest.approx(tSR)

    def test_higher_n_needs_lower_precision(self):
        tSR = 2.0
        p_low_n = sym.implied_precision_symmetric(n=52, tSR=tSR)
        p_high_n = sym.implied_precision_symmetric(n=5200, tSR=tSR)
        assert p_high_n < p_low_n

    def test_zero_frequency_zero_target_raises(self):
        with pytest.raises(ValueError):
            sym.implied_precision_symmetric(n=0, tSR=0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            sym.implied_precision_symmetric(n=-5, tSR=1.0)


# =============================================================================
# simulate_symmetric_sharpe (Snippet 15.1, Monte Carlo cross-check)
# =============================================================================
class TestSimulateSymmetricSharpe:
    def test_cross_validated_against_closed_form(self):
        # Direct simulation of Snippet 15.1 (ported to a seeded Generator)
        # should recover the closed-form theta[p,n=1] within Monte Carlo
        # noise over a large number of draws.
        rng = np.random.default_rng(7)
        mean, std, sharpe = sym.simulate_symmetric_sharpe(p=.55, n_draws=500_000, rng=rng)
        closed_form = sym.sharpe_ratio_symmetric(p=.55, n=1)
        assert sharpe == pytest.approx(closed_form, abs=0.01)

    def test_reproducible_with_seeded_generator(self):
        # Same seed -> byte-identical draws -> identical outputs. Confirms
        # the ported Generator-based version is actually deterministic
        # (project's random_state convention).
        r1 = sym.simulate_symmetric_sharpe(p=.6, n_draws=1000, rng=np.random.default_rng(123))
        r2 = sym.simulate_symmetric_sharpe(p=.6, n_draws=1000, rng=np.random.default_rng(123))
        assert r1 == r2

    def test_p_one_gives_std_zero_and_nan_sharpe(self):
        # Degenerate case: every draw wins -> std=0 -> sharpe is undefined,
        # not a division-by-zero crash.
        rng = np.random.default_rng(1)
        mean, std, sharpe = sym.simulate_symmetric_sharpe(p=1.0, n_draws=100, rng=rng)
        assert mean == pytest.approx(1.0)
        assert std == pytest.approx(0.0)
        assert np.isnan(sharpe)
