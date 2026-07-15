"""
TDD suite for AFML Chapter 13 -- Optimal Trading Rules (OTR).

Known expected values are hand-traced from the book's own formulas
(eq. 13.5-13.7 for estimation, Snippet 13.2's recursion for simulation),
not just shape/type checks -- per the repo's TDD workflow.
"""

# --- import the module(s) under test ---------------------------------------
# Derive the repo root from __file__, put it on sys.path, then import
# fully-qualified.
#
# LOAD-BEARING -- do NOT replace this with a bare `from <module> import ...`,
# and do NOT rely on pytest to put the repo root on sys.path for you. pytest
# walks UP the __init__.py chain to decide which directory it inserts, so the
# import statement that "works" silently depends on which folders happen to
# contain an __init__.py. That makes tests break from a file two directories
# away, and makes the correct import differ per chapter. Deriving ROOT from
# __file__ works from any cwd, with or without pytest, and matches the .py
# path convention in CLAUDE.md and the ch12 pytest-rootdir gotcha this
# pattern was generalized from.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pytest

from ch13.otr.otr import (  # noqa: E402
    build_xy_from_opportunities,
    estimate_ou_params,
    phi_to_half_life,
    half_life_to_phi,
    simulate_ou_path,
    batch,
    best_node,
)


# --------------------------------------------------------------------------- #
# build_xy_from_opportunities (eq. 13.6)
# --------------------------------------------------------------------------- #

def test_build_xy_matches_book_equation_13_6_toy_example():
    # Two toy opportunities, hand-picked so the (X, Y) pooling is easy to
    # verify by eye. Opportunity 0: prices [100,102,101,100.5], target 100.
    # Opportunity 1: prices [50,47,48.5,49.2], target 50.
    paths = [[100, 102, 101, 100.5], [50, 47, 48.5, 49.2]]
    targets = [100, 50]
    X, Y = build_xy_from_opportunities(paths, targets)
    expected_X = np.array([0, 2, 1, 0, -3, -1.5])
    expected_Y = np.array([2, 1, 0.5, -3, -1.5, -0.8])
    np.testing.assert_allclose(X, expected_X)
    np.testing.assert_allclose(Y, expected_Y)


def test_build_xy_skips_single_observation_paths():
    # A path with only 1 price has no (t-1, t) pair to contribute.
    paths = [[100], [50, 51, 49]]
    targets = [100, 50]
    X, Y = build_xy_from_opportunities(paths, targets)
    assert len(X) == 2  # only the second path contributes 2 pairs
    assert len(Y) == 2


def test_build_xy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_xy_from_opportunities([[1, 2]], targets=[1, 2])  # 1 path, 2 targets


# --------------------------------------------------------------------------- #
# estimate_ou_params (eq. 13.7) -- hand-traced against the same toy example
# --------------------------------------------------------------------------- #

def test_estimate_ou_params_hand_traced():
    # Same toy (X, Y) as above. Hand computation (population covariance):
    #   cov[Y,X] = mean(X*Y) - mean(X)*mean(Y) = 1.291666...
    #   cov[X,X] = mean(X*X) - mean(X)**2      = 2.645833...
    #   phi_hat  = cov[Y,X]/cov[X,X]           = 0.48818897637795267
    #   sigma_hat (residual std)               = 1.461536601951913
    X = np.array([0, 2, 1, 0, -3, -1.5])
    Y = np.array([2, 1, 0.5, -3, -1.5, -0.8])
    phi_hat, sigma_hat = estimate_ou_params(X, Y)
    assert phi_hat == pytest.approx(0.48818897637795267, rel=1e-9)
    assert sigma_hat == pytest.approx(1.461536601951913, rel=1e-9)


def test_estimate_ou_params_recovers_known_process():
    # Statistical recovery test: simulate a long synthetic O-U series with a
    # KNOWN phi/sigma, then check estimate_ou_params recovers them within a
    # tolerance appropriate for the sample size. This is the estimation
    # step's real justification -- it should work on data actually generated
    # by the model it assumes.
    rng = np.random.default_rng(42)
    true_phi, true_sigma, forecast = 0.7, 2.0, 5.0
    n = 20_000
    p = np.empty(n)
    p[0] = 0.0
    for t in range(1, n):
        p[t] = (1 - true_phi) * forecast + true_phi * p[t - 1] + true_sigma * rng.normal()
    X, Y = p[:-1], p[1:]
    phi_hat, sigma_hat = estimate_ou_params(X, Y)
    assert phi_hat == pytest.approx(true_phi, abs=0.02)
    assert sigma_hat == pytest.approx(true_sigma, abs=0.05)


def test_estimate_ou_params_rejects_degenerate_x():
    X = np.array([5.0, 5.0, 5.0])  # zero variance
    Y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        estimate_ou_params(X, Y)


# --------------------------------------------------------------------------- #
# phi_to_half_life / half_life_to_phi (Section 13.5.1)
# --------------------------------------------------------------------------- #

def test_phi_to_half_life_known_value():
    # phi=0.5 -> tau = -log(2)/log(0.5) = 1.0 exactly (log(0.5) = -log(2)).
    assert phi_to_half_life(0.5) == pytest.approx(1.0, abs=1e-12)


def test_half_life_to_phi_known_value():
    # tau=1 -> phi = 2**(-1/1) = 0.5 exactly.
    assert half_life_to_phi(1.0) == pytest.approx(0.5, abs=1e-12)


def test_half_life_phi_round_trip():
    for tau in [2.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
        phi = half_life_to_phi(tau)
        assert phi_to_half_life(phi) == pytest.approx(tau, rel=1e-9)


def test_phi_to_half_life_returns_nan_for_non_stationary_phi():
    # By design (see otr.py docstring): non-stationary/degenerate phi
    # returns NaN rather than raising, so real-data findings like our
    # phi_hat ~ 1.04 can be reported, not crash the pipeline.
    assert np.isnan(phi_to_half_life(1.0))
    assert np.isnan(phi_to_half_life(1.05))
    assert np.isnan(phi_to_half_life(0.0))
    assert np.isnan(phi_to_half_life(-0.5))  # stationary per eq 13.4's
    # (-1,1) condition, but NOT valid for THIS half-life formula (see docstring)


def test_half_life_to_phi_rejects_nonpositive_half_life():
    with pytest.raises(ValueError):
        half_life_to_phi(0.0)
    with pytest.raises(ValueError):
        half_life_to_phi(-5.0)


# --------------------------------------------------------------------------- #
# simulate_ou_path -- hand-traced against a deterministic shock sequence
# --------------------------------------------------------------------------- #

def _fixed_shocks(shocks):
    """Deterministic rng: returns each value in `shocks` in order."""
    it = iter(shocks)
    return lambda: next(it)


def test_simulate_ou_path_hand_traced_exit_via_profit_take():
    # phi_hat/sigma_hat from the hand-traced toy example above, forecast=0,
    # pt=1.5, sl=3.0, fixed shock sequence [1.0, 0.5, -0.3, 1.2].
    # Hand computation (see conversation this suite documents):
    #   t=1: p=1.461536...  cP=1.461536...  (below 1.5, continue)
    #   t=2: p=1.444274...  cP=1.444274...  (below 1.5, continue)
    #   t=3: p=0.266617...  cP=0.266617...  (well inside, continue)
    #   t=4: p=1.884003...  cP=1.884003...  (exceeds 1.5 -> exit, profit-take)
    phi, sigma = 0.48818897637795267, 1.461536601951913
    rng = _fixed_shocks([1.0, 0.5, -0.3, 1.2])
    cP, hp = simulate_ou_path(phi, sigma, forecast=0.0, pt=1.5, sl=3.0,
                               max_hp=100, seed=0.0, rng=rng)
    assert hp == 4
    assert cP == pytest.approx(1.8840038128135759, rel=1e-9)
    assert cP > 1.5  # confirms genuine profit-take exit, not a coincidence


def test_simulate_ou_path_exits_via_stop_loss():
    # Same phi/sigma, but a shock sequence engineered to dive past a tight
    # stop-loss instead: forecast=0, phi small (fast reversion doesn't save
    # a big enough single shock), one large negative shock.
    phi, sigma = 0.3, 1.0
    rng = _fixed_shocks([-5.0])  # one huge negative draw
    cP, hp = simulate_ou_path(phi, sigma, forecast=0.0, pt=10.0, sl=2.0,
                               max_hp=100, seed=0.0, rng=rng)
    assert hp == 1
    assert cP == pytest.approx(0.7 * 0.0 + 0.3 * 0.0 + 1.0 * -5.0, rel=1e-9)
    assert cP < -2.0


def test_simulate_ou_path_exits_via_time_barrier():
    # Small, harmless shocks that never cross wide barriers -> exits only
    # because hp exceeds max_hp. max_hp=2 means the loop runs until hp=3
    # (hp > max_hp triggers on the 3rd iteration, hp counted AFTER increment).
    phi, sigma = 0.5, 0.1
    rng = _fixed_shocks([0.1, 0.1, 0.1])
    cP, hp = simulate_ou_path(phi, sigma, forecast=0.0, pt=100.0, sl=100.0,
                               max_hp=2, seed=0.0, rng=rng)
    assert hp == 3
    assert abs(cP) < 1.0  # nowhere near the (huge, deliberately unreachable) barriers


def test_simulate_ou_path_forecast_sign_symmetry():
    # Pins the book's own stated conjecture (Section 13.6.3): "Figure 13.6
    # resembles a rotated photographic negative of Figure 13.16" -- i.e.
    # flipping forecast's sign AND negating every shock should produce the
    # exact negative path at every step (provable: if P'_{t-1} = -P_{t-1},
    # then P'_t = -(1-phi)*forecast - phi*P_{t-1} - sigma*eps_t = -P_t).
    phi, sigma, forecast = 0.6, 1.3, 5.0
    shocks = [0.8, -0.4, 1.1, 0.2, -0.9]
    # max_hp = len(shocks) - 1: exits via time-barrier exactly after
    # consuming all len(shocks) draws (hp > max_hp triggers on the
    # len(shocks)-th iteration). pt/sl=100 are unreachable with these
    # shock magnitudes, so both paths are guaranteed to run the full
    # length and hit the time barrier, not a threshold.
    cP_pos, hp_pos = simulate_ou_path(
        phi, sigma, forecast, pt=100.0, sl=100.0, max_hp=len(shocks) - 1,
        seed=0.0, rng=_fixed_shocks(shocks),
    )
    cP_neg, hp_neg = simulate_ou_path(
        phi, sigma, -forecast, pt=100.0, sl=100.0, max_hp=len(shocks) - 1,
        seed=0.0, rng=_fixed_shocks([-s for s in shocks]),
    )
    assert hp_pos == hp_neg
    assert cP_neg == pytest.approx(-cP_pos, rel=1e-9)


# --------------------------------------------------------------------------- #
# batch -- mesh sweep (Snippet 13.2)
# --------------------------------------------------------------------------- #

def test_batch_returns_expected_structure():
    coeffs = {'forecast': 0.0, 'hl': 5, 'sigma': 1.0}
    r_pt = np.linspace(0, 2, 3)
    r_sl = np.linspace(0, 2, 3)
    results = batch(coeffs, n_iter=200, max_hp=20, r_pt=r_pt, r_sl=r_sl, seed=0.0)
    assert len(results) == len(r_pt) * len(r_sl)
    for pt, sl, mean, std, sharpe in results:
        assert isinstance(pt, (float, np.floating))
        assert isinstance(sl, (float, np.floating))
        assert std >= 0


def test_batch_book_reproduces_approx_sharpe_forecast0_hl5():
    # BOOK VALIDATION (Section 13.6.1 / Figure 13.1): for {forecast=0,
    # half-life=5, sigma=1}, the book reports "Sharpe ratios are high,
    # reaching levels of around 3.2." Reduced nIter (20,000 vs the book's
    # 100,000) and a coarser mesh for test speed; generous tolerance because
    # this is a stochastic simulation, not a closed-form result -- the point
    # is confirming the right BALLPARK and the right qualitative shape
    # (narrow profit-take, wide stop-loss wins), not an exact digit match.
    #
    # NOTE: an earlier version of this test called np.random.seed(1), which
    # had NO effect (simulate_ou_path's old default drew from Python's
    # built-in `random` module, not numpy's) -- see otr.py's LOAD-BEARING
    # note on simulate_ou_path. Fixed by passing random_state explicitly,
    # the parameter that actually controls batch()'s randomness now.
    coeffs = {'forecast': 0.0, 'hl': 5, 'sigma': 1.0}
    r_pt = np.linspace(0.5, 3, 6)
    r_sl = np.linspace(4, 10, 7)
    results = batch(coeffs, n_iter=20_000, max_hp=100, r_pt=r_pt, r_sl=r_sl,
                     seed=0.0, random_state=1)
    pt, sl, mean, std, sharpe = best_node(results)
    assert 2.0 < sharpe < 4.5  # book: "around 3.2"
    assert pt < sl  # book: "small profit-taking with large stop-losses"


def test_best_node_ignores_nan_sharpes():
    results = [(1.0, 1.0, 0.0, 0.0, float('nan')), (2.0, 3.0, 0.5, 1.0, 0.5)]
    pt, sl, mean, std, sharpe = best_node(results)
    assert sharpe == 0.5


def test_best_node_raises_if_all_nan():
    results = [(1.0, 1.0, 0.0, 0.0, float('nan'))]
    with pytest.raises(ValueError):
        best_node(results)
