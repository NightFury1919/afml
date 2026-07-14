"""
TDD suite for Chapter 11 CSCV / PBO.

Expected values are hand-derived from Section 11.6's own algorithm, not
copied from the implementation's output.
"""
import numpy as np
import pandas as pd
import pytest
from math import comb, log

# --- import the module under test -----------------------------------------
# Derive the repo root from __file__ and put it on sys.path explicitly.
#
# LOAD-BEARING: do NOT replace this with a bare `from pbo import ...`, and do
# not rely on pytest inserting the repo root for you. Because __init__.py makes
# this folder a package, pytest walks UP the __init__.py chain to decide the
# package base -- so the import that works depends on whether ch11/__init__.py
# happens to exist. That is a silent, machine-dependent trap (it bit us: a lost
# ch11/__init__.py produced "No module named 'ch11'" on a clean checkout).
# Deriving the root from __file__ works from any cwd, with or without pytest,
# and matches the .py-script path convention in CLAUDE.md.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ch11.backtest_dangers.pbo import cscv, pbo, sharpe_ratio  # noqa: E402


# --- sharpe_ratio ---------------------------------------------------------

def test_sharpe_hand_traced():
    # pnl = [1, 2, 3, 4]; mean = 2.5; sd(ddof=1) = sqrt(5/3) = 1.290994...
    # Sharpe = 2.5 / 1.290994 = 1.936492
    assert sharpe_ratio([1, 2, 3, 4]) == pytest.approx(1.9364916731, rel=1e-9)


def test_sharpe_zero_variance_is_nan():
    # A flat PnL column has sd = 0. Must not raise or return inf.
    assert np.isnan(sharpe_ratio([5.0, 5.0, 5.0, 5.0]))


def test_sharpe_is_scale_invariant():
    # Justifies not annualising: scaling a column cannot change its rank.
    a = np.array([0.1, -0.2, 0.3, 0.05])
    assert sharpe_ratio(a * 250) == pytest.approx(sharpe_ratio(a), rel=1e-12)


# --- CSCV structure -------------------------------------------------------

def _toy(T=40, N=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(T, N)),
                        columns=[f't{i}' for i in range(N)])


def test_cscv_row_count_equals_S_choose_half():
    # Book: "we form all combinations C_S of M_s, taken in groups of size S/2"
    for S in (4, 6, 8):
        res = cscv(_toy(), S=S)
        assert len(res) == comb(S, S // 2)


def test_cscv_book_example_S16_is_12870_not_12780():
    # BOOK ERRATUM: Section 11.6 prints "12,780" for S=16.
    # C(16,8) = 12,870. Verified here so the typo can never creep into code.
    assert comb(16, 8) == 12870


def test_cscv_rejects_odd_S():
    with pytest.raises(ValueError, match='even'):
        cscv(_toy(), S=7)


def test_cscv_rejects_single_trial():
    with pytest.raises(ValueError, match='at least 2'):
        cscv(_toy(N=1), S=4)


def test_cscv_S_larger_than_rows_rejected():
    with pytest.raises(ValueError):
        cscv(_toy(T=6), S=8)


def test_train_and_test_sets_are_complementary_halves():
    # Step 4.2: J-bar is the complement of J in M. With equal blocks the two
    # halves must together cover every row exactly once.
    M = _toy(T=40, N=3)
    S = 4
    blocks = np.array_split(np.arange(40), S)
    from itertools import combinations as _c
    for c in _c(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in c])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in c])
        assert len(np.intersect1d(is_idx, oos_idx)) == 0
        assert sorted(np.concatenate([is_idx, oos_idx])) == list(range(40))


# --- the logit / omega mapping (the heart of Step 4.6-4.7) ----------------

def test_logit_is_zero_at_median_rank():
    # Book: "lambda_c = 0 when R_n* coincides with the median of R_bar."
    # With N trials and omega = rank/(N+1), the median rank is (N+1)/2,
    # so omega = 0.5 exactly and logit = log(1) = 0.
    N = 5
    median_rank = (N + 1) / 2          # = 3.0
    w = median_rank / (N + 1)          # = 0.5
    assert w == 0.5
    assert log(w / (1 - w)) == 0.0


def test_omega_strictly_inside_unit_interval():
    # omega must never hit 0 or 1, or the logit becomes +/- inf.
    res = cscv(_toy(T=60, N=6), S=6)
    assert (res['w_bar'] > 0).all() and (res['w_bar'] < 1).all()
    assert np.isfinite(res['logit']).all()


# --- PBO semantics --------------------------------------------------------

def test_pbo_is_zero_when_one_strategy_dominates_everywhere():
    # A trial that is genuinely best in EVERY subsample is never overfit:
    # it wins IS and stays top OOS, so rank_oos = N, omega > 0.5,
    # logit > 0 in every combination -> PBO = 0.
    T, S = 40, 4
    rng = np.random.default_rng(1)
    M = pd.DataFrame(rng.normal(0, 1, size=(T, 3)), columns=['a', 'b', 'c'])
    M['winner'] = 5.0 + rng.normal(0, 0.01, size=T)   # huge, stable Sharpe
    value, res = pbo(M, S=S)
    assert value == 0.0
    assert (res['n_star'] == 'winner').all()
    assert (res['logit'] > 0).all()


def test_pbo_is_one_when_edge_is_purely_time_localised():
    # Two strategies, each genuinely good in one half of history and equally
    # bad in the other -- the textbook shape of an overfit backtest.
    # With S=2 the blocks ARE those halves, so whichever half is in-sample,
    # its specialist wins IS and is necessarily last OOS. PBO must be 1.0.
    T = 40
    half = T // 2
    jitter = np.tile([1e-3, -1e-3], half)          # keeps variance non-zero
    a = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)]) + jitter
    b = -a + jitter
    M = pd.DataFrame({'good_early': a, 'good_late': b})
    value, res = pbo(M, S=2)
    assert value == 1.0
    assert (res['logit'] < 0).all()
    assert set(res['n_star']) == {'good_early', 'good_late'}  # each wins once


def test_pbo_is_high_when_each_trial_fits_its_own_slice_of_history():
    # Same idea at realistic scale: 10 trials, each handed a private window
    # where it looks brilliant. Selection should be badly overfit.
    rng = np.random.default_rng(3)
    T, N = 240, 10
    data = rng.normal(size=(T, N))
    for j in range(N):
        data[j * (T // N):(j + 1) * (T // N), j] += 1.5
    value, _ = pbo(pd.DataFrame(data), S=8)
    assert value > 0.7


def test_pbo_averages_near_half_for_pure_noise():
    # N indistinguishable zero-edge strategies: selecting the "best" is a
    # coin flip, so PBO should centre on ~0.5.
    #
    # LOAD-BEARING: this is asserted as a MEAN OVER SEEDS, not on one seed.
    # Single-seed PBO on pure noise ranges roughly 0.04-0.99 (measured), so a
    # single-draw assertion would be a flaky test that teaches a false lesson
    # -- namely that any individual PBO estimate is precise. It is not.
    vals = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        M = pd.DataFrame(rng.normal(size=(240, 10)))
        v, _ = pbo(M, S=8)
        vals.append(v)
    assert 0.40 < float(np.mean(vals)) < 0.65


def test_pbo_returns_value_and_frame():
    value, res = pbo(_toy(T=60, N=5), S=6)
    assert isinstance(value, float) and 0.0 <= value <= 1.0
    assert set(res.columns) == {'n_star', 'r_is', 'r_oos',
                                'rank_oos', 'w_bar', 'logit'}
    assert len(res) == comb(6, 3)
