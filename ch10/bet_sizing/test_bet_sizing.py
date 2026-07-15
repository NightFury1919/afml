"""
TDD tests for Chapter 10 -- Bet Sizing.

Known expected values are hand-traced from the book's own formulas
(Section 10.3's z/m derivation, Snippet 10.4's calibration math), not
just shape/type checks -- per the repo's TDD workflow.
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
# path convention in CLAUDE.md.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


from ch10.bet_sizing.bet_sizing import (  # noqa: E402
    getSignal,
    avgActiveSignals,
    mpAvgActiveSignals,
    discreteSignal,
    betSize,
    getTPos,
    invPrice,
    limitPrice,
    getW,
)


# ---------------------------------------------------------------------------
# Snippet 10.3 -- discreteSignal
# ---------------------------------------------------------------------------
class TestDiscreteSignal:

    def test_rounds_to_stepSize(self):
        signal0 = pd.Series([0.23, 0.47, -0.11])
        out = discreteSignal(signal0, stepSize=0.1)
        expected = pd.Series([0.2, 0.5, -0.1])
        pd.testing.assert_series_equal(out, expected)

    def test_caps_at_plus_one(self):
        signal0 = pd.Series([1.4, 2.0])
        out = discreteSignal(signal0, stepSize=0.1)
        assert (out == 1.0).all()

    def test_floors_at_minus_one(self):
        signal0 = pd.Series([-1.4, -2.0])
        out = discreteSignal(signal0, stepSize=0.1)
        assert (out == -1.0).all()


# ---------------------------------------------------------------------------
# Snippet 10.2 -- mpAvgActiveSignals (tested directly, no multiprocessing)
# ---------------------------------------------------------------------------
class TestMpAvgActiveSignals:

    def test_two_overlapping_bets_averaged(self):
        # Bet A: opened t0, closes t2. Bet B: opened t1, closes t3.
        # Between t1 and t2 both are active -> average of the two signals.
        t0, t1, t2, t3 = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02'), \
            pd.Timestamp('2026-01-03'), pd.Timestamp('2026-01-04')
        signals = pd.DataFrame(
            {'signal': [0.6, -0.2], 't1': [t2, t3]},
            index=[t0, t1],
        )
        # at t0: only bet A active -> 0.6
        out_t0 = mpAvgActiveSignals(signals, [t0])
        assert out_t0[t0] == pytest.approx(0.6)

        # at t1: both A and B active -> mean(0.6, -0.2) = 0.2
        out_t1 = mpAvgActiveSignals(signals, [t1])
        assert out_t1[t1] == pytest.approx(0.2)

        # at t2: A has closed (loc < t1 required, t2 < t2 is False) -> only B -> -0.2
        out_t2 = mpAvgActiveSignals(signals, [t2])
        assert out_t2[t2] == pytest.approx(-0.2)

    def test_no_active_bets_returns_zero(self):
        t0, t1 = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02')
        far_future = pd.Timestamp('2026-01-05')
        signals = pd.DataFrame({'signal': [0.5], 't1': [t1]}, index=[t0])
        out = mpAvgActiveSignals(signals, [far_future])
        assert out[far_future] == 0

    def test_open_ended_bet_NaT_stays_active(self):
        t0 = pd.Timestamp('2026-01-01')
        far_future = pd.Timestamp('2026-06-01')
        signals = pd.DataFrame({'signal': [0.5], 't1': [pd.NaT]}, index=[t0])
        out = mpAvgActiveSignals(signals, [far_future])
        assert out[far_future] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# avgActiveSignals edge case: empty input hits mp_pandas_obj's empty-index
# guard, which returns pd.DataFrame() rather than pd.Series(). Documented
# here as a known quirk rather than a silent assumption -- getSignal's own
# prob.shape[0]==0 guard prevents this from being hit in the real pipeline.
# ---------------------------------------------------------------------------
class TestAvgActiveSignalsEmptyEdgeCase:

    def test_empty_signals_returns_empty_dataframe_not_series(self):
        signals = pd.DataFrame({'signal': [], 't1': []})
        out = avgActiveSignals(signals, numThreads=1)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# Snippet 10.1 -- getSignal, hand-traced two-class case
# ---------------------------------------------------------------------------
class TestGetSignal:

    def test_single_bet_two_class_known_value(self):
        # Hand trace (see handoff / chat): prob=0.7, numClasses=2, pred=+1
        # z = (0.7 - 0.5) / sqrt(0.7*0.3) = 0.436435780...
        # m = 1 * (2*Phi(z) - 1) = 0.337479416...
        # discretized to stepSize=0.01 -> round(0.337479.../0.01)*0.01 = 0.34
        t0 = pd.Timestamp('2026-01-01')
        t1 = pd.Timestamp('2026-01-02')
        events = pd.DataFrame({'t1': [t1]}, index=[t0])
        prob = pd.Series([0.7], index=[t0])
        pred = pd.Series([1], index=[t0])

        out = getSignal(events, stepSize=0.01, prob=prob, pred=pred,
                         numClasses=2, numThreads=1)

        assert out[t0] == pytest.approx(0.34, abs=1e-9)
        # at t1 the bet has closed -> no active signal -> discretized to 0
        assert out[t1] == pytest.approx(0.0, abs=1e-9)

    def test_empty_prob_returns_empty_series(self):
        events = pd.DataFrame({'t1': []})
        prob = pd.Series([], dtype=float)
        pred = pd.Series([], dtype=float)
        out = getSignal(events, stepSize=0.01, prob=prob, pred=pred,
                         numClasses=2, numThreads=1)
        assert isinstance(out, pd.Series)
        assert len(out) == 0

    def test_meta_labeling_side_flips_signal(self):
        # Same prob/pred as above, but side=-1 (meta-labeling) should flip sign.
        t0 = pd.Timestamp('2026-01-01')
        t1 = pd.Timestamp('2026-01-02')
        events = pd.DataFrame({'t1': [t1], 'side': [-1]}, index=[t0])
        prob = pd.Series([0.7], index=[t0])
        pred = pd.Series([1], index=[t0])

        out = getSignal(events, stepSize=0.01, prob=prob, pred=pred,
                         numClasses=2, numThreads=1)

        assert out[t0] == pytest.approx(-0.34, abs=1e-9)


# ---------------------------------------------------------------------------
# Snippet 10.4 -- betSize / getW / getTPos / invPrice / limitPrice
# ---------------------------------------------------------------------------
class TestDynamicSizing:

    def test_getW_calibrates_betSize_to_target(self):
        # By construction, betSize(getW(x, m), x) should equal m exactly.
        for x, m in [(10, .95), (5, .5), (20, .8)]:
            w = getW(x, m)
            assert betSize(w, x) == pytest.approx(m, abs=1e-9)

    def test_book_demo_values(self):
        # Book's Snippet 10.4 main(): pos=0, maxPos=100, mP=100, f=115,
        # divergence=10, m=.95. Hand-traced (see handoff / chat):
        # w = 10.803324099723, tPos = 97, limitPrice = 112.36573855883363
        pos, maxPos, mP, f = 0, 100, 100, 115
        w = getW(10, .95)
        assert w == pytest.approx(10.803324099723, abs=1e-6)

        tPos = getTPos(w, f, mP, maxPos)
        assert tPos == 97

        lP = limitPrice(tPos, pos, f, w, maxPos)
        assert lP == pytest.approx(112.36573855883363, abs=1e-6)

    def test_invPrice_inverts_betSize(self):
        # invPrice(f, w, m) should return the price whose divergence from f
        # produces bet size m under betSize.
        w = getW(10, .95)
        f = 115
        m = 0.5
        price = invPrice(f, w, m)
        divergence = f - price
        assert betSize(w, divergence) == pytest.approx(m, abs=1e-9)
