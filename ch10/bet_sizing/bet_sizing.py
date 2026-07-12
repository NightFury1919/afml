"""
Chapter 10 -- Bet Sizing
========================
Implements AFML Snippets 10.1-10.4 (Lopez de Prado):

  10.1  getSignal            -- probability -> bet size (one-vs-rest, OvR)
  10.2  avgActiveSignals /
        mpAvgActiveSignals   -- average bets that are concurrently open
  10.3  discreteSignal       -- discretize bet size to avoid overtrading
  10.4  betSize / getTPos /
        invPrice / limitPrice /
        getW                -- dynamic position sizing + limit price

Book-fidelity notes
--------------------
- Snippets 10.1/10.2 dispatch through this repo's shared multiprocessing
  engine (utils/multiprocess.py: mp_pandas_obj), the snake_case
  reimplementation of the book's mpPandasObj already in use since Ch04.
  Only the dispatch call name changes; molecule handling, kwarg-passing,
  and concatenation behavior are unchanged from what 10.1/10.2 expect.
- Snippet 10.4's limitPrice used Python 2's `xrange`, which does not
  exist in Python 3.10 (this repo's environment). Replaced with the
  Python 3 builtin `range` -- functionally identical, not a semantic
  change to the book's math.
- All other lines are unchanged from the printed snippets. The two-class
  and one-vs-rest z-statistic derivations in 10.1 were checked against
  the book's Section 10.3 text (z = (p - 1/numClasses) / sqrt(p(1-p)),
  m = x*(2*Z[z]-1)) and match exactly.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

# path/portability convention: derive repo root from this file's location
# (this file lives at <root>/ch10/bet_sizing/bet_sizing.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.multiprocess import mp_pandas_obj  # noqa: E402


# ---------------------------------------------------------------------------
# Snippet 10.1 -- From probabilities to bet size
# ---------------------------------------------------------------------------
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    """
    Translate classifier probabilities into a final, discretized bet size.

    Why: a raw predicted probability isn't itself a bet size. We test the
    probability against the null hypothesis that all outcomes are equally
    likely (p = 1/numClasses) -- the further the probability sits from
    that null, the more confident the prediction, and the larger the bet.
    We then (optionally) fold in meta-labeling's side, average all bets
    that are concurrently open (so we don't over-commit before a signal
    has fully strengthened), and discretize so we don't churn the
    position on every tiny probability wiggle.
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)

    # 1) generate signals from multinomial classification (one-vs-rest, OvR)
    signal0 = (prob - 1. / numClasses) / (prob * (1. - prob)) ** .5  # t-value of OvR
    signal0 = pred * (2 * norm.cdf(signal0) - 1)  # signal = side * size

    if 'side' in events:
        signal0 = signal0 * events.loc[signal0.index, 'side']  # meta-labeling

    # 2) compute average signal among those concurrently open
    df0 = signal0.to_frame('signal').join(events[['t1']], how='left')
    df0 = avgActiveSignals(df0, numThreads)
    signal1 = discreteSignal(signal0=df0, stepSize=stepSize)
    return signal1


# ---------------------------------------------------------------------------
# Snippet 10.2 -- Bets are averaged as long as they are still active
# ---------------------------------------------------------------------------
def avgActiveSignals(signals, numThreads):
    """
    Compute the average signal among those active, at every point in time
    where the set of active signals could have changed (a signal starting
    or a signal ending).

    Why: two bets opened at different times but both still open right now
    should be blended, not simply overwritten by whichever was issued
    last -- otherwise we lose the earlier bet's information the moment a
    newer, possibly weaker, signal arrives.
    """
    # 1) time points where signals change (either one starts or one ends)
    tPnts = set(signals['t1'].dropna().values)
    tPnts = tPnts.union(signals.index.values)
    tPnts = list(tPnts)
    tPnts.sort()
    out = mp_pandas_obj(mpAvgActiveSignals, ('molecule', tPnts), numThreads, signals=signals)
    return out


def mpAvgActiveSignals(signals, molecule):
    """
    At time loc, average signal among those still active.
    Signal is active if:
        a) issued before or at loc AND
        b) loc before signal's endtime, or endtime is still unknown (NaT).
    """
    out = pd.Series(dtype=float)
    for loc in molecule:
        df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
        act = signals[df0].index
        if len(act) > 0:
            out[loc] = signals.loc[act, 'signal'].mean()
        else:
            out[loc] = 0  # no signals active at this time
    return out


# ---------------------------------------------------------------------------
# Snippet 10.3 -- Size discretization to prevent overtrading
# ---------------------------------------------------------------------------
def discreteSignal(signal0, stepSize):
    """
    Round the bet size to the nearest stepSize, and cap it at +/-1.

    Why: without this, the position would be re-sized on every marginal
    change in predicted probability, generating transaction costs with no
    corresponding edge. Discretizing means we only trade when the signal
    has moved enough to matter.
    """
    signal1 = (signal0 / stepSize).round() * stepSize  # discretize
    signal1[signal1 > 1] = 1  # cap
    signal1[signal1 < -1] = -1  # floor
    return signal1


# ---------------------------------------------------------------------------
# Snippet 10.4 -- Dynamic position size and limit price
# ---------------------------------------------------------------------------
def betSize(w, x):
    """
    Bet size as a sigmoid-like function of price divergence x, calibrated
    by w. Why sigmoid-shaped: bet size should grow with divergence between
    the forecast price and the market price, but saturate as it approaches
    full size (+/-1) rather than growing without bound.
    """
    return x * (w + x ** 2) ** -.5


def getTPos(w, f, mP, maxPos):
    """
    Target position (in contracts/shares): scale betSize's [-1, 1] output
    by the divergence between forecast price f and market price mP, times
    the max position allowed.
    """
    return int(betSize(w, f - mP) * maxPos)


def invPrice(f, w, m):
    """
    Inverse of betSize: given a target bet size m and calibration w, solve
    for the market price that would produce it, relative to forecast f.
    """
    return f - m * (w / (1 - m ** 2)) ** .5


def limitPrice(tPos, pos, f, w, maxPos):
    """
    Volume-weighted average limit price for moving from the current
    position pos to the target position tPos, one contract at a time --
    each incremental contract is priced via invPrice at its own bet-size
    level, then averaged.

    Book-fidelity note: uses `range` here; the book's printed snippet
    used Python 2's `xrange`, unavailable in Python 3.10.
    """
    sgn = 1 if tPos >= pos else -1
    lP = 0
    for j in range(abs(pos + sgn), abs(tPos + 1)):
        lP += invPrice(f, w, j / float(maxPos))
    lP /= tPos - pos
    return lP


def getW(x, m):
    """
    Calibrate w such that betSize(w, x) == m, i.e. solve for the w that
    makes a divergence of x produce exactly bet size m. Used once, up
    front, to fit the sizing curve to a chosen (divergence, max bet size)
    pair -- e.g. "at a $10 divergence, I want a 95% bet size."
    Requires 0 < alpha < 1 where alpha = m in this parametrization.
    """
    return x ** 2 * (m ** -2 - 1)


def main():
    """Book's Snippet 10.4 demo driver: calibrate w, then get a target
    position and a limit price for it."""
    pos, maxPos, mP, f, wParams = 0, 100, 100, 115, {'divergence': 10, 'm': .95}
    w = getW(wParams['divergence'], wParams['m'])  # calibrate w
    tPos = getTPos(w, f, mP, maxPos)  # get tPos
    lP = limitPrice(tPos, pos, f, w, maxPos)  # limit price for order
    return tPos, lP


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Pytest results (sandbox validation -- Python 3.12.3, pandas 3.0.2,
# scipy 1.17.1, numpy 2.4.4). Confirmed on real mlfinlab env (Python
# 3.10.20 / pandas 1.5.3 / sklearn 1.2.2) via chapter_10_bet_sizing.py --
# see project chat, July 2026.
#
# Fixes applied after the real-machine run surfaced them (not book-snippet
# bugs -- environment/determinism issues in the demo pipeline around this
# module):
#   - Empty pd.Series() calls (lines flagged by pandas FutureWarning) now
#     specify dtype=float explicitly.
#   - chapter_10_bet_sizing.py's SVC(probability=True) now pins
#     random_state=0 -- without it, predict/predict_proba's internal
#     Platt-scaling CV is non-deterministic, and differed enough between
#     sklearn 1.2.2 (real machine) and a later sklearn (sandbox) to flip
#     which class won on this small, single-feature dataset.
#
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 13 items
#
# tests/test_bet_sizing.py::TestDiscreteSignal::test_rounds_to_stepSize PASSED
# tests/test_bet_sizing.py::TestDiscreteSignal::test_caps_at_plus_one PASSED
# tests/test_bet_sizing.py::TestDiscreteSignal::test_floors_at_minus_one PASSED
# tests/test_bet_sizing.py::TestMpAvgActiveSignals::test_two_overlapping_bets_averaged PASSED
# tests/test_bet_sizing.py::TestMpAvgActiveSignals::test_no_active_bets_returns_zero PASSED
# tests/test_bet_sizing.py::TestMpAvgActiveSignals::test_open_ended_bet_NaT_stays_active PASSED
# tests/test_bet_sizing.py::TestAvgActiveSignalsEmptyEdgeCase::test_empty_signals_returns_empty_dataframe_not_series PASSED
# tests/test_bet_sizing.py::TestGetSignal::test_single_bet_two_class_known_value PASSED
# tests/test_bet_sizing.py::TestGetSignal::test_empty_prob_returns_empty_series PASSED
# tests/test_bet_sizing.py::TestGetSignal::test_meta_labeling_side_flips_signal PASSED
# tests/test_bet_sizing.py::TestDynamicSizing::test_getW_calibrates_betSize_to_target PASSED
# tests/test_bet_sizing.py::TestDynamicSizing::test_book_demo_values PASSED
# tests/test_bet_sizing.py::TestDynamicSizing::test_invPrice_inverts_betSize PASSED
#
# ============================== 13 passed in 1.11s ===============================
# ---------------------------------------------------------------------------
