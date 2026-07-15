"""
Chapter 10 -- Bet Sizing: real-data demo
=========================================

Runs the AFML Snippet 10.1 pipeline (getSignal) end-to-end on the real
88-row BTC/TUSD triple-barrier training table (Ch03-05), using Ch09's
real winning SVC hyperparameters (C=100, gamma=0.1 -- the real-data
grid-search winner scored on neg_log_loss, from
ch09_hyperparameter_tuning_stats.csv/pkl).

Why out-of-sample probabilities, not a plain refit
----------------------------------------------------
Feeding getSignal a classifier's in-sample predict_proba would be
lookahead bias: the classifier would be sizing a bet using information
from observations it was trained on, including ones that happened after
the very bet being sized. Ch07's PurgedKFold (n_splits=4,
pctEmbargo=0.12, this dataset's established calibration) gives every
observation a probability from a fold that never trained on it, and
purges/embargoes around it -- matching how a live signal would actually
have been produced.

No saved Ch09 fitted classifier exists on disk (only its winning
hyperparameters were persisted) -- see project chat, July 2026. Refitting
with those real hyperparameters on real data, out-of-sample via
PurgedKFold, is the faithful way to get real prob/pred for getSignal
without a saved model object.
"""
import os
import sys

AFML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from bet_sizing import getSignal  # noqa: E402

sys.path.insert(0, os.path.join(AFML_ROOT, 'ch07', 'cross_validation'))
from purged_kfold import PurgedKFold  # noqa: E402

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')
N_SPLITS = 4          # matches Ch07's real-data calibration for this dataset
PCT_EMBARGO = 0.12     # matches Ch07's real-data calibration for this dataset
STEP_SIZE = 0.01
# Ch09's real-data grid-search winner (neg_log_loss), from
# ch09_hyperparameter_tuning_stats.csv/pkl -- see project chat, July 2026.
SVC_C = 100.0
SVC_GAMMA = 0.1


def load_training_table():
    """The real Ch03-05 BTC/TUSD triple-barrier table: fracdiff feature,
    bin label, w (Ch04 sample-uniqueness weight), t1 (label end time)."""
    path = os.path.join(INPUT_DIR, 'ch07_training_table.csv')
    df = pd.read_csv(path, index_col=0, parse_dates=[0, 't1'])
    return df


def out_of_sample_probs(X, y, w, t1, n_splits=N_SPLITS, pct_embargo=PCT_EMBARGO,
                         C=SVC_C, gamma=SVC_GAMMA, random_state=0):
    """
    For each PurgedKFold test fold, fit SVC(C, gamma, probability=True) on
    the (purged+embargoed) training fold, predict_proba on the held-out
    fold, and record each observation's winning-class probability and
    predicted label. No n_jobs / joblib parallelism anywhere here -- this
    is a plain per-fold .fit()/.predict_proba() loop, not a grid search,
    so the Windows joblib/loky + SVC(probability=True) crash risk flagged
    in the Ch09 handoff doesn't apply, but staying single-threaded here
    regardless.

    random_state is required, not optional: SVC(probability=True) fits an
    internal randomized 5-fold CV (Platt scaling) to calibrate
    predict_proba. Leaving random_state=None makes predict/predict_proba
    non-deterministic run-to-run, and -- confirmed empirically, see
    project chat -- the internal CV split can differ enough between
    sklearn versions (1.2.2 real vs later) to flip which class wins on a
    small, thin (single-feature) dataset like this one. Pinning
    random_state=0 makes every fold's fit reproducible.

    Returns
    -------
    prob : pd.Series, index = X.index -- p(predicted class) per observation
    pred : pd.Series, index = X.index -- predicted label per observation
    """
    gen = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
    prob = pd.Series(index=X.index, dtype=float)
    pred = pd.Series(index=X.index, dtype=float)
    for train, test in gen.split(X=X):
        clf = SVC(C=C, gamma=gamma, probability=True, random_state=random_state)
        clf.fit(X.iloc[train, :], y.iloc[train], sample_weight=w.iloc[train].values)
        proba = clf.predict_proba(X.iloc[test, :])
        idx_max = proba.argmax(axis=1)
        prob.iloc[test] = proba[np.arange(len(test)), idx_max]
        pred.iloc[test] = clf.classes_[idx_max]
    return prob, pred


def main():
    table = load_training_table()
    X = table[['fracdiff']]
    y = table['bin']
    w = table['w']
    t1 = table['t1']
    events = table[['t1']]

    prob, pred = out_of_sample_probs(X, y, w, t1)
    print('Out-of-sample prob/pred summary:')
    print(pd.DataFrame({'prob': prob, 'pred': pred}).describe())

    signal = getSignal(events, stepSize=STEP_SIZE, prob=prob, pred=pred,
                        numClasses=2, numThreads=1)
    print('\nDiscretized bet size summary:')
    print(signal.describe())
    print('\nValue counts:')
    print(signal.value_counts().sort_index())

    fig, ax = plt.subplots(figsize=(9, 4))
    signal.sort_index().plot(ax=ax, drawstyle='steps-post', marker='o', markersize=3)
    ax.set_title('Ch10: discretized bet size over time (real BTC/TUSD data)')
    ax.set_ylabel('bet size')
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color='gray', linewidth=0.8)
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), 'ch10_signal_over_time.png')
    fig.savefig(out_png, dpi=120)
    print(f'\nSaved plot: {out_png}')

    return signal


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# TDD results mirror -- same suite as bet_sizing.py's embedded block and this
# chapter's notebook, duplicated here per the .py/.ipynb mirror convention
# (this script is the ipynb's paired mirror, not bet_sizing.py).
# ---------------------------------------------------------------------------
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
