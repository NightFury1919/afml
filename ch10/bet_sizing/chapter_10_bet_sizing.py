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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(__file__))
from bet_sizing import getSignal  # noqa: E402

sys.path.insert(0, os.path.join(AFML_ROOT, 'ch07', 'cross_validation'))
from purged_kfold import PurgedKFold  # noqa: E402

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')
N_SPLITS = 4          # matches Ch07's real-data calibration for this dataset
PCT_EMBARGO = 0.12     # matches Ch07's real-data calibration for this dataset
STEP_SIZE = 0.01
# Ch09's real-data grid-search winner (neg_log_loss) on the ENRICHED table,
# read from ch09_hyperparameter_tuning_stats.csv (`real,grid,0.01`).
#
# RESOLVED 2026-07-21 -- LOAD-BEARING, and the history matters.
#
# This constant was 100.0 until today. That was Ch09's grid winner on the
# ORIGINAL single-feature table. When Ch19's enrichment landed, Ch09 was
# re-run and its optimum moved to 0.01 -- four orders of magnitude of extra
# regularization, which is what you would expect once there are 12 correlated
# features on 87 rows instead of 1. Ch10 was not re-run at the time, so for
# several days this file cited a source that disagreed with it.
#
# It is now 0.01 because Ch10 loads the enriched table (see
# load_training_table below). The two move together: C is tuned FOR a dataset,
# so the constant and the loader are a single decision, not two.
#
# CORRECTION to a claim made in this comment on 2026-07-21: an earlier version
# said changing this value "cascades into Ch11, which imports getSignal from
# here." The import is real but the cascade is not. Ch11 imports only
# getSignal -- a pure function of (events, stepSize, prob, pred) -- and builds
# its OWN SVC from its own C_GRID and GAMMA. SVC_C never reaches Ch11. The
# blast radius of this constant is Ch10's own demo output and nothing else.
#
# PRECONDITION: 0.01 is only meaningful because out_of_sample_probs now fits
# inside a StandardScaler Pipeline. Ch09 tuned this value on scaled features;
# using it on raw features would be borrowing a constant calibrated under a
# transform we don't apply. If the scaler is ever removed, this number is
# invalid and must be re-tuned.
SVC_C = 0.01
SVC_GAMMA = 0.1


def load_training_table():
    """The real Ch03-05(+19) BTC/TUSD triple-barrier table: 12 features
    (fracdiff + Ch19's 11 microstructural features), bin label, w (Ch04
    sample-uniqueness weight), t1 (label end time).

    MIGRATED 2026-07-21 -- LOAD-BEARING. This used to load
    ch07_training_table.csv (88 rows, fracdiff only). It now loads the
    enriched artifact (87 rows, 12 features; one event was dropped by
    build_enriched_training_table.py for still sitting inside a Ch19
    rolling-window warmup, the same convention Ch05 already uses for
    fracdiff's own FFD warmup).

    Why: before this change the pipeline alternated between the two tables --
    Ch09 enriched, Ch10 thin, Ch11 thin, Ch12 and Ch14 enriched. Chapters 10
    and 11 were a single-feature island in the middle of an enriched run,
    which meant their results were not comparable with the chapters on either
    side of them. Ch07 deliberately stays on the thin table: it is chapter 7,
    the enrichment is built by chapter 19, and pointing Ch07 at the enriched
    artifact would create a forward dependency that breaks book order for
    anyone reading in sequence.

    Falls back to the original table if the enriched artifact is absent (e.g.
    running this chapter standalone before Ch19 exists on a machine), and says
    so out loud rather than silently -- a silent fallback here would produce
    single-feature results that look identical in shape to enriched ones.
    """
    enriched = os.path.join(INPUT_DIR, 'ch07_training_table_enriched.csv')
    if os.path.exists(enriched):
        return pd.read_csv(enriched, index_col=0, parse_dates=[0, 't1'])
    print('  WARNING: ch07_training_table_enriched.csv not found -- falling '
          'back to the original single-feature ch07_training_table.csv. '
          'Results will NOT match the published Ch10 output.')
    path = os.path.join(INPUT_DIR, 'ch07_training_table.csv')
    return pd.read_csv(path, index_col=0, parse_dates=[0, 't1'])


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
        # StandardScaler Pipeline is LOAD-BEARING (added 2026-07-21, same fix
        # Ch09 and Ch12 already carry). A bare SVC on raw X was harmless while
        # X was the single fracdiff column. On the enriched table the feature
        # magnitudes span ~3.8e9 to 1 (kyle_lambda mean |x| ~2.4e3 next to
        # amihud_lambda_20bar ~6.3e-07), and an unscaled RBF kernel's squared
        # distance is then dominated almost entirely by the largest-magnitude
        # column -- the other ten features stop contributing. That yields a
        # near-useless fit that still returns plausible-looking probabilities,
        # which is the dangerous kind of wrong. The scaler is refit per fold on
        # the TRAINING fold only (inside the Pipeline), so no test-fold
        # statistics leak. sample_weight is routed with the svc__ prefix, the
        # standard Pipeline mechanism.
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=C, gamma=gamma, probability=True,
                        random_state=random_state)),
        ])
        pipe.fit(X.iloc[train, :], y.iloc[train],
                 svc__sample_weight=w.iloc[train].values)
        proba = pipe.predict_proba(X.iloc[test, :])
        idx_max = proba.argmax(axis=1)
        prob.iloc[test] = proba[np.arange(len(test)), idx_max]
        pred.iloc[test] = pipe.named_steps['svc'].classes_[idx_max]
    return prob, pred


def main():
    table = load_training_table()
    # All feature columns, not a hardcoded [['fracdiff']] -- same pattern as
    # Ch09 and Ch12, so adding a feature upstream never silently bypasses this
    # chapter again.
    feature_cols = [c for c in table.columns if c not in ('bin', 'w', 't1')]
    X = table[feature_cols]
    print(f'  training table: {X.shape[0]} events x {X.shape[1]} features')
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
