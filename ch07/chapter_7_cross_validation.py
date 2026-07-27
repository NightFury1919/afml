"""
Chapter 7: Cross-Validation in Finance
=======================================

Implements AFML snippets 7.3 (PurgedKFold) and 7.4 (cvScore) and applies
them to the real 88-row BTC/TUSD triple-barrier training table assembled
from Chapters 3-5.

Why standard k-fold CV leaks on financial data
-----------------------------------------------
Standard k-fold cross-validation assumes each observation is independent
and identically distributed (IID). A triple-barrier label isn't a single
point in time -- it's an interval [t0, t1], and neighboring observations'
label intervals overlap. A naive random k-fold split puts overlapping-in-
time observations on both sides of the train/test line, leaking information
and producing an optimistic CV score.

PurgedKFold fixes this two ways:
  1. Purging  -- drop training observations whose label interval overlaps
     the test set's time range.
  2. Embargo  -- drop a further slice of training observations immediately
     after the test set, to account for serial correlation past the
     strict label window.

This is the pipeline-artifact companion to chapter_7_cross_validation.ipynb
(per the ch0N .py/.ipynb pairing convention: pipeline/artifact steps
warrant a .py). Run directly to reproduce the notebook's real-data results.

Fixes applied vs. the raw AFML book snippets 7.3/7.4 (see project handoff,
July 1-4 2026) -- see purged_kfold.py for the full rationale:
  - pctEmbargo defaults to 0. (not None)
  - .iloc used explicitly for positional lookups (pandas >= 2.0 compat)
  - split() enforces identical/sorted index across X/y/w/t1
Sanity-checked against the exact sklearn 1.2.2 _BaseKFold class (this
repo's pinned version) before running on real data.
"""
import os
import sys

# Hybrid path convention: .py scripts derive AFML_ROOT from __file__.
AFML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

sys.path.insert(0, os.path.dirname(__file__))
from purged_kfold import PurgedKFold, cvScore

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')
N_SPLITS = 4
PCT_EMBARGO = 0.12   # int(88 * 0.01) rounds to 0 -- 0.12 covers the Ch05
                      # 10-bar fracdiff rolling window on this 88-row dataset
AVG_U = 0.2288        # Ch04 mean sample uniqueness -> BaggingClassifier max_samples


def load_training_table():
    """Assemble the X/y/w/t1 table from the Ch03-05 artifacts."""
    ch03 = pd.read_pickle(os.path.join(INPUT_DIR, 'ch03_events.pkl'))
    ch04 = pd.read_pickle(os.path.join(INPUT_DIR, 'ch04_weights.pkl'))
    ch05 = pd.read_pickle(os.path.join(INPUT_DIR, 'ch05_features.pkl'))

    X = ch05.loc[ch03.index][['fracdiff']]
    y = ch03['bin']
    w = ch04['w']
    t1 = ch03['t1']

    assert X.index.equals(y.index) and X.index.equals(w.index) and X.index.equals(t1.index), \
        'X/y/w/t1 must share an identical index before going anywhere near PurgedKFold'

    return X, y, w, t1


def fold_size_report(X, t1, n_splits=N_SPLITS, pct_embargo=PCT_EMBARGO, w=None):
    """Print + return a per-fold train/test size table."""
    pkf = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
    rows = []
    for k, (train_idx, test_idx) in enumerate(pkf.split(X)):
        row = {'fold': k, 'train_n': len(train_idx), 'test_n': len(test_idx)}
        if w is not None:
            row['train_sum_w'] = w.iloc[train_idx].sum()
        rows.append(row)
    fold_df = pd.DataFrame(rows).set_index('fold')
    print(fold_df)
    return fold_df


def run_cv(X, y, w, t1, n_splits=N_SPLITS, pct_embargo=PCT_EMBARGO, avg_u=AVG_U):
    """Score both classifiers from the Ch06->Ch07 handoff via PurgedKFold."""
    clf_rf = RandomForestClassifier(
        n_estimators=100, class_weight='balanced_subsample', random_state=1
    )
    scores_rf = cvScore(clf_rf, X, y, sample_weight=w, scoring='accuracy',
                         t1=t1, n_splits=n_splits, pctEmbargo=pct_embargo)
    print('RandomForest accuracy per fold:', np.round(scores_rf, 4))
    print('Mean:', round(scores_rf.mean(), 4), ' Std:', round(scores_rf.std(), 4))

    clf_bag = BaggingClassifier(n_estimators=100, max_samples=avg_u, random_state=1)
    scores_bag = cvScore(clf_bag, X, y, sample_weight=w, scoring='neg_log_loss',
                          t1=t1, n_splits=n_splits, pctEmbargo=pct_embargo)
    print('BaggingClassifier(max_samples=avgU) neg_log_loss per fold:', np.round(scores_bag, 4))
    print('Mean:', round(scores_bag.mean(), 4), ' Std:', round(scores_bag.std(), 4))

    return scores_rf, scores_bag


def plot_fold_sizes(fold_df, n_total, n_splits=N_SPLITS, pct_embargo=PCT_EMBARGO):
    fig, ax = plt.subplots(figsize=(7, 4))
    fold_df[['train_n', 'test_n']].plot(kind='bar', stacked=False, ax=ax,
                                         color=['#2980b9', '#e67e22'])
    ax.axhline(n_total, color='gray', linestyle='--', linewidth=1, label=f'full dataset ({n_total})')
    ax.set_title(f'PurgedKFold fold sizes (n_splits={n_splits}, pctEmbargo={pct_embargo})')
    ax.set_ylabel('rows')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cv_scores(scores_rf, scores_bag, n_splits=N_SPLITS):
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(n_splits)
    width = 0.35
    ax.bar(x_pos - width / 2, scores_rf, width, label='RandomForest (accuracy)', color='#2980b9')
    ax.bar(x_pos + width / 2, -scores_bag, width, label='Bagging (log loss, lower=better)', color='#e67e22')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Fold {k}' for k in range(n_splits)])
    ax.set_title('Purged CV scores per fold')
    ax.legend()
    plt.tight_layout()
    plt.show()


def save_outputs(X, y, w, t1, scores_rf, scores_bag):
    table = pd.concat([X, y.rename('bin'), w.rename('w'), t1.rename('t1')], axis=1)
    table.to_pickle(os.path.join(INPUT_DIR, 'ch07_training_table.pkl'))
    table.to_csv(os.path.join(INPUT_DIR, 'ch07_training_table.csv'))

    cv_results = pd.DataFrame({'rf_accuracy': scores_rf, 'bagging_neg_log_loss': scores_bag})
    cv_results.to_pickle(os.path.join(INPUT_DIR, 'ch07_cv_scores.pkl'))
    cv_results.to_csv(os.path.join(INPUT_DIR, 'ch07_cv_scores.csv'))
    print('Saved ch07_training_table.pkl/.csv and ch07_cv_scores.pkl/.csv to', INPUT_DIR)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    X, y, w, t1 = load_training_table()
    print('Training table:', X.shape[0], 'rows')
    print('bin distribution:', y.value_counts().to_dict())

    fold_df = fold_size_report(X, t1, w=w)
    plot_fold_sizes(fold_df, n_total=len(X))

    # Regression check: must not raise on any pandas version (positional-
    # indexing fix, see purged_kfold.py).
    pkf = PurgedKFold(n_splits=N_SPLITS, t1=t1, pctEmbargo=PCT_EMBARGO)
    list(pkf.split(X))
    print('PurgedKFold.split() runs clean -- no positional-indexing KeyError.')

    scores_rf, scores_bag = run_cv(X, y, w, t1)
    plot_cv_scores(scores_rf, scores_bag)

    save_outputs(X, y, w, t1, scores_rf, scores_bag)


# ============================================================================
# TDD results (test_purged_kfold.py), embedded per project convention.
# ============================================================================
#
# ============================= test session starts ==============================
# test_purged_kfold.py::test_purged_kfold_requires_series_t1 PASSED        [  6%]
# test_purged_kfold.py::test_purged_kfold_default_pctEmbargo_is_zero_not_none PASSED [ 12%]
# test_purged_kfold.py::test_purged_kfold_get_n_splits PASSED              [ 18%]
# test_purged_kfold.py::test_purged_kfold_rejects_bad_shuffle_random_state_combo PASSED [ 25%]
# test_purged_kfold.py::test_split_rejects_misaligned_index PASSED         [ 31%]
# test_purged_kfold.py::test_split_produces_non_overlapping_test_sets PASSED [ 37%]
# test_purged_kfold.py::test_split_purges_overlapping_labels PASSED        [ 43%]
# test_purged_kfold.py::test_split_keeps_train_obs_resolved_before_test_start PASSED [ 50%]
# test_purged_kfold.py::test_split_uses_iloc_not_deprecated_getitem PASSED [ 56%]
# test_purged_kfold.py::test_cvscore_requires_t1 PASSED                    [ 62%]
# test_purged_kfold.py::test_cvscore_rejects_bad_scoring PASSED            [ 68%]
# test_purged_kfold.py::test_cvscore_uniform_weight_default PASSED         [ 75%]
# test_purged_kfold.py::test_cvscore_index_mismatch_guard PASSED           [ 81%]
# test_purged_kfold.py::test_cvscore_real_data_random_forest PASSED        [ 87%]
# test_purged_kfold.py::test_cvscore_real_data_neg_log_loss PASSED         [ 93%]
# test_purged_kfold.py::test_fold_sizes_shrink_from_purging_on_real_data PASSED [100%]
# ======================== 16 passed, 1 warning in 1.74s =========================
