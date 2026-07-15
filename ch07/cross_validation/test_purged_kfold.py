"""
TDD suite for Chapter 7: PurgedKFold and cvScore.

Run with: pytest -v test_purged_kfold.py

Run-order independence: the real-data tests below do NOT depend on
chapter_7_cross_validation.py having been run first. The real_table
fixture builds the 88-row training table on the fly from the ch03/ch04/ch05
artifacts in input_data/. If those upstream artifacts aren't present either,
the real-data tests skip cleanly (rather than erroring) so the synthetic
tests still run and report green.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from ch07.cross_validation.purged_kfold import (  # noqa: E402
    PurgedKFold, cvScore,
)


# Resolve input_data/ relative to this test file: ch07/cross_validation/ -> ../../input_data
_INPUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'input_data')
)


def _build_real_table():
    """
    Assemble the X/y/w/t1 table from the ch03/ch04/ch05 artifacts, mirroring
    load_training_table() in chapter_7_cross_validation.py. Prefers a
    pre-built ch07_training_table.pkl if present, otherwise builds from the
    upstream chapter pkls. Returns None if the inputs aren't available.
    """
    prebuilt = os.path.join(_INPUT_DIR, 'ch07_training_table.pkl')
    if os.path.exists(prebuilt):
        try:
            return pd.read_pickle(prebuilt)
        except Exception:
            pass  # fall through to rebuild from upstream artifacts

    paths = {name: os.path.join(_INPUT_DIR, f'{name}.pkl')
             for name in ('ch03_events', 'ch04_weights', 'ch05_features')}
    if not all(os.path.exists(p) for p in paths.values()):
        return None

    try:
        ch03 = pd.read_pickle(paths['ch03_events'])
        ch04 = pd.read_pickle(paths['ch04_weights'])
        ch05 = pd.read_pickle(paths['ch05_features'])
    except Exception:
        # e.g. numpy 2.x pkls being read under numpy 1.x -- skip rather than
        # fail the whole suite (see project note on cross-env pickle incompat).
        return None

    X = ch05.loc[ch03.index][['fracdiff']]
    table = pd.concat(
        [X, ch03['bin'].rename('bin'), ch04['w'].rename('w'), ch03['t1'].rename('t1')],
        axis=1,
    )
    return table


# ---------- fixtures ----------

@pytest.fixture
def small_t1():
    """20 obs, each label spans 3 days forward (like a fixed vertical barrier)."""
    idx = pd.date_range('2026-01-01', periods=20, freq='D')
    t1 = pd.Series(idx + pd.Timedelta(days=3), index=idx)
    t1.iloc[-3:] = idx[-1]  # can't look past the end of the sample
    return t1


@pytest.fixture
def small_X(small_t1):
    return pd.DataFrame({'feat': np.arange(len(small_t1))}, index=small_t1.index)


@pytest.fixture
def real_table():
    """
    The real 88-row Ch07 training table (fracdiff/bin/w/t1), built on the fly
    from the chapter artifacts. Skips the dependent test if the upstream
    input_data pkls aren't available (so run order never matters).
    """
    table = _build_real_table()
    if table is None:
        pytest.skip(
            'Real-data artifacts not available in input_data/ '
            '(ch03_events/ch04_weights/ch05_features .pkl). '
            'Run the chapter pipeline in the mlfinlab env first.'
        )
    return table


# ---------- PurgedKFold: construction / sklearn contract ----------

def test_purged_kfold_requires_series_t1(small_X):
    with pytest.raises(ValueError, match='pd.Series'):
        PurgedKFold(n_splits=4, t1=[1, 2, 3])  # not a Series


def test_purged_kfold_default_pctEmbargo_is_zero_not_none():
    # July-1 fix: pctEmbargo defaults to 0., not None (int(n*None) used to crash)
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    t1 = pd.Series(idx, index=idx)
    pkf = PurgedKFold(n_splits=2, t1=t1)
    assert pkf.pctEmbargo == 0.
    X = pd.DataFrame({'f': range(10)}, index=idx)
    # should not raise -- int(n_samples * 0.) = 0, valid
    list(pkf.split(X))


def test_purged_kfold_get_n_splits(small_X, small_t1):
    pkf = PurgedKFold(n_splits=4, t1=small_t1)
    assert pkf.get_n_splits() == 4


def test_purged_kfold_rejects_bad_shuffle_random_state_combo(small_t1):
    # Inherited from sklearn's _BaseKFold.__init__ validation
    class BadKFold(PurgedKFold.__bases__[0]):
        def __init__(self, n_splits, t1):
            super().__init__(n_splits, shuffle=False, random_state=42)
    with pytest.raises(ValueError, match='random_state'):
        BadKFold(3, small_t1)


# ---------- PurgedKFold: split() correctness ----------

def test_split_rejects_misaligned_index(small_X, small_t1):
    pkf = PurgedKFold(n_splits=4, t1=small_t1, pctEmbargo=0.1)
    bad_X = small_X.iloc[::-1]  # reversed order -> different index order
    with pytest.raises(ValueError, match='identical'):
        list(pkf.split(bad_X))


def test_split_produces_non_overlapping_test_sets(small_X, small_t1):
    pkf = PurgedKFold(n_splits=4, t1=small_t1, pctEmbargo=0.1)
    all_test = []
    for train, test in pkf.split(small_X):
        all_test.extend(test.tolist())
    assert sorted(all_test) == list(range(len(small_X)))  # test sets partition the full index


def test_split_purges_overlapping_labels(small_X, small_t1):
    """
    Hand-verified case: test = obs[0:5] (Jan1-5), whose labels span up to Jan8.
    No training observation from Jan6-Jan9 (label window + embargo) should
    appear in train.
    """
    pkf = PurgedKFold(n_splits=4, t1=small_t1, pctEmbargo=0.1)
    train, test = next(pkf.split(small_X))
    assert list(test) == [0, 1, 2, 3, 4]
    train_dates = small_X.index[train]
    forbidden = pd.date_range('2026-01-06', '2026-01-09')
    assert not any(d in train_dates for d in forbidden)


def test_split_keeps_train_obs_resolved_before_test_start(small_X, small_t1):
    """
    Fold 1: test = obs[5:10] (Jan6-10), t0=Jan6. Obs 0,1,2 (labels resolve by
    Jan6) should remain in train -- purging only removes *overlapping* labels,
    not everything before the test set.
    """
    pkf = PurgedKFold(n_splits=4, t1=small_t1, pctEmbargo=0.1)
    folds = list(pkf.split(small_X))
    train1, test1 = folds[1]
    assert list(test1) == [5, 6, 7, 8, 9]
    assert {0, 1, 2}.issubset(set(train1.tolist()))


def test_split_uses_iloc_not_deprecated_getitem(small_X, small_t1):
    """
    Regression test for the pandas positional-indexing fix: this must not
    raise KeyError on pandas >= 2.0 (where Series.__getitem__ no longer
    falls back to positional lookup on a non-integer index).
    """
    pkf = PurgedKFold(n_splits=4, t1=small_t1, pctEmbargo=0.1)
    list(pkf.split(small_X))  # should not raise


# ---------- cvScore ----------

def test_cvscore_requires_t1():
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    X = pd.DataFrame({'f': range(10)}, index=idx)
    y = pd.Series(np.random.choice([-1, 1], 10), index=idx)
    with pytest.raises(ValueError, match='t1 is required'):
        cvScore(DummyClassifier(), X, y, t1=None)


def test_cvscore_rejects_bad_scoring():
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    X = pd.DataFrame({'f': range(10)}, index=idx)
    y = pd.Series(np.random.choice([-1, 1], 10), index=idx)
    t1 = pd.Series(idx, index=idx)
    with pytest.raises(ValueError, match='scoring'):
        cvScore(DummyClassifier(), X, y, t1=t1, scoring='rmse')


def test_cvscore_uniform_weight_default(small_X, small_t1):
    y = pd.Series(np.random.choice([-1, 1], len(small_X)), index=small_X.index)
    scores = cvScore(DummyClassifier(strategy='most_frequent'), small_X, y,
                      t1=small_t1, scoring='accuracy', n_splits=4, pctEmbargo=0.1)
    assert len(scores) == 4
    assert np.all(np.isfinite(scores))


def test_cvscore_index_mismatch_guard(small_X, small_t1):
    y = pd.Series(np.random.choice([-1, 1], len(small_X)), index=small_X.index)
    bad_w = pd.Series(1., index=small_X.index[::-1])  # reversed -> misaligned
    with pytest.raises(ValueError, match='sample_weight'):
        cvScore(DummyClassifier(), small_X, y, sample_weight=bad_w,
                t1=small_t1, scoring='accuracy')


def test_cvscore_real_data_random_forest(real_table):
    X = real_table[['fracdiff']]
    y = real_table['bin']
    w = real_table['w']
    t1 = real_table['t1']
    clf = RandomForestClassifier(n_estimators=50, class_weight='balanced_subsample',
                                  random_state=1)
    scores = cvScore(clf, X, y, sample_weight=w, scoring='accuracy',
                      t1=t1, n_splits=4, pctEmbargo=0.12)
    assert len(scores) == 4
    assert np.all((scores >= 0) & (scores <= 1))


def test_cvscore_real_data_neg_log_loss(real_table):
    from sklearn.ensemble import BaggingClassifier
    X = real_table[['fracdiff']]
    y = real_table['bin']
    w = real_table['w']
    t1 = real_table['t1']
    clf = BaggingClassifier(n_estimators=50, max_samples=0.2288, random_state=1)
    scores = cvScore(clf, X, y, sample_weight=w, scoring='neg_log_loss',
                      t1=t1, n_splits=4, pctEmbargo=0.12)
    assert len(scores) == 4
    assert np.all(scores <= 0)  # neg_log_loss is always <= 0


def test_fold_sizes_shrink_from_purging_on_real_data(real_table):
    """Sanity check matching the July-2 conceptual deep-dive: a real middle
    fold purges roughly a third of its would-be training rows."""
    X = real_table[['fracdiff']]
    t1 = real_table['t1']
    pkf = PurgedKFold(n_splits=4, t1=t1, pctEmbargo=0.12)
    fold_sizes = [len(train) for train, test in pkf.split(X)]
    assert all(0 < n < len(X) for n in fold_sizes)
    assert max(fold_sizes) < len(X) - 22  # test fold size is always 22 here
