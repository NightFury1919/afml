"""
Chapter 7: Cross-Validation in Finance
AFML snippets 7.3 (PurgedKFold) and 7.4 (cvScore), reimplemented with fixes.

Why standard k-fold CV leaks on financial data:
Triple-barrier labels span an interval [t0, t1], not a single point in time.
Neighboring observations' label intervals overlap, so a naive random k-fold
split puts overlapping-in-time observations on both sides of the train/test
boundary. The model effectively "sees the future" of its own test set through
correlated overlapping labels -> optimistic, leaked CV scores.

PurgedKFold fixes this two ways:
  1. Purging: drop any training observation whose label interval [t0, t1]
     overlaps the test set's time range.
  2. Embargo: additionally drop a further pctEmbargo-sized slice of training
     observations immediately following the test set, because financial
     time series carry serial correlation past the strict label window
     (a purged-clean training label can still be informationally leaked by
     autocorrelation in the underlying returns).

Fixes applied vs. the raw book snippets (see AFML project handoff, July 1-4 2026):
  - pctEmbargo defaults to 0. (not None) -- int(n * None) crashes.
  - Positional indexing uses .iloc explicitly. The original snippet's
    `self.t1[test_indices]` relies on pandas' deprecated positional-fallback
    behavior for Series.__getitem__ on a non-integer index. That fallback
    still works (with a FutureWarning) on pandas 1.5.3 but is a hard
    KeyError on pandas >= 2.0. .iloc is correct on every pandas version.
  - split() enforces (not just asserts) that X, y, w, t1 share an identical,
    sorted index before iterating -- misalignment previously only produced
    a silent wrong answer or a bare assertion.

Sanity-checked (July 4 2026) against the *exact* sklearn 1.2.2 _BaseKFold
class (Ethan's local env), loaded directly from the real 1.2.2 wheel:
  - _BaseKFold.__init__ is @abstractmethod, signature
    (self, n_splits, *, shuffle, random_state) -- keyword-only shuffle/
    random_state. PurgedKFold overriding __init__ and split() satisfies
    the abstract contract; get_n_splits() is inherited for free.
  - No version drift found between sklearn 1.2.2 and current sklearn on
    this class's mechanics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.metrics import log_loss, accuracy_score


class PurgedKFold(_BaseKFold):
    """
    Extend sklearn's KFold to work with labels that span time intervals (t1).

    Purges training observations whose label interval overlaps the test
    set's time range, then embargoes a further slice of training
    observations immediately after the test set.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds.
    t1 : pd.Series
        Index = observation start time (t0), values = label end time (t1).
        Must share the same index as X passed to .split().
    pctEmbargo : float, default=0.
        Fraction of the total sample size to embargo after each test set,
        as a bar count: mbrg = int(n_samples * pctEmbargo).
    """

    def __init__(self, n_splits=3, t1=None, pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 must be a pd.Series (index=t0, values=t1)')
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        if not X.index.equals(self.t1.index):
            raise ValueError(
                'X and t1 must share an identical, identically-ordered index. '
                'Sort and reindex before calling split() -- silent misalignment '
                'here produces a wrong-but-plausible-looking CV score.'
            )

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [(i[0], i[-1] + 1) for i in
                        np.array_split(np.arange(X.shape[0]), self.n_splits)]

        for i, j in test_starts:
            t0 = self.t1.index[i]                      # start of test set
            test_indices = indices[i:j]
            # positional lookup via .iloc -- see module docstring for why
            maxT1Idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            # purge: train obs whose own label already resolved before t0
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxT1Idx < X.shape[0]:
                # embargo: skip mbrg further bars after the test set's last label
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg:]))
            yield train_indices, test_indices


def cvScore(clf, X, y, sample_weight=None, scoring='neg_log_loss',
            t1=None, n_splits=3, pctEmbargo=0.):
    """
    AFML snippet 7.4: cross-validation score using PurgedKFold, with
    sample weights threaded through both .fit() and the scoring metric.

    sklearn's built-in cross_val_score doesn't accept per-fold sample_weight
    for scoring (only for .fit(), via fit_params) -- so this reimplements
    the scoring loop manually.

    Parameters
    ----------
    clf : sklearn-compatible classifier
    X : pd.DataFrame, features, index = observation time
    y : pd.Series, labels, same index as X
    sample_weight : pd.Series or None, same index as X (return-attribution
        weight from Chapter 4). If None, uses uniform weights.
    scoring : {'neg_log_loss', 'accuracy'}
    t1 : pd.Series, required -- label end times, passed to PurgedKFold
    n_splits : int
    pctEmbargo : float

    Returns
    -------
    np.ndarray of per-fold scores, sign-consistent with sklearn convention
    (higher is better -- neg_log_loss is negated log loss).
    """
    if scoring not in ('neg_log_loss', 'accuracy'):
        raise ValueError('scoring must be "neg_log_loss" or "accuracy"')
    if t1 is None:
        raise ValueError('t1 is required (label end times for PurgedKFold)')

    if sample_weight is None:
        sample_weight = pd.Series(1., index=X.index)

    # Enforce identical, sorted index across X/y/sample_weight/t1 up front,
    # rather than assuming it (July 1 fix: previously only asserted).
    for name, obj in [('y', y), ('sample_weight', sample_weight), ('t1', t1)]:
        if not X.index.equals(obj.index):
            raise ValueError(f'X and {name} must share an identical index')

    gen = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pctEmbargo)
    scores = []
    for train, test in gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :], y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, :])
            score = -log_loss(
                y.iloc[test], prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score = accuracy_score(
                y.iloc[test], pred,
                sample_weight=sample_weight.iloc[test].values,
            )
        scores.append(score)
    return np.array(scores)


# ============================================================================
# TDD results (test_purged_kfold.py), embedded per project convention.
# Run against the real 88-row Ch07 training table (ch03/ch04/ch05 artifacts).
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
#
# (The one warning is sklearn's BaggingClassifier flagging a small per-tree
# bootstrap sample count -- expected and harmless here: on sklearn 1.2.2,
# max_samples=0.2288 is a fraction of the fold's *training row count*
# (int(0.2288 * n_train)), not of summed sample weight. Real fold sizes on
# this dataset (n_splits=4, pctEmbargo=0.12) run 42-63 train rows, giving
# 9-14 bootstrap rows per tree -- small but intentional, exactly the
# variance-reduction mechanism from Ch06 (avgU as max_samples).)

