"""
Chapter 8: Feature Importance
AFML snippets 8.2-8.10, reimplemented with Python-3 / pandas-1.5.3 /
sklearn-1.2.2 fixes and wired to this repo's Chapter 7 PurgedKFold/cvScore.

Why this chapter exists
-----------------------
Chapter 7 gave us an *honest* way to score a model (purged, embargoed CV).
Chapter 8 asks the next question: *which features actually matter?* Getting
this right is hard because there is normally no ground truth to check against.
The trick (getTestData, snippet 8.7) is to *manufacture* a dataset whose
answer key is known: features named I_* are informative, R_* are redundant
(linear combinations of the informative ones, carrying no NEW information),
and N_* are pure noise. A good importance method should rank I_* and R_*
high and push N_* toward zero. Comparing methods on this known dataset is
how we validate a validation method.

Four importance methods, differing on two axes -- in-sample vs out-of-sample,
and whether they suffer the *substitution effect* (two correlated features
"sharing" importance so each looks weaker than it is):

  MDI  (8.2) in-sample, tree impurity. Fast, but noise never scores zero
       and it suffers substitution effects.
  MDA  (8.3) out-of-sample, permutation, uses PurgedKFold. Pushes noise
       nearer zero than MDI; still suffers substitution effects.
  SFI  (8.4) out-of-sample, one feature at a time. Immune to substitution
       effects but blind to joint (pairwise-only) effects.
  Orthogonal features (8.5) decorrelate first (PCA), sidestepping
       substitution effects; snippet 8.6 then checks whether the ML
       importance ranking agrees with the PCA eigenvalue ranking via a
       weighted Kendall's tau.

Fixes applied vs. the raw book snippets
---------------------------------------
1. getTestData (8.7): the book's
      pd.DatetimeIndex(periods=, freq=, end=)
   is not a valid constructor in modern pandas and pd.datetime.today() was
   removed. Rewritten with pd.date_range(end=datetime.today(), periods=,
   freq=BDay()). xrange -> range.

2. featImpMDA (8.3): the book permutes a column with
      np.random.shuffle(X1_[j].values)
   which is a SILENT NO-OP on pandas 1.5.3 (.values can be a copy, not a
   view) and a hard read-only error on pandas >= 2.0 -- in both cases MDA
   would report ~zero importance for every feature while looking like it
   ran. Fixed by reassigning a permutation:
      X1_[j] = np.random.permutation(X1_[j].values)

3. featImportance (8.8): BaggingClassifier's base_estimator= was renamed to
   estimator= in sklearn 1.2 (FutureWarning on 1.2.2, removed in 1.4).
   Uses estimator=.

4. Wiring: the book's cvScore takes a pre-built cvGen= and a cv= alias.
   THIS repo's Chapter 7 cvScore (see ch07/cross_validation/purged_kfold.py)
   instead takes n_splits=/t1=/pctEmbargo= and builds PurgedKFold internally.
   Because PurgedKFold uses shuffle=False, rebuilding it from the same
   (t1, n_splits, pctEmbargo) is deterministic and yields *identical* folds
   to a shared cvGen -- so we thread those args through instead of a cvGen.
   Ch07 is left untouched.

5. testFunc (8.9): izip -> zip, print statement -> print(), added the
   itertools.product import, and out[['a','b',...]] double-bracket fix.

6. plotFeatImportance (8.10): str(tag) so an int default doesn't crash the
   title concatenation.

LOAD-BEARING (max_samples semantics)
------------------------------------
featImportance builds BaggingClassifier(max_samples=1.) as a FLOAT on
purpose. On sklearn 1.2.2 (this repo's env) a float max_samples is a
fraction of the fold's ROW COUNT: int(max_samples * n_rows). Newer sklearn
changed float max_samples to a fraction of the summed SAMPLE WEIGHT. Because
getTestData sets w = 1/n (weights sum to 1.0), on newer sklearn
max_samples=1. collapses to int(1.0 * 1.0) = 1 bootstrap row per tree --
every tree trains on a single observation, every importance goes to
~NaN/zero, oob ~0.50. DO NOT change this float to satisfy a newer-sklearn
warning without also rescaling the weights; on 1.2.2 the float is correct.
"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd

# --- cross-chapter import: reuse Chapter 7's PurgedKFold + cvScore ----------
# .py convention in this repo is __file__-relative paths. This module lives at
#   <repo>/ch08/feature_importance/feature_importance.py
# so the repo root is two directories up. Adjust the two os.pardir hops if your
# ch07 package lives elsewhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ch07.cross_validation.purged_kfold import PurgedKFold, cvScore  # noqa: E402


# ===========================================================================
# Snippet 8.7 -- synthetic dataset with a known answer key
# ===========================================================================
def getTestData(n_features=40, n_informative=10, n_redundant=10,
                n_samples=10000, random_state=0):
    """
    Build a classification dataset whose columns are labelled by role:
    I_* informative, R_* redundant (linear combos of informative), N_* noise.

    make_classification(shuffle=False) lays features out in exactly that
    order, so positional naming is correct. Returns (trnsX, cont) where cont
    is a DataFrame with columns ['bin','w','t1']:
      bin : label (0/1)
      w   : sample weight (1/n; uniform)
      t1  : label end time -- here just the observation's own timestamp, so
            synthetic labels do NOT overlap (purging barely bites; that's
            intentional, it isolates the substitution-effect lesson).
    """
    from sklearn.datasets import make_classification
    trnsX, cont = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=n_informative, n_redundant=n_redundant,
        random_state=random_state, shuffle=False,
    )
    # FIX: pd.DatetimeIndex(periods=,freq=,end=) invalid + pd.datetime removed.
    idx = pd.date_range(end=dt.datetime.today(), periods=n_samples,
                        freq=pd.tseries.offsets.BDay())
    trnsX = pd.DataFrame(trnsX, index=idx)
    cont = pd.Series(cont, index=idx).to_frame('bin')
    # FIX: xrange -> range
    names = (['I_' + str(i) for i in range(n_informative)] +
             ['R_' + str(i) for i in range(n_redundant)])
    names += ['N_' + str(i) for i in range(n_features - len(names))]
    trnsX.columns = names
    cont['w'] = 1. / cont.shape[0]
    cont['t1'] = pd.Series(cont.index, index=cont.index)
    return trnsX, cont


# ===========================================================================
# Snippet 8.2 -- MDI (Mean Decrease Impurity), in-sample
# ===========================================================================
def featImpMDI(fit, featNames):
    """
    Mean impurity reduction per feature, averaged across the forest's trees.

    df0.replace(0, np.nan): with max_features=1 each tree splits on a random
    single feature, so a tree that never split on feature f reports 0 for it.
    Those structural zeros are excluded (set to NaN) so the mean reflects
    trees that actually *used* the feature, not trees that never saw it.

    Note (book behaviour, preserved): the std error uses df0.shape[0]
    (total tree count) rather than the per-feature non-NaN count.
    """
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = pd.concat({'mean': df0.mean(),
                     'std': df0.std() * df0.shape[0] ** -.5}, axis=1)
    imp /= imp['mean'].sum()
    return imp


# ===========================================================================
# Snippet 8.3 -- MDA (Mean Decrease Accuracy), out-of-sample, purged CV
# ===========================================================================
def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo,
               scoring='neg_log_loss'):
    """
    Out-of-sample importance: fit on each purged-CV train fold, score the
    test fold, then re-score with one feature's column permuted. The drop in
    score is that feature's importance. Uses this repo's PurgedKFold.

    Returns (imp, mean_baseline_score) where imp has columns ['mean','std'].
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss, accuracy_score

    cvGen = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)  # purged cv
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns, dtype=float)

    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, w0 = X.iloc[train, :], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test, :], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.values)

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            # FIX: book's np.random.shuffle(X1_[j].values) is a silent no-op
            # on pandas 1.5.3 and read-only error on >=2.0. Reassign instead.
            X1_[j] = np.random.permutation(X1_[j].values)  # permute one column
            if scoring == 'neg_log_loss':
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1.values,
                                           labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1.values)

    imp = (-scr1).add(scr0, axis=0)
    if scoring == 'neg_log_loss':
        imp = imp / -scr1
    else:
        imp = imp / (1. - scr1)
    imp = pd.concat({'mean': imp.mean(),
                     'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
    return imp, scr0.mean()


# ===========================================================================
# Snippet 8.4 -- SFI (Single Feature Importance), out-of-sample
# ===========================================================================
def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, n_splits, pctEmbargo,
                  n_jobs=1):
    """
    Score each feature *in isolation* via purged CV. Immune to substitution
    effects (correlated features can't dilute each other when scored alone)
    but blind to joint effects (a pair that only matters together).

    This is the slow path: one CV (n_splits fits) per feature. Features are
    independent, so we parallelize ACROSS features (the book's mpPandasObj
    approach) rather than across trees -- pass n_jobs=4 with a single-threaded
    clf (featImportance sets clf.n_jobs=1 before calling this) to avoid nested
    parallelism. Each worker gets a fresh clone of clf.

    Wiring note: the book passes a pre-built cvGen to cvScore. This repo's
    cvScore builds PurgedKFold internally from n_splits/t1/pctEmbargo, which
    (shuffle=False) reproduces identical folds for every feature.
    """
    from sklearn.base import clone

    def _one(featName):
        df0 = cvScore(clone(clf), X=trnsX[[featName]], y=cont['bin'],
                      sample_weight=cont['w'], scoring=scoring,
                      t1=cont['t1'], n_splits=n_splits, pctEmbargo=pctEmbargo)
        return featName, df0.mean(), df0.std() * df0.shape[0] ** -.5

    if n_jobs and n_jobs != 1:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(_one)(f) for f in featNames)
    else:
        rows = [_one(f) for f in featNames]

    imp = pd.DataFrame(rows, columns=['_feat', 'mean', 'std']).set_index('_feat')
    imp.index.name = None
    return imp


# ===========================================================================
# Snippet 8.5 -- orthogonal (PCA) features
# ===========================================================================
def get_eVec(dot, varThres):
    """Eigenvectors of the dot-product (correlation) matrix, kept up to a
    cumulative-variance threshold varThres. Returns (eVal, eVec)."""
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]                       # sort eigenvalues desc
    eVal, eVec = eVal[idx], eVec[:, idx]
    eVal = pd.Series(eVal, index=['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    eVec = pd.DataFrame(eVec, index=dot.index, columns=eVal.index)
    eVec = eVec.loc[:, eVal.index]
    cumVar = eVal.cumsum() / eVal.sum()
    dim = cumVar.values.searchsorted(varThres)
    eVal, eVec = eVal.iloc[:dim + 1], eVec.iloc[:, :dim + 1]
    return eVal, eVec


def orthoFeats(dfX, varThres=.95):
    """
    Standardize features, then project onto principal components explaining
    at least varThres of the variance. Returns the orthogonal feature matrix
    dfP (np.ndarray). PCA is unsupervised -- it never sees the labels -- so
    agreement between PCA-importance and ML-importance is meaningful evidence
    the ML result reflects real structure, not overfitting (see snippet 8.6).
    """
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)   # standardize
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=dfX.columns, columns=dfX.columns)
    eVal, eVec = get_eVec(dot, varThres)
    dfP = np.dot(dfZ, eVec)
    return dfP


# ===========================================================================
# Snippet 8.6 -- weighted Kendall's tau: ML importance vs inverse PCA rank
# ===========================================================================
def featPCA_rank_corr(featImp, pcRank):
    """
    Weighted Kendall's tau between feature importance and inverse PCA rank
    (pcRank ** -1). A high positive value means the features the ML model
    found important are also the high-variance PCA directions -- corroborating
    the ML ranking with an unsupervised one. featImp and pcRank are 1-D arrays
    aligned by feature.
    """
    from scipy.stats import weightedtau
    return weightedtau(np.asarray(featImp, dtype=float),
                       np.asarray(pcRank, dtype=float) ** -1.)[0]


# ===========================================================================
# Snippet 8.8 -- driver: feature importance by any method
# ===========================================================================
def featImportance(trnsX, cont, n_estimators=1000, cv=10, max_samples=1.,
                   pctEmbargo=0, scoring='accuracy', method='SFI',
                   minWLeaf=0., n_jobs=1, **kargs):
    """
    Fit a bagged tree ensemble and compute feature importance by MDI, MDA,
    or SFI. Returns (imp, oob, oos). n_jobs controls tree-level parallelism
    (default 1; use 4 or -1 for real runs -- SFI in particular fits one bag
    per feature per CV fold and is the slow path).

    SFI here runs serially (over trnsX.columns). The book parallelizes
    auxFeatImpSFI via its multiprocessing engine; that can be reinstated with
    this repo's utils.multiprocess.mp_pandas_obj -- auxFeatImpSFI already has
    an mp-compatible ('featNames', columns) first argument. Serial is used by
    default to stay robust to Windows spawn re-import quirks in a teaching run.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier

    # max_features=1 (INT) on the tree = one feature per split. This is the
    # deliberate "anti-masking" trick so no single feature hogs importance.
    clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                 class_weight='balanced',
                                 min_weight_fraction_leaf=minWLeaf)
    # estimator= (sklearn>=1.2; was base_estimator=). max_features=1. and
    # max_samples=1. are FLOATS here (100% of features / rows). See the
    # LOAD-BEARING note in the module docstring about float max_samples.
    # n_jobs parallelizes tree-building across cores. Default 1 (deterministic,
    # no process-spawn surprise in tests); pass n_jobs=4 (this repo's
    # convention) or -1 for real runs -- the CV/MDA/SFI refits are the cost.
    clf = BaggingClassifier(estimator=clf, n_estimators=n_estimators,
                            max_features=1., max_samples=max_samples,
                            oob_score=True, n_jobs=n_jobs)
    fit = clf.fit(X=trnsX, y=cont['bin'], sample_weight=cont['w'].values)
    oob = fit.oob_score_

    if method == 'MDI':
        imp = featImpMDI(fit, featNames=trnsX.columns)
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],
                      scoring=scoring, t1=cont['t1'], n_splits=cv,
                      pctEmbargo=pctEmbargo).mean()
    elif method == 'MDA':
        imp, oos = featImpMDA(clf, X=trnsX, y=cont['bin'], cv=cv,
                              sample_weight=cont['w'], t1=cont['t1'],
                              pctEmbargo=pctEmbargo, scoring=scoring)
    elif method == 'SFI':
        oos = cvScore(clf, X=trnsX, y=cont['bin'], sample_weight=cont['w'],
                      scoring=scoring, t1=cont['t1'], n_splits=cv,
                      pctEmbargo=pctEmbargo).mean()
        clf.n_jobs = 1  # parallelize the per-feature loop instead of the trees
        imp = auxFeatImpSFI(trnsX.columns, clf=clf, trnsX=trnsX, cont=cont,
                            scoring=scoring, n_splits=cv, pctEmbargo=pctEmbargo,
                            n_jobs=n_jobs)
    else:
        raise ValueError("method must be 'MDI', 'MDA', or 'SFI'")
    return imp, oob, oos


# ===========================================================================
# Snippet 8.9 -- run all three methods on artificial data and summarize
# ===========================================================================
def testFunc(n_features=40, n_informative=10, n_redundant=10,
             n_estimators=1000, n_samples=10000, cv=10,
             scoring='accuracy', pctEmbargo=0., pathOut=None):
    """
    Run MDI, MDA, SFI on one synthetic dataset and return a summary DataFrame
    with, per method, the fraction of total importance captured by the I / R /
    N feature groups plus oob and oos. On a good method the I column is high
    and the N column low. (izip->zip, print()->func, out[[...]] fixed.)
    """
    from itertools import product
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    out = []
    for method in ['MDI', 'MDA', 'SFI']:
        imp, oob, oos = featImportance(
            trnsX=trnsX, cont=cont, n_estimators=n_estimators, cv=cv,
            scoring=scoring, pctEmbargo=pctEmbargo, method=method)
        df0 = imp[['mean']] / imp['mean'].abs().sum()
        df0['type'] = [i[0] for i in df0.index]
        grp = df0.groupby('type')['mean'].sum().to_dict()
        row = {'method': method, 'scoring': scoring,
               'I': grp.get('I', 0.), 'R': grp.get('R', 0.),
               'N': grp.get('N', 0.), 'oob': oob, 'oos': oos}
        out.append(row)
    out = pd.DataFrame(out)[['method', 'scoring', 'I', 'R', 'N', 'oob', 'oos']]
    if pathOut is not None:
        out.to_csv(pathOut)
    return out


# ===========================================================================
# Snippet 8.10 -- plotting (matplotlib), py3-safe
# ===========================================================================
def plotFeatImportance(imp, oob=None, oos=None, method='', tag=0, simNum=0,
                       savePath=None, ax=None):
    """
    Horizontal bar chart of mean importance with std error bars. For MDI a
    dotted line marks the 1/n_features 'uniform importance' reference.
    Returns the matplotlib Axes so a notebook can display it inline; pass
    savePath to also write a PNG.
    """
    import matplotlib.pyplot as plt
    imp = imp.sort_values('mean', ascending=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(3, imp.shape[0] / 5.)))
    ax.barh(range(imp.shape[0]), imp['mean'].values,
            xerr=imp['std'].values, color='b', alpha=.25,
            error_kw={'ecolor': 'r'})
    ax.set_yticks(range(imp.shape[0]))
    ax.set_yticklabels(imp.index)
    if method == 'MDI':
        ax.axvline(1. / imp.shape[0], linewidth=1, color='r', linestyle='dotted')
    title = 'tag=' + str(tag) + ' | simNum=' + str(simNum)
    if oob is not None:
        title += ' | oob=' + str(round(oob, 4))
    if oos is not None:
        title += ' | oos=' + str(round(oos, 4))
    ax.set_title(title)
    if savePath is not None:
        ax.figure.savefig(savePath, dpi=100, bbox_inches='tight')
    return ax


# ============================================================================
# TDD results (test_feature_importance.py), embedded per project convention.
#
# NOTE: this run is from the delivery sandbox (Python 3.12.3 / sklearn 1.8.0),
# NOT the mlfinlab env. Re-run under mlfinlab (Python 3.10.20 / sklearn 1.2.2)
# to capture the canonical header; there the max_samples float warning below
# does NOT appear (on 1.2.2 a float max_samples is a fraction of the fold ROW
# count, so weights summing to 1 are irrelevant -- see the module docstring).
# Fixtures set w=1.0 precisely so the suite is green on both stacks.
# ============================================================================
#
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 17 items
#
# test_gettestdata_shapes_and_columns PASSED                              [  5%]
# test_gettestdata_column_order_is_I_then_R_then_N PASSED                 [ 11%]
# test_gettestdata_weights_and_t1 PASSED                                  [ 17%]
# test_mdi_normalizes_and_covers_all_features PASSED                      [ 23%]
# test_mdi_informative_beats_noise PASSED                                 [ 29%]
# test_permutation_idiom_actually_shuffles PASSED                         [ 35%]
# test_mda_noise_near_zero_informative_positive PASSED                    [ 41%]
# test_mda_rejects_bad_scoring PASSED                                     [ 47%]
# test_sfi_returns_per_feature_mean_std PASSED                            [ 52%]
# test_orthofeats_shape_and_decorrelation PASSED                          [ 58%]
# test_get_evec_threshold_reduces_dimension PASSED                        [ 64%]
# test_featpca_rank_corr_matches_book_example PASSED                      [ 70%]
# test_featimportance_returns_triplet[MDI] PASSED                         [ 76%]
# test_featimportance_returns_triplet[MDA] PASSED                         [ 82%]
# test_featimportance_returns_triplet[SFI] PASSED                         [ 88%]
# test_featimportance_rejects_bad_method PASSED                           [ 94%]
# test_testfunc_summary_table PASSED                                      [100%]
#
# ======================= 17 passed, 42 warnings in 16.94s =======================
#
# (The 42 warnings are all sklearn 1.8's max_samples frequency-semantics notice,
# emitted only by test_testfunc_summary_table, which alone exercises the
# canonical w=1/n weights. Harmless here; absent on sklearn 1.2.2.)
