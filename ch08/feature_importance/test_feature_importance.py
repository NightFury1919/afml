"""
TDD suite for Chapter 8 feature importance (AFML snippets 8.2-8.10).

Version-robustness note
-----------------------
featImportance builds BaggingClassifier(max_samples=1.) as a FLOAT. On sklearn
1.2.2 that is a fraction of the fold ROW count; on newer sklearn it is a
fraction of the summed SAMPLE WEIGHT. getTestData sets w = 1/n (sum = 1), which
on newer sklearn collapses max_samples=1. to a single bootstrap row per tree.
To keep these tests passing on BOTH Ethan's sklearn 1.2.2 and any newer stack,
fixtures set cont['w'] = 1.0 (uniform, sum = n) -- statistically identical to
the uniform 1/n weights but immune to the frequency-semantics difference.
The pipeline script/notebook keep the book's w = 1/n (correct on 1.2.2).
"""
import numpy as np
import pandas as pd
import pytest

from feature_importance import (
    getTestData, featImpMDI, featImpMDA, auxFeatImpSFI, featImportance,
    get_eVec, orthoFeats, featPCA_rank_corr,
)
from feature_importance import testFunc as _testFunc  # alias: avoid pytest collecting it


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def synth():
    """Modest synthetic set with a known I/R/N answer key. w -> 1.0 for
    version-robust max_samples semantics (see module docstring)."""
    np.random.seed(0)
    trnsX, cont = getTestData(n_features=12, n_informative=5, n_redundant=3,
                              n_samples=600)
    cont = cont.copy()
    cont['w'] = 1.0
    return trnsX, cont


def _by_type_mean(imp):
    d = imp[['mean']].copy()
    d['t'] = [i[0] for i in d.index]
    return d.groupby('t')['mean'].mean()


# --------------------------------------------------------------------------- #
# getTestData (8.7)
# --------------------------------------------------------------------------- #
def test_gettestdata_shapes_and_columns():
    trnsX, cont = getTestData(n_features=40, n_informative=10, n_redundant=10,
                              n_samples=500)
    assert trnsX.shape == (500, 40)
    assert list(cont.columns) == ['bin', 'w', 't1']
    prefixes = [c[0] for c in trnsX.columns]
    assert prefixes.count('I') == 10
    assert prefixes.count('R') == 10
    assert prefixes.count('N') == 20


def test_gettestdata_column_order_is_I_then_R_then_N():
    trnsX, _ = getTestData(n_features=10, n_informative=4, n_redundant=3,
                           n_samples=100)
    assert list(trnsX.columns) == ['I_0', 'I_1', 'I_2', 'I_3',
                                   'R_0', 'R_1', 'R_2', 'N_0', 'N_1', 'N_2']


def test_gettestdata_weights_and_t1():
    _, cont = getTestData(n_samples=250)
    assert cont['w'].nunique() == 1
    assert cont['w'].sum() == pytest.approx(1.0)
    # t1 equals the observation's own timestamp -> no synthetic label overlap
    assert (cont['t1'] == cont.index).all()
    assert set(cont['bin'].unique()).issubset({0, 1})


# --------------------------------------------------------------------------- #
# featImpMDI (8.2)
# --------------------------------------------------------------------------- #
def test_mdi_normalizes_and_covers_all_features(synth):
    trnsX, cont = synth
    imp, _, _ = featImportance(trnsX, cont, n_estimators=50, cv=3,
                               scoring='accuracy', method='MDI')
    assert set(imp.index) == set(trnsX.columns)
    assert imp['mean'].sum() == pytest.approx(1.0, abs=1e-6)
    assert (imp['mean'].dropna() >= 0).all()


def test_mdi_informative_beats_noise(synth):
    trnsX, cont = synth
    imp, _, _ = featImportance(trnsX, cont, n_estimators=60, cv=3,
                               scoring='accuracy', method='MDI')
    m = _by_type_mean(imp)
    assert m['I'] > m['N']  # informative more important than noise


# --------------------------------------------------------------------------- #
# featImpMDA (8.3) -- including a direct regression test for the shuffle bug
# --------------------------------------------------------------------------- #
def test_permutation_idiom_actually_shuffles():
    """Regression guard for the book's np.random.shuffle(X[j].values) no-op:
    our reassignment idiom must actually change the column."""
    np.random.seed(1)
    df = pd.DataFrame({'a': np.arange(50.0)})
    before = df['a'].tolist()
    df['a'] = np.random.permutation(df['a'].values)
    assert before != df['a'].tolist()
    assert sorted(before) == sorted(df['a'].tolist())  # same multiset


def test_mda_noise_near_zero_informative_positive(synth):
    trnsX, cont = synth
    np.random.seed(0)
    imp, _, _ = featImportance(trnsX, cont, n_estimators=60, cv=3,
                               scoring='accuracy', method='MDA')
    m = _by_type_mean(imp)
    assert m['I'] > m['N']              # informative beats noise
    assert abs(m['N']) < 0.05           # noise importance ~ 0 (MDA's strength)


def test_mda_rejects_bad_scoring(synth):
    trnsX, cont = synth
    with pytest.raises(Exception):
        featImpMDA(clf=None, X=trnsX, y=cont['bin'], cv=3,
                   sample_weight=cont['w'], t1=cont['t1'], pctEmbargo=0.,
                   scoring='f1')


# --------------------------------------------------------------------------- #
# auxFeatImpSFI (8.4)
# --------------------------------------------------------------------------- #
def test_sfi_returns_per_feature_mean_std(synth):
    trnsX, cont = synth
    imp, _, _ = featImportance(trnsX, cont, n_estimators=40, cv=3,
                               scoring='accuracy', method='SFI')
    assert set(imp.index) == set(trnsX.columns)
    assert {'mean', 'std'}.issubset(imp.columns)
    assert imp['mean'].notna().all()


# --------------------------------------------------------------------------- #
# Orthogonal features (8.5) + weighted tau (8.6)
# --------------------------------------------------------------------------- #
def test_orthofeats_shape_and_decorrelation(synth):
    trnsX, _ = synth
    dfP = orthoFeats(trnsX, varThres=.95)
    assert dfP.shape[0] == trnsX.shape[0]
    assert dfP.shape[1] <= trnsX.shape[1]
    # principal components are (near) uncorrelated: off-diагonal corr ~ 0
    C = np.corrcoef(dfP, rowvar=False)
    off = C[~np.eye(C.shape[0], dtype=bool)]
    assert np.abs(off).max() < 1e-6


def test_get_evec_threshold_reduces_dimension(synth):
    trnsX, _ = synth
    dfZ = trnsX.sub(trnsX.mean(), axis=1).div(trnsX.std(), axis=1)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index=trnsX.columns,
                       columns=trnsX.columns)
    eVal_full, _ = get_eVec(dot, varThres=1.0)
    eVal_part, _ = get_eVec(dot, varThres=.80)
    assert len(eVal_part) <= len(eVal_full)


def test_featpca_rank_corr_matches_book_example():
    # AFML snippet 8.6 toy: featImp=[.55,.33,.07,.05], pcRank=[1,2,4,3]
    tau = featPCA_rank_corr(np.array([.55, .33, .07, .05]),
                            np.array([1, 2, 4, 3]))
    assert tau > 0.5
    assert tau == pytest.approx(0.8133, abs=0.05)


# --------------------------------------------------------------------------- #
# featImportance driver (8.8) + testFunc (8.9)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('method', ['MDI', 'MDA', 'SFI'])
def test_featimportance_returns_triplet(synth, method):
    trnsX, cont = synth
    imp, oob, oos = featImportance(trnsX, cont, n_estimators=40, cv=3,
                                   scoring='accuracy', method=method)
    assert 0.0 <= oob <= 1.0
    assert np.isfinite(oos)
    assert 'mean' in imp.columns


def test_featimportance_rejects_bad_method(synth):
    trnsX, cont = synth
    with pytest.raises(ValueError):
        featImportance(trnsX, cont, n_estimators=20, cv=3, method='XYZ')


def test_testfunc_summary_table():
    np.random.seed(0)
    out = _testFunc(n_features=10, n_informative=4, n_redundant=3,
                   n_estimators=40, n_samples=300, cv=3)
    assert list(out['method']) == ['MDI', 'MDA', 'SFI']
    for col in ['I', 'R', 'N', 'oob', 'oos']:
        assert col in out.columns
