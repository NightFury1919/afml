"""
TDD suite for Chapter 9 (Hyper-Parameter Tuning).

Covers the three deliverables and the specific book-snippet bugs we fixed:
  * logUniform_gen  -- bounds, analytic CDF, and a KS test that log(x) really
                       is uniform in log-space (plus a regression guard for the
                       scipy "percent sign in a subclass docstring" import bug).
  * MyPipeline      -- the load-bearing check that a BARE sample_weight kwarg
                       actually reaches the final estimator (and that a plain
                       Pipeline does NOT accept it -- the reason MyPipeline
                       exists), plus a differential check that weights change
                       the fit rather than being silently dropped.
  * clfHyperFit     -- scoring branch (F1 vs neg_log_loss), grid & randomized
                       search returning a fitted pipeline with grid params, the
                       cv is actually a PurgedKFold, the Python-3 None>0 bagging
                       guard, the bagging wrap, and a real-data run on the
                       88-row BTC/TUSD table.

Assertions avoid matching sklearn's exact error-message strings (those drift
between versions); they assert behavior. Tests use tree-terminal pipelines
where SVC's tiny-bootstrap fragility would otherwise make a test flaky, and
SVC specifically for the real-data chapter showcase.
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from scipy.stats import kstest

from ch08.feature_importance.feature_importance import getTestData
from ch09.hyper_parameter_tuning.hyper_parameter_tuning import (
    MyPipeline, clfHyperFit, logUniform, logUniform_gen, _pick_scoring,
)
import ch09.hyper_parameter_tuning.hyper_parameter_tuning as ch09mod

_INPUT = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)),
    'input_data')


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def synth():
    """Small synthetic set with meta-labels {0,1}."""
    trnsX, cont = getTestData(n_features=6, n_informative=3, n_redundant=2,
                              n_samples=400, random_state=0)
    return trnsX, cont['bin'], cont['t1'], cont['w']


@pytest.fixture(scope='module')
def real():
    """The real 88-row BTC/TUSD training table (labels {-1,+1})."""
    ch03 = pd.read_csv(os.path.join(_INPUT, 'ch03_events.csv'),
                       index_col=0, parse_dates=True)
    ch04 = pd.read_csv(os.path.join(_INPUT, 'ch04_weights.csv'),
                       index_col=0, parse_dates=True)
    ch05 = pd.read_csv(os.path.join(_INPUT, 'ch05_features.csv'),
                       index_col=0, parse_dates=True)
    X = ch05.loc[ch03.index][['fracdiff']]
    y = ch03['bin']
    w = ch04['w']
    t1 = pd.to_datetime(ch03['t1'])
    return X, y, t1, w


def _tree_pipe():
    return MyPipeline([('sc', StandardScaler()),
                       ('tree', DecisionTreeClassifier(random_state=0))])


class _CaptureClf(BaseEstimator, ClassifierMixin):
    """Records the sample_weight it is handed at fit time."""
    def fit(self, X, y, sample_weight=None):
        self.captured_ = None if sample_weight is None else np.asarray(sample_weight).copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


# --------------------------------------------------------------------------- #
# logUniform (snippets 9.3 / 9.4)
# --------------------------------------------------------------------------- #
def test_loguniform_constructs_without_percent_import_bug():
    # Regression guard: a percent sign in an rv_continuous subclass docstring
    # makes scipy raise at construction. Constructing + drawing proves it's clean.
    dist = logUniform(a=1e-3, b=1e3)
    assert isinstance(dist, logUniform_gen)
    assert dist.rvs(size=5, random_state=0).shape == (5,)


def test_loguniform_docstring_has_no_percent_sign():
    # Belt-and-suspenders: keep the class docstring percent-free forever.
    assert '%' not in (logUniform_gen.__doc__ or '')


def test_loguniform_samples_stay_in_bounds():
    a, b = 1e-3, 1e3
    v = logUniform(a=a, b=b).rvs(size=20000, random_state=0)
    assert v.min() >= a - 1e-12
    assert v.max() <= b + 1e-6


def test_loguniform_cdf_endpoints_and_geometric_midpoint():
    a, b = 1e-3, 1e3
    dist = logUniform(a=a, b=b)
    assert abs(float(dist.cdf(a)) - 0.0) < 1e-9
    assert abs(float(dist.cdf(b)) - 1.0) < 1e-9
    # geometric mean sqrt(a*b) is the median of a log-uniform
    assert abs(float(dist.cdf(np.sqrt(a * b))) - 0.5) < 1e-9


def test_loguniform_is_uniform_in_log_space_ks():
    a, b = 1e-3, 1e3
    v = logUniform(a=a, b=b).rvs(size=50000, random_state=1)
    stat, p = kstest(np.log(v), 'uniform',
                     args=(np.log(a), np.log(b) - np.log(a)))
    assert p > 0.05, f'log(x) should be uniform in log-space (KS p={p:.3f})'


# --------------------------------------------------------------------------- #
# MyPipeline (snippet 9.2)
# --------------------------------------------------------------------------- #
def test_mypipeline_routes_bare_sample_weight_to_final_estimator():
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)
    w = np.arange(1., 21.)
    mp = MyPipeline([('sc', StandardScaler()), ('clf', _CaptureClf())])
    mp.fit(X, y, sample_weight=w)
    captured = mp.named_steps['clf'].captured_
    assert captured is not None, 'bare sample_weight was dropped (silent no-op)'
    assert np.allclose(captured, w)


def test_plain_pipeline_rejects_bare_sample_weight():
    # The motivation for MyPipeline: a vanilla Pipeline cannot take a bare
    # sample_weight kwarg (message text varies by sklearn version, so we only
    # assert that it raises).
    X = np.random.RandomState(0).randn(20, 3)
    y = np.array([0, 1] * 10)
    w = np.arange(1., 21.)
    pp = Pipeline([('sc', StandardScaler()), ('clf', _CaptureClf())])
    with pytest.raises(Exception):
        pp.fit(X, y, sample_weight=w)


def test_mypipeline_weights_actually_change_the_fit():
    # Differential check: weighting toward one class must change predictions,
    # proving the weights are used, not merely accepted.
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = np.array([0, 1] * 30)
    mp1 = MyPipeline([('sc', StandardScaler()),
                      ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))])
    mp2 = MyPipeline([('sc', StandardScaler()),
                      ('tree', DecisionTreeClassifier(max_depth=3, random_state=0))])
    w_skew = np.where(y == 1, 20., 1.)
    mp1.fit(X, y)
    mp2.fit(X, y, sample_weight=w_skew)
    # Heavy weight on class 1 should push at least some predictions toward it.
    assert (mp1.predict(X) != mp2.predict(X)).any()


# --------------------------------------------------------------------------- #
# scoring branch (snippet 9.1)
# --------------------------------------------------------------------------- #
def test_pick_scoring_f1_for_binary_meta_labels():
    assert _pick_scoring(np.array([0, 1, 0, 1])) == 'f1'
    assert _pick_scoring(np.array([0., 1., 1.])) == 'f1'  # float {0,1} still meta


def test_pick_scoring_neg_log_loss_otherwise():
    assert _pick_scoring(np.array([-1, 1, -1, 1])) == 'neg_log_loss'
    assert _pick_scoring(np.array([0, 1, 2])) == 'neg_log_loss'  # multiclass


# --------------------------------------------------------------------------- #
# clfHyperFit (snippets 9.1 / 9.2)
# --------------------------------------------------------------------------- #
def test_clfhyperfit_grid_returns_pipeline_with_grid_params(synth):
    feat, lbl, t1, w = synth
    grid = {'tree__max_depth': [2, 3, 5]}
    best = clfHyperFit(feat, lbl, t1, _tree_pipe(), grid, cv=3, n_jobs=1,
                       tree__sample_weight=w.values)
    assert isinstance(best, Pipeline)
    assert best.named_steps['tree'].max_depth in (2, 3, 5)


def test_clfhyperfit_randomized_runs_with_loguniform(synth):
    feat, lbl, t1, w = synth
    pipe = MyPipeline([('sc', StandardScaler()),
                       ('svc', SVC(probability=True, random_state=0))])
    dist = {'svc__C': logUniform(1e-2, 1e2), 'svc__gamma': logUniform(1e-2, 1e0)}
    best = clfHyperFit(feat, lbl, t1, pipe, dist, cv=3, rndSearchIter=8,
                       n_jobs=1, svc__sample_weight=w.values)
    assert isinstance(best, Pipeline)
    assert 1e-2 <= best.named_steps['svc'].C <= 1e2


def test_clfhyperfit_uses_purged_kfold_as_inner_cv(monkeypatch, synth):
    feat, lbl, t1, w = synth
    captured = {}
    real_gs = ch09mod.GridSearchCV

    def spy(*args, **kwargs):
        captured['cv'] = kwargs.get('cv')
        return real_gs(*args, **kwargs)

    monkeypatch.setattr(ch09mod, 'GridSearchCV', spy)
    clfHyperFit(feat, lbl, t1, _tree_pipe(), {'tree__max_depth': [2, 3]},
                cv=4, pctEmbargo=0.1, n_jobs=1, tree__sample_weight=w.values)
    cv = captured['cv']
    assert isinstance(cv, ch09mod.PurgedKFold)
    assert cv.get_n_splits() == 4
    assert cv.pctEmbargo == 0.1


def test_clfhyperfit_default_bagging_none_does_not_crash(synth):
    # Book default bagging=(0, None, 1.) -> `None > 0` would TypeError on Py3
    # without the guard. Must return the bare (non-bagged) pipeline.
    feat, lbl, t1, w = synth
    best = clfHyperFit(feat, lbl, t1, _tree_pipe(), {'tree__max_depth': [2, 3]},
                       cv=3, n_jobs=1, tree__sample_weight=w.values)
    assert isinstance(best, Pipeline)
    assert 'bag' not in best.named_steps


def test_clfhyperfit_bagging_wraps_winning_pipeline(synth):
    feat, lbl, t1, w = synth
    # Asymmetric max_samples/max_features so the bagging tuple ORDER is pinned:
    # book 9.1/9.3 is (n_estimators, max_samples, max_features).
    best = clfHyperFit(feat, lbl, t1, _tree_pipe(), {'tree__max_depth': [2, 3]},
                       cv=3, n_jobs=1, bagging=(10, 0.5, 1.0),
                       tree__sample_weight=w.values)
    assert isinstance(best, Pipeline)
    assert 'bag' in best.named_steps
    bag = best.named_steps['bag']
    assert bag.n_estimators == 10
    assert bag.max_samples == 0.5    # bagging[1]
    assert bag.max_features == 1.0   # bagging[2]
    assert isinstance(bag.estimator, MyPipeline)  # (fix) estimator=, MyPipeline base


def test_clfhyperfit_requires_index_aligned_t1(synth):
    # PurgedKFold hard-requires X.index == t1.index; a misaligned t1 must fail
    # loudly rather than silently mis-splitting.
    feat, lbl, t1, w = synth
    bad_t1 = t1.copy()
    bad_t1.index = bad_t1.index[::-1]
    with pytest.raises(Exception):
        clfHyperFit(feat, lbl, bad_t1, _tree_pipe(), {'tree__max_depth': [2]},
                    cv=3, n_jobs=1)


def test_clfhyperfit_real_data_svc_neg_log_loss(real):
    # The chapter showcase on real data: labels {-1,+1} -> neg_log_loss,
    # SVC+logUniform randomized search over purged folds, Ch04 weights routed.
    X, y, t1, w = real
    assert _pick_scoring(y.values) == 'neg_log_loss'
    pipe = MyPipeline([('sc', StandardScaler()),
                       ('svc', SVC(probability=True, random_state=0))])
    best = clfHyperFit(X, y, t1, pipe,
                       {'svc__C': logUniform(1e-2, 1e2),
                        'svc__gamma': logUniform(1e-2, 1e1)},
                       cv=4, pctEmbargo=0.12, rndSearchIter=10, n_jobs=1,
                       svc__sample_weight=w.values)
    assert isinstance(best, Pipeline)
    assert best.predict_proba(X.iloc[:3]).shape == (3, 2)
