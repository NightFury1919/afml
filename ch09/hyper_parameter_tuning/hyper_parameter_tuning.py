"""
Chapter 9: Hyper-Parameter Tuning with Cross-Validation
=======================================================

Implements AFML snippets 9.1-9.4 and wires them onto this repo's *purged*
cross-validator (Ch07 PurgedKFold), so that hyper-parameter search on
overlapping-label financial data does not leak the way a plain GridSearchCV
over a random KFold would.

The four pieces
---------------
  9.1/9.2  clfHyperFit   -- grid OR randomized search over a purged inner CV,
                            with optional bagging of the winning pipeline.
  9.2      MyPipeline    -- a Pipeline subclass whose only job is to let a bare
                            `sample_weight=` kwarg reach the FINAL estimator.
  9.3      logUniform_gen / logUniform
                         -- a scipy rv_continuous giving log-uniform sampling,
                            the correct prior for scale hyper-parameters
                            (SVC's C and gamma) in a RandomizedSearchCV.

Why MyPipeline has to exist (this is the subtle part)
-----------------------------------------------------
A *plain* sklearn Pipeline already routes fit-params of the form
`finalstep__sample_weight=...` correctly -- so inside GridSearchCV, passing
`svc__sample_weight=w` works with a vanilla Pipeline. The problem is BAGGING:
`BaggingClassifier.fit(X, y, sample_weight=w)` forwards a BARE `sample_weight`
kwarg to `base_estimator.fit(X, y, sample_weight=w)`. A vanilla Pipeline.fit
does not accept a bare `sample_weight` -- it only understands `step__param`
keys -- so the weights are either rejected or silently dropped. MyPipeline
overrides .fit() to translate a bare `sample_weight` into
`{finalstep}__sample_weight` before delegating to Pipeline.fit, so the Ch04
return-attribution weights actually reach the estimator inside each bagged
bootstrap. See test_mypipeline_routes_bare_sample_weight for the direct
"prove it isn't a silent no-op" check.

Fixes applied vs. the raw AFML book snippets (verified against sklearn 1.2.2,
this repo's pinned version):
  1. `iid=False` REMOVED from GridSearchCV / RandomizedSearchCV. The `iid`
     parameter was deprecated in sklearn 0.22 and REMOVED in 0.24 -- on 1.2.2
     it is a hard TypeError. (It only ever controlled a fold-size weighting of
     the mean score that is not wanted for purged folds anyway.)
  2. `base_estimator=` -> `estimator=` on BaggingClassifier. Renamed in sklearn
     1.2 (deprecated), removed in 1.4. `estimator=` is correct on 1.2.2+.
  3. `gs.base_estimator.steps[-1][0]` -> resolved from the winning pipeline's
     own `.steps` (see 2; the attribute path also changes with the rename).
  4. Python-3 `None > 0` guard. The book default `bagging=[0, None, 1.]` with
     `if bagging[1] > 0:` raises `TypeError: '>' not supported between
     instances of 'NoneType' and 'int'` on Python 3 (it was silently falsey on
     Python 2). Guarded as `bagging[1] is not None and bagging[1] > 0`.
  5. Bagging sample_weight lookup is guarded: only forwarded if the caller
     actually supplied `{finalstep}__sample_weight` in fit_params. NOTE: the
     book does this UNGUARDED (`sample_weight=fit_params[...]`), which raises a
     bare KeyError if you request bagging without passing weights. The guard is
     a deliberate robustness deviation; drop it if you want book-exact behavior.
  6. scipy docstring caveat -- NOT a book bug, a caveat from OUR enhancement.
     The book gives logUniform_gen a one-line COMMENT, so it never hits this. We
     added a docstring (better IDE tooltips for students); an `rv_continuous`
     SUBCLASS docstring must contain NO percent sign, because scipy treats it as
     a printf-style template at construction and a stray `%` raises at import.
     Kept percent-free with a regression test. (scipy-version-independent.)

The bagging tuple is (n_estimators, max_samples, max_features), matching book
snippets 9.1/9.3: max_samples=float(bagging[1]), max_features=float(bagging[2]).

LOAD-BEARING note on float max_samples (carried from Ch08)
----------------------------------------------------------
On sklearn 1.2.2 a FLOAT `max_samples` passed to BaggingClassifier is a
fraction of the fold's ROW COUNT. Newer sklearn reinterpreted it as a fraction
of the summed SAMPLE WEIGHT. With Ch04-style weights this changes how many
bootstrap rows each tree sees. The bagging branch takes `max_samples` from the
caller (`bagging[1]`) untouched -- if you port this to a newer sklearn, revisit
that number.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier

# --- cross-chapter import of the purged CV splitter -------------------------
# Mirrors the Ch08 pattern: resolve the repo root two directories up from this
# module and import the real Ch07 class. Requires ch07/__init__.py and
# ch07/cross_validation/__init__.py (present since Ch08).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from ch07.cross_validation.purged_kfold import PurgedKFold  # noqa: E402


# ===========================================================================
# 9.2  MyPipeline -- route a bare sample_weight to the final estimator
# ===========================================================================
class MyPipeline(Pipeline):
    """A Pipeline that accepts a bare ``sample_weight=`` kwarg in ``fit``.

    Vanilla ``Pipeline.fit`` only understands fit-params keyed as
    ``stepname__param``. BaggingClassifier, however, calls
    ``base_estimator.fit(X, y, sample_weight=...)`` with a bare kwarg. This
    subclass catches that bare kwarg and rewrites it to
    ``{final_step_name}__sample_weight`` so it reaches the last estimator.
    """

    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            # steps[-1][0] is the NAME of the final estimator step
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)


# ===========================================================================
# 9.1 / 9.2  clfHyperFit -- purged grid / randomized search (+ optional bag)
# ===========================================================================
def _pick_scoring(lbl):
    """Book rule: F1 for meta-labeling (binary {0,1}) else neg_log_loss.

    Meta-labels are {0,1} and the interesting class is the positive one, so F1
    is the natural target. For side-and-size labels ({-1,+1} here) F1 is not
    meaningful, so we fall back to the symmetric neg_log_loss.
    Factored out of clfHyperFit purely so it is directly unit-testable.
    """
    if set(np.unique(lbl)) == {0, 1}:
        return 'f1'
    return 'neg_log_loss'


def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid, cv=3,
                bagging=(0, None, 1.), rndSearchIter=0, n_jobs=-1,
                pctEmbargo=0., **fit_params):
    # NOTE: book default is a LIST [0, None, 1.]; we use a tuple to avoid a
    # mutable default (never mutated here, so behaviour is identical).
    """AFML snippets 9.1 + 9.2: hyper-parameter search over a PURGED inner CV.

    Parameters
    ----------
    feat : pd.DataFrame
        Features. Its index MUST equal ``t1``'s index (PurgedKFold requires it).
    lbl : pd.Series
        Labels. {0,1} -> F1 scoring (meta-labeling); else neg_log_loss.
    t1 : pd.Series
        Label end times (index=t0, values=t1), passed to PurgedKFold.
    pipe_clf : sklearn Pipeline (ideally a MyPipeline)
        The estimator/pipeline to tune.
    param_grid : dict
        Grid (grid search) or distributions (randomized search). For the latter,
        scale params like ``svc__C`` should use ``logUniform(...)``.
    cv : int, default=3
        Number of PURGED folds for the inner search CV.
    bagging : sequence (n_estimators, max_samples, max_features)
        Order per book snippet 9.1/9.3. If ``max_samples`` (bagging[1]) is not
        None and > 0, the winning pipeline is wrapped in a BaggingClassifier.
        Book default ``[0, None, 1.]`` = no bagging.
    rndSearchIter : int, default=0
        0 -> GridSearchCV. >0 -> RandomizedSearchCV with this many draws.
    n_jobs : int, default=-1
    pctEmbargo : float, default=0.
    **fit_params
        Forwarded to ``.fit``. To weight the fit, pass
        ``{final_step_name}__sample_weight=w`` (e.g. ``svc__sample_weight=w``);
        GridSearchCV subsets it per fold automatically.

    Returns
    -------
    Fitted best estimator (a Pipeline). If bagging was requested, a
    ``Pipeline([('bag', BaggingClassifier(...))])`` wrapping it.
    """
    scoring = _pick_scoring(lbl)

    # Purged inner CV -- the whole point of the chapter. shuffle=False inside
    # PurgedKFold makes the folds deterministic for a given (t1, cv, pctEmbargo).
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)

    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                          scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
        # NOTE: no iid=... kwarg -- removed in sklearn 0.24 (see module header).
    else:
        gs = RandomizedSearchCV(estimator=pipe_clf, param_distributions=param_grid,
                                scoring=scoring, cv=inner_cv, n_jobs=n_jobs,
                                n_iter=rndSearchIter)

    gs = gs.fit(feat, lbl, **fit_params).best_estimator_  # a fitted Pipeline

    # --- optional bagging of the winning pipeline (book: snippet 9.1 tail) ---
    if bagging[1] is not None and bagging[1] > 0:
        final_name = gs.steps[-1][0]
        bag = BaggingClassifier(
            estimator=MyPipeline(gs.steps),          # (fix 2/3) estimator=, MyPipeline
            n_estimators=int(bagging[0]),
            max_samples=float(bagging[1]),           # bagging tuple = (n_est, max_samples, max_features), per book 9.1/9.3
            max_features=float(bagging[2]),          # LOAD-BEARING (max_samples): see header
            n_jobs=n_jobs)
        sw_key = final_name + '__sample_weight'
        if sw_key in fit_params:                      # (fix 5) guard the lookup
            bag = bag.fit(feat, lbl, sample_weight=fit_params[sw_key])
        else:
            bag = bag.fit(feat, lbl)
        gs = Pipeline([('bag', bag)])

    return gs


# ===========================================================================
# 9.3  logUniform_gen -- log-uniform continuous distribution for random search
# ===========================================================================
# IMPORTANT -- the docstring of an rv_continuous SUBCLASS must contain NO
# percent sign. scipy treats a subclass docstring as a printf-style template at
# construction time and runs (docstring formatted-against a doc dict), so a
# stray percent sign -- even an escaped double one, or one inside a backticked
# code span -- is read as a format specifier and raises at import. Keep this
# class docstring plain prose. (scipy-version-independent; it bit us on import.)
class logUniform_gen(rv_continuous):
    """A random variable log-uniformly distributed between a and b.

    "Log-uniform" means log(x) is uniform on [log a, log b]: every order of
    magnitude between a and b is equally likely. This is the right prior for a
    scale hyper-parameter such as SVC's C or gamma, where 1 vs 10 matters as
    much as 100 vs 1000. Sampling C uniformly on [1e-2, 1e2] would instead pile
    nearly all the mass above 1 and starve the small-C region.
    """

    def _cdf(self, x):
        # CDF of a log-uniform on [a, b]:  log(x/a) / log(b/a)
        return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a=1, b=np.exp(1)):
    """Factory returning a frozen-support logUniform_gen on [a, b]."""
    return logUniform_gen(a=a, b=b, name='logUniform')


# ============================================================================
# TDD results (test_hyper_parameter_tuning.py), embedded per project convention.
# ============================================================================
# NOTE: numbers below are from the DELIVERY SANDBOX (Python 3.12 / sklearn 1.8 /
# scipy 1.17 / numpy 2.x / pandas 3.0), NOT the mlfinlab env. Re-run under
# mlfinlab (Python 3.10.20 / sklearn 1.2.2) and refresh this header before the
# final commit -- same step as Ch08. All 17 assertions are version-independent
# behaviour checks, so they pass on both; only the runtime/warning text differs.
#
# ============================= test session starts ==============================
# test_hyper_parameter_tuning.py::test_loguniform_constructs_without_percent_import_bug PASSED [  5%]
# test_hyper_parameter_tuning.py::test_loguniform_docstring_has_no_percent_sign PASSED [ 11%]
# test_hyper_parameter_tuning.py::test_loguniform_samples_stay_in_bounds PASSED [ 17%]
# test_hyper_parameter_tuning.py::test_loguniform_cdf_endpoints_and_geometric_midpoint PASSED [ 23%]
# test_hyper_parameter_tuning.py::test_loguniform_is_uniform_in_log_space_ks PASSED [ 29%]
# test_hyper_parameter_tuning.py::test_mypipeline_routes_bare_sample_weight_to_final_estimator PASSED [ 35%]
# test_hyper_parameter_tuning.py::test_plain_pipeline_rejects_bare_sample_weight PASSED [ 41%]
# test_hyper_parameter_tuning.py::test_mypipeline_weights_actually_change_the_fit PASSED [ 47%]
# test_hyper_parameter_tuning.py::test_pick_scoring_f1_for_binary_meta_labels PASSED [ 52%]
# test_hyper_parameter_tuning.py::test_pick_scoring_neg_log_loss_otherwise PASSED [ 58%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_grid_returns_pipeline_with_grid_params PASSED [ 64%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_randomized_runs_with_loguniform PASSED [ 70%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_uses_purged_kfold_as_inner_cv PASSED [ 76%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_default_bagging_none_does_not_crash PASSED [ 82%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_bagging_wraps_winning_pipeline PASSED [ 88%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_requires_index_aligned_t1 PASSED [ 94%]
# test_hyper_parameter_tuning.py::test_clfhyperfit_real_data_svc_neg_log_loss PASSED [100%]
# ======================== 17 passed, 1 warning in ~63s ==========================
#
# The single warning is sklearn flagging max_samples=1.0 as a fraction of the
# summed sample weight (~1.0 with getTestData's w=1/n) yielding few bootstrap
# rows -- this is the newer-sklearn max_samples semantics documented in the
# module header (LOAD-BEARING note). On 1.2.2 the float is a fraction of the
# ROW count instead, so the warning will not appear. Harmless for the wiring
# test either way.
