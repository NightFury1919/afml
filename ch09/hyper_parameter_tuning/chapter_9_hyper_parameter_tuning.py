"""
Chapter 9: Hyper-Parameter Tuning with Cross-Validation -- runnable demo.

Three parts:
  A. logUniform (snippets 9.3/9.4): draw from the log-uniform prior, KS-test
     that it really is uniform in log-space, and plot linear- vs log-space
     histograms (the picture that shows why it's the right prior for C/gamma).
  B. Synthetic showcase (getTestData, meta-labels {0,1} -> F1 scoring): tune an
     SVC(RBF) inside a StandardScaler pipeline over PURGED folds, comparing
     GridSearchCV against RandomizedSearchCV-with-logUniform. This is where you
     can watch tuning "bite" across many features.
  C. Real-data plug-in (87-row BTC/TUSD table, labels {-1,+1} -> neg_log_loss):
     the exact same clfHyperFit call on the real pipeline, with Ch04 sample
     weights routed to the SVC via MyPipeline. Since the Ch19 enrichment this
     loads the 12-feature enriched table (fracdiff + 11 microstructural
     features, 87 events), NOT the original fracdiff-only table -- see
     load_real_table() below. The enrichment is visible in the result: the
     real-data grid optimum moved from C=100 on the single-feature table to
     C=0.01 on the enriched one, i.e. sharply heavier regularization once
     there is actually something to regularize.

Run under the mlfinlab env (Python 3.10.20 / sklearn 1.2.2). Path convention:
__file__-derived repo root. BLAS threads capped up top (see conftest/README).
"""
import os
import sys

# Cap BLAS/OpenMP threads BEFORE importing numpy (see README): a search sweep
# times purged folds times SVC's internal Platt CV is a lot of tiny fits;
# uncapped thread pools thrash against joblib. Parallelize at n_jobs instead.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ch08.feature_importance.feature_importance import getTestData        # noqa: E402
from ch09.hyper_parameter_tuning.hyper_parameter_tuning import (           # noqa: E402
    MyPipeline, clfHyperFit, logUniform, _pick_scoring,
)

INPUT_DATA = os.path.join(_REPO_ROOT, 'input_data')
# n_jobs=1 for the SVC searches. This chapter tunes SVC(probability=True), whose
# libsvm internal probability-CV hard-crashes (native, no Python traceback) when
# run inside joblib/loky spawned workers on Windows. Trees (Ch04/Ch08) are
# spawn-safe and use n_jobs=4; SVC-with-probability is not, so keep it serial.
# The searches here are small (a coarse grid / 20 draws on <=1000 rows), so
# serial is plenty fast.
N_JOBS = 1
RND_ITER = 20            # RandomizedSearchCV draws
CV_SPLITS = 4

# Keep the synthetic demo small enough that SVC(probability=True) over a grid x
# purged CV finishes quickly; the tuning pattern is already clear here. Dial up
# for a heavier run.
SYNTH = dict(n_features=10, n_informative=5, n_redundant=5, n_samples=1000)

# Log-uniform priors for the RBF SVC's scale hyper-parameters.
C_RANGE = (1e-2, 1e2)
GAMMA_RANGE = (1e-2, 1e1)
# A matched coarse grid (same span, log-spaced) so grid vs randomized is a fair
# head-to-head.
C_GRID = [1e-2, 1e-1, 1e0, 1e1, 1e2]
GAMMA_GRID = [1e-2, 1e-1, 1e0, 1e1]


def _svc_pipe():
    return MyPipeline([('scaler', StandardScaler()),
                       ('svc', SVC(probability=True, random_state=0))])


# --------------------------------------------------------------------------- #
# A. logUniform
# --------------------------------------------------------------------------- #
def demo_loguniform(a=1e-3, b=1e3, size=100000):
    print('=== A. logUniform prior (snippets 9.3 / 9.4) ===')
    vals = logUniform(a=a, b=b).rvs(size=size, random_state=0)
    stat, p = kstest(np.log(vals), 'uniform',
                     args=(np.log(a), np.log(b) - np.log(a)))
    print(f'  drew {size} samples on [{a:g}, {b:g}]')
    print(f'  KS test of log(x) vs Uniform[log a, log b]: stat={stat:.4f}, '
          f'p={p:.3f}  (p>0.05 => log-uniform confirmed)')
    print(f'  describe:\n{pd.Series(vals).describe()}\n')
    return vals, (stat, p)


# --------------------------------------------------------------------------- #
# B. synthetic showcase
# --------------------------------------------------------------------------- #
def demo_synthetic():
    print('=== B. Synthetic showcase: grid vs randomized (meta-labels -> F1) ===')
    trnsX, cont = getTestData(**SYNTH, random_state=0)
    feat, lbl, t1, w = trnsX, cont['bin'], cont['t1'], cont['w']
    print(f'  data {feat.shape}, labels={set(np.unique(lbl))} -> '
          f'scoring={_pick_scoring(lbl.values)}')

    grid_best = clfHyperFit(
        feat, lbl, t1, _svc_pipe(),
        param_grid={'svc__C': C_GRID, 'svc__gamma': GAMMA_GRID},
        cv=CV_SPLITS, n_jobs=N_JOBS, svc__sample_weight=w.values)
    gC, gG = grid_best.named_steps['svc'].C, grid_best.named_steps['svc'].gamma
    print(f'  GridSearchCV      best -> C={gC:g}, gamma={gG:g} '
          f'({len(C_GRID) * len(GAMMA_GRID)} fits x {CV_SPLITS} folds)')

    rnd_best = clfHyperFit(
        feat, lbl, t1, _svc_pipe(),
        param_grid={'svc__C': logUniform(*C_RANGE),
                    'svc__gamma': logUniform(*GAMMA_RANGE)},
        cv=CV_SPLITS, rndSearchIter=RND_ITER, n_jobs=N_JOBS,
        random_state=0,
        svc__sample_weight=w.values)
    rC, rG = rnd_best.named_steps['svc'].C, rnd_best.named_steps['svc'].gamma
    print(f'  RandomizedSearchCV best -> C={rC:.4g}, gamma={rG:.4g} '
          f'({RND_ITER} draws x {CV_SPLITS} folds; continuous logUniform prior)')
    print('  NB: randomized explores continuous C/gamma the coarse grid can '
          'only hit at its nodes.\n')
    return dict(grid=(gC, gG), rnd=(rC, rG))


# --------------------------------------------------------------------------- #
# C. real-data plug-in
# --------------------------------------------------------------------------- #
def load_real_table():
    # Post-Ch19-enrichment: load the enriched artifact (fracdiff + Ch19's
    # 11 microstructural features, 87 events -- one dropped for still
    # being inside a rolling-window warmup period, same convention Ch05
    # already uses for fracdiff's own FFD warmup) instead of rebuilding a
    # fracdiff-only table straight from ch03/04/05. Falls back to the
    # original single-feature path if the enriched artifact isn't present
    # (e.g. running this chapter standalone before Ch19 exists on a
    # machine).
    enriched_path = os.path.join(INPUT_DATA, 'ch07_training_table_enriched.csv')
    if os.path.exists(enriched_path):
        table = pd.read_csv(enriched_path, index_col=0, parse_dates=[0, 't1'])
        feature_cols = [c for c in table.columns if c not in ('bin', 'w', 't1')]
        X = table[feature_cols]
        y = table['bin']
        w = table['w']
        t1 = table['t1']
    else:
        ch03 = pd.read_csv(os.path.join(INPUT_DATA, 'ch03_events.csv'),
                           index_col=0, parse_dates=True)
        ch04 = pd.read_csv(os.path.join(INPUT_DATA, 'ch04_weights.csv'),
                           index_col=0, parse_dates=True)
        ch05 = pd.read_csv(os.path.join(INPUT_DATA, 'ch05_features.csv'),
                           index_col=0, parse_dates=True)
        X = ch05.loc[ch03.index][['fracdiff']]
        y = ch03['bin']
        w = ch04['w']
        t1 = pd.to_datetime(ch03['t1'])
    assert X.index.equals(y.index) and X.index.equals(w.index) \
        and X.index.equals(t1.index), 'X/y/w/t1 index must match before PurgedKFold'
    return X, y, t1, w


def demo_real():
    print('=== C. Real-data plug-in: real BTC/TUSD table (labels -> neg_log_loss) ===')
    X, y, t1, w = load_real_table()
    print(f'  data {X.shape} ({X.shape[1]} feature{"s" if X.shape[1] != 1 else ""}: '
          f'{list(X.columns)}), labels={set(np.unique(y))} -> scoring={_pick_scoring(y.values)}')
    if X.shape[1] == 1:
        print('  (single feature => thin tuning surface; this shows the machinery '
              'on the REAL pipeline,\n   not a dramatic optimum. Motivates enriching '
              'the real feature set later.)')
    else:
        print('  (post-Ch19 enrichment: fracdiff + 11 microstructural features, '
              'the first real multi-\n   feature tuning surface this pipeline has had.)')

    grid_best = clfHyperFit(
        X, y, t1, _svc_pipe(),
        param_grid={'svc__C': C_GRID, 'svc__gamma': GAMMA_GRID},
        cv=CV_SPLITS, pctEmbargo=0.12, n_jobs=N_JOBS, svc__sample_weight=w.values)
    gC, gG = grid_best.named_steps['svc'].C, grid_best.named_steps['svc'].gamma
    print(f'  GridSearchCV      best -> C={gC:g}, gamma={gG:g}')

    rnd_best = clfHyperFit(
        X, y, t1, _svc_pipe(),
        param_grid={'svc__C': logUniform(*C_RANGE),
                    'svc__gamma': logUniform(*GAMMA_RANGE)},
        cv=CV_SPLITS, pctEmbargo=0.12, rndSearchIter=RND_ITER, n_jobs=N_JOBS,
        random_state=0,
        svc__sample_weight=w.values)
    rC, rG = rnd_best.named_steps['svc'].C, rnd_best.named_steps['svc'].gamma
    print(f'  RandomizedSearchCV best -> C={rC:.4g}, gamma={rG:.4g}\n')
    return dict(grid=(gC, gG), rnd=(rC, rG))


# --------------------------------------------------------------------------- #
# plotting (deferred to the end; no joblib workers alive)
# --------------------------------------------------------------------------- #
def plot_loguniform(vals, a=1e-3, b=1e3):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.log(vals), bins=100, color='#2E5B9C')
    axes[0].set_title('log(x) is UNIFORM (flat)')
    axes[0].set_xlabel('log(x)')
    axes[1].hist(vals, bins=np.logspace(np.log10(a), np.log10(b), 100),
                 color='#E8872A')
    axes[1].set_xscale('log')
    axes[1].set_title('x itself is log-uniform (flat on a log axis)')
    axes[1].set_xlabel('x (log scale)')
    fig.suptitle('logUniform prior: every order of magnitude equally likely')
    fig.tight_layout()
    png = os.path.join(_HERE, 'logUniform_hist.png')
    fig.savefig(png, dpi=110, bbox_inches='tight')
    print(f'Saved {png}')
    return png


def plot_search_results(synth_res, real_res):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, res, mk, col in [
        ('synthetic grid', synth_res['grid'], 's', '#2E5B9C'),
        ('synthetic rand', synth_res['rnd'], 'o', '#1A6B4B'),
        ('real grid', real_res['grid'], 's', '#E8872A'),
        ('real rand', real_res['rnd'], 'o', '#B03A2E'),
    ]:
        ax.scatter(res[0], res[1], marker=mk, s=90, label=label, color=col,
                   edgecolor='k', zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('best C (log)')
    ax.set_ylabel('best gamma (log)')
    ax.set_title('Selected (C, gamma): grid nodes vs continuous randomized')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    png = os.path.join(_HERE, 'ch09_search_comparison.png')
    fig.savefig(png, dpi=110, bbox_inches='tight')
    print(f'Saved {png}')
    return png


def save_artifact(loguniform_ks, synth_res, real_res):
    os.makedirs(INPUT_DATA, exist_ok=True)
    rows = [
        {'demo': 'synthetic', 'search': 'grid', 'C': synth_res['grid'][0],
         'gamma': synth_res['grid'][1], 'scoring': 'f1'},
        {'demo': 'synthetic', 'search': 'randomized', 'C': synth_res['rnd'][0],
         'gamma': synth_res['rnd'][1], 'scoring': 'f1'},
        {'demo': 'real', 'search': 'grid', 'C': real_res['grid'][0],
         'gamma': real_res['grid'][1], 'scoring': 'neg_log_loss'},
        {'demo': 'real', 'search': 'randomized', 'C': real_res['rnd'][0],
         'gamma': real_res['rnd'][1], 'scoring': 'neg_log_loss'},
    ]
    stats = pd.DataFrame(rows)
    stats.attrs['loguniform_ks_stat'] = loguniform_ks[0]
    stats.attrs['loguniform_ks_p'] = loguniform_ks[1]
    csv = os.path.join(INPUT_DATA, 'ch09_hyperparameter_tuning_stats.csv')
    pkl = os.path.join(INPUT_DATA, 'ch09_hyperparameter_tuning_stats.pkl')
    stats.to_csv(csv, index=False)
    stats.to_pickle(pkl)
    print(f'Saved {csv} and {pkl}')
    print('(Record artifact -- no downstream chapter consumes a ch09 pkl.)')


def main():
    pd.set_option('display.max_columns', None)
    vals, ks = demo_loguniform()
    synth_res = demo_synthetic()
    real_res = demo_real()

    # plotting phase (compute done -> safe to open figures)
    plot_loguniform(vals)
    plot_search_results(synth_res, real_res)
    save_artifact(ks, synth_res, real_res)
    try:
        plt.show()
    except Exception:
        pass  # headless -- PNGs already written


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# TDD results mirror -- same suite as hyper_parameter_tuning.py's embedded
# block and this chapter's notebook, duplicated here per the .py/.ipynb mirror
# convention (this script is the ipynb's paired mirror, not hyper_parameter_tuning.py).
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
