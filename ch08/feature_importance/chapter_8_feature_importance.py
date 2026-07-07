"""
Chapter 8: Feature Importance -- runnable demonstration on the book's
synthetic dataset (getTestData). Produces the MDI / MDA / SFI importance
tables and plots, the I/R/N summary, and the orthogonal-features + weighted
Kendall's tau corroboration (snippets 8.2-8.10).

Run this under the mlfinlab env (Python 3.10.20 / sklearn 1.2.2). On newer
sklearn the float max_samples semantics differ (see the LOAD-BEARING note in
feature_importance.py) and every importance collapses to noise.

Path convention: __file__-derived repo root (this repo's .py convention).
"""
import os
import sys

# Cap BLAS/OpenMP threads BEFORE importing numpy. On Windows conda (MKL) builds,
# fitting many small trees otherwise oversubscribes threads across cores and
# thrashes -- capping to 1 and parallelizing at the joblib level (n_jobs below)
# is dramatically faster. See README.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ch08.feature_importance.feature_importance import (  # noqa: E402
    getTestData, featImportance, featImpMDI, orthoFeats, featPCA_rank_corr,
    plotFeatImportance,
)
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from sklearn.ensemble import BaggingClassifier  # noqa: E402

# --- canonical config (book defaults). Dial N_ESTIMATORS down for a faster
#     exploratory run; the I/R/N pattern is already clear by ~200 trees. ---
N_FEATURES = 40
N_INFORMATIVE = 10
N_REDUNDANT = 10
N_SAMPLES = 10000
N_JOBS = 4             # this repo's convention (6 cores available)
SCORING = 'accuracy'
PCT_EMBARGO = 0.
# Per-method compute budget. SFI fits one bag PER FEATURE PER FOLD (e.g. 40x10),
# so it is by far the slowest -- but each fit is a single-column model that
# stabilizes with far fewer trees/folds. Give it a lighter budget. Book
# canonical is 1000 trees / 10 folds everywhere; the I/R/N pattern is clear by
# ~200 trees, and single-feature SFI by ~100.
METHOD_CFG = {
    'MDI': dict(n_estimators=250, cv=10),
    'MDA': dict(n_estimators=250, cv=10),
    'SFI': dict(n_estimators=100, cv=5),
}
INPUT_DATA = os.path.join(_REPO_ROOT, 'input_data')


def type_mean(imp):
    """Mean per-feature importance within each feature type (I / R / N).

    Deliberately the MEAN, not a summed share of the total. There are twice as
    many N features (20) as I or R (10 each), so summing shares makes noise look
    big purely by count. The within-type mean is the honest comparison -- and
    for SFI, whose 'mean' column is single-feature ACCURACY (a level near 0.5,
    not a decrement centred on 0), the summed share is actively misleading while
    the mean is not.
    """
    d = imp[['mean']].copy()
    d['type'] = [i[0] for i in d.index]
    return d.groupby('type')['mean'].mean().reindex(['I', 'R', 'N']).round(4)


def main():
    print('Building synthetic dataset (answer key: I=informative, '
          'R=redundant, N=noise)...')
    trnsX, cont = getTestData(N_FEATURES, N_INFORMATIVE, N_REDUNDANT, N_SAMPLES)
    print(f'  trnsX {trnsX.shape}, bin balance '
          f'{cont["bin"].value_counts().to_dict()}\n')

    results = {}
    for method in ['MDI', 'MDA', 'SFI']:
        cfg = METHOD_CFG[method]
        note = (f"  (one bag per feature per fold, {N_JOBS}-way across features"
                f" -- lighter budget: {cfg['n_estimators']} trees, "
                f"{cfg['cv']} folds)") if method == 'SFI' else ''
        print(f"Running {method}...{note}")
        imp, oob, oos = featImportance(
            trnsX=trnsX, cont=cont, n_estimators=cfg['n_estimators'],
            cv=cfg['cv'], scoring=SCORING, pctEmbargo=PCT_EMBARGO,
            method=method, n_jobs=N_JOBS)
        results[method] = (imp, oob, oos)
        print(f'  oob={oob:.4f}  oos={oos:.4f}')
        print(f'  mean importance by type: {type_mean(imp).to_dict()}')
        if method == 'SFI':
            print('  NB: SFI scores each feature ALONE. make_classification\'s '
                  'informative features are only JOINTLY\n      predictive, so '
                  'single features (I, R, N alike) all land near chance here -- '
                  'this flatness\n      is SFI\'s documented blind spot to joint '
                  'effects, not an error.')
        print()

    # NB: all plotting is deferred to the end (see below). Drawing figures
    # inside this loop opens interactive Tk windows that collide with SFI's
    # joblib worker threads -> harmless-but-noisy "main thread is not in main
    # loop" tracebacks. Compute first, plot once everything parallel is done.

    # --- summary table ---
    summary = pd.DataFrame(
        {m: type_mean(results[m][0]) for m in results}
    ).T
    summary['oob'] = {m: results[m][1] for m in results}
    summary['oos'] = {m: results[m][2] for m in results}
    print('=== mean per-feature importance by type (+ oob/oos) ===')
    print(summary, '\n')
    print('Read WITHIN each method (the scales differ):')
    print('  MDI: noise ~ half of signal (in-sample bias, never zero).')
    print('  MDA: noise ~ 0 (its strength); informative on top.')
    print('  SFI: ~flat across I/R/N -- blind to the joint-only signal here.\n')

    # --- orthogonal features + weighted Kendall's tau (8.5 / 8.6) ---
    # Book-faithful corroboration: transform features to orthogonal PCs, run the
    # same cheap in-sample MDI ON THE PCs, then correlate each PC's importance
    # with its eigenvalue RANK (PC_1 = highest variance). A positive tau means
    # the model finds the high-variance PC directions important -> the importance
    # reflects real structure rather than overfitting to noise. (Correlating a
    # back-mapped PCA score against original-feature importance, as an earlier
    # draft did, is not the book's construction and reads backwards -- avoid it.)
    dfP = orthoFeats(trnsX, varThres=.95)
    dfP = pd.DataFrame(dfP, index=trnsX.index,
                       columns=['PC_%d' % (i + 1) for i in range(dfP.shape[1])])
    print(f'orthoFeats: {dfP.shape[1]} of {trnsX.shape[1]} PCs kept for 95% '
          f'variance.')

    pc_clf = DecisionTreeClassifier(criterion='entropy', max_features=1,
                                    class_weight='balanced')
    pc_bag = BaggingClassifier(
        estimator=pc_clf, n_estimators=METHOD_CFG['MDI']['n_estimators'],
        max_features=1., max_samples=1., oob_score=True, n_jobs=N_JOBS)
    pc_fit = pc_bag.fit(dfP, cont['bin'], sample_weight=cont['w'].values)
    pc_imp = featImpMDI(pc_fit, featNames=dfP.columns)['mean']
    pc_rank = np.arange(1, len(pc_imp) + 1)   # PCs already eigenvalue-ordered
    tau = featPCA_rank_corr(pc_imp.values, pc_rank)
    print(f'weighted Kendall tau (PC importance vs eigenvalue rank) = {tau:.4f}')
    print('  (positive => high-variance PCs are the important ones = real '
          'structure)')

    # --- persist the summary (csv + pkl, per shared-artifact convention) ---
    os.makedirs(INPUT_DATA, exist_ok=True)
    summary.to_csv(os.path.join(INPUT_DATA, 'ch08_feature_importance_stats.csv'))
    summary.to_pickle(os.path.join(INPUT_DATA, 'ch08_feature_importance_stats.pkl'))
    print(f'\nSaved summary -> {INPUT_DATA}\\ch08_feature_importance_stats.'
          f'{{csv,pkl}}')
    print('(No downstream chapter consumes a ch08 pkl -- this is a record '
          'artifact, not a pipeline input.)')

    # --- plotting phase (after all parallel work is done) ---------------------
    # Save a PNG per method (durable, backend-agnostic) and then show them all
    # at once. Running this only now -- with no joblib workers alive -- avoids
    # the Tk/worker-thread collision.
    for method in ['MDI', 'MDA', 'SFI']:
        imp, oob, oos = results[method]
        png = os.path.join(_HERE, f'featImportance_{method}.png')
        ax = plotFeatImportance(imp, oob=oob, oos=oos, method=method,
                                tag='ch08', simNum=method, savePath=png)
        ax.figure.suptitle(f'{method} feature importance', y=1.02)
    print(f'Saved 3 importance charts -> {_HERE}\\featImportance_*.png')
    try:
        plt.show()
    except Exception:
        pass  # headless / no display -- PNGs are already written


if __name__ == '__main__':
    main()
