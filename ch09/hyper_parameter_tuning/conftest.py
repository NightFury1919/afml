"""
pytest conftest for Chapter 9.

Caps BLAS / OpenMP thread pools to a single thread BEFORE numpy is imported.
On the Windows conda (MKL) build, hyper-parameter search fits many small
models (a grid/randomized sweep times purged CV folds, each SVC doing its own
internal Platt-scaling CV, plus bagging). Left uncapped, every tiny matrix op
spins up a thread pool across all cores; those pools then thrash against
joblib's process-level parallelism and the suite slows to a crawl. Cap BLAS to
one thread and let parallelism live at the joblib (n_jobs) level instead.

Same mechanism as ch08/feature_importance/conftest.py.

Manual equivalent if you run the demo outside pytest:
    PowerShell:  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
    bash:        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
