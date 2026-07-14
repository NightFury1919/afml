"""
pytest conftest for Chapter 10.

Caps BLAS / OpenMP thread pools to a single thread BEFORE numpy is imported.
On the Windows conda (MKL) build, the bet-sizing suite fits SVCs with
probability=True (each running its own internal Platt-scaling CV) across
purged CV folds, and dispatches avgActiveSignals through mp_pandas_obj.
Left uncapped, every tiny matrix op spins up a thread pool across all cores;
those pools then thrash against joblib's process-level parallelism and the
suite slows to a crawl. Cap BLAS to one thread and let parallelism live at
the joblib (n_jobs / numThreads) level instead.

Same mechanism as ch08/feature_importance/conftest.py and
ch09/hyper_parameter_tuning/conftest.py.

Manual equivalent if you run the demo outside pytest:
    PowerShell:  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
    bash:        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
