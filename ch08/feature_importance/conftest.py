"""
pytest configuration for the Chapter 8 feature-importance suite.

Caps BLAS / OpenMP thread pools to 1 BEFORE numpy is imported.

Why: on Windows conda (MKL) builds, fitting many small trees makes every tiny
linear-algebra call spawn a thread pool across all cores. On a multi-core
machine those pools thrash against each other (oversubscription), which can make
this suite *10-50x slower* than single-threaded -- the SFI tests, which fit one
CV per feature, suffer most. Capping BLAS to one thread and letting parallelism
happen only at the coarse joblib level (SFI's n_jobs, unaffected here) removes
the contention.

conftest.py is imported by pytest before the test modules, so on a fresh process
these variables are set before numpy/sklearn read them. If your environment has
already imported numpy by the time pytest starts, set them in the shell instead:

    PowerShell:  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
    bash:        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
