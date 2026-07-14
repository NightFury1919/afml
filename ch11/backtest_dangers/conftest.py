"""
pytest configuration for the Chapter 11 CSCV/PBO suite.

Caps BLAS / OpenMP thread pools to 1 BEFORE numpy is imported.

Every numpy linear-algebra call otherwise spins up a thread pool across all
cores; those pools thrash against each other when pytest runs many small
test cases back to back. Cap BLAS to one thread and let parallelism live at
the joblib (n_jobs) level instead.

Mirrors ch08/feature_importance/conftest.py and
ch09/hyper_parameter_tuning/conftest.py VERBATIM in intent (thread cap = 1).

Diffed against the real ch08/ch09 files (not reconstructed). Ch10's copy was
a reconstruction that had drifted to '4'; it has been corrected to match.
VECLIB_MAXIMUM_THREADS is included as ch08 does (macOS Accelerate; a harmless
no-op on Windows).

conftest.py is imported by pytest before the test modules, so on a fresh
process these are set before numpy/sklearn read them. If numpy is already
imported by the time pytest starts, set them in the shell instead:
    PowerShell:  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
    bash:        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
