"""
Caps BLAS/MKL/OpenMP thread pools before any test runs. Without this,
oversubscribed threads on the 6-core mlfinlab machine cause severe
slowdowns during SVC fits (mirrors Ch08/09/10/12's conftest.py).
"""
import os

for _var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'
