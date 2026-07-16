import os

# BLAS thread cap -- mirrors Ch08/Ch09/Ch12/Ch13's conftest.py. On the 6-core
# MKL machine, thread oversubscription across numpy/scipy's BLAS calls causes
# severe slowdowns under pytest. Ch19 leans on numpy (OLS via lstsq, rolling
# regressions) more than most chapters, so this one actually matters.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
