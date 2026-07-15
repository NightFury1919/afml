import os

# BLAS thread cap -- mirrors Ch08/Ch09/Ch12's conftest.py. On the 6-core MKL
# machine, thread oversubscription across numpy/scipy's BLAS calls causes
# severe slowdowns under pytest's own parallelism. Ch13 doesn't lean on BLAS
# heavily (no sklearn here), but capping costs nothing and keeps the
# convention uniform across chapters.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
