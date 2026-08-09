import os

# BLAS thread cap -- mirrors Ch08/09/12/13/17/19's conftest.py. Also
# specifically relevant here: this chapter's own tests spawn real
# multiprocessing.Pool worker processes (test_barrier_touch.py,
# test_multiprocess.py-style patterns); letting each worker's numpy/BLAS
# ALSO try to multithread on top of that would oversubscribe this
# 6-core machine badly.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
