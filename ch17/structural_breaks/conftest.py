import os

# BLAS thread cap -- mirrors Ch08/Ch09/Ch12/Ch13/Ch19's conftest.py. On the
# 6-core MKL machine, thread oversubscription across numpy/scipy's BLAS calls
# causes severe slowdowns under pytest. sadf.py's get_bsadf/get_sadf run many
# small OLS fits (np.linalg.inv) in tight loops, so this matters here too.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
