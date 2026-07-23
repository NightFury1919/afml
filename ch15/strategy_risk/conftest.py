import os

# BLAS thread cap -- mirrors Ch08/Ch09/Ch12/Ch13/Ch19's conftest.py. Ch15's
# own math is scalar/closed-form (no BLAS calls at all), but probFailure's
# mixGaussians draws and this module's Monte Carlo verification (Snippet
# 15.1, 1e6 binomial draws) touch numpy's RNG and reduction internals, so
# the cap is kept for consistency with the rest of the repo's test suite
# running back-to-back on the 6-core Windows/MKL machine.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
