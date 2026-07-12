"""
conftest.py -- BLAS thread cap for the bet_sizing test suite.

Mirrors the Ch08/Ch09 convention: cap BLAS/MKL threads so pytest doesn't
oversubscribe cores when numpy/scipy operations run underneath multiple
test cases, and to avoid resource contention with mp_pandas_obj's own
multiprocessing when numThreads > 1 is exercised in real-data runs.

NOTE: I don't have the actual Ch08/Ch09 conftest.py content to diff
against (only the handoff doc's description: "BLAS thread cap, mirrors
Ch08") -- this is a reasonable reconstruction of that pattern, using this
repo's established sweet spot of 4 threads. If your real conftest.py
differs, let me know and I'll match it exactly rather than leave two
versions drifting.

Must run before numpy/scipy/mkl get imported anywhere in the test
session, hence setting the env vars at collection time in this file.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '4')
