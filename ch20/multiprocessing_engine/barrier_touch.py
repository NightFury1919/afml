# barrier_touch.py -- AFML Section 20.3, Snippets 20.3 (single-thread) and
# 20.4 (multiprocessing) one-touch double-barrier timing example, plus an
# engine-based third version built on this chapter's own mp_job_list
# (utils/multiprocess.py) to tie Section 20.3's motivating example back to
# Section 20.5's formalized engine.
#
# Book-fidelity translation notes (both snippets are printed in Python 2):
#   - `xrange` -> `range`.
#   - `print min(timeit.Timer(...).repeat(5,10))` (print statement) ->
#     `print(min(timeit.Timer(...).repeat(5,10)))` (function call).
#   - Snippet 20.3's `t,p={},np.log(...)` -- a single-line tuple assignment
#     that DOES work as intended in both Python 2 and 3 here (unlike the
#     real Ch5 Snippet 5.3 bug already on file in this project's memory --
#     that one failed because it re-assigned a name mid-expression using
#     its own old value; this one is a plain `t, p = {}, expr`, which is
#     unambiguous). Split onto two lines below purely for readability, not
#     because of a semantics issue.
#   - Snippet 20.4's `main1` hardcodes numThreads=24 to match a specific
#     24-core machine from the book -- made a parameter here instead
#     (default matches this project's own established sweet spot of 4
#     threads on a 6-core machine, not 24; see CLAUDE.md "Working style").

import os
import sys
import time

import numpy as np
import multiprocessing as mp

# path/portability convention: derive repo root from this file's location
# (this file lives at <root>/ch20/multiprocessing_engine/barrier_touch.py)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.multiprocess import mp_job_list  # noqa: E402


def barrier_touch(r, width=.5):
    # AFML Snippet 20.3's barrierTouch -- find the row index of the
    # earliest cumulative-log-return barrier touch, per column of r.
    # Returns a dict keyed by r's own LOCAL column index (0..r.shape[1]-1)
    # -- if r is a column-CHUNK of a larger matrix (as in main1 below),
    # these keys do NOT correspond to the original global column indices.
    # This is a faithful, deliberate port of the book's own snippet, which
    # has the same property (main1 never reassembles a globally-indexed
    # result -- it only measures wall-clock time). See barrier_touch_engine
    # below for a version that preserves global column identity.
    t = {}
    p = np.log((1 + r).cumprod(axis=0))
    for j in range(r.shape[1]):          # go through columns
        for i in range(r.shape[0]):      # go through rows
            if p[i, j] >= width or p[i, j] <= -width:
                t[j] = i
                break
    return t


def main0(num_paths=10000, path_len=1000, width=.5, seed=None):
    # AFML Snippet 20.3's main0 -- sequential (single-thread) implementation.
    rng = np.random.default_rng(seed)
    r = rng.normal(0, .01, size=(path_len, num_paths))
    return barrier_touch(r, width=width)


def main1(num_paths=10000, path_len=1000, width=.5, num_threads=4, seed=None):
    # AFML Snippet 20.4's main1 -- naive raw multiprocessing.Pool version:
    # splits r's COLUMNS into num_threads chunks and runs barrier_touch on
    # each chunk in a separate process. Matches the book's own structure
    # (pool.imap_unordered over pre-sliced column chunks) rather than
    # routing through this chapter's own mp_pandas_obj/mp_job_list --
    # that's deliberate: this snippet's whole teaching point is "here's
    # what raw multiprocessing.Pool code looks like," motivating why
    # Section 20.5 then builds a reusable engine instead of hand-rolling
    # this every time. See run_barrier_touch_engine below for the engine-
    # based version.
    #
    # Windows note: mp.Pool here relies on barrier_touch being a real
    # module-level function (it is) so it's picklable for spawn -- an
    # inline/lambda target would not work on this machine.
    rng = np.random.default_rng(seed)
    r = rng.normal(0, .01, size=(path_len, num_paths))
    num_threads = min(num_threads, r.shape[1])
    parts = np.linspace(0, r.shape[1], num_threads + 1)
    parts = np.ceil(parts).astype(int)
    jobs = [r[:, parts[i - 1]:parts[i]] for i in range(1, len(parts))]

    pool = mp.Pool(processes=num_threads)
    outputs = pool.imap_unordered(barrier_touch, jobs)
    out = list(outputs)
    pool.close()
    pool.join()
    return out


def barrier_touch_engine(molecule, r, width=.5):
    # Worker function for use with utils.multiprocess.mp_job_list.
    # Unlike raw barrier_touch, this preserves GLOBAL column identity:
    # `molecule` is a list of column indices into the FULL r array, and the
    # returned dict is keyed by those same global indices -- fixing the
    # local-index-collision property of the book's own main1 (see
    # barrier_touch's docstring above), so results from different chunks
    # can be safely merged with dict.update.
    sub = r[:, molecule]
    local = barrier_touch(sub, width=width)
    return {molecule[j]: local[j] for j in local}


def run_barrier_touch_engine(r, width=.5, num_threads=1, mp_batches=1):
    # Ties Section 20.3's motivating example back to Section 20.5's
    # formalized engine: uses THIS chapter's own mp_job_list (built earlier
    # this session in utils/multiprocess.py) with a dict.update reducer to
    # merge per-molecule column results into a single correctly-indexed
    # dict, instead of hand-rolling mp.Pool as main1 does.
    return mp_job_list(
        barrier_touch_engine, ('molecule', list(range(r.shape[1]))),
        num_threads=num_threads, mp_batches=mp_batches,
        redux=dict.update, redux_in_place=True,
        r=r, width=width,
    )


def time_single_vs_multi(num_paths=10000, path_len=1000, width=.5,
                          num_threads=4, repeats=3, seed=12345):
    # Real (not estimated) wall-clock comparison, matching Snippet 20.3/20.4's
    # own timeit-based benchmarking intent -- run on the real machine when
    # producing driver-script/notebook output, not fabricated.
    def _single():
        main0(num_paths=num_paths, path_len=path_len, width=width, seed=seed)

    def _multi():
        main1(num_paths=num_paths, path_len=path_len, width=width,
              num_threads=num_threads, seed=seed)

    single_times = []
    for _ in range(repeats):
        t0 = time.time()
        _single()
        single_times.append(time.time() - t0)

    multi_times = []
    for _ in range(repeats):
        t0 = time.time()
        _multi()
        multi_times.append(time.time() - t0)

    return {
        'single_thread_best_sec': min(single_times),
        'multiprocessing_best_sec': min(multi_times),
        'num_threads': num_threads,
        'speedup': min(single_times) / min(multi_times),
    }
