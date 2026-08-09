"""
TDD suite for barrier_touch.py (AFML Snippets 20.3/20.4 one-touch double-
barrier example, plus an engine-based version built on mp_job_list).

Correctness is pinned with a small, hand-computed 2-column example (see
comment on HAND_TRACED_R below) rather than the book's own 1000x10000
scale, which is a timing DEMO, not a unit test fixture. The real 10,000-
path/1000-step timing comparison is exercised separately in the driver
script/notebook, on the real machine, per the real-data-first policy's
"run it for real, report genuine output" principle extended to this
chapter's actual subject matter (wall-clock behavior, not financial data).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from barrier_touch import (
    barrier_touch, main0, main1,
    barrier_touch_engine, run_barrier_touch_engine,
)


# LOAD-BEARING: hand-computed via p = log((1+r).cumprod(axis=0)), width=0.1.
# Column 0 (r=0.05 every step): p = [0.04879, 0.09758, 0.14637] ->
#   crosses +0.1 at row index 2 (first row where p >= 0.1).
# Column 1 (r=-0.01 every step): p = [-0.01005, -0.02010, -0.03015] ->
#   never reaches -0.1 within 3 rows -> absent from the output dict.
# Verified independently with a standalone numpy computation before writing
# these assertions, not derived from running barrier_touch itself.
HAND_TRACED_R = np.array([
    [0.05, -0.01],
    [0.05, -0.01],
    [0.05, -0.01],
])
HAND_TRACED_WIDTH = 0.1
HAND_TRACED_EXPECTED = {0: 2}


class TestBarrierTouch:
    def test_hand_traced_touch_and_no_touch(self):
        assert barrier_touch(HAND_TRACED_R, width=HAND_TRACED_WIDTH) == HAND_TRACED_EXPECTED

    def test_touch_at_first_row_that_crosses(self):
        # A single-column path that crosses immediately at row 0.
        r = np.array([[0.5], [0.5]])
        assert barrier_touch(r, width=0.1) == {0: 0}

    def test_negative_barrier_touch(self):
        r = np.array([[-0.2], [-0.2]])
        assert barrier_touch(r, width=0.1) == {0: 0}

    def test_no_columns_touch_gives_empty_dict(self):
        r = np.array([[0.001, -0.001], [0.001, -0.001]])
        assert barrier_touch(r, width=0.5) == {}

    def test_wider_barrier_delays_or_prevents_touch(self):
        # Same path, wider barrier -> either touches later or not at all.
        r = np.tile(0.05, (5, 1))
        narrow = barrier_touch(r, width=0.1)
        wide = barrier_touch(r, width=10.0)
        assert 0 in narrow
        assert 0 not in wide  # 5 steps of +0.05 log-return never reaches 10.0


class TestMain0Main1Consistency:
    # main0 (sequential) and main1 (raw multiprocessing.Pool, book Snippet
    # 20.4) are NOT expected to return identically-shaped output -- main1's
    # local per-chunk indexing is a known, book-faithful property (see
    # barrier_touch's own docstring). What IS checked here: both run
    # without error on a small case, and main1's chunk count matches what
    # was requested.
    def test_main0_runs_and_returns_dict(self):
        out = main0(num_paths=20, path_len=50, seed=7)
        assert isinstance(out, dict)
        assert all(isinstance(v, (int, np.integer)) for v in out.values())

    def test_main1_runs_and_returns_one_result_per_chunk(self):
        out = main1(num_paths=20, path_len=50, num_threads=4, seed=7)
        assert len(out) == 4  # one dict per column-chunk
        assert all(isinstance(chunk_result, dict) for chunk_result in out)

    def test_main1_num_threads_capped_at_num_paths(self):
        # 3 paths requested with 10 threads -> capped to 3 chunks, not 10.
        out = main1(num_paths=3, path_len=20, num_threads=10, seed=7)
        assert len(out) == 3


class TestBarrierTouchEngine:
    def test_preserves_global_column_index(self):
        # This is the property main1's raw local-index dicts lack: embed
        # the hand-traced 2-column matrix at columns [5,9] of a larger
        # (11-column) matrix, simulating a molecule drawn from the middle
        # of a bigger job, and confirm the returned keys are the GLOBAL
        # indices (5, 9), not local 0/1.
        big_r = np.zeros((3, 11))
        big_r[:, 5] = HAND_TRACED_R[:, 0]
        big_r[:, 9] = HAND_TRACED_R[:, 1]
        out = barrier_touch_engine([5, 9], big_r, width=HAND_TRACED_WIDTH)
        assert out == {5: 2}  # global column 5 touches at row 2;
        # global column 9 never touches -> correctly absent.


class TestRunBarrierTouchEngine:
    def test_single_threaded_matches_direct_call(self):
        out = run_barrier_touch_engine(HAND_TRACED_R, width=HAND_TRACED_WIDTH, num_threads=1)
        assert out == HAND_TRACED_EXPECTED

    def test_multiprocessing_matches_single_threaded(self):
        # Bigger matrix so there's more than one column per thread to chunk.
        rng = np.random.default_rng(2026)
        r = rng.normal(0, .01, size=(200, 12))
        out_mp = run_barrier_touch_engine(r, width=.05, num_threads=3)
        out_seq = run_barrier_touch_engine(r, width=.05, num_threads=1)
        assert out_mp == out_seq

    def test_result_keyed_by_true_global_column_regardless_of_chunking(self):
        # Same 12-column matrix, chunked differently (2 threads vs 4 threads)
        # -- results must be identical regardless of how the columns were
        # grouped into molecules, proving global-index correctness isn't an
        # accident of one particular chunking.
        rng = np.random.default_rng(99)
        r = rng.normal(0, .01, size=(150, 12))
        out_2 = run_barrier_touch_engine(r, width=.05, num_threads=2)
        out_4 = run_barrier_touch_engine(r, width=.05, num_threads=4)
        assert out_2 == out_4
        assert set(out_2.keys()) <= set(range(12))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
