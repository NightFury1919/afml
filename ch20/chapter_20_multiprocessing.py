"""
Chapter 20: Multiprocessing and Vectorization
===============================================

Unlike most chapters in this project, Chapter 20 isn't a financial-ML
formula chapter -- it's about the multiprocessing INFRASTRUCTURE this
project has been quietly relying on since Chapter 3 (mp_pandas_obj is
called in the triple-barrier labeling, sample-weight, feature-importance,
and bet-sizing chapters without ever being explained). This script
explains it, and demonstrates the new pieces added this session
(nested_parts, process_jobs_redux, mp_job_list) that weren't needed by
earlier chapters but complete the book's own Chapter 20 material.

  Part A -- Vectorization (20.1/20.2): un-vectorized vs. vectorized
            Cartesian product, timed, plus a dimension-count the
            un-vectorized (hardcoded 3-loop) version structurally cannot
            reach at all.
  Part B -- Atoms and molecules (20.4): lin_parts (equal-atoms-per-
            molecule) vs. nested_parts (triangular-workload-aware)
            partitioning, shown on a real triangular-shaped task size
            (Ch17's own SADF computation is exactly this shape -- N
            nested-loop iterations, one per lag -- used here as the
            worked example atom count).
  Part C -- REAL single-thread vs. multiprocessing timing benchmark
            (20.3/20.4): the one-touch double-barrier problem, timed on
            this machine, not estimated. Book's own scale (1000-step,
            10,000-path) is faithfully used, but with 1 real timing pass
            rather than the book's repeat(5,10)=50 runs, which would take
            far too long to be a reasonable demo re-run.
  Part D -- The formalized engine closing the loop (20.5/20.6): the SAME
            barrier-touch problem from Part C, this time run through this
            chapter's own mp_job_list with a dict.update reducer
            (utils/multiprocess.py, extended this session) instead of
            hand-rolled mp.Pool code -- and verified to give identical,
            correctly-globally-indexed results regardless of thread count.

Windows multiprocessing note (relevant on this machine): mp.Pool uses
'spawn' on Windows, not 'fork'. All worker functions used here
(barrier_touch, barrier_touch_engine, square-style callbacks in
utils/test_multiprocess.py) are real module-level functions for exactly
this reason -- inline/lambda targets are not spawn-picklable and would
silently hang or error on Windows even though they might appear to work
in a fork-based sandbox.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from multiprocessing_engine.vectorization import (   # noqa: E402
    cartesian_product_unvectorized, cartesian_product_vectorized,
)
from multiprocessing_engine.barrier_touch import (    # noqa: E402
    main0, main1, run_barrier_touch_engine, time_single_vs_multi,
)

sys.path.insert(0, os.path.abspath(os.path.join(ROOT, '..')))
from utils.multiprocess import lin_parts, nested_parts  # noqa: E402


def part_a_vectorization():
    print('=' * 70)
    print('PART A: Vectorization (Section 20.2)')
    print('=' * 70)

    dict0 = {'a': ['1', '2'], 'b': ['+', '*'], 'c': ['!', '@']}

    t0 = time.time()
    jobs_unvec = cartesian_product_unvectorized(dict0)
    t_unvec = time.time() - t0

    t0 = time.time()
    jobs_vec = cartesian_product_vectorized(dict0)
    t_vec = time.time() - t0

    print(f"3-key dict (2x2x2=8 combos): unvectorized={len(jobs_unvec)} jobs "
          f"in {t_unvec * 1e6:.1f}us, vectorized={len(jobs_vec)} jobs "
          f"in {t_vec * 1e6:.1f}us")
    assert jobs_unvec == jobs_vec, "vectorized and unvectorized must agree"
    print("Vectorized and unvectorized outputs match exactly.")

    # The real point: the unvectorized version is HARDCODED to 3 keys (3
    # nested for-loops). It cannot run on a 6-key dict at all without
    # rewriting the function. The vectorized version needs zero changes.
    dict6 = {chr(97 + i): ['0', '1', '2'] for i in range(6)}  # a..f, 3 values each
    t0 = time.time()
    jobs6 = cartesian_product_vectorized(dict6)
    t6 = time.time() - t0
    print(f"6-key dict (3^6={len(jobs6)} combos): vectorized handled it in "
          f"{t6 * 1e3:.2f}ms with the SAME function, no code changes. "
          f"The 3-loop unvectorized version structurally cannot do this.")
    print()


def part_b_partitioning():
    print('=' * 70)
    print('PART B: Atoms and Molecules -- lin_parts vs nested_parts (20.4)')
    print('=' * 70)

    # Real-world-shaped example: Ch17's SADF computation is a genuine
    # two-nested-loop task (for each of N candidate end-points, an inner
    # loop over start-points) -- N here is Ch17's real gold series length
    # used for its SADF sup-search (~1,760 real trading days).
    num_atoms = 1760
    num_threads = 4

    lin = lin_parts(num_atoms, num_threads)
    nest = nested_parts(num_atoms, num_threads)

    lin_sizes = np.diff(lin)
    print(f"lin_parts({num_atoms}, {num_threads}) molecule sizes (atom counts): "
          f"{lin_sizes.tolist()}")
    print("  -> equal ATOM count per molecule, but atom #i's real triangular\n"
          "     workload is i tasks -- so molecule 4 (highest atom indices)\n"
          "     does far more actual work than molecule 1, despite equal atoms.")

    nest_sizes = np.diff(nest)
    nest_task_counts = [
        sum(range(nest[k] + 1, nest[k + 1] + 1)) for k in range(len(nest) - 1)
    ]
    print(f"nested_parts({num_atoms}, {num_threads}) molecule sizes (atom counts): "
          f"{nest_sizes.tolist()}")
    print(f"  -> real triangular TASK counts per molecule: {nest_task_counts} "
          f"(target ~{num_atoms * (num_atoms + 1) // 2 // num_threads} each)")
    total = num_atoms * (num_atoms + 1) // 2
    print(f"  Sum of task counts = {sum(nest_task_counts)}, matches N(N+1)/2 = {total}: "
          f"{sum(nest_task_counts) == total}")
    print()


def part_c_real_timing_benchmark():
    print('=' * 70)
    print('PART C: REAL single-thread vs. multiprocessing timing (20.3/20.4)')
    print('=' * 70)
    print("Book's own scale (1000-step, 10,000-path one-touch double barrier). "
          "Timed once on this real machine (not the book's repeat(5,10)=50 "
          "runs -- that would take far too long for a repeatable demo).")

    import multiprocessing as mp
    detected_cores = mp.cpu_count()
    print(f"Detected CPU count on this machine: {detected_cores}")
    if detected_cores < 2:
        print("WARNING: fewer than 2 CPUs detected -- multiprocessing cannot "
              "show real speedup here (no second core to run on). This is "
              "expected in a single-core sandbox; on the real 6-core machine "
              "this should show genuine speedup. Numbers below are NOT "
              "representative if this warning fired.")

    results = time_single_vs_multi(
        num_paths=10000, path_len=1000, width=.5, num_threads=4,
        repeats=1, seed=12345,
    )
    print(f"Single-thread: {results['single_thread_best_sec']:.2f}s")
    print(f"Multiprocessing ({results['num_threads']} threads): "
          f"{results['multiprocessing_best_sec']:.2f}s")
    print(f"Speedup: {results['speedup']:.2f}x")
    print("(This project's established sweet spot on this 6-core machine is "
          "4 threads, not the book's 24 -- see CLAUDE.md.)")
    print()
    return results


def part_d_engine_closes_the_loop():
    print('=' * 70)
    print('PART D: The formalized engine (20.5/20.6) -- same problem, mp_job_list')
    print('=' * 70)
    print("Same one-touch double-barrier problem as Part C, this time run "
          "through this chapter's own mp_job_list (utils/multiprocess.py) "
          "with a dict.update reducer, instead of hand-rolled mp.Pool code "
          "-- and verified correctly globally-indexed regardless of thread count.")

    rng = np.random.default_rng(12345)
    r = rng.normal(0, .01, size=(500, 2000))

    out_1 = run_barrier_touch_engine(r, width=.5, num_threads=1)
    out_4 = run_barrier_touch_engine(r, width=.5, num_threads=4)

    print(f"1 thread: {len(out_1)} of {r.shape[1]} paths touched the barrier")
    print(f"4 threads: {len(out_4)} of {r.shape[1]} paths touched the barrier")
    print(f"Results identical regardless of thread count: {out_1 == out_4}")
    print()


if __name__ == '__main__':
    part_a_vectorization()
    part_b_partitioning()
    part_c_real_timing_benchmark()
    part_d_engine_closes_the_loop()


# =============================================================================
# Pytest results (REAL-MACHINE CONFIRMED 2026-08-08, Windows, mlfinlab env,
# Python 3.10.20, pytest 9.0.3, two-pass -- repo root below, plus isolated
# passes from inside utils/ (30 collected) and ch20/multiprocessing_engine/
# (21 collected), both matching this combined run exactly). Regression check
# on the four chapters importing utils/multiprocess.py (ch03/04/10): 104
# passed, 3 skipped (expected Windows multiprocessing skips), 0 failed.
#
# $ pytest utils\ ch20\ -v
#
# utils/test_multiprocess.py::TestLinParts::test_hand_traced_10_atoms_3_threads PASSED
# utils/test_multiprocess.py::TestLinParts::test_threads_exceeding_atoms_caps_at_num_atoms PASSED
# utils/test_multiprocess.py::TestLinParts::test_single_thread_gives_one_molecule PASSED
# utils/test_multiprocess.py::TestLinParts::test_partitions_cover_all_atoms_exactly_once PASSED
# utils/test_multiprocess.py::TestMpPandasObj::test_single_threaded_matches_direct_call PASSED
# utils/test_multiprocess.py::TestMpPandasObj::test_multiprocessing_matches_single_threaded PASSED
# utils/test_multiprocess.py::TestMpPandasObj::test_empty_index_returns_empty_dataframe PASSED
# utils/test_multiprocess.py::TestMpPandasObj::test_non_pandas_output_falls_through_as_list PASSED
# utils/test_multiprocess.py::TestMpPandasObj::test_mp_batches_changes_molecule_count_not_result PASSED
# utils/test_multiprocess.py::TestProcessJobs::test_sequential_execution PASSED
# utils/test_multiprocess.py::TestProcessJobsMp::test_multiprocessing_matches_sequential PASSED
# utils/test_multiprocess.py::TestNestedParts::test_hand_traced_10_atoms_3_threads PASSED
# utils/test_multiprocess.py::TestNestedParts::test_upper_triang_reverses_molecule_sizes PASSED
# utils/test_multiprocess.py::TestNestedParts::test_hand_traced_20_atoms_6_threads PASSED
# utils/test_multiprocess.py::TestNestedParts::test_partition_covers_all_atoms PASSED
# utils/test_multiprocess.py::TestNestedParts::test_triangular_task_count_conserved PASSED
# utils/test_multiprocess.py::TestNestedParts::test_threads_exceeding_atoms_caps_at_num_atoms PASSED
# utils/test_multiprocess.py::TestReportProgress::test_message_contains_percent_and_task_name PASSED
# utils/test_multiprocess.py::TestReportProgress::test_final_job_reports_100_percent PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_no_redux_falls_back_to_list PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_numeric_add_reduction_matches_manual_sum PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_series_add_reduction PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_dict_update_in_place_reduction PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_explicit_list_append_reducer_documents_real_quirky_behavior PASSED
# utils/test_multiprocess.py::TestProcessJobsRedux::test_multiprocessing_matches_sequential PASSED
# utils/test_multiprocess.py::TestMpJobList::test_matches_mp_pandas_obj_when_no_redux_used_directly PASSED
# utils/test_multiprocess.py::TestMpJobList::test_with_redux_reduces_across_molecules PASSED
# utils/test_multiprocess.py::TestMpJobList::test_nested_partitioning_option PASSED
# utils/test_multiprocess.py::TestMpJobList::test_empty_atom_list_returns_none PASSED
# utils/test_multiprocess.py::TestMpJobList::test_multiprocessing_matches_sequential PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouch::test_hand_traced_touch_and_no_touch PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouch::test_touch_at_first_row_that_crosses PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouch::test_negative_barrier_touch PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouch::test_no_columns_touch_gives_empty_dict PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouch::test_wider_barrier_delays_or_prevents_touch PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestMain0Main1Consistency::test_main0_runs_and_returns_dict PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestMain0Main1Consistency::test_main1_runs_and_returns_one_result_per_chunk PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestMain0Main1Consistency::test_main1_num_threads_capped_at_num_paths PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestBarrierTouchEngine::test_preserves_global_column_index PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestRunBarrierTouchEngine::test_single_threaded_matches_direct_call PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestRunBarrierTouchEngine::test_multiprocessing_matches_single_threaded PASSED
# ch20/multiprocessing_engine/test_barrier_touch.py::TestRunBarrierTouchEngine::test_result_keyed_by_true_global_column_regardless_of_chunking PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductUnvectorized::test_matches_hand_enumerated_jobs PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductUnvectorized::test_job_count_is_product_of_list_lengths PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorized::test_matches_unvectorized_version PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorized::test_matches_hand_enumerated_jobs PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorized::test_generalizes_beyond_three_keys PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorized::test_single_valued_lists_give_single_combo PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorized::test_empty_list_gives_no_combos PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorizedGenerator::test_is_lazy_not_a_list PASSED
# ch20/multiprocessing_engine/test_vectorization.py::TestCartesianProductVectorizedGenerator::test_materializes_to_same_result_as_list_version PASSED
#
# 51 passed in 9.61s (real machine: Windows, mlfinlab env, Python 3.10.20)
# =============================================================================
