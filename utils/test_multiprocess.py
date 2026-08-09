"""
TDD suite for utils/multiprocess.py -- AFML Chapter 20's mpPandasObj /
linParts / nestedParts / processJobs / processJobsRedux / mpJobList family.

Two groups of tests:
  1. lin_parts / mp_pandas_obj / process_jobs / process_jobs_mp -- these
     already existed in the repo (used since Ch03) but had NO dedicated
     test file until this session. Covered here for the first time.
  2. nested_parts / report_progress / process_jobs_redux / mp_job_list --
     new this session (Ch20), added additively per the "extend, don't
     modify" decision (see utils/multiprocess.py header comment).

All numeric partition-boundary values below are hand-traced against the
book's own closed-form formulas (Snippets 20.5/20.6), not just re-derived
from the code under test -- see inline comments for the arithmetic.

Worker functions used with multiprocessing.Pool are imported from
_mp_test_workers.py (module-level, picklable) rather than defined inline --
inline/lambda targets are not spawn-safe on Windows (this machine's actual
platform), even though they might appear to work under Linux/fork in a
sandbox. Kept `num_threads=1` (sequential) as the default for most
correctness tests to keep the suite fast and deterministic; a smaller
number of tests explicitly exercise `num_threads=2` to confirm the
multiprocessing path itself works, not just the sequential fallback.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from multiprocess import (
    lin_parts, mp_pandas_obj, process_jobs, process_jobs_mp,
    nested_parts, report_progress, process_jobs_redux, mp_job_list,
)
from _mp_test_workers import (
    square, double_list, sum_molecule, series_from_molecule,
    list_of_molecule, dict_from_molecule,
)


# =============================================================================
# Pre-existing functions -- no dedicated test coverage existed before this
# session despite being load-bearing for Ch03/04/08/10.
# =============================================================================
class TestLinParts:
    def test_hand_traced_10_atoms_3_threads(self):
        # linspace(0,10,4) = [0, 3.333, 6.667, 10] -> ceil -> [0,4,7,10]
        assert lin_parts(10, 3).tolist() == [0, 4, 7, 10]

    def test_threads_exceeding_atoms_caps_at_num_atoms(self):
        # min(10,5)+1 = 6 boundaries -> one atom per molecule
        assert lin_parts(5, 10).tolist() == [0, 1, 2, 3, 4, 5]

    def test_single_thread_gives_one_molecule(self):
        assert lin_parts(7, 1).tolist() == [0, 7]

    def test_partitions_cover_all_atoms_exactly_once(self):
        parts = lin_parts(97, 4)
        # boundaries must be monotonic and span the full atom range
        assert parts[0] == 0
        assert parts[-1] == 97
        assert list(parts) == sorted(parts)


class TestMpPandasObj:
    def test_single_threaded_matches_direct_call(self):
        idx = list(range(10))
        out = mp_pandas_obj(square, ('molecule', idx), num_threads=1)
        expected = pd.Series({i: i ** 2 for i in idx})
        pd.testing.assert_series_equal(out.sort_index(), expected.sort_index())

    def test_multiprocessing_matches_single_threaded(self):
        idx = list(range(20))
        out_mp = mp_pandas_obj(square, ('molecule', idx), num_threads=2)
        out_st = mp_pandas_obj(square, ('molecule', idx), num_threads=1)
        pd.testing.assert_series_equal(out_mp.sort_index(), out_st.sort_index())

    def test_empty_index_returns_empty_dataframe(self):
        # LOAD-BEARING: guards a real scenario from this pipeline (e.g. an
        # events index empty after a min_ret filter) -- must not crash.
        out = mp_pandas_obj(square, ('molecule', []), num_threads=1)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_non_pandas_output_falls_through_as_list(self):
        idx = list(range(6))
        out = mp_pandas_obj(double_list, ('molecule', idx), num_threads=1, mp_batches=1)
        # 6 atoms / 1 thread / 1 batch -> single molecule -> single list back
        assert out == [[0, 2, 4, 6, 8, 10]]

    def test_mp_batches_changes_molecule_count_not_result(self):
        idx = list(range(12))
        out_1batch = mp_pandas_obj(square, ('molecule', idx), num_threads=1, mp_batches=1)
        out_4batch = mp_pandas_obj(square, ('molecule', idx), num_threads=1, mp_batches=4)
        pd.testing.assert_series_equal(out_1batch.sort_index(), out_4batch.sort_index())


class TestProcessJobs:
    def test_sequential_execution(self):
        jobs = [{'molecule': [0, 1, 2]}, {'molecule': [3, 4]}]
        out = process_jobs(square, jobs)
        assert len(out) == 2
        pd.testing.assert_series_equal(out[0], pd.Series({0: 0, 1: 1, 2: 4}))
        pd.testing.assert_series_equal(out[1], pd.Series({3: 9, 4: 16}))


class TestProcessJobsMp:
    def test_multiprocessing_matches_sequential(self):
        jobs = [{'molecule': [0, 1]}, {'molecule': [2, 3]}, {'molecule': [4, 5]}]
        out_mp = process_jobs_mp(square, jobs, num_threads=2)
        out_seq = process_jobs(square, jobs)
        # imap/map order should both preserve job-list order here
        for a, b in zip(sorted(out_mp, key=lambda s: s.index[0]),
                         sorted(out_seq, key=lambda s: s.index[0])):
            pd.testing.assert_series_equal(a, b)


# =============================================================================
# New this session: nested_parts
# =============================================================================
class TestNestedParts:
    def test_hand_traced_10_atoms_3_threads(self):
        # Book Snippet 20.6 closed-form, hand-traced (see PR/handoff notes):
        # r1=5.576->6, r2=8.078->8, r3=10.0->10. Molecule sizes 6,2,2
        # (target ~18.33 triangular-tasks/molecule; actual 21,15,19 -- the
        # book itself notes rounding causes deviation from the exact target).
        assert nested_parts(10, 3).tolist() == [0, 6, 8, 10]

    def test_upper_triang_reverses_molecule_sizes(self):
        # diff([0,6,8,10])=[6,2,2] -> reversed=[2,2,6] -> cumsum=[2,4,10]
        assert nested_parts(10, 3, upper_triang=True).tolist() == [0, 2, 4, 10]

    def test_hand_traced_20_atoms_6_threads(self):
        # Cross-checked directly against the closed-form formula by hand
        # (independently reproduced, not just re-running the code under test).
        assert nested_parts(20, 6).tolist() == [0, 8, 11, 14, 16, 18, 20]

    def test_partition_covers_all_atoms(self):
        parts = nested_parts(50, 7)
        assert parts[0] == 0
        assert parts[-1] == 50
        assert list(parts) == sorted(parts)

    def test_triangular_task_count_conserved(self):
        # Every atom's triangular task-count (row i contributes i tasks,
        # 1-indexed) must sum to exactly N(N+1)/2 across all molecules --
        # this is the invariant the whole partition scheme exists to balance.
        n, m = 30, 4
        parts = nested_parts(n, m)
        total_tasks = sum(
            sum(range(parts[k] + 1, parts[k + 1] + 1))
            for k in range(len(parts) - 1)
        )
        assert total_tasks == n * (n + 1) // 2

    def test_threads_exceeding_atoms_caps_at_num_atoms(self):
        parts = nested_parts(4, 10)
        assert parts[-1] == 4
        assert len(parts) - 1 <= 4


# =============================================================================
# New this session: report_progress
# =============================================================================
class TestReportProgress:
    def test_message_contains_percent_and_task_name(self):
        import time as _time
        msg = report_progress(5, 10, _time.time() - 1, 'square')
        assert '50.0%' in msg
        assert 'square' in msg
        assert 'done after' in msg
        assert 'Remaining' in msg

    def test_final_job_reports_100_percent(self):
        import time as _time
        msg = report_progress(10, 10, _time.time() - 1, 'square')
        assert '100.0%' in msg


# =============================================================================
# New this session: process_jobs_redux
# =============================================================================
class TestProcessJobsRedux:
    def test_no_redux_falls_back_to_list(self):
        jobs = [{'molecule': [0, 1]}, {'molecule': [2, 3]}]
        out = process_jobs_redux(double_list, jobs, num_threads=1)
        assert sorted(out, key=lambda x: x[0]) == [[0, 2], [4, 6]]

    def test_numeric_add_reduction_matches_manual_sum(self):
        jobs = [{'molecule': [0, 1, 2]}, {'molecule': [3, 4]}, {'molecule': [5, 6, 7]}]
        import operator
        out = process_jobs_redux(sum_molecule, jobs, num_threads=1,
                                  redux=operator.add)
        assert out == sum(range(8))  # 0+1+...+7 = 28

    def test_series_add_reduction(self):
        # Two molecules each touching disjoint keys -> pd.Series.add with
        # fill_value semantics not needed since keys don't overlap here.
        jobs = [{'molecule': [0, 1]}, {'molecule': [2, 3]}]
        out = process_jobs_redux(series_from_molecule, jobs, num_threads=1,
                                  redux=pd.Series.add, redux_args={'fill_value': 0})
        # dtype float64 is expected here, not a bug: pd.Series.add with a
        # fill_value promotes to float regardless of the inputs' own dtype.
        expected = pd.Series({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})
        pd.testing.assert_series_equal(out.sort_index(), expected.sort_index())

    def test_dict_update_in_place_reduction(self):
        # This is the book's own worked example of an in-place reducer
        # (dict.update requires reduxInPlace=True) -- and the one case where
        # the "first output becomes the accumulator directly" branch (no
        # list-wrapping) is unambiguously correct, since dict.update's job
        # IS to merge into an existing dict in place.
        jobs = [{'molecule': [0, 1]}, {'molecule': [2, 3]}, {'molecule': [4]}]
        out = process_jobs_redux(dict_from_molecule, jobs, num_threads=1,
                                  redux=dict.update, redux_in_place=True)
        assert out == {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

    def test_explicit_list_append_reducer_documents_real_quirky_behavior(self):
        # NOT a book-fidelity bug fix -- this documents genuinely surprising
        # real behavior, verified against the book's own printed Snippet
        # 20.12 logic (kept verbatim, not "improved"). When redux is left as
        # the default None, the first output gets WRAPPED in a list first
        # (out=[out_]) before list.append is used on subsequent outputs.
        # But when the USER explicitly passes redux=list.append (the same
        # reducer!) with redux_in_place=True, the book's own branching skips
        # the list-wrapping step (out=copy.deepcopy(out_) instead) -- so an
        # explicit list.append does NOT behave the same as the implicit
        # default. Confirmed correct interpretation: the explicit
        # dict.update/list.append path is really designed for reducers whose
        # very first output already IS the correct-shaped accumulator (e.g.
        # dict.update's first dict), not for building a wrapping list from
        # scratch -- for that, leave redux=None and get the default behavior.
        jobs = [{'molecule': [0, 1]}, {'molecule': [2, 3]}]
        out_explicit = process_jobs_redux(list_of_molecule, jobs, num_threads=1,
                                           redux=list.append, redux_in_place=True)
        out_default = process_jobs_redux(list_of_molecule, jobs, num_threads=1)
        # Explicit list.append: first molecule's list IS out (unwrapped),
        # second molecule's list gets appended as a single nested element.
        assert out_explicit == [0, 1, [2, 3]]
        # Default (redux=None): first molecule's list gets wrapped, second
        # appended the same way -- structurally different result for the
        # exact same inputs and "equivalent-sounding" reducer.
        assert out_default == [[0, 1], [2, 3]]

    def test_multiprocessing_matches_sequential(self):
        jobs = [{'molecule': [0, 1, 2]}, {'molecule': [3, 4]}, {'molecule': [5, 6, 7]}]
        import operator
        out_mp = process_jobs_redux(sum_molecule, jobs, num_threads=2,
                                     redux=operator.add)
        out_seq = process_jobs_redux(sum_molecule, jobs, num_threads=1,
                                      redux=operator.add)
        assert out_mp == out_seq == 28


# =============================================================================
# New this session: mp_job_list
# =============================================================================
class TestMpJobList:
    def test_matches_mp_pandas_obj_when_no_redux_used_directly(self):
        # Without redux, mp_job_list falls back to the same list-accumulation
        # as process_jobs_redux's default -- verify against a manual sum of
        # the pieces rather than mp_pandas_obj (different return shape by design).
        idx = list(range(10))
        out = mp_job_list(square, ('molecule', idx), num_threads=1)
        combined = pd.concat(out).sort_index()
        expected = pd.Series({i: i ** 2 for i in idx})
        pd.testing.assert_series_equal(combined, expected.sort_index())

    def test_with_redux_reduces_across_molecules(self):
        idx = list(range(8))
        import operator
        out = mp_job_list(sum_molecule, ('molecule', idx), num_threads=1,
                           mp_batches=2, redux=operator.add)
        assert out == sum(idx)

    def test_nested_partitioning_option(self):
        # lin_mols=False routes through nested_parts instead of lin_parts --
        # confirm it still produces a correct (if differently-molecule-sized)
        # reduction over the same atoms.
        idx = list(range(10))
        import operator
        out = mp_job_list(sum_molecule, ('molecule', idx), num_threads=1,
                           mp_batches=3, lin_mols=False, redux=operator.add)
        assert out == sum(idx)

    def test_empty_atom_list_returns_none(self):
        out = mp_job_list(square, ('molecule', []), num_threads=1)
        assert out is None

    def test_multiprocessing_matches_sequential(self):
        idx = list(range(12))
        import operator
        out_mp = mp_job_list(sum_molecule, ('molecule', idx), num_threads=2,
                              redux=operator.add)
        out_seq = mp_job_list(sum_molecule, ('molecule', idx), num_threads=1,
                               redux=operator.add)
        assert out_mp == out_seq == sum(idx)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
