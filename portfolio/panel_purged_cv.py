"""Purged k-fold cross-validation for a PANEL of assets (adapted from AFML Snippet 7.3).

Why the book's PurgedKFold is not enough here
    Snippet 7.3 assumes one row per timestamp: it cuts the sample into contiguous
    blocks of ROWS. A panel has one row per ETF per date, so a row-based cut can
    end in the middle of a date: some ETFs from that day land in the test set and
    the others in the training set. Labels are measured relative to the whole
    universe, so ETFs on the same day (and on overlapping days) share information.
    That is a leak, and it would make every score look better than it is.

What this class does instead
    1. Folds are contiguous blocks of DATES. Every ETF's rows on a date go to the
       same side, so a date is never split.
    2. Purge, by date across ALL ETFs. The test window runs from the first test
       date to the latest label end (t1) among every test row, any ETF. A training
       row survives only if its own label resolved at or before the test start
       (t1 <= start), or it starts at or after the end of that test window. The
       longest label of any ETF therefore sets the cutoff for all of them.
       Touching exactly at a boundary date is allowed, as in the book (<=, >=).
    3. Embargo, after the test block only, counted in DATES (embargo_dates).
       The book's pctEmbargo equals int(n_dates * pct). Features look back up to
       252 days (mom_12_1), so training rows soon after the test block carry
       test-period returns inside their features. An embargo at least as long as
       the longest feature lookback closes that route; choose it deliberately.

Input
    X  : DataFrame indexed by (date, asset), sorted by date, unique index.
    t1 : Series on the same index, the date each label resolves (>= its own date).
Output of split(): (train_positions, test_positions), integer positions into X,
    so X.iloc[train] works. A fold with no training rows raises ValueError.
"""
import numpy as np
import pandas as pd


class PanelPurgedKFold:
    def __init__(self, n_splits=5, t1=None, embargo_dates=0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pd.Series indexed like X, values = label end dates")
        if embargo_dates < 0:
            raise ValueError("embargo_dates must be >= 0")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_dates = int(embargo_dates)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _validate(self, X):
        idx = X.index
        if not isinstance(idx, pd.MultiIndex) or list(idx.names) != ["date", "asset"]:
            raise ValueError("X must be indexed by a (date, asset) MultiIndex")
        if not idx.equals(self.t1.index):
            raise ValueError("X and t1 must share an identical, identically-ordered index")
        if not idx.is_unique:
            raise ValueError("X has duplicate (date, asset) rows")
        if not idx.get_level_values("date").is_monotonic_increasing:
            raise ValueError("X must be sorted by date")
        if self.t1.isna().any():
            raise ValueError("t1 has missing label end dates")

    def split(self, X, y=None, groups=None):
        self._validate(X)
        row_date = X.index.get_level_values("date").to_numpy()
        row_t1 = self.t1.to_numpy()
        if (row_t1 < row_date).any():
            raise ValueError("a label ends before it starts (t1 < date)")

        dates = np.unique(row_date)
        n_dates = len(dates)
        if self.n_splits > n_dates:
            raise ValueError(f"n_splits={self.n_splits} is more than the {n_dates} dates available")

        for block in np.array_split(np.arange(n_dates), self.n_splits):
            test_start, test_last = dates[block[0]], dates[block[-1]]
            test_mask = (row_date >= test_start) & (row_date <= test_last)
            test_idx = np.flatnonzero(test_mask)

            # Latest label end among ALL test rows (every ETF) sets the right-hand cutoff.
            test_end = row_t1[test_mask].max()

            before = (row_date < test_start) & (row_t1 <= test_start)

            first_after = int(np.searchsorted(dates, test_end))   # first date >= test_end
            cut_pos = first_after + self.embargo_dates
            if cut_pos < n_dates:
                after = (row_date > test_last) & (row_date >= dates[cut_pos])
            else:
                after = np.zeros(len(row_date), dtype=bool)

            train_idx = np.flatnonzero(before | after)
            if len(train_idx) == 0:
                raise ValueError("purging and embargo left no training rows for a fold")
            yield train_idx, test_idx

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-04, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_panel_purged_cv.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 19 items
#
# test_panel_purged_cv.py::test_known_values_two_folds_no_embargo PASSED              [  5%]
# test_panel_purged_cv.py::test_embargo_removes_dates_after_the_test_block_only PASSED [ 10%]
# test_panel_purged_cv.py::test_a_date_is_never_split_between_train_and_test PASSED   [ 15%]
# test_panel_purged_cv.py::test_no_training_label_window_overlaps_the_test_window_brute_force PASSED [ 21%]
# test_panel_purged_cv.py::test_longest_label_among_all_assets_decides_the_purge PASSED [ 26%]
# test_panel_purged_cv.py::test_test_sets_cover_every_row_exactly_once PASSED         [ 31%]
# test_panel_purged_cv.py::test_indices_are_positions_in_x PASSED                     [ 36%]
# test_panel_purged_cv.py::test_ragged_panel_late_starting_assets PASSED              [ 42%]
# test_panel_purged_cv.py::test_naive_row_split_cuts_a_date_in_half_which_is_why_this_class_splits_by_date PASSED [ 47%]
# test_panel_purged_cv.py::test_get_n_splits PASSED                                   [ 52%]
# test_panel_purged_cv.py::test_t1_must_be_a_series PASSED                            [ 57%]
# test_panel_purged_cv.py::test_index_must_be_date_asset_multiindex PASSED            [ 63%]
# test_panel_purged_cv.py::test_x_and_t1_must_share_the_same_index PASSED             [ 68%]
# test_panel_purged_cv.py::test_rows_must_be_sorted_by_date PASSED                    [ 73%]
# test_panel_purged_cv.py::test_duplicate_rows_are_rejected PASSED                    [ 78%]
# test_panel_purged_cv.py::test_missing_label_end_dates_are_rejected PASSED           [ 84%]
# test_panel_purged_cv.py::test_label_ending_before_it_starts_is_rejected PASSED      [ 89%]
# test_panel_purged_cv.py::test_more_splits_than_dates_is_rejected PASSED             [ 94%]
# test_panel_purged_cv.py::test_an_empty_training_set_is_an_error_not_a_silent_fold PASSED [100%]
#
# 19 passed in 1.11s
#
# Mutation check (sandbox, 2026-10-04): breaking the purge three ways (no
# before-test purge; cutoff from the first test row instead of the latest label
# across all ETFs; folds cut by rows instead of dates) made at least one test
# fail each time. Original code restored: 19 passed.
# ---------------------------------------------------------------------------
