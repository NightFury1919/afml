import numpy as np
import pandas as pd
import pytest

from panel_purged_cv import PanelPurgedKFold


def make_panel(n_dates, assets, horizon, long_horizon=None, start="2020-01-01"):
    """Panel indexed by (date, asset), sorted by date then asset.

    t1 is the date `horizon` calendar rows ahead (the calendar runs past the last
    date, so the last rows resolve after the panel ends). `long_horizon` is an
    optional {asset: horizon} override.
    """
    cal = pd.bdate_range(start, periods=n_dates + 100)
    rows, t1 = [], []
    for k in range(n_dates):
        for a in assets:
            h = (long_horizon or {}).get(a, horizon)
            rows.append((cal[k], a))
            t1.append(cal[k + h])
    idx = pd.MultiIndex.from_tuples(rows, names=["date", "asset"])
    X = pd.DataFrame({"f": np.arange(len(idx), dtype=float)}, index=idx)
    return X, pd.Series(t1, index=idx), cal[:n_dates]


def dates_of(X, positions):
    return set(X.index.get_level_values("date")[positions])


def test_known_values_two_folds_no_embargo():
    X, t1, cal = make_panel(10, ["A", "B"], horizon=2)
    folds = list(PanelPurgedKFold(n_splits=2, t1=t1).split(X))
    (tr1, te1), (tr2, te2) = folds
    # fold 1: test dates d0-d4 (rows 0-9). Latest test label ends d6, so train starts at d6 (rows 12-19).
    assert te1.tolist() == list(range(0, 10))
    assert tr1.tolist() == list(range(12, 20))
    # fold 2: test dates d5-d9 (rows 10-19). d3 resolves exactly at d5 (kept), d4 resolves d6 (purged).
    assert te2.tolist() == list(range(10, 20))
    assert tr2.tolist() == list(range(0, 8))


def test_embargo_removes_dates_after_the_test_block_only():
    X, t1, cal = make_panel(10, ["A", "B"], horizon=2)
    (tr1, _), (tr2, _) = list(PanelPurgedKFold(n_splits=2, t1=t1, embargo_dates=1).split(X))
    assert tr1.tolist() == list(range(14, 20))   # train now starts at d7, not d6
    assert tr2.tolist() == list(range(0, 8))     # before-test side is unchanged


def test_a_date_is_never_split_between_train_and_test():
    X, t1, _ = make_panel(60, list("ABCDE"), horizon=5)
    for tr, te in PanelPurgedKFold(n_splits=5, t1=t1, embargo_dates=2).split(X):
        assert dates_of(X, tr).isdisjoint(dates_of(X, te))


def test_no_training_label_window_overlaps_the_test_window_brute_force():
    X, t1, _ = make_panel(120, list("ABCDEF"), horizon=10)
    row_date = X.index.get_level_values("date")
    for emb in (0, 5):
        for tr, te in PanelPurgedKFold(n_splits=5, t1=t1, embargo_dates=emb).split(X):
            test_start = row_date[te].min()
            test_end = t1.iloc[te].max()
            for r in tr:
                before_ok = t1.iloc[r] <= test_start
                after_ok = row_date[r] >= test_end
                assert before_ok or after_ok


def test_longest_label_among_all_assets_decides_the_purge():
    # A resolves in 2 days, B in 10. B's long windows must purge A's rows too.
    X, t1, cal = make_panel(60, ["A", "B"], horizon=2, long_horizon={"B": 10})
    folds = list(PanelPurgedKFold(n_splits=3, t1=t1).split(X))
    tr, te = folds[1]                       # test dates d20-d39
    test_last = cal[39]
    latest_b = cal[39 + 10]                 # B's last test label ends d49
    train_dates = dates_of(X, tr)
    forbidden = {d for d in cal if test_last < d < latest_b}
    assert train_dates.isdisjoint(forbidden)
    assert cal[50] in train_dates           # and it resumes at d49 or later
    assert cal[49] in train_dates


def test_test_sets_cover_every_row_exactly_once():
    X, t1, _ = make_panel(37, list("ABC"), horizon=3)
    seen = np.concatenate([te for _, te in PanelPurgedKFold(n_splits=4, t1=t1).split(X)])
    assert sorted(seen.tolist()) == list(range(len(X)))


def test_indices_are_positions_in_x():
    X, t1, _ = make_panel(30, ["A", "B"], horizon=2)
    tr, te = next(iter(PanelPurgedKFold(n_splits=3, t1=t1).split(X)))
    assert len(X.iloc[tr]) == len(tr) and len(X.iloc[te]) == len(te)


def test_ragged_panel_late_starting_assets():
    X, t1, cal = make_panel(40, ["A", "B", "C"], horizon=3)
    drop = [(d, "C") for d in cal[:15]]
    X, t1 = X.drop(drop), t1.drop(drop)
    folds = list(PanelPurgedKFold(n_splits=4, t1=t1).split(X))
    row_date = X.index.get_level_values("date")
    for tr, te in folds:
        assert dates_of(X, tr).isdisjoint(dates_of(X, te))
    # each test block is a block of dates: test sizes differ because C is missing early on
    assert len({len(te) for _, te in folds}) > 1


def test_naive_row_split_cuts_a_date_in_half_which_is_why_this_class_splits_by_date():
    X, _, _ = make_panel(10, list("ABC"), horizon=2)
    first_block = np.array_split(np.arange(len(X)), 4)[0]
    last_row_date = X.index.get_level_values("date")[first_block[-1]]
    rows_that_day = (X.index.get_level_values("date") == last_row_date).sum()
    in_block_that_day = (X.index.get_level_values("date")[first_block] == last_row_date).sum()
    assert in_block_that_day < rows_that_day


def test_get_n_splits():
    X, t1, _ = make_panel(20, ["A"], horizon=1)
    assert PanelPurgedKFold(n_splits=4, t1=t1).get_n_splits() == 4


def test_t1_must_be_a_series():
    with pytest.raises(ValueError):
        PanelPurgedKFold(n_splits=3, t1=[1, 2, 3])


def test_index_must_be_date_asset_multiindex():
    X, t1, _ = make_panel(20, ["A", "B"], horizon=2)
    flat = X.reset_index(drop=True)
    flat_t1 = t1.reset_index(drop=True)
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=flat_t1).split(flat))


def test_x_and_t1_must_share_the_same_index():
    X, t1, _ = make_panel(20, ["A", "B"], horizon=2)
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=t1.iloc[::-1]).split(X))


def test_rows_must_be_sorted_by_date():
    X, t1, _ = make_panel(20, ["A", "B"], horizon=2)
    shuffled = X.iloc[::-1]
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=t1.loc[shuffled.index]).split(shuffled))


def test_duplicate_rows_are_rejected():
    X, t1, _ = make_panel(10, ["A"], horizon=2)
    dup_X = pd.concat([X, X.iloc[[0]]]).sort_index()
    dup_t1 = pd.concat([t1, t1.iloc[[0]]]).sort_index()
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=dup_t1).split(dup_X))


def test_missing_label_end_dates_are_rejected():
    X, t1, _ = make_panel(10, ["A"], horizon=2)
    t1 = t1.copy()
    t1.iloc[3] = pd.NaT
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=t1).split(X))


def test_label_ending_before_it_starts_is_rejected():
    X, t1, cal = make_panel(10, ["A"], horizon=2)
    t1 = t1.copy()
    t1.iloc[4] = cal[0]
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=t1).split(X))


def test_more_splits_than_dates_is_rejected():
    X, t1, _ = make_panel(3, ["A", "B"], horizon=1)
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=5, t1=t1).split(X))


def test_an_empty_training_set_is_an_error_not_a_silent_fold():
    X, t1, _ = make_panel(10, ["A", "B"], horizon=2)
    with pytest.raises(ValueError):
        list(PanelPurgedKFold(n_splits=2, t1=t1, embargo_dates=50).split(X))

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
