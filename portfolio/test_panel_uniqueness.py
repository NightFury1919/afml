import numpy as np
import pandas as pd
import pytest

from panel_uniqueness import get_panel_average_uniqueness
from panel_time_decay import get_time_decay_by_date

# Six business days, shared by every asset in these examples.
IDX = pd.bdate_range("2020-01-01", periods=6)  # d0 .. d5
D0, D1, D2, D3, D4, D5 = IDX


def make_close():
    return pd.Series(np.arange(6, dtype=float) + 100.0, index=IDX)


def make_events(starts, ends):
    return pd.DataFrame({"t1": ends}, index=pd.DatetimeIndex(starts))


# Asset A: e1 covers d0-d2, e2 covers d2-d3, e3 covers d4-d5.
#   concurrency by bar: 1, 1, 2, 1, 1, 1
#   e1 = (1 + 1 + 1/2) / 3 = 5/6    e2 = (1/2 + 1) / 2 = 3/4    e3 = 1
EVENTS_A = make_events([D0, D2, D4], [D2, D3, D5])

# Asset B: f1 covers d0-d3, f2 covers d1-d2.
#   concurrency by bar: 1, 2, 2, 1
#   f1 = (1 + 1/2 + 1/2 + 1) / 4 = 3/4    f2 = (1/2 + 1/2) / 2 = 1/2
EVENTS_B = make_events([D0, D1], [D3, D2])


def test_single_asset_matches_the_book_function_known_values():
    out = get_panel_average_uniqueness({"A": make_close()}, {"A": EVENTS_A})
    assert np.allclose(out.values, [5 / 6, 3 / 4, 1.0])


def test_two_assets_known_values():
    out = get_panel_average_uniqueness(
        {"A": make_close(), "B": make_close()}, {"A": EVENTS_A, "B": EVENTS_B}
    )
    assert np.allclose(out.xs("A", level="asset").values, [5 / 6, 3 / 4, 1.0])
    assert np.allclose(out.xs("B", level="asset").values, [3 / 4, 1 / 2])


def test_one_asset_never_changes_another_assets_uniqueness():
    alone = get_panel_average_uniqueness({"A": make_close()}, {"A": EVENTS_A})
    together = get_panel_average_uniqueness(
        {"A": make_close(), "B": make_close()}, {"A": EVENTS_A, "B": EVENTS_B}
    )
    assert np.allclose(alone.values, together.xs("A", level="asset").values)


def test_index_is_date_then_asset_and_counts_every_event():
    out = get_panel_average_uniqueness(
        {"A": make_close(), "B": make_close()}, {"A": EVENTS_A, "B": EVENTS_B}
    )
    assert list(out.index.names) == ["date", "asset"]
    assert len(out) == len(EVENTS_A) + len(EVENTS_B)
    assert (D0, "A") in out.index and (D0, "B") in out.index


def test_asset_with_no_events_is_skipped():
    empty = pd.DataFrame({"t1": pd.Series([], dtype="datetime64[ns]")},
                         index=pd.DatetimeIndex([]))
    out = get_panel_average_uniqueness(
        {"A": make_close(), "B": make_close()}, {"A": EVENTS_A, "B": empty}
    )
    assert set(out.index.get_level_values("asset")) == {"A"}
    assert len(out) == 3


def test_mismatched_asset_names_are_rejected():
    with pytest.raises(ValueError):
        get_panel_average_uniqueness({"A": make_close()}, {"B": EVENTS_B})


def test_feeds_the_decay_wrapper_events_on_one_date_share_a_weight():
    out = get_panel_average_uniqueness(
        {"A": make_close(), "B": make_close()}, {"A": EVENTS_A, "B": EVENTS_B}
    )
    decay = get_time_decay_by_date(out, clf_last_w=0.0)
    # A's e1 and B's f1 both start on d0, so they share one weight.
    assert decay.loc[(D0, "A")] == pytest.approx(decay.loc[(D0, "B")])
    # The newest event date in the panel is d4 (A's e3); it gets weight 1.
    assert decay.loc[(D4, "A")] == pytest.approx(1.0)
    assert decay.index.equals(out.index)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-03, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_panel_uniqueness.py test_panel_time_decay.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 22 items
#
# test_panel_uniqueness.py::test_single_asset_matches_the_book_function_known_values PASSED  [  4%]
# test_panel_uniqueness.py::test_two_assets_known_values PASSED                              [  9%]
# test_panel_uniqueness.py::test_one_asset_never_changes_another_assets_uniqueness PASSED    [ 13%]
# test_panel_uniqueness.py::test_index_is_date_then_asset_and_counts_every_event PASSED      [ 18%]
# test_panel_uniqueness.py::test_asset_with_no_events_is_skipped PASSED                      [ 22%]
# test_panel_uniqueness.py::test_mismatched_asset_names_are_rejected PASSED                  [ 27%]
# test_panel_uniqueness.py::test_feeds_the_decay_wrapper_events_on_one_date_share_a_weight PASSED [ 31%]
# test_panel_time_decay.py::test_no_decay_when_c_is_one PASSED                               [ 36%]
# test_panel_time_decay.py::test_same_date_events_share_one_weight_known_values PASSED       [ 40%]
# test_panel_time_decay.py::test_newest_date_always_gets_weight_one PASSED                   [ 45%]
# test_panel_time_decay.py::test_positive_c_known_values PASSED                              [ 50%]
# test_panel_time_decay.py::test_negative_c_erases_oldest_known_values PASSED                [ 54%]
# test_panel_time_decay.py::test_weights_never_decrease_toward_the_present PASSED            [ 59%]
# test_panel_time_decay.py::test_result_does_not_depend_on_row_order PASSED                  [ 63%]
# test_panel_time_decay.py::test_output_keeps_input_index_and_order PASSED                   [ 68%]
# test_panel_time_decay.py::test_matches_book_function_when_dates_are_unique PASSED          [ 72%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.0] PASSED            [ 77%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.5] PASSED            [ 81%]
# test_panel_time_decay.py::test_zero_total_uniqueness_is_rejected PASSED                    [ 86%]
# test_panel_time_decay.py::test_nan_uniqueness_is_rejected PASSED                           [ 90%]
# test_panel_time_decay.py::test_empty_input_returns_empty PASSED                            [ 95%]
# test_panel_time_decay.py::test_multiindex_date_asset_input_is_supported PASSED             [100%]
#
# 22 passed in 1.45s
# ---------------------------------------------------------------------------
