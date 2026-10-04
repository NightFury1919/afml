"""Time decay for a panel of many assets (AFML Chapter 4, Section 4.7).

Why a wrapper?  The book's get_time_decay() (Snippet 4.11, in
ch04/sample_weights/time_decay.py) sorts events by date and decays along the
cumulative sum of average uniqueness.  That is correct for ONE asset, where
each date appears once.  In a pooled ETF panel many events share a date, and
sorting only by date leaves the order of those ties arbitrary, so two events
from the same day could get slightly different decay weights.

This wrapper fixes that without touching the book code:
    1. add up the uniqueness of all events that fall on each date,
    2. run the book's get_time_decay() on those one-row-per-date totals,
    3. give every event the weight of its date.

So every event on the same date shares one weight, and the result does not
depend on the order of the rows you pass in.

Input  : tw, a pd.Series of average uniqueness per event, indexed by event date
         (duplicate dates allowed).  For a panel, compute uniqueness across the
         whole panel, not per asset.
Output : pd.Series of decay weights, same index and same row order as tw.
         The newest date gets 1.0.  The meaning of clf_last_w (the book's c) is
         the same as in the book:
             1        no decay
             0 to 1   linear decay, oldest data keeps a positive weight
             0        oldest data fades to (almost) zero
             -1 to 0  the oldest slice gets weight exactly 0 (erased)
         c must be greater than -1; c = -1 would divide by zero.
"""
import sys
from pathlib import Path

import pandas as pd

# Let this file find the book code when run from the portfolio folder.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ch04.sample_weights.time_decay import get_time_decay  # noqa: E402


def get_time_decay_by_date(tw, clf_last_w=1.0):
    if not clf_last_w > -1:
        raise ValueError(f"clf_last_w must be greater than -1, got {clf_last_w}")
    if len(tw) == 0:
        return tw.astype(float).copy()
    if tw.isna().any():
        raise ValueError("tw contains NaN values")

    # Step 1: one uniqueness total per date.
    per_date = tw.groupby(level=0).sum()
    if not per_date.sum() > 0:
        raise ValueError("total uniqueness must be positive")

    # Step 2: the book's decay, on unique dates only.
    decay_by_date = get_time_decay(per_date, clf_last_w=clf_last_w)

    # Step 3: hand each event the weight of its date, keeping the input order.
    return pd.Series(decay_by_date.reindex(tw.index).to_numpy(), index=tw.index)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-03, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_panel_time_decay.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 14 items
#
# test_panel_time_decay.py::test_no_decay_when_c_is_one PASSED                       [  7%]
# test_panel_time_decay.py::test_same_date_events_share_one_weight_known_values PASSED [ 14%]
# test_panel_time_decay.py::test_newest_date_always_gets_weight_one PASSED           [ 21%]
# test_panel_time_decay.py::test_positive_c_known_values PASSED                      [ 28%]
# test_panel_time_decay.py::test_negative_c_erases_oldest_known_values PASSED        [ 35%]
# test_panel_time_decay.py::test_weights_never_decrease_toward_the_present PASSED    [ 42%]
# test_panel_time_decay.py::test_result_does_not_depend_on_row_order PASSED          [ 50%]
# test_panel_time_decay.py::test_output_keeps_input_index_and_order PASSED           [ 57%]
# test_panel_time_decay.py::test_matches_book_function_when_dates_are_unique PASSED  [ 64%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.0] PASSED    [ 71%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.5] PASSED    [ 78%]
# test_panel_time_decay.py::test_zero_total_uniqueness_is_rejected PASSED            [ 85%]
# test_panel_time_decay.py::test_nan_uniqueness_is_rejected PASSED                   [ 92%]
# test_panel_time_decay.py::test_empty_input_returns_empty PASSED                    [100%]
#
# 14 passed in 1.43s
# ---------------------------------------------------------------------------
