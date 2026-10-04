"""Average uniqueness for a panel of assets (AFML Chapter 4, Sections 4.3 and 4.4).

What this does
    Runs the book's get_average_uniqueness() (Snippets 4.1 and 4.2, in
    ch04/sample_weights/uniqueness.py) once per asset, then stacks the results
    into one Series indexed by (date, asset).  The book code is not modified.

What "overlap" means here
    The book calls two labels concurrent when both depend on at least one
    common RETURN.  If each ETF's label is built from that ETF's own returns,
    labels from different ETFs share no return, so uniqueness is computed
    inside each ETF and one ETF never changes another's values.

What this does NOT do
    ETFs still move together, so 49 ETFs are not 49 independent samples.  That
    shows up in effective breadth, not here.  If labels are later defined
    relative to the universe (for example minus the universe median), labels
    on different ETFs DO share information and this function must be revisited.

Inputs
    close_by_asset  : dict {asset: pd.Series of closes indexed by date}
    events_by_asset : dict {asset: pd.DataFrame with a 't1' column, indexed by
                      event start date} (the Chapter 3 events object)
    Both dicts must contain the same assets.  An asset with no events is skipped.

Output
    pd.Series of average uniqueness, MultiIndex names ['date', 'asset'], assets
    in the order of close_by_asset.  Pass it to get_time_decay_by_date().
"""
import sys
from pathlib import Path

import pandas as pd

# Let this file find the book code when run from the portfolio folder.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ch04.sample_weights.uniqueness import get_average_uniqueness  # noqa: E402


def get_panel_average_uniqueness(close_by_asset, events_by_asset, num_threads=1):
    if set(close_by_asset) != set(events_by_asset):
        raise ValueError("close_by_asset and events_by_asset must have the same assets")

    pieces = []
    for asset, close in close_by_asset.items():
        events = events_by_asset[asset]
        if len(events) == 0:
            continue
        tw = get_average_uniqueness(close, events, num_threads=num_threads)
        tw.index = pd.MultiIndex.from_arrays(
            [tw.index, [asset] * len(tw)], names=["date", "asset"]
        )
        pieces.append(tw)

    if not pieces:
        empty_index = pd.MultiIndex.from_arrays([[], []], names=["date", "asset"])
        return pd.Series([], dtype=float, index=empty_index)
    return pd.concat(pieces)

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
