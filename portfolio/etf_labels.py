"""Cross-sectional labels for the ETF panel (pre-registered design).

For ETF i on date t, with horizon h trading days:
    fwd_ret    = log(price at t+h / price at t)
    median     = median of fwd_ret across the ETFs in the universe on date t
    excess_ret = fwd_ret - median
    label      = 1 if excess_ret > 0 else 0   (ties with the median get 0)
    t1         = the date h trading days after t (when the label is resolved;
                purging and uniqueness need it)

The universe on date t is exactly the (date, asset) rows you pass in, the ETFs
whose features exist that day. An ETF with no price at t+h gets no label.
Dates with fewer than min_assets labeled ETFs are dropped. The last h dates
have no forward return, so they never appear here; they are only used live.

The default horizon is 21 trading days (about one month), a judgment call made
for cost reasons, not a tested result.
"""
import numpy as np
import pandas as pd

from etf_features import DEFAULT_MIN_ASSETS, DEFAULT_WINDOWS, compute_features
from panel_data import wide_to_long

DEFAULT_HORIZON = 21


def forward_returns(prices, horizon):
    """Wide frame of log(price at t+horizon / price at t); NaN where unavailable."""
    logp = np.log(prices)
    return logp.shift(-horizon) - logp


def make_labels(prices, index, horizon=DEFAULT_HORIZON, min_assets=DEFAULT_MIN_ASSETS):
    """Labels for the (date, asset) rows in `index`. Columns: fwd_ret, excess_ret, label, t1."""
    fwd = wide_to_long(forward_returns(prices, horizon), "fwd_ret")
    fwd = fwd.reindex(index).dropna()

    cols = ["fwd_ret", "excess_ret", "label", "t1"]
    if fwd.empty:
        return pd.DataFrame(columns=cols, index=fwd.index)

    by_date = fwd.groupby(level="date")
    median = by_date.transform("median")
    count = by_date.transform("size")
    keep = count >= min_assets
    fwd, median = fwd[keep], median[keep]
    if fwd.empty:
        return pd.DataFrame(columns=cols, index=fwd.index)

    dates = prices.index
    t1_map = pd.Series(dates[horizon:], index=dates[:len(dates) - horizon])
    out = pd.DataFrame({"fwd_ret": fwd, "excess_ret": fwd - median})
    out["label"] = (out["excess_ret"] > 0).astype(int)
    out["t1"] = pd.Series(out.index.get_level_values("date")).map(t1_map).values
    return out[cols]


def build_dataset(prices, horizon=DEFAULT_HORIZON, windows=DEFAULT_WINDOWS,
                  min_assets=DEFAULT_MIN_ASSETS):
    """Features and labels on the same (date, asset) rows, ready for modeling."""
    feats = compute_features(prices, windows, min_assets)
    labels = make_labels(prices, feats.index, horizon, min_assets)
    return feats.reindex(labels.index).join(labels)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-04, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_panel_data.py test_etf_features.py test_etf_labels.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 31 items
#
# test_panel_data.py::test_loads_dates_as_sorted_index_and_tickers_as_columns PASSED  [  3%]
# test_panel_data.py::test_missing_prices_stay_nan_for_late_starting_etfs PASSED      [  6%]
# test_panel_data.py::test_duplicate_dates_are_rejected PASSED                        [  9%]
# test_panel_data.py::test_non_positive_prices_are_rejected PASSED                    [ 12%]
# test_panel_data.py::test_column_with_no_data_is_dropped PASSED                      [ 16%]
# test_panel_data.py::test_drop_hedge_removes_sh_and_leaves_the_rest_untouched PASSED [ 19%]
# test_panel_data.py::test_drop_hedge_ignores_a_hedge_that_is_not_present PASSED      [ 22%]
# test_etf_features.py::test_default_windows_are_the_preregistered_values PASSED      [ 25%]
# test_etf_features.py::test_trailing_vol_known_value PASSED                          [ 29%]
# test_etf_features.py::test_trend_distance_known_value PASSED                        [ 32%]
# test_etf_features.py::test_lagged_log_return_known_values PASSED                    [ 35%]
# test_etf_features.py::test_cross_sectional_rank_known_values PASSED                 [ 38%]
# test_etf_features.py::test_raw_momentum_is_divided_by_trailing_vol PASSED           [ 41%]
# test_etf_features.py::test_vol_feature_is_not_divided_by_itself PASSED              [ 45%]
# test_etf_features.py::test_features_are_ranks_strictly_between_zero_and_one PASSED  [ 48%]
# test_etf_features.py::test_each_date_has_ranks_spread_evenly_over_the_eligible_assets PASSED [ 51%]
# test_etf_features.py::test_no_lookahead_changing_later_prices_leaves_earlier_features_alone PASSED [ 54%]
# test_etf_features.py::test_late_starting_etf_only_appears_once_it_has_enough_history PASSED [ 58%]
# test_etf_features.py::test_dates_with_too_few_eligible_etfs_are_dropped PASSED      [ 61%]
# test_etf_features.py::test_flat_price_etf_is_left_out_not_given_an_infinite_feature PASSED [ 64%]
# test_etf_labels.py::test_default_horizon_is_21_trading_days PASSED                  [ 67%]
# test_etf_labels.py::test_forward_return_known_values PASSED                         [ 70%]
# test_etf_labels.py::test_labels_known_values_balanced_case PASSED                   [ 74%]
# test_etf_labels.py::test_ties_with_the_median_get_label_zero PASSED                 [ 77%]
# test_etf_labels.py::test_last_dates_without_a_forward_return_are_excluded PASSED    [ 80%]
# test_etf_labels.py::test_t1_is_the_date_h_trading_days_ahead PASSED                 [ 83%]
# test_etf_labels.py::test_median_uses_only_the_assets_in_the_index PASSED            [ 87%]
# test_etf_labels.py::test_dates_with_too_few_assets_are_dropped PASSED               [ 90%]
# test_etf_labels.py::test_asset_with_missing_future_price_gets_no_label PASSED       [ 93%]
# test_etf_labels.py::test_label_uses_future_prices_but_features_index_is_all_that_decides_the_universe PASSED [ 96%]
# test_etf_labels.py::test_build_dataset_joins_features_and_labels_on_the_same_rows PASSED [100%]
#
# 31 passed in 1.92s
# ---------------------------------------------------------------------------
