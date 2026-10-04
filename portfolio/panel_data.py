"""Load the daily price panel and reshape wide <-> long.

Wide  : one row per date, one column per ETF (what pull_prices.py writes).
Long  : one row per (date, asset), the shape the models and labels use.

ETFs launched at different times, so early cells in a wide frame are NaN.
Nothing here fills them: an ETF only joins the panel once it has real prices.
"""
import numpy as np
import pandas as pd


def load_prices(path):
    """Read prices_daily.csv into a wide frame (date index, ticker columns).

    Raises ValueError on duplicate dates or non-positive prices. A column with
    no data at all is dropped. Missing prices stay NaN.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("prices file has duplicate dates")
    df = df.set_index("date").sort_index().astype(float)
    df = df.dropna(axis=1, how="all")
    if (df.fillna(1.0) <= 0).any().any():
        raise ValueError("prices file has non-positive prices")
    df.index.name = "date"
    df.columns.name = "asset"
    return df


def wide_to_long(wide, name):
    """Wide frame -> Series indexed by (date, asset), named `name`. Keeps NaN rows."""
    w = wide.rename_axis(index="date", columns="asset")
    long = w.reset_index().melt(id_vars="date", var_name="asset", value_name=name)
    return long.set_index(["date", "asset"])[name].sort_index()


# SH is the inverse-S&P hedge candidate. It is pulled with the universe but is
# not one of the 48 ranked ETFs, so it must not be ranked or labeled with them.
HEDGE_TICKERS = ("SH",)


def drop_hedge(prices, hedge=HEDGE_TICKERS):
    """Prices without the hedge ETFs, so only the ranked universe is left."""
    return prices.drop(columns=[c for c in hedge if c in prices.columns])

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
