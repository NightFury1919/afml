import numpy as np
import pandas as pd
import pytest

from panel_data import drop_hedge, load_prices


def write(tmp_path, text):
    p = tmp_path / "prices.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_dates_as_sorted_index_and_tickers_as_columns(tmp_path):
    p = write(tmp_path, "date,SPY,QQQ\n2020-01-03,102,202\n2020-01-02,101,201\n")
    out = load_prices(p)
    assert list(out.columns) == ["SPY", "QQQ"]
    assert out.index.is_monotonic_increasing
    assert out.index.name == "date" and out.columns.name == "asset"
    assert out.loc[pd.Timestamp("2020-01-02"), "QQQ"] == 201


def test_missing_prices_stay_nan_for_late_starting_etfs(tmp_path):
    p = write(tmp_path, "date,SPY,EMB\n2020-01-02,101,\n2020-01-03,102,50\n")
    out = load_prices(p)
    assert np.isnan(out.loc[pd.Timestamp("2020-01-02"), "EMB"])


def test_duplicate_dates_are_rejected(tmp_path):
    p = write(tmp_path, "date,SPY\n2020-01-02,101\n2020-01-02,102\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_non_positive_prices_are_rejected(tmp_path):
    p = write(tmp_path, "date,SPY\n2020-01-02,101\n2020-01-03,0\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_column_with_no_data_is_dropped(tmp_path):
    p = write(tmp_path, "date,SPY,ZZZ\n2020-01-02,101,\n2020-01-03,102,\n")
    out = load_prices(p)
    assert list(out.columns) == ["SPY"]


def test_drop_hedge_removes_sh_and_leaves_the_rest_untouched():
    prices = pd.DataFrame({"SPY": [1.0, 2.0], "SH": [3.0, 4.0], "QQQ": [5.0, 6.0]})
    out = drop_hedge(prices)
    assert list(out.columns) == ["SPY", "QQQ"]
    assert list(prices.columns) == ["SPY", "SH", "QQQ"]  # input not modified


def test_drop_hedge_ignores_a_hedge_that_is_not_present():
    prices = pd.DataFrame({"SPY": [1.0, 2.0]})
    assert list(drop_hedge(prices).columns) == ["SPY"]

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
