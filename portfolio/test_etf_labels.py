import numpy as np
import pandas as pd
import pytest

from etf_labels import DEFAULT_HORIZON, forward_returns, make_labels, build_dataset

D = pd.bdate_range("2020-01-01", periods=5)  # d0 .. d4


def prices4():
    df = pd.DataFrame({
        "A": [100, 100, 110, 120, 130],
        "B": [100, 100, 100, 100, 100],
        "C": [100, 100, 90, 80, 70],
        "D": [100, 100, 105, 105, 105],
    }, index=D, dtype=float)
    df.index.name, df.columns.name = "date", "asset"
    return df


def full_index(prices, dates=None):
    dates = list(prices.index if dates is None else dates)
    return pd.MultiIndex.from_product([dates, prices.columns], names=["date", "asset"])


def test_default_horizon_is_21_trading_days():
    assert DEFAULT_HORIZON == 21


def test_forward_return_known_values():
    fwd = forward_returns(prices4(), horizon=2)
    assert fwd.loc[D[0], "A"] == pytest.approx(np.log(1.10))
    assert fwd.loc[D[0], "C"] == pytest.approx(np.log(0.90))
    assert np.isnan(fwd.loc[D[3], "A"])  # no price two days ahead


def test_labels_known_values_balanced_case():
    p = prices4()
    out = make_labels(p, full_index(p), horizon=2, min_assets=4)
    day0 = out.xs(D[0], level="date")
    median = (0.0 + np.log(1.05)) / 2
    assert day0.loc["A", "excess_ret"] == pytest.approx(np.log(1.10) - median)
    assert day0.loc["B", "excess_ret"] == pytest.approx(0.0 - median)
    assert day0["label"].to_dict() == {"A": 1, "B": 0, "C": 0, "D": 1}


def test_ties_with_the_median_get_label_zero():
    p = prices4()
    out = make_labels(p, full_index(p), horizon=2, min_assets=4)
    day2 = out.xs(D[2], level="date")  # B and D both return 0, median is 0
    assert day2["label"].to_dict() == {"A": 1, "B": 0, "C": 0, "D": 0}


def test_last_dates_without_a_forward_return_are_excluded():
    p = prices4()
    out = make_labels(p, full_index(p), horizon=2, min_assets=4)
    assert sorted(set(out.index.get_level_values("date"))) == list(D[:3])
    assert len(out) == 12


def test_t1_is_the_date_h_trading_days_ahead():
    p = prices4()
    out = make_labels(p, full_index(p), horizon=2, min_assets=4)
    for d0, d2 in zip(D[:3], D[2:]):
        assert (out.xs(d0, level="date")["t1"] == d2).all()


def test_median_uses_only_the_assets_in_the_index():
    p = prices4()
    idx = full_index(p, dates=[D[0]]).drop([(D[0], "D")])
    out = make_labels(p, idx, horizon=2, min_assets=3)
    day0 = out.xs(D[0], level="date")
    assert set(day0.index) == {"A", "B", "C"}
    assert day0["label"].to_dict() == {"A": 1, "B": 0, "C": 0}  # median is B's 0.0


def test_dates_with_too_few_assets_are_dropped():
    p = prices4()
    out = make_labels(p, full_index(p), horizon=2, min_assets=5)
    assert len(out) == 0


def test_asset_with_missing_future_price_gets_no_label():
    p = prices4()
    p.loc[D[2], "C"] = np.nan
    out = make_labels(p, full_index(p), horizon=2, min_assets=3)
    assert ("C" not in out.xs(D[0], level="date").index)


def test_label_uses_future_prices_but_features_index_is_all_that_decides_the_universe():
    p = prices4()
    idx = full_index(p, dates=[D[0]]).drop([(D[0], "A")])
    out = make_labels(p, idx, horizon=2, min_assets=3)
    assert "A" not in out.xs(D[0], level="date").index


def test_build_dataset_joins_features_and_labels_on_the_same_rows():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=60)
    data = {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))) for c in "ABCDEF"}
    p = pd.DataFrame(data, index=idx)
    p.index.name, p.columns.name = "date", "asset"
    tiny = {"mom_long": 6, "mom_skip": 2, "mom_med": 4, "ret_short": 2, "vol": 3, "trend": 5}
    ds = build_dataset(p, horizon=5, windows=tiny, min_assets=3)
    assert {"mom_12_1", "mom_3m", "ret_5d", "vol_60", "trend_200",
            "fwd_ret", "excess_ret", "label", "t1"} <= set(ds.columns)
    assert not ds.isna().any().any()
    assert set(ds["label"].unique()) <= {0, 1}
    # the last 5 dates have no forward return, so none of them can appear
    assert ds.index.get_level_values("date").max() <= idx[-6]

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
