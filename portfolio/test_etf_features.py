import numpy as np
import pandas as pd
import pytest

from etf_features import (
    DEFAULT_WINDOWS,
    FEATURE_NAMES,
    compute_features,
    cross_sectional_rank,
    lagged_log_return,
    raw_features,
    trailing_vol,
    trend_distance,
)

TINY = {"mom_long": 6, "mom_skip": 2, "mom_med": 4, "ret_short": 2, "vol": 3, "trend": 5}


def wide(cols, n=40, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    data = {c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for c in cols}
    df = pd.DataFrame(data, index=idx)
    df.index.name, df.columns.name = "date", "asset"
    return df


def test_default_windows_are_the_preregistered_values():
    assert DEFAULT_WINDOWS == {"mom_long": 252, "mom_skip": 21, "mom_med": 63,
                               "ret_short": 5, "vol": 60, "trend": 200}
    assert FEATURE_NAMES == ["mom_12_1", "mom_3m", "ret_5d", "vol_60", "trend_200"]


def test_trailing_vol_known_value():
    # log returns +0.1 then -0.1 -> sample std = sqrt((0.1^2 + 0.1^2) / 1) = sqrt(0.02)
    p = pd.DataFrame({"A": [100.0, 100 * np.exp(0.1), 100.0]})
    out = trailing_vol(p, window=2)
    assert out["A"].iloc[-1] == pytest.approx(np.sqrt(0.02))
    assert np.isnan(out["A"].iloc[0]) and np.isnan(out["A"].iloc[1])


def test_trend_distance_known_value():
    p = pd.DataFrame({"A": [10.0, 12.0, 14.0]})
    out = trend_distance(p, window=3)
    assert out["A"].iloc[-1] == pytest.approx(14 / 12 - 1)
    assert np.isnan(out["A"].iloc[1])


def test_lagged_log_return_known_values():
    p = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})
    assert lagged_log_return(p, near=0, far=2)["A"].iloc[-1] == pytest.approx(np.log(1.21))
    assert lagged_log_return(p, near=1, far=3)["A"].iloc[-1] == pytest.approx(np.log(1.21))


def test_cross_sectional_rank_known_values():
    w = pd.DataFrame({"A": [1.0, 5.0], "B": [3.0, np.nan], "C": [2.0, 4.0]})
    out = cross_sectional_rank(w)
    # row 0: ranks 1, 3, 2 over n = 3 -> (r - 0.5) / 3
    assert out.iloc[0].tolist() == pytest.approx([0.5 / 3, 2.5 / 3, 1.5 / 3])
    # row 1: two assets, ranks 2 and 1 over n = 2
    assert out.iloc[1]["A"] == pytest.approx(0.75)
    assert out.iloc[1]["C"] == pytest.approx(0.25)
    assert np.isnan(out.iloc[1]["B"])


def test_raw_momentum_is_divided_by_trailing_vol():
    p = wide(["A"], n=30)
    raw = raw_features(p, TINY)
    sigma = trailing_vol(p, TINY["vol"])
    expected = lagged_log_return(p, near=0, far=TINY["mom_med"]) / sigma
    assert np.allclose(raw["mom_3m"]["A"].dropna().values, expected["A"].dropna().values)


def test_vol_feature_is_not_divided_by_itself():
    p = wide(["A"], n=30)
    raw = raw_features(p, TINY)
    assert np.allclose(raw["vol_60"]["A"].dropna().values, trailing_vol(p, TINY["vol"])["A"].dropna().values)


def test_features_are_ranks_strictly_between_zero_and_one():
    p = wide(list("ABCDEF"), n=40)
    f = compute_features(p, TINY, min_assets=3)
    assert list(f.columns) == FEATURE_NAMES
    assert ((f > 0) & (f < 1)).all().all()
    assert list(f.index.names) == ["date", "asset"]


def test_each_date_has_ranks_spread_evenly_over_the_eligible_assets():
    p = wide(list("ABCDEF"), n=40)
    f = compute_features(p, TINY, min_assets=3)
    one_date = f.xs(f.index.get_level_values("date")[-1], level="date")
    assert sorted(one_date["mom_3m"].tolist()) == pytest.approx(
        [(i + 0.5) / 6 for i in range(6)])


def test_no_lookahead_changing_later_prices_leaves_earlier_features_alone():
    p = wide(list("ABCDEF"), n=40)
    base = compute_features(p, TINY, min_assets=3)
    changed_prices = p.copy()
    cut = p.index[30]
    changed_prices.loc[changed_prices.index > cut] *= 1.7
    changed = compute_features(changed_prices, TINY, min_assets=3)
    early = base.index.get_level_values("date") <= cut
    early_changed = changed.index.get_level_values("date") <= cut
    pd.testing.assert_frame_equal(base[early], changed[early_changed])


def test_late_starting_etf_only_appears_once_it_has_enough_history():
    p = wide(list("ABCDE"), n=40)
    p.loc[p.index[:15], "E"] = np.nan  # E launches on day 15
    f = compute_features(p, TINY, min_assets=3)
    e_dates = f.xs("E", level="asset").index
    # needs trend window 5 prices and vol 3 returns and mom_long 6 -> first usable date is day 15 + 6
    assert e_dates.min() == p.index[15 + TINY["mom_long"]]
    # and E never distorts the ranks of earlier dates
    early = f.xs(p.index[16], level="date")
    assert "E" not in early.index


def test_dates_with_too_few_eligible_etfs_are_dropped():
    p = wide(list("ABCDE"), n=40)
    p.loc[p.index[:25], ["C", "D", "E"]] = np.nan  # only A and B early on
    f = compute_features(p, TINY, min_assets=4)
    assert f.index.get_level_values("date").min() >= p.index[25]


def test_flat_price_etf_is_left_out_not_given_an_infinite_feature():
    p = wide(list("ABCDE"), n=40)
    p["E"] = 100.0  # zero volatility
    f = compute_features(p, TINY, min_assets=3)
    assert "E" not in f.index.get_level_values("asset")
    assert np.isfinite(f.values).all()

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
