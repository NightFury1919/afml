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



# ------------------------------------------------------------ within-group labels
def prices6():
    df = pd.DataFrame({
        "A": [100, 100, 110, 120, 130], "B": [100, 100, 100, 100, 100], "C": [100, 100, 90, 80, 70],
        "D": [100, 100, 120, 130, 140], "E": [100, 100, 105, 105, 105], "F": [100, 100, 95, 90, 85],
    }, index=D, dtype=float)
    df.index.name, df.columns.name = "date", "asset"
    return df


def test_labels_use_the_median_of_the_etfs_own_group():
    p = prices6()
    groups = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2", "F": "g2"}
    out = make_labels(p, full_index(p), horizon=2, min_assets=3, groups=groups, min_group_assets=3)
    day0 = out.xs(D[0], level="date")
    # g1 returns over 2 days: A +9.53%, B 0, C -10.5% -> median is B's 0.0; g2: D +18.2%, E +4.9%, F -5.1% -> median E's 4.9%
    assert day0.loc["A", "excess_ret"] == pytest.approx(np.log(1.10))
    assert day0.loc["B", "excess_ret"] == pytest.approx(0.0)
    assert day0.loc["D", "excess_ret"] == pytest.approx(np.log(1.20) - np.log(1.05))
    assert day0.loc["F", "excess_ret"] == pytest.approx(np.log(0.95) - np.log(1.05))
    assert day0["label"].to_dict() == {"A": 1, "B": 0, "C": 0, "D": 1, "E": 0, "F": 0}


def test_pooled_and_grouped_labels_differ_when_groups_move_differently():
    p = prices6()
    groups = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2", "F": "g2"}
    pooled = make_labels(p, full_index(p), horizon=2, min_assets=3)
    grouped = make_labels(p, full_index(p), horizon=2, min_assets=3, groups=groups, min_group_assets=3)
    d0 = D[0]
    assert pooled.xs(d0, level="date")["label"].to_dict() != grouped.xs(d0, level="date")["label"].to_dict()


def test_a_group_with_too_few_labeled_etfs_is_dropped_on_that_date():
    p = prices6()
    groups = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2", "F": "g2"}
    idx = full_index(p).drop([(D[0], "F")])             # g2 has only D and E on d0
    out = make_labels(p, idx, horizon=2, min_assets=3, groups=groups, min_group_assets=3)
    assert set(out.xs(D[0], level="date").index) == {"A", "B", "C"}
    assert set(out.xs(D[1], level="date").index) == set("ABCDEF")


def test_labels_without_groups_are_unchanged_by_the_new_parameters():
    p = prices4()
    a = make_labels(p, full_index(p), horizon=2, min_assets=4)
    b = make_labels(p, full_index(p), horizon=2, min_assets=4, groups=None)
    pd.testing.assert_frame_equal(a, b)


def test_build_dataset_with_groups_runs_end_to_end_and_keeps_the_same_columns():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=80)
    cols = list("ABCDEF")
    p = pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))) for c in cols}, index=idx)
    p.index.name, p.columns.name = "date", "asset"
    tiny = {"mom_long": 6, "mom_skip": 2, "mom_med": 4, "ret_short": 2, "vol": 3, "trend": 5}
    groups = {c: ("g1" if c in "ABC" else "g2") for c in cols}
    ds = build_dataset(p, horizon=5, windows=tiny, min_assets=3, groups=groups, min_group_assets=3)
    assert {"mom_12_1", "mom_3m", "ret_5d", "vol_60", "trend_200", "fwd_ret", "excess_ret", "label", "t1"} <= set(ds.columns)
    assert not ds.isna().any().any()
    # within each group and date the label balance is about half
    share = ds.assign(g=ds.index.get_level_values("asset").map(groups)).groupby(
        [ds.index.get_level_values("date"), "g"])["label"].mean()
    assert share.between(0.0, 0.67).all()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_etf_labels.py::test_default_horizon_is_21_trading_days PASSED       [ 75%]
# test_etf_labels.py::test_forward_return_known_values PASSED              [ 76%]
# test_etf_labels.py::test_labels_known_values_balanced_case PASSED        [ 77%]
# test_etf_labels.py::test_ties_with_the_median_get_label_zero PASSED      [ 78%]
# test_etf_labels.py::test_last_dates_without_a_forward_return_are_excluded PASSED [ 79%]
# test_etf_labels.py::test_t1_is_the_date_h_trading_days_ahead PASSED      [ 80%]
# test_etf_labels.py::test_median_uses_only_the_assets_in_the_index PASSED [ 81%]
# test_etf_labels.py::test_dates_with_too_few_assets_are_dropped PASSED    [ 82%]
# test_etf_labels.py::test_asset_with_missing_future_price_gets_no_label PASSED [ 84%]
# test_etf_labels.py::test_label_uses_future_prices_but_features_index_is_all_that_decides_the_universe PASSED [ 85%]
# test_etf_labels.py::test_build_dataset_joins_features_and_labels_on_the_same_rows PASSED [ 86%]
# test_etf_labels.py::test_labels_use_the_median_of_the_etfs_own_group PASSED [ 87%]
# test_etf_labels.py::test_pooled_and_grouped_labels_differ_when_groups_move_differently PASSED [ 88%]
# test_etf_labels.py::test_a_group_with_too_few_labeled_etfs_is_dropped_on_that_date PASSED [ 89%]
# test_etf_labels.py::test_labels_without_groups_are_unchanged_by_the_new_parameters PASSED [ 90%]
# test_etf_labels.py::test_build_dataset_with_groups_runs_end_to_end_and_keeps_the_same_columns PASSED [ 92%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): using the pooled median when groups are given made tests fail; original restored.
# ---------------------------------------------------------------------------
