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



# ------------------------------------------------------------ within-group ranking
def test_cross_sectional_rank_within_groups_known_values():
    w = pd.DataFrame({"A": [3.0], "B": [1.0], "C": [2.0], "D": [10.0], "E": [20.0]})
    groups = {"A": "g1", "B": "g1", "C": "g1", "D": "g2", "E": "g2"}
    out = cross_sectional_rank(w, groups)
    assert out.iloc[0][["A", "B", "C"]].tolist() == pytest.approx([2.5 / 3, 0.5 / 3, 1.5 / 3])
    assert out.iloc[0][["D", "E"]].tolist() == pytest.approx([0.25, 0.75])


def test_grouped_rank_with_groups_none_equals_the_pooled_rank():
    w = pd.DataFrame({"A": [3.0, 1.0], "B": [1.0, 2.0], "C": [2.0, 3.0]})
    pd.testing.assert_frame_equal(cross_sectional_rank(w), cross_sectional_rank(w, None))


def test_grouped_features_rank_inside_each_group_and_each_group_is_evenly_spread():
    p = wide(list("ABCDEFGH"), n=60)
    groups = {c: ("g1" if c in "ABCD" else "g2") for c in "ABCDEFGH"}
    f = compute_features(p, TINY, min_assets=4, groups=groups, min_group_assets=3)
    last = f.xs(f.index.get_level_values("date")[-1], level="date")
    for g, members in (("g1", list("ABCD")), ("g2", list("EFGH"))):
        vals = sorted(last.loc[members, "mom_3m"].tolist())
        assert vals == pytest.approx([(i + 0.5) / 4 for i in range(4)])


def test_a_group_with_too_few_eligible_etfs_is_dropped_that_day():
    p = wide(list("ABCDEF"), n=60)
    p.loc[p.index[:30], ["E", "F"]] = np.nan          # g2 = {D, E, F} has only D for a while
    groups = {c: ("g1" if c in "ABC" else "g2") for c in "ABCDEF"}
    f = compute_features(p, TINY, min_assets=3, groups=groups, min_group_assets=3)
    early = f.xs(p.index[32], level="date")
    assert set(early.index) == {"A", "B", "C"}          # D alone is not a ranking, so it is left out
    late = f.xs(p.index[-1], level="date")
    assert set(late.index) == set("ABCDEF")


def test_every_priced_etf_must_have_a_group():
    p = wide(list("ABCD"), n=40)
    with pytest.raises(ValueError):
        compute_features(p, TINY, min_assets=2, groups={"A": "g", "B": "g", "C": "g"}, min_group_assets=2)


def test_default_behaviour_without_groups_is_unchanged():
    p = wide(list("ABCDEF"), n=40)
    a = compute_features(p, TINY, min_assets=3)
    b = compute_features(p, TINY, min_assets=3, groups=None)
    pd.testing.assert_frame_equal(a, b)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_etf_features.py::test_default_windows_are_the_preregistered_values PASSED [ 53%]
# test_etf_features.py::test_trailing_vol_known_value PASSED               [ 54%]
# test_etf_features.py::test_trend_distance_known_value PASSED             [ 55%]
# test_etf_features.py::test_lagged_log_return_known_values PASSED         [ 56%]
# test_etf_features.py::test_cross_sectional_rank_known_values PASSED      [ 57%]
# test_etf_features.py::test_raw_momentum_is_divided_by_trailing_vol PASSED [ 59%]
# test_etf_features.py::test_vol_feature_is_not_divided_by_itself PASSED   [ 60%]
# test_etf_features.py::test_features_are_ranks_strictly_between_zero_and_one PASSED [ 61%]
# test_etf_features.py::test_each_date_has_ranks_spread_evenly_over_the_eligible_assets PASSED [ 62%]
# test_etf_features.py::test_no_lookahead_changing_later_prices_leaves_earlier_features_alone PASSED [ 63%]
# test_etf_features.py::test_late_starting_etf_only_appears_once_it_has_enough_history PASSED [ 64%]
# test_etf_features.py::test_dates_with_too_few_eligible_etfs_are_dropped PASSED [ 65%]
# test_etf_features.py::test_flat_price_etf_is_left_out_not_given_an_infinite_feature PASSED [ 67%]
# test_etf_features.py::test_cross_sectional_rank_within_groups_known_values PASSED [ 68%]
# test_etf_features.py::test_grouped_rank_with_groups_none_equals_the_pooled_rank PASSED [ 69%]
# test_etf_features.py::test_grouped_features_rank_inside_each_group_and_each_group_is_evenly_spread PASSED [ 70%]
# test_etf_features.py::test_a_group_with_too_few_eligible_etfs_is_dropped_that_day PASSED [ 71%]
# test_etf_features.py::test_every_priced_etf_must_have_a_group PASSED     [ 72%]
# test_etf_features.py::test_default_behaviour_without_groups_is_unchanged PASSED [ 73%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): ignoring groups in the grouped ranking, and not dropping small groups, each made tests fail; original restored.
# ---------------------------------------------------------------------------
