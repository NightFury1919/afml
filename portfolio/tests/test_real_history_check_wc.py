import numpy as np
import pandas as pd
import pytest

from etf_features import FEATURE_NAMES, compute_features
from etf_labels import make_labels
from real_history_check import MAX_VOL_TILT
from real_history_check_wc import (active_series_within_class, block_bootstrap_ir_ci, build_wc_dataset,
                                   decide_wc, evaluate_wc, oof_with_features, save_results_wc,
                                   within_class_tilt)

SMALL = dict(n_splits=4, embargo_dates=21, min_assets=5)
DATE = pd.Timestamp("2020-01-02")


def one_date(rows):
    """rows: list of (asset, class, pred, fwd). Returns pred, fwd, groups."""
    idx = pd.MultiIndex.from_tuples([(DATE, r[0]) for r in rows], names=["date", "asset"])
    return (pd.Series([r[2] for r in rows], index=idx, dtype=float),
            pd.Series([r[3] for r in rows], index=idx, dtype=float),
            {r[0]: r[1] for r in rows})


def class_prices(n_dates=1500, n_classes=4, per_class=6, seed=0, k=0.2):
    """Volatility rises with the ETF number inside each class, and drift rises with volatility."""
    rng = np.random.default_rng(seed)
    n = n_classes * per_class
    sig = np.tile(np.linspace(0.005, 0.02, per_class), n_classes)
    idx = pd.bdate_range("2012-01-02", periods=n_dates)
    rets = rng.normal(0, 1, (n_dates, n)) * sig + k * sig
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=[f"E{i:02d}" for i in range(n)])
    px.index.name, px.columns.name = "date", "asset"
    groups = {f"E{i:02d}": f"C{i // per_class}" for i in range(n)}
    return px, groups


# ------------------------------------------- class-neutral active return
def test_active_return_known_value_equal_class_sizes():
    rows = [(f"X{i}", "X", i, f) for i, f in enumerate([0.01, 0.02, 0.03, 0.04, 0.05], 1)]
    rows += [(f"Y{i}", "Y", i, f) for i, f in enumerate([0.0, 0.0, 0.0, 0.0, 0.10], 1)]
    pred, fwd, groups = one_date(rows)
    act = active_series_within_class(pred, fwd, groups, horizon=1)
    # X: top pick 0.05 minus class mean 0.03 = 0.02.  Y: 0.10 minus 0.02 = 0.08.  Equal sizes: 0.5 each.
    assert act.iloc[0] == pytest.approx(0.5 * 0.02 + 0.5 * 0.08)


def test_active_return_weights_classes_by_their_size_and_picks_a_rounded_up_fifth():
    rows = [(f"X{i}", "X", i, f) for i, f in enumerate([0.01, 0.02, 0.03, 0.04, 0.05], 1)]      # 5 ETFs: 1 pick
    rows += [(f"Y{i}", "Y", i, 0.12 if i > 12 else 0.0) for i in range(1, 16)]                  # 15 ETFs: 3 picks
    pred, fwd, groups = one_date(rows)
    act = active_series_within_class(pred, fwd, groups, horizon=1)
    # X diff 0.02 (weight 5/20). Y: 3 picks at 0.12, class mean 0.024, diff 0.096 (weight 15/20).
    assert act.iloc[0] == pytest.approx(0.25 * 0.02 + 0.75 * 0.096)


def test_a_class_wide_return_difference_does_not_change_the_active_return():
    rows = [(f"X{i}", "X", i, 0.01 * i) for i in range(1, 6)] + [(f"Y{i}", "Y", i, 0.02 * i) for i in range(1, 6)]
    pred, fwd, groups = one_date(rows)
    shifted = fwd.copy()
    shifted[[a for a in fwd.index if a[1].startswith("X")]] += 0.10                          # class X beats class Y
    a = active_series_within_class(pred, fwd, groups, horizon=1)
    b = active_series_within_class(pred, shifted, groups, horizon=1)
    assert a.iloc[0] == pytest.approx(b.iloc[0])


def test_a_class_of_six_picks_two_not_one():
    rows = [(f"X{i}", "X", i, {6: 0.12, 5: 0.06}.get(i, 0.0)) for i in range(1, 7)]
    pred, fwd, groups = one_date(rows)
    act = active_series_within_class(pred, fwd, groups, horizon=1)
    assert act.iloc[0] == pytest.approx(0.09 - 0.03)          # picks 0.12 and 0.06 (mean 0.09), class mean 0.03


def test_float_noise_does_not_add_a_pick():
    rows = [(f"X{i}", "X", i, 0.1 if i > 18 else 0.0) for i in range(1, 26)]
    pred, fwd, groups = one_date(rows)
    act = active_series_within_class(pred, fwd, groups, horizon=1, top_share=0.28)    # 0.28 * 25 = 7.000000000000001
    assert act.iloc[0] == pytest.approx(0.1 - 0.028)          # 7 picks, not 8


def test_only_every_horizon_th_date_is_used():
    idx = pd.MultiIndex.from_product([pd.bdate_range("2020-01-01", periods=10), ["A", "B", "C"]], names=["date", "asset"])
    pred = pd.Series(np.tile([3.0, 2.0, 1.0], 10), index=idx)
    fwd = pd.Series(np.tile([0.03, 0.0, 0.0], 10), index=idx)
    groups = {"A": "K", "B": "K", "C": "K"}
    assert len(active_series_within_class(pred, fwd, groups, horizon=4)) == 3                  # dates 0, 4, 8


# ----------------------------------------------------------- tilt statistic
def test_within_class_tilt_is_plus_one_for_a_volatility_sort_and_minus_one_for_its_reverse():
    px, groups = class_prices(300, 3, 5)
    idx = pd.MultiIndex.from_product([px.index[:50], px.columns], names=["date", "asset"])
    vol = pd.Series(np.tile(np.tile(np.arange(5) / 5 + 0.1, 3), 50), index=idx)
    assert within_class_tilt(vol, vol, groups) == pytest.approx(1.0)
    assert within_class_tilt(-vol, vol, groups) == pytest.approx(-1.0)


def test_tilt_is_measured_inside_each_class_not_across_classes():
    px, groups = class_prices(300, 3, 5)
    idx = pd.MultiIndex.from_product([px.index[:50], px.columns], names=["date", "asset"])
    class_no = pd.Series([int(groups[a][1]) for a in idx.get_level_values("asset")], index=idx)
    within = pd.Series(np.tile(np.tile(np.arange(5) / 5 + 0.1, 3), 50), index=idx)
    vol = within + class_no * 10.0                            # volatile classes AND volatile ETFs inside a class
    rng = np.random.default_rng(0)
    pred = class_no * 10.0 + pd.Series(rng.normal(size=len(idx)), index=idx)   # sorts classes, random inside them
    assert abs(within_class_tilt(pred, vol, groups)) < 0.15   # a pooled correlation would be near +1


# ----------------------------------------------------------------- dataset
def test_dataset_uses_class_ranks_and_class_median_labels():
    px, groups = class_prices(900, 4, 6)
    ds = build_wc_dataset(px, groups, min_assets=5, min_group_assets=3)
    feats = compute_features(px, min_assets=5, groups=groups, min_group_assets=3)
    labels = make_labels(px, feats.index, 21, 5, groups, 3)
    pd.testing.assert_frame_equal(ds[FEATURE_NAMES], feats.reindex(labels.index)[FEATURE_NAMES])
    pd.testing.assert_series_equal(ds["label"], labels["label"])
    cls = ds.index.get_level_values("asset").map(groups)
    date = ds.index.get_level_values("date")
    ones = ds["label"].groupby([date, cls]).sum()
    size = ds["label"].groupby([date, cls]).size()
    assert (ones <= size // 2).all()                          # at most the half above each class median


def test_the_pooled_dataset_would_differ_which_proves_groups_are_used():
    px, groups = class_prices(900, 4, 6)
    ds = build_wc_dataset(px, groups, min_assets=5, min_group_assets=3)
    pooled = build_wc_dataset(px, {a: "ALL" for a in groups}, min_assets=5, min_group_assets=3)
    assert not ds["label"].equals(pooled["label"].reindex(ds.index))


# ----------------------------------------------------------------- the arms
def test_the_no_volatility_arm_ignores_the_volatility_feature():
    px, groups = class_prices(1100, 3, 6, seed=2)
    ds = build_wc_dataset(px, groups, min_assets=5, min_group_assets=3)
    feats = [f for f in FEATURE_NAMES if f != "vol_60"]
    base = oof_with_features(ds, feats, n_splits=4, embargo_dates=21)
    scrambled = ds.copy()
    scrambled["vol_60"] = np.random.default_rng(0).permutation(scrambled["vol_60"].to_numpy())
    pd.testing.assert_series_equal(base, oof_with_features(scrambled, feats, n_splits=4, embargo_dates=21))
    with_vol = oof_with_features(ds, FEATURE_NAMES, n_splits=4, embargo_dates=21)
    with_vol_scr = oof_with_features(scrambled, FEATURE_NAMES, n_splits=4, embargo_dates=21)
    assert not with_vol.equals(with_vol_scr)                  # the full model does react to it


def test_evaluate_gives_the_no_vol_arm_four_features_and_the_full_arm_five(monkeypatch):
    import real_history_check_wc as m
    seen = []
    real = m.oof_with_features

    def spy(ds, features, *a, **k):
        seen.append(list(features))
        return real(ds, features, *a, **k)
    monkeypatch.setattr(m, "oof_with_features", spy)
    px, groups = class_prices(1300, 3, 6)
    evaluate_wc(px, groups, **SMALL, min_group_assets=3, n_boot=20)
    assert sorted(len(f) for f in seen) == [4, 5]
    assert all("vol_60" not in f for f in seen if len(f) == 4)
    assert all("vol_60" in f for f in seen if len(f) == 5)


def test_standalone_ir_interval_excludes_zero_for_a_steady_edge_and_straddles_it_for_noise():
    rng = np.random.default_rng(0)
    good = pd.Series(rng.normal(0.02, 0.03, 300))
    lo, hi = block_bootstrap_ir_ci(good, horizon=21, block=4, n_boot=500, seed=1)
    assert lo > 0 and hi > lo
    noise = pd.Series(rng.normal(0.0, 0.03, 300))
    lo, hi = block_bootstrap_ir_ci(noise, horizon=21, block=4, n_boot=500, seed=1)
    assert lo < 0 < hi
    assert block_bootstrap_ir_ci(good, 21, 4, 500, 1) == block_bootstrap_ir_ci(good, 21, 4, 500, 1)


def test_evaluate_runs_with_three_arms_and_the_volatility_rule_is_a_pure_volatility_sort():
    px, groups = class_prices(1500, 4, 6, k=0.3)
    res = evaluate_wc(px, groups, **SMALL, min_group_assets=3, n_boot=200)
    assert set(res["arms"]) == {"wc_current", "wc_no_vol", "wc_vol_rule"}
    assert res["arms"]["wc_vol_rule"]["vol_tilt"] == pytest.approx(1.0)
    assert res["arms"]["wc_vol_rule"]["mean_ic"] > 0.1           # inside each class volatility predicts the excess return
    assert res["arms"]["wc_vol_rule"]["ir"] > 0.5                # volatility really pays inside each class here
    assert set(res["paired"]) == {"wc_current-wc_vol_rule", "wc_no_vol-wc_vol_rule", "wc_no_vol-wc_current"}
    assert set(res["standalone"]) == set(res["arms"])
    assert len(res["active"]) > 20


# ---------------------------------------------------------------- decision
def fake(lo_cur, tilt_cur, sa_cur, lo_nv, tilt_nv, sa_nv):
    return dict(arms={"wc_current": dict(vol_tilt=tilt_cur), "wc_no_vol": dict(vol_tilt=tilt_nv),
                      "wc_vol_rule": dict(vol_tilt=1.0)},
                paired={"wc_current-wc_vol_rule": (lo_cur, 1.0), "wc_no_vol-wc_vol_rule": (lo_nv, 1.0),
                        "wc_no_vol-wc_current": (-1.0, 1.0)},
                standalone={"wc_current": (sa_cur, 2.0), "wc_no_vol": (sa_nv, 2.0), "wc_vol_rule": (0.0, 1.0)})


def test_an_edge_needs_all_three_conditions():
    d = decide_wc(fake(0.1, 0.5, 0.1, 0.1, 0.5, -0.1))
    assert d["wc_current_edge_found"] is True and d["wc_no_vol_edge_found"] is False         # no_vol fails standalone
    assert decide_wc(fake(-0.1, 0.5, 0.1, 0, 0, 0))["wc_current_edge_found"] is False         # does not beat vol rule
    assert decide_wc(fake(0.1, 0.9, 0.1, 0, 0, 0))["wc_current_edge_found"] is False         # still a vol sort
    assert decide_wc(fake(0.1, 0.5, -0.1, 0, 0, 0))["wc_current_edge_found"] is False         # not positive alone
    assert MAX_VOL_TILT == 0.7


def test_the_decision_reports_each_part_separately():
    d = decide_wc(fake(0.1, 0.9, 0.1, 0.1, 0.2, 0.1))
    assert d["wc_current_adds_beyond_volatility"] is False and d["wc_no_vol_adds_beyond_volatility"] is True
    assert d["wc_current_positive_alone"] is True and d["wc_no_vol_positive_alone"] is True


# --------------------------------------------------------------------- once
def test_save_results_refuses_to_overwrite(tmp_path):
    px, groups = class_prices(1300, 4, 6)
    res = evaluate_wc(px, groups, **SMALL, min_group_assets=3, n_boot=50)
    save_results_wc(res, tmp_path)
    assert (tmp_path / "real_history_check_wc_results.csv").exists()
    assert (tmp_path / "real_history_check_wc_active_returns.csv").exists()
    with pytest.raises(FileExistsError):
        save_results_wc(res, tmp_path)
