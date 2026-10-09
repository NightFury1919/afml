import numpy as np
import pandas as pd
import pytest

from etf_features import FEATURE_NAMES
from panel_data import wide_to_long
from positive_control_etf import top_quintile_active_ir
from real_history_check import (CI_LEVEL, MAX_VOL_TILT, MIN_TILT_REDUCTION, active_series, annualized_ir,
                                build_arm_datasets, daily_vol_panel, decide, evaluate,
                                paired_block_bootstrap_ci, save_results, vol_scaled_labels, vol_tilt)

SMALL = dict(n_splits=4, embargo_dates=21, min_assets=5)


def vol_driven_prices(n_dates=1500, n_assets=20, seed=0, k=0.2):
    """Higher-volatility ETFs also have higher drift, so 'hold the most volatile' really works."""
    rng = np.random.default_rng(seed)
    sig = np.linspace(0.005, 0.02, n_assets)
    idx = pd.bdate_range("2012-01-02", periods=n_dates)
    rets = rng.normal(0, 1, (n_dates, n_assets)) * sig + k * sig
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                      columns=[f"E{i:02d}" for i in range(n_assets)])
    px.index.name, px.columns.name = "date", "asset"
    return px


# --------------------------------------------------------------- daily vol
def test_daily_vol_is_an_exponentially_weighted_std_of_one_day_returns():
    idx = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(0)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (30, 2)), axis=0)), index=idx, columns=["A", "B"])
    got = daily_vol_panel(px, span0=5)
    want = px.pct_change().ewm(span=5, min_periods=5).std()
    pd.testing.assert_frame_equal(got, want)
    assert got.iloc[:5].isna().all().all() and got.iloc[5:].notna().all().all()


def test_daily_vol_reacts_on_the_day_of_a_jump_not_two_days_later():
    idx = pd.bdate_range("2020-01-01", periods=40)
    p = pd.Series(100.0, index=idx)
    p.iloc[30:] = 110.0                                     # a one-day jump on bar 30
    v = daily_vol_panel(p.to_frame("A"), span0=10)["A"]
    assert v.iloc[29] == 0.0 and v.iloc[30] > 0.0


def test_daily_vol_uses_only_data_up_to_each_date():
    px = vol_driven_prices(300, 3)
    full = daily_vol_panel(px)
    short = daily_vol_panel(px.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], short)


# ------------------------------------------------------- volatility-scaled label
def test_scaling_by_volatility_flips_the_label_of_the_volatile_etf():
    # One date with three ETFs. Raw 21-day returns +10%, +6%, +2%; daily vol 4%, 1%, 1%.
    h = 21
    idx = pd.bdate_range("2020-01-01", periods=h + 1)
    px = pd.DataFrame({"A": 100.0, "B": 100.0, "C": 100.0}, index=idx)
    px.iloc[-1] = [110.0, 106.0, 102.0]
    px.index.name = "date"
    vol = pd.DataFrame({"A": 0.04, "B": 0.01, "C": 0.01}, index=idx)
    index = pd.MultiIndex.from_product([[idx[0]], ["A", "B", "C"]], names=["date", "asset"])
    out = vol_scaled_labels(px, index, vol, horizon=h, min_assets=3)
    raw = {a: (np.log(px[a].iloc[-1] / 100.0)) for a in "ABC"}
    # raw: A beats the median (B); scaled: A's return per unit of risk is below B's, so A loses
    assert out.loc[(idx[0], "A"), "raw_label"] == 1 and out.loc[(idx[0], "A"), "label"] == 0
    assert out.loc[(idx[0], "B"), "label"] == 1
    assert out.loc[(idx[0], "C"), "label"] == 0
    assert out["t1"].eq(idx[-1]).all()


def test_rows_without_a_volatility_estimate_are_dropped_and_dates_stay_balanced():
    h = 21
    idx = pd.bdate_range("2020-01-01", periods=h + 1)
    px = pd.DataFrame({k: 100.0 for k in "ABCD"}, index=idx)
    px.iloc[-1] = [110.0, 106.0, 102.0, 104.0]
    px.index.name = "date"
    vol = pd.DataFrame(0.01, index=idx, columns=list("ABCD"))
    vol["D"] = np.nan
    index = pd.MultiIndex.from_product([[idx[0]], list("ABCD")], names=["date", "asset"])
    out = vol_scaled_labels(px, index, vol, horizon=h, min_assets=3)
    assert sorted(out.index.get_level_values("asset")) == ["A", "B", "C"]
    short = vol_scaled_labels(px, index, vol, horizon=h, min_assets=4)
    assert short.empty


def test_scaled_labels_never_use_future_volatility():
    px = vol_driven_prices(700, 8)
    vol = daily_vol_panel(px)
    index = wide_to_long(px, "x").index
    a = vol_scaled_labels(px, index, vol, horizon=21, min_assets=5)
    vol2 = vol.copy()
    vol2.iloc[-30:] = vol2.iloc[-30:] * 5                    # change volatility only in the last 30 days
    b = vol_scaled_labels(px, index, vol2, horizon=21, min_assets=5)
    early = a.index.get_level_values("date") < px.index[-30]
    pd.testing.assert_frame_equal(a[early], b[b.index.get_level_values("date") < px.index[-30]])


# ---------------------------------------------------------------- statistics
def test_active_series_matches_the_positive_control_top_quintile_ir():
    px = vol_driven_prices(900, 10, seed=3)
    idx = wide_to_long(px, "x").index
    rng = np.random.default_rng(1)
    pred = pd.Series(rng.random(len(idx)), index=idx)
    fwd = wide_to_long(np.log(px).shift(-21) - np.log(px), "f").reindex(idx).dropna()
    pred = pred.reindex(fwd.index)
    act = active_series(pred, fwd, 21)
    ir_ref, mean_ref, n_ref = top_quintile_active_ir(pred, fwd, 21)
    assert annualized_ir(act, 21) == pytest.approx(ir_ref)
    assert act.mean() == pytest.approx(mean_ref) and len(act) == n_ref


def test_vol_tilt_is_plus_one_for_a_vol_sort_and_minus_one_for_its_reverse():
    px = vol_driven_prices(500, 8)
    idx = wide_to_long(px, "x").index
    rank = pd.Series(np.tile(np.arange(8) / 8 + 0.05, 500), index=idx)
    assert vol_tilt(rank, rank) == pytest.approx(1.0)
    assert vol_tilt(-rank, rank) == pytest.approx(-1.0)


def test_bootstrap_ci_of_a_constant_advantage_excludes_zero_and_of_identical_series_straddles_it():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.002, 0.03, 200))
    ci_same = paired_block_bootstrap_ci(a, a, horizon=21, block=4, n_boot=500, seed=1)
    assert ci_same[0] == pytest.approx(0.0) and ci_same[1] == pytest.approx(0.0)
    b = a + 0.01                                                          # a steady edge of one point per period
    lo, hi = paired_block_bootstrap_ci(b, a, horizon=21, block=4, n_boot=500, seed=1)
    assert lo > 0
    again = paired_block_bootstrap_ci(b, a, horizon=21, block=4, n_boot=500, seed=1)
    assert (lo, hi) == again


# ------------------------------------------------------------- the three arms
def test_arms_share_features_and_rows_and_differ_only_in_the_label():
    px = vol_driven_prices(1300, 12)
    arms = build_arm_datasets(px, **{k: SMALL[k] for k in ("min_assets",)})
    a, b = arms["current"], arms["vol_scaled"]
    assert a.index.equals(b.index)
    pd.testing.assert_frame_equal(a[FEATURE_NAMES], b[FEATURE_NAMES])
    assert not a["label"].equals(b["label"])
    assert (a["t1"] == b["t1"]).all()


def test_evaluate_runs_and_the_volatility_rule_wins_when_volatility_really_pays():
    px = vol_driven_prices(1500, 20, k=0.3)
    res = evaluate(px, **SMALL, n_boot=200)
    assert set(res["arms"]) == {"current", "vol_scaled", "vol_rule"}
    assert res["arms"]["vol_rule"]["vol_tilt"] > 0.95
    assert res["arms"]["vol_rule"]["ir"] > 1.0
    assert res["arms"]["vol_rule"]["mean_vol_rank_of_picks"] > 0.85
    assert set(res["paired"]) == {"current-vol_rule", "vol_scaled-vol_rule", "vol_scaled-current"}
    assert len(res["active"]) > 20


# -------------------------------------------------------------- the decision
def fake(ir_gap_lo_a, tilt_a, ir_gap_lo_b, tilt_b):
    return dict(arms={"current": dict(vol_tilt=tilt_a), "vol_scaled": dict(vol_tilt=tilt_b), "vol_rule": dict(vol_tilt=1.0)},
                paired={"current-vol_rule": (ir_gap_lo_a, 1.0), "vol_scaled-vol_rule": (ir_gap_lo_b, 1.0),
                        "vol_scaled-current": (-1.0, 1.0)})


def test_decision_requires_both_a_clear_gain_over_the_volatility_rule_and_a_small_volatility_tilt():
    d = decide(fake(0.1, 0.5, -0.2, 0.2))
    assert d["current_adds_beyond_volatility"] is True and d["vol_scaled_adds_beyond_volatility"] is False
    assert decide(fake(0.1, 0.9, 0.1, 0.3))["current_adds_beyond_volatility"] is False      # gain but still a vol sort
    assert decide(fake(-0.1, 0.3, 0.1, 0.3))["current_adds_beyond_volatility"] is False     # CI includes zero


def test_decision_on_whether_scaling_removes_the_volatility_tilt():
    assert decide(fake(0, 0.9, 0, 0.4))["scaling_reduces_tilt"] is True                    # reduced by 0.5
    assert decide(fake(0, 0.9, 0, 0.5))["scaling_reduces_tilt"] is True                    # reduced by exactly 0.4
    assert decide(fake(0, 0.9, 0, 0.6))["scaling_reduces_tilt"] is False                  # reduced by only 0.3
    assert MAX_VOL_TILT == 0.7 and MIN_TILT_REDUCTION == 0.4 and CI_LEVEL == 0.90


# -------------------------------------------------------------------- run once
def test_save_results_refuses_to_overwrite(tmp_path):
    px = vol_driven_prices(1300, 12)
    res = evaluate(px, **SMALL, n_boot=50)
    save_results(res, tmp_path)
    assert (tmp_path / "real_history_check_results.csv").exists()
    assert (tmp_path / "real_history_check_active_returns.csv").exists()
    with pytest.raises(FileExistsError):
        save_results(res, tmp_path)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_real_history_check.py -v
#
# collected 14 items
#
# test_real_history_check.py::test_daily_vol_is_an_exponentially_weighted_std_of_one_day_returns PASSED [  7%]
# test_real_history_check.py::test_daily_vol_reacts_on_the_day_of_a_jump_not_two_days_later PASSED [ 14%]
# test_real_history_check.py::test_daily_vol_uses_only_data_up_to_each_date PASSED [ 21%]
# test_real_history_check.py::test_scaling_by_volatility_flips_the_label_of_the_volatile_etf PASSED [ 28%]
# test_real_history_check.py::test_rows_without_a_volatility_estimate_are_dropped_and_dates_stay_balanced PASSED [ 35%]
# test_real_history_check.py::test_scaled_labels_never_use_future_volatility PASSED [ 42%]
# test_real_history_check.py::test_active_series_matches_the_positive_control_top_quintile_ir PASSED [ 50%]
# test_real_history_check.py::test_vol_tilt_is_plus_one_for_a_vol_sort_and_minus_one_for_its_reverse PASSED [ 57%]
# test_real_history_check.py::test_bootstrap_ci_of_a_constant_advantage_excludes_zero_and_of_identical_series_straddles_it PASSED [ 64%]
# test_real_history_check.py::test_arms_share_features_and_rows_and_differ_only_in_the_label PASSED [ 71%]
# test_real_history_check.py::test_evaluate_runs_and_the_volatility_rule_wins_when_volatility_really_pays PASSED [ 78%]
# test_real_history_check.py::test_decision_requires_both_a_clear_gain_over_the_volatility_rule_and_a_small_volatility_tilt PASSED [ 85%]
# test_real_history_check.py::test_decision_on_whether_scaling_removes_the_volatility_tilt PASSED [ 92%]
# test_real_history_check.py::test_save_results_refuses_to_overwrite PASSED [100%]
#
# 14 passed in 2.54s
# Mutation checks (sandbox): two-bar returns, no volatility scaling in the label, decision using the
#   upper interval bound, tilt check removed, unpaired bootstrap, overwrite allowed, future volatility,
#   and overlapping periods each made tests fail; original restored.
# Sandbox calibration (synthetic no-edge worlds, not the real data): ir spread about 0.3 per arm; tilt
#   difference between label designs sd 0.20 over 16 worlds, which set MIN_TILT_REDUCTION = 0.4.
# Full-size synthetic run (48 ETFs x 4,700 dates) took about 5 seconds.
# NOT yet run on the real snapshot. Pre-register first (preregistration_real_history_check.md).
# ---------------------------------------------------------------------------
