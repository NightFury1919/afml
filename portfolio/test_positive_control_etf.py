import numpy as np
import pandas as pd
import pytest

from positive_control_etf import (
    analyze,
    block_bootstrap_returns,
    date_ics,
    demean_returns,
    ic_tstat,
    one_replicate,
    out_of_fold_predictions,
    plant_signal,
    planted_score,
    returns_to_prices,
    run_pipeline,
    top_quintile_active_ir,
)

SMALL = dict(n_splits=4, embargo_dates=21, min_assets=5)


def noise_returns(n_dates=1500, n_assets=20, seed=0, sd=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n_dates)
    df = pd.DataFrame(rng.normal(0, sd, (n_dates, n_assets)), index=idx,
                      columns=[f"E{i:02d}" for i in range(n_assets)])
    df.index.name, df.columns.name = "date", "asset"
    return df


# ---------------------------------------------------------------- bootstrap
def test_block_bootstrap_keeps_shape_index_and_uses_only_original_rows():
    r = pd.DataFrame({"A": np.arange(100.0), "B": np.arange(100.0) * 2},
                     index=pd.bdate_range("2020-01-01", periods=100))
    out = block_bootstrap_returns(r, block_length=10, rng=np.random.default_rng(0))
    assert out.shape == r.shape and out.index.equals(r.index)
    assert set(out["A"]).issubset(set(r["A"]))
    assert (out["B"] == out["A"] * 2).all()          # whole cross-sections move together


def test_block_bootstrap_blocks_are_consecutive_runs_of_the_original():
    r = pd.DataFrame({"A": np.arange(100.0)}, index=pd.bdate_range("2020-01-01", periods=100))
    out = block_bootstrap_returns(r, block_length=10, rng=np.random.default_rng(1))
    pos = out["A"].to_numpy()
    for b in range(10):
        block = pos[b * 10:(b + 1) * 10]
        assert (np.diff(block) == 1).all()


def test_block_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds():
    r = noise_returns(200, 3)
    a = block_bootstrap_returns(r, 20, np.random.default_rng(5))
    b = block_bootstrap_returns(r, 20, np.random.default_rng(5))
    c = block_bootstrap_returns(r, 20, np.random.default_rng(6))
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_returns_to_prices_known_values():
    r = pd.DataFrame({"A": [0.1, -0.1]})
    p = returns_to_prices(r)
    assert p["A"].iloc[0] == pytest.approx(100 * np.exp(0.1))
    assert p["A"].iloc[1] == pytest.approx(100.0)


# ------------------------------------------------------------------ planting
def test_planted_score_is_centered_and_ordered_by_momentum():
    idx = pd.bdate_range("2015-01-01", periods=400)
    common = np.random.default_rng(0).normal(0, 0.01, 400)
    drifts = [0.0000, 0.0003, 0.0006, 0.0009, 0.0012, 0.0015]
    prices = pd.DataFrame({f"E{k}": 100 * np.exp(np.cumsum(common + d)) for k, d in enumerate(drifts)}, index=idx)
    prices.index.name, prices.columns.name = "date", "asset"
    z = planted_score(prices, min_assets=3)
    last = z.iloc[-1]
    assert last.mean() == pytest.approx(0.0, abs=1e-12)
    assert last.is_monotonic_increasing                       # more drift, higher score
    assert 0.85 < last.std(ddof=0) < 1.0                       # about unit variance


def test_plant_signal_known_values():
    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.DataFrame({"A": [0.01, 0.03, 0.02], "B": [0.00, -0.01, 0.01]}, index=idx)
    z = pd.DataFrame({"A": [1.0, 2.0, np.nan], "B": [-1.0, -2.0, np.nan]}, index=idx)
    out = plant_signal(r, z, ic_horizon=0.2, horizon=4)      # a = 0.2 / sqrt(4) = 0.1
    sig = r.std()
    assert out.loc[idx[0]].tolist() == r.loc[idx[0]].tolist()                    # no score before day 0
    assert out.loc[idx[1], "A"] == pytest.approx(0.03 + 0.1 * sig["A"] * 1.0)    # uses yesterday's score
    assert out.loc[idx[1], "B"] == pytest.approx(-0.01 + 0.1 * sig["B"] * -1.0)
    assert out.loc[idx[2], "A"] == pytest.approx(0.02 + 0.1 * sig["A"] * 2.0)


def test_zero_ic_changes_nothing_and_missing_scores_mean_no_tilt():
    r = noise_returns(50, 3)
    z = pd.DataFrame(np.nan, index=r.index, columns=r.columns)
    pd.testing.assert_frame_equal(plant_signal(r, z, 0.1), r)
    z2 = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    pd.testing.assert_frame_equal(plant_signal(r, z2, 0.0), r)


# ------------------------------------------------------------ statistics
def test_date_ics_known_values():
    idx = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-01")], list("ABCD")], names=["date", "asset"])
    pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    same = pd.Series([1.0, 3.0, 2.0, 4.0], index=idx)       # ranks differ by one swap -> 1 - 6*2/(4*15) = 0.8
    assert date_ics(pred, same).iloc[0] == pytest.approx(0.8)
    assert date_ics(pred, pred * 10).iloc[0] == pytest.approx(1.0)
    assert date_ics(pred, -pred).iloc[0] == pytest.approx(-1.0)


def test_ic_tstat_takes_every_nth_date_and_matches_the_formula():
    s = pd.Series([0.1, 9.9, 0.3, 9.9, 0.2, 9.9, 0.4, 9.9], index=pd.bdate_range("2020-01-01", periods=8))
    t, mean, n = ic_tstat(s, spacing=2)
    kept = np.array([0.1, 0.3, 0.2, 0.4])
    assert n == 4 and mean == pytest.approx(kept.mean())
    assert t == pytest.approx(kept.mean() / (kept.std(ddof=1) / np.sqrt(4)))


def test_top_quintile_active_ir_known_values():
    d1, d2 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")
    idx = pd.MultiIndex.from_product([[d1, d2], list("ABCDE")], names=["date", "asset"])
    pred = pd.Series([0.9, 0.7, 0.5, 0.3, 0.1] * 2, index=idx)
    fwd = pd.Series([0.05, 0.01, 0.0, -0.02, -0.04, 0.04, 0.0, 0.0, 0.0, -0.04], index=idx)
    ir, mean_active, n = top_quintile_active_ir(pred, fwd, horizon=1)
    # top ETF is A each day: active = 0.05 - 0.0 and 0.04 - 0.0 (equal-weight means are 0.0 and 0.0)
    a = np.array([0.05, 0.04])
    assert n == 2 and mean_active == pytest.approx(a.mean())
    assert ir == pytest.approx(a.mean() / a.std(ddof=1) * np.sqrt(252))


# -------------------------------------------------------------- pipeline
def test_out_of_fold_predictions_cover_every_row_once_and_are_probabilities():
    from etf_labels import build_dataset
    prices = returns_to_prices(noise_returns(1100, 12, seed=3))
    ds = build_dataset(prices, 21, min_assets=5)
    pred = out_of_fold_predictions(ds, n_splits=4, embargo_dates=21)
    assert pred.index.equals(ds.index)
    assert pred.notna().all() and ((pred > 0) & (pred < 1)).all()


def test_pipeline_finds_a_strong_planted_signal():
    r = noise_returns(1500, 20, seed=1)
    score = planted_score(returns_to_prices(r), min_assets=5)
    prices = returns_to_prices(plant_signal(r, score, ic_horizon=0.3))
    res = run_pipeline(prices, oracle_score=score, **SMALL)
    assert res["mean_ic"] > 0.1 and res["t"] > 3
    assert res["oracle_ic"] > 0.15


def test_pipeline_finds_nothing_when_nothing_is_planted():
    r = noise_returns(1500, 20, seed=2)
    prices = returns_to_prices(r)
    res = run_pipeline(prices, oracle_score=planted_score(prices, min_assets=5), **SMALL)
    assert abs(res["t"]) < 3


def test_realized_oracle_ic_is_close_to_the_nominal_planted_ic():
    # The planted score is not perfectly persistent over 21 days (about 0.87 correlation),
    # so the realized IC runs a little below nominal. Averaged over all dates to keep noise low.
    from etf_labels import build_dataset
    from panel_data import wide_to_long
    r = noise_returns(4000, 30, seed=7)
    score = planted_score(returns_to_prices(r), min_assets=5)
    prices = returns_to_prices(plant_signal(r, score, ic_horizon=0.10))
    ds = build_dataset(prices, 21, min_assets=5)
    z = wide_to_long(score, "z").reindex(ds.index)
    realized = date_ics(z, ds["excess_ret"]).mean()
    assert 0.07 < realized < 0.12


def test_one_replicate_returns_one_row_per_ic_level_and_is_reproducible():
    r = noise_returns(1300, 12, seed=5)
    a = one_replicate(r, seed=11, ic_levels=[0.0, 0.1], block_length=63, **SMALL)
    b = one_replicate(r, seed=11, ic_levels=[0.0, 0.1], block_length=63, **SMALL)
    assert [row["ic_nominal"] for row in a] == [0.0, 0.1]
    assert [row["seed"] for row in a] == [11, 11]
    assert a == b


# -------------------------------------------------------------- analysis
def test_analyze_power_uses_the_null_95th_percentile_threshold():
    rows = []
    for k in range(100):                                    # null t-stats 0..99
        rows.append(dict(seed=k, ic_nominal=0.0, t=float(k), mean_ic=0.0, oracle_ic=0.0,
                         active_ir=0.0, oracle_active_ir=0.0, n_ic=200))
    for k in range(100):                                    # planted: half above the threshold
        rows.append(dict(seed=k, ic_nominal=0.05, t=float(k) + 50, mean_ic=0.04, oracle_ic=0.05,
                         active_ir=0.5, oracle_active_ir=1.0, n_ic=200))
    out = analyze(pd.DataFrame(rows)).set_index("ic_nominal")
    thr = np.percentile(np.arange(100.0), 95)               # 94.05
    assert out["null95"].iloc[0] == pytest.approx(thr)
    assert out.loc[0.05, "power"] == pytest.approx(np.mean(np.arange(100.0) + 50 > thr))
    assert out.loc[0.05, "capture_ir"] == pytest.approx(0.5)
    assert out.loc[0.0, "power"] == pytest.approx(0.05, abs=0.011)


# ------------------------------------------------------- demeaned no-edge world
def test_demean_returns_known_values_and_shape():
    idx = pd.bdate_range("2020-01-01", periods=3)
    r = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.00, -0.02, 0.05]}, index=idx)
    out = demean_returns(r)
    assert out["A"].tolist() == pytest.approx([-0.01, 0.0, 0.01])
    assert out["B"].tolist() == pytest.approx([-0.01, -0.03, 0.04])      # B's mean is 0.01
    assert out.index.equals(r.index) and list(out.columns) == ["A", "B"]
    assert out.mean().abs().max() < 1e-15
    assert out.std().tolist() == pytest.approx(r.std().tolist())         # volatility untouched


def test_demeaned_world_has_no_static_edge_but_the_raw_world_does():
    # Give every ETF its own constant drift (a static difference between ETFs).
    r = noise_returns(2500, 20, seed=8)
    drift = np.linspace(-0.0005, 0.0005, 20)
    r = r + drift
    raw = one_replicate(r, seed=3, ic_levels=[0.0], block_length=63, **SMALL)[0]
    dem = one_replicate(demean_returns(r), seed=3, ic_levels=[0.0], block_length=63, **SMALL)[0]
    assert raw["oracle_ic"] > 0.05            # static differences make momentum look predictive
    assert abs(dem["oracle_ic"]) < 0.04       # removing them takes that away
    assert raw["mean_ic"] > dem["mean_ic"] + 0.04

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_positive_control_etf.py::test_block_bootstrap_keeps_shape_index_and_uses_only_original_rows PASSED [ 28%]
# test_positive_control_etf.py::test_block_bootstrap_blocks_are_consecutive_runs_of_the_original PASSED [ 29%]
# test_positive_control_etf.py::test_block_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds PASSED [ 30%]
# test_positive_control_etf.py::test_returns_to_prices_known_values PASSED [ 31%]
# test_positive_control_etf.py::test_planted_score_is_centered_and_ordered_by_momentum PASSED [ 32%]
# test_positive_control_etf.py::test_plant_signal_known_values PASSED      [ 34%]
# test_positive_control_etf.py::test_zero_ic_changes_nothing_and_missing_scores_mean_no_tilt PASSED [ 35%]
# test_positive_control_etf.py::test_date_ics_known_values PASSED          [ 36%]
# test_positive_control_etf.py::test_ic_tstat_takes_every_nth_date_and_matches_the_formula PASSED [ 37%]
# test_positive_control_etf.py::test_top_quintile_active_ir_known_values PASSED [ 38%]
# test_positive_control_etf.py::test_out_of_fold_predictions_cover_every_row_once_and_are_probabilities PASSED [ 39%]
# test_positive_control_etf.py::test_pipeline_finds_a_strong_planted_signal PASSED [ 40%]
# test_positive_control_etf.py::test_pipeline_finds_nothing_when_nothing_is_planted PASSED [ 42%]
# test_positive_control_etf.py::test_realized_oracle_ic_is_close_to_the_nominal_planted_ic PASSED [ 43%]
# test_positive_control_etf.py::test_one_replicate_returns_one_row_per_ic_level_and_is_reproducible PASSED [ 44%]
# test_positive_control_etf.py::test_analyze_power_uses_the_null_95th_percentile_threshold PASSED [ 45%]
# test_positive_control_etf.py::test_demean_returns_known_values_and_shape PASSED [ 46%]
# test_positive_control_etf.py::test_demeaned_world_has_no_static_edge_but_the_raw_world_does PASSED [ 47%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): same-day planting score, independent per-ETF bootstrap, a 50th-percentile threshold, and a no-op demean each made tests fail; original restored.
# ---------------------------------------------------------------------------
