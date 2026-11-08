"""Tests for crypto_family.py with hand-checkable values. Run from the portfolio folder:
    pytest test_crypto_family.py -v
"""
import numpy as np
import pandas as pd
import pytest

import crypto_family as cf


def days(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


def long_index(dates, assets):
    return pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])


# ------------------------------------------------------------------- universe
def universe_panel():
    idx = days(12)
    px = pd.DataFrame(100.0, index=idx, columns=list("ABCDX"))
    px.iloc[:3, px.columns.get_loc("B")] = np.nan          # B lists on day 3
    px.iloc[[8, 9], px.columns.get_loc("D")] = np.nan      # D has a 2-day gap
    qv = pd.DataFrame(1000.0, index=idx, columns=px.columns)
    qv["C"] = 10.0                                         # C is illiquid
    return px, qv


def test_universe_applies_history_liquidity_gap_and_exclusion_rules():
    px, qv = universe_panel()
    elig, _ = cf.build_universe(px, qv, exclude=["X"], min_obs=5, min_dv=100, dv_window=4, max_missing=1, min_assets=2)
    assert "X" not in elig.columns
    assert elig["A"].iloc[4:].all() and not elig["A"].iloc[:4].any()       # needs 5 observed days, then a full window
    assert list(np.flatnonzero(elig["B"])) == [7, 8, 9, 10, 11]            # 5 days of history from day 3, clean window
    assert not elig["C"].any()                                              # volume below the floor
    assert list(np.flatnonzero(elig["D"])) == [4, 5, 6, 7]                  # two missing days in the window kill it


def test_a_date_with_too_few_coins_is_removed_entirely():
    px, qv = universe_panel()
    elig, dv = cf.build_universe(px, qv, exclude=["X"], min_obs=5, min_dv=100, dv_window=4, max_missing=1, min_assets=3)
    assert not elig.iloc[4:7].any().any()      # only A and D qualify on days 4-6: 2 < 3
    assert dv.iloc[4:7].isna().all().all()


def test_universe_uses_only_past_data():
    px, qv = universe_panel()
    qv = qv + np.arange(12)[:, None]            # distinct values, so a window that peeks ahead changes the median
    kw = dict(exclude=["X"], min_obs=5, min_dv=800, dv_window=4, max_missing=1, min_assets=2)
    base, dv_base = cf.build_universe(px, qv, **kw)
    px2, qv2 = px.copy(), qv.copy()
    px2.iloc[10:, :] = np.nan
    qv2.iloc[10:, :] = 0.0                      # destroy the future
    again, dv_again = cf.build_universe(px2, qv2, **kw)
    assert base.iloc[:10].equals(again.iloc[:10])
    pd.testing.assert_frame_equal(dv_base.iloc[:10], dv_again.iloc[:10])
    assert base.iloc[4:10].any().any()          # the comparison is not vacuous


def test_tiers_are_terciles_by_volume_with_zero_the_lowest():
    dv = pd.DataFrame([[6.0, 1.0, 5.0, 2.0, 4.0, 3.0]], columns=list("abcdef"), index=days(1))
    t = cf.assign_tiers(dv)
    assert t.iloc[0].to_dict() == {"a": 2, "b": 0, "c": 2, "d": 0, "e": 1, "f": 1}


def test_tiers_leave_ineligible_coins_blank():
    dv = pd.DataFrame([[3.0, np.nan, 1.0, 2.0]], columns=list("abcd"), index=days(1))
    t = cf.assign_tiers(dv).iloc[0]
    assert np.isnan(t["b"]) and t["c"] == 0


# --------------------------------------------------------------- forward returns
def test_forward_returns_known_values_and_delisting_rule():
    idx = days(10)
    z = pd.Series([100 * 1.1 ** i for i in range(5)] + [np.nan] * 5, index=idx)           # delists after day 4
    live = pd.Series([100 * 1.01 ** i for i in range(10)], index=idx)
    halt = live.copy(); halt.iloc[5] = np.nan                                                # one missing day
    px = pd.DataFrame({"Z": z, "L": live, "H": halt})
    f = cf.forward_log_returns(px, 3)
    assert f["Z"].iloc[0] == pytest.approx(3 * np.log(1.1))          # normal: t+3 exists
    assert f["Z"].iloc[2] == pytest.approx(2 * np.log(1.1))          # t+3 is past the last price: use the last price
    assert f["Z"].iloc[4] == pytest.approx(0.0)                      # on its last day
    assert f["Z"].iloc[5:].isna().all()
    assert f["L"].iloc[7:].isna().all() and f["L"].iloc[0] == pytest.approx(3 * np.log(1.01))
    assert np.isnan(f["H"].iloc[2])                                  # the end date is missing: no label, no guess


# ------------------------------------------------------------------- ranking
def test_within_tier_rank_known_values():
    idx = long_index(days(1), list("abcde"))
    v = pd.Series([3.0, 1.0, 2.0, 10.0, 20.0], index=idx)
    tier = pd.Series([0, 0, 0, 1, 1], index=idx)
    r = cf.within_tier_rank(v, tier)
    assert r.round(4).tolist() == [0.8333, 0.1667, 0.5, 0.25, 0.75]


# ------------------------------------------------------------------- dataset
def tiny_panel(n_days=30, n_coins=10, seed=3):
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_coins)), axis=0)),
                      index=days(n_days), columns=[f"c{i}" for i in range(n_coins)])
    px.columns.name = "asset"
    elig = pd.DataFrame(True, index=px.index, columns=px.columns)
    elig.iloc[:3] = False
    tiers = pd.DataFrame(np.tile(np.where(np.arange(n_coins) < 5, 0.0, 1.0), (n_days, 1)),
                         index=px.index, columns=px.columns).where(elig)
    return px, elig, tiers


CFG = dict(horizon=2, features=["f"], vol_feature="f", raw=lambda p: {"f": np.log(p).diff()})


def test_dataset_labels_are_tier_median_excess_and_t1_is_h_days_later():
    px, elig, tiers = tiny_panel()
    ds = cf.build_dataset(px, elig, tiers, CFG, min_tier=3)
    d = ds.index.get_level_values("date")
    assert d.max() == px.index[-3]                                    # the last 2 days have no label
    assert ((ds["t1"] - d) == pd.Timedelta(days=2)).all()
    assert ds["f"].between(0, 1).all()
    date = px.index[10]
    sub = ds.xs(date, level="date")
    for t in (0, 1):
        s = sub[sub["tier"] == t]
        fwd = np.log(px.shift(-2) / px).loc[date, s.index]
        assert s["fwd_ret"].to_numpy() == pytest.approx(fwd.to_numpy())
        assert s["excess_ret"].to_numpy() == pytest.approx((fwd - fwd.median()).to_numpy())
        assert (s["label"] == (s["excess_ret"] > 0)).all()


def test_dataset_drops_a_tier_that_is_too_small():
    px, elig, tiers = tiny_panel()
    ds = cf.build_dataset(px, elig, tiers, CFG, min_tier=6)           # tiers have 5 coins each
    assert ds.empty


# ---------------------------------------------------------- ICs and portfolios
def two_tier_frame(scores_by_date, fwd_by_date):
    dates = days(len(scores_by_date))
    assets = [f"a{i}" for i in range(10)]
    idx = long_index(dates, assets)
    score = pd.Series(np.concatenate(scores_by_date), index=idx, dtype=float)
    fwd = pd.Series(np.concatenate(fwd_by_date), index=idx, dtype=float)
    tier = pd.Series(np.tile([0] * 5 + [2] * 5, len(dates)), index=idx)
    return score, fwd, tier


def test_ic_is_plus_one_for_a_perfect_score_and_minus_one_for_its_reverse():
    s = np.arange(10.0)
    score, fwd, tier = two_tier_frame([s, s], [s * 2, s * 2])
    assert cf.date_tier_ics(score, fwd, tier).round(6).tolist() == [1.0] * 4
    assert cf.date_tier_ics(-score, fwd, tier).round(6).tolist() == [-1.0] * 4


def test_ic_is_computed_inside_tiers_not_across_them():
    # tier 2 returns are all far above tier 0, but inside each tier the score is perfectly wrong
    score = np.array([4, 3, 2, 1, 0, 9, 8, 7, 6, 5.0])
    fwd = np.array([0, 1, 2, 3, 4, 100, 101, 102, 103, 104.0])
    sc, fw, tier = two_tier_frame([score], [fwd])
    assert cf.date_tier_ics(sc, fw, tier).round(6).tolist() == [-1.0, -1.0]


def test_ic_is_a_rank_correlation_inside_each_tier_even_when_tiers_interleave():
    score = np.array([0, 1, 2, 3, 100, 4, 5, 6, 7, 8.0])          # tier 0 holds both the lowest and the highest score
    fwd = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14.0])
    sc, fw, tier = two_tier_frame([score], [fwd])
    assert cf.date_tier_ics(sc, fw, tier).round(9).tolist() == [1.0, 1.0]


def test_long_weights_pick_the_top_fifth_of_each_tier_and_sum_to_one():
    score, _, tier = two_tier_frame([np.arange(10.0)], [np.zeros(10)])
    w = cf.long_weights(score, tier)
    assert w.sum() == pytest.approx(1.0)
    assert w[w > 0].index.get_level_values("asset").tolist() == ["a4", "a9"]
    assert w[w > 0].tolist() == [0.5, 0.5]


def test_a_tier_of_six_picks_two_and_float_noise_does_not_add_a_pick():
    idx = long_index(days(1), [f"a{i}" for i in range(6)])
    score, tier = pd.Series(np.arange(6.0), index=idx), pd.Series(0, index=idx)
    assert (cf.long_weights(score, tier) > 0).sum() == 2
    idx = long_index(days(1), [f"a{i}" for i in range(25)])
    score, tier = pd.Series(np.arange(25.0), index=idx), pd.Series(0, index=idx)
    assert (cf.long_weights(score, tier, top_share=0.28) > 0).sum() == 7      # 0.28 * 25 = 7.000000000000001


def test_active_return_and_costs_known_values():
    base = np.arange(10.0)                                          # picks: a4 (tier 0) and a9 (tier 2)
    moved = np.array([4, 3, 2, 1, 0, 9, 5, 6, 7, 8.0])              # picks: a0 (tier 0) and a5 (tier 2)
    fwd = np.array([0.0, 0, 0, 0, 0.10, 0.20, 0, 0, 0, 0.30])
    score, f, tier = two_tier_frame([base, base, moved], [fwd, fwd, fwd])
    act = cf.active_returns(score, f, tier, horizon=1)
    # date 1: weights 0.5 on a4 (0.10) and a9 (0.30); universe mean is 0.06
    assert act["gross"].iloc[0] == pytest.approx(0.5 * 0.10 + 0.5 * 0.30 - 0.06)
    assert act["turnover"].iloc[0] == pytest.approx(1.0)                     # the first period buys everything
    assert act["cost"].iloc[0] == pytest.approx(0.5 * cf.COST_PER_SIDE[0] + 0.5 * cf.COST_PER_SIDE[2])
    assert act["turnover"].iloc[1] == pytest.approx(0.0) and act["cost"].iloc[1] == pytest.approx(0.0)
    # date 3: both picks change, so each tier sells 0.5 and buys 0.5
    assert act["turnover"].iloc[2] == pytest.approx(2.0)
    assert act["cost"].iloc[2] == pytest.approx(cf.COST_PER_SIDE[0] + cf.COST_PER_SIDE[2])
    assert act["gross"].iloc[2] == pytest.approx(0.5 * 0.0 + 0.5 * 0.20 - 0.06)    # a0 returns 0, a5 returns 0.20
    assert act["net"].iloc[0] == pytest.approx(act["gross"].iloc[0] - act["cost"].iloc[0])


def test_cost_schedules_follow_kraken_fees_and_the_taker_sensitivity_is_dearer():
    assert cf.COST_PER_SIDE == pytest.approx({0: 0.0080, 1: 0.0055, 2: 0.0045})
    assert cf.COST_TAKER == pytest.approx({0: 0.0120, 1: 0.0095, 2: 0.0085})
    assert all(cf.COST_TAKER[k] > cf.COST_PER_SIDE[k] for k in cf.COST_PER_SIDE)


def test_taker_costs_lower_net_returns_by_the_known_amount():
    base = np.arange(10.0)
    fwd = np.array([0.0, 0, 0, 0, 0.10, 0.20, 0, 0, 0, 0.30])
    score, f, tier = two_tier_frame([base, base], [fwd, fwd])
    maker = cf.active_returns(score, f, tier, horizon=1)
    taker = cf.active_returns(score, f, tier, horizon=1, cost=cf.COST_TAKER)
    assert (maker["gross"] == taker["gross"]).all()
    # first period buys 0.5 in tier 0 and 0.5 in tier 2; the taker schedule is dearer by 0.0040 in each tier
    assert (maker["net"] - taker["net"]).iloc[0] == pytest.approx(0.5 * 0.0040 + 0.5 * 0.0040)
    assert (maker["net"] - taker["net"]).iloc[1] == pytest.approx(0.0)


def test_periods_are_spaced_one_horizon_apart():
    s = np.arange(10.0)
    score, fwd, tier = two_tier_frame([s] * 7, [s] * 7)
    assert len(cf.active_returns(score, fwd, tier, horizon=3)) == 3
    assert len(cf.period_ic_series(score, fwd, tier, 3)) == 3


# ----------------------------------------------------------------- bootstrap
def test_bootstrap_lower_bound_is_positive_for_a_steady_edge_and_negative_for_noise():
    rng = np.random.default_rng(0)
    edge = 0.05 + rng.normal(0, 0.05, 150)
    noise = rng.normal(0, 0.05, 150)
    assert cf.lower_bound(edge, n_boot=1000) > 0
    assert cf.lower_bound(noise, n_boot=1000) < 0


def test_paired_lower_bound_is_zero_for_identical_series_and_positive_for_a_clear_gap():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 0.05, 120))
    assert cf.paired_lower_bound(a, a, n_boot=500) == 0.0
    assert cf.paired_lower_bound(a + 0.2, a, n_boot=500) == pytest.approx(0.2)


def test_bootstrap_is_reproducible_and_handles_a_block_longer_than_the_series():
    x = np.arange(4.0)
    assert np.array_equal(cf.block_bootstrap_means(x, block=10, n_boot=50), cf.block_bootstrap_means(x, block=10, n_boot=50))
    assert cf.block_bootstrap_means(x, block=10, n_boot=50).tolist() == [1.5] * 50


# ------------------------------------------------------------------ decisions
def base_res(**kw):
    r = dict(ic_lower=0.01, mean_ic=0.02, ic_vs_control_lower=0.005, net_lower=0.001)
    r.update(kw)
    return r


def test_an_edge_needs_all_three_conditions_and_tradable_needs_net_returns():
    assert cf.decide(base_res()) == {"edge_found": True, "tradable": True}
    assert cf.decide(base_res(ic_lower=-0.001))["edge_found"] is False
    assert cf.decide(base_res(mean_ic=0.009))["edge_found"] is False            # below the minimum worth trading
    assert cf.decide(base_res(ic_vs_control_lower=-0.001))["edge_found"] is False   # no better than a volatility sort
    assert cf.decide(base_res(net_lower=-0.001)) == {"edge_found": True, "tradable": False}


def test_alpha_is_the_bonferroni_level_for_three_arms():
    assert cf.ALPHA == pytest.approx(0.05 / 3)


# --------------------------------------------------------------------- model
def planted_ds(n_dates=120, n_assets=12, dev_end="2024-02-15", flip_from=70, seed=0):
    rng = np.random.default_rng(seed)
    dates = days(n_dates, "2024-01-01")
    idx = long_index(dates, [f"c{i}" for i in range(n_assets)])
    f = pd.Series(rng.uniform(0, 1, len(idx)), index=idx)
    label = (f > 0.5).astype(int)
    d = idx.get_level_values("date")
    t1 = pd.Series(d + pd.Timedelta(days=1), index=idx)
    # rows late in development whose labels resolve AFTER dev_end and are deliberately wrong
    late = (d > pd.Timestamp(dev_end) - pd.Timedelta(days=30)) & (d <= pd.Timestamp(dev_end))
    t1[late] = pd.Timestamp(dev_end) + pd.Timedelta(days=20)
    label = label.where(~late, 1 - label)
    return pd.DataFrame({"f": f, "label": label, "t1": t1, "tier": 0}), pd.Timestamp(dev_end)


def test_holdout_fit_excludes_labels_that_resolve_after_the_cutoff():
    ds, dev_end = planted_ds()
    s = cf.holdout_scores(ds, ["f"], dev_end=dev_end, held_start=dev_end + pd.Timedelta(days=1))
    assert s.index.get_level_values("date").min() == dev_end + pd.Timedelta(days=1)
    x = ds.loc[s.index, "f"]
    assert np.corrcoef(s, x)[0, 1] > 0.95            # learned the true relation, not the flipped late rows


def test_oof_scores_cover_every_row_and_find_a_planted_relation():
    ds, _ = planted_ds()
    s = cf.oof_scores(ds, ["f"], n_splits=5, embargo_dates=1)
    assert s.notna().all() and len(s) == len(ds)
    clean = ds["t1"] <= ds["t1"].min() + pd.Timedelta(days=5)
    assert np.corrcoef(s[clean], ds.loc[clean, "f"])[0, 1] > 0.5


# ------------------------------------------------------------------ end to end
def synthetic_market(n_days=1500, n_coins=40, phi=0.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_days, freq="D")
    eps = rng.normal(0, 0.03, (n_days, n_coins))
    r = np.zeros_like(eps)
    for t in range(1, n_days):
        r[t] = phi * r[t - 1] + eps[t]
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"C{i}USDT" for i in range(n_coins)])
    px.columns.name = "asset"
    qv = pd.DataFrame(rng.uniform(2e6, 2e7, px.shape), index=idx, columns=px.columns)
    return px, qv


def test_no_signal_gives_no_edge_in_any_arm():
    px, qv = synthetic_market(phi=0.0)
    res = cf.run_family(px, qv, n_boot=300)
    assert set(res) == {"A", "B", "C"}
    assert not any(r["rules"]["edge_found"] for r in res.values())
    assert all(r["n_periods"] > 10 for r in res.values())


def test_a_planted_one_day_signal_is_found_by_arm_a_but_does_not_pass_the_volatility_control_by_luck():
    px, qv = synthetic_market(phi=0.15, seed=1)
    res = cf.run_family(px, qv, n_boot=300, arms=("A",))
    a = res["A"]
    assert a["mean_ic"] > 0.05 and a["rules"]["edge_found"]
    assert a["control_mean_ic"] < a["mean_ic"]
    assert a["net_taker_mean"] < a["net_mean"] < a["gross_mean"]      # taker fees cost more than maker fees, and both cost something


# --------------------------------------------------------------------- files
def test_check_not_run_refuses_to_overwrite(tmp_path):
    (tmp_path / "crypto_family_results.csv").write_text("x")
    with pytest.raises(FileExistsError):
        cf.check_not_run(tmp_path)


def test_load_panels_rejects_a_date_hole(tmp_path):
    p = pd.DataFrame({"date": ["2024-01-01", "2024-01-03"], "AUSDT": [1.0, 2.0]})
    p.to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame({"date": ["2024-01-01"], "symbol": ["AUSDT"], "quote_volume": [5.0]}).to_csv(tmp_path / "o.csv", index=False)
    with pytest.raises(ValueError, match="one row per calendar day"):
        cf.load_panels(tmp_path / "p.csv", tmp_path / "o.csv")

# ------------------------------------------------------- Kraken confirmation (rule 3)
def kraken_like(n_days=520, n_coins=40, phi=0.15, seed=2, volume=(2e5, 2e6)):
    """A small USD panel like the Kraken pull: one row per day, 'USD' pair names, dollar volumes near the $100K floor."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    eps = rng.normal(0, 0.03, (n_days, n_coins))
    r = np.zeros_like(eps)
    for t in range(1, n_days):
        r[t] = phi * r[t - 1] + eps[t]
    px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"K{i}USD" for i in range(n_coins)])
    px.columns.name = "asset"
    qv = pd.DataFrame(rng.uniform(*volume, px.shape), index=idx, columns=px.columns)
    return px, qv


def fitted_model(arm="A", phi=0.15, seed=1):
    """A model fitted on a Binance-like development period with a planted one-day signal."""
    px, qv = synthetic_market(phi=phi, seed=seed)
    elig, dv = cf.build_universe(px, qv)
    ds = cf.build_dataset(px, elig, cf.assign_tiers(dv), cf.ARMS[arm])
    return cf.fit_holdout_model(ds, cf.ARMS[arm]["features"])


def test_the_kraken_floor_is_100k_and_the_binance_floor_stays_1m():
    assert cf.KRAKEN_MIN_DV == 100_000.0 and cf.MIN_DV == 1_000_000.0
    px = pd.DataFrame(100.0, index=days(8), columns=list("ABC"))
    qv = pd.DataFrame(150_000.0, index=px.index, columns=px.columns)
    kw = dict(exclude=[], min_obs=3, dv_window=4, max_missing=1, min_assets=2)
    assert cf.build_universe(px, qv, min_dv=cf.KRAKEN_MIN_DV, **kw)[0].iloc[-1].all()
    assert not cf.build_universe(px, qv, min_dv=cf.MIN_DV, **kw)[0].any().any()


def test_kraken_exclusions_are_the_flagged_stablecoins_plus_the_binance_exclusion_bases():
    m = pd.DataFrame({"pair": ["USDCUSD", "ZEURUSD", "WBTCUSD", "PAXGUSD", "LUNAUSD", "UUSD", "ADAUSD", "UNIUSD", "XBTUSD"],
                      "base": ["USDC", "ZEUR", "WBTC", "PAXG", "LUNA", "U", "ADA", "UNI", "XXBT"],
                      "stable_or_fiat_base": [True, True, False, False, False, False, False, False, False]})
    assert cf.kraken_exclusions(m) == ["LUNAUSD", "PAXGUSD", "USDCUSD", "UUSD", "WBTCUSD", "ZEURUSD"]   # not UNI, not ADA, not XBT
    assert cf.kraken_exclusions(m, exclude=[]) == ["USDCUSD", "ZEURUSD"]                                # flagged ones only


def test_load_panels_reads_the_kraken_column_names(tmp_path):
    p = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "AUSD": [1.0, 2.0, 3.0], "BUSD": [4.0, 5.0, 6.0]})
    p.to_csv(tmp_path / "p.csv", index=False)
    o = pd.DataFrame({"date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2, "pair": ["AUSD"] * 3 + ["BUSD"] * 3,
                      "dollar_volume": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    o.to_csv(tmp_path / "o.csv", index=False)
    px, qv = cf.load_panels(tmp_path / "p.csv", tmp_path / "o.csv", symbol_col="pair", volume_col="dollar_volume")
    assert px.shape == (3, 2) and qv.loc["2024-01-02", "BUSD"] == 50.0 and qv.loc["2024-01-03", "AUSD"] == 30.0


def test_kraken_confirmed_needs_a_positive_ic_and_a_positive_net_return():
    assert cf.kraken_confirmed({"mean_ic": 0.02, "net_mean": 0.001}) is True
    assert cf.kraken_confirmed({"mean_ic": -0.02, "net_mean": 0.001}) is False
    assert cf.kraken_confirmed({"mean_ic": 0.02, "net_mean": -0.001}) is False       # an IC that costs eat is not confirmed
    assert cf.kraken_confirmed({"mean_ic": 0.0, "net_mean": 0.001}) is False         # strictly above zero
    assert cf.kraken_confirmed({"mean_ic": float("nan"), "net_mean": 0.001}) is False


def test_the_frozen_model_gives_the_same_scores_as_holdout_scores():
    ds, dev_end = planted_ds()
    model = cf.fit_holdout_model(ds, ["f"], dev_end=dev_end)
    s = cf.holdout_scores(ds, ["f"], dev_end=dev_end, held_start=dev_end + pd.Timedelta(days=1))
    again = pd.Series(model.predict_proba(ds.loc[s.index, ["f"]].to_numpy())[:, 1], index=s.index)
    pd.testing.assert_series_equal(s, again)


class NoRefit:
    """Scores like the real model but raises if anything tries to fit it on Kraken data."""
    def __init__(self, model):
        self.model = model
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def fit(self, *a, **k):
        raise AssertionError("the Kraken check must not refit the model")


def test_a_signal_that_carries_over_is_confirmed_and_costs_can_still_sink_it():
    model = NoRefit(fitted_model())
    px, qv = kraken_like(phi=0.15)
    free = cf.kraken_check("A", model, px, qv, exclude=[], n_boot=300, cost={0: 0.0, 1: 0.0, 2: 0.0})
    assert free["mean_ic"] > 0.05 and free["net_mean"] > 0 and free["kraken_confirmed"] is True
    assert free["n_periods"] > 100
    costly = cf.kraken_check("A", model, px, qv, exclude=[], n_boot=300)                  # the real maker schedule
    assert costly["mean_ic"] == pytest.approx(free["mean_ic"]) and costly["mean_ic"] > 0.05
    assert costly["net_mean"] < 0 and costly["kraken_confirmed"] is False                 # edge exists, costs consume it


def test_a_signal_that_reverses_on_kraken_is_not_confirmed():
    px, qv = kraken_like(phi=-0.15)
    res = cf.kraken_check("A", fitted_model(), px, qv, exclude=[], n_boot=300, cost={0: 0.0, 1: 0.0, 2: 0.0})
    assert res["mean_ic"] < -0.05 and res["kraken_confirmed"] is False


def test_kraken_check_leaves_out_excluded_pairs_and_coins_below_the_floor():
    px, qv = kraken_like(n_coins=37, volume=(2e5, 2e6))
    qv["K0USD"] = 5_000.0                                  # below the $100K floor
    with_all = cf.kraken_check("A", fitted_model(), px, qv, exclude=[], n_boot=100)
    skipped = cf.kraken_check("A", fitted_model(), px, qv, exclude=["K1USD", "K2USD", "K3USD", "K4USD"], n_boot=100)
    assert with_all["n_coins_median"] == 36 and skipped["n_coins_median"] == 32      # K0 never in; K1-K4 removed on request


def test_kraken_check_with_too_few_coins_is_not_confirmed_and_keeps_the_same_keys():
    px, qv = kraken_like(n_coins=25)                                                 # under the 30-coin minimum on every date
    res = cf.kraken_check("A", fitted_model(), px, qv, exclude=[], n_boot=100)
    full = cf.kraken_check("A", fitted_model(), *kraken_like(), exclude=[], n_boot=100)
    assert res["n_periods"] == 0 and res["kraken_confirmed"] is False and set(res) == set(full)


def test_run_family_checks_kraken_only_for_arms_a_and_b_that_found_an_edge():
    px, qv = synthetic_market(phi=0.15, seed=1)
    kpx, kqv = kraken_like(phi=0.15)
    res = cf.run_family(px, qv, n_boot=300, arms=("A", "C"), kraken=dict(prices=kpx, qvol=kqv, exclude=[]))
    assert res["A"]["rules"]["edge_found"] and "kraken" in res["A"]
    assert isinstance(res["A"]["rules"]["kraken_confirmed"], bool)
    assert "kraken" not in res["C"] and res["C"]["rules"].get("kraken_confirmed") is None       # arm C has no Kraken step
    px0, qv0 = synthetic_market(phi=0.0, seed=0)
    none = cf.run_family(px0, qv0, n_boot=300, arms=("A",), kraken=dict(prices=kpx, qvol=kqv, exclude=[]))
    assert not none["A"]["rules"]["edge_found"] and "kraken" not in none["A"]
    assert none["A"]["rules"]["kraken_confirmed"] is None                                       # not run, not "False"

def test_the_command_line_run_writes_results_with_the_kraken_rows_and_refuses_a_second_run(tmp_path, monkeypatch):
    px, qv = synthetic_market(phi=0.15, seed=1)
    kpx, kqv = kraken_like(phi=0.15)
    px.to_csv(tmp_path / "bp.csv", index_label="date")
    qv.stack().rename("quote_volume").rename_axis(["date", "symbol"]).reset_index().to_csv(tmp_path / "bo.csv", index=False)
    kpx.to_csv(tmp_path / "kp.csv", index_label="date")
    kqv.stack().rename("dollar_volume").rename_axis(["date", "pair"]).reset_index().to_csv(tmp_path / "ko.csv", index=False)
    pd.DataFrame({"pair": kpx.columns, "base": [c[:-3] for c in kpx.columns], "stable_or_fiat_base": False}).to_csv(
        tmp_path / "km.csv", index=False)
    args = ["crypto_family.py", "--prices", str(tmp_path / "bp.csv"), "--ohlcv", str(tmp_path / "bo.csv"),
            "--kraken-prices", str(tmp_path / "kp.csv"), "--kraken-ohlcv", str(tmp_path / "ko.csv"),
            "--kraken-manifest", str(tmp_path / "km.csv"), "--folder", str(tmp_path)]
    real = cf.run_family
    monkeypatch.setattr(cf, "run_family", lambda p, q, kraken=None: real(p, q, n_boot=200, kraken=kraken))
    monkeypatch.setattr("sys.argv", args)
    cf.main()
    out = pd.read_csv(tmp_path / "crypto_family_results.csv")
    metrics = set(out["metric"])
    assert {"mean_ic", "rule_edge_found", "rule_tradable", "rule_kraken_confirmed", "kraken_mean_ic", "kraken_net_mean",
            "kraken_kraken_confirmed"} <= metrics
    assert (tmp_path / "crypto_family_A_ic.csv").exists() and (tmp_path / "crypto_family_C_active.csv").exists()
    with pytest.raises(FileExistsError):
        cf.main()


def test_the_command_line_run_needs_the_kraken_files(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["crypto_family.py", "--prices", "a", "--ohlcv", "b", "--folder", str(tmp_path)])
    with pytest.raises(SystemExit):
        cf.main()


# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-11, mlfinlab env on Ethan's machine: Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2). 41 tests: the original 29 plus 12 for the Kraken confirmation (pre-registration section 8, rule 3).
# The same 41 passed in the sandbox on the same versions before this run.
# $ conda activate mlfinlab ; cd C:\ws\AFML\portfolio ; pytest test_crypto_family.py -v --color=no
# ============================= test session starts =============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML\portfolio
# collecting ... collected 41 items
#
# test_crypto_family.py::test_universe_applies_history_liquidity_gap_and_exclusion_rules PASSED [  2%]
# test_crypto_family.py::test_a_date_with_too_few_coins_is_removed_entirely PASSED [  4%]
# test_crypto_family.py::test_universe_uses_only_past_data PASSED          [  7%]
# test_crypto_family.py::test_tiers_are_terciles_by_volume_with_zero_the_lowest PASSED [  9%]
# test_crypto_family.py::test_tiers_leave_ineligible_coins_blank PASSED    [ 12%]
# test_crypto_family.py::test_forward_returns_known_values_and_delisting_rule PASSED [ 14%]
# test_crypto_family.py::test_within_tier_rank_known_values PASSED         [ 17%]
# test_crypto_family.py::test_dataset_labels_are_tier_median_excess_and_t1_is_h_days_later PASSED [ 19%]
# test_crypto_family.py::test_dataset_drops_a_tier_that_is_too_small PASSED [ 21%]
# test_crypto_family.py::test_ic_is_plus_one_for_a_perfect_score_and_minus_one_for_its_reverse PASSED [ 24%]
# test_crypto_family.py::test_ic_is_computed_inside_tiers_not_across_them PASSED [ 26%]
# test_crypto_family.py::test_ic_is_a_rank_correlation_inside_each_tier_even_when_tiers_interleave PASSED [ 29%]
# test_crypto_family.py::test_long_weights_pick_the_top_fifth_of_each_tier_and_sum_to_one PASSED [ 31%]
# test_crypto_family.py::test_a_tier_of_six_picks_two_and_float_noise_does_not_add_a_pick PASSED [ 34%]
# test_crypto_family.py::test_active_return_and_costs_known_values PASSED  [ 36%]
# test_crypto_family.py::test_cost_schedules_follow_kraken_fees_and_the_taker_sensitivity_is_dearer PASSED [ 39%]
# test_crypto_family.py::test_taker_costs_lower_net_returns_by_the_known_amount PASSED [ 41%]
# test_crypto_family.py::test_periods_are_spaced_one_horizon_apart PASSED  [ 43%]
# test_crypto_family.py::test_bootstrap_lower_bound_is_positive_for_a_steady_edge_and_negative_for_noise PASSED [ 46%]
# test_crypto_family.py::test_paired_lower_bound_is_zero_for_identical_series_and_positive_for_a_clear_gap PASSED [ 48%]
# test_crypto_family.py::test_bootstrap_is_reproducible_and_handles_a_block_longer_than_the_series PASSED [ 51%]
# test_crypto_family.py::test_an_edge_needs_all_three_conditions_and_tradable_needs_net_returns PASSED [ 53%]
# test_crypto_family.py::test_alpha_is_the_bonferroni_level_for_three_arms PASSED [ 56%]
# test_crypto_family.py::test_holdout_fit_excludes_labels_that_resolve_after_the_cutoff PASSED [ 58%]
# test_crypto_family.py::test_oof_scores_cover_every_row_and_find_a_planted_relation PASSED [ 60%]
# test_crypto_family.py::test_no_signal_gives_no_edge_in_any_arm PASSED    [ 63%]
# test_crypto_family.py::test_a_planted_one_day_signal_is_found_by_arm_a_but_does_not_pass_the_volatility_control_by_luck PASSED [ 65%]
# test_crypto_family.py::test_check_not_run_refuses_to_overwrite PASSED    [ 68%]
# test_crypto_family.py::test_load_panels_rejects_a_date_hole PASSED       [ 70%]
# test_crypto_family.py::test_the_kraken_floor_is_100k_and_the_binance_floor_stays_1m PASSED [ 73%]
# test_crypto_family.py::test_kraken_exclusions_are_the_flagged_stablecoins_plus_the_binance_exclusion_bases PASSED [ 75%]
# test_crypto_family.py::test_load_panels_reads_the_kraken_column_names PASSED [ 78%]
# test_crypto_family.py::test_kraken_confirmed_needs_a_positive_ic_and_a_positive_net_return PASSED [ 80%]
# test_crypto_family.py::test_the_frozen_model_gives_the_same_scores_as_holdout_scores PASSED [ 82%]
# test_crypto_family.py::test_a_signal_that_carries_over_is_confirmed_and_costs_can_still_sink_it PASSED [ 85%]
# test_crypto_family.py::test_a_signal_that_reverses_on_kraken_is_not_confirmed PASSED [ 87%]
# test_crypto_family.py::test_kraken_check_leaves_out_excluded_pairs_and_coins_below_the_floor PASSED [ 90%]
# test_crypto_family.py::test_kraken_check_with_too_few_coins_is_not_confirmed_and_keeps_the_same_keys PASSED [ 92%]
# test_crypto_family.py::test_run_family_checks_kraken_only_for_arms_a_and_b_that_found_an_edge PASSED [ 95%]
# test_crypto_family.py::test_the_command_line_run_writes_results_with_the_kraken_rows_and_refuses_a_second_run PASSED [ 97%]
# test_crypto_family.py::test_the_command_line_run_needs_the_kraken_files PASSED [100%]
#
# ============================= 41 passed in 10.56s =============================
#
# Mutation checks (sandbox), original 9: delisting rule removed, ceil guard removed, universe window peeking ahead, purge at
#   the cutoff removed, free first buy, IC pooled across tiers, Bonferroni removed, upper bound used, and the taker
#   sensitivity wired to the maker schedule each made a test fail; original restored. One mutation is EQUIVALENT, not a
#   gap: dropping the 'feature must exist' mask changes nothing because rows with a missing feature are dropped again
#   when the frames are joined.
# Mutation checks (sandbox), Kraken confirmation, 17 of 17 caught: IC rule '>=' ; OR instead of AND ; net return ignored ;
#   Kraken floor 1M ; floor ignored ; exclusions ignored ; stablecoin flag ignored ; Binance exclusion bases ignored ;
#   base matched by substring ; model refit on Kraken data ; costs ignored ; Kraken step run for arm C ; run with no
#   edge ; 'not run' recorded as False ; Binance column names used for Kraken ; purge removed from the frozen fit ;
#   empty Kraken result marked confirmed.
# Real-data check on the Kraken files (counts only, no returns): 633 pairs, 19 excluded, 461 dates with a universe
#   (2025-07-06 to 2026-10-09), 65 to 124 coins a date (median 84), at least 21 per tier, arm A 460 periods, arm B 92.
# Full-size synthetic run of the Binance part (700 coins x 3,332 days, no edge): about 52 seconds, 1.2 GB; no arm found an edge.
# NOT yet run on the real Binance panel. Pre-registered in preregistration_crypto_family.md (commits a170d2e, b42b39a);
#   the run is once only.
# ---------------------------------------------------------------------------
