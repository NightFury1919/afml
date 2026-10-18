"""Tests for noise_test_tier_b.py. The end-to-end test uses small windows on a 12-asset synthetic panel."""
import numpy as np
import pandas as pd
import pytest

import noise_test_tier_b as nb
import shadow_evaluate as ev
import shadow_scoring as ss

W = {"mom_long": 30, "mom_skip": 5, "mom_med": 15, "ret_short": 3, "vol": 10, "trend": 20}
SKW = dict(windows=W, min_assets=6, min_group_assets=3, horizon=5)
GROUPS = {f"A{i}": ("g1" if i < 6 else "g2") for i in range(12)}


def returns(n=600, seed=1, mean=0.0005):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(rng.normal(mean, 0.01, (n, 12)), index=idx, columns=list(GROUPS))


def test_registered_settings():
    assert nb.HIST == 1260 and nb.PERIODS == 36 and nb.STEP == 21 == ss.REFIT_EVERY
    assert nb.BLOCK == 63 and nb.WORLDS == 8 and nb.BASE_SEED == 20261018


def test_world_length_leaves_room_for_the_last_forward_return():
    assert nb.world_length(1260, 36, 21, 21) == 1260 + 21 * 35 + 21 + 1
    assert nb.world_length(100, 3, 5, 5) == 100 + 10 + 5 + 1


def test_common_returns_demeaned_and_drops_hedge_and_early_nans():
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    cols = [a for a in nb.CLASS_OF] + ["SH"]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, (300, len(cols))), axis=0)), index=idx, columns=cols)
    px.iloc[:100, 3] = np.nan                                  # one ETF starts late
    r = nb.common_returns(px)
    assert "SH" not in r.columns and r.shape[1] == 48
    assert len(r) == 199                                       # first 100 rows lost to the late ETF, plus the diff row
    assert r.mean().abs().max() < 1e-12


def test_world_prices_shape_dates_and_reproducibility():
    r = returns()
    p1 = nb.make_world_prices(r, 400, seed=5, block=20)
    p2 = nb.make_world_prices(r, 400, seed=5, block=20)
    p3 = nb.make_world_prices(r, 400, seed=6, block=20)
    assert p1.shape == (400, 12) and p1.index[0] == pd.Timestamp("2000-01-03") and p1.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(p1, p2)
    assert not p1.equals(p3)
    assert (p1 > 0).all().all()


def test_world_prices_use_whole_rows_in_blocks():
    r = returns()
    p = nb.make_world_prices(r, 120, seed=2, block=20)
    lr = np.log(p).diff().dropna()
    # every block of 20 days is a run of consecutive rows of the source: match the first block's second return
    src = r.to_numpy()
    first = lr.iloc[0:19].to_numpy()
    found = any(np.allclose(src[s + 1:s + 20], first) for s in range(len(src) - 20))
    assert found


def test_scoring_dates_are_step_apart_from_position_hist():
    idx = pd.bdate_range("2000-01-03", periods=200)
    d = nb.scoring_dates(idx, hist=100, periods=4, step=7)
    assert d == [idx[100], idx[107], idx[114], idx[121]]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("w")
    hist, periods, step = 120, 4, 5
    prices = nb.make_world_prices(returns(), nb.world_length(hist, periods, step, 5), seed=11, block=20)
    logs = nb.run_world(prices, folder / "prices.csv", folder / "shadow", hist, periods, step, GROUPS, SKW)
    return prices, logs, folder, (hist, periods, step)


def test_run_world_writes_one_refit_log_per_scoring_date(world):
    prices, logs, folder, (hist, periods, step) = world
    assert len(logs) == 4
    days = nb.scoring_dates(prices.index, hist, periods, step)
    for p, d in zip(logs, days):
        h, s = ss.read_scores(p)
        assert h["as_of"] == str(d.date()) and h["refit_today"] == "True" and h["late_fit"] == "False"
        assert h["first_scoring_date"] == str(days[0].date()) and h["code_version"] == "tier_b"
        assert len(s) == 12


def test_no_lookahead_in_the_world_logs(world):
    prices, logs, folder, (hist, periods, step) = world
    h, _ = ss.read_scores(logs[0])
    assert pd.Timestamp(h["fit_prices_through"]) == pd.Timestamp(h["as_of"])      # fit never past the scoring day
    assert pd.Timestamp(h["max_label_end"]) <= pd.Timestamp(h["as_of"])


def test_evaluate_world_returns_a_complete_row(world):
    prices, logs, folder, (hist, periods, step) = world
    # periods=4 is below the formal 36, so decide_formal is given periods=len(active); that is what evaluate_world does
    row = nb.evaluate_world(prices, logs, GROUPS, step, horizon=5, min_assets=6, min_group_assets=3, n_boot=200)
    assert row["n_periods"] == 4 and row["all_dates_logged"] and row["all_refit_today"] and not row["any_late_fit"]
    assert row["min_etfs_scored"] == 12
    for arm in ev.FITTED:
        assert np.isfinite(row[f"{arm}_ir"]) and -1 <= row[f"{arm}_mean_ic"] <= 1
        assert -1 <= row[f"{arm}_vol_tilt"] <= 1
    assert isinstance(row["any_edge_found"], bool)


def test_evaluate_world_matches_a_hand_computation_of_the_vol_rule_arm(world):
    prices, logs, folder, (hist, periods, step) = world
    row = nb.evaluate_world(prices, logs, GROUPS, step, horizon=5, min_assets=6, min_group_assets=3, n_boot=200)
    # rebuild wc_vol_rule's ir by hand: top ceil(0.2*6)=2 of each 6-asset class by vol_60_rank, minus class mean
    vals = []
    for p in logs:
        h, s = ss.read_scores(p)
        d = pd.Timestamp(h["as_of"])
        i = prices.index.get_loc(d)
        fwd = np.log(prices.iloc[i + 5] / prices.iloc[i])          # the pipeline's forward return is a log return
        s = s.set_index("asset")
        tot = 0.0
        for g in ("g1", "g2"):
            names = [a for a in GROUPS if GROUPS[a] == g]
            sc = s.loc[names, "wc_vol_rule"].sort_values(ascending=False)
            top = sc.index[:2]
            tot += (fwd[top].mean() - fwd[names].mean()) * len(names)
        vals.append(tot / 12)
    assert row["ir_vol_rule"] == pytest.approx(ev.ir(vals), abs=1e-9)


def _row(seed, ir=0.0, ic=0.0, **over):
    r = dict(seed=seed, n_periods=36, all_dates_logged=True, min_etfs_scored=48, all_refit_today=True,
             any_late_fit=False, ir_vol_rule=0.0)
    for a in ev.FITTED:
        r[f"{a}_ir"], r[f"{a}_mean_ic"] = ir, ic
    r.update(over)
    return r


def test_judge_passes_centred_noise_with_integrity():
    rng = np.random.default_rng(0)
    df = pd.DataFrame([_row(i, ir=x, ic=x / 20) for i, x in enumerate(rng.normal(0, 0.4, 8) - rng.normal(0, 0.4, 8).mean())])
    df.loc[:, [f"{a}_ir" for a in ev.FITTED]] = df.loc[:, [f"{a}_ir" for a in ev.FITTED]] - df["wc_current_ir"].mean()
    df.loc[:, [f"{a}_mean_ic" for a in ev.FITTED]] = df.loc[:, [f"{a}_mean_ic" for a in ev.FITTED]] \
        - df["wc_current_mean_ic"].mean()
    v = nb.judge(df)
    assert v["integrity"] and v["no_edge_claimed_by_noise"] and v["passed"]


def test_judge_fails_when_noise_looks_skilled():
    df = pd.DataFrame([_row(i, ir=1.0 + 0.01 * i, ic=0.05) for i in range(8)])      # every world shows a steady ir of 1
    v = nb.judge(df)
    assert not v["wc_current_ir_ok"] and not v["passed"]


@pytest.mark.parametrize("field,value", [("all_dates_logged", False), ("all_refit_today", False), ("any_late_fit", True),
                                         ("min_etfs_scored", 47), ("n_periods", 35)])
def test_judge_fails_on_each_integrity_problem(field, value):
    rng = np.random.default_rng(1)
    rows = [_row(i, ir=x, ic=x / 20) for i, x in enumerate(rng.normal(0, 0.4, 8))]
    rows[3][field] = value
    v = nb.judge(pd.DataFrame(rows))
    assert not v["integrity"] and not v["passed"]


def test_judge_single_world_has_infinite_se_and_is_ok():
    v = nb.judge(pd.DataFrame([_row(0, ir=5.0)]))
    assert v["wc_current_ir_se"] == np.inf and v["wc_current_ir_ok"]


def test_one_world_end_to_end_and_never_overwrites(tmp_path):
    args = (returns(), 77, tmp_path, 120, 6, 5, GROUPS, SKW)
    row = nb.one_world(args)
    assert row["seed"] == 77 and row["n_periods"] == 6
    assert (tmp_path / "world_77" / "result.csv").exists()
    assert len(list((tmp_path / "world_77" / "shadow").glob("scores_asof_*.csv"))) == 6
    with pytest.raises(FileExistsError):
        nb.one_world(args)


def test_cli_refuses_when_summary_exists(tmp_path):
    (tmp_path / "summary.csv").write_text("x")
    assert nb.main(["--prices", "nope.csv", "--folder", str(tmp_path)]) == 1


def _assemble(logs):
    frames = []
    for p in logs:
        h, s = ss.read_scores(p)
        s["date"] = pd.Timestamp(h["as_of"])
        frames.append(s)
    return pd.concat(frames).set_index(["date", "asset"]).sort_index()


def test_vol_tilt_is_each_arms_score_against_the_vol_60_rank_column(world):
    from real_history_check_wc import within_class_tilt
    prices, logs, folder, (hist, periods, step) = world
    row = nb.evaluate_world(prices, logs, GROUPS, step, horizon=5, min_assets=6, min_group_assets=3, n_boot=200)
    sc = _assemble(logs)
    for arm in ev.FITTED:
        assert row[f"{arm}_vol_tilt"] == pytest.approx(within_class_tilt(sc[arm], sc["vol_60_rank"], GROUPS), abs=1e-12)
    assert not np.isclose(row["wc_current_vol_tilt"], within_class_tilt(sc["wc_current"], sc["wc_current"], GROUPS))


def test_mean_ic_is_spearman_against_return_in_excess_of_the_class_median(world):
    from scipy.stats import spearmanr
    prices, logs, folder, (hist, periods, step) = world
    row = nb.evaluate_world(prices, logs, GROUPS, step, horizon=5, min_assets=6, min_group_assets=3, n_boot=200)
    sc = _assemble(logs)
    ics = []
    for d, day in sc.groupby(level="date"):
        i = prices.index.get_loc(d)
        fwd = np.log(prices.iloc[i + 5] / prices.iloc[i])
        for g in ("g1", "g2"):
            names = [a for a in GROUPS if GROUPS[a] == g]
            excess = fwd[names] - fwd[names].median()
            ics.append(spearmanr(day.loc[(d, names), "wc_current"].to_numpy(), excess.to_numpy())[0])
    assert row["wc_current_mean_ic"] == pytest.approx(np.mean(ics), abs=1e-9)


@pytest.mark.parametrize("old,new,field,expected", [
    ("# refit_today: True", "# refit_today: False", "all_refit_today", False),
    ("# late_fit: False", "# late_fit: True", "any_late_fit", True)])
def test_integrity_flags_read_the_log_headers(world, tmp_path, old, new, field, expected):
    prices, logs, folder, (hist, periods, step) = world
    copies = []
    for k, p in enumerate(logs):
        text = open(p).read()
        if k == 2:
            assert old in text
            text = text.replace(old, new)
        q = tmp_path / p.name
        q.write_text(text)
        copies.append(q)
    row = nb.evaluate_world(prices, copies, GROUPS, step, horizon=5, min_assets=6, min_group_assets=3, n_boot=200)
    assert row[field] is expected


def _shifted(delta):
    base = np.array([-1.0, 1.0] * 4)                          # mean 0, sd 1.069, se = 1.069 / sqrt(8) = 0.378
    return pd.DataFrame([_row(i, ir=base[i] + delta, ic=0.0) for i in range(8)])


def test_judge_boundary_is_two_standard_errors():
    assert nb.judge(_shifted(0.70))["wc_current_ir_ok"]        # 0.70 < 2 * 0.378 = 0.756
    assert not nb.judge(_shifted(0.80))["wc_current_ir_ok"]    # 0.80 > 0.756 (would pass at 3 SE)
    assert nb.judge(_shifted(0.70))["wc_current_ir_se"] == pytest.approx(1.0690449676496976 / np.sqrt(8))
