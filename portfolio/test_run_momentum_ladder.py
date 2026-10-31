"""Tests for run_momentum_ladder.py, written BEFORE the code (TDD).

Synthetic worlds with answers worked out by hand. Nothing here touches the real price file.
Run from the portfolio folder in the mlfinlab env:
    python -m pytest test_run_momentum_ladder.py -v
"""
import json
import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml
import run_momentum_ladder as rn

GOOD_SHA = "a" * 64


# ---------- helpers ----------
def world(n_days=700, n_etf=24):
    """24 ETFs growing at different steady rates, plus SPY at 0.03% a day."""
    idx = pd.bdate_range("2018-01-01", periods=n_days)
    rates = {f"E{i:02d}": 0.0002 + 0.0001 * i for i in range(n_etf)}
    etfs = pd.DataFrame({k: (1 + g) ** np.arange(n_days) for k, g in rates.items()}, index=idx)
    spy = pd.Series(1.0003 ** np.arange(n_days), index=idx, name="SPY")
    return etfs, spy


def write_prices(tmp_path, etfs, spy, extra=("SH",)):
    df = etfs.copy()
    df.insert(0, "SPY", spy)
    for e in extra:
        df[e] = 1.0
    p = tmp_path / "prices_daily_asof_2099-01-01.csv"
    df.index.name = "date"
    df.to_csv(p)
    return p


# ---------- loading ----------
def test_load_prices_drops_spy_and_sh_from_universe(tmp_path):
    etfs, spy = world()
    p = write_prices(tmp_path, etfs, spy)
    u, s = rn.load_prices(p)
    assert "SPY" not in u.columns and "SH" not in u.columns
    assert list(u.columns) == list(etfs.columns)
    assert np.allclose(s.values, spy.values)
    assert isinstance(u.index, pd.DatetimeIndex)


def test_newest_price_file_picks_latest_date(tmp_path):
    for d in ("2026-10-19", "2026-10-30", "2026-10-02"):
        (tmp_path / f"prices_daily_asof_{d}.csv").write_text("date,SPY\n")
    assert rn.newest_price_file(tmp_path).name == "prices_daily_asof_2026-10-30.csv"


def test_sha256_of_known_bytes(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert rn.sha256_of(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# ---------- the freeze guard ----------
def test_refuses_when_preregistration_is_a_draft():
    text = f"Status: **DRAFT, not frozen.**\nSHA-256: {GOOD_SHA}\n"
    with pytest.raises(RuntimeError, match="not frozen"):
        rn.check_frozen(text, GOOD_SHA)


def test_draft_that_merely_mentions_the_frozen_line_is_still_refused():
    text = (f"Status: **DRAFT.** To freeze, change this line to exactly `Status: FROZEN`.\n"
            f"SHA-256: {GOOD_SHA}\n")
    with pytest.raises(RuntimeError, match="not frozen"):
        rn.check_frozen(text, GOOD_SHA)


def test_refuses_when_hash_line_missing():
    with pytest.raises(RuntimeError, match="SHA-256"):
        rn.check_frozen("Status: FROZEN\n", GOOD_SHA)


def test_refuses_when_price_file_differs_from_the_frozen_hash():
    text = f"Status: FROZEN\nSHA-256: {'b' * 64}\n"
    with pytest.raises(RuntimeError, match="different"):
        rn.check_frozen(text, GOOD_SHA)


def test_accepts_frozen_with_matching_hash():
    rn.check_frozen(f"Status: FROZEN\nSHA-256: {GOOD_SHA}\n", GOOD_SHA)


# ---------- small statistics ----------
def test_max_drawdown_hand_checked():
    v = pd.Series([1.0, 2.0, 1.0, 1.5, 0.9, 3.0])
    assert rn.max_drawdown(v) == pytest.approx(0.9 / 2.0 - 1.0)


def test_annual_stats_constant_growth():
    v = pd.Series(1.001 ** np.arange(253))                  # 252 steps of +0.1%
    s = rn.annual_stats(v)
    assert s["ann_return"] == pytest.approx(1.001 ** 252 - 1.0)
    assert s["ann_vol"] == pytest.approx(0.0, abs=1e-12)
    assert s["max_drawdown"] == 0.0


def test_beat_spy_share_and_average():
    # strategy ends windows +10%, +10%, +0%; SPY +5%, +15%, +0%  (window = 1 step here)
    strat = pd.Series([1.0, 1.1, 1.21, 1.21])
    spy = pd.Series([1.0, 1.05, 1.2075, 1.2075])
    share, avg = rn.beat_spy(strat, spy, window=1)
    assert share == pytest.approx(1 / 3)                     # only window 1 is strictly better
    assert avg == pytest.approx(((0.10 - 0.05) + (0.10 - 0.15) + 0.0) / 3)


# ---------- evaluation ----------
def test_evaluate_reports_both_floors_and_bootstrap_for_a_flat_world():
    n = 200
    v = pd.Series(np.ones(n))
    s = rn.evaluate(v, v.copy(), window=35, n_resamples=200, seed=1)
    assert s["touch_floor"] == 0.0 and s["touch_hard"] == 0.0
    assert s["median"] == 0.0
    assert s["diff_ci"] == (0.0, 0.0)


def test_evaluate_hard_floor_vs_soft_floor():
    # falls 20% then flat: soft floor (-10%) touched by early windows, hard floor (-50%) never
    v = pd.Series(np.r_[np.linspace(1.0, 0.8, 20), np.full(100, 0.8)])
    s = rn.evaluate(v, pd.Series(np.ones(len(v))), window=35, n_resamples=100, seed=1)
    assert s["touch_floor"] > 0.0
    assert s["touch_hard"] == 0.0


# ---------- the decision rules ----------
def fake(touch, worst, median, lo):
    return {"touch_floor": touch, "worst": worst, "median": median, "diff_ci": (lo, lo + 0.1)}


def test_decide_risk_eligibility_and_beats():
    spy = fake(0.05, -0.2, 0.02, 0.0)
    res = {"SPY": spy,
           "D10": fake(0.04, -0.20, 0.03, 0.01),    # eligible, higher median, beats
           "D25": fake(0.12, -0.30, 0.04, -0.02),   # eligible (<= 0.10? no: limit is 2x5% = 10%)
           "D50": fake(0.08, -0.55, 0.05, 0.02),    # worst below -50%: not eligible
           "D100": fake(0.09, -0.40, 0.01, 0.03)}   # eligible but median below SPY
    out = rn.decide(res, worst_limit=-0.50, touch_multiple=2.0)
    assert out["D10"] == {"risk_eligible": True, "beats_spy": True, "median_above_spy": True}
    assert out["D25"]["risk_eligible"] is False      # 12% > 10%
    assert out["D50"]["risk_eligible"] is False
    assert out["D100"]["risk_eligible"] is True and out["D100"]["median_above_spy"] is False
    assert out["recommended"] == {"D": "D10", "C": None}


def test_decide_recommends_largest_qualifying_rung_per_sleeve():
    spy = fake(0.05, -0.2, 0.02, 0.0)
    res = {"SPY": spy}
    for name, med in (("C10", 0.03), ("C25", 0.04), ("C50", 0.05), ("C100", 0.01)):
        res[name] = fake(0.05, -0.3, med, 0.0)
    out = rn.decide(res, worst_limit=-0.50, touch_multiple=2.0)
    assert out["recommended"]["C"] == "C50"          # C100 median is below SPY
    assert out["recommended"]["D"] is None


# ---------- the full run on a made-up world ----------
def test_run_all_builds_every_variant_range_and_cost():
    etfs, spy = world()
    res = rn.run_all(etfs, spy, since="2019-09-01", n_resamples=100, seed=3)
    names = {"SPY"} | {f"{s}{int(r * 100)}" for s in "DC" for r in ml.RUNGS}
    assert set(res["full"]["cost"]) == names and len(names) == 9
    assert set(res["since"]["nocost"]) == names
    # costs can only lower the median of a sleeve that trades
    assert res["full"]["cost"]["D100"]["ann_return"] <= res["full"]["nocost"]["D100"]["ann_return"]
    # SPY benchmark is the zero-sleeve simulation: the same list in both sleeves
    assert res["full"]["nocost"]["SPY"]["median"] == pytest.approx(
        ml.window_summary(ml.simulate(etfs, spy, 10, 0.0, cost_bps=0.0).value)["median"])
    # the made-up world has steady winners: a bigger sleeve means a higher growth rate
    assert (res["full"]["nocost"]["D100"]["ann_return"] > res["full"]["nocost"]["D10"]["ann_return"])


def test_since_range_starts_at_or_after_the_date():
    etfs, spy = world()
    res = rn.run_all(etfs, spy, since="2019-09-01", n_resamples=50, seed=3)
    assert pd.Timestamp(res["since_start"]) >= pd.Timestamp("2019-09-01")
    assert res["since"]["cost"]["SPY"]["n_windows"] < res["full"]["cost"]["SPY"]["n_windows"]


# ---------- writing and refusing to run twice ----------
def test_write_results_then_refuses_second_run(tmp_path):
    out = tmp_path / "momentum_results"
    rn.write_results(out, {"x": 1}, "report text")
    assert json.loads((out / "results.json").read_text()) == {"x": 1}
    assert (out / "report.txt").read_text() == "report text"
    with pytest.raises(FileExistsError):
        rn.write_results(out, {"x": 2}, "again")
    assert json.loads((out / "results.json").read_text()) == {"x": 1}   # untouched


def test_main_refuses_a_draft_before_touching_anything(tmp_path):
    etfs, spy = world()
    p = write_prices(tmp_path, etfs, spy)
    pre = tmp_path / "pre.md"
    pre.write_text("Status: **DRAFT**\n")
    with pytest.raises(RuntimeError, match="not frozen"):
        rn.main(["--prices", str(p), "--prereg", str(pre), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_run_constants_match_preregistration():
    assert rn.SEED == 20261028
    assert rn.N_RESAMPLES == 5000
    assert rn.BLOCK == 4
    assert rn.LEVEL == 0.90
    assert rn.SINCE == "2016-01-04"
    assert rn.HARD_FLOOR == -0.50
    assert rn.WORST_LIMIT == -0.50
    assert rn.TOUCH_MULTIPLE == 2.0
