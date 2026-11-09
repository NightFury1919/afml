"""Tests for run_momentum_stop.py: freeze guard, run-once rule, verdict logic, and an end-to-end run on made-up prices."""
import json

import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml
import run_momentum_stop as rs
import run_momentum_ladder as rl


def fake_world(tmp_path, rows=700, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=rows)
    cols = [f"E{i}" for i in range(24)] + ["SHY", "SH", "SPY"]
    steps = rng.normal(0.0004, 0.012, size=(rows, len(cols)))
    df = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    f = tmp_path / "prices_daily_asof_test.csv"
    df.to_csv(f)
    return f


def fake_vol_fn(close, span0=100):
    return close.pct_change().ewm(span=span0).std()


def prereg(tmp_path, sha, status="FROZEN"):
    p = tmp_path / "prereg.md"
    p.write_text(f"# x\n\nStatus: {status}\n\nSHA-256: {sha}\n")
    return p


def test_refuses_when_not_frozen(tmp_path):
    f = fake_world(tmp_path)
    p = prereg(tmp_path, rl.sha256_of(f), status="DRAFT")
    with pytest.raises(RuntimeError, match="not frozen"):
        rs.run(f, p, tmp_path / "out", tmp_path, get_daily_vol=fake_vol_fn)
    assert not (tmp_path / "out").exists()


def test_refuses_when_price_file_differs(tmp_path):
    f = fake_world(tmp_path)
    p = prereg(tmp_path, "0" * 64)
    with pytest.raises(RuntimeError, match="different"):
        rs.run(f, p, tmp_path / "out", tmp_path, get_daily_vol=fake_vol_fn)


def test_end_to_end_then_refuses_second_run(tmp_path):
    f = fake_world(tmp_path)
    p = prereg(tmp_path, rl.sha256_of(f))
    out = tmp_path / "out"
    r = rs.run(f, p, out, tmp_path, n_resamples=200, get_daily_vol=fake_vol_fn)
    saved = json.loads((out / "results.json").read_text())
    assert saved["price_sha256"] == rl.sha256_of(f)
    assert set(saved["verdict"]) == {"risk_eligible", "beats_spy", "stop_adds_value", "recommended_for_live_trial"}
    assert "C50s" in (out / "report.txt").read_text()
    for label in ("main_cost", "zero_cost"):
        assert saved[label]["full"]["C50s"]["n_windows"] > 0
        assert saved[label]["stops"]["n_stops"] >= 0
    with pytest.raises(FileExistsError):
        rs.run(f, p, out, tmp_path, n_resamples=200, get_daily_vol=fake_vol_fn)
    assert r["verdict"] == saved["verdict"]


def test_spy_is_not_in_the_universe_but_shy_is_exactly_as_in_the_ladder(tmp_path):
    f = fake_world(tmp_path)
    etfs, spy = rl.load_prices(f)
    assert "SPY" not in etfs.columns and "SH" not in etfs.columns and "SHY" in etfs.columns


def full(touch_s, touch_spy, worst, ci_spy, ci_base):
    return {"C50s": {"touch_floor": touch_s, "worst": worst}, "SPY": {"touch_floor": touch_spy},
            "C50s_minus_SPY": {"ci": ci_spy}, "C50s_minus_C50": {"ci": ci_base}}


def test_verdict_rules_hand_checked():
    v = rs.verdict(full(0.10, 0.06, -0.30, (0.01, 0.05), (-0.01, 0.02)))
    assert v == {"risk_eligible": True, "beats_spy": True, "stop_adds_value": False, "recommended_for_live_trial": True}
    # touch rate 0.13 > 2 x 0.06
    assert rs.verdict(full(0.13, 0.06, -0.30, (0.01, 0.05), (0.01, 0.02)))["recommended_for_live_trial"] is False
    # worst window exactly at the limit is allowed; just below is not
    assert rs.verdict(full(0.05, 0.06, -0.50, (-0.01, 0.05), (0.01, 0.02)))["risk_eligible"] is True
    assert rs.verdict(full(0.05, 0.06, -0.51, (0.01, 0.05), (0.01, 0.02)))["risk_eligible"] is False
    # interval lower end exactly 0 is not "beats"
    assert rs.verdict(full(0.05, 0.06, -0.1, (0.0, 0.05), (0.0, 0.02)))["beats_spy"] is False


def test_compare_hand_checked_mean_times_twelve():
    # a grows 2% and b grows 1% every 21 rows, so every period difference is exactly 0.01 and the mean x 12 is 0.12
    k = np.arange(41)
    a = pd.Series(np.repeat(1.02 ** k, 21)[: 21 * 40 + 1])
    b = pd.Series(np.repeat(1.01 ** k, 21)[: 21 * 40 + 1])
    out = rs.compare(a, b, n_resamples=100)
    assert out["n_periods"] == 40
    assert out["annual_diff"] == pytest.approx(0.12)
    lo, hi = out["ci"]
    assert lo <= 0.12 + 1e-9 <= hi + 1e-9


def test_cut_rebases_to_one():
    s = pd.Series([2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=3))
    c = rs.cut(s, "2020-01-02")
    assert c.iloc[0] == 1.0 and c.iloc[1] == pytest.approx(4.0 / 3.0)
