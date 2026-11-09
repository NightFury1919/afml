"""Tests for run_momentum_variants.py: freeze guards, run-once rule, verdict logic, end-to-end on made-up prices."""
import json

import numpy as np
import pandas as pd
import pytest

import run_momentum_ladder as rl
import run_momentum_variants as rv


def fake_world(tmp_path, rows=700, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=rows)
    cols = [f"E{i}" for i in range(24)] + ["SHY", "SH", "SPY"]
    steps = rng.normal(0.0004, 0.012, size=(rows, len(cols)))
    df = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)
    f = tmp_path / "prices_daily_asof_test.csv"
    df.to_csv(f)
    groups = {f"E{i}": f"g{i % 5}" for i in range(24)}
    groups["SHY"] = "cash"
    g = tmp_path / "groups.json"
    g.write_text(json.dumps(groups))
    return f, g


def prereg(tmp_path, price_sha, groups_sha, status="FROZEN"):
    p = tmp_path / "prereg.md"
    p.write_text(f"# x\n\nStatus: {status}\n\nSHA-256: {price_sha}\nGroups file SHA-256: {groups_sha}\n")
    return p


def test_refuses_when_not_frozen(tmp_path):
    f, g = fake_world(tmp_path)
    p = prereg(tmp_path, rl.sha256_of(f), rl.sha256_of(g), status="DRAFT")
    with pytest.raises(RuntimeError, match="not frozen"):
        rv.run(f, g, p, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_refuses_when_price_file_differs(tmp_path):
    f, g = fake_world(tmp_path)
    p = prereg(tmp_path, "0" * 64, rl.sha256_of(g))
    with pytest.raises(RuntimeError, match="different"):
        rv.run(f, g, p, tmp_path / "out")


def test_refuses_when_group_table_differs(tmp_path):
    f, g = fake_world(tmp_path)
    p = prereg(tmp_path, rl.sha256_of(f), rl.sha256_of(g))
    g.write_text(g.read_text().replace("g1", "g2"))                  # someone edits the table after the freeze
    with pytest.raises(RuntimeError, match="group table"):
        rv.run(f, g, p, tmp_path / "out")


def test_refuses_when_no_group_hash_line(tmp_path):
    f, g = fake_world(tmp_path)
    p = tmp_path / "prereg.md"
    p.write_text(f"Status: FROZEN\n\nSHA-256: {rl.sha256_of(f)}\n")
    with pytest.raises(RuntimeError, match="Groups file"):
        rv.run(f, g, p, tmp_path / "out")


def test_end_to_end_then_refuses_second_run(tmp_path):
    f, g = fake_world(tmp_path)
    p = prereg(tmp_path, rl.sha256_of(f), rl.sha256_of(g))
    out = tmp_path / "out"
    r = rv.run(f, g, p, out, n_resamples=200)
    saved = json.loads((out / "results.json").read_text())
    assert saved["price_sha256"] == rl.sha256_of(f) and saved["groups_sha256"] == rl.sha256_of(g)
    assert set(saved["verdict"]) == {"trend", "cap"}
    for v in saved["verdict"].values():
        assert set(v) == {"risk_eligible", "beats_spy", "adds_value_over_C50", "recommended_for_live_trial"}
    txt = (out / "report.txt").read_text()
    assert "trend" in txt and "cap" in txt and "Two variants" in txt
    assert saved["main_cost"]["full"]["trend"]["n_windows"] > 0
    with pytest.raises(FileExistsError):
        rv.run(f, g, p, out, n_resamples=200)
    assert r["verdict"] == saved["verdict"]


def test_bad_group_table_is_refused(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError):
        rv.load_groups(g)
    g.write_text(json.dumps({"A": 1}))
    with pytest.raises(ValueError):
        rv.load_groups(g)


def full(name, touch, spy_touch, worst, ci_spy, ci_c50):
    return {name: {"touch_floor": touch, "worst": worst}, "SPY": {"touch_floor": spy_touch},
            f"{name}_minus_SPY": {"ci": ci_spy}, f"{name}_minus_C50": {"ci": ci_c50}}


def test_verdict_rules_hand_checked():
    v = rv.verdict(full("trend", 0.10, 0.06, -0.30, (0.01, 0.05), (-0.01, 0.02)), "trend")
    assert v == {"risk_eligible": True, "beats_spy": True, "adds_value_over_C50": False, "recommended_for_live_trial": True}
    assert rv.verdict(full("cap", 0.13, 0.06, -0.30, (0.01, 0.05), (0.01, 0.02)), "cap")["risk_eligible"] is False
    assert rv.verdict(full("cap", 0.05, 0.06, -0.50, (0.0, 0.05), (0.0, 0.02)), "cap") == {
        "risk_eligible": True, "beats_spy": False, "adds_value_over_C50": False, "recommended_for_live_trial": False}
    assert rv.verdict(full("cap", 0.05, 0.06, -0.51, (0.01, 0.05), (0.01, 0.02)), "cap")["risk_eligible"] is False


def test_compare_hand_checked_mean_times_twelve():
    k = np.arange(41)
    a = pd.Series(np.repeat(1.02 ** k, 21)[: 21 * 40 + 1])
    b = pd.Series(np.repeat(1.01 ** k, 21)[: 21 * 40 + 1])
    out = rv.compare(a, b, n_resamples=100)
    assert out["n_periods"] == 40
    assert out["annual_diff"] == pytest.approx(0.12)


def test_cut_rebases_to_one():
    s = pd.Series([2.0, 3.0, 4.0], index=pd.bdate_range("2020-01-01", periods=3))
    c = rv.cut(s, "2020-01-02")
    assert c.iloc[0] == 1.0 and c.iloc[1] == pytest.approx(4.0 / 3.0)
