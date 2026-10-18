"""Tests for noise_test_tier_a.py (simulator for the formal-rule noise test)."""
import numpy as np
import pandas as pd
import pytest

import noise_test_tier_a as nt
from etf_classes import CLASS_OF


@pytest.fixture(scope="module")
def params():
    assets = [a for a in CLASS_OF if a != "SH"]
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.02, 0.10, len(assets))
    return assets, [CLASS_OF[a] for a in assets], sigma


def test_registered_settings():
    assert nt.PERIODS == 36 and nt.DF == 5 and nt.Q == 0.5
    assert nt.TILTS == {"wc_current": 0.5, "wc_no_vol": 0.2, "wc_book": 0.5}
    assert nt.RHO_ARMS == 0.9 and nt.RHO_BOOK == 0.98 and nt.BASE_SEED == 20261017
    assert nt.BETAS == [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30] and nt.N_WORLDS == 2000


def test_world_shape_and_reproducible(params):
    a1, t1 = nt.simulate_world(*params, 0.0, seed=1)
    a2, t2 = nt.simulate_world(*params, 0.0, seed=1)
    assert a1.shape == (36, 4) and set(a1.columns) == {"wc_current", "wc_no_vol", "wc_book", "wc_vol_rule"}
    pd.testing.assert_frame_equal(a1, a2)
    assert t1 == t2
    a3, _ = nt.simulate_world(*params, 0.0, seed=2)
    assert not a1.equals(a3)


def test_vol_tilts_near_configured_values(params):
    tilts = pd.DataFrame([nt.simulate_world(*params, 0.0, seed=s)[1] for s in range(30)])
    m = tilts.mean()
    assert m["wc_current"] == pytest.approx(0.5, abs=0.06)
    assert m["wc_book"] == pytest.approx(0.5, abs=0.06)
    assert m["wc_no_vol"] == pytest.approx(0.2, abs=0.06)
    assert m["wc_no_vol"] < m["wc_current"]


def test_arm_scores_are_close_cousins(params):
    a, _ = nt.simulate_world(*params, 0.0, seed=3)
    # same shared noise: the active returns of current and book are strongly related, no_vol less so
    c = a.corr()
    assert c.loc["wc_current", "wc_book"] > c.loc["wc_current", "wc_no_vol"] - 0.5      # sanity, not tight


def test_no_edge_world_has_mean_ir_near_zero_and_planted_one_is_positive(params):
    zero = np.mean([nt.ev.ir(nt.simulate_world(*params, 0.0, seed=s)[0]["wc_current"]) for s in range(150)])
    planted = np.mean([nt.ev.ir(nt.simulate_world(*params, 0.2, seed=s)[0]["wc_current"]) for s in range(150)])
    assert abs(zero) < 0.15
    assert planted > 1.0
    assert planted > zero + 0.8


def test_vol_rule_never_gets_the_signal(params):
    r = np.mean([nt.ev.ir(nt.simulate_world(*params, 0.3, seed=s)[0]["wc_vol_rule"]) for s in range(100)])
    assert abs(r) < 0.2


def test_universe_params_from_prices():
    idx = pd.bdate_range("2020-01-01", periods=1400)
    rng = np.random.default_rng(1)
    cols = list(CLASS_OF) + ["SH"]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (1400, len(cols))), axis=0)), index=idx, columns=cols)
    assets, classes, sigma = nt.universe_params(px)
    assert "SH" not in assets and len(assets) == 48 == len(classes) == len(sigma)
    assert sigma == pytest.approx(0.01 * np.sqrt(21), rel=0.25)


def test_one_world_row_and_analyze(params):
    rows = [nt.one_world((*params, beta, 100 + i)) for beta in (0.0, 0.1) for i in range(6)]
    df = pd.DataFrame(rows)
    assert {"beta", "seed", "any_edge_found", "wc_book_edge_found"} <= set(df.columns)
    out = nt.analyze(df)
    assert list(out["beta"]) == [0.0, 0.1] and (out["n_worlds"] == 6).all()
    first = df[df.beta == 0.0]
    assert out.loc[0, "any_edge_found"] == first["any_edge_found"].mean()
    p = out.loc[0, "any_edge_found"]
    assert out.loc[0, "any_edge_found_se"] == pytest.approx(np.sqrt(p * (1 - p) / 6))


def test_any_edge_found_is_or_over_arms(params):
    r = nt.one_world((*params, 0.3, 5))
    assert r["any_edge_found"] == any(r[f"{a}_edge_found"] for a in nt.ev.FITTED)


def test_cli_refuses_to_overwrite(tmp_path):
    (tmp_path / "noise_test_tier_a_results.csv").write_text("x")
    assert nt.main(["--prices", "nope.csv", "--folder", str(tmp_path)]) == 1
    assert (tmp_path / "noise_test_tier_a_results.csv").read_text() == "x"


def test_analyze_family_wise_rate_and_se_on_a_constructed_frame():
    base = dict(beta=0.0, seed=1, ir_vol_rule=0.0, wc_current_ir=0.0, wc_current_vol_tilt=0.5)
    rows = []
    for i, flags in enumerate([(True, False, False), (False, False, False), (False, True, False), (False, False, False)]):
        r = dict(base, seed=i)
        for arm, flag in zip(nt.ev.FITTED, flags):
            r.update({f"{arm}_adds_beyond_volatility": flag, f"{arm}_positive_alone": flag, f"{arm}_edge_found": flag})
        r["any_edge_found"] = any(flags)
        rows.append(r)
    out = nt.analyze(pd.DataFrame(rows)).iloc[0]
    assert out["any_edge_found"] == 0.5 and out["wc_current_edge_found"] == 0.25
    assert out["any_edge_found_se"] == pytest.approx(0.25)          # sqrt(0.5 * 0.5 / 4)


def test_one_world_any_edge_is_an_or_not_an_and(params, monkeypatch):
    fake = {a: dict(ir=0.0, paired_low=0.0, standalone_low=0.0, vol_tilt=0.1, adds_beyond_volatility=False,
                    positive_alone=False, edge_found=(a == "wc_no_vol")) for a in nt.ev.FITTED}
    monkeypatch.setattr(nt.ev, "decide_formal", lambda active, tilt: fake)
    assert nt.one_world((*params, 0.0, 1))["any_edge_found"] is True
