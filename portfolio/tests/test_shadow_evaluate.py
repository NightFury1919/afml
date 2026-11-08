"""Tests for shadow_evaluate.py (the registered formal evaluation rule)."""
import numpy as np
import pandas as pd
import pytest

import real_history_check as rh
import real_history_check_wc as wc
import shadow_evaluate as ev


def series(x):
    return pd.Series(x, index=pd.date_range("2026-01-01", periods=len(x), freq="21B"))


def frame(rng, n=36, edge=None, noise=0.01):
    cols = {a: rng.normal(0, noise, n) for a in ev.ARMS}
    for a, e in (edge or {}).items():
        cols[a] = cols[a] + e
    return pd.DataFrame(cols, index=pd.date_range("2026-01-01", periods=n, freq="21B"))


def test_constants_match_registration():
    assert ev.FORMAL_PERIODS == 36 and ev.FORMAL_N_BOOT == 20000 and ev.BOOT_BLOCK == 4 and ev.BOOT_SEED == 20261009
    assert ev.FORMAL_ALPHA == pytest.approx(0.05 / 3)
    assert ev.FORMAL_LEVEL == pytest.approx(0.966667, abs=1e-6)
    assert ev.FITTED == ["wc_current", "wc_no_vol", "wc_book"] and ev.REFERENCE == "wc_vol_rule"
    assert ev.MAX_VOL_TILT == 0.7 and ev.HORIZON == 21


def test_vectorised_standalone_equals_the_loop_version():
    x = series(np.random.default_rng(1).normal(0.002, 0.01, 36))
    loop = wc.block_bootstrap_ir_ci(x, 21, block=4, n_boot=500, seed=20261009, level=ev.FORMAL_LEVEL)
    fast = ev.standalone_low(x, n_boot=500)
    assert fast == pytest.approx(loop, abs=1e-12)


def test_vectorised_paired_equals_the_loop_version():
    r = np.random.default_rng(2)
    a, b = series(r.normal(0.003, 0.01, 36)), series(r.normal(0.0, 0.01, 36))
    loop = rh.paired_block_bootstrap_ci(a, b, 21, block=4, n_boot=500, seed=20261009, level=ev.FORMAL_LEVEL)
    fast = ev.paired_low(a, b, n_boot=500)
    assert fast == pytest.approx(loop, abs=1e-12)


def test_equality_also_holds_for_a_length_not_divisible_by_the_block():
    x = series(np.random.default_rng(3).normal(0.001, 0.01, 37))
    loop = wc.block_bootstrap_ir_ci(x, 21, block=4, n_boot=300, seed=20261009, level=ev.FORMAL_LEVEL)
    assert ev.standalone_low(x, n_boot=300) == pytest.approx(loop, abs=1e-12)


def test_rows_are_blocks_of_four_consecutive_periods():
    rows = ev.resample_rows(36, n_boot=50)
    assert rows.shape == (50, 36) and rows.min() >= 0 and rows.max() <= 35
    assert (np.diff(rows[:, :4], axis=1) == 1).all()          # first block consecutive
    assert (np.diff(rows[:, 4:8], axis=1) == 1).all()


def test_interval_level_puts_the_lower_end_at_1_67_percentile():
    x = np.random.default_rng(4).normal(0.002, 0.01, 36)
    irs = ev._irs(x, ev.resample_rows(36, n_boot=2000), 21)
    lo, _ = ev.standalone_low(x, n_boot=2000)
    assert lo == pytest.approx(np.percentile(irs, 100 * ev.FORMAL_ALPHA))


def test_clear_edge_in_one_arm_is_found_only_there():
    rng = np.random.default_rng(5)
    f = frame(rng, edge={"wc_current": 0.01})                # ir far above 1
    res = ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=2000)
    assert res["wc_current"]["edge_found"] and res["wc_current"]["adds_beyond_volatility"]
    assert not res["wc_no_vol"]["edge_found"] and not res["wc_book"]["edge_found"]


def test_pure_noise_finds_nothing_for_a_fixed_world():
    f = frame(np.random.default_rng(6))
    res = ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=2000)
    assert not any(v["edge_found"] for v in res.values())


def test_vol_tilt_above_limit_blocks_adds_beyond_volatility():
    f = frame(np.random.default_rng(7), edge={"wc_current": 0.01})
    res = ev.decide_formal(f, {"wc_current": 0.71, "wc_no_vol": 0.1, "wc_book": 0.1}, n_boot=2000)
    assert res["wc_current"]["positive_alone"] and not res["wc_current"]["adds_beyond_volatility"]
    assert not res["wc_current"]["edge_found"]
    res = ev.decide_formal(f, {"wc_current": 0.70, "wc_no_vol": 0.1, "wc_book": 0.1}, n_boot=2000)
    assert res["wc_current"]["adds_beyond_volatility"]       # 0.7 itself is allowed ("at most 0.7")


def test_edge_equal_to_the_vol_rule_is_not_adding_beyond_it():
    rng = np.random.default_rng(8)
    f = frame(rng)
    shared = rng.normal(0.01, 0.01, 36)                       # arm and reference both earn the same big edge
    f["wc_current"] = shared + rng.normal(0, 0.001, 36)
    f["wc_vol_rule"] = shared
    res = ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=2000)
    assert res["wc_current"]["positive_alone"] and not res["wc_current"]["adds_beyond_volatility"]


def test_positive_alone_needs_a_positive_lower_end():
    f = frame(np.random.default_rng(9), edge={"wc_current": -0.01})
    res = ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=2000)
    assert not res["wc_current"]["positive_alone"]


def test_uses_exactly_the_first_36_periods():
    rng = np.random.default_rng(10)
    f = frame(rng, n=50)
    a = ev.decide_formal(f, {x: 0.2 for x in ev.FITTED}, n_boot=500)
    f2 = f.copy()
    f2.iloc[36:] = 9.0                                        # later periods must not matter
    b = ev.decide_formal(f2, {x: 0.2 for x in ev.FITTED}, n_boot=500)
    assert a == b


def test_refuses_fewer_than_36_periods_and_missing_columns_and_nan():
    f = frame(np.random.default_rng(11), n=35)
    with pytest.raises(ValueError, match="needs 36"):
        ev.decide_formal(f, {x: 0.2 for x in ev.FITTED})
    g = frame(np.random.default_rng(12)).drop(columns="wc_book")
    with pytest.raises(ValueError, match="lack columns"):
        ev.decide_formal(g, {x: 0.2 for x in ev.FITTED})
    h = frame(np.random.default_rng(13))
    h.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ev.decide_formal(h, {x: 0.2 for x in ev.FITTED})


def test_paired_rejects_different_lengths():
    with pytest.raises(ValueError):
        ev.paired_low([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4])


def test_ir_known_value():
    x = [0.01, 0.03]                                          # mean 0.02, sd 0.0141421
    assert ev.ir(x) == pytest.approx(0.02 / np.std(x, ddof=1) * np.sqrt(12.0))


def test_reported_lows_are_the_lower_ends_of_the_same_intervals():
    f = frame(np.random.default_rng(14), edge={"wc_current": 0.002})
    res = ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=800)
    act = f[ev.ARMS].iloc[:36]
    assert res["wc_current"]["standalone_low"] == ev.standalone_low(act["wc_current"], n_boot=800)[0]
    assert res["wc_current"]["paired_low"] == ev.paired_low(act["wc_current"], act["wc_vol_rule"], n_boot=800)[0]
    lo, hi = ev.standalone_low(act["wc_current"], n_boot=800)
    assert lo < hi
    assert res["wc_current"]["positive_alone"] == (lo > 0)


def test_positive_alone_follows_the_lower_end_not_the_upper_end():
    # a series with a small positive mean and big noise: upper end above 0, lower end below 0
    x = np.random.default_rng(15).normal(0.0005, 0.02, 36)
    f = frame(np.random.default_rng(16))
    f["wc_current"] = x
    lo, hi = ev.standalone_low(x, n_boot=2000)
    assert lo < 0 < hi
    assert not ev.decide_formal(f, {a: 0.2 for a in ev.FITTED}, n_boot=2000)["wc_current"]["positive_alone"]
