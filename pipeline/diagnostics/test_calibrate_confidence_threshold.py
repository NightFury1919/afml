"""
pipeline/diagnostics/test_calibrate_confidence_threshold.py

TDD for the budget-based zero-exposure anchor. Known values are hand-derived
(see comments); synthetic data is fine here because these are unit tests.

Run:  python -m pytest pipeline/diagnostics/test_calibrate_confidence_threshold.py -v
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'calibrate_confidence_threshold', os.path.join(HERE, 'calibrate_confidence_threshold.py'))
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def test_exposure_scalar_known_values():
    # anchor 0.5: 0.4 -> 0 ; 0.5 -> 0 ; 0.75 -> 0.5 ; 1.0 -> 1 ; capped at 1
    out = cal.exposure_scalar([0.4, 0.5, 0.75, 1.0, 1.2], 0.5)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_expected_null_exposure_hand_computed():
    # c = [0, 0.8], a = 0.5 -> scalars [0, 0.3/0.5 = 0.6] -> mean 0.3
    assert cal.expected_null_exposure([0.0, 0.8], 0.5) == pytest.approx(0.3)


def test_solve_anchor_known_value():
    # c = [0, 0.8]. For a < 0.8: E(a) = 0.5*(0.8-a)/(1-a).
    # E = 0.1  ->  (0.8-a)/(1-a) = 0.2  ->  0.6 = 0.8a  ->  a = 0.75
    assert cal.solve_anchor([0.0, 0.8], 0.1) == pytest.approx(0.75, abs=1e-9)


def test_solve_anchor_hits_the_budget_on_random_data():
    rng = np.random.default_rng(0)
    conf = rng.beta(2, 5, 200)
    for budget in (0.01, 0.03, 0.08):
        a = cal.solve_anchor(conf, budget)
        assert cal.expected_null_exposure(conf, a) == pytest.approx(budget, abs=1e-9)


def test_lower_budget_means_higher_anchor():
    rng = np.random.default_rng(1)
    conf = rng.beta(2, 4, 100)
    anchors = [cal.solve_anchor(conf, b) for b in (0.10, 0.05, 0.03, 0.01)]
    assert anchors == sorted(anchors)
    assert len(set(anchors)) == 4


def test_budget_edge_cases():
    conf = np.array([0.1, 0.3, 0.5])
    assert cal.solve_anchor(conf, 0.0) == pytest.approx(0.5)      # zero budget -> above the max
    assert cal.solve_anchor(conf, conf.mean()) == 0.0             # budget >= mean -> anchor 0
    assert cal.solve_anchor(conf, 0.99) == 0.0


def test_cluster_bootstrap_reproducible_and_bracketed():
    draws = pd.DataFrame({'confidence': [0.05, 0.10, 0.6, 0.7, 0.2, 0.15],
                          'source': ['a', 'a', 'b', 'b', 'c', 'c']})
    b1 = cal.cluster_bootstrap_anchor(draws, 0.03, n_boot=300, seed=7)
    b2 = cal.cluster_bootstrap_anchor(draws, 0.03, n_boot=300, seed=7)
    np.testing.assert_array_equal(b1, b2)
    assert b1.min() >= 0.0 and b1.max() <= 0.7
    assert b1.std() > 0                                           # sources disagree -> real spread


def test_cluster_bootstrap_resamples_whole_sources_not_rows():
    # Source 'hi' is all 0.9, 'lo' all 0.1. A resample of whole sources can only
    # contain values from {0.1, 0.9}; a row bootstrap of a mixed pool could not
    # produce a pool that is ALL 0.9 -- the cluster one does when 'hi' is drawn twice.
    draws = pd.DataFrame({'confidence': [0.9, 0.9, 0.1, 0.1], 'source': ['hi', 'hi', 'lo', 'lo']})
    b = cal.cluster_bootstrap_anchor(draws, 0.03, n_boot=400, seed=3)
    all_hi = cal.solve_anchor(np.array([0.9, 0.9, 0.9, 0.9]), 0.03)
    assert np.isclose(b, all_hi).any()


def test_load_null_draws_dedupes_skips_and_computes_confidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'dsr': [0.5, 0.5, 0.8], 'pbo': [0.2, 0.2, 0.5]}).to_csv('a.csv', index=False)
    pd.DataFrame({'dsr': [0.9, np.nan], 'pbo': [0.1, 0.3]}).to_csv('b.csv', index=False)
    pd.DataFrame({'x': [1]}).to_csv('c.csv', index=False)          # no dsr/pbo -> skipped
    d = cal.load_null_draws(['a.csv', 'b.csv', 'c.csv', 'missing.csv'], verbose=False)
    assert len(d) == 3                                             # a: 2 unique, b: 1 non-NaN
    np.testing.assert_allclose(sorted(d['confidence']), sorted([0.5 * 0.8, 0.8 * 0.5, 0.9 * 0.9]))
    assert set(d['source']) == {'a.csv', 'b.csv'}


def test_calibrate_end_to_end_writes_consistent_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(5)
    for i in range(4):
        pd.DataFrame({'dsr': rng.uniform(0.3, 0.8, 5), 'pbo': rng.uniform(0, 0.9, 5)}
                     ).to_csv(f's{i}.csv', index=False)
    out = tmp_path / 'out'
    out.mkdir()
    r = cal.calibrate([f's{i}.csv' for i in range(4)], budget=0.03, n_boot=50,
                      out_dir=str(out), verbose=False)
    written = float((out / 'confidence_threshold.txt').read_text().strip())
    assert written == pytest.approx(r['anchor'])
    assert r['expected_null_exposure'] == pytest.approx(0.03, abs=1e-9)
    q05, q50, q95 = r['boot_q05_q50_q95']
    assert q05 <= q50 <= q95
    assert (out / 'confidence_threshold_calibration.csv').exists()


def test_legacy_sigma_mode_reproduces_old_rule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame({'dsr': [0.5, 0.6, 0.7, 0.8], 'pbo': [0.0, 0.0, 0.0, 0.0]}).to_csv('a.csv', index=False)
    r = cal.calibrate(['a.csv'], n_sigma=2.0, out_dir=str(tmp_path), verbose=False)
    conf = np.array([0.5, 0.6, 0.7, 0.8])
    assert r['anchor'] == pytest.approx(conf.mean() + 2 * conf.std(ddof=1))
    assert r['boot_q05_q50_q95'] is None


def test_default_sweep_csvs_are_found_from_any_working_directory(tmp_path, monkeypatch):
    # The Kraken/Binance sweep CSVs moved from the repo root into results/kraken_calibration/
    # (2026-11-10). The defaults must not depend on the current folder. Real tracked data.
    monkeypatch.chdir(tmp_path)
    assert all(os.path.isabs(f) and f.startswith(cal.RESULTS_DIR) for f in cal.DEFAULT_CSVS)
    assert all(os.path.exists(f) for f in cal.DEFAULT_CSVS)
    d = cal.load_null_draws(cal.DEFAULT_CSVS, verbose=False)
    assert len(d) == 50 and d['source'].nunique() == 10           # windows 1-8, binance, pooled x 5 rows


def test_pool_script_default_pattern_finds_all_20_windows_from_any_directory(tmp_path, monkeypatch):
    import glob
    monkeypatch.chdir(tmp_path)
    spec_p = importlib.util.spec_from_file_location(
        'pool_kraken_pbo_across_windows', os.path.join(HERE, 'pool_kraken_pbo_across_windows.py'))
    pool = importlib.util.module_from_spec(spec_p)
    spec_p.loader.exec_module(pool)
    assert os.path.dirname(pool.DEFAULT_PATTERN) == cal.RESULTS_DIR
    assert len(glob.glob(pool.DEFAULT_PATTERN)) == 20


# =============================================================================
# TDD RESULTS (synthetic + tmp_path data; mlfinlab env, Python 3.10.20, pytest 9.0.3)
# 11 passed in 6.48s  (run 2026-09-19)
# 13 passed in 4.33s  (run 2026-11-10, sandbox Python 3.11, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1; re-run on Windows)
#   2026-11-10: sweep CSVs moved from the repo root to results/kraken_calibration/; two tests added.
#   Both new tests FAIL on the pre-move scripts (no RESULTS_DIR) and pass after. Output of
#   pool_kraken_pbo_across_windows.py and calibrate_confidence_threshold.py is byte-identical before and after.
# =============================================================================
# test_exposure_scalar_known_values PASSED
# test_expected_null_exposure_hand_computed PASSED
# test_solve_anchor_known_value PASSED
# test_solve_anchor_hits_the_budget_on_random_data PASSED
# test_lower_budget_means_higher_anchor PASSED
# test_budget_edge_cases PASSED
# test_cluster_bootstrap_reproducible_and_bracketed PASSED
# test_cluster_bootstrap_resamples_whole_sources_not_rows PASSED
# test_load_null_draws_dedupes_skips_and_computes_confidence PASSED
# test_calibrate_end_to_end_writes_consistent_files PASSED
# test_legacy_sigma_mode_reproduces_old_rule PASSED
# test_default_sweep_csvs_are_found_from_any_working_directory PASSED
# test_pool_script_default_pattern_finds_all_20_windows_from_any_directory PASSED
