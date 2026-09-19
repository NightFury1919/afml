"""
pipeline/diagnostics/test_pool_multiwindow_alignment_null.py

TDD tests for pool_multiwindow_alignment_null.py on SYNTHETIC data with
known expected behaviour (synthetic is acceptable here only because these
are unit tests, not the real-data validation -- the script's own
draw-0 self-check does that on the real windows).

Run:  python -m pytest pipeline/diagnostics/test_pool_multiwindow_alignment_null.py -v
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'pool_multiwindow_alignment_null', os.path.join(HERE, 'pool_multiwindow_alignment_null.py'))
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
real = an.real

N_BARS, EVERY, HORIZON = 500, 5, 5
FEATURE_COLS = ['f1', 'f2', 'f3']


def make_window(seed, planted):
    """Synthetic window: iid returns; an event every EVERY bars with
    horizon HORIZON. planted=True -> label and f1 both carry the sign of
    the FORWARD return (a real, exploitable edge); False -> pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=N_BARS, freq='h')
    rets = rng.normal(0, 0.002, N_BARS)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx, name='close')
    starts = np.arange(0, N_BARS - HORIZON - 1, EVERY)
    fwd = np.array([close.iloc[s + HORIZON] / close.iloc[s] - 1 for s in starts])
    if planted:
        bin_ = np.where(fwd > 0, 1, -1)
        f1 = np.sign(fwd) + rng.normal(0, 0.7, len(starts))
    else:
        bin_ = rng.choice([-1, 1], len(starts))
        f1 = rng.normal(0, 1, len(starts))
    events = pd.DataFrame({
        'f1': f1, 'f2': rng.normal(0, 1, len(starts)), 'f3': rng.normal(0, 1, len(starts)),
        'bin': bin_, 'w': np.ones(len(starts)),
        't1': [idx[s + HORIZON] for s in starts],
    }, index=idx[starts])
    return events, close


@pytest.fixture()
def small_real(monkeypatch):
    """Shrink the real module's window set and grids so tests run fast;
    both the real procedure and the script read these at call time."""
    monkeypatch.setattr(real, 'WINDOW_DIRS', {1: 'a', 2: 'b', 3: 'c', 4: 'd'})
    monkeypatch.setattr(real, 'C_GRID', [0.1, 1.0])
    monkeypatch.setattr(real, 'STEP_GRID', [0.05, 0.2])
    return real


def tables(planted):
    return {w: make_window(100 + w, planted) for w in (1, 2, 3, 4)}


def test_zero_shift_reproduces_real_procedure_exactly(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    draw0 = an.evaluate_draw(cache, {})
    for f in draw0['folds']:
        ref = real.run_one_test_fold(f['test_window'], tb, FEATURE_COLS)
        assert f['winner_C'] == ref['winner_C']
        assert f['winner_step'] == ref['winner_step']
        assert f['n_test_active_bars'] == ref['n_test_active_bars']
    ref_pooled = np.concatenate([real.run_one_test_fold(w, tb, FEATURE_COLS)['pnl']
                                 for w in real.WINDOW_DIRS])
    np.testing.assert_allclose(draw0['pooled_pnl'], ref_pooled, rtol=0, atol=1e-15)


def test_shifting_changes_pnl_but_not_its_length_or_position_cache(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    pos_before = {k: v.copy() for k, v in cache['pos'].items()}
    a = an.evaluate_draw(cache, {})
    b = an.evaluate_draw(cache, {1: 100, 2: 200, 3: 50, 4: 300})
    assert len(a['pooled_pnl']) == len(b['pooled_pnl'])
    assert not np.allclose(a['pooled_pnl'], b['pooled_pnl'])
    for k, v in pos_before.items():          # rolling must never touch positions
        np.testing.assert_array_equal(cache['pos'][k], v)


def test_full_length_shift_is_identity(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    n = {w: len(cache['ret'][w]) for w in cache['ret']}
    a = an.evaluate_draw(cache, {})
    b = an.evaluate_draw(cache, n)           # np.roll by len == no shift
    np.testing.assert_allclose(a['pooled_pnl'], b['pooled_pnl'])


def test_draw_shifts_respect_bounds_and_are_reproducible():
    lens = {1: 100, 2: 300}
    s1 = an.draw_shifts(np.random.default_rng(5), lens, 20)
    s2 = an.draw_shifts(np.random.default_rng(5), lens, 20)
    assert s1 == s2
    for w, k in s1.items():
        assert 20 <= k <= lens[w] - 20
    with pytest.raises(ValueError):
        an.draw_shifts(np.random.default_rng(0), {1: 30}, 20)


def test_empirical_p_known_values():
    null = np.arange(-9, 10, dtype=float)    # 19 values, -9..9
    p1, p2, n = an.empirical_p(9.0, null)    # only 9 >= 9 -> (1+1)/(19+1)
    assert n == 19 and p1 == pytest.approx(2 / 20)
    assert p2 == pytest.approx(3 / 20)       # |-9| and |9| >= 9
    p1, _, _ = an.empirical_p(100.0, null)   # nothing exceeds -> 1/20
    assert p1 == pytest.approx(1 / 20)


def test_noise_only_null_is_centred_near_zero(small_real):
    tb = tables(planted=False)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    lens = {w: len(cache['ret'][w]) for w in cache['ret']}
    rng = np.random.default_rng(1)
    ts = [an.evaluate_draw(cache, an.draw_shifts(rng, lens, 20))['t_stat'] for _ in range(60)]
    ts = np.array([t for t in ts if np.isfinite(t)])
    assert abs(ts.mean()) < 0.75
    real_t = an.evaluate_draw(cache, {})['t_stat']
    p1, p2, _ = an.empirical_p(real_t, ts)
    assert p2 > 0.05                          # pure noise must not look significant


def test_planted_edge_is_detected_against_the_alignment_null(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    lens = {w: len(cache['ret'][w]) for w in cache['ret']}
    real_t = an.evaluate_draw(cache, {})['t_stat']
    assert real_t > 3.0                       # the planted edge is real and large
    rng = np.random.default_rng(2)
    ts = [an.evaluate_draw(cache, an.draw_shifts(rng, lens, 20))['t_stat'] for _ in range(60)]
    ts = np.array([t for t in ts if np.isfinite(t)])
    assert abs(ts.mean()) < 0.75              # rolling kills the edge -> centred near 0
    p1, _, _ = an.empirical_p(real_t, ts)
    assert p1 <= 0.05


# =============================================================================
# TDD RESULTS (synthetic data, sandbox run 2026-09-18: Python 3.12.3, pytest 9.1.1
# -- NOT the mlfinlab env; re-run there and overwrite this block if it differs)
# =============================================================================
# test_pool_multiwindow_alignment_null.py::test_zero_shift_reproduces_real_procedure_exactly PASSED
# test_pool_multiwindow_alignment_null.py::test_shifting_changes_pnl_but_not_its_length_or_position_cache PASSED
# test_pool_multiwindow_alignment_null.py::test_full_length_shift_is_identity PASSED
# test_pool_multiwindow_alignment_null.py::test_draw_shifts_respect_bounds_and_are_reproducible PASSED
# test_pool_multiwindow_alignment_null.py::test_empirical_p_known_values PASSED
# test_pool_multiwindow_alignment_null.py::test_noise_only_null_is_centred_near_zero PASSED
# test_pool_multiwindow_alignment_null.py::test_planted_edge_is_detected_against_the_alignment_null PASSED
# ============================== 7 passed in 33.13s ==============================
