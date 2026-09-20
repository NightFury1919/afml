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


def make_window(seed, planted, drift=0.0, p_short=0.5):
    """Synthetic window: iid returns; an event every EVERY bars with
    horizon HORIZON. planted=True -> label and f1 both carry the sign of
    the FORWARD return (a real, exploitable edge); False -> pure noise."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2026-01-01', periods=N_BARS, freq='h')
    rets = rng.normal(drift, 0.002, N_BARS)
    close = pd.Series(100 * np.cumprod(1 + rets), index=idx, name='close')
    starts = np.arange(0, N_BARS - HORIZON - 1, EVERY)
    fwd = np.array([close.iloc[s + HORIZON] / close.iloc[s] - 1 for s in starts])
    if planted:
        bin_ = np.where(fwd > 0, 1, -1)
        f1 = np.sign(fwd) + rng.normal(0, 0.7, len(starts))
    else:
        bin_ = rng.choice([-1, 1], len(starts), p=[p_short, 1 - p_short])
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


def tables(planted, drift=0.0, p_short=0.5):
    return {w: make_window(100 + w, planted, drift, p_short) for w in (1, 2, 3, 4)}


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


def test_demean_equals_prescoring_on_demeaned_returns(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    pre = {'pos': cache['pos'],
           'ret': {w: r - r.mean() for w, r in cache['ret'].items()}}
    shifts = {1: 40, 2: 90, 3: 150, 4: 200}
    a = an.evaluate_draw(cache, shifts, demean=True)
    b = an.evaluate_draw(pre, shifts)
    np.testing.assert_allclose(a['pooled_pnl'], b['pooled_pnl'], rtol=0, atol=1e-15)
    raw = an.evaluate_draw(cache, shifts)
    assert not np.allclose(a['pooled_pnl'], raw['pooled_pnl'])   # flag really does something


def test_demeaning_removes_the_drift_offset_a_circular_shift_cannot(small_real):
    # Strong upward drift + labels mostly -1 -> classifiers are net short
    # -> position-average x return-average is negative in EVERY roll, since
    # circular shifts preserve each window's mean return.
    tb = tables(planted=False, drift=0.001, p_short=0.85)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    lens = {w: len(cache['ret'][w]) for w in cache['ret']}

    def null_ts(demean):
        rng = np.random.default_rng(3)
        ts = [an.evaluate_draw(cache, an.draw_shifts(rng, lens, 20), demean=demean)['t_stat']
              for _ in range(40)]
        return np.array([t for t in ts if np.isfinite(t)])

    raw_null, dm_null = null_ts(False), null_ts(True)
    assert raw_null.mean() < -2.0            # drift x net-short bias, as hypothesised
    assert abs(dm_null.mean()) < 0.75        # demeaning recentres the null


# ---------------------------------------------------------------------------
# Transaction-cost model
# ---------------------------------------------------------------------------
def test_cost_zero_is_identical_to_gross(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    sh = {1: 40, 2: 90, 3: 150, 4: 200}
    a = an.evaluate_draw(cache, sh)
    b = an.evaluate_draw(cache, sh, cost=0.0)
    np.testing.assert_array_equal(a['pooled_pnl'], b['pooled_pnl'])


def test_cost_hand_computed_known_values(monkeypatch):
    """Two windows, one grid point, positions/returns small enough to do by
    hand. net = pos*ret - cost*|delta pos| (first bar's delta is from flat)."""
    monkeypatch.setattr(real, 'WINDOW_DIRS', {1: 'a', 2: 'b'})
    monkeypatch.setattr(real, 'C_GRID', [1.0])
    monkeypatch.setattr(real, 'STEP_GRID', [0.1])
    p1 = np.array([0.0, 1.0, 1.0, -1.0])
    p2 = np.array([-1.0, -1.0, 0.0, 0.5])
    r1 = np.array([0.01, 0.02, -0.01, 0.03])
    r2 = np.array([0.02, -0.01, 0.04, 0.02])
    filler = np.array([0.5, -0.5, 0.25, -0.25])
    cache = {'pos': {('inner', 1.0, 2, 1, 0.1): filler,
                     ('inner', 1.0, 1, 2, 0.1): filler,
                     ('refit', 1.0, 1, 0.1): p1,
                     ('refit', 1.0, 2, 0.1): p2},
             'ret': {1: r1, 2: r2}}
    c = 0.001
    out = an.evaluate_draw(cache, {}, cost=c)
    net1 = [0.0, 0.019, -0.010, -0.032]      # gross [0,.02,-.01,-.03] - cost*[0,1,0,2]
    net2 = [-0.021, 0.010, -0.001, 0.0095]   # gross [-.02,.01,0,.01]  - cost*[1,0,1,.5]
    np.testing.assert_allclose(out['pooled_pnl'], net1 + net2, atol=1e-15)
    assert out['n_active'] == 7              # the flat, no-trade bar is inactive
    assert out['mean_pnl'] == pytest.approx(-0.0255 / 7)
    gross = an.evaluate_draw(cache, {})
    assert gross['mean_pnl'] > out['mean_pnl']


def test_turnover_memo_never_modifies_positions(small_real):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    before = {k: v.copy() for k, v in cache['pos'].items()}
    an.evaluate_draw(cache, {1: 30, 2: 60, 3: 90, 4: 120}, cost=0.0005)
    for k, v in before.items():
        np.testing.assert_array_equal(cache['pos'][k], v)
    assert cache['turn']                     # memo populated


def test_breakeven_interpolation_known_value():
    df = pd.DataFrame({'cost_bps_one_way': [0.0, 10.0, 20.0],
                       'real_net_mean_pnl_per_active_bar': [2.0, 1.0, -1.0]})
    assert an.breakeven_bps(df) == pytest.approx(15.0)
    df2 = df.assign(real_net_mean_pnl_per_active_bar=[2.0, 1.5, 1.0])
    assert an.breakeven_bps(df2) is None


def test_cost_sweep_kills_a_planted_edge_at_high_cost(small_real, tmp_path):
    tb = tables(planted=True)
    cache = an.build_cache(tb, FEATURE_COLS, verbose=False)
    df = an.run_cost_sweep(cache, [0.0, 5.0, 500.0], n_rolls=10, min_shift=20,
                           seed=4, demean=False, out_path=str(tmp_path / 'sweep.csv'))
    pnl = df['real_net_mean_pnl_per_active_bar'].values
    assert pnl[0] > 0                        # gross edge is real
    assert pnl[-1] < 0                       # 5% one-way cost must destroy it
    assert pnl[0] > pnl[1] > pnl[-1]         # more cost, less money
    assert df['n_rolls'].tolist() == [10, 10, 10]


def test_turnover_report_hand_computed_known_values(monkeypatch):
    monkeypatch.setattr(real, 'WINDOW_DIRS', {1: 'a', 2: 'b'})
    monkeypatch.setattr(real, 'C_GRID', [1.0])
    monkeypatch.setattr(real, 'STEP_GRID', [0.1])
    p1 = np.array([0.0, 1.0, 1.0, -1.0])     # turnover [0,1,0,2]
    p2 = np.array([-1.0, -1.0, 0.0, 0.5])    # turnover [1,0,1,0.5]
    filler = np.array([0.5, -0.5, 0.25, -0.25])
    cache = {'pos': {('inner', 1.0, 2, 1, 0.1): filler, ('inner', 1.0, 1, 2, 0.1): filler,
                     ('refit', 1.0, 1, 0.1): p1, ('refit', 1.0, 2, 0.1): p2},
             'ret': {1: np.array([0.01, 0.02, -0.01, 0.03]),
                     2: np.array([0.02, -0.01, 0.04, 0.02])}}
    rep = an.turnover_report(cache)
    assert rep['n_bars_total'] == 8
    assert rep['pooled_turnover_sum'] == pytest.approx(5.5)
    assert rep['mean_turnover_per_bar'] == pytest.approx(5.5 / 8)
    assert rep['pooled_gross_pnl_sum'] == pytest.approx(-0.02)   # fold1 -0.02, fold2 0.0
    assert rep['breakeven_bps_first_order'] == pytest.approx(1e4 * -0.02 / 5.5)
    f1 = rep['per_fold'].set_index('test_window').loc[1]
    assert f1['frac_bars_in_market'] == pytest.approx(3 / 4)
    assert f1['frac_bars_with_trade'] == pytest.approx(2 / 4)
    assert f1['mean_abs_position'] == pytest.approx(3 / 4)


# =============================================================================
# TDD RESULTS (synthetic data; mlfinlab env, Python 3.10.20, pytest 9.0.3)
# 15 passed in 136.17s  (run 2026-09-19)
# =============================================================================
# test_zero_shift_reproduces_real_procedure_exactly PASSED
# test_shifting_changes_pnl_but_not_its_length_or_position_cache PASSED
# test_full_length_shift_is_identity PASSED
# test_draw_shifts_respect_bounds_and_are_reproducible PASSED
# test_empirical_p_known_values PASSED
# test_noise_only_null_is_centred_near_zero PASSED
# test_planted_edge_is_detected_against_the_alignment_null PASSED
# test_demean_equals_prescoring_on_demeaned_returns PASSED
# test_demeaning_removes_the_drift_offset_a_circular_shift_cannot PASSED
# test_cost_zero_is_identical_to_gross PASSED
# test_cost_hand_computed_known_values PASSED
# test_turnover_memo_never_modifies_positions PASSED
# test_breakeven_interpolation_known_value PASSED
# test_cost_sweep_kills_a_planted_edge_at_high_cost PASSED
# test_turnover_report_hand_computed_known_values PASSED
