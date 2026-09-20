"""
pipeline/diagnostics/test_positive_control_exposure_power.py

TDD for positive_control_exposure_power.py. Known values are hand-derived;
the two simulation tests use the real one_replicate() at tiny scale.

Run:  python -m pytest pipeline/diagnostics/test_positive_control_exposure_power.py -v
"""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    'positive_control_exposure_power', os.path.join(HERE, 'positive_control_exposure_power.py'))
pw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pw)

# bars_per_day chosen so sqrt(bars_per_day * 365) == 10 exactly
BPD10 = 100.0 / 365.0


def test_annual_equivalent_known_value():
    # z=2, T=400 (sqrt=20): 2 * 10 / 20 = 1.0
    assert pw.annual_equivalent(2.0, 400.0, BPD10) == pytest.approx(1.0)


def test_t_needed_is_the_inverse_of_annual_equivalent():
    # z*=3, target annual Sharpe 1.5, sqrt(bpd*365)=10  ->  T = 9*100/2.25 = 400
    T = pw.t_needed(1.5, 3.0, BPD10)
    assert T == pytest.approx(400.0)
    assert pw.annual_equivalent(3.0, T, BPD10) == pytest.approx(1.5)


def test_cell_seed_is_deterministic_and_distinct():
    assert pw.cell_seed('a', 2.0) == pw.cell_seed('a', 2.0)
    assert pw.cell_seed('a', 2.0) != pw.cell_seed('a', 3.0)
    assert pw.cell_seed('a', 2.0) != pw.cell_seed('b', 2.0)
    assert 0 <= pw.cell_seed('pooled_x4', 6.0) < 2 ** 31


def test_exposure_known_values():
    np.testing.assert_allclose(pw.exposure([0.4, 0.5, 0.75, 1.0], 0.5), [0.0, 0.0, 0.5, 1.0])


def test_min_detectable_z_interpolation_and_edges():
    z = [0, 1, 2, 3]
    p = [0.05, 0.2, 0.6, 0.9]
    assert pw.min_detectable_z(z, p, 0.5) == pytest.approx(1.75)   # 1 + (0.5-0.2)/0.4
    assert pw.min_detectable_z(z, p, 0.9) == pytest.approx(3.0)
    assert np.isnan(pw.min_detectable_z(z, p, 0.95))               # never reached
    assert pw.min_detectable_z(z, p, 0.05) == 0.0                  # already met at the first point
    assert pw.min_detectable_z([3, 0, 2, 1], [0.9, 0.05, 0.6, 0.2], 0.5) == pytest.approx(1.75)  # unsorted


def _reps():
    rows = []
    for c in [0.1, 0.2, 0.3, 0.4]:
        rows.append(('T1', 100.0, 0.0, 0.0, 0.0, c))
    for c in [0.6, 0.8, 1.0, 0.7]:
        rows.append(('T1', 100.0, 4.0, 0.0, 0.0, c))
    df = pd.DataFrame(rows, columns=['T_label', 'T_effective', 'z', 'rep', 'pbo', 'confidence'])
    df['dsr'] = df['confidence']
    return df


def test_analyze_hand_computed():
    s = pw.analyze(_reps(), [('T1', 100.0, BPD10)], anchors=[0.5])
    z0 = s[s['z'] == 0].iloc[0]
    z4 = s[s['z'] == 4].iloc[0]
    # null 95th percentile of [.1,.2,.3,.4]: index .95*3 = 2.85 -> .3 + .85*.1 = .385
    assert z4['null95'] == pytest.approx(0.385)
    assert z4['power_null95'] == pytest.approx(1.0)                # all four exceed .385
    assert z0['power_null95'] == pytest.approx(0.25)               # only 0.4 exceeds it
    # anchor 0.5, z=4: exposures (0.6,0.8,1.0,0.7 - 0.5)/0.5 = .2,.6,1.0,.4 -> mean .55
    assert z4['mean_exposure@0.5000'] == pytest.approx(0.55)
    assert z4['frac_exposed@0.5000'] == pytest.approx(1.0)
    assert z0['frac_exposed@0.5000'] == pytest.approx(0.0)
    assert z4['mean_conf'] == pytest.approx(0.775)
    # z=4, T=100 (sqrt=10), sqrt(bpd*365)=10 -> annual = 4*10/10 = 4
    assert z4['annual_sharpe_equiv'] == pytest.approx(4.0)


def test_detectability_table_known_value():
    reps = _reps()
    s = pw.analyze(reps, [('T1', 100.0, BPD10)], anchors=[0.5])
    d = pw.detectability_table(s, [('T1', 100.0, BPD10)], levels=(0.5,))
    # power: z=0 -> 0.25, z=4 -> 1.0 ; 50% power at 0 + (0.5-0.25)*4/0.75 = 1.3333
    assert d.iloc[0]['z_at_50pct_power'] == pytest.approx(4 / 3)
    assert d.iloc[0]['annual_sharpe_at_50pct_power'] == pytest.approx((4 / 3) * 10 / 10)


def test_run_grid_writes_cells_and_resumes(tmp_path):
    calls = []

    def fake(args):
        calls.append(args)
        return 0.5, 0.25

    path = str(tmp_path / 'reps.csv')
    tp = [('A', 100.0, 1.0)]
    pw.run_grid(tp, [0.0, 2.0], n_reps=5, workers=1, reps_path=path, replicate_fn=fake, verbose=False)
    assert len(calls) == 10
    df = pd.read_csv(path)
    assert len(df) == 10 and set(df['z']) == {0.0, 2.0}
    assert (df['confidence'] == 0.5 * 0.75).all()
    # true_sharpe passed to the generator is z / sqrt(T)
    assert calls[-1][1] == pytest.approx(2.0 / 10.0)
    # a re-run skips both complete cells; asking for a new z runs only that cell
    pw.run_grid(tp, [0.0, 2.0, 4.0], n_reps=5, workers=1, reps_path=path,
                replicate_fn=fake, verbose=False)
    assert len(calls) == 15
    assert len(pd.read_csv(path)) == 15


def test_real_generator_separates_a_strong_edge_from_null():
    """Smoke test with the REAL one_replicate (real pbo() and DSR), tiny scale."""
    null = [pw._replicate_task((197.3, 0.0, 100 + i)) for i in range(6)]
    edge = [pw._replicate_task((197.3, 6.0 / np.sqrt(197.3), 200 + i)) for i in range(6)]
    conf = lambda rs: np.mean([d * (1 - p) for d, p in rs])
    assert conf(edge) > conf(null) + 0.3


def test_main_end_to_end_tiny(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pw, 'T_POINTS', [('tiny', 197.3, 65.27)])
    monkeypatch.setattr(pw, 'load_live_anchor', lambda: 0.5363)
    summary, det = pw.main(['--n-reps', '6', '--workers', '1', '--z-grid', '0', '6',
                            '--out-dir', str(tmp_path)])
    out = capsys.readouterr().out
    assert 'MINIMUM DETECTABLE EDGE' in out and 'POWER BY z' in out
    for f in ('positive_control_exposure_power_reps.csv',
              'positive_control_exposure_power_summary.csv',
              'positive_control_exposure_power_detectability.csv'):
        assert (tmp_path / f).exists()
    assert set(summary['z']) == {0.0, 6.0}
    assert 'frac_exposed@0.5363' in summary.columns and 'frac_exposed@0.7235' in summary.columns
    z6 = summary[summary['z'] == 6.0].iloc[0]
    z0 = summary[summary['z'] == 0.0].iloc[0]
    assert z6['mean_conf'] > z0['mean_conf']
    # z=0 is required: a grid without it must refuse
    with pytest.raises(SystemExit):
        pw.main(['--n-reps', '2', '--workers', '1', '--z-grid', '2', '4', '--out-dir', str(tmp_path)])


# =============================================================================
# TDD RESULTS (real one_replicate() at tiny scale + hand-derived known values)
# mlfinlab env, Python 3.10.20, pytest 9.0.3: 10 passed in 48.69s  (run 2026-09-19)
# =============================================================================
# test_annual_equivalent_known_value PASSED
# test_t_needed_is_the_inverse_of_annual_equivalent PASSED
# test_cell_seed_is_deterministic_and_distinct PASSED
# test_exposure_known_values PASSED
# test_min_detectable_z_interpolation_and_edges PASSED
# test_analyze_hand_computed PASSED
# test_detectability_table_known_value PASSED
# test_run_grid_writes_cells_and_resumes PASSED
# test_real_generator_separates_a_strong_edge_from_null PASSED
# test_main_end_to_end_tiny PASSED
