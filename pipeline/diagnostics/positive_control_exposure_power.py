"""
pipeline/diagnostics/positive_control_exposure_power.py

POWER CURVE for the confidence-exposure gate (Option 4).

QUESTION
--------
confidence = DSR * (1 - PBO) drives exposure_scalar = clip((c - a)/(1 - a), 0, 1).
The 2026-09-13 positive control showed a planted edge of annualized Sharpe
1-3 earns confidence no higher than the null at this project's real sample
sizes. This script maps the WHOLE power curve, so the next design decision
(how much data, which anchor, what ramp) rests on numbers instead of a
guess. It answers:

  1. How strong must a real edge be, at each sample size, before the score
     reliably separates from a no-edge score? (minimum detectable Sharpe)
  2. How much MORE data would a realistic edge (annualized Sharpe 1-3) need?
  3. Does the answer depend on T only through the edge's t-statistic z? (If
     yes, results extrapolate to sample sizes too slow to simulate here.)

DESIGN
------
Same generator the 2026-09-13 positive control uses (imported unmodified via
positive_control_confidence_threshold.one_replicate: iid N(0,1) PnL for
N=20 trials, trial 0 given a mean shift, real ch11 pbo() with S=12, real
ch14 deflated_sharpe_ratio()), but the grid is in

    z = true_per_bar_sharpe * sqrt(T)        (the planted edge's t-stat)

instead of annualized Sharpe. For iid Gaussian PnL, DSR and PBO depend on
the edge mainly through z, so a z-grid covers every T with the same
effort; each z is then CONVERTED to an annualized Sharpe per T with this
project's existing convention (sqrt(bars_per_day * 365), from
calibrate_n_trials_floor.py). z itself is convention-free -- if you doubt
the annualization convention, trust the z results.

T points: the two real T_effective values (single window 197.30, 8-window
pool 4004.48) plus 4x the pool (16017.92, "what three more years of data
would look like"). If the z-curves coincide across the three T's, the
z-law holds and you can extrapolate to any T.

THRESHOLD-FREE POWER: power_null95 = P(confidence > the 95th percentile of
that T's own z=0 scores). It asks "does a real edge beat 95% of no-edge
scores?" without committing to any anchor. Exposure at specific anchors
(the live one from confidence_threshold.txt, the legacy 0.7235, and the
per-T null-95th percentile) is reported alongside.

REPRODUCIBILITY FIX: the 2026-09-13 script seeded each cell with Python's
hash((label, sharpe)); str hashing is randomized per process, so its seeds
changed every run. Seeds here come from zlib.crc32 and are deterministic.

RESUMABLE: each completed (T, z) cell is appended to the reps CSV; a re-run
skips cells that already have n_reps rows. Replicates run in a process pool
(pure numpy/pandas -- spawn-safe; the SVC/loky restriction does not apply).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\positive_control_exposure_power.py            # ~25-30 min, 4 workers
    python pipeline\\diagnostics\\positive_control_exposure_power.py --quick    # ~3 min sanity pass
    python pipeline\\diagnostics\\positive_control_exposure_power.py --analyze-only --anchors 0.45 0.60
"""
import argparse
import importlib.util
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    'positive_control_confidence_threshold',
    os.path.join(HERE, 'positive_control_confidence_threshold.py'))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

POOLED_T = 4004.48
POOLED_BPD = 37884.0 / 240.0
T_POINTS = [
    ('single_window_tb2000', 197.30, 65.27),
    ('pooled_8window_tb40000', POOLED_T, POOLED_BPD),
    ('pooled_x4', POOLED_T * 4, POOLED_BPD),
]
Z_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
N_REPS = 300
LEGACY_ANCHOR = 0.7235
THRESHOLD_FILE = os.path.join(HERE, 'confidence_threshold.txt')
TARGET_ANNUAL_SHARPES = [1.0, 2.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# helpers (pure, unit-tested)
# ---------------------------------------------------------------------------
def annual_equivalent(z, T, bars_per_day):
    """Annualized Sharpe of a planted edge with t-stat z at sample size T."""
    return z * np.sqrt(bars_per_day * 365.0) / np.sqrt(T)


def t_needed(target_annual_sharpe, z_star, bars_per_day):
    """Sample size at which an edge of the target annualized Sharpe reaches
    the detection t-stat z_star (inverse of annual_equivalent)."""
    return (z_star ** 2) * bars_per_day * 365.0 / (target_annual_sharpe ** 2)


def cell_seed(label, z):
    """Deterministic (unlike hash() of a str, which is randomized per process)."""
    return (zlib.crc32(label.encode('utf-8')) * 1_000_003 + int(round(z * 1000))) % (2 ** 31 - 1000)


def exposure(conf, anchor):
    return np.clip((np.asarray(conf, dtype=float) - anchor) / (1.0 - anchor), 0.0, 1.0)


def min_detectable_z(z_values, power_values, level):
    """Smallest z whose power reaches `level`, linearly interpolated between grid
    points (nan if never reached)."""
    z = np.asarray(z_values, dtype=float)
    p = np.asarray(power_values, dtype=float)
    order = np.argsort(z)
    z, p = z[order], p[order]
    for i in range(len(z)):
        if p[i] >= level:
            if i == 0:
                return float(z[0])
            return float(z[i - 1] + (level - p[i - 1]) * (z[i] - z[i - 1]) / (p[i] - p[i - 1]))
    return float('nan')


def load_live_anchor():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as fh:
            return float(fh.read().strip())
    return LEGACY_ANCHOR


def analyze(reps, t_points, anchors):
    """reps: DataFrame[T_label, z, dsr, pbo, confidence]. Returns a summary
    DataFrame, one row per (T, z), with threshold-free and per-anchor power."""
    rows = []
    meta = {lab: (T, bpd) for lab, T, bpd in t_points}
    for lab, g in reps.groupby('T_label', sort=False):
        T, bpd = meta[lab]
        null_conf = g.loc[g['z'] == 0, 'confidence'].values
        null95 = float(np.quantile(null_conf, 0.95)) if len(null_conf) else float('nan')
        for z, gz in g.groupby('z'):
            c = gz['confidence'].values
            row = {'T_label': lab, 'T_effective': T, 'z': float(z),
                   'annual_sharpe_equiv': float(annual_equivalent(z, T, bpd)),
                   'n_reps': len(c), 'mean_dsr': float(gz['dsr'].mean()),
                   'mean_pbo': float(gz['pbo'].mean()), 'mean_conf': float(c.mean()),
                   'std_conf': float(c.std(ddof=1)) if len(c) > 1 else float('nan'),
                   'null95': null95,
                   'power_null95': float((c > null95).mean())}
            for a in anchors:
                row[f'frac_exposed@{a:.4f}'] = float((c > a).mean())
                row[f'mean_exposure@{a:.4f}'] = float(exposure(c, a).mean())
            if null95 < 1.0:
                row['mean_exposure@null95'] = float(exposure(c, null95).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def detectability_table(summary, t_points, levels=(0.5, 0.8)):
    meta = {lab: (T, bpd) for lab, T, bpd in t_points}
    rows = []
    for lab, g in summary.groupby('T_label', sort=False):
        T, bpd = meta[lab]
        row = {'T_label': lab, 'T_effective': T}
        for lv in levels:
            zs = min_detectable_z(g['z'].values, g['power_null95'].values, lv)
            row[f'z_at_{int(lv * 100)}pct_power'] = zs
            row[f'annual_sharpe_at_{int(lv * 100)}pct_power'] = (
                float(annual_equivalent(zs, T, bpd)) if np.isfinite(zs) else float('nan'))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def _replicate_task(args):
    T, true_sharpe, seed = args
    dsr, pbo_val = pc.one_replicate(T, true_sharpe, seed=seed)
    return float(dsr), float(pbo_val)


def run_grid(t_points, z_grid, n_reps, workers, reps_path, replicate_fn=None, verbose=True):
    """Simulate every (T, z) cell not already complete in reps_path; append each
    finished cell to the CSV. replicate_fn is injectable when workers == 1."""
    done = {}
    if os.path.exists(reps_path):
        prev = pd.read_csv(reps_path)
        done = prev.groupby(['T_label', 'z']).size().to_dict()
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for lab, T, bpd in t_points:
            for z in z_grid:
                if done.get((lab, float(z)), 0) >= n_reps:
                    if verbose:
                        print(f'  skip {lab} z={z:g} (already complete)')
                    continue
                true_sharpe = z / np.sqrt(T)
                base = cell_seed(lab, z)
                tasks = [(T, true_sharpe, base + i) for i in range(n_reps)]
                if pool is not None:
                    res = list(pool.map(_replicate_task, tasks,
                                        chunksize=max(1, n_reps // (workers * 4))))
                else:
                    fn = replicate_fn or _replicate_task
                    res = [fn(t) for t in tasks]
                cell = pd.DataFrame({'T_label': lab, 'T_effective': T, 'z': float(z),
                                     'rep': np.arange(n_reps),
                                     'dsr': [r[0] for r in res], 'pbo': [r[1] for r in res]})
                cell['confidence'] = cell['dsr'] * (1 - cell['pbo'])
                cell.to_csv(reps_path, mode='a', header=not os.path.exists(reps_path), index=False)
                if verbose:
                    print(f'  {lab} z={z:g}: mean conf {cell["confidence"].mean():.3f} '
                          f'({n_reps} reps)', flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-reps', type=int, default=N_REPS)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--z-grid', type=float, nargs='+', default=None)
    ap.add_argument('--anchors', type=float, nargs='+', default=None,
                    help='extra anchors to report (live anchor and 0.7235 always included)')
    ap.add_argument('--quick', action='store_true', help='40 reps, z in {0,2,4,6}')
    ap.add_argument('--analyze-only', action='store_true')
    ap.add_argument('--out-dir', default=HERE)
    args = ap.parse_args(argv)

    n_reps = 40 if args.quick else args.n_reps
    z_grid = args.z_grid or ([0.0, 2.0, 4.0, 6.0] if args.quick else Z_GRID)
    if 0.0 not in [float(z) for z in z_grid]:
        raise SystemExit('z grid must include 0 (the per-T null the power is measured against).')
    reps_path = os.path.join(args.out_dir, 'positive_control_exposure_power_reps'
                             + ('_quick' if args.quick else '') + '.csv')
    live = load_live_anchor()
    anchors = sorted({round(live, 4), LEGACY_ANCHOR, *(args.anchors or [])})

    print(f'Power curve: {len(T_POINTS)} T points x {len(z_grid)} z values x {n_reps} reps, '
          f'{args.workers} workers; live anchor {live:.4f}')
    if not args.analyze_only:
        run_grid(T_POINTS, z_grid, n_reps, args.workers, reps_path)
    if not os.path.exists(reps_path):
        raise SystemExit(f'{reps_path} not found -- run without --analyze-only first.')

    reps = pd.read_csv(reps_path)
    summary = analyze(reps, T_POINTS, anchors)
    det = detectability_table(summary, T_POINTS)
    summary.to_csv(os.path.join(args.out_dir, 'positive_control_exposure_power_summary.csv'), index=False)
    det.to_csv(os.path.join(args.out_dir, 'positive_control_exposure_power_detectability.csv'), index=False)

    pd.set_option('display.width', 200)
    print(f'\n{"=" * 78}\nPOWER BY z (threshold-free: P[score > 95th pct of no-edge scores])\n{"=" * 78}')
    print(summary.pivot(index='z', columns='T_label', values='power_null95').round(3).to_string())
    print('\nIf these columns agree, power depends on T only through z -> extrapolation is valid.')

    print(f'\n{"=" * 78}\nEXPOSURE AT EACH ANCHOR (mean exposure | fraction of reps with any exposure)\n{"=" * 78}')
    for a in anchors:
        print(f'\nanchor {a:.4f}{"  <- live" if abs(a - round(live, 4)) < 1e-9 else ""}:')
        me = summary.pivot(index='z', columns='T_label', values=f'mean_exposure@{a:.4f}')
        fe = summary.pivot(index='z', columns='T_label', values=f'frac_exposed@{a:.4f}')
        print(pd.concat({'mean_exposure': me, 'frac_exposed': fe}, axis=1).round(3).to_string())

    print(f'\n{"=" * 78}\nMINIMUM DETECTABLE EDGE\n{"=" * 78}')
    print(det.round(3).to_string(index=False))
    z50 = det['z_at_50pct_power'].dropna()
    if len(z50):
        z_star = float(z50.mean())
        print(f'\nMean z at 50% power = {z_star:.2f}. Sample size needed for a target '
              f'annualized Sharpe to reach it (project convention: bars/day = pooled '
              f'{POOLED_BPD:.1f}; the z result itself is convention-free):')
        for tgt in TARGET_ANNUAL_SHARPES:
            need = t_needed(tgt, z_star, POOLED_BPD)
            print(f'  annualized Sharpe {tgt:g}: T_effective ~ {need:,.0f}  '
                  f'({need / POOLED_T:,.0f}x the 8-window pool)')
    print(f'\nSaved: {reps_path}\n       positive_control_exposure_power_summary.csv, '
          f'positive_control_exposure_power_detectability.csv')
    return summary, det


if __name__ == '__main__':
    main()


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
