"""
pipeline/diagnostics/calibrate_confidence_threshold.py

Option 4 (2026-09-11, revised 2026-09-19): replace the binary DSR>=0.95
strategy-level gate with a continuous exposure scalar

    confidence      = DSR * (1 - PBO)
    exposure_scalar = clip((confidence - anchor) / (1 - anchor), 0, 1)

The whole design question is where to put `anchor` (the "zero-exposure"
point below which the strategy gets no capital).

WHAT CHANGED (2026-09-19)
-------------------------
The 2026-09-11 version set anchor = null_mean + 2*null_std of the real
(DSR, PBO) sweep pairs. That landed at 0.7235 -- above 92% of the null
draws AND above what real runs ever reached (a live run scored 0.734 and
got exposure 0.04), so the strategy would almost never trade. And a "2
sigma" rule is borrowed from a normal-distribution convention that these
bounded, skewed scores do not follow.

The anchor is now chosen from an explicit RISK BUDGET, in the strategy's
own units:

    "When there is NO edge (a genuine null draw), the AVERAGE exposure
     I am willing to carry is B of full size."     (default B = 3%)

    E_null(a) = mean over null draws of clip((c - a) / (1 - a), 0, 1)

E_null(a) falls continuously from mean(c) at a=0 to 0 at a=1, so the
anchor solving E_null(a) = B is unique (bisection). Lowering B raises the
anchor; a budget >= mean(c) puts the anchor at 0.

The null draws are every real (DSR, PBO) sweep pair on disk from data the
2026-09-11 multi-window meta-analysis showed contains no edge: windows
1-8, Binance, and the 8-window pool (50 rows -- windows 4-8 were not
available to the first calibration, which used 25).

UNCERTAINTY: the sources are few and overlapping (the pool contains the
windows; the rows within a source are five trials of the same data), so
the anchor's sampling noise is reported with a CLUSTER bootstrap over
sources (resample whole sources, not rows). Treat the interval, not just
the point value, as the calibration.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_confidence_threshold.py
    python pipeline\\diagnostics\\calibrate_confidence_threshold.py --budget 0.05
    python pipeline\\diagnostics\\calibrate_confidence_threshold.py --sigma 2   # legacy rule

Writes confidence_threshold.txt (read by apply_confidence_exposure.py and
log_live_prediction.py) and confidence_threshold_calibration.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Explicit list, not a fragile glob (a broad glob once swept in an unrelated
# leftover file). Resolved relative to CWD (the repo root), as before.
DEFAULT_CSVS = [f'kraken_target_bars_calibration_window{i}.csv' for i in range(1, 9)] + [
    'kraken_target_bars_calibration_binance.csv',
    'kraken_target_bars_calibration_pooled.csv',
]
DEFAULT_BUDGET = 0.03
N_SIGMA_LEGACY = 2.0


def load_null_draws(csv_names, verbose=True):
    frames = []
    for name in csv_names:
        if not os.path.exists(name):
            if verbose:
                print(f'  SKIP: {name} not found (not run yet, or different CWD?)')
            continue
        df = pd.read_csv(name)
        if 'dsr' not in df.columns or 'pbo' not in df.columns:
            if verbose:
                print(f'  SKIP: {name} missing dsr/pbo columns')
            continue
        df = df.dropna(subset=['dsr', 'pbo'])
        n_before = len(df)
        df = df.drop_duplicates(subset=['dsr', 'pbo'])
        if verbose and len(df) < n_before:
            print(f'    (dropped {n_before - len(df)} exact-duplicate dsr/pbo rows '
                  f'within {name})')
        df = df.assign(source=os.path.basename(name))
        frames.append(df[['dsr', 'pbo', 'source']])
        if verbose:
            print(f'  Loaded {len(df)} real (dsr, pbo) rows from {name}')
    if not frames:
        raise SystemExit('No real (dsr, pbo) data found -- run the sweeps first.')
    pooled = pd.concat(frames, ignore_index=True)
    pooled['confidence'] = pooled['dsr'] * (1 - pooled['pbo'])
    return pooled


def exposure_scalar(confidence, anchor):
    """Linear ramp: 0 at/below the anchor, 1 at confidence = 1."""
    return np.clip((np.asarray(confidence, dtype=float) - anchor) / (1.0 - anchor), 0.0, 1.0)


def expected_null_exposure(conf, anchor):
    """Average exposure the ramp would assign to genuine null draws."""
    return float(exposure_scalar(conf, anchor).mean())


def solve_anchor(conf, budget):
    """Unique anchor a with expected_null_exposure(conf, a) == budget."""
    conf = np.asarray(conf, dtype=float)
    if budget <= 0:
        return float(conf.max())
    if budget >= conf.mean():
        return 0.0
    lo, hi = 0.0, float(conf.max())          # E(lo) > budget >= E(hi) = 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_null_exposure(conf, mid) > budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cluster_bootstrap_anchor(draws, budget, n_boot=2000, seed=20260919):
    """Resample whole SOURCES with replacement; re-solve the anchor each time."""
    rng = np.random.default_rng(seed)
    groups = {s: g['confidence'].values for s, g in draws.groupby('source')}
    keys = list(groups)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        conf = np.concatenate([groups[keys[i]] for i in pick])
        out.append(solve_anchor(conf, budget))
    return np.array(out)


def calibrate(csv_names, budget=DEFAULT_BUDGET, n_boot=2000, seed=20260919,
              n_sigma=None, out_dir=HERE, verbose=True):
    draws = load_null_draws(csv_names, verbose=verbose)
    conf = draws['confidence'].values
    n = len(conf)
    null_mean, null_std = float(conf.mean()), float(conf.std(ddof=1))
    legacy_anchor = null_mean + N_SIGMA_LEGACY * null_std

    if n_sigma is not None:
        anchor, mode = null_mean + n_sigma * null_std, f'legacy mean + {n_sigma:g}*std'
    else:
        anchor, mode = solve_anchor(conf, budget), f'risk budget {budget:.1%}'

    boot = cluster_bootstrap_anchor(draws, budget, n_boot, seed) if n_sigma is None else None
    result = {'anchor': anchor, 'mode': mode, 'n_draws': n, 'n_sources': draws['source'].nunique(),
              'null_mean': null_mean, 'null_std': null_std, 'legacy_anchor': legacy_anchor,
              'null_max': float(conf.max()),
              'frac_null_exposed': float((conf > anchor).mean()),
              'expected_null_exposure': expected_null_exposure(conf, anchor),
              'boot_q05_q50_q95': (tuple(np.quantile(boot, [0.05, 0.5, 0.95])) if boot is not None else None),
              'draws': draws}

    if verbose:
        print(f'\n{n} real null (dsr, pbo) draws from {result["n_sources"]} sources '
              f'(all from data the multi-window meta-analysis showed has no edge).')
        print(f'confidence = dsr * (1 - pbo): mean {null_mean:.4f}, std {null_std:.4f}, '
              f'max {conf.max():.4f}')
        print(f'\n  candidate anchors -> how often a null draw gets ANY exposure, and the '
              f'average null exposure:')
        for a in (0.7235, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3):
            print(f'    anchor {a:.4f}: {100 * (conf > a).mean():5.1f}% of null draws exposed, '
                  f'mean null exposure {expected_null_exposure(conf, a):.2%}')
        print(f'\n  legacy anchor (mean + {N_SIGMA_LEGACY:.0f}*std) = {legacy_anchor:.4f}  '
              f'-> mean null exposure {expected_null_exposure(conf, legacy_anchor):.2%}')
        print(f'  CHOSEN ({mode}): anchor = {anchor:.4f}  '
              f'-> {100 * result["frac_null_exposed"]:.1f}% of null draws exposed, '
              f'mean null exposure {result["expected_null_exposure"]:.2%}')
        if boot is not None:
            q = result['boot_q05_q50_q95']
            print(f'  cluster-bootstrap over sources ({n_boot} resamples): anchor 5/50/95% = '
                  f'{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}  '
                  f'(a wide interval = the sources disagree; take it seriously)')

    draws.to_csv(os.path.join(out_dir, 'confidence_threshold_calibration.csv'), index=False)
    with open(os.path.join(out_dir, 'confidence_threshold.txt'), 'w') as f:
        f.write(f'{anchor}\n')
    if verbose:
        print(f'\nWrote confidence_threshold.txt (zero_exposure_threshold={anchor:.6f}) and '
              'confidence_threshold_calibration.csv in ' + out_dir)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='*', help='sweep CSVs (default: windows 1-8, binance, pooled)')
    ap.add_argument('--budget', type=float, default=DEFAULT_BUDGET,
                    help='max AVERAGE exposure tolerated on genuine null draws (default 0.03)')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--sigma', type=float, default=None,
                    help='legacy rule: anchor = null mean + SIGMA * null std (ignores --budget)')
    args = ap.parse_args()
    calibrate(args.csvs or DEFAULT_CSVS, budget=args.budget, n_boot=args.n_boot,
              n_sigma=args.sigma)


if __name__ == '__main__':
    main()


# =============================================================================
# TDD RESULTS (synthetic + tmp_path data; mlfinlab env, Python 3.10.20, pytest 9.0.3)
# 11 passed in 6.48s  (run 2026-09-19)
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
