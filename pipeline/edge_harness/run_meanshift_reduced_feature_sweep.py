"""
pipeline/edge_harness/run_meanshift_reduced_feature_sweep.py

Direct follow-on to isolate_single_feature_meanshift.py's real result
(2026-09-06, edge_strength=2.0/seed=0): isolating roll_c ALONE moved OOS
accuracy 0.5052 -> 0.5679 and corr(prob,label) -0.0166 -> +0.1526 versus
the full 14-feature table -- real evidence that the other 13, mostly-
uncorrelated features were DILUTING roll_c's genuine signal inside a
14-dimensional RBF kernel. But DSR/PBO barely moved at that single point
(0.4930->0.5079, 0.8485->0.8323), and the winning trial's Sharpe stayed
tiny either way -- a real directional accuracy edge doesn't automatically
show up as a large Sharpe once it's run through Ch10's real getSignal
position-sizing mechanics. That single-point result doesn't yet show
whether DSR would actually start RESPONDING to edge_strength, which is
the real question this project has been chasing since the original
T_effective-controlled meanshift_edge_sweep_results.csv came back with
corr(edge_strength, dsr) = +0.0111.

THIS SCRIPT: re-runs the EXACT same T_effective-controlled sweep design
as run_meanshift_edge_sweep.py (same edge_strengths/seeds/n_trades_map
discipline), but restricts every combo's staged training table to a
FIXED, small set of features chosen for their real bar-level correlation
with next-bar return in the es=2.0 trace, instead of all 14:

    roll_c                  r=+0.2894  (Stage 2, es=2.0/seed=0 trace)
    becker_parkinson_sigma  r=+0.1308
    structural_break_stat   r=+0.1154
    parkinson_vol_20bar     r=+0.0947

*** LOAD-BEARING DESIGN DECISION: feature set is FIXED across the whole
sweep, chosen ONCE from a single reference trace (es=2.0/seed=0), NOT
re-selected per edge_strength/seed. Re-selecting per combo would be a
real form of feature-selection leakage/data-snooping -- picking whichever
features happen to correlate best with THAT combo's own realized noise
would mechanically inflate apparent detection power at every
edge_strength, including edge_strength=0.0, and the resulting DSR-vs-
edge_strength curve would tell us nothing trustworthy. A single upfront,
disclosed selection avoids that at the cost of the selection itself
possibly being suboptimal for other edge_strengths/seeds -- an accepted,
documented tradeoff, not an oversight.

If DSR/PBO respond meaningfully to edge_strength here (dsr rising, pbo
falling as edge_strength grows) where the full-feature sweep was flat,
that confirms dilution -- not a fundamental model/CV-stage ceiling -- was
the real explanation, a materially different conclusion than the
"third independent mechanism hits an unexplained ceiling" framing the
full-feature sweep supported. If DSR stays flat here too, that pushes
back toward a genuine detection floor at this pipeline's real scale,
even with dilution addressed.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\run_meanshift_reduced_feature_sweep.py --smoke-test
    python pipeline\\edge_harness\\run_meanshift_reduced_feature_sweep.py --edge-strengths 0.5,0.75,1.0,1.5,2.0 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)

from rebuild import build_bars_and_labels             # noqa: E402
from features import build_enriched_events             # noqa: E402
from live_staging import stage_live_training_tables     # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402

from generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades  # noqa: E402

DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)
RESULTS_CSV_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'meanshift_reduced_feature_sweep_results.csv'
)

SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work_meanshift_reduced')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# Same T_effective-controlled grid as run_meanshift_edge_sweep.py's real
# 2026-08-27 run (meanshift_edge_sweep_results.csv) -- deliberately
# identical so this sweep's DSR-vs-edge_strength curve is directly
# comparable to the full-14-feature one, with feature count as the only
# thing that changed.
N_TRADES = 120_000  # fallback only; --n-trades-map should cover every
                     # edge_strength below via meanshift_n_trades_map.json
EDGE_STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0]
SEEDS = [0, 1, 2, 3, 4]

TARGET_BARS = 1000
PBO_S = 12

# Fixed, disclosed feature set -- see module LOAD-BEARING note above.
# Chosen ONCE from trace_meanshift_signal_leakage.py's real Stage 2
# output at edge_strength=2.0/seed=0, ranked by |r| vs next-bar return:
#   roll_c                  r=+0.2894
#   becker_parkinson_sigma  r=+0.1308
#   structural_break_stat   r=+0.1154
#   parkinson_vol_20bar     r=+0.0947
# (the next-largest, amihud_lambda_20bar at +0.0900, was left out to keep
# the set to the four clearly-above-the-rest features; --features lets
# this be overridden without editing the script if a different cut point
# is wanted later.)
DEFAULT_FEATURES = [
    'roll_c', 'becker_parkinson_sigma', 'structural_break_stat',
    'parkinson_vol_20bar',
]


def load_calibration():
    if not os.path.exists(BASELINE_PARAMS_PATH):
        raise SystemExit(
            f'{BASELINE_PARAMS_PATH} not found. Run '
            'calibrate_synthetic_trade_params.py --source live first.'
        )
    with open(BASELINE_PARAMS_PATH) as f:
        params = json.load(f)
    mean_rate_per_sec = params['n_trades'] / (params['span_hours'] * 3600.0)
    return {
        'baseline_imbalance': params['baseline_imbalance'],
        'price_diff_std': params['price_diff_std'],
        'avg_trade_size': params['avg_trade_size'],
        'avg_trade_rate_per_sec': mean_rate_per_sec,
        'start_price': params['price_start'],
    }


def restrict_staged_table(enriched_csv_path, features):
    """Overwrite the just-staged ch07_training_table_enriched.csv in
    place, keeping only t1/bin/w plus the given feature columns. Ch11's
    part_c_build_trials() derives feature_cols generically from the
    CSV's own columns, so this is a legitimate restricted input, not a
    workaround (same pattern as isolate_single_feature_meanshift.py,
    generalized to more than one feature)."""
    df = pd.read_csv(enriched_csv_path, index_col=0, parse_dates=True)
    missing = [f for f in features if f not in df.columns]
    if missing:
        available = [c for c in df.columns if c not in ('t1', 'bin', 'w')]
        raise SystemExit(
            f'feature(s) {missing} not found in the staged table. '
            f'Available features: {available}'
        )
    restricted = df[['t1', 'bin', 'w'] + list(features)].copy()
    restricted.to_csv(enriched_csv_path)
    return restricted.shape


def run_one_combo(edge_strength, seed, calib, ch11, n_trades, target_bars, features):
    t_start = time.time()

    raw_trades, diag = generate_bar_aligned_meanshift_trades(
        n_trades=n_trades,
        target_bars=target_bars,
        edge_strength=edge_strength,
        seed=seed,
        baseline_imbalance=calib['baseline_imbalance'],
        price_diff_std=calib['price_diff_std'],
        avg_trade_rate_per_sec=calib['avg_trade_rate_per_sec'],
        avg_trade_size=calib['avg_trade_size'],
        start_price=calib['start_price'],
        return_diagnostics=True,
    )

    raw_corr = diag['lag1_autocorr']

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    staged = stage_live_training_tables(
        rebuild_result, enriched_result, STAGING_DIR,
    )
    restrict_staged_table(staged['enriched_csv_path'], features)

    ch11_M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)

    tw_aligned = rebuild_result['tw'].reindex(
        enriched_result['enriched_events'].index
    )
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')

    eval_result = evaluate_overfitting(ch11_M, meta, ch11, S=PBO_S, tw=tw_aligned)

    # Real OOS accuracy/prob-label correlation for the winning trial --
    # same direct diagnostic isolate_single_feature_meanshift.py reports,
    # logged per-combo here so the reduced-feature sweep's own results
    # CSV carries it alongside dsr/pbo, not just the summary trial table.
    best_trial = eval_result['best_trial']
    C = meta.loc[best_trial, 'C']
    enriched_path = os.path.join(STAGING_DIR, 'ch07_training_table_enriched.csv')
    events_for_oos = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events_for_oos['t1'] = pd.to_datetime(events_for_oos['t1'])
    feat_cols_oos = [c for c in events_for_oos.columns if c not in ('bin', 'w', 't1')]
    X = events_for_oos[feat_cols_oos]
    y = events_for_oos['bin']
    w = events_for_oos['w']
    t1 = events_for_oos['t1']
    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)
    oos_accuracy = np.nan
    prob_label_corr = np.nan
    if len(prob) >= 3:
        y_aligned = y.loc[prob.index]
        oos_accuracy = float((pred == y_aligned).mean())
        if prob.nunique() > 1:
            prob_label_corr = float(np.corrcoef(
                prob.values.astype(float), y_aligned.values.astype(float)
            )[0, 1])

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'n_features': len(features),
        'n_raw_trades_used': diag['n_used_trades'],
        'n_scaffold_bars': diag['n_windows'],
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'n_events_enriched': enriched_result['n_events_after'],
        'fracdiff_d': enriched_result['fracdiff_d'],
        'raw_signal_corr': raw_corr,
        'T_raw': eval_result['T_raw'],
        'tw_mean': eval_result['tw_mean'],
        'T_effective': eval_result['T'],
        'sr_hat': eval_result['sr_hat'],
        'dsr': eval_result['dsr'],
        'pbo': eval_result['prob_overfit'],
        'oos_accuracy': oos_accuracy,
        'prob_label_corr': prob_label_corr,
        'skew': eval_result['skew'],
        'kurtosis': eval_result['kurtosis'],
        'n_trials': eval_result['n_trials'],
        'wall_clock_sec': wall_clock_sec,
        'error': '',
    }


def append_result_row(row, results_path):
    df_row = pd.DataFrame([row])
    file_exists = os.path.exists(results_path)
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    df_row.to_csv(results_path, mode=mode, header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--edge-strengths', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-trades', type=int, default=None)
    parser.add_argument(
        '--n-trades-map', type=str, default=None,
        help='Path to a JSON file mapping edge_strength (string) -> n_trades '
             '(int), produced by calibrate_meanshift_n_trades.py. Same '
             'purpose as run_meanshift_edge_sweep.py\'s own flag -- holds '
             'T_effective roughly constant across the sweep. Any '
             'edge_strength not present falls back to --n-trades/N_TRADES.',
    )
    parser.add_argument('--target-bars', type=int, default=None)
    parser.add_argument(
        '--features', type=str, default=None,
        help='Comma-separated feature names to restrict the staged table '
             f'to. Default: {",".join(DEFAULT_FEATURES)} (see module '
             'LOAD-BEARING note for how these were chosen).',
    )
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS
    features = (
        [f.strip() for f in args.features.split(',')]
        if args.features else DEFAULT_FEATURES
    )

    n_trades_map = {}
    if args.n_trades_map:
        with open(args.n_trades_map) as f:
            n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}
        print(f'Loaded n_trades map from {args.n_trades_map}:')
        for es, nt in sorted(n_trades_map.items()):
            print(f'  edge_strength={es}: n_trades={nt:,}')
        print(f'  (any edge_strength not listed above falls back to '
              f'n_trades={default_n_trades:,})\n')

    print(f'Restricted feature set ({len(features)}): {features}\n')

    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)
    calib = load_calibration()
    print('Live-calibrated generator parameters:')
    for k, v in calib.items():
        print(f'  {k}: {v}')
    print()

    print('Loading Ch11 driver (once, reused across all combos)...')
    ch11 = load_ch11_driver()

    if args.smoke_test:
        combos = [(0.5, 0)]
        print(f'SMOKE TEST: running 1 combo only (edge_strength=0.5, seed=0), '
              f'n_trades={n_trades_map.get(0.5, default_n_trades)}, '
              f'target_bars={target_bars}\n')
    else:
        edge_strengths = (
            [float(x) for x in args.edge_strengths.split(',')]
            if args.edge_strengths else EDGE_STRENGTHS
        )
        seeds = (
            [int(x) for x in args.seeds.split(',')]
            if args.seeds else SEEDS
        )
        combos = [(es, s) for es in edge_strengths for s in seeds]
        print(f'SWEEP: {len(edge_strengths)} edge_strengths x '
              f'{len(seeds)} seeds = {len(combos)} combos, '
              f'target_bars={target_bars}\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        n_trades = n_trades_map.get(edge_strength, default_n_trades)
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed}, n_trades={n_trades:,} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, ch11, n_trades,
                                 target_bars, features)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(dsr={row['dsr']:.4f}, pbo={row['pbo']:.4f}, "
                  f"oos_acc={row['oos_accuracy']:.4f}, "
                  f"prob_corr={row['prob_label_corr']:+.4f}, "
                  f"T_eff={row['T_effective']:.2f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'n_features': len(features),
                'n_raw_trades_used': np.nan, 'n_scaffold_bars': np.nan,
                'n_bars': np.nan, 'n_events': np.nan,
                'n_events_enriched': np.nan, 'fracdiff_d': np.nan,
                'raw_signal_corr': np.nan, 'T_raw': np.nan, 'tw_mean': np.nan,
                'T_effective': np.nan, 'sr_hat': np.nan, 'dsr': np.nan,
                'pbo': np.nan, 'oos_accuracy': np.nan, 'prob_label_corr': np.nan,
                'skew': np.nan, 'kurtosis': np.nan,
                'n_trials': np.nan, 'wall_clock_sec': np.nan,
                'error': f'{type(e).__name__}: {e}',
            }
        results.append(row)
        append_result_row(row, results_path)

    print(f'\nAll combos done. Results written incrementally to {results_path}')
    n_failed = sum(1 for r in results if r['error'])
    if n_failed:
        print(f'WARNING: {n_failed}/{len(results)} combos failed -- see '
              f"the 'error' column in {results_path}")

    if not args.smoke_test and len(results) >= 2:
        df = pd.DataFrame(results).dropna(subset=['dsr', 'edge_strength'])
        if len(df) >= 2 and df['edge_strength'].nunique() >= 2:
            corr_dsr = df['edge_strength'].corr(df['dsr'])
            corr_pbo = df['edge_strength'].corr(df['pbo'])
            corr_acc = df['edge_strength'].corr(df['oos_accuracy'])
            print(f'\ncorr(edge_strength, dsr)          = {corr_dsr:+.4f}')
            print(f'corr(edge_strength, pbo)           = {corr_pbo:+.4f}')
            print(f'corr(edge_strength, oos_accuracy)  = {corr_acc:+.4f}')
            print('(compare against the full-14-feature sweep\'s '
                  'corr(edge_strength, dsr) = +0.0111 -- '
                  'meanshift_edge_sweep_results.csv)')

    print("""
INTERPRETATION GUIDE
---------------------
  - If dsr/pbo respond meaningfully to edge_strength here (dsr rising,
    pbo falling, oos_accuracy climbing as edge_strength grows) where the
    full-14-feature sweep was flat (corr(edge_strength, dsr)=+0.0111),
    that confirms the single-point isolation result generalizes across
    the whole sweep: dilution among uncorrelated features -- not a
    fundamental model/CV-stage ceiling -- explains the original null.
  - If dsr/pbo stay flat here too, despite the single-point isolation
    test showing a real accuracy/correlation improvement at es=2.0, that
    would mean the dilution fix doesn't survive averaging across seeds
    or doesn't hold at lower edge_strengths -- pointing back toward a
    genuine detection floor, with the es=2.0/seed=0 isolation result as
    a single favorable draw rather than a general fix.
""")


if __name__ == '__main__':
    main()
