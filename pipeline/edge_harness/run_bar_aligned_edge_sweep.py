"""
pipeline/edge_harness/run_bar_aligned_edge_sweep.py

Same sweep structure as run_edge_sweep.py, but using
generate_bar_aligned_trades.generate_bar_aligned_synthetic_trades()
instead of the fixed-trade-count-window generator -- tests whether
injecting the edge at REAL dollar-bar granularity (the unit Ch19's
features actually operate on) changes the detection picture found in
the 2026-08-22 sweep (zero DSR/PBO response even at saturating
edge_strength up to 2.0).

Uses the SAME edge_strength values and seeds as the original sweep
(pipeline/diagnostics/edge_sweep_results.csv) for a direct, apples-to-
apples comparison -- same live-calibrated baseline stats, same
target_bars=1000/CUSUM_H=313 production defaults, same S=12 PBO
precision, same 5 seeds.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\run_bar_aligned_edge_sweep.py --smoke-test
    python pipeline\\edge_harness\\run_bar_aligned_edge_sweep.py
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

from generate_bar_aligned_trades import generate_bar_aligned_synthetic_trades  # noqa: E402

DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'bar_aligned_edge_sweep_results.csv')

SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work_bar_aligned')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# LOAD-BEARING (2026-08-22): n_trades=120,000 matches the original
# trade-window sweep's scale exactly (same live calibration pull), for a
# direct comparison. target_bars=1000 matches production
# (run_pipeline_live.py's current default).
N_TRADES = 120_000

# Same edge_strength/seed grid as the original sweep
# (edge_sweep_results.csv), for direct comparability.
EDGE_STRENGTHS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2, 3, 4]

TARGET_BARS = 1000
PBO_S = 12


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


def run_one_combo(edge_strength, seed, calib, ch11, n_trades, target_bars):
    t_start = time.time()

    raw_trades, diag = generate_bar_aligned_synthetic_trades(
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

    imb = diag['realized_imbalance']
    wmp = diag['window_mean_price']
    drift_next = np.diff(wmp)
    raw_corr = float(np.corrcoef(imb[:-1], drift_next)[0, 1])

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    staged = stage_live_training_tables(
        rebuild_result, enriched_result, STAGING_DIR,
    )

    M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)

    tw_aligned = rebuild_result['tw'].reindex(
        enriched_result['enriched_events'].index
    )
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')

    eval_result = evaluate_overfitting(M, meta, ch11, S=PBO_S, tw=tw_aligned)

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'n_raw_trades_used': diag['n_used_trades'],
        'n_scaffold_bars': diag['n_windows'],  # pass-1 scaffold bar count
        'n_bars': len(rebuild_result['bars']),  # REAL bar count on the final (edge-injected) trades
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
    parser.add_argument(
        '--n-trades', type=int, default=None,
        help='Total synthetic trade count (overrides N_TRADES=120,000). '
             'Scale this UP TOGETHER WITH --target-bars (keeping their '
             'ratio similar to N_TRADES/TARGET_BARS=120) to extend the '
             'tape\'s implied time span rather than just re-slicing the '
             'same span into more/fewer bars -- see module docstring.',
    )
    parser.add_argument(
        '--target-bars', type=int, default=None,
        help='Target bar count for the dynamic dollar-bar threshold '
             '(overrides TARGET_BARS=1000).',
    )
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    n_trades = args.n_trades if args.n_trades else N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS

    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)
    calib = load_calibration()
    print('Live-calibrated generator parameters:')
    for k, v in calib.items():
        print(f'  {k}: {v}')
    print()

    print('Loading Ch11 driver (once, reused across all combos)...')
    ch11 = load_ch11_driver()

    if args.smoke_test:
        combos = [(0.2, 0)]
        print(f'SMOKE TEST: running 1 combo only (edge_strength=0.2, seed=0), '
              f'n_trades={n_trades}, target_bars={target_bars}\n')
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
              f'n_trades={n_trades}, target_bars={target_bars}\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, ch11, n_trades, target_bars)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(dsr={row['dsr']:.4f}, pbo={row['pbo']:.4f}, "
                  f"T_eff={row['T_effective']:.2f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f}, "
                  f"n_scaffold_bars={row['n_scaffold_bars']}, "
                  f"n_real_bars={row['n_bars']})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'n_raw_trades_used': np.nan, 'n_scaffold_bars': np.nan,
                'n_bars': np.nan, 'n_events': np.nan,
                'n_events_enriched': np.nan, 'fracdiff_d': np.nan,
                'raw_signal_corr': np.nan, 'T_raw': np.nan, 'tw_mean': np.nan,
                'T_effective': np.nan, 'sr_hat': np.nan, 'dsr': np.nan,
                'pbo': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
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


if __name__ == '__main__':
    main()
