"""
pipeline/edge_harness/run_meanshift_edge_sweep.py

Same sweep structure as run_bar_aligned_edge_sweep.py, but using
generate_bar_aligned_meanshift_trades.generate_bar_aligned_meanshift_trades()
-- a clean single-lag price-return-momentum edge instead of OFI's
engineered-imbalance edge or momentum's Markov-persistence edge. See
generate_bar_aligned_meanshift_trades.py's module docstring for why this
is a genuinely different, cleaner positive control than either existing
sweep.

PILOT SCALE FIRST (2026-08-27, Ethan's/this session's call): uses the
SAME edge_strength/seed grid as the ORIGINAL OFI pilot
(edge_sweep_results.csv: 8 edge_strengths x 5 seeds = 40 combos), not
yet the scaled 50-seed/400-combo precision OFI and momentum eventually
reached. Rationale: this is a brand-new generator that has never run
against the real chain -- confirm the apparatus and result shape at
pilot scale first (mirrors the exact same incremental discipline this
project used before scaling either prior sweep). Re-run at
--seeds 0-49 (mirroring bar_aligned_scaled_50seeds.csv's real scale)
once the pilot's shape looks sane.

*** WIDENED (2026-08-27, later same session): --n-trades-map added ***
The 0.75-2.0 edge_strength extension (run this same session) found
T_effective dropping sharply as edge_strength rose (mean ~110 at es=0.5
down to ~63 at es=2.0) -- a real confound, since a stronger injected
signal was also shrinking the effective sample size at the same time,
making the sweep's own DSR-vs-edge_strength reading impossible to
interpret cleanly. --n-trades-map accepts a JSON file (produced by
calibrate_meanshift_n_trades.py) mapping edge_strength -> n_trades,
overriding the single global --n-trades/N_TRADES value PER edge_strength
so T_effective can be held roughly constant while only edge_strength
varies. Falls back to the global n_trades for any edge_strength not
present in the map (so old-style single-n_trades runs still work
unchanged).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py --smoke-test
    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py
    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py --edge-strengths 0.5,0.75,1.0,1.5,2.0 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
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
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'meanshift_edge_sweep_results.csv')

SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work_meanshift')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# Same scale as OFI's original pilot (edge_sweep_results.csv) and
# production defaults -- see module docstring on why pilot scale first.
N_TRADES = 120_000
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

    raw_trades, diag = generate_bar_aligned_meanshift_trades(
        n_trades=n_trades,  # resolved per-edge_strength by the caller, see --n-trades-map
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

    ch11_M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)

    tw_aligned = rebuild_result['tw'].reindex(
        enriched_result['enriched_events'].index
    )
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')

    eval_result = evaluate_overfitting(ch11_M, meta, ch11, S=PBO_S, tw=tw_aligned)

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
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
             '(int), produced by calibrate_meanshift_n_trades.py. Overrides '
             '--n-trades/N_TRADES PER edge_strength, holding T_effective '
             'roughly constant across the sweep instead of letting it drift '
             'with edge_strength (see module docstring, 2026-08-27 fix). Any '
             'edge_strength not present in the map falls back to the global '
             '--n-trades/N_TRADES value.',
    )
    parser.add_argument('--target-bars', type=int, default=None)
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS

    n_trades_map = {}
    if args.n_trades_map:
        with open(args.n_trades_map) as f:
            n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}
        print(f'Loaded n_trades map from {args.n_trades_map}:')
        for es, nt in sorted(n_trades_map.items()):
            print(f'  edge_strength={es}: n_trades={nt:,}')
        print(f'  (any edge_strength not listed above falls back to '
              f'n_trades={default_n_trades:,})\n')

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
              f'n_trades={n_trades_map.get(0.2, default_n_trades)}, '
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

    print("""
INTERPRETATION GUIDE
---------------------
  - raw_signal_corr is the lag-1 autocorrelation of REALIZED window
    returns (a generic, non-engineered price feature) -- confirms the
    injected signal is really present and growing with edge_strength.
  - If dsr/pbo respond meaningfully to edge_strength here (dsr rising,
    pbo falling as edge_strength grows), that's a genuine positive
    result: the full pipeline CAN detect at least this shape of edge,
    and the OFI/momentum nulls are specific to those mechanisms, not a
    universal detection ceiling.
  - If dsr/pbo stay flat here too (same signature as OFI: 0.508-0.526
    flat, corr~0.016; momentum: similar), that is a MUCH more serious
    finding -- it would mean the ceiling is structural (bar
    construction, feature set, or CV/model stage itself), not specific
    to either prior injection mechanism, and would need to be run down
    before trusting ANY of this project's null results, including the
    already-closed BTC/XRP/TAO ones.
""")


if __name__ == '__main__':
    main()
