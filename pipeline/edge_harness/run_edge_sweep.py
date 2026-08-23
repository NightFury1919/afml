"""
pipeline/edge_harness/run_edge_sweep.py

The actual edge-detection sweep: generates a synthetic raw-trade tape at
real production scale (~120k trades, matching the 2026-08-22 live
calibration pull) for each (edge_strength, seed) combination, runs it
through the REAL pipeline chain -- build_bars_and_labels -> 
build_enriched_events -> stage_live_training_tables -> run_live_trials ->
evaluate_overfitting -- and records DSR/PBO/T_effective for each run.

This answers: how large does the injected order-flow-imbalance edge need
to be before this pipeline's real machinery (dollar bars, CUSUM, triple-
barrier labels, Ch19 features, Ch11's SVC grid, Ch14 DSR) actually
detects it?

Design provenance
------------------
- generate_synthetic_trades() / synthetic_trade_adapter.py: built and
  TDD-confirmed 2026-08-22 (see pipeline/edge_harness/
  test_generate_synthetic_trades.py -- a real bug was found and fixed
  there: the null case wasn't actually null until fixed).
- Real pipeline function signatures (build_bars_and_labels,
  build_enriched_events, stage_live_training_tables, load_ch11_driver,
  run_live_trials, evaluate_overfitting) confirmed by reading the actual
  current source from GitHub on 2026-08-22, not assumed from memory.
- Trade-count scale (~120k) and rate/price/imbalance calibration sourced
  from a FRESH LIVE pull (calibrate_synthetic_trade_params.py
  --source live, 2026-08-22), not the March static CSV -- the static
  CSV's arrival rate is ~13x slower than live BTCUSDT (static: 0.00344
  trades/sec; live: 0.04666 trades/sec), so using it would have produced
  an internally-inconsistent synthetic tape (real trade COUNT but wrong
  time density/price level/volatility regime).

*** LOAD-BEARING (2026-08-22): mean-based (not median-based) arrival
rate ***
Both the static AND live calibration pulls showed the SAME pattern:
median inter-trade gap implies a much higher tick rate than the trade
count / span actually supports (static: 4.4x discrepancy; live: 16x).
Real trade arrivals are bursty, not a steady stream. This generator uses
the honest average rate (n_trades / span_hours) as a homogeneous Poisson
arrival rate -- reproduces the real trade COUNT over the real SPAN, does
NOT reproduce real burst clustering. See generate_synthetic_trades.py's
own module docstring for the full reasoning.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\run_edge_sweep.py --smoke-test
        # runs ONE combo (edge_strength=0.2, seed=0) to confirm the
        # wiring works and to get a real per-run timing number before
        # committing to the full sweep
    python pipeline\\edge_harness\\run_edge_sweep.py
        # full sweep: 8 edge_strengths x 5 seeds = 40 runs
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # pipeline/edge_harness
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))    # pipeline/
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')

# LOAD-BEARING (2026-08-22): matches run_pipeline_live.py's own exact
# sys.path pattern -- rebuild.py/features.py/stages.py/live_staging.py
# all use BARE imports between themselves (e.g. rebuild.py does
# `from ingestion import _disambiguate_timestamps`, not a package-
# relative import), which only resolve if pipeline/orchestration itself
# is directly on sys.path. Each of those modules independently inserts
# the repo ROOT for its OWN chapter-module imports (ch02/ch03/etc.) --
# not this script's job to do that too.
sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)  # for the local generate_synthetic_trades / adapter

from rebuild import build_bars_and_labels             # noqa: E402
from features import build_enriched_events             # noqa: E402
from live_staging import stage_live_training_tables     # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402

from generate_synthetic_trades import generate_synthetic_trades  # noqa: E402
from synthetic_trade_adapter import synthetic_to_raw_trades_schema  # noqa: E402

DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'edge_sweep_results.csv')

# LOAD-BEARING (2026-08-22): scratch dirs, regenerable output -- matches
# this project's "one real frozen snapshot per investigation, delete
# after" convention (e.g. t_effective_snapshot_2026-08-21). Safe to
# delete pipeline/edge_harness/sweep_work/ once the sweep is done and
# edge_sweep_results.csv has been reviewed/committed.
SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')  # ch11's PNG output target

# LOAD-BEARING (2026-08-22): n_windows x trades_per_window = 120,000,
# matching the 2026-08-22 live calibration pull's 120,941 real trades
# (Ethan chose "match real scale" explicitly). trades_per_window=300 is
# on the same order as this pipeline's real bars-per-trade-count ratio
# (today's production run: 114,116 trades / 855 bars =~ 133 trades/bar),
# so an injected window's imbalance signal has a realistic chance of
# showing up within a few bars of the next window, rather than being
# smeared across many bars or crammed into a fraction of one.
N_WINDOWS = 400
TRADES_PER_WINDOW = 300

# LOAD-BEARING (2026-08-22): edge_strength sweep values -- includes a
# TRUE zero (null-case sanity check against this project's established
# real findings: PBO~0.83, DSR not surviving at S=12) plus a spread
# spanning what the generator's own TDD (2026-08-22) confirmed produces
# a clearly detectable raw imbalance-drift correlation (edge_strength=
# 0.2 -> mean raw correlation ~0.12 over many seeds) down to much
# smaller values, to bracket the pipeline's actual detection threshold
# rather than assuming it in advance.
EDGE_STRENGTHS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
SEEDS = [0, 1, 2, 3, 4]

TARGET_BARS = 1000  # matches run_pipeline_live.py's current production default
PBO_S = 12           # matches run_pipeline_live.py's current production default


def load_calibration():
    """Load the live-calibrated baseline stats
    (calibrate_synthetic_trade_params.py --source live, 2026-08-22) and
    derive the honest MEAN-based arrival rate (not the file's own
    median-based tick_density_per_sec field -- see module LOAD-BEARING
    note on burstiness)."""
    if not os.path.exists(BASELINE_PARAMS_PATH):
        raise SystemExit(
            f'{BASELINE_PARAMS_PATH} not found. Run '
            'calibrate_synthetic_trade_params.py --source live first.'
        )
    with open(BASELINE_PARAMS_PATH) as f:
        params = json.load(f)
    if params.get('source') != 'live':
        print(
            f"WARNING: {BASELINE_PARAMS_PATH} was calibrated from "
            f"source={params.get('source')!r}, not 'live'. The sweep's "
            "trade-count scale (~120k) was chosen to match a LIVE "
            "calibration pull -- re-run calibrate_synthetic_trade_"
            "params.py --source live if this is stale.",
            file=sys.stderr,
        )
    mean_rate_per_sec = params['n_trades'] / (params['span_hours'] * 3600.0)
    return {
        'baseline_imbalance': params['baseline_imbalance'],
        'price_diff_std': params['price_diff_std'],
        'avg_trade_size': params['avg_trade_size'],
        'avg_trade_rate_per_sec': mean_rate_per_sec,
        'start_price': params['price_start'],
    }


def run_one_combo(edge_strength, seed, calib, ch11):
    """Run the full real chain for one (edge_strength, seed) combination.
    Returns a result dict. Raises nothing -- callers should still wrap in
    try/except since chapter-module functions themselves can raise
    (empty bars, zero CUSUM events, etc.) on an unlucky draw."""
    t_start = time.time()

    df, diag = generate_synthetic_trades(
        n_windows=N_WINDOWS,
        trades_per_window=TRADES_PER_WINDOW,
        edge_strength=edge_strength,
        seed=seed,
        baseline_imbalance=calib['baseline_imbalance'],
        price_diff_std=calib['price_diff_std'],
        avg_trade_rate_per_sec=calib['avg_trade_rate_per_sec'],
        avg_trade_size=calib['avg_trade_size'],
        start_price=calib['start_price'],
        return_diagnostics=True,
    )
    raw_trades = synthetic_to_raw_trades_schema(df)

    # Raw-signal sanity check: is the injected edge actually present in
    # the raw generated data for THIS specific run, independent of
    # whether the downstream pipeline detects it? (Same diagnostic used
    # in the generator's own TDD.)
    imb = diag['realized_imbalance']
    wmp = diag['window_mean_price']
    drift_next = np.diff(wmp)
    raw_corr = float(np.corrcoef(imb[:-1], drift_next)[0, 1])

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
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
        raise ValueError(
            'tw has NaN after reindexing to the enriched event index '
            '(same invariant run_pipeline_live.py enforces).'
        )

    eval_result = evaluate_overfitting(M, meta, ch11, S=PBO_S, tw=tw_aligned)

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'n_raw_trades': len(raw_trades),
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


def append_result_row(row, results_path, is_first_row_of_run):
    """Incremental checkpointing -- flush immediately so an interrupted
    sweep never loses completed runs.

    LOAD-BEARING (2026-08-22), BUG FOUND AND FIXED same day: the original
    version decided whether to write a header based on `i==0` (this run's
    OWN loop position), which meant mode='w' fired on the first row of
    EVERY invocation of this script -- a second sweep run (e.g. an
    extreme-edge sanity check) would have silently OVERWRITTEN the first
    sweep's already-completed 40 rows. Fixed: base the decision on
    whether the destination FILE already exists on disk, not on this
    run's loop index -- multiple invocations now safely accumulate
    (append) rather than clobber each other, as long as they pass the
    same --output path (or omit it, both default to the same file)."""
    df_row = pd.DataFrame([row])
    file_exists = os.path.exists(results_path)
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    df_row.to_csv(results_path, mode=mode, header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test', action='store_true',
        help='Run ONE combo (edge_strength=0.2, seed=0) only, to confirm '
             'the wiring works and get a real per-run timing number '
             'before committing to the full sweep.',
    )
    parser.add_argument(
        '--edge-strengths', type=str, default=None,
        help='Comma-separated edge_strength values to sweep, e.g. '
             '"1.0,2.0". Overrides the default EDGE_STRENGTHS list. '
             'Ignored if --smoke-test is set.',
    )
    parser.add_argument(
        '--seeds', type=str, default=None,
        help='Comma-separated integer seeds, e.g. "0,1,2". Overrides '
             'the default SEEDS list. Ignored if --smoke-test is set.',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path. Defaults to '
             'pipeline/diagnostics/edge_sweep_results.csv. Use a '
             'different path (e.g. for an extreme-edge sanity check) to '
             'keep results separate from the main sweep -- see '
             "append_result_row()'s LOAD-BEARING note: re-running with "
             'the SAME path safely appends (does not overwrite) as long '
             'as that file already exists.',
    )
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH

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
        print('SMOKE TEST: running 1 combo only (edge_strength=0.2, seed=0)\n')
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
              f'{len(seeds)} seeds = {len(combos)} combos\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, ch11)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(dsr={row['dsr']:.4f}, pbo={row['pbo']:.4f}, "
                  f"T_eff={row['T_effective']:.2f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'n_raw_trades': np.nan, 'n_bars': np.nan, 'n_events': np.nan,
                'n_events_enriched': np.nan, 'fracdiff_d': np.nan,
                'raw_signal_corr': np.nan, 'T_raw': np.nan, 'tw_mean': np.nan,
                'T_effective': np.nan, 'sr_hat': np.nan, 'dsr': np.nan,
                'pbo': np.nan, 'skew': np.nan, 'kurtosis': np.nan,
                'n_trials': np.nan, 'wall_clock_sec': np.nan,
                'error': f'{type(e).__name__}: {e}',
            }
        results.append(row)
        append_result_row(row, results_path, is_first_row_of_run=(i == 0))

    print(f'\nAll combos done. Results written incrementally to '
          f'{results_path}')
    n_failed = sum(1 for r in results if r['error'])
    if n_failed:
        print(f'WARNING: {n_failed}/{len(results)} combos failed -- see '
              f"the 'error' column in {RESULTS_CSV_PATH}")


if __name__ == '__main__':
    main()
