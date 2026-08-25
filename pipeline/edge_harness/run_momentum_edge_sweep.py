"""
pipeline/edge_harness/run_momentum_edge_sweep.py

PHASE 2 of the momentum sanity-check (see 2026-08-23 handoff). The
2026-08-15 positive control (positive_control_data.generate_momentum_
trades via run_pipeline_positive_control.py) proved the pipeline CAN
detect a strong, saturating momentum edge (continuation_prob=0.85, PBO
0.029/DSR 0.854) -- but that single run used STALE pipeline defaults
(CUSUM_H=500, target_bars=250, S=8, all since superseded), a single
seed, and a signal strength far outside the range the 2026-08-23 OFI
sweep tested (raw_signal_corr up to ~0.52). This script asks the real
question left open by both: at CURRENT production defaults and at
CORRELATION-MATCHED signal strength, does the pipeline detect momentum
where it failed to detect order-flow imbalance?

continuation_prob values below are matched, via map_momentum_
correlation.py's phase-1 mapping (see momentum_correlation_mapping.csv
and the finer-resolution follow-up sweep in the 2026-08-23 handoff), to
land on realized bar-level lag-1 autocorrelation (momentum_correlation.
bar_lag1_autocorr) values comparable to the OFI sweep's raw_signal_corr
range at each of its 8 edge_strength points (0.0, 0.02, 0.05, 0.1, 0.15,
0.2, 0.3, 0.5, plus the extreme sanity check's 1.0/2.0) --
bar_aligned_scaled_50seeds.csv / edge_sweep_extreme_sanity_check.csv.

*** LOAD-BEARING (2026-08-23): CONTINUATION_PROBS below are SANDBOX-
DERIVED, NOT YET real-machine confirmed ***
map_momentum_correlation.py's mapping was run in Claude's sandbox
(numpy 2.4/pandas 3.0), not mlfinlab's pinned numpy 1.23.5/pandas 1.5.3
-- the RNG stream is expected to match (default_rng's PCG64 has proven
bit-identical across numpy versions in this project before, per the
2026-08-22 handoff), but this has NOT been explicitly re-verified for
this specific script. Run map_momentum_correlation.py for real in
mlfinlab FIRST and compare its output against
momentum_correlation_mapping.csv before trusting this grid -- if the
real mapping shifts meaningfully, update CONTINUATION_PROBS below to
match before running this (expensive, ~50-seed) sweep.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\run_momentum_edge_sweep.py --smoke-test
    python pipeline\\edge_harness\\run_momentum_edge_sweep.py
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

from positive_control_data import generate_momentum_trades    # noqa: E402
from rebuild import build_bars_and_labels                     # noqa: E402
from features import build_enriched_events                     # noqa: E402
from live_staging import stage_live_training_tables             # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402
from momentum_correlation import bar_lag1_autocorr               # noqa: E402

DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'momentum_edge_sweep_50seeds.csv')

SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work_momentum')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# LOAD-BEARING (2026-08-23, CORRECTED): originally set to N_TRADES=24,000
# / TARGET_BARS=1000 to match map_momentum_correlation.py's phase-1
# mapping scale and the LIVE PIPELINE's current production target_bars
# default -- but that scale is NOT the one the OFI harness's actual
# well-powered comparison used. The first full run at 24k/1000 came back
# with mean T_effective only 64-124 across the grid, with ~1% of all 400
# runs reaching the T_effective>=200 reliability bar -- underpowered, not
# a trustworthy null. bar_aligned_scaled_50seeds.csv (the OFI result this
# script is meant to compare against) specifically scaled to
# N_TRADES=360,000/target_bars=3000 to reach T_effective 200-550; this
# script now matches that scale exactly (same trades/bar density, ~120,
# too) for a genuinely apples-to-apples comparison. NOTE: this means the
# correlation mapping from map_momentum_correlation.py (run at 24k/1000)
# is NOT directly valid at this new scale -- CONTINUATION_PROBS below
# were re-verified at 360k/3000 (see the "corrected scale" spot-check in
# the 2026-08-23 handoff) before trusting this grid at the new scale.
N_TRADES = 360_000
TARGET_BARS = 3000
TOTAL_SPAN_HOURS = 720.0
PBO_S = 12

# LOAD-BEARING (2026-08-24): tick_bp/noise_std loaded from the real
# calibration run (calibrate_synthetic_momentum_params.py), NOT
# hardcoded -- the previous hardcoded generator defaults (tick_bp=0.0005,
# noise_std=0.0003) were exactly the staleness this calibration fixed:
# measured CUSUM firing rate 59.8% at the old values vs. real BTC's
# 14.7%, ~14.5% after calibration (k=0.2791, 4-seed spot-check spread
# only 2.4pp -- see 2026-08-24 handoff). Loading from the JSON here
# instead of copying the calibrated numbers in as new literals avoids
# reintroducing the same silent-drift failure mode this fix addressed.
MOMENTUM_BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_momentum_baseline_params.json'
)
if not os.path.exists(MOMENTUM_BASELINE_PARAMS_PATH):
    raise SystemExit(
        f'{MOMENTUM_BASELINE_PARAMS_PATH} not found -- run '
        'calibrate_synthetic_momentum_params.py first (see 2026-08-24 '
        'handoff).'
    )
with open(MOMENTUM_BASELINE_PARAMS_PATH) as _f:
    _momentum_baseline_params = json.load(_f)
if (_momentum_baseline_params['n_trades'] != N_TRADES
        or _momentum_baseline_params['target_bars'] != TARGET_BARS):
    raise SystemExit(
        "synthetic_momentum_baseline_params.json was calibrated at "
        f"n_trades={_momentum_baseline_params['n_trades']}, "
        f"target_bars={_momentum_baseline_params['target_bars']}, but "
        f"this script's scale is n_trades={N_TRADES}, "
        f"target_bars={TARGET_BARS}. CUSUM firing rate is scale-"
        "dependent (measured directly in the 2026-08-24 calibration run "
        "-- 59.8% at n_trades=360000 vs. a much lower rate at the old "
        "24k/1000 scale) -- re-run calibrate_synthetic_momentum_params.py "
        "at THIS exact scale before trusting these calibrated params "
        "here, don't assume the existing JSON still applies."
    )
CALIBRATED_TICK_BP = _momentum_baseline_params['calibrated_tick_bp']
CALIBRATED_NOISE_STD = _momentum_baseline_params['calibrated_noise_std']

# CORRECTED (2026-08-23): the mapping done at the OLD N_TRADES=24,000/
# target_bars=1000 scale does NOT hold at this corrected N_TRADES=
# 360,000/target_bars=3000 scale -- the same continuation_prob produces
# a MUCH higher realized correlation at this larger/denser scale (e.g.
# cp=0.530 gave autocorr~0.06 at the old scale, ~0.19 at this scale).
# Re-mapped directly at 360k/3000 (3-seed spot-checks, sandbox + one
# real-machine-confirmed anchor point -- see 2026-08-23 handoff for the
# full fine-grained sweep): these 8 values give realized bar_lag1_
# autocorr of approximately [0.00, 0.03, 0.06, 0.10, 0.16, 0.25, 0.41,
# 0.48], tracking the OFI sweep's tested raw_signal_corr range (0.0 up
# to ~0.51, from bar_aligned_scaled_50seeds.csv / edge_sweep_extreme_
# sanity_check.csv) closely again at the corrected scale.
# CORRECTED (2026-08-24): the top grid point (originally 0.605, carried
# over from the mapping done BEFORE the momentum generator's price
# dynamics were calibrated -- see calibrate_synthetic_momentum_params.py)
# overshot its target once tick_bp/noise_std were calibrated: a 3-seed
# pilot at cp=0.605 with the CALIBRATED params realized autocorr~0.623,
# not the intended ~0.48 (a ~30% relative miss, consistent across all 3
# seeds, not noise). Working hypothesis: the diff_std calibration was
# done at a near-null continuation_prob (~0.5001, no persistent drift),
# so it never exercised the price-level compounding a strong, sustained
# regime produces over a 360k-trade/30-day span -- a second-order
# nonlinearity the simple "scale tick_bp/noise_std together by k"
# correction didn't account for. Re-bracketed empirically (NOT just
# extrapolated): cp=0.563 -> autocorr 0.464 (3-seed mean), cp=0.572 ->
# autocorr 0.507 (3-seed mean); linear interpolation between these two
# real, close-bracketing points gives cp~0.566 for the 0.48 target. The
# other 7 grid points were spot-checked against a 3-seed pilot and held
# closely to their targets (largest gap ~0.011) -- only this top point
# needed correction.
CONTINUATION_PROBS = [0.505, 0.510, 0.515, 0.521, 0.527, 0.537, 0.558, 0.566]
SEEDS = list(range(50))  # matches bar_aligned_scaled_50seeds.csv's power


def run_one_combo(continuation_prob, seed, ch11, n_trades, target_bars):
    t_start = time.time()

    synth = generate_momentum_trades(
        n_trades,
        continuation_prob=continuation_prob,
        tick_bp=CALIBRATED_TICK_BP,
        noise_std=CALIBRATED_NOISE_STD,
        total_span_hours=TOTAL_SPAN_HOURS,
        random_state=seed,
    )
    raw_trades = synth['raw_trades']

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    raw_corr = bar_lag1_autocorr(rebuild_result['close'].values)

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
        'continuation_prob': continuation_prob,
        'seed': seed,
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'n_events_enriched': enriched_result['n_events_after'],
        'fracdiff_d': enriched_result['fracdiff_d'],
        'bar_lag1_autocorr': raw_corr,
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
    parser.add_argument('--continuation-probs', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-trades', type=int, default=None)
    parser.add_argument('--target-bars', type=int, default=None)
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    n_trades = args.n_trades if args.n_trades else N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS

    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('Loading Ch11 driver (once, reused across all combos)...')
    ch11 = load_ch11_driver()

    if args.smoke_test:
        combos = [(0.605, 0)]
        print(f'SMOKE TEST: running 1 combo only (continuation_prob=0.605, '
              f'seed=0), n_trades={n_trades}, target_bars={target_bars}\n')
    else:
        cps = (
            [float(x) for x in args.continuation_probs.split(',')]
            if args.continuation_probs else CONTINUATION_PROBS
        )
        seeds = (
            [int(x) for x in args.seeds.split(',')]
            if args.seeds else SEEDS
        )
        combos = [(cp, s) for cp in cps for s in seeds]
        print(f'SWEEP: {len(cps)} continuation_probs x {len(seeds)} seeds '
              f'= {len(combos)} combos, n_trades={n_trades}, '
              f'target_bars={target_bars}\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (cp, seed) in enumerate(combos):
        print(f'[{i+1}/{len(combos)}] continuation_prob={cp}, seed={seed} '
              f'... ', end='', flush=True)
        try:
            row = run_one_combo(cp, seed, ch11, n_trades, target_bars)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(dsr={row['dsr']:.4f}, pbo={row['pbo']:.4f}, "
                  f"T_eff={row['T_effective']:.2f}, "
                  f"autocorr={row['bar_lag1_autocorr']:.4f}, "
                  f"n_bars={row['n_bars']})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'continuation_prob': cp, 'seed': seed,
                'n_bars': np.nan, 'n_events': np.nan,
                'n_events_enriched': np.nan, 'fracdiff_d': np.nan,
                'bar_lag1_autocorr': np.nan, 'T_raw': np.nan,
                'tw_mean': np.nan, 'T_effective': np.nan, 'sr_hat': np.nan,
                'dsr': np.nan, 'pbo': np.nan, 'skew': np.nan,
                'kurtosis': np.nan, 'n_trials': np.nan,
                'wall_clock_sec': np.nan,
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
