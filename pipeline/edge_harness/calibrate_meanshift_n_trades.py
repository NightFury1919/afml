"""
pipeline/edge_harness/calibrate_meanshift_n_trades.py

Fixes a real confound found in the 2026-08-27 meanshift edge sweep
(meanshift_edge_sweep_results.csv, edge_strength 0.75-2.0): T_effective
dropped sharply as edge_strength rose (mean ~110 at es=0.5 down to ~63 at
es=2.0, corr(edge_strength, T_effective) = -0.74 on that batch alone) --
turning up the injected signal ALSO shrank the effective sample size,
because a stronger drift triggers CUSUM events more often and more
densely, which (same trade-off already documented for CUSUM_H/target_bars
elsewhere in this project) dilutes average per-event uniqueness (tw_mean).
This meant "does DSR respond to a stronger signal" and "does DSR respond
to a smaller sample" were both changing at once, unable to be
disentangled from the sweep's own DSR-vs-edge_strength readings.

FIX: rather than derive a closed-form n_trades(edge_strength) relationship
(the earlier data doesn't support one -- event counts didn't scale
cleanly with edge_strength either), calibrate it empirically per
edge_strength, same "search a small candidate grid, real pipeline call,
pick the closest match" discipline calibrate_tao_cusum_h.py already used.

METHOD: for each edge_strength in the target grid, run ONE real seed
(seed=0) at each candidate n_trades, record T_effective, and pick whichever
candidate lands closest to TARGET_T_EFFECTIVE. This is a calibration pass
only (1 seed/candidate) -- NOT the full multi-seed precision run; once
this produces a mapping, run_meanshift_edge_sweep.py's --n-trades-map
argument uses it for the real sweep.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\calibrate_meanshift_n_trades.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)

from rebuild import build_bars_and_labels             # noqa: E402
from features import build_enriched_events             # noqa: E402
from live_staging import stage_live_training_tables     # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402

from generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades  # noqa: E402

BASELINE_PARAMS_PATH = os.path.join(DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json')
OUTPUT_CSV = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_calibration.csv')
OUTPUT_MAP = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_map.json')
SWEEP_WORK_DIR = os.path.join(HERE, 'sweep_work_meanshift_calib')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# The edge_strength grid we actually care about holding T_effective stable
# across -- matches the earlier sweep's 0.5-2.0 extension.
EDGE_STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0]

# Candidate n_trades per edge_strength -- starts at the original 120,000
# baseline (which gave T_eff~110 at es=0.5) and scales UP, since stronger
# edges shrink T_effective at fixed n_trades; more raw trades gives the
# CUSUM/bar-construction chain more room before density dilution bites.
CANDIDATE_N_TRADES = [120_000, 160_000, 220_000, 300_000, 400_000]

TARGET_T_EFFECTIVE = 110.0  # matches es=0.5's own real T_eff at the 120k baseline
TARGET_BARS = 1000
PBO_S = 12
CALIB_SEED = 0  # single seed per candidate -- calibration pass, not precision


def load_calibration():
    with open(BASELINE_PARAMS_PATH) as f:
        params = json.load(f)
    mean_rate = params['n_trades'] / (params['span_hours'] * 3600.0)
    return {
        'baseline_imbalance': params['baseline_imbalance'],
        'price_diff_std': params['price_diff_std'],
        'avg_trade_size': params['avg_trade_size'],
        'avg_trade_rate_per_sec': mean_rate,
        'start_price': params['price_start'],
    }


def run_one(edge_strength, n_trades, calib, ch11):
    t0 = time.time()
    raw_trades, diag = generate_bar_aligned_meanshift_trades(
        n_trades=n_trades, target_bars=TARGET_BARS, edge_strength=edge_strength,
        seed=CALIB_SEED, return_diagnostics=True, **calib,
    )
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    stage_live_training_tables(rebuild_result, enriched_result, STAGING_DIR)
    M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)
    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    eval_result = evaluate_overfitting(M, meta, ch11, S=PBO_S, tw=tw_aligned)
    return {
        'edge_strength': edge_strength, 'n_trades': n_trades,
        'n_events': len(rebuild_result['events']),
        'T_effective': eval_result['T'], 'tw_mean': eval_result['tw_mean'],
        'wall_clock_sec': time.time() - t0, 'error': '',
    }


def main():
    calib = load_calibration()
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)
    print('Loading Ch11 driver (once, reused across all candidates)...')
    ch11 = load_ch11_driver()

    rows = []
    chosen_map = {}
    for es in EDGE_STRENGTHS:
        print(f'\n=== edge_strength={es} ===')
        candidates = []
        for n_trades in CANDIDATE_N_TRADES:
            print(f'  n_trades={n_trades:,} ... ', end='', flush=True)
            try:
                row = run_one(es, n_trades, calib, ch11)
                candidates.append(row)
                print(f"T_eff={row['T_effective']:.2f}, tw_mean={row['tw_mean']:.4f}, "
                      f"n_events={row['n_events']} ({row['wall_clock_sec']:.1f}s)")
            except Exception as e:
                print(f'FAILED: {type(e).__name__}: {e}')
                candidates.append({
                    'edge_strength': es, 'n_trades': n_trades, 'n_events': np.nan,
                    'T_effective': np.nan, 'tw_mean': np.nan, 'wall_clock_sec': np.nan,
                    'error': f'{type(e).__name__}: {e}',
                })
        rows.extend(candidates)

        valid = [c for c in candidates if not np.isnan(c['T_effective'])]
        if not valid:
            print(f'  WARNING: no valid candidates for edge_strength={es} -- '
                  f'falling back to the largest n_trades tried.')
            chosen_map[str(es)] = CANDIDATE_N_TRADES[-1]
            continue
        best = min(valid, key=lambda c: abs(c['T_effective'] - TARGET_T_EFFECTIVE))
        chosen_map[str(es)] = best['n_trades']
        print(f'  -> chosen n_trades={best["n_trades"]:,} '
              f'(T_eff={best["T_effective"]:.2f}, target={TARGET_T_EFFECTIVE})')

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_MAP, 'w') as f:
        json.dump(chosen_map, f, indent=2)

    print(f'\nCalibration grid written to {OUTPUT_CSV}')
    print(f'n_trades map written to {OUTPUT_MAP}')
    print('\nChosen mapping (edge_strength -> n_trades):')
    for es, nt in chosen_map.items():
        print(f'  {es}: {nt:,}')
    print("""
NEXT STEP: re-run the real sweep using this map instead of a fixed
n_trades, so edge_strength varies while T_effective stays roughly
constant -- isolating "does DSR respond to a stronger signal" from "does
DSR respond to a smaller sample," which the 0.75-2.0 batch could not
previously distinguish:

    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py ^
        --edge-strengths 0.5,0.75,1.0,1.5,2.0 ^
        --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
""")


if __name__ == '__main__':
    main()
