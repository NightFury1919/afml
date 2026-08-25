"""
pipeline/edge_harness/map_momentum_correlation.py

PHASE 1 of the momentum sanity-check (see 2026-08-23 handoff, "momentum
positive control recalibration"). Maps positive_control_data.py's
continuation_prob dial to REALIZED bar-level lag-1 return autocorrelation
(momentum_correlation.bar_lag1_autocorr), at CURRENT production defaults
(CUSUM_H=313 via rebuild.py's module constant, target_bars=1000) -- the
2026-08-15 single run used the since-superseded CUSUM_H=500/target_bars
=250 defaults.

Deliberately CHEAP: only runs raw_trades -> build_bars_and_labels (bars
only), never touches features/live_staging/SVC/DSR/PBO. This is meant to
be fast enough to run for real in a couple of minutes, so the actual
continuation_prob values used in the expensive phase-2 sweep (matching
run_bar_aligned_edge_sweep.py's rigor -- n=50 seeds, full pipeline) can be
CHOSEN from real numbers, not guessed.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\map_momentum_correlation.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)

from positive_control_data import generate_momentum_trades  # noqa: E402
from rebuild import build_bars_and_labels                    # noqa: E402
from momentum_correlation import bar_lag1_autocorr             # noqa: E402

DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
OUTPUT_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'momentum_correlation_mapping.csv')

# LOAD-BEARING (2026-08-23): n_trades=24,000 chosen to target ~24
# trades/bar (this project's usual density, per positive_control_data.py's
# own docstring) at TARGET_BARS=1000 -- i.e. matched to CURRENT production
# scale, not the 2026-08-15 single run's n_trades=6000 (which targeted
# only ~244 bars under the since-superseded target_bars=250 default).
# compute_dynamic_threshold sets threshold = total_dollar_volume /
# target_bars, so realized bar count should land near TARGET_BARS
# regardless of continuation_prob -- verify this holds in the real dry
# run below and adjust if bar counts come back badly off.
N_TRADES = 24_000
TARGET_BARS = 1000
TOTAL_SPAN_HOURS = 720.0  # matches this project's LOOKBACK_HOURS convention

# Sweep from near-null (0.51) up to the original 2026-08-15 setting
# (0.85) -- fine enough resolution to see where realized correlation
# crosses into the OFI sweep's tested range (raw_signal_corr up to
# ~0.52, per bar_aligned_scaled_50seeds.csv / edge_sweep_extreme_
# sanity_check.csv).
CONTINUATION_PROBS = [0.51, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
SEEDS = [0, 1, 2, 3, 4]


def run_one(continuation_prob, seed):
    synth = generate_momentum_trades(
        N_TRADES,
        continuation_prob=continuation_prob,
        total_span_hours=TOTAL_SPAN_HOURS,
        random_state=seed,
    )
    raw_trades = synth['raw_trades']
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    close = rebuild_result['close']
    corr = bar_lag1_autocorr(close.values)
    return {
        'continuation_prob': continuation_prob,
        'seed': seed,
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'bar_lag1_autocorr': corr,
    }


def main():
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
    rows = []
    combos = [(cp, s) for cp in CONTINUATION_PROBS for s in SEEDS]
    print(f'Mapping {len(CONTINUATION_PROBS)} continuation_prob values x '
          f'{len(SEEDS)} seeds = {len(combos)} combos '
          f'(n_trades={N_TRADES}, target_bars={TARGET_BARS}, CUSUM_H=313 '
          f'production default)\n')
    for i, (cp, seed) in enumerate(combos):
        print(f'[{i+1}/{len(combos)}] continuation_prob={cp}, seed={seed} '
              f'... ', end='', flush=True)
        row = run_one(cp, seed)
        print(f"n_bars={row['n_bars']}, n_events={row['n_events']}, "
              f"autocorr={row['bar_lag1_autocorr']:.4f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f'\nWrote {OUTPUT_CSV_PATH}\n')

    summary = df.groupby('continuation_prob').agg(
        mean_autocorr=('bar_lag1_autocorr', 'mean'),
        std_autocorr=('bar_lag1_autocorr', 'std'),
        mean_n_bars=('n_bars', 'mean'),
        mean_n_events=('n_events', 'mean'),
    ).reset_index()
    print('=' * 78)
    print('SUMMARY: continuation_prob -> realized bar-level lag-1 autocorrelation')
    print('=' * 78)
    print(summary.to_string(index=False))
    print()
    print('Compare against the OFI sweep\'s raw_signal_corr range '
          '(edge_strength 0.0-0.5 -> corr roughly 0.0-0.3; extreme sanity '
          'check up to edge_strength=2.0 -> corr up to ~0.52).')
    print('Use this table to pick continuation_prob values for phase 2 '
          '(the full 50-seed DSR/PBO sweep) whose realized correlations '
          'overlap that range -- see run_momentum_edge_sweep.py once built.')


if __name__ == '__main__':
    main()
