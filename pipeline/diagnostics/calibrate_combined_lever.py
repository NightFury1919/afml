"""
pipeline/diagnostics/calibrate_combined_lever.py

Direct follow-on to CALIBRATION_AUDIT.md's "CUSUM_H Staleness Correction
vs. T_effective (2026-08-21)" section, which flagged this exact test as
untested: does combining CUSUM_H=313 (the staleness-corrected value, which
alone COSTS T_effective -34.9%) with target_bars=500 (the one lever
independently shown to HELP T_effective, +77% on 2026-08-20) net out
positive, negative, or somewhere in between?

Also re-runs target_bars_500 ALONE on TODAY's snapshot (not just reusing
2026-08-20's row), since the prior section's own caveat noted the h=250
vs h=313/375 comparison already crossed two different snapshot days --
this keeps the full four-way comparison (baseline / target_bars_500 alone
/ cusum_h_313 alone / combined) on ONE snapshot, avoiding that same
cross-day confound here.

Reuses calibrate_t_effective_levers.py's real _run_one_config() directly
-- no reimplementation. baseline and cusum_h_313 rows are NOT re-run here
(already real-machine-confirmed today in cusum_h_correction_calibration.csv
on this exact snapshot) -- this script only runs the two NEW configs
needed to complete the four-way comparison, and the write-up will pull
the other two rows from that existing CSV.

Run (same snapshot as today's other CUSUM_H work -- do NOT use a
different/fresh snapshot, or this loses its clean same-day comparison
against cusum_h_correction_calibration.csv's existing rows):
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\calibrate_combined_lever.py pipeline\diagnostics\t_effective_snapshot_2026-08-21
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_t_effective_levers import _run_one_config  # noqa: E402

OUTPUT_CSV = os.path.join(HERE, 'combined_lever_calibration.csv')
SWEEP_COLUMNS = [
    'config', 'target_bars', 'cusum_h', 'vertical_barrier_num_days',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

CONFIGS = [
    ('target_bars_500_today', {'target_bars': 500}),
    ('combined_tb500_h313', {'target_bars': 500, 'CUSUM_H': 313}),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python calibrate_combined_lever.py <snapshot_dir>\n'
            'Use the SAME snapshot as today\'s other CUSUM_H work '
            '(t_effective_snapshot_2026-08-21), not a fresh one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    work_root = os.path.join(HERE, 'combined_lever_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for config_name, overrides in CONFIGS:
        print(f'\n=== Running config: {config_name} ({overrides}) ===')
        row = _run_one_config(raw_trades, config_name, overrides, work_root)
        row['config'] = config_name
        rows.append(row)
        print(f"  [{config_name}] T_raw={row['T_raw']}, tw_mean={row['tw_mean']:.4f}, "
              f"T_effective={row['T_effective']:.2f}, DSR={row['dsr']:.4f}, "
              f"PBO={row['pbo']:.4f}")

    file_exists = os.path.exists(OUTPUT_CSV)
    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
    print(f'\nResults appended to {OUTPUT_CSV}')

    tb500 = df[df['config'] == 'target_bars_500_today'].iloc[0]
    combined = df[df['config'] == 'combined_tb500_h313'].iloc[0]
    print(f"\ntarget_bars=500 alone (today): T_effective={tb500['T_effective']:.2f}")
    print(f"combined (tb=500 + h=313): T_effective={combined['T_effective']:.2f}")
    print('\nCompare both against baseline=62.14 and cusum_h_313_alone=40.47 '
          '(from cusum_h_correction_calibration.csv, same snapshot) to see '
          'whether the combination nets out ahead of baseline or not.')


if __name__ == '__main__':
    main()