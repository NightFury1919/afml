"""
pipeline/diagnostics/calibrate_target_bars_750.py

Direct follow-on to two threads converging today: (1) CALIBRATION_AUDIT.md's
still-open "push target_bars further (750, 1000)" item, deferred since
2026-08-20, and (2) today's "Combined Lever" finding that target_bars=500
+ CUSUM_H=313 nets +40.9% T_effective over baseline, giving back ~33% of
target_bars=500-alone's gain to the staleness correction.

This tests target_bars=750 -- both alone (does the target_bars->T_effective
relationship keep scaling roughly linearly past 500, or start to plateau?)
and combined with CUSUM_H=313 (does a bigger target_bars base absorb the
staleness correction's cost more easily, recovering a larger FRACTION of
its alone-gain than target_bars=500 did?).

Reuses calibrate_t_effective_levers.py's real _run_one_config() directly
-- no reimplementation. Run against the SAME snapshot as today's other
CUSUM_H/target_bars work, for a clean same-day four-point comparison
(250/500/750, each alone and at 500/750 combined with h=313).

Run:
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\calibrate_target_bars_750.py pipeline\diagnostics\t_effective_snapshot_2026-08-21
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_t_effective_levers import _run_one_config  # noqa: E402

OUTPUT_CSV = os.path.join(HERE, 'target_bars_750_calibration.csv')
SWEEP_COLUMNS = [
    'config', 'target_bars', 'cusum_h', 'vertical_barrier_num_days',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

CONFIGS = [
    ('target_bars_750_today', {'target_bars': 750}),
    ('combined_tb750_h313', {'target_bars': 750, 'CUSUM_H': 313}),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python calibrate_target_bars_750.py <snapshot_dir>\n'
            'Use the SAME snapshot as today\'s other CUSUM_H/target_bars '
            'work (t_effective_snapshot_2026-08-21), not a fresh one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    work_root = os.path.join(HERE, 'target_bars_750_work')
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

    tb750 = df[df['config'] == 'target_bars_750_today'].iloc[0]
    combined = df[df['config'] == 'combined_tb750_h313'].iloc[0]
    print(f"\ntarget_bars=750 alone: T_effective={tb750['T_effective']:.2f} "
          f"(vs target_bars=500 alone: 131.05, target_bars=250 baseline: 62.14)")
    print(f"combined (tb=750 + h=313): T_effective={combined['T_effective']:.2f} "
          f"(vs combined tb=500+h=313: 87.57)")
    fraction_750 = combined['T_effective'] / tb750['T_effective']
    fraction_500 = 87.57 / 131.05
    print(f"\nFraction of alone-gain retained after adding h=313: "
          f"tb=500 -> {fraction_500:.1%}, tb=750 -> {fraction_750:.1%}")


if __name__ == '__main__':
    main()