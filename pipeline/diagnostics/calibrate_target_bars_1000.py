"""
pipeline/diagnostics/calibrate_target_bars_1000.py

Direct follow-on to the target_bars=750 section (2026-08-21), which found
target_bars scaling starting to plateau past 500 (250->500 was slightly
super-linear, 500->750 clearly sub-linear) and the combined-lever retained
fraction WORSENING at 750 vs 500 (58.2% vs 66.8%) -- the opposite of the
pre-registered hypothesis. This tests target_bars=1000, the other value
explicitly flagged as untested in that section's caveat, to see whether
the plateau continues/flattens further and whether the retained-fraction
trend keeps declining or reverses.

Reuses calibrate_t_effective_levers.py's real _run_one_config() directly
-- no reimplementation. Same snapshot as every other target_bars/CUSUM_H
result today, for a clean same-day four-point scaling curve (250/500/750/
1000).

Run:
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\calibrate_target_bars_1000.py pipeline\diagnostics\t_effective_snapshot_2026-08-21
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_t_effective_levers import _run_one_config  # noqa: E402

OUTPUT_CSV = os.path.join(HERE, 'target_bars_1000_calibration.csv')
SWEEP_COLUMNS = [
    'config', 'target_bars', 'cusum_h', 'vertical_barrier_num_days',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

CONFIGS = [
    ('target_bars_1000_today', {'target_bars': 1000}),
    ('combined_tb1000_h313', {'target_bars': 1000, 'CUSUM_H': 313}),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python calibrate_target_bars_1000.py <snapshot_dir>\n'
            'Use the SAME snapshot as today\'s other CUSUM_H/target_bars '
            'work (t_effective_snapshot_2026-08-21), not a fresh one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    work_root = os.path.join(HERE, 'target_bars_1000_work')
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

    tb1000 = df[df['config'] == 'target_bars_1000_today'].iloc[0]
    combined = df[df['config'] == 'combined_tb1000_h313'].iloc[0]
    print(f"\ntarget_bars=1000 alone: T_effective={tb1000['T_effective']:.2f} "
          f"(vs tb=750 alone: 180.48, tb=500 alone: 131.05, baseline: 62.14)")
    print(f"combined (tb=1000 + h=313): T_effective={combined['T_effective']:.2f} "
          f"(vs combined tb=750+h=313: 105.07, tb=500+h=313: 87.57)")
    fraction_1000 = combined['T_effective'] / tb1000['T_effective']
    print(f"\nFraction of alone-gain retained after adding h=313: "
          f"tb=500 -> 66.8%, tb=750 -> 58.2%, tb=1000 -> {fraction_1000:.1%}")

    step_ratio_tb = 1000 / 750
    step_ratio_teff = tb1000['T_effective'] / 180.48
    print(f"\ntarget_bars step 750->1000: {step_ratio_tb:.2f}x target_bars, "
          f"{step_ratio_teff:.2f}x T_effective "
          f"({'super' if step_ratio_teff > step_ratio_tb else 'sub'}-linear)")


if __name__ == '__main__':
    main()