"""
pipeline/diagnostics/calibrate_cusum_h_correction.py

Direct follow-on to CALIBRATION_AUDIT.md's "CUSUM_H Staleness Audit
(2026-08-21)" section, which measured h~313 as the value that would
restore March's relative CUSUM firing rate on today's live data -- a
MEASUREMENT, explicitly not a calibration decision on its own.

This script closes the open question that measurement's own caveats
section flagged: the 2026-08-20 T_effective lever sweep already showed
CUSUM_H=250 makes T_effective WORSE (-45%), via a tw_mean collapse from
label overlap (more events packed into the same bar window). h~313 is a
much smaller reduction than h=250, motivated by a completely different
reason (staleness correction, not event-count maximization) -- does it
avoid that same uniqueness-collapse mechanism, or does ANY reduction below
500 trigger it?

Reuses calibrate_t_effective_levers.py's real _run_one_config() directly
(same monkeypatch-and-restore pattern, same full rebuild -> enrich ->
stage -> Ch11 trials -> evaluate chain) -- no reimplementation. Only the
CONFIGS list and output CSV differ.

Run (after capture_t_effective_snapshot.py has produced a snapshot dir --
reuse an existing one if you have a fresh one from today, or run it again
to freeze a new one; do NOT reuse the deleted 2026-08-20 snapshot, since
that would silently mix today's staleness finding with three-day-old
market data):
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\calibrate_cusum_h_correction.py <snapshot_dir>
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_t_effective_levers import _run_one_config  # noqa: E402

OUTPUT_CSV = os.path.join(HERE, 'cusum_h_correction_calibration.csv')
SWEEP_COLUMNS = [
    'config', 'target_bars', 'cusum_h', 'vertical_barrier_num_days',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

# h=313: the 2026-08-21 staleness audit's measured value to restore March's
# relative CUSUM firing rate on current data (CALIBRATION_AUDIT.md,
# "CUSUM_H Staleness Audit" section). h=375 included as a bracketing point
# roughly midway between 313 and the established baseline of 500, to see
# whether any effect is monotonic/graded or a step change.
CONFIGS = [
    ('baseline', {}),
    ('cusum_h_313_staleness_corrected', {'CUSUM_H': 313}),
    ('cusum_h_375_bracket', {'CUSUM_H': 375}),
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python calibrate_cusum_h_correction.py <snapshot_dir>\n'
            'Run capture_t_effective_snapshot.py first to produce one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    work_root = os.path.join(HERE, 'cusum_h_correction_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for config_name, overrides in CONFIGS:
        print(f'\n=== Running config: {config_name} ({overrides or "baseline"}) ===')
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

    baseline = df[df['config'] == 'baseline'].iloc[0]
    for _, row in df[df['config'] != 'baseline'].iterrows():
        pct = (row['T_effective'] - baseline['T_effective']) / baseline['T_effective']
        print(f"\n{row['config']}: T_effective {baseline['T_effective']:.2f} -> "
              f"{row['T_effective']:.2f} ({pct:+.1%}), "
              f"tw_mean {baseline['tw_mean']:.4f} -> {row['tw_mean']:.4f}")


if __name__ == '__main__':
    main()