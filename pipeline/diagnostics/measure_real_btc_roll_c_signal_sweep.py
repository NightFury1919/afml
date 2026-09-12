"""
pipeline/diagnostics/measure_real_btc_roll_c_signal_sweep.py

Follow-on to measure_real_btc_roll_c_signal.py (2026-09-08, same session).
That script found roll_c's real bar-level correlation vs next-bar return
at target_bars=1000 (r=+0.0147) does not survive even a basic
significance check once corrected for testing all 14 Ch19 features
(Bonferroni alpha=0.05/14=0.00357; strongest real feature,
corwin_schultz_spread at r=+0.0746, p=0.019 uncorrected -- still fails).

Ethan's question: kraken_target_bars_calibration.csv (2026-08-25 snapshot)
already showed T_effective climbing cleanly from 123 (target_bars=1000)
to 758 (target_bars=5000) with NO plateau -- materially more real
statistical power extractable from the same 30 real days, just sliced
finer. Does roll_c's (and the other features') correlation against
next-bar return sharpen up as target_bars increases, or stay flat
regardless of resolution?

This is a real, non-cheated test: more bars from the SAME real trades is
more statistical power, not a loosened threshold. It is also not a free
lunch -- finer bars mean each one reflects less real trading volume, so
per-bar microstructure noise can rise even as aggregate sample size
does. Reports both directions honestly.

Reuses measure_real_btc_roll_c_signal.py's corr_report() function
unchanged (imported, not reimplemented) and rebuild.py/features.py's
real build_bars_and_labels()/build_enriched_events() exactly as every
other calibration script in this project does -- target_bars is the
only thing varied here, same approach calibrate_kraken_target_bars.py
took for T_effective.

Diagnostic-only: no new AFML formula, no change to any committed chapter
or pipeline module. Read-only against the real chain.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\measure_real_btc_roll_c_signal_sweep.py
    python pipeline\\diagnostics\\measure_real_btc_roll_c_signal_sweep.py --target-bars 1000 2000 3000 4000 5000
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')

sys.path.insert(0, ORCH_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rebuild import build_bars_and_labels                       # noqa: E402
from features import build_enriched_events                       # noqa: E402
from measure_real_btc_roll_c_signal import corr_report            # noqa: E402

DEFAULT_SNAPSHOT_DIR = os.path.join(
    HERE, 'kraken_snapshot_xbtusd_720h_2026-09-08'
)
DEFAULT_TARGET_BARS = [1000, 2000, 3000, 4000, 5000]

# LOAD-BEARING (2026-09-13): OUT_CSV used to be a single fixed filename
# ('btc_roll_c_signal_sweep_results.csv') regardless of --snapshot-dir, so
# running this script against a different snapshot (window2, window3, ...)
# silently overwrote the previous snapshot's results with no warning -- the
# bug flagged 2026-09-08, unresolved for three sessions. Fixed by deriving
# the default output filename from the snapshot directory's own basename,
# and adding an explicit --out-csv override for full control.
def _default_out_csv(snapshot_dir):
    snap_name = os.path.basename(os.path.normpath(snapshot_dir))
    return os.path.join(HERE, f'btc_roll_c_signal_sweep_results_{snap_name}.csv')


def run_one(raw_trades, target_bars):
    row = {'target_bars': target_bars}
    try:
        rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
        close = rebuild_result['close']
        next_ret = close.pct_change().shift(-1)
        row['n_bars'] = len(close)

        enriched_result = build_enriched_events(
            raw_trades, rebuild_result['threshold'], rebuild_result['events'],
        )
        feature_table = enriched_result['feature_table']

        rows = corr_report(feature_table, next_ret,
                            f'next bar raw return (target_bars={target_bars})')
        for col, r, n in rows:
            row[f'{col}_r'] = r
            row[f'{col}_n'] = n
        row['status'] = 'ok'
    except Exception as e:                                  # noqa: BLE001
        row['status'] = f'{type(e).__name__}: {e}'
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-dir', type=str, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument('--target-bars', type=int, nargs='+',
                         default=DEFAULT_TARGET_BARS)
    parser.add_argument('--out-csv', type=str, default=None,
                         help='Output CSV path. Defaults to a filename '
                              'derived from --snapshot-dir so different '
                              'snapshots never silently overwrite each '
                              'other\'s results.')
    args = parser.parse_args()

    out_csv = args.out_csv or _default_out_csv(args.snapshot_dir)

    raw_path = os.path.join(args.snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_path):
        raise SystemExit(f'{raw_path} not found.')

    print('=' * 78)
    print(f'REAL BTC SIGNAL vs. TARGET_BARS SWEEP: {args.snapshot_dir}')
    print(f'target_bars values: {args.target_bars}')
    print('=' * 78)

    raw_trades = pd.read_parquet(raw_path)
    print(f'\nLoaded {len(raw_trades)} raw trades')

    results = []
    for tb in args.target_bars:
        print(f'\n{"#" * 78}\n# target_bars={tb}\n{"#" * 78}')
        row = run_one(raw_trades, tb)
        if row['status'] != 'ok':
            print(f'  FAILED: {row["status"]}')
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f'\nFull results written to {out_csv}')

    print('\n' + '=' * 78)
    print('SUMMARY: roll_c correlation vs. target_bars')
    print('=' * 78)
    if 'roll_c_r' in df.columns:
        summary = df[['target_bars', 'n_bars', 'roll_c_r', 'roll_c_n']] \
            if 'n_bars' in df.columns else df[['target_bars', 'roll_c_r', 'roll_c_n']]
        print(summary.to_string(index=False))

        valid = df.dropna(subset=['roll_c_r'])
        if len(valid) >= 3:
            corr_tb_vs_r = np.corrcoef(valid['target_bars'], valid['roll_c_r'].abs())[0, 1]
            print(f'\ncorr(target_bars, |roll_c_r|) = {corr_tb_vs_r:+.4f}')
            print('(positive => correlation sharpens at finer resolution; '
                  'near zero/negative => flat or noisier, not a resolution '
                  'problem)')
    else:
        print('roll_c_r not found in any successful run -- see per-target_bars '
              'status above.')

    print('\nReminder: none of target_bars=1000\'s individual feature '
          'correlations survived a Bonferroni-corrected significance check '
          '(see measure_real_btc_roll_c_signal.py\'s 2026-09-08 run). The '
          'same correction should be applied before reading any single '
          'point in this sweep as \'real\' rather than noise.')


if __name__ == '__main__':
    main()