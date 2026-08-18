"""
pipeline/diagnostics/finish_ffd_sweep.py

One-off follow-up to sensitivity_scan.py's 2026-08-18 run, which crashed
partway through the FFD_THRES leg. The other 6 sweep points (baseline, S,
ROLL_WINDOW, VPIN_WINDOW) are already safely logged in
sensitivity_scan.csv -- this script ONLY finishes the FFD_THRES leg,
importing sensitivity_scan.py's now-patched functions (including the
try/except guard around FFD_THRES) so nothing gets duplicated or re-run.

Usage
-----
    conda activate mlfinlab
    cd C:\ws\AFML
    python pipeline\diagnostics\finish_ffd_sweep.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sensitivity_scan as ss   # noqa: E402 -- reuse its patched functions

snapshot_date = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
ss.SNAPSHOT_DIR = os.path.join(HERE, f'sensitivity_snapshot_{snapshot_date}')
ss.SWEEP_STAGING_DIR = os.path.join(ss.SNAPSHOT_DIR, 'sweep_staging')
ss.SWEEP_PLOTS_DIR = os.path.join(ss.SNAPSHOT_DIR, 'sweep_plots')

if not os.path.isdir(ss.SNAPSHOT_DIR):
    raise SystemExit(f'{ss.SNAPSHOT_DIR} not found.')

print(f'Loading frozen snapshot from {ss.SNAPSHOT_DIR}...')
raw_trades, rebuild_result = ss.load_snapshot(ss.SNAPSHOT_DIR)
print(f"  {len(raw_trades)} raw trades, {len(rebuild_result['bars'])} bars, "
      f"{len(rebuild_result['events'])} events (all frozen)")

for val in (1e-5, 0.05):
    print(f'\n[FFD_THRES={val}] (ROLL_WINDOW=20 VPIN_WINDOW=10 S=8)')
    try:
        eval_result, enriched_result, _ = ss.run_one_sweep_point(
            raw_trades, rebuild_result,
            ffd_thres=val, roll_window=ss.DEFAULT_ROLL_WINDOW,
            vpin_window=ss.DEFAULT_VPIN_WINDOW, S=8,
        )
        ss.log_row('FFD_THRES', val, eval_result, enriched_result)
    except ValueError as e:
        ss.log_failed_row('FFD_THRES', val, str(e))

print(f'\nDone. Appended to {ss.RESULTS_CSV}')