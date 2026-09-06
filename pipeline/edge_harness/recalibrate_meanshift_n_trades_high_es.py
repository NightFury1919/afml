"""
pipeline/edge_harness/recalibrate_meanshift_n_trades_high_es.py

Follow-on to calibrate_meanshift_n_trades.py's real run (2026-08-27):
edge_strength=1.5 and 2.0 did NOT converge to the T_effective=110 target
even at the largest candidate tried (400,000 trades) -- they landed at
72.03 and 81.88 respectively, while 0.5/0.75/1.0 all converged cleanly
(108-116). Rather than accept a partial fix (which would leave a
smaller, but real, version of the same T_effective-vs-edge_strength
confound the original calibration was built to eliminate), this script
extends the candidate grid higher for JUST those two edge_strengths --
no need to re-run 0.5/0.75/1.0, which already converged.

Reuses run_one() and load_calibration() from calibrate_meanshift_n_trades.py
UNMODIFIED (imported, not reimplemented) -- only the edge_strength list and
candidate grid change.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\recalibrate_meanshift_n_trades_high_es.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
DIAGNOSTICS_DIR = os.path.join(PIPELINE_DIR, 'diagnostics')
sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)

from calibrate_meanshift_n_trades import (  # noqa: E402
    load_calibration, run_one, TARGET_T_EFFECTIVE,
)
from stages import load_ch11_driver  # noqa: E402

EXISTING_MAP_PATH = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_map.json')
OUTPUT_CSV = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_calibration_high_es.csv')

# Only the two that didn't converge. Extends well past the original grid's
# 400,000 ceiling -- if these still don't reach anywhere near 110, that
# itself is a real finding worth knowing (see interpretation note at the
# bottom of main()).
EDGE_STRENGTHS_TO_REDO = [1.5, 2.0]
CANDIDATE_N_TRADES = [500_000, 650_000, 800_000, 1_000_000]


def main():
    with open(EXISTING_MAP_PATH) as f:
        n_trades_map = json.load(f)

    calib = load_calibration()
    print('Loading Ch11 driver (once, reused across all candidates)...')
    ch11 = load_ch11_driver()

    rows = []
    for es in EDGE_STRENGTHS_TO_REDO:
        print(f'\n=== RE-CALIBRATING edge_strength={es} (previous best: '
              f'n_trades={n_trades_map.get(str(es), "?")}) ===')
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
            print(f'  WARNING: no valid candidates -- keeping previous choice.')
            continue
        best = min(valid, key=lambda c: abs(c['T_effective'] - TARGET_T_EFFECTIVE))
        gap = abs(best['T_effective'] - TARGET_T_EFFECTIVE)
        n_trades_map[str(es)] = best['n_trades']
        print(f'  -> chosen n_trades={best["n_trades"]:,} '
              f'(T_eff={best["T_effective"]:.2f}, target={TARGET_T_EFFECTIVE}, '
              f'gap={gap:.1f})')
        if gap > 20:
            print(f'  *** STILL {gap:.1f} away from target even at the extended '
                  f'grid -- see interpretation note below before trusting this '
                  f'as a clean fix. ***')

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    with open(EXISTING_MAP_PATH, 'w') as f:
        json.dump(n_trades_map, f, indent=2)

    print(f'\nExtended-grid results written to {OUTPUT_CSV}')
    print(f'Updated n_trades map written to {EXISTING_MAP_PATH} '
          f'(0.5/0.75/1.0 entries preserved unchanged)')
    print('\nFull updated mapping (edge_strength -> n_trades):')
    for es, nt in sorted(n_trades_map.items(), key=lambda kv: float(kv[0])):
        print(f'  {es}: {nt:,}')
    print("""
INTERPRETATION: if 1.5/2.0 STILL can't reach anywhere near T_effective=110
even at up to 1,000,000 trades, that is itself informative -- it would mean
a strong enough injected signal genuinely can't be run through this bar-
construction chain at a stable, comparable sample size, which is a real
structural limit worth documenting rather than chasing further. In that
case, proceed with whatever n_trades got closest, but treat the es=1.5/2.0
points in the eventual sweep as having a materially different (lower)
T_effective than es=0.5-1.0, and read their DSR values with that caveat
explicitly in mind rather than comparing them directly.
""")


if __name__ == '__main__':
    main()
