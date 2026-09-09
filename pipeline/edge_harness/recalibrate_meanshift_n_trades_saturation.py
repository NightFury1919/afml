"""
pipeline/edge_harness/recalibrate_meanshift_n_trades_saturation.py

SATURATION TEST (2026-09-06 session, following the reduced-feature-sweep
result): the T_effective-controlled full-14-feature sweep found DSR flat
across edge_strength 0.5-2.0 (corr=+0.0111). A follow-on reduced-4-feature
sweep, built to test whether feature dilution explained that null, showed
a real single-point improvement (roll_c isolated: OOS accuracy 0.5052 ->
0.5679 at es=2.0/seed=0) but did NOT generalize into a clean DSR-vs-
edge_strength trend once run at proper scale (15 seeds): corr(edge_strength,
dsr) dropped from +0.2811 (5-seed pilot) to +0.1072 (p=0.36, 15 seeds) --
statistically indistinguishable from the original full-feature null.

OPEN QUESTION this script's calibration feeds into: is there ANY
edge_strength at which this pipeline's real DSR/PBO chain can detect an
injected signal at all, or does it stay flat even at an absurdly strong,
impossible-to-miss signal? This is a different, more basic question than
the dilution investigation -- it asks about the FULL 14-feature pipeline
(not a reduced feature set), pushed far past the point any real trading
signal would plausibly reach, purely to find out whether a detection
ceiling exists at some finite strength or whether something is
structurally broken in the classifier/CV/DSR chain regardless of signal
strength.

  - If DSR clearly turns on at some higher edge_strength (rising cleanly,
    PBO falling), that is a REASSURING, closable result: the pipeline
    works as designed, this project's real BTC/XRP/TAO signals (and the
    synthetic ones tested so far) were simply below its real detection
    floor -- consistent with, and would finally confirm concretely, the
    project's own "T>=200 needed for meaningful discrimination" framing.
  - If DSR NEVER responds, even at edge_strength=8.0 or higher (raw
    signal correlation likely approaching ceiling), that is a much more
    serious finding -- it would point at something structurally wrong in
    the classifier/CV/DSR wiring itself, not a subtle power/dilution
    issue, and would need direct investigation (see the planned
    Idea 2 follow-on: bypass the SVC entirely with a trivial roll_c-sign
    strategy, to check whether ANY method extracts a Sharpe response at
    these signal strengths).

METHOD: same "search a small candidate grid, real pipeline call, pick the
closest T_effective match" discipline as calibrate_meanshift_n_trades.py
and recalibrate_meanshift_n_trades_high_es.py (whose real, already-
calibrated run_one()/load_calibration()/TARGET_T_EFFECTIVE are reused
UNMODIFIED here, not reimplemented) -- extended to THREE new, much
higher edge_strengths: 3.0, 5.0, 8.0. Candidate n_trades starts where the
2.0 calibration left off (500,000) and goes much higher (up to 4,000,000),
since stronger injected drift triggers denser CUSUM sampling and further
shrinks T_effective at any fixed n_trades -- the same mechanism that
required 2.0 to reach 500,000 trades where 0.5 only needed 300,000.

*** RUNTIME WARNING: candidates at the top of this grid (up to 4,000,000
trades) will likely be considerably slower per call than anything run so
far in this project (0.5-2.0's calibration topped out at 800,000-1,000,000
trades). Expect this script to take meaningfully longer than
recalibrate_meanshift_n_trades_high_es.py did. Consider running with a
subset of EDGE_STRENGTHS_TO_ADD first (e.g. just [3.0]) if wall-clock time
becomes a concern -- the script's per-edge_strength loop is independent,
so it's safe to interrupt and resume by trimming the list. ***

Writes into the SAME meanshift_n_trades_map.json used by
run_meanshift_edge_sweep.py -- merges in 3.0/5.0/8.0 entries, preserving
0.5/0.75/1.0/1.5/2.0 unchanged. The existing run_meanshift_edge_sweep.py
needs NO changes to use the extended map -- just pass a wider
--edge-strengths list.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\edge_harness\\recalibrate_meanshift_n_trades_saturation.py

    REM Then run the actual saturation sweep with the existing script:
    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py ^
        --edge-strengths 0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0 ^
        --seeds 0,1,2,3,4 ^
        --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json ^
        --output pipeline\\diagnostics\\meanshift_saturation_sweep_results.csv
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
OUTPUT_CSV = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_calibration_saturation.csv')

# New, much higher edge_strengths -- well past the original 0.5-2.0 grid,
# deliberately approaching the point where raw_signal_corr should be
# close to whatever ceiling this generator can reach at all.
EDGE_STRENGTHS_TO_ADD = [3.0, 5.0, 8.0]

# Starts where the 2.0 calibration left off (500,000) and extends well
# past the 1.5/2.0 recalibration's ceiling of 1,000,000 -- see module
# RUNTIME WARNING above before running the full grid unattended.
CANDIDATE_N_TRADES = [500_000, 800_000, 1_200_000, 1_800_000, 2_500_000, 4_000_000]


def main():
    with open(EXISTING_MAP_PATH) as f:
        n_trades_map = json.load(f)

    calib = load_calibration()
    print('Loading Ch11 driver (once, reused across all candidates)...')
    ch11 = load_ch11_driver()

    rows = []
    for es in EDGE_STRENGTHS_TO_ADD:
        print(f'\n=== CALIBRATING saturation edge_strength={es} '
              f'(new -- not in the existing map) ===')
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
                  f'not added to the map. Investigate before running the sweep '
                  f'at this edge_strength.')
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
                  f'as a clean, comparable T_effective at this edge_strength. ***')

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    with open(EXISTING_MAP_PATH, 'w') as f:
        json.dump(n_trades_map, f, indent=2)

    print(f'\nSaturation calibration grid written to {OUTPUT_CSV}')
    print(f'Updated n_trades map written to {EXISTING_MAP_PATH} '
          f'(0.5/0.75/1.0/1.5/2.0 entries preserved unchanged)')
    print('\nFull updated mapping (edge_strength -> n_trades):')
    for es, nt in sorted(n_trades_map.items(), key=lambda kv: float(kv[0])):
        print(f'  {es}: {nt:,}')
    print("""
NEXT STEP: run the saturation sweep using the EXISTING sweep script (no
new script needed -- it already accepts an arbitrary --edge-strengths
list and --n-trades-map):

    python pipeline\\edge_harness\\run_meanshift_edge_sweep.py ^
        --edge-strengths 0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0 ^
        --seeds 0,1,2,3,4 ^
        --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json ^
        --output pipeline\\diagnostics\\meanshift_saturation_sweep_results.csv

Start with 5 seeds (pilot scale, matching this project's own established
discipline) -- the reduced-feature sweep's experience this session showed
a 5-seed correlation estimate can be seriously misleading (moved from
+0.28 to +0.11 once seeds went 5->15), so treat any 5-seed saturation
result as preliminary and plan to extend seeds before drawing a real
conclusion from it, same as the reduced-feature sweep needed.

INTERPRETATION: if even edge_strength=8.0 (which should push raw_signal_
corr toward this generator's real ceiling, given raw_signal_corr climbed
smoothly from 0.155 at es=0.5 to 0.481 at es=2.0) still shows DSR flat
and statistically indistinguishable from the null, that is strong
evidence the detection ceiling is NOT a signal-strength/power problem at
all -- pointing at the classifier/CV/DSR chain itself, worth investigating
directly next (Idea 2: a trivial roll_c-sign strategy bypassing the SVC
entirely, to check whether ANY method extracts a Sharpe response at these
signal strengths).
""")


if __name__ == '__main__':
    main()
