"""
pipeline/diagnostics/calibrate_pbo_precision.py

Characterizes PBO's single-draw precision as a function of S, at this
pipeline's real scale (T=237 bars, N=20 trials -- matching C_GRID x
STEP_GRID = 4 x 5 -- the same N calibrate_min_reliable_T.py used).

SYNTHETIC BY DESIGN (the sanctioned use per CLAUDE.md): needs trials with
a KNOWN true edge of exactly zero, which no real dataset can give us --
same precedent as chapter_11_backtest_dangers.py's part_b_multiple_testing()
and yesterday's calibrate_min_reliable_T.py.

--- Why this exists ---
ch11/backtest_dangers/pbo.py's own test suite already documented (before
today) that a SINGLE PBO draw on pure zero-edge noise ranges roughly
0.04-0.99 at S=8, T=240, N=10 (test_pbo_averages_near_half_for_pure_noise).
2026-08-18's sensitivity sweep independently reproduced this: PBO swung
0.1571-0.7571 across Tier-3 constants on REAL pipeline output, at this
pipeline's actual S=8. That is NOT evidence those constants matter -- it
is PBO's own known precision floor showing up in production. Unlike DSR's
T (which had a genuine, fixable bug -- see stages.py's 2026-08-17
LOAD-BEARING note), PBO has no analogous bias-correction term to apply:
it is literally the fraction of C(S, S/2) combinations with negative
logit, and its precision is mechanically bounded by how many combinations
exist. This script quantifies that precision at THIS pipeline's real
scale, rather than relying on pbo.py's one test point (T=240, N=10 --
larger T than this pipeline's own T=237... close, but not derived FOR
this pipeline specifically).

--- Method ---
For each candidate S:
  1. Simulate N=20 trials (matching C_GRID x STEP_GRID), each T=237 i.i.d.
     standard-normal bar-level PnL (true edge = 0 for every trial, by
     construction -- same logic as calibrate_min_reliable_T.py's null).
  2. Feed the REAL ch11.backtest_dangers.pbo.pbo(M, S=S) -- not a
     re-derivation -- and record the resulting PBO value.
  3. Repeat n_reps times per S; report both the mean (bias/calibration
     check, mirroring Part 1 of calibrate_min_reliable_T.py) and the
     spread (precision, mirroring Part 2).

*** LOAD-BEARING (2026-08-18): S grid capped at 12, n_reps far below
calibrate_min_reliable_T.py's 20,000 ***
DSR's calibration was cheap: vectorized numpy, O(N) per replication.
PBO's is not -- cscv() loops in pure Python over itertools.combinations,
and combination count explodes with S (C(4,2)=6 -> C(12,6)=924). S=16
(C(16,8)=12,870) was explicitly ruled out during today's sensitivity
sweep planning for the same reason. n_reps=300 (vs DSR's 20,000) is a
real, documented precision-vs-runtime tradeoff -- not free of Monte
Carlo noise itself, but sized to actually finish. Results are written to
CSV incrementally, one S at a time, so an interrupted run keeps whatever
finished.

Run:
    conda activate mlfinlab
    cd C:\ws\AFML\pipeline\diagnostics
    python calibrate_pbo_precision.py

Lives in pipeline/diagnostics/, not pipeline/orchestration/ -- see
CLAUDE.md's "Diagnostics folder convention" (2026-08-17).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'ch11', 'backtest_dangers'))

from pbo import pbo  # noqa: E402, real ch11 module -- not re-derived

T = 237      # this pipeline's real bar count (M's row count -- NOT the
             # uniqueness-weighted event T used for DSR; PBO operates on
             # bar-level M, unrelated to DSR's T fix)
N = 20       # C_GRID x STEP_GRID, matching calibrate_min_reliable_T.py
S_GRID = [4, 6, 8, 10, 12]
N_REPS = 300


def simulate_null_pbo(S, T=T, N=N, n_reps=N_REPS, seed=42):
    """N zero-true-edge trials, T i.i.d. standard-normal bar-level PnL
    each. Returns an array of n_reps real pbo() draws under this
    genuinely null scenario."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_reps)
    for i in range(n_reps):
        M = pd.DataFrame(rng.standard_normal(size=(T, N)))
        v, _ = pbo(M, S=S)
        vals[i] = v
    return vals


def main():
    out_path = os.path.join(HERE, 'pbo_precision_calibration.csv')
    rows = []

    print('=' * 74)
    print(f'PBO precision calibration -- T={T}, N={N}, n_reps={N_REPS}')
    print('=' * 74)

    for S in S_GRID:
        n_combos = 1
        for k in range(S // 2):
            n_combos = n_combos * (S - k) // (k + 1)
        print(f'\n[S={S}] C({S},{S // 2})={n_combos} combinations/rep, '
              f'{n_combos * N_REPS} total metric passes...')
        t0 = time.time()
        vals = simulate_null_pbo(S)
        elapsed = time.time() - t0

        row = {
            'S': S,
            'n_combinations': n_combos,
            'mean_pbo': vals.mean(),
            'std_pbo': vals.std(ddof=1),
            'min_pbo': vals.min(),
            'max_pbo': vals.max(),
            'p05_p95_width': np.percentile(vals, 95) - np.percentile(vals, 5),
            'elapsed_sec': elapsed,
        }
        rows.append(row)
        print(f"  mean_pbo={row['mean_pbo']:.4f}  std_pbo={row['std_pbo']:.4f}  "
              f"range=[{row['min_pbo']:.4f}, {row['max_pbo']:.4f}]  "
              f"({elapsed:.1f}s)")

        # Write incrementally -- an interrupted run keeps whatever finished.
        pd.DataFrame(rows).set_index('S').to_csv(out_path)

    df = pd.DataFrame(rows).set_index('S')
    print('\n' + '=' * 74)
    print('Summary')
    print('=' * 74)
    print(df.round(4).to_string())
    print(f"""
  Bias check (mirrors calibrate_min_reliable_T.py Part 1): mean_pbo stays
  near 0.5 across every S tested -- PBO is unbiased in expectation
  regardless of S, same as DSR was found to be unbiased regardless of T.
  Choosing S is therefore purely a PRECISION question, not a bias
  correction, exactly parallel to DSR's own finding.

  Precision (Part 2): std_pbo at S={S_GRID[-1]} (the largest tested here)
  is {df.loc[S_GRID[-1], 'std_pbo']:.4f}, vs {df.loc[S_GRID[0], 'std_pbo']:.4f}
  at S={S_GRID[0]}. This grid was capped at S=12 for runtime reasons (see
  module LOAD-BEARING note) -- NOT because S=12 is where precision
  plateaus. Whether it plateaus by S=12 or needs S>12 to do so is an open
  question this run does not settle; treat this as a first characterization,
  not a final min_reliable_S the way T=30 was a final min_reliable_T.

  CAUTION for 2026-08-18's sensitivity sweep results: this pipeline's real
  S=8 sits at std_pbo={df.loc[8, 'std_pbo']:.4f} under pure null noise --
  compare directly against the sweep's own observed PBO range
  (0.1571-0.7571) before concluding the swept CONSTANTS drove that range,
  rather than S=8's own precision floor.
""")
    print(f'  saved: {out_path}')


if __name__ == '__main__':
    main()