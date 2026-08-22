"""
pipeline/diagnostics/calibrate_pbo_precision_s14.py

Extends calibrate_pbo_precision.py's S grid one point further: S=14,
closing part of the "PBO precision past S=12" open item (deferred since
2026-08-18, that script's own S_GRID cap).

Runs ONLY the new S=14 point -- does NOT re-run S=4..12, which are
already real-machine-confirmed in pbo_precision_calibration.csv. Appends
one new row to that same CSV, matching its established incremental-write
convention, rather than creating a separate output file.

*** LOAD-BEARING (2026-08-21): n_reps=100 at S=14, not 300 ***
C(14,7)=3,432 vs C(12,6)=924 -- a 3.71x combinatorial jump. The real
recorded S=12 runtime (386 sec at n_reps=300, from this project's own
2026-08-18 run) extrapolates to ~24 minutes for S=14 at the SAME n_reps --
too long for this session's available time. n_reps=100 (1/3 of the
original) targets ~8 minutes instead, at the cost of wider Monte Carlo
noise on the reported std_pbo/percentile estimates themselves -- same
documented tradeoff calibrate_pbo_precision.py's own module docstring
already makes for n_reps=300 vs DSR's 20,000. Treat this S=14 point as
noisier than the S=4..12 points it's being compared against, not as a
same-precision extension of that grid.

S=16 (C(16,8)=12,870) remains explicitly out of scope, per
calibrate_pbo_precision.py's own module docstring ("explicitly ruled out
during today's sensitivity sweep planning for the same reason").

Reuses ch11.backtest_dangers.pbo.pbo() directly -- no reimplementation.
Same T=237, N=20 as the original script (this pipeline's real bar-level
trial-grid shape).

Run:
    conda activate mlfinlab
    cd C:\ws\AFML\pipeline\diagnostics
    python calibrate_pbo_precision_s14.py
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

T = 237
N = 20
S = 14
N_REPS = 100  # see module LOAD-BEARING note -- 1/3 of the original 300,
              # sized to fit this session's remaining time, not a claim
              # of equal precision to the S=4..12 points.


def simulate_null_pbo(S, T=T, N=N, n_reps=N_REPS, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_reps)
    for i in range(n_reps):
        M = pd.DataFrame(rng.standard_normal(size=(T, N)))
        v, _ = pbo(M, S=S)
        vals[i] = v
    return vals


def main():
    out_path = os.path.join(HERE, 'pbo_precision_calibration.csv')
    if not os.path.exists(out_path):
        raise SystemExit(
            f'{out_path} not found -- run calibrate_pbo_precision.py first '
            '(S=4..12 grid) before extending it.'
        )
    existing = pd.read_csv(out_path, index_col='S')
    if S in existing.index:
        raise SystemExit(f'S={S} already present in {out_path} -- refusing '
                          'to overwrite. Delete that row manually first if '
                          'you really want to re-run it.')

    n_combos = 1
    for k in range(S // 2):
        n_combos = n_combos * (S - k) // (k + 1)
    print(f'[S={S}] C({S},{S // 2})={n_combos} combinations/rep, '
          f'{n_combos * N_REPS} total metric passes, n_reps={N_REPS} '
          f'(reduced from 300 -- see module LOAD-BEARING note)...')
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
    print(f"  mean_pbo={row['mean_pbo']:.4f}  std_pbo={row['std_pbo']:.4f}  "
          f"range=[{row['min_pbo']:.4f}, {row['max_pbo']:.4f}]  "
          f"({elapsed:.1f}s)")

    updated = pd.concat([existing, pd.DataFrame([row]).set_index('S')])
    updated.to_csv(out_path)
    print(f'\nAppended S=14 row to {out_path}')
    print(f"\nFull grid so far:\n{updated.round(4).to_string()}")

    print(f"""
  Compare std_pbo({S})={row['std_pbo']:.4f} against std_pbo(12)=
  {existing.loc[12, 'std_pbo']:.4f} to see whether precision is still
  improving meaningfully at S=14, or whether S=12 was already close to a
  plateau. Remember n_reps=100 here vs 300 at S<=12 -- some of any
  observed difference could be this point's own added Monte Carlo noise,
  not a real precision change. A same-n_reps re-run would be needed to
  fully separate the two if this matters later.
""")


if __name__ == '__main__':
    main()