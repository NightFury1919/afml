"""
pipeline/diagnostics/calibrate_min_reliable_T.py

Derives report.py's min_reliable_T via null-hypothesis Monte Carlo, rather
than picking a number that "feels right" (the old min_reliable_T=150's
actual origin -- see report.py's own 2026-08-17 LOAD-BEARING note).

SYNTHETIC BY DESIGN (the sanctioned use per CLAUDE.md): this needs trials
with a KNOWN true Sharpe of exactly zero, which no real dataset can give
us -- same precedent as chapter_11_backtest_dangers.py's own
part_b_multiple_testing().

--- What this answers ---
DSR's own multiple-testing correction (the sqrt(T-1) term inside the PSR
formula it's built on) already dampens confidence appropriately at small
T -- a genuinely zero-edge strategy does NOT produce falsely confident DSR
readings just because T is small (verified below: P[DSR>0.5] stays ~50%
and P[DSR>0.95] stays ~0% uniformly across T=5..200). What DOES change
with T is the PRECISION of a single DSR draw -- how much it would bounce
around if you reran the same experiment with fresh random luck. That
precision curve is what min_reliable_T should actually be gating on, and
that is what this script measures.

--- Method ---
For each candidate effective T:
  1. Simulate N=20 trials (matching this project's real C_GRID x
     STEP_GRID = 4 x 5 = 20), each T i.i.d. standard-normal returns
     (true Sharpe = 0 for every trial, by construction).
  2. Compute each trial's realized Sharpe; take the best as sr_hat and the
     real spread across all 20 as var_sr_trials -- both regenerated fresh
     per replication, since DSR's own deflation term is itself part of
     what small T distorts.
  3. Feed sr_hat, var_sr_trials, N=20, T into the REAL
     ch14.backtest_statistics.deflated_sharpe_ratio() (Gaussian
     skew=0/kurtosis=3 -- this project's real winning trial measured
     skew=0.0386/kurtosis=3.1186 on 2026-08-17, close enough to Gaussian
     that a skewed-return generator isn't worth building for this).
  4. Repeat 20,000 times per T; report both the tail-threshold rates and
     the estimate's own spread (std, 5th/95th percentile width).

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML\\pipeline\\diagnostics
    python calibrate_min_reliable_T.py

Lives in pipeline/diagnostics/, not pipeline/orchestration/ -- see
CLAUDE.md's "Diagnostics folder convention" (2026-08-17): this is a
calibration/validation script, not part of the core pipeline package
orchestration/ imports at runtime.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'ch14', 'backtest_statistics'))

from backtest_statistics import deflated_sharpe_ratio  # noqa: E402, real ch14 module


def simulate_null_dsr(T, N=20, n_reps=20000, seed=42):
    """N zero-true-edge trials, T i.i.d. standard-normal returns each.
    Returns an array of n_reps DSR draws under this genuinely null
    scenario."""
    rng = np.random.default_rng(seed)
    dsrs = np.empty(n_reps)
    for i in range(n_reps):
        pnl = rng.standard_normal(size=(int(round(T)), N))
        sharpes = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
        sr_hat = sharpes.max()
        var_sr_trials = sharpes.var(ddof=1)
        dsrs[i] = deflated_sharpe_ratio(sr_hat, var_sr_trials, N, T, skew=0., kurtosis=3.)
    return dsrs


def main():
    T_grid = [5, 10, 15, 19, 20, 25, 30, 40, 50, 75, 100, 150, 200, 500, 1000]

    print('=' * 74)
    print('PART 1 -- tail-threshold false-positive rates under the null')
    print('=' * 74)
    rows = []
    for T in T_grid:
        dsrs = simulate_null_dsr(T)
        rows.append({
            'T': T,
            'P[DSR>0.5]': (dsrs > 0.5).mean(),
            'P[DSR>0.95]': (dsrs > 0.95).mean(),
            'mean_dsr': dsrs.mean(),
        })
    df1 = pd.DataFrame(rows).set_index('T')
    print(df1.round(4).to_string())
    print("""
  DSR is ALREADY well-calibrated at every T tested -- P[DSR>0.5] hovers
  near 50% and P[DSR>0.95] near 0% uniformly from T=5 to T=200. Small T
  does not produce falsely confident readings; the sqrt(T-1) deflation
  term already handles this correctly. This rules out "false positive
  rate" as the right basis for min_reliable_T -- see Part 2.
""")

    print('=' * 74)
    print('PART 2 -- estimator PRECISION (spread) under the null')
    print('=' * 74)
    asymptotic_std = simulate_null_dsr(1000, n_reps=20000, seed=42).std(ddof=1)
    rows2 = []
    for T in T_grid:
        dsrs = simulate_null_dsr(T)
        std_t = dsrs.std(ddof=1)
        rows2.append({
            'T': T,
            'std_dsr': std_t,
            'pct_of_asymptotic_spread': std_t / asymptotic_std,
            'p05_p95_width': np.percentile(dsrs, 95) - np.percentile(dsrs, 5),
        })
    df2 = pd.DataFrame(rows2).set_index('T')
    print(f'  asymptotic std_dsr (T=1000): {asymptotic_std:.4f}\n')
    print(df2.round(4).to_string())
    print(f"""
  T=30's std_dsr sits at {df2.loc[30, 'std_dsr']:.4f}, only
  {(df2.loc[30, 'pct_of_asymptotic_spread'] - 1) * 100:.1f}% above the
  T=1000 asymptotic floor -- past the steep early drop, near the
  precision floor. This is the basis for report.py's min_reliable_T=30
  (see report.py's own LOAD-BEARING note, 2026-08-17).
""")

    out_path = os.path.join(HERE, 'min_reliable_T_calibration.csv')
    df1.join(df2, rsuffix='_precision').to_csv(out_path)
    print(f'  saved: {out_path}')


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Real-machine output (mlfinlab env), 2026-08-17
#
# (mlfinlab) PS C:\ws\AFML\pipeline\orchestration> python calibrate_min_reliable_T.py
# (path was pipeline/orchestration/ at the time this run happened -- the
# file was relocated to pipeline/diagnostics/ immediately afterward, see
# CLAUDE.md's 2026-08-17 Diagnostics folder convention; ROOT-relative
# imports are unaffected since both live at the same depth under pipeline/)
#
# PART 1 -- tail-threshold false-positive rates under the null
#       P[DSR>0.5]  P[DSR>0.95]  mean_dsr
# T
# 5         0.5588       0.0000    0.5264
# 10        0.5222       0.0018    0.5155
# 15        0.5020       0.0012    0.5076
# 19        0.4966       0.0010    0.5059
# 20        0.4922       0.0011    0.5049
# 25        0.4864       0.0012    0.5023
# 30        0.4832       0.0010    0.5018
# 40        0.4754       0.0006    0.4979
# 50        0.4715       0.0010    0.4978
# 75        0.4712       0.0012    0.4964
# 100       0.4712       0.0007    0.4971
# 150       0.4726       0.0006    0.4974
# 200       0.4714       0.0004    0.4970
# 500       0.4734       0.0005    0.4973
# 1000      0.4686       0.0006    0.4961
# DSR is already well-calibrated (P[DSR>0.5]~50%, P[DSR>0.95]~0%) at
# every real T tested -- confirms Part 1's conclusion is not a sandbox
# artifact.
#
# PART 2 -- estimator PRECISION (spread) under the null
# asymptotic std_dsr (T=1000): 0.1568
#       std_dsr  pct_of_asymptotic_spread  p05_p95_width
# T
# 5      0.2040                    1.3013         0.6723
# 10     0.1762                    1.1238         0.5830
# 15     0.1702                    1.0857         0.5633
# 19     0.1658                    1.0574         0.5459
# 20     0.1663                    1.0605         0.5483
# 25     0.1625                    1.0366         0.5359
# 30     0.1622                    1.0343         0.5360  <- adopted threshold
# 40     0.1600                    1.0203         0.5300
# 50     0.1600                    1.0206         0.5285
# 75     0.1575                    1.0047         0.5196
# 100    0.1584                    1.0104         0.5282
# 150    0.1575                    1.0045         0.5199
# 200    0.1561                    0.9955         0.5189
# 500    0.1567                    0.9992         0.5159
# 1000   0.1568                    1.0000         0.5199
#
# T=30 real-machine std_dsr = 0.1622, 3.4% above the T=1000 asymptotic
# floor -- matches the sandbox preview essentially exactly. Confirms
# report.py's min_reliable_T=30.
#
# Confirmed against real data via pipeline\run_pipeline.py immediately
# after: real effective T=26.17 (< 30) -- correctly still flags the
# small-sample warning, an honest result given T sits just under the new
# threshold, not a regression from the old (uncalibrated) 150 cutoff.
# ---------------------------------------------------------------------------
