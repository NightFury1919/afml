"""
pipeline/diagnostics/calibrate_detection_power.py

Ethan's question (2026-08-19 session): this pipeline's real live runs keep
finding "no exploitable edge" (DSR sub-0.5 on every run so far). Is that
because there genuinely is no edge, or because a real-but-small edge would
be invisible at this pipeline's actual sample size regardless? This script
answers that directly by measuring DSR's DETECTION POWER -- not "is DSR
calibrated under the null" (calibrate_min_reliable_T.py already answered
that: yes) but "if a real edge of a given size existed, how often would
DSR actually detect it, at the T/S/N this project actually runs at?"

SYNTHETIC BY DESIGN (the sanctioned use per CLAUDE.md): this needs trials
with a KNOWN true Sharpe injected by construction, which no real dataset
can give us -- same precedent as calibrate_min_reliable_T.py and
chapter_11_backtest_dangers.py's own part_b_multiple_testing().

--- Method ---
For each candidate T and each candidate true_sharpe (the size of a
genuine, real edge if one existed):
  1. Simulate N=20 trials (matching this project's real C_GRID x
     STEP_GRID = 4 x 5 = 20). ONE trial (index 0) is given a real
     population Sharpe of true_sharpe; the other N-1 are genuinely
     zero-edge, by construction -- this mirrors the real backtest-
     overfitting scenario DSR is meant to guard against (most configs
     you try are noise; at most one might be real).
  2. Compute each trial's realized Sharpe; take the best (by realized
     Sharpe, NOT by knowing which one is truly real -- same as a real
     researcher would) as sr_hat, and the spread across all 20 as
     var_sr_trials.
  3. Feed sr_hat, var_sr_trials, N=20, T into the REAL
     ch14.backtest_statistics.deflated_sharpe_ratio().
  4. Repeat 20,000 times per (T, true_sharpe) cell; report P[DSR>0.5]
     (detection power) and P[best trial selected == the real-edge trial]
     (selection accuracy, a useful secondary diagnostic).

--- Part 1 vs Part 2 ---
Part 1 uses i.i.d. standard-normal returns (skew=0, kurtosis=3 fed to
deflated_sharpe_ratio) -- the same simplification calibrate_min_reliable_T.py
used, justified there by that session's real winning trial measuring
skew=0.0386/kurtosis=3.1186 (close to Gaussian). That justification does
NOT hold anymore: this project's 2026-08-19 live runs measured real
skew as extreme as -1.3035 and kurtosis as high as 13.02 -- a materially
different, fat-tailed, negatively-skewed regime. Part 2 re-runs the same
power curve using a Bernoulli-jump return generator tuned to produce
comparable skew/kurtosis (see calibrate_jump_mixture() below), with
REALIZED skew/kurtosis recomputed per-replication from the selected best
trial's own simulated sample and fed into deflated_sharpe_ratio -- exactly
mirroring how stages.py's evaluate_overfitting() computes skew/kurtosis
from the real winning trial's bet_ret, rather than assuming Gaussian.

KNOWN SIMPLIFICATION (documented, not hidden): the jump-mixture parameters
(p_jump=0.05, jump_size=5.0) were tuned to land IN THE NEIGHBORHOOD of
2026-08-19's observed live skew/kurtosis (achieved: skew ~-1.65,
kurtosis ~7.4 vs live's observed range of skew -1.30 to +0.53, kurtosis
6.5 to 13.0) -- not an exact distributional match. The point is testing
sensitivity to "meaningfully fat-tailed and negatively skewed," not
reproducing this project's exact empirical distribution, which changes
run to run anyway.

Real observed runtime: ~1 hour on Ethan's machine (Part 2's 66 cells x
20,000 reps each, with per-rep scipy.stats.skew/kurtosis calls --
meaningfully heavier than prior calibration scripts like
calibrate_min_reliable_T.py). A pre-run sandbox estimate under-predicted
this significantly (~20-25 min) -- budget generously for anything at
this compute scale.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML\\pipeline\\diagnostics
    python calibrate_detection_power.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'ch14', 'backtest_statistics'))

from backtest_statistics import deflated_sharpe_ratio  # noqa: E402, real ch14 module

N_TRIALS = 20        # matches this project's real C_GRID x STEP_GRID = 4 x 5
N_REPS = 20_000       # matches calibrate_min_reliable_T.py's convention

# T grid: canonical values (30 = min_reliable_T threshold, 1000 = asymptotic
# reference) PLUS this project's actual observed live T_effective range as
# of 2026-08-19 (see pipeline/diagnostics/live_run_log.csv: 52.47, 55.03,
# 57.51, 63.32, 66.33, 79.38 across six real runs) -- the whole point of
# this script is answering the question AT THE T THIS PIPELINE ACTUALLY
# RUNS AT, not just in the abstract.
T_GRID = [30, 50, 55, 60, 66, 70, 80, 100, 150, 200, 1000]

# true_sharpe grid: 0.0 included as a baseline cross-check against
# calibrate_min_reliable_T.py's null-hypothesis result (should recover
# P[DSR>0.5] ~= 0.5 at every T, same as that script found).
TRUE_SHARPE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def simulate_power_gaussian(T, true_sharpe, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Part 1: i.i.d. standard-normal trials, one (index 0) given a real
    population Sharpe of true_sharpe via a mean shift (std stays 1, so
    population Sharpe = mean = true_sharpe exactly). skew=0/kurtosis=3
    fed to deflated_sharpe_ratio() -- see module docstring on why this
    simplification is being re-tested in Part 2, not just relied on."""
    rng = np.random.default_rng(seed)
    dsrs = np.empty(n_reps)
    correct_selection = np.empty(n_reps, dtype=bool)
    T_int = int(round(T))
    for i in range(n_reps):
        pnl = rng.standard_normal(size=(T_int, N))
        pnl[:, 0] += true_sharpe
        sharpes = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
        best_idx = sharpes.argmax()
        sr_hat = sharpes[best_idx]
        var_sr_trials = sharpes.var(ddof=1)
        dsrs[i] = deflated_sharpe_ratio(sr_hat, var_sr_trials, N, T, skew=0., kurtosis=3.)
        correct_selection[i] = (best_idx == 0)
    return dsrs, correct_selection


def calibrate_jump_mixture(p_jump=0.05, jump_size=5.0, n=2_000_000, seed=0):
    """One-time, large-sample estimate of the unshifted jump-mixture's
    population mean/std -- needed so simulate_power_fat_tailed() can shift
    the LOCATION (not re-standardize per-replication, which would destroy
    the sampling noise this whole script exists to measure) to hit a
    target population Sharpe while preserving the mixture's shape
    (skew/kurtosis are location-invariant)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    jumps = rng.random(n) < p_jump
    x = x.copy()
    x[jumps] -= jump_size
    return {
        'p_jump': p_jump, 'jump_size': jump_size,
        'mean0': x.mean(), 'std0': x.std(ddof=1),
        'skew0': stats.skew(x), 'kurtosis0': stats.kurtosis(x, fisher=False),
    }


def simulate_power_fat_tailed(T, true_sharpe, mix, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Part 2: same design as simulate_power_gaussian(), but every trial's
    returns are drawn from the jump-mixture calibrated above (shifted so
    the true-edge trial's POPULATION Sharpe is exactly true_sharpe, and
    every noise trial's population Sharpe is exactly 0 -- see
    calibrate_jump_mixture()'s docstring for why a location shift, not
    re-standardization, is used). skew/kurtosis fed to
    deflated_sharpe_ratio() are the REALIZED values from the SELECTED best
    trial's own T-length sample -- mirroring stages.py's real
    evaluate_overfitting() convention exactly (bet_ret.skew(),
    bet_ret.kurtosis()+3), not the fixed Gaussian assumption."""
    rng = np.random.default_rng(seed)
    dsrs = np.empty(n_reps)
    correct_selection = np.empty(n_reps, dtype=bool)
    T_int = int(round(T))
    p_jump, jump_size = mix['p_jump'], mix['jump_size']
    mean0, std0 = mix['mean0'], mix['std0']

    shift_noise = -mean0                              # population Sharpe 0
    shift_edge = true_sharpe * std0 - mean0            # population Sharpe true_sharpe

    for i in range(n_reps):
        base = rng.standard_normal(size=(T_int, N))
        jumps = rng.random(size=(T_int, N)) < p_jump
        base[jumps] -= jump_size
        base[:, 1:] += shift_noise
        base[:, 0] += shift_edge

        sharpes = base.mean(axis=0) / base.std(axis=0, ddof=1)
        best_idx = sharpes.argmax()
        sr_hat = sharpes[best_idx]
        var_sr_trials = sharpes.var(ddof=1)

        best_col = base[:, best_idx]
        if T_int > 2:
            realized_skew = float(stats.skew(best_col))
            realized_kurtosis = float(stats.kurtosis(best_col, fisher=False))
        else:
            realized_skew, realized_kurtosis = 0., 3.

        dsrs[i] = deflated_sharpe_ratio(
            sr_hat, var_sr_trials, N, T, skew=realized_skew, kurtosis=realized_kurtosis,
        )
        correct_selection[i] = (best_idx == 0)
    return dsrs, correct_selection


def main():
    print('=' * 74)
    print('PART 1 -- detection power, Gaussian returns (skew=0, kurtosis=3)')
    print('=' * 74)
    rows1 = []
    for true_sharpe in TRUE_SHARPE_GRID:
        for T in T_GRID:
            dsrs, correct = simulate_power_gaussian(T, true_sharpe)
            rows1.append({
                'true_sharpe': true_sharpe, 'T': T,
                'P[DSR>0.5]': (dsrs > 0.5).mean(),
                'P[correct_selection]': correct.mean(),
                'mean_dsr': dsrs.mean(),
            })
    df1 = pd.DataFrame(rows1)
    pivot1 = df1.pivot(index='T', columns='true_sharpe', values='P[DSR>0.5]')
    print('\n  P[DSR>0.5] (detection power) by T (rows) x true_sharpe (cols):\n')
    print(pivot1.round(4).to_string())

    print('\n' + '=' * 74)
    print('PART 2 -- detection power, fat-tailed/skewed returns (jump-mixture)')
    print('=' * 74)
    mix = calibrate_jump_mixture()
    print(f"\n  jump-mixture calibration: p_jump={mix['p_jump']}, "
          f"jump_size={mix['jump_size']}\n"
          f"  achieved (unshifted) skew={mix['skew0']:.4f}, "
          f"kurtosis={mix['kurtosis0']:.4f}\n"
          f"  (2026-08-19 live runs observed skew -1.30 to +0.53, "
          f"kurtosis 6.5 to 13.0 -- neighborhood match, not exact, "
          f"see module docstring)\n")

    print('  PART 2a -- null-hypothesis calibration check (true_sharpe=0.0 column)')
    print('  ' + '-' * 70)
    print("""
  *** REAL FINDING (2026-08-19), not a simulation bug -- verified by
  directly inspecting selected trials: unlike Part 1's Gaussian case
  (where P[DSR>0.5] stays ~0.5 at every T, matching
  calibrate_min_reliable_T.py's null-hypothesis result exactly), DSR is
  NOT well-calibrated under this fat-tailed regime -- P[DSR>0.5 | true
  edge=0] runs well ABOVE 0.5 (see the true_sharpe=0.0 column below).

  Mechanism: selecting the trial with the BEST realized Sharpe out of N
  candidates implicitly selects for trials that, by chance, avoided the
  jump/tail events in their own finite sample (confirmed directly: at
  T=66, the selected trial averaged ~1.1 jump events vs ~3.3 expected
  unconditionally). Since deflated_sharpe_ratio() is fed the SELECTED
  trial's own REALIZED skew/kurtosis (exactly matching stages.py's real
  evaluate_overfitting() convention), that realized sample systematically
  UNDERSTATES the true population tail risk -- DSR's own
  selection-bias correction is itself distorted by the very selection
  process it exists to correct for, specifically in jump-risk regimes.

  Practical read for this project: DSR is biased TOWARD false positives
  (too willing to say "yes, real edge") under conditions resembling this
  project's own live fat-tailed returns. Every real live run so far
  (2026-08-19, six runs) has STILL read below 0.5 despite this upward
  bias -- if anything this makes the "no edge" finding MORE robust, not
  less: a metric biased toward saying yes is still consistently saying
  no. See pipeline/diagnostics/live_run_log.csv for the real DSR history
  this compares against. ***
""")

    rows2 = []
    for true_sharpe in TRUE_SHARPE_GRID:
        for T in T_GRID:
            dsrs, correct = simulate_power_fat_tailed(T, true_sharpe, mix)
            rows2.append({
                'true_sharpe': true_sharpe, 'T': T,
                'P[DSR>0.5]': (dsrs > 0.5).mean(),
                'P[correct_selection]': correct.mean(),
                'mean_dsr': dsrs.mean(),
            })
    df2 = pd.DataFrame(rows2)
    pivot2 = df2.pivot(index='T', columns='true_sharpe', values='P[DSR>0.5]')
    print('  P[DSR>0.5] (detection power) by T (rows) x true_sharpe (cols):\n')
    print(pivot2.round(4).to_string())

    print(f"""
  Reading this: Part 1's true_sharpe=0.0 column reads close to 0.5 at
  every T (matches calibrate_min_reliable_T.py's null-hypothesis finding
  -- confirms this script's harness is built correctly). Part 2's
  true_sharpe=0.0 column does NOT read close to 0.5 -- see Part 2a above
  for why (real finding, not a bug) -- so Part 2's numbers must be read
  RELATIVE TO ITS OWN inflated baseline, not against the fixed 0.5 line:
  the gap between the true_sharpe=0.0 column and larger true_sharpe
  columns, at this project's real observed T range (~50-80), is what
  actually measures discriminating power in the fat-tailed regime. A
  small gap at realistic true_sharpe values (0.05-0.15, the range this
  project's own live runs' best trials have actually landed in) is the
  honest answer to Ethan's 2026-08-19 question: not "there is no edge"
  but "even a real edge this size would barely move the needle above
  fat-tailed noise, so we could not tell either way yet."
""")

    out_path = os.path.join(HERE, 'detection_power_calibration.csv')
    df1['regime'] = 'gaussian'
    df2['regime'] = 'fat_tailed'
    pd.concat([df1, df2], ignore_index=True).to_csv(out_path, index=False)
    print(f'  saved: {out_path}')


if __name__ == '__main__':
    main()