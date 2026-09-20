"""
pipeline/diagnostics/positive_control_confidence_threshold.py

Follow-on to Option 4's confidence-exposure prototype (2026-09-11 session):
that work calibrated zero_exposure_threshold=0.7235 entirely against NULL
data (25 real confirmed-null (dsr, pbo) pairs). Nobody had checked what a
GENUINE edge's confidence score looks like, at a realistic strength, on
this project's real T_effective values -- this script is that check.

*** UNIT CATCH (2026-09-13), found while building this script ***
A naive first attempt set true_sharpe=1,2,3 directly in
simulate_power_gaussian()'s units and got confidence=1.0 in every single
replicate (DSR=1.0, PBO=0.0, zero variance). That is NOT evidence a
realistic edge trivially clears the threshold -- true_sharpe in that
function is a raw per-replicate mean-shift statistic, not an annualized
Sharpe ratio. calibrate_n_trials_floor.py already established the real
conversion: implied_annualized_sharpe = true_sharpe_sim *
sqrt(bars_per_day * 365). Inverting it here (true_sharpe_sim =
target_annual_sharpe / sqrt(bars_per_day * 365)) is the ONLY way to test
a realistic annualized Sharpe of 1-3 -- the range real strategies
actually run, per that same script's own framing. Skipping this
conversion would silently retest the same "detection floor" question in
DSR units alone; the whole point here is PBO's contribution too.

*** WHY THIS NEEDS A REAL, NOT PROXY, PBO ***
The existing detection-power scripts (calibrate_kraken_detection_power.py,
calibrate_n_trials_floor.py) only ever computed DSR on synthetic data --
PBO was never part of that machinery, because those scripts predate
Option 4's confidence = DSR*(1-PBO) formula. This script generates the
exact same (T, N) synthetic PnL matrix those scripts already validate,
but ALSO runs it through ch11's real pbo() (S=12, matching this
project's own established LOAD-BEARING precision choice) -- giving a
genuine (dsr, pbo) pair from an actual injected edge, not an assumed or
estimated one.

Reuses real, unmodified module code throughout:
  - ch11/backtest_dangers/pbo.py's real pbo() (S=12)
  - ch14/backtest_statistics/backtest_statistics.py's real
    deflated_sharpe_ratio()
The Gaussian PnL generator below mirrors calibrate_kraken_detection_
power.py's own simulate_power_gaussian() (same generative model, same
mean-shift-in-trial-0 mechanism) but is NOT imported from that file --
that file currently has an unreviewed, uncommitted parallelization
change sitting in it (see project handoff, 2026-09-13), and this script
should not take on a dependency on code still under review. The ~10
lines duplicated here are the well-established, already-committed part
of that file, not the new part.

*** n_reps=300 CHOICE (2026-09-13) ***
Measured directly, not guessed: a real 60-replicate pilot run at T=197.3
gave std(confidence)=0.14; a 40-replicate pilot at T=4004.48 gave
std(confidence)=0.155. At n_reps=300, SE on mean confidence is
~0.009 (0.155/sqrt(300)) -- tight enough to distinguish a real crossing
from noise. Runtime at ~0.6-0.9s/rep (pbo()'s S=12 CSCV -- C(12,6)=924
combinations per replicate -- dominates cost, not T's row count) gives
an estimated 20-25 minutes total across 3 annual_sharpe values x 2
T-points x 300 reps. Budget generously, same caution as every other
Monte Carlo calibration in this project.

T grid: this project's two real, already-established T_effective
values -- window1 tb2000 (single-window, T=197.30, bars_per_day=65.27)
and the 8-window pooled series (T=4004.48, bars_per_day=37884/240 real
bars over real days, not a rounded estimate).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\positive_control_confidence_threshold.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'ch11', 'backtest_dangers'))
sys.path.insert(0, os.path.join(ROOT, 'ch14', 'backtest_statistics'))

from pbo import pbo                                       # noqa: E402, real ch11 module
from backtest_statistics import deflated_sharpe_ratio       # noqa: E402, real ch14 module

N_TRIALS = 20
S_PBO = 12
N_REPS = 300
ANNUAL_SHARPE_GRID = [1.0, 2.0, 3.0]

THRESHOLD_FILE = os.path.join(HERE, 'confidence_threshold.txt')
FALLBACK_ZERO_EXPOSURE_THRESHOLD = 0.7235

# Real, already-established T_effective values from this project's own
# sweeps -- not synthetic round numbers.
T_GRID = [
    {'label': 'single_window_tb2000', 'T_effective': 197.30, 'bars_per_day': 65.27},
    {'label': 'pooled_8window_tb40000', 'T_effective': 4004.48,
     'bars_per_day': 37884.0 / 240.0},  # real n_bars / real total_days, exact
]


def _load_zero_exposure_threshold():
    """Reads the SAME threshold file log_live_prediction.py reads, so this
    positive control always tests against whatever threshold is actually
    live -- not a hardcoded copy that can silently drift if the threshold
    is ever recalibrated."""
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return float(f.read().strip())
    return FALLBACK_ZERO_EXPOSURE_THRESHOLD


def simulate_gaussian_matrix(T, true_sharpe, N, seed):
    """Mirrors calibrate_kraken_detection_power.py's simulate_power_
    gaussian() generative step exactly (i.i.d. standard-normal trials,
    trial 0 given a real population Sharpe via a mean shift) -- see
    module docstring for why this is duplicated rather than imported.
    Returns the full (T, N) PnL matrix, not just its Sharpe ratios,
    because this script needs the matrix itself for pbo()."""
    rng = np.random.default_rng(seed)
    T_int = int(round(T))
    pnl = rng.standard_normal(size=(T_int, N))
    pnl[:, 0] += true_sharpe
    return pnl


def one_replicate(T, true_sharpe, N=N_TRIALS, S=S_PBO, seed=0):
    """One real (dsr, pbo) pair from one injected-edge synthetic trial
    matrix. DSR side: identical math to simulate_power_gaussian(). PBO
    side: the SAME matrix, fed to ch11's real, unmodified pbo()."""
    pnl = simulate_gaussian_matrix(T, true_sharpe, N, seed)
    sharpes = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
    best_idx = sharpes.argmax()
    sr_hat = sharpes[best_idx]
    var_sr_trials = sharpes.var(ddof=1)
    dsr = deflated_sharpe_ratio(sr_hat, var_sr_trials, N, T, skew=0., kurtosis=3.)
    pbo_val, _ = pbo(pd.DataFrame(pnl), S=S)
    return dsr, pbo_val


def main():
    zero_exposure_threshold = _load_zero_exposure_threshold()
    print('=' * 78)
    print('POSITIVE CONTROL: does a REALISTIC injected edge clear the '
          'confidence threshold?')
    print(f'zero_exposure_threshold = {zero_exposure_threshold} '
          f'(read from {THRESHOLD_FILE if os.path.exists(THRESHOLD_FILE) else "fallback constant"})')
    print(f'N_TRIALS={N_TRIALS}, S_PBO={S_PBO}, N_REPS={N_REPS}, '
          f'ANNUAL_SHARPE_GRID={ANNUAL_SHARPE_GRID}')
    print('=' * 78)

    rows = []
    for t_point in T_GRID:
        T = t_point['T_effective']
        bars_per_day = t_point['bars_per_day']
        annualize_factor = np.sqrt(bars_per_day * 365)
        print(f"\n--- {t_point['label']}: T_effective={T}, "
              f"bars_per_day={bars_per_day:.2f}, "
              f"annualize_factor={annualize_factor:.2f} ---")

        for annual_sharpe in ANNUAL_SHARPE_GRID:
            true_sharpe = annual_sharpe / annualize_factor
            dsrs = np.empty(N_REPS)
            pbos = np.empty(N_REPS)
            seed_base = hash((t_point['label'], annual_sharpe)) % (2**31)
            for i in range(N_REPS):
                dsrs[i], pbos[i] = one_replicate(T, true_sharpe, seed=seed_base + i)
            confidence = dsrs * (1 - pbos)
            frac_above = float((confidence > zero_exposure_threshold).mean())

            row = {
                'window_label': t_point['label'],
                'T_effective': T,
                'bars_per_day': bars_per_day,
                'annual_sharpe': annual_sharpe,
                'sim_true_sharpe': true_sharpe,
                'mean_dsr': dsrs.mean(),
                'mean_pbo': pbos.mean(),
                'mean_confidence': confidence.mean(),
                'std_confidence': confidence.std(ddof=1),
                'frac_above_threshold': frac_above,
                'zero_exposure_threshold': zero_exposure_threshold,
                'n_reps': N_REPS,
            }
            rows.append(row)
            print(f'  annual_sharpe={annual_sharpe:.1f} (sim true_sharpe={true_sharpe:.5f}): '
                  f'mean_dsr={row["mean_dsr"]:.4f}  mean_pbo={row["mean_pbo"]:.4f}  '
                  f'mean_confidence={row["mean_confidence"]:.4f} '
                  f'(std={row["std_confidence"]:.4f})  '
                  f'frac_above_threshold={frac_above:.3f}')

    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, 'positive_control_confidence_results.csv')
    df.to_csv(out_csv, index=False)

    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(df[['window_label', 'annual_sharpe', 'mean_dsr', 'mean_pbo',
              'mean_confidence', 'frac_above_threshold']].to_string(index=False))
    print(f'\nWritten to {out_csv}')

    max_frac = df['frac_above_threshold'].max()
    if max_frac < 0.05:
        print(f'\nA realistic injected edge (annualized Sharpe 1-3) crosses '
              f'the {zero_exposure_threshold} threshold in at most '
              f'{max_frac:.1%} of replicates across every T-point tested. '
              f'This threshold is not reliably reachable by a realistic '
              f'edge at these real sample sizes -- the same structural '
              f'problem the original DSR-floor diagnosis found, one level '
              f'down.')
    else:
        print(f'\nA realistic injected edge crosses the '
              f'{zero_exposure_threshold} threshold in up to {max_frac:.1%} '
              f'of replicates somewhere in this grid -- worth reading the '
              f'per-row breakdown above to see exactly where.')


if __name__ == '__main__':
    main()
