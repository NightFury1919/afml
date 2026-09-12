"""
pipeline/diagnostics/calibrate_n_trials_floor.py

Direct test of Option 1 from the 2026-09-11 "the evaluation setup, not
the market, is unrealistic" discussion: DSR's detection floor scales
with how many trials were tested (N). This project's live pipeline
currently uses N=20 (a hand-picked 4x5 C/stepSize grid) without that
count ever being chosen deliberately. This script holds T_effective
FIXED at a real observed value and sweeps N instead, using the exact
same parallelized Monte Carlo machinery calibrate_kraken_detection_
power.py already has (simulate_power_gaussian_mp, reused via import,
not reimplemented) -- Ch20's mp_job_list, same as that script.

*** WHY THIS MATTERS IN REAL TERMS, NOT JUST DSR UNITS ***
A DSR-crossing true Sharpe at T=197.30 (window1 tb2000's real
T_effective) means nothing on its own -- convert it through the
window's own real bars-per-day (1958 bars / 30 days = 65.3/day) via
sqrt(bars_per_day * 365) to get an ANNUALIZED Sharpe, which is what a
human can actually sanity-check against real strategies (good ones run
1-3). Earlier today, at N=20, that conversion gave an implied floor in
the high 30s to high 40s -- nothing in real markets clears that. This
script's whole point is checking whether shrinking N moves that number
toward something real, or barely moves it at all.

Uses the GAUSSIAN regime only (not the fat-tailed jump-mixture) --
earlier real output showed the two regimes landing close together, and
the fat-tailed mixture requires re-calibrating skew/kurtosis per
window/target_bars, which isn't the point of this particular question.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_n_trials_floor.py
    python pipeline\\diagnostics\\calibrate_n_trials_floor.py --t-effective 618.83 --bars-per-day 159.5
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from calibrate_kraken_detection_power import simulate_power_gaussian_mp  # noqa: E402

N_GRID = [4, 6, 8, 10, 12, 16, 20]   # 20 = current live pipeline's real value
# *** FIXED (2026-09-11): resolution gap between 0.30 and 0.50 masked
# real N-vs-N differences ***
# First run showed mean_dsr at true_sharpe=0.30 genuinely differs by N
# (N=4: 0.9378, N=20: 0.9020) -- both real crossings fall somewhere in
# that gap, so the old sparse grid reported the same coarse "0.50" for
# every N, hiding the actual differentiation this script exists to
# measure. Added points through the 0.20-0.50 range specifically.
# *** FIXED (2026-09-11, second occurrence) ***
# Same resolution-gap problem as the T=197.3 run: at T=4004.48, mean_dsr
# jumps from ~0.78-0.87 at true_sharpe=0.05 to ~0.99+ at 0.10, and every
# N's real crossing falls somewhere in that gap -- masking a real,
# fairly large N-dependence visible at 0.05 (N=4: 0.865, N=20: 0.776)
# that the coarse grid couldn't resolve into an actual crossing point.
TRUE_SHARPE_GRID = [0.00, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                     0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
DSR_THRESHOLD = 0.95

# Real observed default: window1 tb2000 (T_effective=197.30, 1958 bars /
# 30 days = 65.27 bars/day) -- a real, moderate-resolution live
# observation, not a synthetic round number.
DEFAULT_T_EFFECTIVE = 197.30
DEFAULT_BARS_PER_DAY = 65.27


def _first_crossing(true_sharpe_grid, mean_dsr_row, threshold=DSR_THRESHOLD):
    """First true_sharpe grid point where mean_dsr >= threshold, or None
    if the grid never reaches it. Deliberately NOT interpolated -- the
    grid is sparse enough that interpolation would imply false
    precision; report the real tested points instead."""
    for ts, dsr in zip(true_sharpe_grid, mean_dsr_row):
        if dsr >= threshold:
            return ts
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-effective', type=float, default=DEFAULT_T_EFFECTIVE,
                         help=f'Fixed T_effective to test across (default '
                              f'{DEFAULT_T_EFFECTIVE}, window1 tb2000 real value).')
    parser.add_argument('--bars-per-day', type=float, default=DEFAULT_BARS_PER_DAY,
                         help=f'Real bars/day for the annualization conversion '
                              f'(default {DEFAULT_BARS_PER_DAY}, window1 tb2000).')
    parser.add_argument('--num-threads', type=int, default=4)
    args = parser.parse_args()

    T = args.t_effective
    annualize_factor = np.sqrt(args.bars_per_day * 365)
    print(f'Fixed T_effective={T}, bars_per_day={args.bars_per_day} '
          f'(annualization factor sqrt(bars_per_day*365)={annualize_factor:.2f})')
    print(f'N_GRID={N_GRID}, num_threads={args.num_threads}\n')

    rows = []
    grid = pd.DataFrame(index=N_GRID, columns=TRUE_SHARPE_GRID, dtype=float)
    for N in N_GRID:
        for true_sharpe in TRUE_SHARPE_GRID:
            dsrs, _ = simulate_power_gaussian_mp(
                T, true_sharpe, N=N, num_threads=args.num_threads,
            )
            grid.loc[N, true_sharpe] = dsrs.mean()
        crossing = _first_crossing(TRUE_SHARPE_GRID, grid.loc[N].values)
        implied_annual_sharpe = crossing * annualize_factor if crossing is not None else None
        rows.append({
            'N': N,
            'min_true_sharpe_for_dsr_0.95': crossing,
            'implied_annualized_sharpe': implied_annual_sharpe,
        })
        crossing_str = f'{crossing:.2f}' if crossing is not None else '>1.00 (not reached)'
        annual_str = f'{implied_annual_sharpe:.1f}' if implied_annual_sharpe is not None else 'N/A'
        print(f'  N={N:2d}: min true_sharpe for DSR>=0.95 = {crossing_str:>18s}  '
              f'-> implied annualized Sharpe = {annual_str}')

    print('\n' + '=' * 74)
    print(f'mean_dsr grid (rows=N, cols=true_sharpe), T_effective={T}')
    print('=' * 74)
    print(grid.round(4).to_string())

    summary = pd.DataFrame(rows)
    print('\n' + '=' * 74)
    print('SUMMARY')
    print('=' * 74)
    print(summary.to_string(index=False))
    print(f'\nCurrent live pipeline uses N=20. Real strategies typically run '
          f'annualized Sharpe 1-3. If N=4-8\'s implied floor is still far above '
          f'that range, reducing N alone does not fix the realism problem -- '
          f'it would need combining with T_effective pooling (Option 2) to '
          f'close the gap. If N=4-8 lands close to 1-3, trial-count reduction '
          f'alone is a large, cheap part of the fix.')

    out_csv = os.path.join(HERE, 'n_trials_floor_calibration.csv')
    summary.to_csv(out_csv, index=False)
    grid.to_csv(os.path.join(HERE, 'n_trials_floor_full_grid.csv'))
    print(f'\nWritten to {out_csv} and n_trials_floor_full_grid.csv')


if __name__ == '__main__':
    main()
