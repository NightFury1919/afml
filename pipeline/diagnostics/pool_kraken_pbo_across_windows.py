"""
pipeline/diagnostics/pool_kraken_pbo_across_windows.py

The point of capturing 8 independent windows instead of 2: single-window
findings kept looking real and then getting contradicted by the next
check (see 2026-09-10 session). With genuinely independent draws, a real
effect should show up as the POOLED DISTRIBUTION being shifted relative
to pure noise, not as a couple of low outliers among a handful of
correlated-by-construction target_bars resamplings.

Null baseline reused from the real, already-run calibration
(pbo_precision_calibration.csv, calibrate_pbo_precision.py) at S=12 --
the same S this project's evaluate_overfitting() calls use everywhere:
mean_pbo=0.4925, std_pbo=0.2051 under pure null (zero-edge) noise.
Nothing re-simulated here -- these are real numbers already on disk.

*** HONESTY NOTE ON DEGREES OF FREEDOM -- READ BEFORE TRUSTING A Z-SCORE ***
Each window contributes 5 target_bars rows, but those 5 rows are NOT
independent draws -- they're the same underlying 30-day trade stream
resampled into bars at 5 different resolutions, so they're correlated
with each other in a way plain n=40 statistics would overstate. This
script reports BOTH:
  (a) the naive pooled view across all 40 (window, target_bars) rows,
      and
  (b) the more defensible PER-WINDOW-MEAN view (8 genuinely independent
      values, one mean PBO per window, averaged across that window's 5
      correlated target_bars rows first) -- treat (b) as the real
      answer and (a) as a sanity check, not the other way around.

Run once all 8 windows' calibrate_kraken_target_bars.py sweeps exist:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\pool_kraken_pbo_across_windows.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

# Real, already-calibrated null baseline at S=12 (this project's own
# standard S) -- from pbo_precision_calibration.csv, not re-simulated.
NULL_MEAN_PBO = 0.4925252525252525
NULL_STD_PBO = 0.2051176032389558

# *** FIXED (2026-09-11): search CWD by default, not this script's own
# directory ***
# Every calibrate_kraken_target_bars.py invocation this session was
# called with a bare relative filename (e.g.
# 'kraken_target_bars_calibration_window1.csv') while the terminal's
# CWD was the repo root (C:\ws\AFML) -- so that's where the CSVs
# actually landed, not next to this script in pipeline\diagnostics\.
# The first version of this pattern used HERE (this file's own
# directory) and found nothing as a result.
DEFAULT_PATTERN = 'kraken_target_bars_calibration_window*.csv'


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATTERN
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'No files matched {pattern!r} -- run the target_bars '
                          'sweep on each window first.')

    frames = []
    for p in paths:
        window_name = os.path.basename(p).replace(
            'kraken_target_bars_calibration_', '').replace('.csv', '')
        df = pd.read_csv(p)
        df['window'] = window_name
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    print(f'Loaded {len(paths)} window CSVs, {len(pooled)} total (window, '
          f'target_bars) rows: {[os.path.basename(p) for p in paths]}')
    if len(paths) < 2:
        print('WARNING: only 1 window found -- pooling across windows needs '
              'more than one file to mean anything.')

    print('\n' + '=' * 74)
    print('(a) NAIVE POOLED VIEW -- all (window, target_bars) rows treated as iid')
    print('    (see module docstring: this OVERSTATES independence -- sanity '
          'check only)')
    print('=' * 74)
    all_pbo = pooled['pbo'].values
    print(f'  n={len(all_pbo)}, mean={all_pbo.mean():.4f}, std={all_pbo.std(ddof=1):.4f}, '
          f'min={all_pbo.min():.4f}, max={all_pbo.max():.4f}')
    naive_z = (all_pbo.mean() - NULL_MEAN_PBO) / (NULL_STD_PBO / np.sqrt(len(all_pbo)))
    print(f'  naive z-score vs null (mean={NULL_MEAN_PBO:.4f}, std={NULL_STD_PBO:.4f}): '
          f'{naive_z:+.2f}  <- treat with real suspicion, see docstring')

    print('\n' + '=' * 74)
    print('(b) PER-WINDOW-MEAN VIEW -- the real answer (8 independent values)')
    print('=' * 74)
    per_window = pooled.groupby('window')['pbo'].mean().sort_index()
    print(per_window.to_string())
    n_windows = len(per_window)
    window_mean = per_window.mean()
    window_std = per_window.std(ddof=1)
    print(f'\n  n_windows={n_windows}, mean_of_window_means={window_mean:.4f}, '
          f'std_across_windows={window_std:.4f}')

    if n_windows >= 2:
        se = NULL_STD_PBO / np.sqrt(n_windows)
        z = (window_mean - NULL_MEAN_PBO) / se
        p_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f'  z-score vs null (using the null\'s own std, since we are '
              f'testing whether these {n_windows} means come from that null '
              f'distribution): {z:+.2f}')
        print(f'  two-sided p-value: {p_two_sided:.4f}')

        # Also report a plain one-sample t-test using the OBSERVED
        # across-window std (more honest if the real windows are more/
        # less variable than pure null noise predicts -- worth comparing
        # both).
        t_stat, t_p = stats.ttest_1samp(per_window.values, NULL_MEAN_PBO)
        print(f'  (cross-check) one-sample t-test using OBSERVED across-window '
              f'std instead of the null\'s: t={t_stat:+.2f}, p={t_p:.4f}')

    print('\n' + '=' * 74)
    print('READ THIS AS')
    print('=' * 74)
    print('  A |z| comfortably below ~2 (p > 0.05) across (b) means: 8 independent '
          'windows worth of real Kraken BTC PBO readings are NOT distinguishable '
          'from pure null noise at this S. That would be a real, defensible '
          'answer to the entire 2026-09-10 investigation -- the single-window '
          'anomalies were noise, full stop.')
    print('  A |z| clearly above ~2 would be the first evidence in this whole '
          'investigation that survived a genuinely independent multi-window '
          'test -- worth taking seriously, though still only n=' + str(n_windows) +
          ' windows, all from one exchange, one continuous ~7-month stretch of '
          'calendar time (not independent draws from "all possible market '
          'regimes").')


if __name__ == '__main__':
    main()
