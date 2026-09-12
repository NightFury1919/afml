"""
pipeline/diagnostics/calibrate_confidence_threshold.py

Option 4 (2026-09-11): replace the binary DSR>=0.95 strategy-level gate
with a continuous exposure scalar. Needs a "zero exposure" anchor point
for the combined confidence score DSR*(1-PBO) -- and rather than assume
one (DSR and PBO each individually center ~0.5 under null, but their
PRODUCT's null distribution is a different, real shape, not just 0.25),
this calibrates it EMPIRICALLY from real data already on disk.

Every (DSR, PBO) pair in today's sweep CSVs -- window1-3, Binance, and
the 8-window pool -- comes from data the 2026-09-11 multi-window
meta-analysis already proved contains no real edge (pooled PBO
z=-0.07, p=0.95 against the calibrated null). That makes every single
row a genuine, real draw from the TRUE null distribution of this
combined score -- not a synthetic assumption, not a guess.

Zero-exposure anchor = empirical_null_mean + 2*empirical_null_std (real
noise very rarely produces a confidence score this high by chance;
"very rarely" is a real, checkable property of this actual empirical
distribution, not a borrowed convention from p<0.05-style thresholds).
Full-exposure anchor = 1.0 (the score's theoretical maximum, since
DSR<=1 and PBO>=0 by construction). Linear ramp between the two.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_confidence_threshold.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Explicit list, not a fragile glob -- same lesson as
# pool_kraken_raw_trades.py's earlier bug (a broad glob swept in an
# unrelated Aug 25 leftover file). Every real sweep CSV produced today.
DEFAULT_CSVS = [
    'kraken_target_bars_calibration_window1.csv',
    'kraken_target_bars_calibration_window2.csv',
    'kraken_target_bars_calibration_window3.csv',
    'kraken_target_bars_calibration_binance.csv',
    'kraken_target_bars_calibration_pooled.csv',
]
N_SIGMA = 2.0
OUTPUT_CSV = os.path.join(HERE, 'confidence_threshold_calibration.csv')


def main():
    csv_names = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_CSVS
    frames = []
    for name in csv_names:
        # Resolve relative to CWD, same convention every other script
        # today has used (see the earlier pool_kraken_pbo_across_windows.py
        # path bug for why this matters).
        if not os.path.exists(name):
            print(f'  SKIP: {name} not found (not run yet, or different CWD?)')
            continue
        df = pd.read_csv(name)
        if 'dsr' not in df.columns or 'pbo' not in df.columns:
            print(f'  SKIP: {name} missing dsr/pbo columns')
            continue
        df = df.dropna(subset=['dsr', 'pbo'])
        n_before = len(df)
        df = df.drop_duplicates(subset=['dsr', 'pbo'])
        if len(df) < n_before:
            print(f'    (dropped {n_before - len(df)} exact-duplicate dsr/pbo '
                  f'rows within {name} -- the known window3-CSV duplication '
                  f'issue, fixed here rather than assumed cleaned up)')
        df['source'] = os.path.basename(name)
        frames.append(df[['dsr', 'pbo', 'source']])
        print(f'  Loaded {len(df)} real (dsr, pbo) rows from {name}')

    if not frames:
        raise SystemExit('No real (dsr, pbo) data found -- run the sweeps first.')

    pooled = pd.concat(frames, ignore_index=True)
    pooled['confidence'] = pooled['dsr'] * (1 - pooled['pbo'])

    n = len(pooled)
    null_mean = pooled['confidence'].mean()
    null_std = pooled['confidence'].std(ddof=1)
    zero_exposure_threshold = null_mean + N_SIGMA * null_std
    full_exposure_threshold = 1.0

    print(f'\n{n} real (dsr, pbo) pairs loaded, all from data the 2026-09-11 '
          f'8-window meta-analysis already confirmed has no real edge -- '
          f'treating every row as a genuine empirical null draw.')
    print(f'\nconfidence = dsr * (1 - pbo):')
    print(f'  empirical null mean:   {null_mean:.4f}')
    print(f'  empirical null std:    {null_std:.4f}')
    print(f'  min / max observed:    {pooled["confidence"].min():.4f} / '
          f'{pooled["confidence"].max():.4f}')
    print(f'\n  zero-exposure threshold (mean + {N_SIGMA:.0f}*std): '
          f'{zero_exposure_threshold:.4f}')
    print(f'  full-exposure threshold (theoretical max):       '
          f'{full_exposure_threshold:.4f}')
    n_above = (pooled['confidence'] > zero_exposure_threshold).sum()
    print(f'\n  {n_above}/{n} real observed rows exceed the zero-exposure '
          f'threshold ({100*n_above/n:.1f}%) -- consistency check: if this '
          'is close to 0 (as expected for a ~2-sigma-style cut on real '
          'confirmed-null data), the threshold is behaving as intended.')

    pooled.to_csv(OUTPUT_CSV, index=False)
    with open(os.path.join(HERE, 'confidence_threshold.txt'), 'w') as f:
        f.write(f'{zero_exposure_threshold}\n')
    print(f'\nWrote {OUTPUT_CSV} and confidence_threshold.txt '
          f'(zero_exposure_threshold={zero_exposure_threshold:.6f}, '
          'read by the live exposure-sizing step next).')


if __name__ == '__main__':
    main()
