"""
diagnose_cusum_threshold.py -- ONE-OFF DIAGNOSTIC, not part of the pipeline.

Reuses real, already-tested ch02.bars.filters.cusum_filter and
ch03.labeling.triple_barrier.get_daily_vol directly against TODAY's real
staged close series (pipeline/live_staging_data/ch05_features.csv) to
compare:
  - CURRENT: fixed CUSUM_H=500 (dollar terms, price series)
  - PATH 1:  h = close.mean() * daily_vol.mean()  (dollar terms, price series)
  - PATH 2a: h = daily_vol.mean() * 1  (fractional terms, RETURNS series, Ex 3.1a)
  - PATH 2b: h = daily_vol.mean() * 2  (fractional terms, RETURNS series, Ex 5.6a)

Prints real daily_vol stats and resulting event counts for each candidate,
so the multiplier/path decision is made from real numbers, not a guess.
Delete this file after use -- it's not meant to be committed.

Run from repo root:
    python diagnose_cusum_threshold.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ch02.bars import filters as ch02_filters
from ch03.labeling import triple_barrier

FEATURES_CSV = os.path.join(ROOT, 'pipeline', 'live_staging_data', 'ch05_features.csv')

DAILY_VOL_SPAN0 = 100
CUSUM_H_CURRENT = 500


def main():
    feats = pd.read_csv(FEATURES_CSV, index_col=0, parse_dates=True)
    close = feats['close']
    print(f"Loaded {len(close)} bar closes from {FEATURES_CSV}")
    print(f"Price range: {close.min():.2f} - {close.max():.2f}, mean={close.mean():.2f}")

    daily_vol = triple_barrier.get_daily_vol(close, span0=DAILY_VOL_SPAN0)
    daily_vol = daily_vol.dropna()
    print(f"\nget_daily_vol() real stats ({len(daily_vol)} non-NaN values):")
    print(f"  mean={daily_vol.mean():.6f}  median={daily_vol.median():.6f}  "
          f"std={daily_vol.std():.6f}  min={daily_vol.min():.6f}  max={daily_vol.max():.6f}")

    def run_cusum(series, h, label):
        cusum_df = pd.DataFrame({'Date': series.index, 'Price': series.values})
        events = ch02_filters.cusum_filter(cusum_df, h=h)
        print(f"  {label:45s} h={h:.6f}  -> {len(events)} CUSUM events")
        return events

    print(f"\n=== CURRENT (price series, fixed dollar h) ===")
    run_cusum(close, CUSUM_H_CURRENT, "CUSUM_H=500 (current)")

    print(f"\n=== SCAN: looser dollar thresholds on price series (same series shape as current) ===")
    for h_scan in [400, 300, 200, 150, 100, 75, 50, 30, 20, 10]:
        run_cusum(close, h_scan, f"h={h_scan}")

    print(f"\n=== PATH 1 (price series, vol-scaled dollar h) ===")
    h_path1_1x = close.mean() * daily_vol.mean()
    h_path1_2x = close.mean() * daily_vol.mean() * 2
    run_cusum(close, h_path1_1x, "Path1 1x: mean(close)*mean(daily_vol)")
    run_cusum(close, h_path1_2x, "Path1 2x: mean(close)*mean(daily_vol)*2")

    print(f"\n=== PATH 2 (RETURNS series, fractional h -- Ex 3.1a/5.6a literal) ===")
    log_returns = np.log(close).diff().dropna()
    h_path2_1x = daily_vol.mean()       # Ex 3.1a: h = std of daily returns
    h_path2_2x = daily_vol.mean() * 2   # Ex 5.6a: h = 2x std
    run_cusum(log_returns, h_path2_1x, "Path2 1x (Ex 3.1a): h=mean(daily_vol)")
    run_cusum(log_returns, h_path2_2x, "Path2 2x (Ex 5.6a): h=2*mean(daily_vol)")


if __name__ == '__main__':
    main()
