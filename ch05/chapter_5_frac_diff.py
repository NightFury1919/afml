"""
chapter_5_frac_diff.py
=======================
Example script: Chapter 5 -- Fractionally Differentiated Features.

Runs the full toolkit (Snippets 5.1-5.4, plus our own calibration
helper) end-to-end against REAL Binance BTC/TUSD tick data, per the
project's real-data-first policy.
"""

import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CH05_ROOT = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(CH05_ROOT, 'frac_diff'))

import numpy as np
import pandas as pd

from get_weights import get_weights
from frac_diff import frac_diff
from get_weights_ffd import get_weights_ffd
from frac_diff_ffd import frac_diff_ffd
from find_min_ffd import find_min_ffd, find_minimum_d
from calibration import calibrate_ffd_thres


def load_btc_price_series(csv_path: str) -> pd.Series:
    """
    Load the raw Binance BTC/TUSD trades CSV and return a chronologically
    ordered price series.

    NOTE: this real data has 561 duplicate microsecond timestamps (out
    of 9205 trades) -- multiple trades genuinely executed in the same
    microsecond. frac_diff/frac_diff_ffd require a UNIQUE index, so we
    use a plain positional index here: trade ORDER is what matters for
    fractional differencing, not exact wall-clock time, the same way
    earlier chapters operate on bar-indexed series rather than literal
    calendar time.
    """
    cols = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp',
             'is_buyer_maker', 'is_best_match']
    df = pd.read_csv(csv_path, header=None, names=cols)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return pd.Series(df['price'].values, name='price')


def main():
    csv_path = os.path.join(root, 'input_data', 'BTCTUSD-trades-2026-03.csv')
    price = load_btc_price_series(csv_path)
    log_price = np.log(price)

    print(f"Loaded {len(price)} real BTC/TUSD trades "
          f"(March 2026, Binance).")
    print()

    # ------------------------------------------------------------------
    # Part 1: get_weights (Snippet 5.1) -- sanity check on a small size
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Part 1: get_weights -- weight decay for a few d values")
    print("=" * 70)
    for d in [0.2, 0.4, 1.0]:
        w = get_weights(d, 5).flatten()
        print(f"  d={d}: {np.round(w, 4)}")
    print()

    # ------------------------------------------------------------------
    # Part 2: frac_diff vs frac_diff_ffd -- three-way comparison
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Part 2: frac_diff (expanding) vs frac_diff_ffd (fixed-width)")
    print("=" * 70)
    d = 0.2

    expanding = frac_diff(log_price, d=d, thres=0.01)

    print(f"--- NAIVE comparison (thres=0.01 passed to BOTH, exactly "
          f"what the book's own Snippet 5.4 demo does) ---")
    fixed_naive = frac_diff_ffd(log_price, d=d, thres=0.01)
    common_naive = expanding.index.intersection(fixed_naive.index)
    corr_naive = np.corrcoef(expanding.loc[common_naive], fixed_naive.loc[common_naive])[0, 1]
    print(f"  expanding output length: {len(expanding)}")
    print(f"  fixed (naive) output length: {len(fixed_naive)}")
    print(f"  correlation on {len(common_naive)} overlapping points: {corr_naive:.4f}")
    print(f"  NOTE: thres means a RELATIVE weight-loss FRACTION in frac_diff,")
    print(f"  but an ABSOLUTE weight MAGNITUDE in frac_diff_ffd -- same number,")
    print(f"  different units. This is a real trap, not a contrived one.")
    print()

    print(f"--- CALIBRATED comparison (same effective weight-mass retention) ---")
    calibrated_thres, k = calibrate_ffd_thres(d, mass_retain=0.99, max_size=len(log_price))
    print(f"  calibrated thres = {calibrated_thres:.6f} (expected width ~{k})")
    fixed_calibrated = frac_diff_ffd(log_price, d=d, thres=calibrated_thres)
    common_cal = expanding.index.intersection(fixed_calibrated.index)
    corr_cal = np.corrcoef(expanding.loc[common_cal], fixed_calibrated.loc[common_cal])[0, 1]
    print(f"  fixed (calibrated) output length: {len(fixed_calibrated)}")
    print(f"  correlation on {len(common_cal)} overlapping points: {corr_cal:.4f}")
    print(f"  NOTE: calibration does NOT make these match closely, because")
    print(f"  frac_diff's window keeps GROWING throughout the series while")
    print(f"  frac_diff_ffd's window is permanently FIXED -- these are two")
    print(f"  structurally different algorithms, not just differently-tuned")
    print(f"  versions of the same one. FFD's constant window is a deliberate")
    print(f"  design choice (consistent operator at every point), not just")
    print(f"  a faster approximation of the expanding version.")
    print()

    # ------------------------------------------------------------------
    # Part 3: find_min_ffd (Snippet 5.4) -- the actual chapter payoff
    # ------------------------------------------------------------------
    print("=" * 70)
    print("Part 3: find_min_ffd -- minimum d that achieves stationarity")
    print("=" * 70)
    results = find_min_ffd(price, thres=0.01)
    print(results.to_string(float_format=lambda x: f"{x:.6g}"))
    print()

    min_d = find_minimum_d(results)
    min_d_row = results.loc[min_d]
    print(f"Minimum d that passes the ADF test (p < 0.05): {min_d}")
    print(f"  At this d: correlation with original log price = {min_d_row['corr']:.4f}")
    print(f"  Compare to d=1.0 (full differencing): "
          f"correlation = {results.loc[1.0, 'corr']:.4f}")
    print()
    print(f"  This is the chapter's whole point: d={min_d} achieves stationarity")
    print(f"  while keeping {min_d_row['corr']:.1%} of the original series' memory,")
    print(f"  vs. full differencing (d=1.0) which only keeps "
          f"{results.loc[1.0, 'corr']:.1%}.")

    return results


if __name__ == '__main__':
    main()
