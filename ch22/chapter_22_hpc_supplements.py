"""
AFML Chapter 22 supplements -- driver script.

Chapter 22 (High-Performance Computational Intelligence and Forecasting
Technologies) has no printed code snippets -- see ../README.md for why
this chapter's core deliverable is a README, not an implementation. This
script runs the two OPTIONAL teaching supplements built to illustrate two
of the chapter's real ideas on our own real data:

  Part 1: I/O format benchmark (echoes Section 22.6.4's HDF5-vs-ASCII
          21x-speedup story -- CSV vs. Parquet, single machine)
  Part 2: Non-uniform Fourier transform (Section 22.6.6, applied to our
          real irregularly-timestamped BTC/TUSD trades)

Path convention: this script derives its own root via __file__ so it
works for anyone who clones the repo, regardless of OS or username.
"""
import os
import sys
import tempfile

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'ch22', 'hpc_supplements'))

import numpy as np  # noqa: E402

from io_benchmark import (  # noqa: E402
    benchmark_column_subset_read,
    benchmark_write_read,
    load_real_trades,
    replicate_data,
)
from nufft_analysis import power_spectrum, price_return_series, trade_size_series  # noqa: E402


def part1_io_benchmark():
    print('=' * 70)
    print('PART 1: I/O format benchmark (echoes Section 22.6.4)')
    print('=' * 70)
    csv_path = os.path.join(root, 'input_data', 'BTCTUSD-trades-2026-03.csv')
    trades = load_real_trades(csv_path)
    print(f'Real BTC/TUSD trades: {len(trades)} rows')

    # The real dataset (9,205 rows) is too small on its own to show a
    # measurable format difference -- replicate, matching the book's own
    # approach in Section 22.6.4 ("we replicated the data 10 times").
    n_replicas = 200
    big = replicate_data(trades[['trade_id', 'price', 'qty', 'quote_qty', 'time_us']], n_replicas)
    print(f'Replicated {n_replicas}x for a measurable benchmark: {len(big):,} rows')

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = benchmark_write_read(big, tmp_dir, n_repeats=3)
        print('\nWrite/read timings (median of 3 runs):')
        for fmt in ('csv', 'parquet'):
            r = result[fmt]
            print(f"  {fmt:8s}: write={r['write_s']:.4f}s  read={r['read_s']:.4f}s  "
                  f"size={r['size_bytes'] / 1e6:.2f} MB")
        read_speedup = result['csv']['read_s'] / result['parquet']['read_s']
        size_ratio = result['csv']['size_bytes'] / result['parquet']['size_bytes']
        print(f'\n  Parquet read speedup vs CSV: {read_speedup:.2f}x')
        print(f'  Parquet file size vs CSV: {size_ratio:.2f}x smaller')

        col_result = benchmark_column_subset_read(big, tmp_dir, column='price', n_repeats=3)
        col_speedup = col_result['csv_column_read_s'] / col_result['parquet_column_read_s']
        print(f"\n  Single-column ('price') read: CSV={col_result['csv_column_read_s']:.4f}s, "
              f"Parquet={col_result['parquet_column_read_s']:.4f}s -> {col_speedup:.2f}x speedup")
        print('  (This is the closest single-machine analogy to the book\'s own HDF5-indexing')
        print('   result: a columnar format can skip data you didn\'t ask for; a row-based')
        print('   text format has to scan every row regardless.)')
    print()
    return result, col_result


def part2_nufft(freq_max_per_day=4.0, n_freqs=400):
    print('=' * 70)
    print('PART 2: Non-uniform Fourier transform (Section 22.6.6)')
    print('=' * 70)
    csv_path = os.path.join(root, 'input_data', 'BTCTUSD-trades-2026-03.csv')
    trades = load_real_trades(csv_path)
    span_days = (trades['ts'].max() - trades['ts'].min()).total_seconds() / 86400
    print(f'Real trade data span: {span_days:.1f} days, {len(trades)} trades '
          f'(~{len(trades) / span_days:.0f} trades/day average)')
    print('NOTE: this is far sparser than the book\'s own year-long, higher-frequency futures')
    print('data -- a once-per-minute search (like the book\'s TWAP finding) is not realistically')
    print('resolvable here. Searching daily/sub-daily frequencies instead.')

    freqs = np.linspace(0.05, freq_max_per_day, n_freqs)

    times_days, log_returns = price_return_series(trades)
    return_spec = power_spectrum(times_days, log_returns, freqs)
    peak_idx = np.argmax(return_spec['magnitude'])
    print(f'\nPrice-return spectrum: peak at {freqs[peak_idx]:.3f} cycles/day '
          f'(magnitude {return_spec["magnitude"][peak_idx]:.4f})')

    times_days2, qty = trade_size_series(trades)
    qty_centered = qty - qty.mean()  # remove DC component so the spectrum isn't dominated by it
    qty_spec = power_spectrum(times_days2, qty_centered, freqs)
    peak_idx2 = np.argmax(qty_spec['magnitude'])
    print(f'Trade-size spectrum:   peak at {freqs[peak_idx2]:.3f} cycles/day '
          f'(magnitude {qty_spec["magnitude"][peak_idx2]:.4f})')

    # Honest check: is either peak actually a standout, or is the spectrum
    # basically flat (no real periodic structure at this trade density)?
    return_flatness = return_spec['magnitude'].std() / return_spec['magnitude'].mean()
    qty_flatness = qty_spec['magnitude'].std() / qty_spec['magnitude'].mean()
    print(f'\nSpectral flatness (std/mean of magnitude; low = flat/no standout peak):')
    print(f'  price returns: {return_flatness:.3f}')
    print(f'  trade size:    {qty_flatness:.3f}')
    print()
    return freqs, return_spec, qty_spec


if __name__ == '__main__':
    part1_io_benchmark()
    part2_nufft()
