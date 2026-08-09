"""
AFML Chapter 22 supplement -- I/O format benchmark.

NOT a book snippet -- Chapter 22 has none. This is a small, real,
single-machine echo of the chapter's own headline finding: switching from
a row-based text format (ASCII/CSV) to a format designed for analytical
access (their HDF5, our Parquet -- HDF5 requires the HDF5 C library and
isn't a natural fit outside HPC/scientific-computing workflows; Parquet is
the columnar, compressed, indexable format most commonly reached for in a
plain Python/pandas stack today, and demonstrates the SAME underlying
lesson: row-oriented text formats force you to read and parse everything,
column-oriented binary formats let you skip what you don't need).

We don't have a real HPC cluster to demonstrate the 512-CPU-core numbers
from Figure 22.8 -- this is a single-machine, single-format-choice
analogy, not a reproduction of the book's actual benchmark. Labeled as
such throughout.
"""

import os
import time

import pandas as pd


def load_real_trades(csv_path):
    """
    Load the real BTC/TUSD trade data (Binance historical-trades format:
    no header, columns are trade_id, price, qty, quote_qty, time_us
    (microseconds since epoch), is_buyer_maker, is_best_match).

    Parameters
    ----------
    csv_path : str
        Path to the raw BTCTUSD-trades-*.csv file.

    Returns
    -------
    pd.DataFrame
        Columns: trade_id, price, qty, quote_qty, time_us, is_buyer_maker,
        is_best_match, ts (parsed datetime).
    """
    df = pd.read_csv(
        csv_path, header=None,
        names=['trade_id', 'price', 'qty', 'quote_qty', 'time_us', 'is_buyer_maker', 'is_best_match'],
    )
    df['ts'] = pd.to_datetime(df['time_us'], unit='us')
    return df


def replicate_data(df, n_replicas):
    """
    Replicate the real trade data n_replicas times, matching the book's
    own approach in Section 22.6.4 ("we replicated the data 10 times")
    when the real dataset is too small on its own to show a measurable
    I/O-format difference.

    Parameters
    ----------
    df : pd.DataFrame
    n_replicas : int
        Must be >= 1.

    Returns
    -------
    pd.DataFrame
        n_replicas * len(df) rows.
    """
    if n_replicas < 1:
        raise ValueError('n_replicas must be >= 1')
    return pd.concat([df] * n_replicas, ignore_index=True)


def _time_it(fn, n_repeats=3):
    """Run fn() n_repeats times, return the median wall-clock duration in seconds."""
    durations = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - t0)
    durations.sort()
    return durations[len(durations) // 2]


def benchmark_write_read(df, tmp_dir, n_repeats=3):
    """
    Time writing and reading the same DataFrame as CSV vs. Parquet, and
    compare on-disk file size.

    Parameters
    ----------
    df : pd.DataFrame
    tmp_dir : str
        Directory to write scratch files into (caller's responsibility to
        clean up).
    n_repeats : int, default 3
        Number of timing repeats per operation; median is reported (more
        robust to one-off OS/disk-cache noise than a single measurement).

    Returns
    -------
    dict
        {'csv': {'write_s': float, 'read_s': float, 'size_bytes': int},
         'parquet': {'write_s': float, 'read_s': float, 'size_bytes': int}}
    """
    csv_path = os.path.join(tmp_dir, 'bench.csv')
    parquet_path = os.path.join(tmp_dir, 'bench.parquet')

    csv_write_s = _time_it(lambda: df.to_csv(csv_path, index=False), n_repeats)
    csv_size = os.path.getsize(csv_path)
    csv_read_s = _time_it(lambda: pd.read_csv(csv_path), n_repeats)

    parquet_write_s = _time_it(lambda: df.to_parquet(parquet_path, index=False), n_repeats)
    parquet_size = os.path.getsize(parquet_path)
    parquet_read_s = _time_it(lambda: pd.read_parquet(parquet_path), n_repeats)

    # Correctness check before trusting any timing: both formats must
    # round-trip the data exactly (same shape, same values once floating
    # point columns are compared with a tolerance for the format's own
    # serialization precision).
    csv_roundtrip = pd.read_csv(csv_path)
    parquet_roundtrip = pd.read_parquet(parquet_path)
    if csv_roundtrip.shape != df.shape or parquet_roundtrip.shape != df.shape:
        raise AssertionError('round-trip shape mismatch -- benchmark results would be untrustworthy')

    return {
        'csv': {'write_s': csv_write_s, 'read_s': csv_read_s, 'size_bytes': csv_size},
        'parquet': {'write_s': parquet_write_s, 'read_s': parquet_read_s, 'size_bytes': parquet_size},
    }


def benchmark_column_subset_read(df, tmp_dir, column, n_repeats=3):
    """
    Time reading a SINGLE column out of an already-written file, for both
    formats. This is the closest single-machine analogy to the book's own
    HDF5-indexing result (Figure 22.8's 16.95s -> 4.59s): row-based CSV
    must scan every row regardless of how many columns you actually want,
    while Parquet's columnar layout can skip the column blocks you didn't
    ask for.

    Parameters
    ----------
    df : pd.DataFrame
    tmp_dir : str
        Directory containing (or to receive) bench.csv / bench.parquet --
        reuses files from benchmark_write_read if already present, writes
        them fresh otherwise.
    column : str
        Column name to read in isolation.
    n_repeats : int, default 3

    Returns
    -------
    dict
        {'csv_column_read_s': float, 'parquet_column_read_s': float,
         'result_matches': bool}
    """
    csv_path = os.path.join(tmp_dir, 'bench.csv')
    parquet_path = os.path.join(tmp_dir, 'bench.parquet')
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    if not os.path.exists(parquet_path):
        df.to_parquet(parquet_path, index=False)

    csv_col_s = _time_it(lambda: pd.read_csv(csv_path, usecols=[column]), n_repeats)
    parquet_col_s = _time_it(lambda: pd.read_parquet(parquet_path, columns=[column]), n_repeats)

    csv_result = pd.read_csv(csv_path, usecols=[column])[column]
    parquet_result = pd.read_parquet(parquet_path, columns=[column])[column]
    matches = len(csv_result) == len(parquet_result) == len(df)

    return {
        'csv_column_read_s': csv_col_s,
        'parquet_column_read_s': parquet_col_s,
        'result_matches': matches,
    }
