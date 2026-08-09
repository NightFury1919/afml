"""
AFML Chapter 22 supplement -- non-uniform Fourier transform (Section 22.6.6).

NOT a book snippet -- none exists for this. The book applies a
non-uniform FFT to irregularly-timestamped trading prices/volumes to look
for periodicities (daily cycles, algorithmic once-per-minute footprints)
that a regularly-sampled FFT can't honestly represent, since trade arrival
times are NOT evenly spaced. Our real BTC/TUSD trade data has exactly
that property (irregular inter-trade timing), so this is a genuinely
applicable technique, implemented directly from the underlying math
(a direct, non-uniform discrete Fourier transform) rather than a library,
since the book gives no snippet to translate and our dataset (9,205
points) is small enough that the O(N x F) direct-sum cost is trivial --
no need for a fast approximate algorithm.

Frequency units: cycles per day (not per year, unlike the book's
natural-gas example) -- our real dataset spans about a month (2026-03-01
to 2026-03-31), roughly 300 trades/day on average, which is far sparser
than the book's own year-long, much-higher-frequency futures data. A
once-per-minute search (like the book's TWAP finding) is not realistically
resolvable at this trade density; we search daily/sub-daily frequencies
instead and report honestly whatever the spectrum shows (including "no
strong signal," per this project's standing practice of reporting genuine
real-data findings rather than only the ones that look interesting).
"""

import numpy as np
import pandas as pd


def days_since_start(timestamps):
    """
    Convert a pandas datetime Series/Index to float days since the first
    timestamp -- gives frequency units of cycles/day directly.

    Parameters
    ----------
    timestamps : pd.Series or pd.DatetimeIndex

    Returns
    -------
    np.ndarray, float64
    """
    ts = pd.DatetimeIndex(timestamps)
    t0 = ts.min()
    return ((ts - t0) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def nudft(times, values, freqs):
    """
    Direct non-uniform discrete Fourier transform.

    X(f) = sum_n  values[n] * exp(-2j * pi * f * times[n])

    Parameters
    ----------
    times : array-like, shape (N,)
        Non-uniformly spaced sample times (any consistent unit -- use
        days_since_start() for cycles/day frequency units).
    values : array-like, shape (N,)
        Signal values at each time (need not be real; typically is here).
    freqs : array-like, shape (F,)
        Candidate frequencies to evaluate, same unit as 1/times.

    Returns
    -------
    np.ndarray, shape (F,), complex128
        X(f) for each candidate frequency.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    # Outer product times[:, None] * freqs[None, :] -> (N, F) phase matrix,
    # summed over N (axis=0) to get one complex value per frequency.
    phase = -2j * np.pi * np.outer(times, freqs)
    return (values[:, None] * np.exp(phase)).sum(axis=0)


def power_spectrum(times, values, freqs):
    """
    |X(f)|^2 at each candidate frequency, via nudft(). The magnitude
    (not power) is often more directly comparable to the book's own
    Figure 22.10 (log-scale amplitude plot), so both are exposed.

    Returns
    -------
    dict
        {'X': complex spectrum (F,), 'magnitude': |X| (F,), 'power': |X|^2 (F,)}
    """
    x = nudft(times, values, freqs)
    magnitude = np.abs(x)
    return {'X': x, 'magnitude': magnitude, 'power': magnitude ** 2}


def price_return_series(trades_df):
    """
    Convert a raw price series to log returns -- a raw (non-stationary,
    trending) price level's Fourier transform is dominated by a huge
    near-zero-frequency component and isn't informative about periodic
    structure; returns are the standard de-trended alternative and are
    what actually reveals periodic activity, if any exists.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Must have 'ts' (datetime) and 'price' columns, in time order.

    Returns
    -------
    times_days : np.ndarray, days since first trade (length N-1)
    log_returns : np.ndarray (length N-1)
    """
    trades_df = trades_df.sort_values('ts')
    log_price = np.log(trades_df['price'].to_numpy())
    log_returns = np.diff(log_price)
    times_days = days_since_start(trades_df['ts'])[1:]  # aligned to the return ENDING at that time
    return times_days, log_returns


def trade_size_series(trades_df):
    """
    The trade-size (qty) analog of price_return_series -- the book also
    runs its non-uniform FFT on trading VOLUMES, not just prices
    (Section 22.6.6, final paragraph).

    Returns
    -------
    times_days : np.ndarray
    qty : np.ndarray
    """
    trades_df = trades_df.sort_values('ts')
    times_days = days_since_start(trades_df['ts'])
    return times_days, trades_df['qty'].to_numpy()
