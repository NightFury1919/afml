import numpy as np
import pandas as pd
import sys
import os

# Add project root to path so we can import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.multiprocess import mp_pandas_obj

# Triple Barrier Labeling — AFML Chapter 3, Sections 3.3-3.5, pages 44-50
#
# The triple barrier method labels each observation as +1, 0, or -1 based on
# which of three barriers is hit first after a trade entry:
#
#   Upper barrier  (+1) — profit target: price rises by trgt * ptSl[0]
#   Lower barrier  (-1) — stop loss:     price falls by trgt * ptSl[1]
#   Vertical barrier (0) — time limit:   neither barrier hit within numDays
#
# This is more realistic than the fixed-time horizon method because it
# accounts for the PATH the price took, not just where it ended up.
#
# --- Pipeline ---
# 1. get_daily_vol      → compute volatility to set barrier widths
# 2. add_vertical_barrier → find the timestamp num_days after each event
# 3. get_events         → set up all three barriers, find first touch
#                         (uses mpPandasObj to call apply_pt_sl_on_t1 in parallel)
# 4. apply_pt_sl_on_t1  → checks each bar for barrier crosses (worker function)
# 5. get_bins           → assign final +1/-1/0 labels


def get_daily_vol(close, span0=100):
    # Daily Volatility Estimates — AFML Chapter 3, Snippet 3.1, page 44
    #
    # Computes the exponentially weighted moving standard deviation of
    # daily returns. Used to set the width of the horizontal barriers
    # dynamically — so profit targets and stop losses scale with current
    # market volatility rather than being fixed dollar amounts.
    #
    # --- Inputs ---
    # close : pd.Series — closing prices indexed by datetime
    # span0 : int — EWMA span for volatility estimation (default 100 bars)
    #
    # --- Output ---
    # pd.Series of daily volatility estimates, same index as close

    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0 - 1],
        index=close.index[close.shape[0] - df0.shape[0]:]
    )
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    df0 = df0.ewm(span=span0).std()
    return df0


def apply_pt_sl_on_t1(molecule, close, events, pt_sl):
    # Apply Profit-Taking and Stop-Loss Barriers — AFML Chapter 3, Snippet 3.2, page 45
    #
    # Worker function called by mpPandasObj in get_events().
    # Processes only the subset of events in 'molecule' (its assigned chunk).
    # For each event in molecule, walks forward through price bars between
    # entry and the vertical barrier, checking whether the upper or lower
    # barrier was crossed first.
    #
    # --- Why 'molecule'? ---
    # The book uses the term 'molecule' for the chunk of the index assigned
    # to one parallel worker. Each worker gets a different slice of events.index
    # and processes only those rows independently. mpPandasObj then stitches
    # all results back together.
    #
    # --- Inputs ---
    # molecule : pd.DatetimeIndex — the subset of events.index this worker handles
    # close    : pd.Series — closing prices indexed by datetime
    # events   : pd.DataFrame — columns: t1, trgt, side (full DataFrame, not just molecule)
    # pt_sl    : list [pt_multiplier, sl_multiplier]
    #
    # --- Output ---
    # pd.DataFrame with columns: t1, sl, pt (timestamps of first touch or NaT)
    #   Indexed by the dates in molecule only.

    # Work only on the rows assigned to this worker (molecule)
    out = events.loc[molecule, ['t1']].copy(deep=True)

    # Compute barrier widths
    # pt (profit target): how far price must rise to trigger upper barrier
    # sl (stop loss):     how far price must fall to trigger lower barrier
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index, dtype=float)

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index, dtype=float)

    # Loop over only the events assigned to this molecule
    for loc, t1 in events.loc[molecule, 't1'].fillna(close.index[-1]).items():

        # Get path prices from entry (loc) to vertical barrier (t1)
        # .loc[loc:t1] gives all bars in that time window
        df0 = close.loc[loc:t1]

        # Compute returns relative to entry price, scaled by trade side
        # side=+1 (long): profit when price rises, loss when falls
        # side=-1 (short): profit when price falls, loss when rises
        df0 = (df0 / close[loc] - 1) * events.at[loc, 'side']

        # Find earliest stop loss touch (return drops below sl threshold)
        sl_hits = df0[df0 < sl[loc]]
        out.loc[loc, 'sl'] = sl_hits.index.min() if len(sl_hits) > 0 else pd.NaT

        # Find earliest profit target touch (return rises above pt threshold)
        pt_hits = df0[df0 > pt[loc]]
        out.loc[loc, 'pt'] = pt_hits.index.min() if len(pt_hits) > 0 else pd.NaT

    return out


def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads=1, t1=False):
    # Get Time of First Touch — AFML Chapter 3, Snippet 3.3, page 46
    #
    # Main orchestrator. Sets up the three barriers for each event and
    # finds which one is touched first, using mpPandasObj to parallelize
    # the barrier-checking across CPU cores.
    #
    # --- Inputs ---
    # close       : pd.Series — closing prices indexed by datetime
    # t_events    : pd.DatetimeIndex — entry dates (from CUSUM filter)
    # pt_sl       : list [pt_multiplier, sl_multiplier]
    # trgt        : pd.Series — barrier width per event (daily volatility)
    # min_ret     : float — minimum target return to include an event
    # num_threads : int — number of parallel workers (default 1)
    #               Set higher to speed up large datasets
    # t1          : pd.Series or False — vertical barrier timestamps
    #               Pass the output of add_vertical_barrier() here.
    #               If False, no vertical barrier (trade runs until a
    #               horizontal barrier is hit, which may never happen).
    #
    # --- Output ---
    # pd.DataFrame with columns:
    #   t1   : timestamp of first barrier touch
    #   trgt : barrier width used for this event

    # Step 1: Filter trgt to event dates only (bfill = use next available vol)
    trgt = trgt.reindex(t_events, method='bfill')

    # Step 2: Apply minimum return filter — skip events where vol is too low
    # This removes events during very quiet markets where barriers would be
    # too tight to be meaningful
    trgt = trgt[trgt > min_ret]

    # Step 3: Set up vertical barriers
    # If no t1 provided, set NaT for all events (no time limit)
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Step 4: Build events DataFrame
    # side is always +1 here — we assume long for barrier setup.
    # (Meta-labeling handles direction separately in get_events_meta)
    side_ = pd.Series(1., index=trgt.index)

    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_},
        axis=1
    ).dropna(subset=['trgt'])

    # Step 5: Find first barrier touch using mpPandasObj
    # This is the key parallelization point from Snippet 3.3.
    # mpPandasObj splits events.index into num_threads chunks and calls
    # apply_pt_sl_on_t1 on each chunk in a separate process.
    # Each worker returns a DataFrame; mpPandasObj concatenates them.
    #
    # pdObj = ('molecule', events.index) tells mpPandasObj that the
    # first argument of apply_pt_sl_on_t1 is called 'molecule' and
    # should receive the chunk of events.index assigned to that worker.
    df0 = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl
    )

    # Step 6: Take the earliest touch across all three barriers
    # min(axis=1) returns the smallest (earliest) timestamp in each row,
    # ignoring NaT values (pd.min ignores NaN/NaT by default)
    events['t1'] = df0.dropna(how='all').min(axis=1)
    events = events.drop('side', axis=1)

    return events


def add_vertical_barrier(close, t_events, num_days):
    # Add Vertical Barrier — AFML Chapter 3, Snippet 3.4, page 47
    #
    # For each event date, finds the timestamp of the bar that falls
    # approximately num_days later. This becomes the vertical barrier —
    # the maximum holding period for the trade.
    #
    # --- Inputs ---
    # close    : pd.Series — closing prices indexed by datetime
    # t_events : pd.DatetimeIndex — entry dates
    # num_days : int — maximum holding period in calendar days
    #
    # --- Output ---
    # pd.Series — vertical barrier timestamps, indexed by t_events

    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(
        close.index[t1],
        index=t_events[:t1.shape[0]]
    )
    return t1


def get_bins(events, close):
    # Labeling for Side and Size — AFML Chapter 3, Snippet 3.5, page 48
    #
    # Assigns final labels (+1, -1, 0) based on which barrier was hit first
    # and computes the actual return achieved.
    #
    # --- Inputs ---
    # events : pd.DataFrame — output of get_events()
    # close  : pd.Series — closing prices indexed by datetime
    #
    # --- Output ---
    # pd.DataFrame with columns:
    #   ret : actual return from entry to first barrier touch
    #   bin : label in {-1, 0, 1}
    #         +1 = upper barrier hit first (profit target reached)
    #         -1 = lower barrier hit first (stop loss triggered)
    #          0 = vertical barrier hit first (time expired, neutral)

    events_ = events.dropna(subset=['t1'])

    # Build price series covering all entry and exit dates
    px = events_.index.union(events_['t1'].values).drop_duplicates()

    # Reindex close to cover all needed dates, forward-fill gaps
    px = close.reindex(px, method='bfill')

    out = pd.DataFrame(index=events_.index)

    # Get exit prices — use nearest available bar if exact timestamp not in index
    exit_prices = []
    for t1 in events_['t1'].values:
        idx = close.index.searchsorted(t1)
        idx = min(idx, len(close) - 1)
        exit_prices.append(close.iloc[idx])

    entry_prices = close.reindex(events_.index, method='bfill').values

    out['ret'] = (np.array(exit_prices) / entry_prices) - 1
    out['bin'] = np.sign(out['ret'])

    return out
