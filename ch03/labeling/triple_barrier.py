import numpy as np
import pandas as pd

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
# 1. getDailyVol   → compute volatility to set barrier widths
# 2. getEvents     → for each CUSUM event, set up the three barriers
# 3. applyPtSlOnT1 → find which barrier was hit first (and when)
# 4. getBins       → assign final labels based on which barrier was hit


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


def apply_pt_sl_on_t1(close, events, pt_sl):
    # Apply Profit-Taking and Stop-Loss Barriers — AFML Chapter 3, Snippet 3.2
    #
    # For each event, walks forward through price bars between entry and the
    # vertical barrier, checking whether the upper or lower barrier was crossed.
    # Returns the timestamp of the first touch for each barrier type.
    #
    # --- Inputs ---
    # close   : pd.Series — closing prices indexed by datetime
    # events  : pd.DataFrame — columns: t1, trgt, side
    # pt_sl   : list [pt_multiplier, sl_multiplier]
    #
    # --- Output ---
    # pd.DataFrame with columns: sl, pt (timestamps of first touch or NaT)

    out = events[['t1']].copy(deep=True)

    if pt_sl[0] > 0:
        pt = pt_sl[0] * events['trgt']
    else:
        pt = pd.Series(index=events.index, dtype=float)

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events['trgt']
    else:
        sl = pd.Series(index=events.index, dtype=float)

    for loc, t1 in events['t1'].fillna(close.index[-1]).items():

        # Get path prices from entry to vertical barrier
        # Use .loc slicing — handles irregular timestamps gracefully
        df0 = close.loc[loc:t1]

        # Compute returns relative to entry price, scaled by trade side
        df0 = (df0 / close[loc] - 1) * events.at[loc, 'side']

        # Find earliest stop loss touch (return drops below sl threshold)
        sl_hits = df0[df0 < sl[loc]]
        out.loc[loc, 'sl'] = sl_hits.index.min() if len(sl_hits) > 0 else pd.NaT

        # Find earliest profit target touch (return rises above pt threshold)
        pt_hits = df0[df0 > pt[loc]]
        out.loc[loc, 'pt'] = pt_hits.index.min() if len(pt_hits) > 0 else pd.NaT

    return out


def get_events(close, t_events, pt_sl, trgt, min_ret, t1=False):
    # Get Time of First Touch — AFML Chapter 3, Snippet 3.3
    #
    # Main orchestrator. Sets up the three barriers for each event and
    # finds which one is touched first.
    #
    # --- Inputs ---
    # close    : pd.Series — closing prices indexed by datetime
    # t_events : pd.DatetimeIndex — entry dates (from CUSUM filter)
    # pt_sl    : list [pt_multiplier, sl_multiplier]
    # trgt     : pd.Series — barrier width per event (daily volatility)
    # min_ret  : float — minimum target return to include an event
    # t1       : pd.Series or False — vertical barrier timestamps
    #
    # --- Output ---
    # pd.DataFrame with columns: t1 (first touch timestamp), trgt

    # Filter trgt to event dates only
    trgt = trgt.reindex(t_events, method='bfill')

    # Apply minimum return filter
    trgt = trgt[trgt > min_ret]

    # Set up vertical barriers
    if t1 is False:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Build events DataFrame — side is always +1 for now
    side_ = pd.Series(1., index=trgt.index)

    events = pd.concat(
        {'t1': t1, 'trgt': trgt, 'side': side_},
        axis=1
    ).dropna(subset=['trgt'])

    # Find first barrier touch (single-threaded version)
    df0 = apply_pt_sl_on_t1(
        close=close,
        events=events,
        pt_sl=pt_sl
    )

    # Take the earliest touch across all three barriers
    events['t1'] = df0.dropna(how='all').min(axis=1)
    events = events.drop('side', axis=1)

    return events


def add_vertical_barrier(close, t_events, num_days):
    # Add Vertical Barrier — AFML Chapter 3, Snippet 3.4
    #
    # For each event date, finds the timestamp of the bar that falls
    # approximately num_days later.
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
    # Labeling for Side and Size — AFML Chapter 3, Snippet 3.5
    #
    # Assigns final labels (+1, -1, 0) based on which barrier was hit first
    # and computes the actual return achieved.
    #
    # --- Inputs ---
    # events : pd.DataFrame — output of get_events()
    # close  : pd.Series — closing prices indexed by datetime
    #
    # --- Output ---
    # pd.DataFrame with columns: ret (actual return), bin (label in {-1, 0, 1})

    events_ = events.dropna(subset=['t1'])

    # Build price series covering all entry and exit dates
    px = events_.index.union(events_['t1'].values).drop_duplicates()

    # Reindex close to cover all needed dates, forward-fill gaps
    px = close.reindex(px, method='bfill')

    out = pd.DataFrame(index=events_.index)

    # Get exit prices — use nearest available bar if exact timestamp not in index
    exit_prices = []
    for t1 in events_['t1'].values:
        # Find nearest bar at or after t1
        idx = close.index.searchsorted(t1)
        idx = min(idx, len(close) - 1)
        exit_prices.append(close.iloc[idx])

    entry_prices = close.reindex(events_.index, method='bfill').values

    out['ret'] = (np.array(exit_prices) / entry_prices) - 1
    out['bin'] = np.sign(out['ret'])

    return out
