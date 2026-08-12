"""
pipeline/orchestration/rebuild.py

Rebuilds bars -> CUSUM events -> triple-barrier labels -> sample weights
from a raw trades DataFrame (either the existing static CSV or a fresh
ingestion.pull_recent_trades() pull), reusing Ch02/03/04's real functions
directly -- no reimplementation.

Real modules reused:
  - ch02/bars/utils.py           (delta)
  - ch02/bars/standard_bars.py   (dollar_bars)
  - ch02/bars/filters.py         (cusum_filter)
  - ch03/labeling/triple_barrier.py  (get_daily_vol, add_vertical_barrier,
    get_events, get_bins)
  - ch04/sample_weights/uniqueness.py       (get_average_uniqueness)
  - ch04/sample_weights/return_attribution.py (get_sample_weights)

Calibration constants (CUSUM h=500, get_daily_vol span0=100, pt_sl=[1,1],
min_ret=0.005, num_days=3) are carried over UNCHANGED from this project's
established real calibration on the March 2026 static dataset (see
ch03/examples_chapter_3_labeling.py). These are real, already-validated
choices for this asset -- reusing them here, rather than re-deriving new
ones, is deliberate. KNOWN OPEN QUESTION (documented, not resolved): CUSUM's
h=500 is a flat DOLLAR threshold, calibrated when BTC was trading near
$65,000 in March 2026. If a live pull happens at a meaningfully different
BTC price level, h=500 may fire CUSUM events too often or too rarely
relative to the original calibration's intent (an h-per-day-volatility
scaling would be more defensible but is a new design decision this project
hasn't made yet -- flagged here rather than silently invented).

The DOLLAR-BAR THRESHOLD, unlike the constants above, is NOT carried over
fixed at $10,000 -- Phase 2 scope explicitly calls for it to scale with
the live pull's size (a fixed $10,000 threshold would produce wildly
different bar counts for a 1-hour pull vs. a 30-day pull). See
compute_dynamic_threshold().
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ch02/bars, ch03/labeling, and ch04/sample_weights all have their own
# __init__.py with relative imports (from .co_events import ...) -- unlike
# ch07/ch10/ch11, which don't. Importing via the full package path (not a
# bare `import uniqueness` after sys.path.insert) is REQUIRED here, or
# Python raises "attempted relative import with no known parent package".
from ch02.bars import utils as ch02_utils            # real module
from ch02.bars import standard_bars                  # real module
from ch02.bars import filters as ch02_filters         # real module
from ch03.labeling import triple_barrier              # real module
from ch04.sample_weights import uniqueness            # real module
from ch04.sample_weights import return_attribution    # real module

# Established real calibration, carried over unchanged (see module docstring)
CUSUM_H = 500
DAILY_VOL_SPAN0 = 100
PT_SL = [1, 1]
MIN_RET = 0.005
VERTICAL_BARRIER_NUM_DAYS = 3


def compute_dynamic_threshold(raw_trades, target_bars=250):
    """Dollar-bar threshold scaled to the pull's own total dollar volume,
    targeting roughly `target_bars` bars regardless of how much history was
    pulled -- a fixed $10,000 threshold (this project's static-data
    convention) would produce wildly different bar counts for a 1-hour vs.
    a 30-day pull. target_bars=250 matches the established static dataset's
    real bar count (249 bars on ~9,205 March 2026 trades), so downstream
    CUSUM/triple-barrier calibration (tuned for that bar density) stays in
    a comparable regime."""
    total_dollar_volume = (raw_trades['Price'] * raw_trades['Volume']).sum()
    if total_dollar_volume <= 0:
        raise ValueError('Raw trades have zero or negative total dollar volume')
    return float(total_dollar_volume) / target_bars


def preprocess_raw_trades(raw_trades):
    """Real preprocessing from ch03/examples_chapter_3_labeling.py: parse
    Timestamp (microseconds) to a Date column, derive tick-rule Label from
    IsBuyerMaker, compute Dollar value, run ch02's real delta() (needed by
    downstream tick-rule-based features, not used directly here but kept
    for parity with the established real pipeline shape)."""
    raw = raw_trades.copy()
    raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')
    raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)

    df = raw[['Date', 'Price', 'Volume', 'Label']].copy()
    df['Dollar'] = df['Price'] * df['Volume']
    df = ch02_utils.delta(df)
    return df


def build_bars_and_labels(raw_trades, target_bars=250):
    """Full real chain: raw trades -> dynamic-threshold dollar bars ->
    CUSUM events -> daily vol -> vertical barrier -> triple-barrier events
    -> bins -> sample weights. Every step calls a real, already-tested
    Ch02/03/04 function; this function only sequences them and computes
    the dynamic threshold.

    Returns
    -------
    dict with keys:
      'bars'       : pd.DataFrame, dollar bars (Date, Open, Low, High,
                     Close, Vwap)
      'close'      : pd.Series, bar close prices (index = Date)
      'threshold'  : float, the dynamic dollar-bar threshold used
      'events'     : pd.DataFrame, triple-barrier events + bins (t1, trgt,
                     ret, bin) -- schema-identical to ch03_events.csv
      'w'          : pd.Series, return-attribution sample weight (Ch04)
      'tw'         : pd.Series, average uniqueness (Ch04)
    """
    df = preprocess_raw_trades(raw_trades)

    threshold = compute_dynamic_threshold(raw_trades, target_bars=target_bars)
    bars = standard_bars.dollar_bars(df, thresh=threshold)
    if bars.empty:
        raise ValueError(
            'Dynamic-threshold dollar bar construction produced zero bars '
            '-- the pulled trade window may be too short for target_bars'
        )
    bars = bars.set_index('Date')
    close = bars['Close']

    cusum_df = pd.DataFrame({'Date': close.index, 'Price': close.values})
    cusum_events = ch02_filters.cusum_filter(cusum_df, h=CUSUM_H)
    if len(cusum_events) == 0:
        raise ValueError(
            f'CUSUM filter (h={CUSUM_H}) produced zero events on this bar '
            'series -- likely too little price movement in the pulled '
            'window relative to the established h calibration (see module '
            'docstring\'s KNOWN OPEN QUESTION on h scaling)'
        )

    daily_vol = triple_barrier.get_daily_vol(close, span0=DAILY_VOL_SPAN0)
    t1 = triple_barrier.add_vertical_barrier(
        close, cusum_events, num_days=VERTICAL_BARRIER_NUM_DAYS,
    )
    tb_events = triple_barrier.get_events(
        close=close, t_events=cusum_events, pt_sl=PT_SL,
        trgt=daily_vol, min_ret=MIN_RET, t1=t1,
    )
    bins = triple_barrier.get_bins(tb_events, close)
    if bins.empty:
        raise ValueError(
            'Triple-barrier labeling produced zero events on this pulled '
            'window -- likely too few CUSUM events survived min_ret '
            'filtering for a window this short'
        )
    # get_bins() only returns ret/bin (see its docstring) -- t1/trgt come
    # from tb_events, matching this project's established ch03_events.csv
    # schema (t1, trgt, ret, bin). Real driver precedent: Ch04's own
    # get_sample_weights/get_average_uniqueness calls use tb_events (which
    # HAS t1), not the post-get_bins frame -- an earlier draft of this
    # function passed the wrong one and hit a real KeyError('t1'), caught
    # in sandbox testing before this was ever handed off.
    events = tb_events[['t1', 'trgt']].join(bins, how='inner')

    w = return_attribution.get_sample_weights(close, tb_events, num_threads=1)
    tw = uniqueness.get_average_uniqueness(close, tb_events, num_threads=1)
    # get_sample_weights/get_average_uniqueness are computed on tb_events
    # (pre-get_bins), matching Ch04's own real driver precedent. On the
    # established static dataset tb_events and the post-get_bins `events`
    # happen to have identical length (no in-flight events near the
    # dataset's end). A live pull's cutoff can leave the last few CUSUM
    # events without a resolved t1 (still "in flight" when the pull ended)
    # -- get_bins() drops those via dropna(subset=['t1']), so w/tw are
    # explicitly reindexed here to guarantee alignment with the final
    # `events` table rather than relying on that lucky static-data
    # coincidence.
    w = w.reindex(events.index)
    tw = tw.reindex(events.index)

    return {
        'bars': bars,
        'close': close,
        'threshold': threshold,
        'events': events,
        'w': w,
        'tw': tw,
    }
