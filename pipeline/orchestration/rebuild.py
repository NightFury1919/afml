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

Calibration constants (get_daily_vol span0=100, pt_sl=[1,1], min_ret=0.005,
num_days=3) are carried over UNCHANGED from this project's established real
calibration on the March 2026 static dataset (see
ch03/examples_chapter_3_labeling.py). These are real, already-validated
choices for this asset -- reusing them here, rather than re-deriving new
ones, is deliberate.

*** LOAD-BEARING (2026-08-21): CUSUM_H changed 500 -> 313 -- a measured
staleness correction, not a re-derivation from scratch *** The original
h=500 was a flat DOLLAR threshold calibrated when BTC was trading near
$65,000 in March 2026 (this module's own KNOWN OPEN QUESTION, first flagged
2026-08-13, now resolved by real measurement rather than left open).
`pipeline/diagnostics/audit_cusum_h_staleness.py` ran both the March static
baseline and a fresh 720h live BTCUSDT pull through this exact
build_bars_and_labels() chain: h=500 fired on 30.3% of live bars vs.
March's 42.6% (CALIBRATION_AUDIT.md's "CUSUM_H Staleness Audit" section)
-- the threshold had drifted too HIGH for current data, not too low, and
h=313 was the measured value restoring March's relative firing rate.
CAVEAT carried forward: a single-day measurement, not a multi-day average
-- may need revisiting if a later measurement shows further drift. The
deliberate h-per-day-volatility redesign the original KNOWN OPEN QUESTION
called for remains undesigned; this is a measured point-fix, not that
redesign.

Adopting h=313 ALONE was shown to cost T_effective (-34.9% at the prior
target_bars=250 default -- CALIBRATION_AUDIT.md's "CUSUM_H Staleness
Correction vs. T_effective" section, via a tw_mean uniqueness collapse).
This is why target_bars' production default changed too -- see
run_pipeline_live.py's own LOAD-BEARING note on target_bars=250->1000,
adopted together with this change specifically to absorb that cost.

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

# Reused from ingestion.py, not reimplemented -- see preprocess_raw_trades'
# own LOAD-BEARING note (2026-08-21) for why this is needed here too.
from ingestion import _disambiguate_timestamps

# Established real calibration, carried over unchanged (see module docstring)
CUSUM_H = 313  # LOAD-BEARING (2026-08-21): was 500 -- see module docstring
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
    for parity with the established real pipeline shape).

    *** LOAD-BEARING (2026-08-21): disambiguates Timestamp FIRST, reusing
    ingestion.py's real _disambiguate_timestamps() *** -- ingestion.py's
    own LOAD-BEARING note documents that live pulls can have trades sharing
    an identical millisecond, disambiguated there via consecutive
    microsecond offsets. That fix only ran inside pull_recent_trades()
    itself, leaving any OTHER raw_trades source (the static March CSV, any
    future test fixture) unprotected. Real, confirmed bug found 2026-08-21
    while testing target_bars=1000 on the static dataset for adoption as
    this pipeline's live default: the static CSV has 561 duplicate raw
    timestamps out of 9,205 (~6%); at a fine enough bar granularity, these
    occasionally land on a bar-close boundary, producing two bars sharing
    one Date index value, which crashes triple_barrier.get_daily_vol()
    with a real ValueError (shape mismatch in its .loc lookup -- Ch03's own
    Snippet 3.1 code is correct; the bug is upstream data hygiene, not a
    book bug). Confirmed the fix resolves target_bars=1000 on the static
    dataset (789 bars, 171 events, no crash) with ZERO change to the
    already-passing target_bars=250/500 results -- safe, non-disruptive.
    Calling this here makes preprocess_raw_trades() -- and therefore
    build_bars_and_labels() -- genuinely data-source agnostic, matching
    this module's own test file's stated intent, rather than relying on
    every caller having already gone through ingestion.py. Idempotent on
    already-disambiguated live data (already-unique timestamps get a zero
    offset, confirmed by inspection of _disambiguate_timestamps' groupby-
    cumcount logic)."""
    raw = _disambiguate_timestamps(raw_trades)
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