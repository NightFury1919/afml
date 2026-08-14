"""
pipeline/orchestration/features.py

Phase 2c: rebuilds Ch19's 11 microstructural features + Ch05's frac-diff
feature on live-pulled data, joining them onto rebuild.py's triple-barrier
events -- the same enrichment build_enriched_training_table.py performs
for the static dataset, adapted for a live pull's dynamic bar threshold.

Real modules reused directly (no reimplementation):
  - ch19/microstructural_features/microstructural_features.py (all 11
    feature functions, exact same calling convention as
    chapter_19_microstructural_features.py's real driver)
  - ch05/frac_diff/{find_min_ffd,frac_diff_ffd,get_weights_ffd}.py

*** LOAD-BEARING (2026-08-13): bar-id trade-tagging is duplicated, not
reused, from ch19's own Part A / build_enriched_training_table.py's
build_bar_id_to_date ***
ch02's dollar_bars() (reused elsewhere in this pipeline, see rebuild.py)
runs the identical cumulative-dollar-threshold accumulate-and-reset loop
internally but does NOT expose a per-trade bar_id. Ch19's own driver and
build_enriched_training_table.py both independently re-implement this
same loop rather than modifying dollar_bars() to add that output. This
module follows that SAME established precedent (not a new violation of
the project's reuse-over-reimplementation rule) so that BuyVolume/
SellVolume/DollarVolume/n_trades -- inputs several Ch19 features need
that dollar_bars() doesn't produce -- can be computed per bar.

CONSISTENCY REQUIREMENT: build_enriched_events() must be called with the
exact same raw_trades DataFrame (same row order, unmodified) and the same
dynamic threshold that rebuild.py's build_bars_and_labels() used for this
pull, or the resulting bar-close timestamps will NOT line up with
rebuild.py's `events` index and the join will silently produce mostly-NaN
rows rather than a loud error.

*** LOAD-BEARING (2026-08-13): minimum stationary d is RE-DERIVED on the
live bar series, not assumed to be the static baseline's d=0.2 ***
Matches chapter_5_bar_features.py's own stated principle: dollar bars
aggregate away tick microstructure, so the minimal d is not guaranteed to
transfer between datasets -- check, don't assume. A live pull with far
fewer bars than the static 249-bar baseline may plausibly find NO d in
[0, 1] that passes the ADF test (p<0.05) -- if so, compute_fracdiff_feature
surfaces that explicitly (returns d=None), not silently defaulted to 0.2.

*** LOAD-BEARING (2026-08-13): ROLL_WINDOW=20 carried over unchanged from
Ch19's established calibration, NOT rescaled for live data ***
On a live pull with far fewer bars than the static 249-bar baseline, a
20-bar rolling window consumes a proportionally larger share of the
series (more NaN-warmup rows relative to total bars) than it did on the
static dataset. Rescaling this window is a genuine design decision this
project hasn't made -- flagged here rather than silently invented,
mirroring rebuild.py's own open-question convention for CUSUM's h.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ch19/microstructural_features has no __init__.py (matches this project's
# no-__init__.py-in-test-containing-folders convention) -- it works as an
# implicit namespace package, exactly like chapter_19_microstructural_
# features.py's own import line.
from ch19.microstructural_features import microstructural_features as mf   # real module

# ch05/frac_diff's modules use BARE imports between themselves
# (find_min_ffd.py does `from frac_diff_ffd import frac_diff_ffd`, not a
# package-relative one) -- they only resolve if the frac_diff/ folder
# itself is on sys.path, not just ROOT. Matches chapter_5_bar_features.py's
# own import style exactly (NOT the dotted-namespace style used for ch19).
CH05_FRAC_DIFF = os.path.join(ROOT, 'ch05', 'frac_diff')
if CH05_FRAC_DIFF not in sys.path:
    sys.path.insert(0, CH05_FRAC_DIFF)

from find_min_ffd import find_min_ffd, find_minimum_d   # real module
from frac_diff_ffd import frac_diff_ffd                 # real module
from get_weights_ffd import get_weights_ffd              # real module

ROLL_WINDOW = 20        # carried over unchanged from Ch19 (see LOAD-BEARING note)
FFD_THRES = 0.01        # carried over unchanged from Ch05's established calibration
VPIN_WINDOW = 10         # carried over unchanged from Ch19's established calibration


def _retag_trades_with_bar_id(raw_trades, threshold):
    """Re-run the SAME cumulative-dollar-threshold accumulate-and-reset
    loop dollar_bars() uses internally (see module LOAD-BEARING note),
    against the pull's dynamic threshold, to tag every trade with the
    bar it belongs to. Drops the trailing incomplete bar, matching Ch19's
    own driver and build_enriched_training_table.py's convention.

    Returns
    -------
    trades : pd.DataFrame, columns Date/Price/Volume/Label/bar_id, one
        row per trade in the completed bars (incomplete trailing bar's
        trades dropped)
    """
    raw = raw_trades.copy()
    raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')
    raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)
    trades = raw[['Date', 'Price', 'Volume', 'Label']].copy()

    cumm_dollar, bar_id, bar_ids = 0.0, 0, []
    for price, volume in zip(trades['Price'], trades['Volume']):
        cumm_dollar += price * volume
        bar_ids.append(bar_id)
        if cumm_dollar >= threshold:
            bar_id += 1
            cumm_dollar = 0.0
    trades['bar_id'] = bar_ids

    n_complete_bars = trades['bar_id'].max()
    trades = trades[trades['bar_id'] < n_complete_bars].copy()
    return trades


def _build_bars_with_volume(trades):
    """Bar-level OHLC + volume aggregates Ch19's features need that
    rebuild.py's bars (Open/Low/High/Close/Vwap only) don't carry.
    Indexed by bar-close Date, matching rebuild.py's `close`/`events`
    index exactly (same trades, same threshold => same bar boundaries).
    """
    bars = trades.groupby('bar_id').agg(
        Date=('Date', 'last'),
        Open=('Price', 'first'), High=('Price', 'max'),
        Low=('Price', 'min'), Close=('Price', 'last'),
        n_trades=('Price', 'size'),
    )
    dollar_per_trade = trades['Price'] * trades['Volume']
    bars['DollarVolume'] = dollar_per_trade.groupby(trades['bar_id']).sum()
    bars['BuyVolume'] = (
        trades[trades['Label'] == 1].groupby('bar_id')['Volume'].sum()
        .reindex(bars.index).fillna(0)
    )
    bars['SellVolume'] = (
        trades[trades['Label'] == -1].groupby('bar_id')['Volume'].sum()
        .reindex(bars.index).fillna(0)
    )
    bars = bars.set_index('Date')
    return bars


def compute_fracdiff_feature(close, thres=FFD_THRES):
    """Re-derive the minimum stationary d ON THIS LIVE BAR SERIES (see
    module LOAD-BEARING note -- never assume the static baseline's
    d=0.2 transfers) and build the frac-diff feature at that d.

    Returns
    -------
    dict with keys:
      'fracdiff' : pd.Series (may be empty if d is None)
      'd'        : float or None (None if no d in [0,1] passed ADF)
      'width'    : int or None
      'results'  : pd.DataFrame, the full find_min_ffd ADF search table
        (kept for inspection/debugging, not just the winning d)
    """
    results = find_min_ffd(close, thres=thres)
    d = find_minimum_d(results)
    if d is None:
        return {'fracdiff': pd.Series(dtype=float), 'd': None,
                'width': None, 'results': results}

    log_close = np.log(close)
    width = len(get_weights_ffd(d, thres=thres)) - 1
    fracdiff = frac_diff_ffd(log_close, d=d, thres=thres)
    fracdiff.name = 'fracdiff'
    return {'fracdiff': fracdiff, 'd': float(d), 'width': int(width),
            'results': results}


def _rolling_bar(a, b, func, window):
    """Same rolling-window helper as chapter_19_microstructural_features.py
    (duplicated, not imported, since it's a small script-local helper in
    the original, not part of the real mf module)."""
    out = [np.nan] * len(a)
    for i in range(window - 1, len(a)):
        try:
            out[i] = func(a[i - window + 1:i + 1], b[i - window + 1:i + 1])
        except Exception:
            out[i] = np.nan
    return out


def compute_ch19_features(trades, bars_vol, roll_window=ROLL_WINDOW,
                           vpin_window=VPIN_WINDOW):
    """All 11 of Ch19's real feature functions, same calling convention
    as chapter_19_microstructural_features.py's Part C (rolling estimators
    at roll_window, whole-series-shaped estimators, and per-bar trade-level
    estimators) -- just against live bars_vol/trades instead of the static
    249-bar table.

    Returns
    -------
    pd.DataFrame indexed by bar-close Date, columns matching Ch19's
    established real feature table exactly (roll_c, roll_sigma_u,
    parkinson_vol_20bar, corwin_schultz_spread, becker_parkinson_sigma,
    kyle_lambda, amihud_lambda_20bar, vpin_10bar, round_number_fraction,
    serial_corr_signed_flow, tick_rule_accuracy).
    """
    closes = bars_vol['Close'].values
    highs = bars_vol['High'].values
    lows = bars_vol['Low'].values
    dollar_vols = bars_vol['DollarVolume'].values

    roll_c, roll_sigma_u = [], []
    for i in range(len(closes)):
        if i < roll_window:
            roll_c.append(np.nan)
            roll_sigma_u.append(np.nan)
            continue
        res = mf.roll_measure(closes[i - roll_window:i + 1])
        roll_c.append(res['c'])
        roll_sigma_u.append(res['sigma_u'])

    parkinson_roll = _rolling_bar(highs, lows, mf.parkinson_volatility, roll_window)
    amihud_roll = _rolling_bar(closes, dollar_vols, mf.amihud_lambda, roll_window)

    cs_spread_full = mf.corwin_schultz(
        pd.Series(highs, index=bars_vol.index),
        pd.Series(lows, index=bars_vol.index), sl=1,
    ).reindex(bars_vol.index)
    bp_sigma_full = mf.becker_parkinson_sigma(
        mf.get_beta(pd.Series(highs, index=bars_vol.index),
                    pd.Series(lows, index=bars_vol.index), 1),
        mf.get_gamma(pd.Series(highs, index=bars_vol.index),
                     pd.Series(lows, index=bars_vol.index)),
    ).reindex(bars_vol.index)

    signed_vol = trades['Volume'] * trades['Label']
    kyle_by_bar_id = mf.kyle_lambda_by_bar(
        trades['Price'].values, signed_vol.values, trades['bar_id'].values,
        min_trades=5,
    )
    # kyle_by_bar_id is bar_id-indexed (see mf.kyle_lambda_by_bar's real
    # docstring) -- map back to Date via bars_vol's own bar_id index,
    # which groupby('bar_id') preserved as bars_vol's positional order.
    bar_id_order = trades.groupby('bar_id').size().index  # sorted bar_ids present
    kyle_full = pd.Series(
        kyle_by_bar_id.reindex(bar_id_order).values, index=bars_vol.index,
    )

    vpin_full = mf.vpin(bars_vol['BuyVolume'].values,
                         bars_vol['SellVolume'].values, window=vpin_window)

    round_frac, serial_corr, tick_acc_bar = [], [], []
    for bid, grp in trades.groupby('bar_id'):
        round_frac.append(mf.round_number_frequency(grp['Volume'].values)['round_fraction'])
        serial_corr.append(
            mf.serial_correlation_signed_flow(grp['Label'].values, lag=1)
            if len(grp) > 3 else np.nan
        )
        tick_acc_bar.append(
            mf.tick_rule_accuracy(mf.tick_rule(grp['Price'].values), grp['Label'].values)
        )

    feature_table = pd.DataFrame({
        'roll_c': roll_c,
        'roll_sigma_u': roll_sigma_u,
        'parkinson_vol_20bar': parkinson_roll,
        'corwin_schultz_spread': cs_spread_full.values,
        'becker_parkinson_sigma': bp_sigma_full.values,
        'kyle_lambda': kyle_full.values,
        'amihud_lambda_20bar': amihud_roll,
        'vpin_10bar': vpin_full.values,
        'round_number_fraction': round_frac,
        'serial_corr_signed_flow': serial_corr,
        'tick_rule_accuracy': tick_acc_bar,
    }, index=bars_vol.index)
    return feature_table


def build_enriched_events(raw_trades, threshold, events):
    """Full Phase 2c chain: raw trades -> bar-id-tagged trades -> bar
    volume aggregates -> re-derived frac-diff -> Ch19's 11 features ->
    join onto rebuild.py's triple-barrier events (Date-indexed already,
    so a direct reindex is enough -- no bar_id<->Date remapping layer
    needed here, unlike build_enriched_training_table.py's static-data
    version).

    Parameters
    ----------
    raw_trades : pd.DataFrame, EXACT same object passed to rebuild.py's
        build_bars_and_labels() for this pull (see CONSISTENCY
        REQUIREMENT in module docstring)
    threshold : float, EXACT same dynamic threshold rebuild.py used
        (result['threshold'] from build_bars_and_labels())
    events : pd.DataFrame, rebuild.py's result['events'] (Date-indexed
        t1/trgt/ret/bin)

    Returns
    -------
    dict with keys:
      'enriched_events' : pd.DataFrame, events joined with fracdiff +
        Ch19's 11 features, incomplete (warmup/NaN) rows dropped
      'fracdiff_d'       : float or None
      'n_events_before'  : int
      'n_events_after'   : int
      'feature_table'    : pd.DataFrame, the full bar-indexed feature
        table before the join (for inspection)
    """
    trades = _retag_trades_with_bar_id(raw_trades, threshold)
    bars_vol = _build_bars_with_volume(trades)

    fd = compute_fracdiff_feature(bars_vol['Close'])
    ch19_features = compute_ch19_features(trades, bars_vol)

    table = ch19_features.copy()
    if fd['d'] is not None:
        table = table.join(fd['fracdiff'], how='left')
    else:
        table['fracdiff'] = np.nan

    enriched = events.join(table, how='left')
    n_before = len(enriched)
    feature_cols = [c for c in table.columns]
    enriched = enriched.dropna(subset=feature_cols)
    n_after = len(enriched)

    return {
        'enriched_events': enriched,
        'fracdiff_d': fd['d'],
        'n_events_before': n_before,
        'n_events_after': n_after,
        'feature_table': table,
    }
