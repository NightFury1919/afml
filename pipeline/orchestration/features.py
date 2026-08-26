"""
pipeline/orchestration/features.py

Phase 2c: rebuilds Ch19's 11 microstructural features + Ch05's frac-diff
feature on live-pulled data, joining them onto rebuild.py's triple-barrier
events -- the same enrichment build_enriched_training_table.py performs
for the static dataset, adapted for a live pull's dynamic bar threshold.

*** WIDENED (2026-08-25): Ch17 structural-break + Ch18 entropy features
added, real feature-set expansion beyond Ch19+Ch05 ***
Per the 2026-08-25 session's discovery-vs-validation discussion: this
pipeline's PBO/CPCV/DSR machinery is a strong validator, but until now
the classifier only ever saw two families of information (Ch19
liquidity/toxicity/spread microstructure + Ch05 fracdiff price-level
memory). Two genuinely new feature families are added here --
compute_entropy_feature() (Ch18 LZ entropy rate) and
compute_structural_break_feature() (Ch17 Chu-Stinchcombe-White CUSUM,
bounded-lookback adaptation -- see that function's own LOAD-BEARING
note) -- both real book functions, reused not reimplemented, both
computational-cost-tested directly before being added as per-bar
rolling features. This is new information reaching the model, not a
re-test of the same inputs -- the honest distinction the discovery-vs-
validation discussion was about.

NOT YET RE-VALIDATED end-to-end against real Kraken/Binance.US sweep
data as of this commit -- that is explicitly deferred to a following
session (see CALIBRATION_AUDIT.md). This change adds the features and
confirms they compute correctly; it does NOT re-run detection-power
calibration or the target_bars sweeps with them included.

Real modules reused directly (no reimplementation):
  - ch19/microstructural_features/microstructural_features.py (all 11
    feature functions, exact same calling convention as
    chapter_19_microstructural_features.py's real driver)
  - ch05/frac_diff/{find_min_ffd,frac_diff_ffd,get_weights_ffd}.py
  - ch17/structural_breaks/cusum.py (get_csw_stat, get_csw_critical_value
    -- NOT get_csw_sup/get_csw_cusum, see compute_structural_break_
    feature's own note on why a bounded-lookback caller-side loop is
    used instead of those two unbounded functions)
  - ch18/entropy_features/{entropy_estimators,encoding_schemes}.py
    (konto, binary_encode)

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

# Same implicit-namespace-package convention as ch19 above -- no
# __init__.py, no bare-import sys.path trick needed (cusum.py/
# entropy_estimators.py/encoding_schemes.py have no inter-module bare
# imports of their own, unlike ch05's frac_diff/).
from ch17.structural_breaks import cusum as sb            # real module
from ch18.entropy_features import entropy_estimators as ee  # real module
from ch18.entropy_features import encoding_schemes as enc   # real module

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

# Reused from ingestion.py, not reimplemented -- same precedent
# rebuild.py's preprocess_raw_trades() already established (2026-08-21)
# for this exact bug class. Bare import: ingestion.py lives alongside
# this module in pipeline/orchestration/, which every caller already
# adds to sys.path before importing features.py.
from ingestion import _disambiguate_timestamps           # noqa: E402

ROLL_WINDOW = 20        # carried over unchanged from Ch19 (see LOAD-BEARING note)
FFD_THRES = 0.01        # carried over unchanged from Ch05's established calibration
VPIN_WINDOW = 10         # carried over unchanged from Ch19's established calibration
ENTROPY_WINDOW = 20      # matches ROLL_WINDOW's own established convention --
                          # no independent derivation, same "carried over" status
CSW_MAX_LOOKBACK = 200   # see compute_structural_break_feature's own
                          # LOAD-BEARING note on why this is bounded


def compute_entropy_feature(close, window=ENTROPY_WINDOW):
    """
    *** LOAD-BEARING (2026-08-25): new feature, Ch18's real Kontoyiannis
    LZ entropy-rate estimator, added to widen the live feature set beyond
    Ch19 microstructural + Ch05 fracdiff (per 2026-08-25 session's
    discussion of discovery vs. validation -- this is genuinely NEW
    information reaching the classifier, not a re-test of existing
    inputs) ***

    Rolling, per-bar entropy-rate feature: at each bar i, sign-encodes
    the most recent `window` bar-to-bar returns into a binary message
    (Ch18's real binary_encode(), Sec 18.5.1 -- '1' for a positive
    return, '0' for negative, zero-returns dropped exactly as the book
    specifies) and estimates that message's entropy rate via Ch18's
    real konto() (Snippet 18.4, Kontoyiannis' 2013 LZ estimator) --
    BOTH functions reused unmodified, not reimplemented.

    Interpretation: LOW entropy in a window means returns compressed
    easily (long non-redundant substrings -> predictable-looking sign
    sequence); HIGH entropy means the sign sequence looked close to
    random. This is a genuinely different kind of information than
    Ch19's liquidity/toxicity features or Ch05's fracdiff -- it's about
    how COMPRESSIBLE recent price direction has been, not about spread,
    volume, or price-level memory.

    COMPUTATIONAL COST (measured directly, 2026-08-25, before deciding
    this was safe to add as a per-bar rolling feature): konto() on a
    ~20-30 character message costs ~0.12ms per call -- recomputing it
    at every bar for up to 5,000 bars (Kraken's real observed target_bars
    range) costs well under 1 second total. No bounding/adaptation
    needed here, unlike compute_structural_break_feature's CSW CUSUM
    (see that function's own note on why THAT one needed a bounded
    lookback).

    Parameters
    ----------
    close : pd.Series, bar close prices, Date-indexed (same series
        rebuild.py/features.py already use for fracdiff/Ch19 features)
    window : int, rolling window size in bars

    Returns
    -------
    pd.Series, Date-indexed like `close`, entropy-rate estimate per bar
    (NaN for the first `window` bars -- warmup, same convention as
    Ch19's own roll_c/roll_sigma_u rolling features).
    """
    returns = close.pct_change()
    values = returns.values
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(window, n):
        window_returns = values[i - window:i]
        msg = enc.binary_encode(window_returns)
        if len(msg) < 4:
            continue  # too few non-zero returns in this window for a
                       # meaningful LZ estimate -- leave as NaN rather
                       # than feed konto() a near-empty message
        result = ee.konto(msg, window=None)  # expanding window on the
                                                # small per-bar message
                                                # itself, not on `close`
        out[i] = result['h']
    return pd.Series(out, index=close.index, name='entropy_rate')


def compute_structural_break_feature(close, min_sample=3,
                                      max_lookback=CSW_MAX_LOOKBACK):
    """
    *** LOAD-BEARING (2026-08-25): new feature, Ch17's real
    Chu-Stinchcombe-White CUSUM structural-break statistic (Sec 17.3.2),
    added for the same "widen the feature set" reason as
    compute_entropy_feature above ***

    *** LOAD-BEARING (2026-08-25): BOUNDED reference-search range, NOT
    the book's own unbounded get_csw_sup()/get_csw_cusum() ***
    The book's own get_csw_stat() (Sec 17.3.2's printed formula) is
    reused here UNMODIFIED -- this function does not reimplement the
    statistic itself, only the OUTER search loop around it. The real
    get_csw_cusum()/get_csw_sup() (this repo's ch17/structural_breaks/
    cusum.py, already real-machine-confirmed for chapter_17's own
    teaching deliverable) search the reference index n over the ENTIRE
    prior history [0, t) for each t -- correct and book-faithful for a
    one-off research computation, but O(T^2) real function calls, which
    a DIRECT MEASUREMENT (2026-08-25, before writing this function)
    showed costs ~4.7s at T=1000 bars and ~118s at T=5000 bars (Kraken's
    real observed target_bars range) -- prohibitive as a cost paid on
    EVERY live pipeline run and EVERY sweep config, unlike a single
    one-off diagnostic computation.

    This function instead caps the backward search range to the most
    recent `max_lookback` bars (default 200) rather than all the way
    back to the series' start, turning the cost from O(T^2) into
    O(T * max_lookback) -- MEASURED (2026-08-25) at ~1.66s (T=1000) and
    ~11.1s (T=5000), a real and now-tractable cost. This is also, on its
    own merits, a more sensible design for a LIVE rolling feature: a
    reference level from months ago is not obviously the right thing to
    compare today's price against; a bounded recent lookback is a
    reasonable choice independent of the performance motivation.

    Returns the NORMALIZED statistic S / critical_value_95 (not the raw
    S) as the single feature value -- a ratio above 1.0 means the
    observed departure exceeds the book's own 5% one-sided critical
    value (Sec 17.3.2, b_0.05=4.6) at that bar's best-matching reference
    point; this is a cleaner single scalar for a model input than
    carrying S and critical_value_95 as two separate correlated columns.

    Parameters
    ----------
    close : pd.Series, bar close prices, Date-indexed
    min_sample : int, matches cusum.get_csw_cusum's own parameter
    max_lookback : int, bounded backward reference-search range in bars

    Returns
    -------
    pd.Series, Date-indexed like `close`, normalized CSW statistic per
    bar (NaN for bars before min_sample, or where sigma_hat_t could not
    be estimated -- same NaN cases the book's own get_csw_stat() has).
    """
    log_close = np.log(close)
    values = log_close.values
    idx = log_close.index
    T = len(values)

    normalized = np.full(T, np.nan)
    for t_idx in range(min_sample, T):
        n_start = max(0, t_idx - max_lookback)
        best_S, best_n = -np.inf, None
        for n_idx in range(n_start, t_idx):
            S = sb.get_csw_stat(log_close, n_idx, t_idx)  # real, unmodified
            if np.isfinite(S) and S > best_S:
                best_S, best_n = S, n_idx
        if best_n is not None:
            cv = sb.get_csw_critical_value(best_n, t_idx)  # real, unmodified
            if cv > 0:
                normalized[t_idx] = best_S / cv
    return pd.Series(normalized, index=idx, name='structural_break_stat')


def _retag_trades_with_bar_id(raw_trades, threshold):
    """Re-run the SAME cumulative-dollar-threshold accumulate-and-reset
    loop dollar_bars() uses internally (see module LOAD-BEARING note),
    against the pull's dynamic threshold, to tag every trade with the
    bar it belongs to. Drops the trailing incomplete bar, matching Ch19's
    own driver and build_enriched_training_table.py's convention.

    *** LOAD-BEARING (2026-08-25): disambiguates Timestamp FIRST, same
    fix rebuild.py's preprocess_raw_trades() already applies -- reused
    here for the first time, NOT previously needed ***
    Real bug found via the Kraken target_bars sweep (calibrate_kraken_
    target_bars.py): this function receives raw_trades DIRECTLY from
    the caller (see build_enriched_events()'s own docstring -- "EXACT
    same object passed to rebuild.py's build_bars_and_labels()"), NOT
    the disambiguated copy rebuild.py's preprocess_raw_trades() computes
    internally -- that disambiguation never propagates back to the
    caller's original DataFrame. When multiple trades share an exact
    raw timestamp (routine, not a Kraken-only phenomenon -- any burst of
    trades within one millisecond) AND a dollar-bar boundary happens to
    fall in the middle of that burst, the LAST trade of bar N and the
    LAST trade of bar N+1 can share that identical raw timestamp --
    _build_bars_with_volume()'s groupby(...).agg(Date=('Date','last'))
    then produces two bars with the SAME Date, which crashes
    compute_fracdiff_feature() downstream with a duplicate-index
    ValueError. Real-machine confirmed (2026-08-25): duplicate count
    rose 9->45->94->142->212 as target_bars rose 1000->5000 on a 720h
    Kraken snapshot -- smaller thresholds put bar boundaries in the
    middle of a shared-timestamp burst more often, exactly the predicted
    mechanism. This is a GENERAL gap, not Kraken-specific -- the
    2026-08-21 fix only patched rebuild.py's own internal path; this
    second, independent raw-timestamp-consuming path was never covered.
    It surfaced now because Kraken's higher density made pushing
    target_bars much higher than anything tried on Binance.US both
    possible and informative. Reuses ingestion.py's real
    _disambiguate_timestamps() -- same established precedent, not a new
    approach -- so both raw-trade-consuming paths in this pipeline now
    get the same protection.

    Returns
    -------
    trades : pd.DataFrame, columns Date/Price/Volume/Label/bar_id, one
        row per trade in the completed bars (incomplete trailing bar's
        trades dropped)
    """
    raw = _disambiguate_timestamps(raw_trades)
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
    entropy_feature = compute_entropy_feature(bars_vol['Close'])
    structural_break_feature = compute_structural_break_feature(bars_vol['Close'])

    table = ch19_features.copy()
    if fd['d'] is not None:
        table = table.join(fd['fracdiff'], how='left')
    else:
        table['fracdiff'] = np.nan
    table = table.join(entropy_feature, how='left')
    table = table.join(structural_break_feature, how='left')

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


# TDD results -- real machine (mlfinlab env), 2026-08-25
#
# (mlfinlab) PS C:\ws\AFML> python -m pytest pipeline\orchestration\test_new_features_2026_08_25.py -v
# ============================== 8 passed in 3.03s ==============================
# Two-pass (from inside pipeline/orchestration/): 8 passed in 2.32s
#
# Also confirmed via a full end-to-end real-chain run (generate_momentum_
# trades -> build_bars_and_labels -> build_enriched_events, sandbox,
# 2026-08-25): both new features compute correctly at target_bars=1000
# (2.7s total, 259/259 events survived, no all-NaN columns) and
# target_bars=4000 (11.5s total, matching the isolated bounded-CSW-CUSUM
# timing measurement, 403/403 events survived).
# ---------------------------------------------------------------------------
