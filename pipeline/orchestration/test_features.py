"""
pipeline/orchestration/test_features.py

TDD test suite for features.py (Phase 2c: live frac-diff + Ch19
microstructural feature enrichment). Hand-traced synthetic values,
matching this project's TDD convention: for the two genuinely NEW
functions this module introduces (_retag_trades_with_bar_id,
_build_bars_with_volume) and the generic _rolling_bar helper, expected
values are hand-computed by the test author, not just shape-checked.

For compute_fracdiff_feature and build_enriched_events -- thin
orchestration wrappers around ALREADY-TESTED Ch05 (frac-diff) and Ch19
(11 microstructural features) real functions -- tests use monkeypatching
to isolate THIS module's own wiring/branching logic (does it handle
d=None correctly? does the join/dropna/count logic work?) rather than
re-deriving Ch05/Ch19's own formula correctness, which has its own test
suites already. This mirrors Phase 1's stages.py test philosophy for
reused Ch11 functions.

Run (two-pass, per project convention):
    From repo root:              pytest pipeline/orchestration/test_features.py -v
    From pipeline/orchestration: pytest test_features.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import features  # real module under test


# ---------------------------------------------------------------------
# _retag_trades_with_bar_id
# ---------------------------------------------------------------------

def _make_raw_trades(prices, volumes, is_buyer_maker, start_ts_us=1_000_000):
    """Small helper to build a raw_trades-shaped DataFrame matching
    ingestion.py's real RAW_TRADE_COLUMNS schema (only the columns
    _retag_trades_with_bar_id actually reads: Timestamp/Price/Volume/
    IsBuyerMaker), one microsecond apart so ordering is unambiguous."""
    n = len(prices)
    return pd.DataFrame({
        'TradeID': range(n),
        'Price': prices,
        'Volume': volumes,
        'Timestamp': [start_ts_us + i for i in range(n)],
        'IsBuyerMaker': is_buyer_maker,
    })


def test_retag_trades_with_bar_id_hand_traced():
    """threshold=100. Cumulative dollar volume:
    (10,3)->30, cumm=30
    (10,3)->30, cumm=60
    (10,5)->50, cumm=110 >= 100 -> bar_id=0 for THIS trade, then bar rolls to 1
    (10,2)->20, cumm=20
    (10,9)->90, cumm=110 >= 100 -> bar_id=1 for THIS trade, then bar rolls to 2
    (10,1)->10, cumm=10 (never crosses -- trailing incomplete bar)
    Expected bar_id sequence: [0,0,0,1,1,2]. n_complete_bars = max = 2,
    so the last row (bar_id=2) is dropped -- 5 rows survive."""
    raw = _make_raw_trades(
        prices=[10, 10, 10, 10, 10, 10],
        volumes=[3, 3, 5, 2, 9, 1],
        is_buyer_maker=[False, False, False, True, False, True],
    )
    trades = features._retag_trades_with_bar_id(raw, threshold=100)

    assert len(trades) == 5
    assert list(trades['bar_id']) == [0, 0, 0, 1, 1]
    assert 2 not in trades['bar_id'].values
    # Label derived from IsBuyerMaker: False -> +1 (buy), True -> -1 (sell)
    assert list(trades['Label']) == [1, 1, 1, -1, 1]
    assert trades['Date'].iloc[0] == pd.to_datetime(1_000_000, unit='us')


def test_retag_trades_with_bar_id_no_complete_bar_returns_empty():
    """If cumulative dollar volume never reaches threshold, bar_id stays
    0 the whole way through -> n_complete_bars = max(bar_id) = 0 ->
    trades[trades['bar_id'] < 0] is empty. A live pull with too-short a
    lookback could hit this (see rebuild.py's own get_daily_vol
    empty-window finding from the 2026-08-13 session)."""
    raw = _make_raw_trades(prices=[10, 10], volumes=[1, 1], is_buyer_maker=[False, False])
    trades = features._retag_trades_with_bar_id(raw, threshold=1_000_000)
    assert len(trades) == 0


# ---------------------------------------------------------------------
# _build_bars_with_volume
# ---------------------------------------------------------------------

def test_build_bars_with_volume_hand_traced():
    """Two bars, hand-traced:
    Bar 0: prices [10,12,8], volumes [3,3,5], labels [1,1,-1] (buy,buy,sell)
      Open=10, High=12, Low=8, Close=8, n_trades=3
      DollarVolume = 10*3+12*3+8*5 = 30+36+40 = 106
      BuyVolume = 3+3 = 6, SellVolume = 5
    Bar 1: prices [20,25], volumes [2,9], labels [-1,1] (sell,buy)
      Open=20, High=25, Low=20, Close=25, n_trades=2
      DollarVolume = 20*2+25*9 = 40+225 = 265
      BuyVolume = 9, SellVolume = 2
    """
    trades = pd.DataFrame({
        'Date': pd.to_datetime([1, 2, 3, 4, 5], unit='us'),
        'Price': [10, 12, 8, 20, 25],
        'Volume': [3, 3, 5, 2, 9],
        'Label': [1, 1, -1, -1, 1],
        'bar_id': [0, 0, 0, 1, 1],
    })
    bars = features._build_bars_with_volume(trades)

    assert len(bars) == 2
    bar0, bar1 = bars.iloc[0], bars.iloc[1]

    assert bar0['Open'] == 10
    assert bar0['High'] == 12
    assert bar0['Low'] == 8
    assert bar0['Close'] == 8
    assert bar0['n_trades'] == 3
    assert bar0['DollarVolume'] == 106
    assert bar0['BuyVolume'] == 6
    assert bar0['SellVolume'] == 5

    assert bar1['Open'] == 20
    assert bar1['High'] == 25
    assert bar1['Low'] == 20
    assert bar1['Close'] == 25
    assert bar1['n_trades'] == 2
    assert bar1['DollarVolume'] == 265
    assert bar1['BuyVolume'] == 9
    assert bar1['SellVolume'] == 2

    assert bars.index.name == 'Date'
    assert bars.index[0] == pd.to_datetime(3, unit='us')  # last trade in bar 0
    assert bars.index[1] == pd.to_datetime(5, unit='us')  # last trade in bar 1


def test_build_bars_with_volume_missing_side_fills_zero():
    """A bar with ONLY buy trades (no sell trades at all) must fill
    SellVolume with 0, not NaN or a missing column -- the reindex/fillna
    in the real code exists specifically for this case."""
    trades = pd.DataFrame({
        'Date': pd.to_datetime([1, 2], unit='us'),
        'Price': [10, 10],
        'Volume': [5, 5],
        'Label': [1, 1],   # both buys, no sells at all
        'bar_id': [0, 0],
    })
    bars = features._build_bars_with_volume(trades)
    assert bars.iloc[0]['BuyVolume'] == 10
    assert bars.iloc[0]['SellVolume'] == 0  # NOT NaN


# ---------------------------------------------------------------------
# _rolling_bar
# ---------------------------------------------------------------------

def test_rolling_bar_hand_traced():
    """a=[1,2,3,4,5], b=[10,20,30,40,50], window=3, func=sum(x)+sum(y).
    out[0],out[1] = nan (window-1=2 warmup rows).
    out[2] = (1+2+3)+(10+20+30) = 66
    out[3] = (2+3+4)+(20+30+40) = 99
    out[4] = (3+4+5)+(30+40+50) = 132
    """
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])
    out = features._rolling_bar(a, b, lambda x, y: x.sum() + y.sum(), window=3)

    assert np.isnan(out[0])
    assert np.isnan(out[1])
    assert out[2] == 66
    assert out[3] == 99
    assert out[4] == 132


def test_rolling_bar_exception_falls_back_to_nan():
    """If func raises for a given window, that position must become nan
    (not propagate the exception) -- the real code wraps each call in a
    try/except specifically for this.
    window=2: i=1 -> a[0:2]=[1,2], x[0]=1, no raise, sum=3
              i=2 -> a[1:3]=[2,3], x[0]=2, RAISES -> nan
              i=3 -> a[2:4]=[3,4], x[0]=3, no raise, sum=7
    """
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3, 4])

    def flaky(x, y):
        if x[0] == 2:   # raises on the window whose slice starts at index 1
            raise ValueError('boom')
        return x.sum()

    out = features._rolling_bar(a, b, flaky, window=2)
    assert np.isnan(out[0])          # warmup
    assert out[1] == 3               # 1+2, no raise
    assert np.isnan(out[2])          # flaky raised here
    assert out[3] == 7               # 3+4


# ---------------------------------------------------------------------
# compute_fracdiff_feature (monkeypatched isolation of Ch05 reuse)
# ---------------------------------------------------------------------

def test_compute_fracdiff_feature_d_none_path(monkeypatch):
    """When find_minimum_d returns None (no d in [0,1] passed ADF),
    compute_fracdiff_feature must surface that explicitly -- empty
    fracdiff Series, d=None, width=None -- NOT silently default to
    d=0.2 (see module's own LOAD-BEARING note on this exact point)."""
    sentinel_results = pd.DataFrame({'adfStat': [1.0, 2.0]})

    monkeypatch.setattr(features, 'find_min_ffd', lambda close, thres: sentinel_results)
    monkeypatch.setattr(features, 'find_minimum_d', lambda results: None)

    close = pd.Series([100.0, 101.0, 102.0])
    out = features.compute_fracdiff_feature(close)

    assert out['d'] is None
    assert out['width'] is None
    assert isinstance(out['fracdiff'], pd.Series)
    assert len(out['fracdiff']) == 0
    assert out['results'] is sentinel_results  # passed through untouched


def test_compute_fracdiff_feature_d_found_matches_real_ch05_functions(monkeypatch):
    """When a d IS found, compute_fracdiff_feature's output must match
    calling the REAL get_weights_ffd/frac_diff_ffd directly with that d
    -- this verifies the wrapper's own log-transform, width calc, and
    naming, WITHOUT re-deriving Ch05's own ADF-search correctness
    (already covered by Ch05's own test suite)."""
    fixed_d = 0.4
    monkeypatch.setattr(features, 'find_min_ffd', lambda close, thres: pd.DataFrame())
    monkeypatch.setattr(features, 'find_minimum_d', lambda results: fixed_d)

    close = pd.Series([100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0, 105.5])
    out = features.compute_fracdiff_feature(close, thres=features.FFD_THRES)

    expected_width = len(features.get_weights_ffd(fixed_d, thres=features.FFD_THRES)) - 1
    expected_fracdiff = features.frac_diff_ffd(np.log(close), d=fixed_d, thres=features.FFD_THRES)

    assert out['d'] == fixed_d
    assert out['width'] == expected_width
    assert out['fracdiff'].name == 'fracdiff'
    pd.testing.assert_series_equal(
        out['fracdiff'], expected_fracdiff, check_names=False,
    )


# ---------------------------------------------------------------------
# build_enriched_events (monkeypatched integration test)
# ---------------------------------------------------------------------

def test_build_enriched_events_join_and_dropna(monkeypatch):
    """Isolates build_enriched_events' OWN new logic (join alignment,
    dropna-based event survival, before/after counts, fracdiff_d
    propagation) by monkeypatching compute_ch19_features and
    compute_fracdiff_feature to return small FIXED tables with known
    NaN warmup rows -- NOT re-testing Ch19's 11 real feature formulas
    or Ch05's fracdiff math, which have their own test suites.

    _retag_trades_with_bar_id and _build_bars_with_volume (this
    module's genuinely NEW code) run for REAL against tiny synthetic
    trades, matching this project's real-code-where-possible principle.

    threshold=50, 9 trades of $30 each (price=10, volume=3):
    cumm: 30,60>=50(bar0->1,reset),30,60>=50(bar1->2,reset),
          30,60>=50(bar2->3,reset),30,60>=50(bar3->4,reset),30(trailing)
    bar_ids: [0,0,1,1,2,2,3,3,4] -> n_complete_bars=4, 8 rows survive,
    4 complete bars (0,1,2,3).
    """
    raw = _make_raw_trades(
        prices=[10] * 9,
        volumes=[3] * 9,
        is_buyer_maker=[False] * 9,
    )
    threshold = 50
    trades = features._retag_trades_with_bar_id(raw, threshold=threshold)
    bars_vol = features._build_bars_with_volume(trades)
    n_bars = len(bars_vol)
    assert n_bars == 4  # hand-traced above

    # Fake Ch19 feature table: first bar is NaN (simulated warmup), rest are 1.0
    fake_ch19 = pd.DataFrame(
        {'fake_feature': [np.nan] + [1.0] * (n_bars - 1)},
        index=bars_vol.index,
    )
    monkeypatch.setattr(features, 'compute_ch19_features', lambda trades, bars_vol: fake_ch19)

    fake_fracdiff = pd.Series(2.0, index=bars_vol.index, name='fracdiff')
    monkeypatch.setattr(
        features, 'compute_fracdiff_feature',
        lambda close, thres=features.FFD_THRES: {
            'fracdiff': fake_fracdiff, 'd': 0.3, 'width': 5, 'results': pd.DataFrame(),
        },
    )

    # events: one Date matching the (NaN) warmup bar, the rest matching
    # good bars, plus ONE Date with NO matching bar at all (must be dropped)
    event_dates = list(bars_vol.index) + [pd.Timestamp('2099-01-01')]
    events = pd.DataFrame(
        {'t1': event_dates, 'trgt': 0.01, 'ret': 0.0, 'bin': 1},
        index=pd.Index(event_dates, name='Date'),
    )

    out = features.build_enriched_events(raw, threshold, events)

    assert out['n_events_before'] == len(events)
    # survivors = all events EXCEPT the warmup-bar event and the no-match event
    assert out['n_events_after'] == n_bars - 1
    assert out['fracdiff_d'] == 0.3
    assert out['enriched_events']['fake_feature'].notna().all()
    assert (out['enriched_events']['fake_feature'] == 1.0).all()
    assert (out['enriched_events']['fracdiff'] == 2.0).all()
    assert bars_vol.index[0] not in out['enriched_events'].index
    assert pd.Timestamp('2099-01-01') not in out['enriched_events'].index
# =============================================================================
# TDD VERIFICATION -- pytest results, real-machine-confirmed 2026-08-14
# =============================================================================
# Two-pass run (per project convention):
#
# PASS 1 -- from repo root (pytest pipeline/orchestration/test_features.py -v):
#   test_retag_trades_with_bar_id_hand_traced PASSED
#   test_retag_trades_with_bar_id_no_complete_bar_returns_empty PASSED
#   test_build_bars_with_volume_hand_traced PASSED
#   test_build_bars_with_volume_missing_side_fills_zero PASSED
#   test_rolling_bar_hand_traced PASSED
#   test_rolling_bar_exception_falls_back_to_nan PASSED
#   test_compute_fracdiff_feature_d_none_path PASSED
#   test_compute_fracdiff_feature_d_found_matches_real_ch05_functions PASSED
#   test_build_enriched_events_join_and_dropna PASSED
#   9 passed in 2.01s
#
# PASS 2 -- from pipeline/orchestration/ (pytest test_features.py -v):
#   Same 9 tests, all PASSED, 9 passed in 1.57s
#
# One real bug found and fixed during test-writing: the ORIGINAL hand-trace
# for test_rolling_bar_exception_falls_back_to_nan mislabeled which loop
# index (i) the flaky() exception fires on, conflating it with the window
# slice's start index in `a`. Corrected trace: window=2, i=1 -> a[0:2]=[1,2]
# (x[0]=1, no raise, sum=3); i=2 -> a[1:3]=[2,3] (x[0]=2, RAISES -> nan);
# i=3 -> a[2:4]=[3,4] (x[0]=3, no raise, sum=7). This was a test-authoring
# error, not a bug in features.py's _rolling_bar -- the real code's
# try/except-per-window behavior was correct throughout.
# =============================================================================
