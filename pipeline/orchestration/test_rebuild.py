"""
pipeline/orchestration/test_rebuild.py

Tests for rebuild.py (Phase 2a: raw trades -> bars -> CUSUM events ->
triple-barrier labels -> sample weights, reusing Ch02/03/04's real
functions). Run against this project's existing real static raw trades
CSV as a stand-in for a live pull -- rebuild.py itself is data-source
agnostic (it only needs a DataFrame in RAW_TRADE_COLUMNS schema), so
testing it against the known static dataset is a strong structural check:
the results are compared against this project's already-established real
pipeline shape (249 bars / 88 events at the fixed $10,000 threshold), not
just checked for "doesn't crash".

ingestion.py (the actual Binance network calls) is NOT tested here -- this
environment cannot reach api.binance.com. See pipeline/README.md's Phase 2
section for what still needs real-machine, real-network testing.
"""
import os

import pandas as pd
import pytest

from rebuild import (
    compute_dynamic_threshold, preprocess_raw_trades, build_bars_and_labels,
    CUSUM_H, MIN_RET,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RAW_TRADES_PATH = os.path.join(ROOT, 'input_data', 'BTCTUSD-trades-2026-03.csv')

RAW_COLUMNS = ['TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
               'IsBuyerMaker', 'IsBestMatch']


@pytest.fixture(scope='module')
def raw_trades():
    return pd.read_csv(RAW_TRADES_PATH, header=None, names=RAW_COLUMNS)


class TestComputeDynamicThreshold:

    def test_close_to_established_fixed_threshold(self, raw_trades):
        # This project's established static-data convention is a fixed
        # $10,000 threshold, chosen to produce ~250 bars on this exact
        # dataset. The dynamic formula targeting 250 bars on the SAME
        # dataset should land in the same ballpark, not an order of
        # magnitude off -- a sanity check that the formula (total dollar
        # volume / target_bars) is doing something sensible, not that it
        # exactly reproduces $10,000 (it won't, and shouldn't have to).
        thresh = compute_dynamic_threshold(raw_trades, target_bars=250)
        assert 5000 < thresh < 20000

    def test_scales_with_target_bars(self, raw_trades):
        thresh_250 = compute_dynamic_threshold(raw_trades, target_bars=250)
        thresh_500 = compute_dynamic_threshold(raw_trades, target_bars=500)
        # doubling target_bars should roughly halve the threshold (same
        # total dollar volume, twice as many bars)
        assert thresh_500 == pytest.approx(thresh_250 / 2, rel=0.01)

    def test_rejects_zero_volume(self):
        empty = pd.DataFrame({'Price': [0.0], 'Volume': [0.0]})
        with pytest.raises(ValueError):
            compute_dynamic_threshold(empty)


class TestPreprocessRawTrades:

    def test_adds_expected_columns(self, raw_trades):
        df = preprocess_raw_trades(raw_trades)
        for col in ('Date', 'Price', 'Volume', 'Label', 'Dollar', 'Delta'):
            assert col in df.columns

    def test_label_matches_is_buyer_maker(self, raw_trades):
        df = preprocess_raw_trades(raw_trades)
        expected = raw_trades['IsBuyerMaker'].apply(lambda x: -1 if x else 1)
        assert (df['Label'].values == expected.values).all()

    def test_row_count_preserved(self, raw_trades):
        df = preprocess_raw_trades(raw_trades)
        assert len(df) == len(raw_trades)


class TestBuildBarsAndLabels:

    def test_shape_close_to_established_real_pipeline(self, raw_trades):
        # Established real numbers on this exact dataset at the fixed
        # $10,000 threshold: 249 bars, 88 final labeled events. The
        # dynamic threshold here lands at ~$10,900 (see
        # TestComputeDynamicThreshold), so exact equality isn't expected
        # -- but the result should be in the same regime, not off by an
        # order of magnitude, which would indicate a real bug in the
        # rebuild chain rather than expected threshold-driven variation.
        result = build_bars_and_labels(raw_trades, target_bars=250)
        assert 150 < len(result['bars']) < 300
        assert 50 < len(result['events']) < 120

    def test_events_schema_matches_ch03_events_csv(self, raw_trades):
        result = build_bars_and_labels(raw_trades, target_bars=250)
        assert list(result['events'].columns) == ['t1', 'trgt', 'ret', 'bin']
        assert set(result['events']['bin'].unique()) <= {-1.0, 0.0, 1.0}

    def test_w_and_tw_aligned_with_events_no_nan(self, raw_trades):
        result = build_bars_and_labels(raw_trades, target_bars=250)
        assert result['w'].index.equals(result['events'].index)
        assert result['tw'].index.equals(result['events'].index)
        assert not result['w'].isna().any()
        assert not result['tw'].isna().any()

    def test_t1_strictly_after_event_start(self, raw_trades):
        result = build_bars_and_labels(raw_trades, target_bars=250)
        events = result['events']
        assert (pd.to_datetime(events['t1']).values > events.index.values).all()

    def test_close_prices_positive(self, raw_trades):
        result = build_bars_and_labels(raw_trades, target_bars=250)
        assert (result['close'] > 0).all()


# ---------------------------------------------------------------------------
# TDD results -- SANDBOX pre-check only, NOT YET real-machine confirmed.
# Sandbox: Python 3.12.3, pandas 3.0.2, numpy 2.4.4.
#
# Real numbers observed in sandbox against the established static dataset
# (target_bars=250):
#   dynamic threshold: $10,903.01 (established fixed value: $10,000)
#   bars: 230 (established: 249)
#   events: 84 (established: 88)
#   w/tw: both length 84, fully aligned with events.index, zero NaN
#
# A real bug was caught and fixed during this sandbox pass: an earlier
# draft passed the post-get_bins `events` frame (ret/bin only, no t1) to
# get_sample_weights/get_average_uniqueness instead of tb_events (which
# has t1) -- raised a real KeyError('t1'). Fixed by using tb_events for
# those two calls and explicitly reindexing w/tw to the final events index
# (robust to live-pull edge cases the static dataset didn't exercise --
# see rebuild.py's own comments).
#
# STILL NEEDED before this is real-machine confirmed:
#   conda activate mlfinlab
#   cd C:\ws\AFML
#   python -m pytest pipeline\orchestration\test_rebuild.py -v
#   cd pipeline\orchestration
#   python -m pytest test_rebuild.py -v
# ---------------------------------------------------------------------------
