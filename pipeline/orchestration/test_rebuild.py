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

    def test_disambiguates_duplicate_timestamps(self, raw_trades):
        # LOAD-BEARING (2026-08-21): regression guard for the real bug
        # found while testing target_bars=1000 for production adoption --
        # the static CSV has 561 duplicate raw Timestamp values out of
        # 9,205, which (at a fine enough bar granularity) can produce two
        # bars sharing one Date index value and crash
        # triple_barrier.get_daily_vol(). preprocess_raw_trades() must make
        # Timestamp -- and therefore the derived Date column -- unique.
        assert raw_trades['Timestamp'].duplicated().sum() > 0  # sanity:
            # confirms the input fixture actually exercises this case
        df = preprocess_raw_trades(raw_trades)
        assert not df['Date'].duplicated().any()


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

    def test_target_bars_1000_does_not_crash(self, raw_trades):
        # LOAD-BEARING (2026-08-21): regression guard for the real bug
        # this exact call surfaced during production adoption -- target_bars
        # =1000 is now this pipeline's live default (see
        # run_pipeline_live.py's own LOAD-BEARING note), and previously
        # crashed on this static dataset with a real ValueError from
        # triple_barrier.get_daily_vol() (duplicate bar-close timestamps,
        # fixed via preprocess_raw_trades' own timestamp disambiguation --
        # see TestPreprocessRawTrades.test_disambiguates_duplicate_timestamps).
        # Confirmed real numbers on this dataset: 789 bars, 171 events.
        result = build_bars_and_labels(raw_trades, target_bars=1000)
        assert len(result['bars']) > 0
        assert not result['bars'].index.duplicated().any()


class TestCusumHStalenessCorrection:

    def test_cusum_h_is_staleness_corrected_value(self):
        # LOAD-BEARING (2026-08-21): regression guard against an accidental
        # revert to the pre-staleness-correction value. CUSUM_H was 500
        # (the original March 2026 calibration) until this date; changed to
        # 313 per real measurement (CALIBRATION_AUDIT.md's "CUSUM_H
        # Staleness Audit" section). This locks in TODAY's measured value,
        # not a claim that 313 is permanent -- a future staleness
        # re-measurement may find a different value appropriate, and should
        # update this assertion deliberately, not silently.
        assert CUSUM_H == 313


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
# ---------------------------------------------------------------------------
# 2026-08-21 UPDATE -- production adoption of CUSUM_H=313/target_bars=1000,
# sandbox pre-check only (same environment as above), full suite re-run:
#
#   14 passed in 2.10s
#
#   TestComputeDynamicThreshold::test_close_to_established_fixed_threshold PASSED
#   TestComputeDynamicThreshold::test_scales_with_target_bars PASSED
#   TestComputeDynamicThreshold::test_rejects_zero_volume PASSED
#   TestPreprocessRawTrades::test_adds_expected_columns PASSED
#   TestPreprocessRawTrades::test_label_matches_is_buyer_maker PASSED
#   TestPreprocessRawTrades::test_row_count_preserved PASSED
#   TestPreprocessRawTrades::test_disambiguates_duplicate_timestamps PASSED  [NEW]
#   TestBuildBarsAndLabels::test_shape_close_to_established_real_pipeline PASSED
#   TestBuildBarsAndLabels::test_events_schema_matches_ch03_events_csv PASSED
#   TestBuildBarsAndLabels::test_w_and_tw_aligned_with_events_no_nan PASSED
#   TestBuildBarsAndLabels::test_t1_strictly_after_event_start PASSED
#   TestBuildBarsAndLabels::test_close_prices_positive PASSED
#   TestBuildBarsAndLabels::test_target_bars_1000_does_not_crash PASSED  [NEW]
#   TestCusumHStalenessCorrection::test_cusum_h_is_staleness_corrected_value PASSED  [NEW]
#
# A real, previously-undiscovered bug was caught and fixed during this
# pass: target_bars=1000 crashed on the static dataset with a real
# ValueError from triple_barrier.get_daily_vol() (duplicate bar-close
# timestamps -- the static CSV has 561 duplicate raw Timestamp values out
# of 9,205, which at this bar granularity occasionally collide on a bar
# close). Root-caused by direct inspection (confirmed 4 duplicate bar
# index entries at target_bars=1000, matching the exact 757-vs-753 shape
# mismatch). Fixed by reusing ingestion.py's already-tested
# _disambiguate_timestamps() inside preprocess_raw_trades() -- confirmed
# in sandbox to resolve target_bars=1000 (789 bars, 171 events) with ZERO
# change to the already-passing target_bars=250/500 results. The live
# pipeline itself was never at risk from this specific bug (ingestion.py
# already disambiguates live pulls) -- this closes the same gap for any
# OTHER raw_trades source, matching this module's stated data-source-
# agnostic intent.
#
# STILL NEEDED before this is real-machine confirmed:
#   conda activate mlfinlab
#   cd C:\ws\AFML
#   python -m pytest pipeline\orchestration\test_rebuild.py -v
#   cd pipeline\orchestration
#   python -m pytest test_rebuild.py -v
# THEN a fresh live run under the new defaults to confirm end-to-end on
# real Binance.US data, not just the static dataset:
#   cd C:\ws\AFML
#   python pipeline\run_pipeline_live.py --snapshot
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# REAL-MACHINE CONFIRMED (2026-08-21): mlfinlab conda env, Python 3.10.20,
# pandas 1.5.3, numpy 1.23.5, Windows 11.
#
# Two-pass pytest, both directions, both 14/14 passed (including both new
# regression tests -- test_disambiguates_duplicate_timestamps and
# test_target_bars_1000_does_not_crash):
#
#   From repo root (python -m pytest pipeline\orchestration\test_rebuild.py -v):
#     14 passed in 3.44s
#   From pipeline\orchestration (python -m pytest test_rebuild.py -v):
#     14 passed in 3.06s
#
# Real-machine results match the sandbox pre-check exactly -- no
# environment-specific surprises. The target_bars=1000 duplicate-timestamp
# bug and its fix are confirmed real, not a sandbox artifact.
#
# STILL NEEDED: a fresh live run under the new run_pipeline_live.py
# defaults (target_bars=1000, via rebuild.py's now-active CUSUM_H=313) to
# confirm end-to-end on real, current Binance.US data:
#   cd C:\ws\AFML
#   python pipeline\run_pipeline_live.py --snapshot
# ---------------------------------------------------------------------------