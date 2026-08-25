"""
pipeline/orchestration/test_trade_archive.py

Hand-traced TDD suite for trade_archive.py -- per this project's
standing convention, values below are hand-computed, not just asserted
against whatever the code happens to produce.
"""
import os
import tempfile

import pandas as pd
import pytest

from trade_archive import load_archive, append_to_archive, RAW_TRADE_COLUMNS


def _make_trades(trade_ids, timestamps_us=None):
    """Small synthetic trades DataFrame, real RAW_TRADE_COLUMNS schema.
    timestamps_us defaults to trade_ids themselves (strictly increasing,
    easy to hand-trace) if not given."""
    if timestamps_us is None:
        timestamps_us = trade_ids
    n = len(trade_ids)
    return pd.DataFrame({
        'TradeID': trade_ids,
        'Price': [100.0 + i for i in range(n)],
        'Volume': [1.0] * n,
        'QuoteVolume': [100.0 + i for i in range(n)],
        'Timestamp': timestamps_us,
        'IsBuyerMaker': [True] * n,
        'IsBestMatch': [True] * n,
    })


@pytest.fixture
def archive_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, 'archive.parquet')


class TestLoadArchive:
    def test_returns_empty_frame_with_correct_schema_when_missing(self, archive_path):
        result = load_archive(archive_path)
        assert len(result) == 0
        assert list(result.columns) == RAW_TRADE_COLUMNS

    def test_loads_previously_written_archive(self, archive_path):
        trades = _make_trades([1, 2, 3])
        append_to_archive(trades, archive_path)
        loaded = load_archive(archive_path)
        assert len(loaded) == 3
        assert set(loaded['TradeID']) == {1, 2, 3}


class TestAppendToArchive:
    def test_creates_new_archive_when_none_exists(self, archive_path):
        trades = _make_trades([1, 2, 3])
        result = append_to_archive(trades, archive_path)
        # hand-traced: 3 brand-new trades, 0 existing, 0 duplicates
        assert result['n_new_added'] == 3
        assert result['n_duplicates_skipped'] == 0
        assert result['n_total_after'] == 3
        assert os.path.exists(archive_path)

    def test_appends_non_overlapping_trades(self, archive_path):
        append_to_archive(_make_trades([1, 2, 3]), archive_path)
        result = append_to_archive(_make_trades([4, 5]), archive_path)
        # hand-traced: 3 existing + 2 brand-new, no overlap
        assert result['n_new_added'] == 2
        assert result['n_duplicates_skipped'] == 0
        assert result['n_total_after'] == 5

    def test_dedupes_overlapping_trades_by_trade_id(self, archive_path):
        # Pull A: IDs 100-104 (5 trades). Pull B: IDs 102-106 (5 trades,
        # 3 overlapping: 102, 103, 104). Hand-traced expected union:
        # {100,101,102,103,104,105,106} = 7 distinct trades.
        append_to_archive(_make_trades(list(range(100, 105))), archive_path)
        result = append_to_archive(_make_trades(list(range(102, 107))), archive_path)
        assert result['n_new_added'] == 2          # 105, 106 are the only new ones
        assert result['n_duplicates_skipped'] == 3  # 102, 103, 104 already present
        assert result['n_total_after'] == 7
        loaded = load_archive(archive_path)
        assert set(loaded['TradeID']) == set(range(100, 107))

    def test_fully_overlapping_pull_adds_nothing(self, archive_path):
        append_to_archive(_make_trades([1, 2, 3]), archive_path)
        result = append_to_archive(_make_trades([1, 2, 3]), archive_path)
        # hand-traced: identical pull re-run, e.g. a retried/duplicate
        # cron invocation -- must be a safe no-op, not an error or a
        # silent double-count
        assert result['n_new_added'] == 0
        assert result['n_duplicates_skipped'] == 3
        assert result['n_total_after'] == 3

    def test_sorts_by_timestamp_after_merge_even_if_appended_out_of_order(self, archive_path):
        # Second pull's timestamps are EARLIER than the first pull's
        # (e.g. backfilling older history after already archiving recent
        # data) -- hand-traced: final archive must still be chronological.
        append_to_archive(_make_trades([10, 11], timestamps_us=[500, 600]), archive_path)
        append_to_archive(_make_trades([8, 9], timestamps_us=[300, 400]), archive_path)
        loaded = load_archive(archive_path)
        assert list(loaded['Timestamp']) == [300, 400, 500, 600]
        assert list(loaded['TradeID']) == [8, 9, 10, 11]

    def test_span_days_hand_traced(self, archive_path):
        # 1,000,000 microseconds = 1 second. Trades 24h apart:
        # 24 * 3600 * 1_000_000 = 86,400,000,000 us difference -> 1.0 day.
        result = append_to_archive(
            _make_trades([1, 2], timestamps_us=[0, 86_400_000_000]), archive_path
        )
        assert result['span_days'] == pytest.approx(1.0)

    def test_raises_on_missing_required_columns(self, archive_path):
        bad_trades = pd.DataFrame({'TradeID': [1], 'Price': [100.0]})
        with pytest.raises(ValueError, match='missing required columns'):
            append_to_archive(bad_trades, archive_path)
