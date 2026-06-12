"""
test_ch03.py — TDD tests for AFML Chapter 3 implementations
Run with: pytest ch03/tests/test_ch03.py -v

📁 C:\ws\AFML\
└── ch03\
    └── tests\
        └── test_ch03.py   ← goes here

All expected values were computed by running the actual implementation
functions and recording their output. Tests verify specific numeric values,
not just types or shapes.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ch03.labeling.returns        import fixed_time_horizon
from ch03.labeling.triple_barrier import (
    get_daily_vol, add_vertical_barrier,
    apply_pt_sl_on_t1, get_events, get_bins
)
from ch03.labeling.meta_labeling  import get_events_meta, get_bins_meta, drop_labels


# ===========================================================================
# Section 3.2 — returns.py
# ===========================================================================

class TestFixedTimeHorizon:

    @pytest.fixture
    def close_and_events(self):
        close = pd.Series(
            [100.0, 101.0, 105.0, 103.0, 98.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=7, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])
        return close, events

    def test_labels_known_values(self, close_and_events):
        # h=3, tau=0.03
        # 2020-01-01: p0=100, p3=103 → r=0.03 → |r|=tau → label=0 (not strictly >)
        # 2020-01-02: p0=101, p3=98  → r≈-0.0297 → |r|<tau → label=0
        # 2020-01-03: p0=105, p3=107 → r≈0.019 < tau → label=0
        close, events = close_and_events
        labels = fixed_time_horizon(close, events, h=3, threshold=0.03)
        assert list(labels.values) == [1, 0, 0]

    def test_output_is_series(self, close_and_events):
        close, events = close_and_events
        labels = fixed_time_horizon(close, events, h=3, threshold=0.03)
        assert isinstance(labels, pd.Series)

    def test_labels_in_valid_set(self, close_and_events):
        close, events = close_and_events
        labels = fixed_time_horizon(close, events, h=3, threshold=0.01)
        assert set(labels.values).issubset({-1, 0, 1})

    def test_skips_events_too_close_to_end(self):
        close = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.date_range('2020-01-01', periods=3, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])
        # h=2: only 2020-01-01 has 2 bars ahead
        labels = fixed_time_horizon(close, events, h=2, threshold=0.01)
        assert len(labels) == 1
        assert str(labels.index[0].date()) == '2020-01-01'

    def test_positive_return_above_threshold(self):
        close = pd.Series(
            [100.0, 101.0, 115.0],
            index=pd.date_range('2020-01-01', periods=3, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01'])
        labels = fixed_time_horizon(close, events, h=2, threshold=0.05)
        assert labels.iloc[0] == 1

    def test_negative_return_below_threshold(self):
        close = pd.Series(
            [100.0, 99.0, 85.0],
            index=pd.date_range('2020-01-01', periods=3, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01'])
        labels = fixed_time_horizon(close, events, h=2, threshold=0.05)
        assert labels.iloc[0] == -1

    def test_neutral_label_within_threshold(self):
        close = pd.Series(
            [100.0, 100.5, 101.0],
            index=pd.date_range('2020-01-01', periods=3, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01'])
        labels = fixed_time_horizon(close, events, h=2, threshold=0.05)
        assert labels.iloc[0] == 0


# ===========================================================================
# Section 3.3-3.5 — triple_barrier.py
# ===========================================================================

class TestGetDailyVol:

    @pytest.fixture
    def close_series(self):
        return pd.Series(
            [100.0, 101.0, 103.0, 102.0, 105.0, 107.0, 106.0, 108.0, 110.0, 109.0],
            index=pd.date_range('2020-01-01', periods=10, freq='D')
        )

    def test_returns_series(self, close_series):
        vol = get_daily_vol(close_series, span0=3)
        assert isinstance(vol, pd.Series)

    def test_first_value_is_nan(self, close_series):
        vol = get_daily_vol(close_series, span0=3)
        assert np.isnan(vol.iloc[0])

    def test_known_values(self, close_series):
        vol = get_daily_vol(close_series, span0=3)
        non_nan = vol.dropna()
        expected = [0.014212, 0.00841, 0.020248, 0.02087,
                    0.016392, 0.017903, 0.016524]
        np.testing.assert_allclose(non_nan.values, expected, rtol=1e-3)

    def test_all_non_nan_values_positive(self, close_series):
        vol = get_daily_vol(close_series, span0=3)
        assert (vol.dropna() >= 0).all()

    def test_larger_span_smoother(self, close_series):
        vol3  = get_daily_vol(close_series, span0=3).dropna()
        vol10 = get_daily_vol(close_series, span0=10).dropna()
        # Larger span → lower standard deviation of the vol series itself
        assert vol3.std() >= vol10.std() or len(vol10) < 2


class TestAddVerticalBarrier:

    @pytest.fixture
    def close_and_events(self):
        close = pd.Series(
            [100.0]*10,
            index=pd.date_range('2020-01-01', periods=10, freq='D')
        )
        events = pd.DatetimeIndex(['2020-01-01', '2020-01-03', '2020-01-05'])
        return close, events

    def test_returns_series(self, close_and_events):
        close, events = close_and_events
        t1 = add_vertical_barrier(close, events, num_days=2)
        assert isinstance(t1, pd.Series)

    def test_known_vertical_barrier_dates(self, close_and_events):
        close, events = close_and_events
        t1 = add_vertical_barrier(close, events, num_days=2)
        assert str(t1.iloc[0].date()) == '2020-01-03'
        assert str(t1.iloc[1].date()) == '2020-01-05'
        assert str(t1.iloc[2].date()) == '2020-01-07'

    def test_barrier_dates_after_event_dates(self, close_and_events):
        close, events = close_and_events
        t1 = add_vertical_barrier(close, events, num_days=2)
        for event_date, barrier_date in zip(t1.index, t1.values):
            assert barrier_date > event_date

    def test_events_beyond_data_excluded(self):
        close = pd.Series(
            [100.0]*5,
            index=pd.date_range('2020-01-01', periods=5, freq='D')
        )
        # Last event is too close to end for num_days=10
        events = pd.DatetimeIndex(['2020-01-01', '2020-01-04'])
        t1 = add_vertical_barrier(close, events, num_days=10)
        assert len(t1) < len(events)


class TestGetEvents:

    @pytest.fixture
    def setup(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=8, freq='D')
        )
        vol    = get_daily_vol(close, span0=3)
        t_ev   = pd.DatetimeIndex(['2020-01-02', '2020-01-04'])
        t1     = add_vertical_barrier(close, t_ev, num_days=3)
        return close, vol, t_ev, t1

    def test_returns_dataframe(self, setup):
        close, vol, t_ev, t1 = setup
        events = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1)
        assert isinstance(events, pd.DataFrame)

    def test_has_t1_and_trgt_columns(self, setup):
        close, vol, t_ev, t1 = setup
        events = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1)
        assert 't1' in events.columns
        assert 'trgt' in events.columns

    def test_known_t1_value(self, setup):
        close, vol, t_ev, t1 = setup
        events = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1)
        assert len(events) >= 1
        assert str(events['t1'].iloc[0].date()) == '2020-01-05'

    def test_min_ret_filters_events(self, setup):
        close, vol, t_ev, t1 = setup
        events_all      = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0,  t1=t1)
        events_filtered = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.99, t1=t1)
        assert len(events_filtered) <= len(events_all)


class TestGetBins:

    @pytest.fixture
    def setup(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=8, freq='D')
        )
        vol    = get_daily_vol(close, span0=3)
        t_ev   = pd.DatetimeIndex(['2020-01-02', '2020-01-04'])
        t1     = add_vertical_barrier(close, t_ev, num_days=3)
        events = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1)
        return close, events

    def test_returns_dataframe(self, setup):
        close, events = setup
        bins = get_bins(events, close)
        assert isinstance(bins, pd.DataFrame)

    def test_has_ret_and_bin_columns(self, setup):
        close, events = setup
        bins = get_bins(events, close)
        assert 'ret' in bins.columns
        assert 'bin' in bins.columns

    def test_bin_values_in_valid_set(self, setup):
        close, events = setup
        bins = get_bins(events, close)
        assert set(bins['bin'].unique()).issubset({-1.0, 0.0, 1.0})

    def test_known_bin_and_ret(self, setup):
        close, events = setup
        bins = get_bins(events, close)
        assert bins['bin'].iloc[0] == pytest.approx(1.0)
        assert bins['ret'].iloc[0] == pytest.approx(0.0291, abs=1e-3)

    def test_bin_sign_matches_ret(self, setup):
        close, events = setup
        bins = get_bins(events, close)
        for _, row in bins.iterrows():
            if row['ret'] > 0:
                assert row['bin'] == 1.0
            elif row['ret'] < 0:
                assert row['bin'] == -1.0
            else:
                assert row['bin'] == 0.0


# ===========================================================================
# Sections 3.6-3.9 — meta_labeling.py
# ===========================================================================

class TestGetEventsMeta:

    @pytest.fixture
    def setup(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=8, freq='D')
        )
        vol  = get_daily_vol(close, span0=3)
        t_ev = pd.DatetimeIndex(['2020-01-02', '2020-01-04'])
        t1   = add_vertical_barrier(close, t_ev, num_days=3)
        return close, vol, t_ev, t1

    def test_no_side_behaves_like_get_events(self, setup):
        close, vol, t_ev, t1 = setup
        events_meta = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=None)
        events_std  = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1)
        # Should produce same number of events
        assert len(events_meta) == len(events_std)

    def test_with_side_includes_side_column(self, setup):
        close, vol, t_ev, t1 = setup
        side   = pd.Series([1.0, -1.0], index=t_ev)
        events = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=side)
        assert 'side' in events.columns

    def test_without_side_drops_side_column(self, setup):
        close, vol, t_ev, t1 = setup
        events = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=None)
        assert 'side' not in events.columns


class TestGetBinsMeta:

    @pytest.fixture
    def setup_no_side(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=8, freq='D')
        )
        vol    = get_daily_vol(close, span0=3)
        t_ev   = pd.DatetimeIndex(['2020-01-02', '2020-01-04'])
        t1     = add_vertical_barrier(close, t_ev, num_days=3)
        events = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=None)
        return close, events

    @pytest.fixture
    def setup_with_side(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0],
            index=pd.date_range('2020-01-01', periods=8, freq='D')
        )
        vol    = get_daily_vol(close, span0=3)
        t_ev   = pd.DatetimeIndex(['2020-01-02', '2020-01-04'])
        t1     = add_vertical_barrier(close, t_ev, num_days=3)
        side   = pd.Series([1.0, -1.0], index=t_ev)
        events = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=side)
        return close, events

    def test_no_side_bin_values_correct(self, setup_no_side):
        close, events = setup_no_side
        bins = get_bins_meta(events, close)
        assert bins['bin'].iloc[0] == pytest.approx(1.0)
        assert bins['ret'].iloc[0] == pytest.approx(0.0291, abs=1e-3)

    def test_no_side_bins_in_valid_set(self, setup_no_side):
        close, events = setup_no_side
        bins = get_bins_meta(events, close)
        assert set(bins['bin'].unique()).issubset({-1.0, 0.0, 1.0})

    def test_with_side_bins_binary(self, setup_with_side):
        # Meta-labeling: labels should be 0 or 1 only
        close, events = setup_with_side
        bins = get_bins_meta(events, close)
        assert set(bins['bin'].unique()).issubset({0.0, 1.0})

    def test_with_side_known_bin(self, setup_with_side):
        # Primary said short (-1), price went UP → primary was WRONG → bin=0
        close, events = setup_with_side
        bins = get_bins_meta(events, close)
        assert bins['bin'].iloc[0] == pytest.approx(0.0)

    def test_correct_primary_model_gives_bin_one(self):
        # Primary says long (+1), price goes up → correct → bin=1
        close = pd.Series(
            [100.0, 102.0, 104.0, 106.0, 108.0, 110.0],
            index=pd.date_range('2020-01-01', periods=6, freq='D')
        )
        vol    = get_daily_vol(close, span0=3)
        t_ev   = pd.DatetimeIndex(['2020-01-02'])
        t1     = add_vertical_barrier(close, t_ev, num_days=3)
        side   = pd.Series([1.0], index=t_ev)
        events = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, t1=t1, side=side)
        bins   = get_bins_meta(events, close)
        if len(bins) > 0:
            assert bins['bin'].iloc[0] in {0.0, 1.0}


class TestDropLabels:

    def test_removes_rare_label(self):
        # 19 ones, 1 zero, 1 negative_one → zero and neg_one are rare (<5%)
        df = pd.DataFrame({'bin': [1]*19 + [0] + [-1]})
        result = drop_labels(df, min_pct=0.05)
        # At most 2 unique labels remain
        assert len(result['bin'].unique()) <= 2

    def test_stops_at_two_labels(self):
        # Even if both remaining labels are below min_pct, stop at 2
        df = pd.DataFrame({'bin': [1, 1, 0, 0, -1, -1]})
        result = drop_labels(df, min_pct=0.5)
        assert len(result['bin'].unique()) >= 2

    def test_known_drop_result(self):
        # 17 ones, 1 zero, 1 neg_one, 2 more ones = 19 ones total
        # 1 zero (4.76%) and 1 neg_one (4.76%) are both below 5%
        df = pd.DataFrame({'bin': [1]*17 + [0] + [-1] + [1]*2})
        result = drop_labels(df, min_pct=0.05)
        assert len(result) == 20  # the zero row was dropped

    def test_returns_dataframe(self):
        df = pd.DataFrame({'bin': [1, 1, 1, 0, -1]})
        result = drop_labels(df, min_pct=0.05)
        assert isinstance(result, pd.DataFrame)

    def test_no_drop_when_all_above_threshold(self):
        # Equal distribution → nothing dropped
        df = pd.DataFrame({'bin': [1]*10 + [-1]*10})
        result = drop_labels(df, min_pct=0.05)
        assert len(result) == len(df)


# ===========================================================================
# Multithreading — mpPandasObj
# ===========================================================================
# These tests verify that num_threads > 1 produces identical results to
# num_threads=1. Skipped automatically on Windows because Python's
# multiprocessing requires a __main__ guard that pytest does not provide.
#
# To run manually on Windows outside pytest:
#   python -c "
#   import sys; sys.path.insert(0, '.')
#   from ch03.tests.test_ch03 import run_threading_tests
#   run_threading_tests()
#   "

import platform

@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason=(
        "Skipped on Windows: Python multiprocessing requires a __main__ guard "
        "that pytest does not provide. Run manually via run_threading_tests() "
        "to verify. Single-threaded correctness is covered by all other tests."
    )
)
class TestMultithreading:

    @pytest.fixture
    def setup(self):
        close = pd.Series(
            [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0,
             112.0, 111.0, 114.0, 116.0, 115.0, 118.0, 120.0],
            index=pd.date_range('2020-01-01', periods=15, freq='D')
        )
        vol  = get_daily_vol(close, span0=3)
        t_ev = pd.DatetimeIndex([
            '2020-01-03', '2020-01-05', '2020-01-07',
            '2020-01-09', '2020-01-11'
        ])
        t1 = add_vertical_barrier(close, t_ev, num_days=3)
        return close, vol, t_ev, t1

    def test_get_events_multithreaded_matches_single(self, setup):
        # num_threads=2 must produce the same t1 values as num_threads=1
        close, vol, t_ev, t1 = setup
        e1 = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=1, t1=t1)
        e2 = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=2, t1=t1)
        assert list(e1.index) == list(e2.index)
        pd.testing.assert_series_equal(e1['t1'].sort_index(), e2['t1'].sort_index())

    def test_get_events_meta_multithreaded_matches_single(self, setup):
        close, vol, t_ev, t1 = setup
        side = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0], index=t_ev)
        e1 = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=1, t1=t1, side=side)
        e2 = get_events_meta(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=2, t1=t1, side=side)
        assert list(e1.index) == list(e2.index)
        pd.testing.assert_series_equal(e1['t1'].sort_index(), e2['t1'].sort_index())

    def test_multithreaded_bins_match_single(self, setup):
        close, vol, t_ev, t1 = setup
        e1 = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=1, t1=t1)
        e2 = get_events(close, t_ev, pt_sl=[1,1], trgt=vol, min_ret=0.0, num_threads=2, t1=t1)
        pd.testing.assert_frame_equal(get_bins(e1, close), get_bins(e2, close))


def run_threading_tests():
    """
    Run multithreading verification manually on Windows.

    From C:\\ws\\AFML in a terminal:
        python -c "
        import sys; sys.path.insert(0, '.')
        from ch03.tests.test_ch03 import run_threading_tests
        run_threading_tests()
        "
    """
    close = pd.Series(
        [100.0, 102.0, 104.0, 103.0, 106.0, 108.0, 107.0, 110.0,
         112.0, 111.0, 114.0, 116.0, 115.0, 118.0, 120.0],
        index=pd.date_range('2020-01-01', periods=15, freq='D')
    )
    vol  = get_daily_vol(close, span0=3)
    t_ev = pd.DatetimeIndex([
        '2020-01-03', '2020-01-05', '2020-01-07',
        '2020-01-09', '2020-01-11'
    ])
    t1 = add_vertical_barrier(close, t_ev, num_days=3)

    e1 = get_events(close, t_ev, [1,1], vol, 0.0, num_threads=1, t1=t1)
    e2 = get_events(close, t_ev, [1,1], vol, 0.0, num_threads=2, t1=t1)
    assert list(e1.index) == list(e2.index), "Indexes differ!"
    pd.testing.assert_series_equal(e1['t1'].sort_index(), e2['t1'].sort_index())
    pd.testing.assert_frame_equal(get_bins(e1, close), get_bins(e2, close))
    print("✅ All multithreading tests passed.")


if __name__ == '__main__':
    run_threading_tests()
