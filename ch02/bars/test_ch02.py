r"""
test_ch02.py — TDD tests for AFML Chapter 2 implementations
Run with: pytest ch02/bars/test_ch02.py -v

📁 C:\ws\AFML\
└── ch02\
    └── bars\
        └── test_ch02.py   ← goes here (moved from ch02\tests\, 2026-07 layout refactor;
                              covers ch02/bars and ch02/multi_product via fully-qualified
                              ch02.* imports, not co-location)

All expected values were computed by running the actual implementation
functions and recording their output. Tests verify specific numeric values,
not just types or shapes.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — allows imports from ch02/bars and ch02/multi_product
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ch02.bars.utils           import ewma, delta, tick_rule, estimate_buy_sell_probs
from ch02.bars.filters         import cusum_filter
from ch02.bars.standard_bars   import time_bars, tick_bars, volume_bars, dollar_bars
from ch02.bars.imbalance_bars  import tick_imbalance_bars, volume_imbalance_bars
from ch02.bars.run_bars        import tick_run_bars, volume_run_bars
from ch02.multi_product.pca_weights import pca_weights
from ch02.multi_product.roll   import roll_gaps, get_rolled_series, non_negative_rolled_prices
from ch02.multi_product.etf_trick  import etf_trick


# ===========================================================================
# Section 2.3 — utils.py
# ===========================================================================

class TestEwma:

    def test_empty_array_returns_zero(self):
        assert ewma([], 3) == 0

    def test_single_element_returns_that_element(self):
        assert ewma([10], 3) == 10

    def test_three_element_window_three(self):
        # alpha = 2/(3+1) = 0.5
        # ewma_0 = 10
        # ewma_1 = 0.5*12 + 0.5*10 = 11.0
        # ewma_2 = 0.5*8 + 0.5*11  = 9.5
        result = ewma([10, 12, 8], 3)
        assert result == pytest.approx(9.5, abs=1e-9)

    def test_five_elements_window_two(self):
        # alpha = 2/(2+1) = 2/3
        result = ewma([1, 2, 3, 4, 5], 2)
        assert result == pytest.approx(4.5061728395, rel=1e-6)

    def test_larger_window_smoother(self):
        # Larger window → result closer to simple average
        arr = [10, 20, 10, 20, 10]
        small = ewma(arr, 2)
        large = ewma(arr, 10)
        # Large window should be closer to the mean (14)
        assert abs(large - 14) < abs(small - 14)


class TestDelta:

    def test_first_row_is_zero(self):
        df = pd.DataFrame({'Price': [100, 102, 101, 105], 'Volume': [1,1,1,1]})
        result = delta(df)
        assert result['Delta'].iloc[0] == 0.0

    def test_price_changes_correct(self):
        df = pd.DataFrame({'Price': [100, 102, 101, 105], 'Volume': [1,1,1,1]})
        result = delta(df)
        assert list(result['Delta']) == [0, 2, -1, 4]

    def test_output_length_matches_input(self):
        df = pd.DataFrame({'Price': [1, 2, 3, 4, 5], 'Volume': [1,1,1,1,1]})
        result = delta(df)
        assert len(result) == 5

    def test_does_not_modify_price_column(self):
        df = pd.DataFrame({'Price': [100.0, 105.0], 'Volume': [1,1]})
        result = delta(df)
        assert result['Price'].iloc[0] == 100.0
        assert result['Price'].iloc[1] == 105.0


class TestTickRule:

    def test_uptick_gives_positive_one(self):
        df = pd.DataFrame({'Price': [100, 102], 'Volume': [1,1]})
        df = delta(df)
        df = tick_rule(df)
        assert df['Label'].iloc[1] == 1.0

    def test_downtick_gives_negative_one(self):
        df = pd.DataFrame({'Price': [100, 98], 'Volume': [1,1]})
        df = delta(df)
        df = tick_rule(df)
        assert df['Label'].iloc[1] == -1.0

    def test_flat_price_carries_forward(self):
        # price: up, flat, flat → labels: +1, +1, +1
        df = pd.DataFrame({'Price': [100, 102, 102, 102], 'Volume': [1,1,1,1]})
        df = delta(df)
        df = tick_rule(df)
        assert list(df['Label']) == [1.0, 1.0, 1.0, 1.0]

    def test_full_sequence(self):
        # price: -, up, flat, down → labels: +1, +1, +1, -1
        df = pd.DataFrame({'Price': [100, 102, 102, 99], 'Volume': [1,1,1,1]})
        df = delta(df)
        df = tick_rule(df)
        assert list(df['Label']) == [1.0, 1.0, 1.0, -1.0]

    def test_default_first_label_is_one(self):
        df = pd.DataFrame({'Price': [100, 99], 'Volume': [1,1]})
        df = delta(df)
        df = tick_rule(df)
        assert df['Label'].iloc[0] == 1.0


class TestEstimateBuySellProbs:

    def test_three_buys_one_sell(self):
        df = pd.DataFrame({'Price': [1,2,3,4], 'Label': [1,1,1,-1]})
        pb, ps = estimate_buy_sell_probs(df)
        assert float(pb) == pytest.approx(0.75, abs=1e-9)
        assert float(ps) == pytest.approx(0.25, abs=1e-9)

    def test_probs_sum_to_one(self):
        df = pd.DataFrame({'Price': [1,2,3,4,5], 'Label': [1,1,-1,-1,-1]})
        pb, ps = estimate_buy_sell_probs(df)
        assert float(pb) + float(ps) == pytest.approx(1.0, abs=1e-9)

    def test_all_buys(self):
        df = pd.DataFrame({'Price': [1,2,3], 'Label': [1,1,1]})
        pb, ps = estimate_buy_sell_probs(df)
        assert float(pb) == pytest.approx(1.0, abs=1e-9)
        assert float(ps) == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# Section 2.5 — filters.py
# ===========================================================================

class TestCusumFilter:

    def _make_df(self, prices):
        dates = pd.date_range('2020-01-01', periods=len(prices), freq='D')
        return pd.DataFrame({'Price': prices, 'Date': dates})

    def test_fires_on_upward_drift(self):
        # Prices jump +6 on day 6 and again on day 11 → 2 events
        prices = [100,101,102,103,104,110,111,112,113,114,120]
        df = self._make_df(prices)
        events = cusum_filter(df, h=5)
        assert len(events) == 2

    def test_event_dates_correct(self):
        prices = [100,101,102,103,104,110,111,112,113,114,120]
        df = self._make_df(prices)
        events = cusum_filter(df, h=5)
        assert str(events[0].date()) == '2020-01-06'
        assert str(events[1].date()) == '2020-01-11'

    def test_no_events_when_drift_below_threshold(self):
        prices = [100, 101, 100, 101, 100]  # oscillates, never drifts by h
        df = self._make_df(prices)
        events = cusum_filter(df, h=10)
        assert len(events) == 0

    def test_returns_datetimeindex(self):
        prices = [100, 110]
        df = self._make_df(prices)
        events = cusum_filter(df, h=5)
        assert isinstance(events, pd.DatetimeIndex)

    def test_downward_drift_also_fires(self):
        prices = [100, 99, 98, 97, 96, 90]
        df = self._make_df(prices)
        events = cusum_filter(df, h=5)
        assert len(events) >= 1


# ===========================================================================
# Section 2.3 — standard_bars.py
# ===========================================================================

class TestTickBars:

    def _make_trades(self, prices, volumes=None):
        n = len(prices)
        return pd.DataFrame({
            'Price':  prices,
            'Volume': volumes if volumes else [1.0]*n,
            'Date':   pd.date_range('2020-01-01', periods=n, freq='D')
        })

    def test_correct_number_of_bars(self):
        df = self._make_trades([100,101,102,103,104,105])
        bars = tick_bars(df, thresh=3)
        assert len(bars) == 2

    def test_ohlc_values_first_bar(self):
        df = self._make_trades([100.0,101.0,102.0,103.0,104.0,105.0])
        bars = tick_bars(df, thresh=3)
        assert bars['Open'].iloc[0]  == pytest.approx(100.0)
        assert bars['High'].iloc[0]  == pytest.approx(102.0)
        assert bars['Low'].iloc[0]   == pytest.approx(100.0)
        assert bars['Close'].iloc[0] == pytest.approx(102.0)

    def test_vwap_equal_volumes(self):
        # Equal volumes → VWAP = arithmetic mean
        df = self._make_trades([100.0,101.0,102.0])
        bars = tick_bars(df, thresh=3)
        assert bars['Vwap'].iloc[0] == pytest.approx(101.0)

    def test_vwap_unequal_volumes(self):
        # 10 units at 100, 1 unit at 200 → VWAP ≈ 109.09
        df = self._make_trades([100.0, 200.0, 100.0], volumes=[10.0, 1.0, 1.0])
        bars = tick_bars(df, thresh=3)
        expected_vwap = (10*100 + 1*200 + 1*100) / 12
        assert bars['Vwap'].iloc[0] == pytest.approx(expected_vwap, rel=1e-6)

    def test_incomplete_last_bar_not_included(self):
        # 7 trades with thresh=3 → 2 complete bars, 1 trade left over
        df = self._make_trades([100]*7)
        bars = tick_bars(df, thresh=3)
        assert len(bars) == 2


class TestVolumeBars:

    def _make_trades(self, prices, volumes):
        return pd.DataFrame({
            'Price':  prices, 'Volume': volumes,
            'Date':   pd.date_range('2020-01-01', periods=len(prices), freq='D')
        })

    def test_closes_on_volume_threshold(self):
        df = self._make_trades([100.0]*6, [1.0]*6)
        bars = volume_bars(df, thresh=3)
        assert len(bars) == 2

    def test_ohlc_values(self):
        df = self._make_trades([100.0,101.0,102.0,103.0,104.0,105.0], [1.0]*6)
        bars = volume_bars(df, thresh=3)
        assert bars['Open'].iloc[0]  == pytest.approx(100.0)
        assert bars['High'].iloc[0]  == pytest.approx(102.0)
        assert bars['Low'].iloc[0]   == pytest.approx(100.0)
        assert bars['Close'].iloc[0] == pytest.approx(102.0)
        assert bars['Vwap'].iloc[0]  == pytest.approx(101.0)

    def test_large_single_trade_closes_bar(self):
        # One trade of 10 units closes bar immediately (thresh=5)
        df = self._make_trades([100.0, 200.0], [10.0, 1.0])
        bars = volume_bars(df, thresh=5)
        assert len(bars) >= 1
        assert bars['Open'].iloc[0] == pytest.approx(100.0)


class TestDollarBars:

    def _make_trades(self, prices, volumes):
        return pd.DataFrame({
            'Price':  prices, 'Volume': volumes,
            'Date':   pd.date_range('2020-01-01', periods=len(prices), freq='D')
        })

    def test_closes_on_dollar_threshold(self):
        # 100*1=100, 200*1=200 → cumulative=300 ≥ 300 → bar closes after 2 trades
        df = self._make_trades([100.0, 200.0, 100.0, 200.0], [1.0]*4)
        bars = dollar_bars(df, thresh=300)
        assert len(bars) == 2

    def test_ohlcv_values(self):
        df = self._make_trades([100.0, 200.0, 100.0, 200.0], [1.0]*4)
        bars = dollar_bars(df, thresh=300)
        assert bars['Open'].iloc[0]  == pytest.approx(100.0)
        assert bars['High'].iloc[0]  == pytest.approx(200.0)
        assert bars['Low'].iloc[0]   == pytest.approx(100.0)
        assert bars['Close'].iloc[0] == pytest.approx(200.0)
        assert bars['Vwap'].iloc[0]  == pytest.approx(150.0)

    def test_single_large_trade_closes_bar(self):
        # 1 trade: price=1000, volume=10 → dollar=10000 ≥ thresh=5000
        df = self._make_trades([1000.0], [10.0])
        bars = dollar_bars(df, thresh=5000)
        assert len(bars) == 1


# ===========================================================================
# Section 2.3.2 — imbalance_bars.py
# ===========================================================================

class TestTickImbalanceBars:

    @pytest.fixture
    def labeled_df(self):
        np.random.seed(42)
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        labels = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        df = pd.DataFrame({
            'Price': prices, 'Volume': np.ones(n),
            'Label': labels,
            'Date': pd.date_range('2020-01-01', periods=n, freq='h')
        })
        from ch02.bars.utils import delta
        return delta(df)

    def test_produces_bars(self, labeled_df):
        bars = tick_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert len(bars) >= 1

    def test_first_bar_ohlc(self, labeled_df):
        bars = tick_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert bars['Open'].iloc[0]  == pytest.approx(100.2484, abs=1e-3)
        assert bars['High'].iloc[0]  == pytest.approx(102.2403, abs=1e-3)
        assert bars['Low'].iloc[0]   == pytest.approx(100.1792, abs=1e-3)
        assert bars['Close'].iloc[0] == pytest.approx(101.8967, abs=1e-3)

    def test_open_lte_high(self, labeled_df):
        bars = tick_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert (bars['High'] >= bars['Open']).all()
        assert (bars['Low']  <= bars['Open']).all()

    def test_close_within_high_low(self, labeled_df):
        bars = tick_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert (bars['Close'] <= bars['High']).all()
        assert (bars['Close'] >= bars['Low']).all()


class TestVolumeImbalanceBars:

    @pytest.fixture
    def labeled_df(self):
        np.random.seed(42)
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        labels = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        df = pd.DataFrame({
            'Price': prices, 'Volume': np.ones(n),
            'Label': labels,
            'Date': pd.date_range('2020-01-01', periods=n, freq='h')
        })
        from ch02.bars.utils import delta
        return delta(df)

    def test_produces_bars(self, labeled_df):
        bars = volume_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert len(bars) >= 1

    def test_has_vwap_column(self, labeled_df):
        bars = volume_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert 'Vwap' in bars.columns

    def test_high_gte_low(self, labeled_df):
        bars = volume_imbalance_bars(labeled_df, expected_num_ticks_init=10)
        assert (bars['High'] >= bars['Low']).all()


# ===========================================================================
# Section 2.3.2 — run_bars.py
# ===========================================================================

class TestTickRunBars:

    @pytest.fixture
    def labeled_df(self):
        np.random.seed(42)
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        labels = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        df = pd.DataFrame({
            'Price': prices, 'Volume': np.ones(n),
            'Label': labels,
            'Date': pd.date_range('2020-01-01', periods=n, freq='h')
        })
        from ch02.bars.utils import delta
        return delta(df)

    def test_correct_number_of_bars(self, labeled_df):
        bars = tick_run_bars(labeled_df, expected_num_ticks_init=10)
        assert len(bars) == 6

    def test_first_bar_ohlcv(self, labeled_df):
        bars = tick_run_bars(labeled_df, expected_num_ticks_init=10)
        assert bars['Open'].iloc[0]  == pytest.approx(100.2484, abs=1e-3)
        assert bars['High'].iloc[0]  == pytest.approx(102.2403, abs=1e-3)
        assert bars['Low'].iloc[0]   == pytest.approx(100.1792, abs=1e-3)
        assert bars['Close'].iloc[0] == pytest.approx(102.2403, abs=1e-3)
        assert bars['Vwap'].iloc[0]  == pytest.approx(101.2606, abs=1e-3)

    def test_high_gte_open_and_close(self, labeled_df):
        bars = tick_run_bars(labeled_df, expected_num_ticks_init=10)
        assert (bars['High'] >= bars['Open']).all()
        assert (bars['High'] >= bars['Close']).all()


class TestVolumeRunBars:

    @pytest.fixture
    def labeled_df(self):
        np.random.seed(42)
        n = 50
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        labels = np.where(np.random.randn(n) > 0, 1.0, -1.0)
        df = pd.DataFrame({
            'Price': prices, 'Volume': np.ones(n),
            'Label': labels,
            'Date': pd.date_range('2020-01-01', periods=n, freq='h')
        })
        from ch02.bars.utils import delta
        return delta(df)

    def test_correct_number_of_bars(self, labeled_df):
        bars = volume_run_bars(labeled_df, expected_num_ticks_init=10)
        assert len(bars) == 26

    def test_first_bar_ohlcv(self, labeled_df):
        bars = volume_run_bars(labeled_df, expected_num_ticks_init=10)
        assert bars['Open'].iloc[0]  == pytest.approx(100.2484, abs=1e-3)
        assert bars['High'].iloc[0]  == pytest.approx(102.2403, abs=1e-3)
        assert bars['Low'].iloc[0]   == pytest.approx(100.1792, abs=1e-3)
        assert bars['Vwap'].iloc[0]  == pytest.approx(101.2606, abs=1e-3)

    def test_has_vwap(self, labeled_df):
        bars = volume_run_bars(labeled_df, expected_num_ticks_init=10)
        assert 'Vwap' in bars.columns
        assert bars['Vwap'].notna().all()


# ===========================================================================
# Section 2.4.2 — pca_weights.py
# ===========================================================================

class TestPcaWeights:

    @pytest.fixture
    def cov_3x3(self):
        return np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0]
        ])

    def test_output_shape(self, cov_3x3):
        w = pca_weights(cov_3x3)
        assert w.shape == (3, 1)

    def test_min_variance_weights(self, cov_3x3):
        w = pca_weights(cov_3x3).flatten()
        assert w[0] == pytest.approx(-0.858454, abs=1e-4)
        assert w[1] == pytest.approx( 1.101382, abs=1e-4)
        assert w[2] == pytest.approx(-0.353295, abs=1e-4)

    def test_equal_risk_weights(self, cov_3x3):
        rd = np.array([1/3, 1/3, 1/3])
        w  = pca_weights(cov_3x3, risk_dist=rd, risk_target=1.0).flatten()
        assert w[0] == pytest.approx(-1.12432,  abs=1e-4)
        assert w[1] == pytest.approx( 0.252292, abs=1e-4)
        assert w[2] == pytest.approx( 0.127822, abs=1e-4)

    def test_identity_cov_min_var(self):
        # Identity covariance: all components equal risk, min-var = last eigenvector
        cov = np.eye(2)
        w   = pca_weights(cov)
        # Result should be a valid 2-vector
        assert w.shape == (2, 1)

    def test_risk_target_scales_weights(self, cov_3x3):
        w1 = pca_weights(cov_3x3, risk_target=1.0).flatten()
        w2 = pca_weights(cov_3x3, risk_target=2.0).flatten()
        np.testing.assert_allclose(w2, w1 * 2, rtol=1e-6)


# ===========================================================================
# Section 2.4.3 — roll.py
# ===========================================================================

class TestRollGaps:

    @pytest.fixture
    def two_contract_series(self):
        return pd.DataFrame({
            'Instrument': ['SP98H','SP98H','SP98H','SP98M','SP98M','SP98M'],
            'Open':       [100.0, 101.0, 102.0, 105.0, 106.0, 107.0],
            'Close':      [100.5, 101.5, 102.5, 105.5, 106.5, 107.5],
        }, index=pd.date_range('2020-01-01', periods=6, freq='D'))

    def test_gaps_backward_correct(self, two_contract_series):
        gaps = roll_gaps(two_contract_series, match_end=True)
        assert list(gaps.round(4)) == [-2.5, -2.5, -2.5, 0.0, 0.0, 0.0]

    def test_last_gap_is_zero_when_match_end(self, two_contract_series):
        gaps = roll_gaps(two_contract_series, match_end=True)
        assert gaps.iloc[-1] == pytest.approx(0.0)

    def test_forward_roll_first_gap_is_zero(self, two_contract_series):
        gaps = roll_gaps(two_contract_series, match_end=False)
        assert gaps.iloc[0] == pytest.approx(0.0)


class TestGetRolledSeries:

    @pytest.fixture
    def series(self):
        return pd.DataFrame({
            'Instrument': ['SP98H','SP98H','SP98H','SP98M','SP98M','SP98M'],
            'Open':       [100.0, 101.0, 102.0, 105.0, 106.0, 107.0],
            'Close':      [100.5, 101.5, 102.5, 105.5, 106.5, 107.5],
        }, index=pd.date_range('2020-01-01', periods=6, freq='D'))

    def test_rolled_close_values(self, series):
        rolled = get_rolled_series(series, match_end=True)
        expected = [103.0, 104.0, 105.0, 105.5, 106.5, 107.5]
        np.testing.assert_allclose(rolled['Close'].values, expected, rtol=1e-6)

    def test_no_jump_at_roll_date(self, series):
        rolled = get_rolled_series(series, match_end=True)
        # Difference at roll (index 2→3) should be small, not a gap
        jump = abs(rolled['Close'].iloc[3] - rolled['Close'].iloc[2])
        assert jump < 2.0  # original gap was 3.0

    def test_does_not_modify_original(self, series):
        original_close = series['Close'].copy()
        get_rolled_series(series, match_end=True)
        pd.testing.assert_series_equal(series['Close'], original_close)


class TestNonNegativeRolledPrices:

    @pytest.fixture
    def series(self):
        return pd.DataFrame({
            'Instrument': ['SP98H','SP98H','SP98H','SP98M','SP98M','SP98M'],
            'Open':       [100.0, 101.0, 102.0, 105.0, 106.0, 107.0],
            'Close':      [100.5, 101.5, 102.5, 105.5, 106.5, 107.5],
        }, index=pd.date_range('2020-01-01', periods=6, freq='D'))

    def test_rprices_always_positive(self, series):
        result = non_negative_rolled_prices(series, match_end=True)
        rp = result['rPrices'].dropna()
        assert (rp > 0).all()

    def test_rprices_known_values(self, series):
        result = non_negative_rolled_prices(series, match_end=True)
        rp = result['rPrices'].dropna().values
        expected = [1.00995, 1.0199, 1.024876, 1.03459, 1.044305]
        np.testing.assert_allclose(rp, expected, rtol=1e-4)

    def test_first_rprices_is_nan(self, series):
        result = non_negative_rolled_prices(series, match_end=True)
        assert np.isnan(result['rPrices'].iloc[0])

    def test_has_returns_column(self, series):
        result = non_negative_rolled_prices(series, match_end=True)
        assert 'Returns' in result.columns


# ===========================================================================
# Section 2.4.1 — etf_trick.py
# ===========================================================================

class TestEtfTrick:

    @pytest.fixture
    def simple_inputs(self):
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        instruments = ['A', 'B']
        op = pd.DataFrame({'A': [100.0]*5, 'B': [200.0]*5}, index=dates)
        cl = pd.DataFrame({
            'A': [101.0, 102.0, 100.0, 103.0, 104.0],
            'B': [201.0, 202.0, 200.0, 203.0, 204.0]
        }, index=dates)
        aw = pd.DataFrame({'A': [0.5]*5, 'B': [0.5]*5}, index=dates)
        pv = pd.DataFrame({'A': [1.0]*5, 'B': [1.0]*5}, index=dates)
        dv = pd.DataFrame({'A': [0.0]*5, 'B': [0.0]*5}, index=dates)
        tc = pd.Series({'A': 0.0, 'B': 0.0})
        rb = [dates[0]]
        return op, cl, aw, pv, dv, rb, tc

    def test_starts_at_one_dollar(self, simple_inputs):
        op, cl, aw, pv, dv, rb, tc = simple_inputs
        result = etf_trick(op, cl, aw, pv, dv, rb, trans_costs=tc)
        assert result['K'].iloc[0] == pytest.approx(1.0)

    def test_k_series_values(self, simple_inputs):
        op, cl, aw, pv, dv, rb, tc = simple_inputs
        result = etf_trick(op, cl, aw, pv, dv, rb, trans_costs=tc)
        expected = [1.0, 1.0, 0.985, 1.0075, 1.015]
        np.testing.assert_allclose(result['K'].values, expected, rtol=1e-4)

    def test_output_columns(self, simple_inputs):
        op, cl, aw, pv, dv, rb, tc = simple_inputs
        result = etf_trick(op, cl, aw, pv, dv, rb, trans_costs=tc)
        assert 'K' in result.columns
        assert 'rebalance_cost' in result.columns
        assert 'bid_ask_cost' in result.columns
        assert 'volume' in result.columns

    def test_output_length(self, simple_inputs):
        op, cl, aw, pv, dv, rb, tc = simple_inputs
        result = etf_trick(op, cl, aw, pv, dv, rb, trans_costs=tc)
        assert len(result) == 5

    def test_no_costs_when_zero_trans_costs(self, simple_inputs):
        op, cl, aw, pv, dv, rb, tc = simple_inputs
        result = etf_trick(op, cl, aw, pv, dv, rb, trans_costs=tc)
        assert result['bid_ask_cost'].sum() == pytest.approx(0.0)
