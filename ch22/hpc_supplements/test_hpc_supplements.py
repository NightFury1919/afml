"""
TDD tests for AFML Chapter 22 supplements (io_benchmark.py, nufft_analysis.py).

These are NOT book-snippet tests (Chapter 22 has none) -- they verify the
supplementary code built for this chapter's optional teaching material.
Where a supplement produces a real-world TIMING result (io_benchmark), the
tests check structural correctness (shapes, round-trip fidelity, sane
non-negative durations) rather than exact timing numbers, since wall-clock
timings are inherently non-deterministic -- this is a deliberate departure
from every other chapter's hand-traced-exact-value convention, and is
noted here explicitly so it doesn't read as an oversight. Where a
supplement is pure math (nufft_analysis), hand-traced exact values are
used as usual.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_benchmark import (  # noqa: E402
    benchmark_column_subset_read,
    benchmark_write_read,
    load_real_trades,
    replicate_data,
)
from nufft_analysis import (  # noqa: E402
    days_since_start,
    nudft,
    power_spectrum,
    price_return_series,
    trade_size_series,
)


# ---------------------------------------------------------------------------
# io_benchmark.py
# ---------------------------------------------------------------------------
class TestLoadRealTrades:
    def test_parses_binance_format(self, tmp_path):
        f = tmp_path / 'trades.csv'
        f.write_text(
            '1,100.0,0.5,50.0,1772323209088203,True,True\n'
            '2,101.0,0.25,25.25,1772323316320184,False,True\n'
        )
        df = load_real_trades(str(f))
        assert list(df.columns) >= ['trade_id', 'price', 'qty', 'quote_qty', 'time_us',
                                     'is_buyer_maker', 'is_best_match', 'ts'][:len(df.columns)]
        assert len(df) == 2
        assert df['price'].iloc[0] == 100.0
        assert pd.api.types.is_datetime64_any_dtype(df['ts'])


class TestReplicateData:
    def test_row_count_multiplies(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        out = replicate_data(df, 4)
        assert len(out) == 12

    def test_values_repeat(self):
        df = pd.DataFrame({'a': [1, 2]})
        out = replicate_data(df, 3)
        assert list(out['a']) == [1, 2, 1, 2, 1, 2]

    def test_rejects_invalid_n(self):
        df = pd.DataFrame({'a': [1]})
        with pytest.raises(ValueError):
            replicate_data(df, 0)


class TestBenchmarkWriteRead:
    def _synthetic_df(self, n=500):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            'price': rng.uniform(60000, 70000, n),
            'qty': rng.uniform(0.001, 1.0, n),
            'time_us': np.arange(n) * 1000,
        })

    def test_returns_well_formed_structure(self, tmp_path):
        df = self._synthetic_df()
        result = benchmark_write_read(df, str(tmp_path), n_repeats=2)
        assert set(result.keys()) == {'csv', 'parquet'}
        for fmt in ('csv', 'parquet'):
            assert result[fmt]['write_s'] >= 0
            assert result[fmt]['read_s'] >= 0
            assert result[fmt]['size_bytes'] > 0

    def test_files_actually_written(self, tmp_path):
        df = self._synthetic_df()
        benchmark_write_read(df, str(tmp_path), n_repeats=1)
        assert os.path.exists(os.path.join(str(tmp_path), 'bench.csv'))
        assert os.path.exists(os.path.join(str(tmp_path), 'bench.parquet'))


class TestBenchmarkColumnSubsetRead:
    def _synthetic_df(self, n=500):
        rng = np.random.default_rng(1)
        return pd.DataFrame({
            'price': rng.uniform(60000, 70000, n),
            'qty': rng.uniform(0.001, 1.0, n),
            'time_us': np.arange(n) * 1000,
        })

    def test_result_matches_and_well_formed(self, tmp_path):
        df = self._synthetic_df()
        result = benchmark_column_subset_read(df, str(tmp_path), column='price', n_repeats=2)
        assert result['result_matches'] is True
        assert result['csv_column_read_s'] >= 0
        assert result['parquet_column_read_s'] >= 0


# ---------------------------------------------------------------------------
# nufft_analysis.py
# ---------------------------------------------------------------------------
class TestDaysSinceStart:
    def test_hand_traced(self):
        ts = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])
        result = days_since_start(ts)
        np.testing.assert_allclose(result, [0.0, 1.0, 2.0])

    def test_fractional_days(self):
        ts = pd.to_datetime(['2020-01-01 00:00:00', '2020-01-01 12:00:00'])
        result = days_since_start(ts)
        np.testing.assert_allclose(result, [0.0, 0.5])


class TestNudft:
    def test_hand_traced_two_point_dc(self):
        # times=[0,1], values=[1,1], freq=0: X(0) = 1*e^0 + 1*e^0 = 2.0
        result = nudft([0, 1], [1, 1], [0])
        np.testing.assert_allclose(result, [2.0 + 0j], atol=1e-10)

    def test_hand_traced_two_point_cancellation(self):
        # times=[0,1], values=[1,1], freq=0.5:
        #   e^(-2j*pi*0.5*0) = 1
        #   e^(-2j*pi*0.5*1) = e^(-j*pi) = -1
        #   X(0.5) = 1*1 + 1*(-1) = 0
        result = nudft([0, 1], [1, 1], [0.5])
        np.testing.assert_allclose(result, [0.0 + 0j], atol=1e-10)

    def test_recovers_known_frequency_from_irregular_samples(self):
        # Construct an irregularly-sampled pure cosine at f0=2.0 cycles/unit
        # and confirm the power spectrum peaks at f0 among a coarse grid.
        rng = np.random.default_rng(42)
        times = np.sort(rng.uniform(0, 10, 300))  # irregular, NOT evenly spaced
        f0 = 2.0
        values = np.cos(2 * np.pi * f0 * times)
        freqs = np.linspace(0.1, 5.0, 200)
        spec = power_spectrum(times, values, freqs)
        peak_freq = freqs[np.argmax(spec['magnitude'])]
        assert peak_freq == pytest.approx(f0, abs=0.1)


class TestPowerSpectrum:
    def test_magnitude_and_power_consistent(self):
        result = power_spectrum([0, 1, 2], [1.0, 0.5, -0.5], [0.1, 0.3])
        np.testing.assert_allclose(result['power'], result['magnitude'] ** 2)
        np.testing.assert_allclose(result['magnitude'], np.abs(result['X']))


class TestPriceReturnSeries:
    def test_hand_traced(self):
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03']),
            'price': [100.0, 110.0, 99.0],
        })
        times_days, log_returns = price_return_series(df)
        expected_returns = [np.log(110.0 / 100.0), np.log(99.0 / 110.0)]
        np.testing.assert_allclose(log_returns, expected_returns)
        np.testing.assert_allclose(times_days, [1.0, 2.0])

    def test_length_is_n_minus_1(self):
        df = pd.DataFrame({
            'ts': pd.date_range('2020-01-01', periods=10, freq='h'),
            'price': np.linspace(100, 110, 10),
        })
        times_days, log_returns = price_return_series(df)
        assert len(log_returns) == 9
        assert len(times_days) == 9


class TestTradeSizeSeries:
    def test_matches_qty_column_sorted(self):
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2020-01-02', '2020-01-01']),  # deliberately unsorted
            'qty': [5.0, 3.0],
        })
        times_days, qty = trade_size_series(df)
        # After sorting by ts: 2020-01-01 (qty=3.0) comes first, then 2020-01-02 (qty=5.0)
        np.testing.assert_allclose(qty, [3.0, 5.0])
        np.testing.assert_allclose(times_days, [0.0, 1.0])
