"""
TDD suite for Chapter 14 -- Backtest Statistics.

All tests use hand-traced, synthetic data with known expected values
(per project convention: unit tests use synthetic/hand-traced data;
real BTC/TUSD data is exercised in chapter_14_backtest_statistics.py,
consuming Ch12's real CPCV path signal/returns).
"""
# --- import the module(s) under test ---------------------------------------
# Derive the repo root from __file__, put it on sys.path, then import
# fully-qualified.
#
# LOAD-BEARING (2026-07-22): this used to be a bare 'from backtest_statistics
# import ...' with no sys.path handling, which the 2026-07-21 audit flagged
# as fragile -- it only passes when pytest is invoked from inside this
# module's own folder; from the repo root it raises ModuleNotFoundError
# (both this folder and ch14/ have __init__.py, per the audit's verified
# truth table). See ch10/bet_sizing/test_bet_sizing.py or ch13/otr/test_otr.py
# for the pattern this was generalized from.
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ch14.backtest_statistics.backtest_statistics import (  # noqa: E402
    getBetTiming, getHoldingPeriod, getHHI, hhi_concentration_stats,
    computeDD_TuW, probabilistic_sharpe_ratio, expected_max_sharpe,
    deflated_sharpe_ratio, EULER_MASCHERONI,
)


# --------------------------------------------------------------------------- #
class TestGetBetTiming:
    def test_flattening_and_flip(self):
        """long -> flat -> short -> flip long: expect a flattening bet at the
        flat timestamp, and a flip bet at the flip timestamp (also the last
        index, so it's not double-appended)."""
        idx = pd.date_range('2026-01-01', periods=6, freq='D')
        tPos = pd.Series([1, 1, 0, -1, -1, 1], index=idx)
        bets = getBetTiming(tPos)
        assert list(bets) == [idx[2], idx[5]]

    def test_last_bet_appended_when_no_natural_end(self):
        """Position still open (nonzero) at the series' last index, with no
        flattening/flip there -> the last index must still be appended as
        a bet-ending timestamp."""
        idx = pd.date_range('2026-01-01', periods=4, freq='D')
        tPos = pd.Series([0, 1, 1, 1], index=idx)  # opens day1, never closes
        bets = getBetTiming(tPos)
        assert idx[-1] in bets

    def test_no_double_count_when_last_index_already_a_bet(self):
        idx = pd.date_range('2026-01-01', periods=3, freq='D')
        tPos = pd.Series([1, -1, -1], index=idx)  # flip at day1 (last isn't 0/flip)
        bets = getBetTiming(tPos)
        # day1 (index 1) is a flip; day2 (last) is appended separately since not a natural bet end
        assert idx[1] in bets
        assert idx[-1] in bets
        assert len(bets) == len(set(bets))  # no duplicates


# --------------------------------------------------------------------------- #
class TestGetHoldingPeriod:
    def test_single_trade_known_duration(self):
        """Enter day1, exit (flatten) day4 -> holding period = 3 days."""
        idx = pd.date_range('2026-01-01', periods=5, freq='D')
        tPos = pd.Series([0, 1, 1, 1, 0], index=idx)
        assert getHoldingPeriod(tPos) == pytest.approx(3.0)

    def test_two_trades_weighted_average(self):
        """Trade A: size 2, duration 1 day. Trade B: size 1, duration 3 days.
        Weighted avg = (1*2 + 3*1) / (2+1) = 5/3."""
        idx = pd.date_range('2026-01-01', periods=8, freq='D')
        tPos = pd.Series([0, 2, 0, 0, 1, 1, 1, 0], index=idx)
        assert getHoldingPeriod(tPos) == pytest.approx(5.0 / 3.0)

    def test_never_enters_position_returns_nan(self):
        idx = pd.date_range('2026-01-01', periods=4, freq='D')
        tPos = pd.Series([0, 0, 0, 0], index=idx)
        assert np.isnan(getHoldingPeriod(tPos))


# --------------------------------------------------------------------------- #
class TestGetHHI:
    def test_uniform_returns_near_zero(self):
        uniform = pd.Series([1.0] * 10)
        assert getHHI(uniform) == pytest.approx(0.0, abs=1e-12)

    def test_single_dominant_return_near_one(self):
        dominant = pd.Series([100.0] + [0.001] * 4)
        assert getHHI(dominant) == pytest.approx(1.0, abs=1e-2)

    def test_small_sample_returns_nan(self):
        assert np.isnan(getHHI(pd.Series([1.0, 2.0])))
        assert np.isnan(getHHI(pd.Series([1.0])))

    def test_bounded_zero_to_one(self):
        rng = np.random.default_rng(0)
        vals = pd.Series(rng.normal(1, 0.3, size=50).clip(min=0.01))
        h = getHHI(vals)
        assert 0.0 <= h <= 1.0


class TestHHIConcentrationStats:
    def test_splits_positive_and_negative_correctly(self):
        idx = pd.date_range('2026-01-01', periods=10, freq='D')
        ret = pd.Series([1, -1, 2, -2, 3, -3, 4, -4, 5, -5], index=idx, dtype=float)
        stats = hhi_concentration_stats(ret)
        assert set(stats.keys()) == {'hhi_positive', 'hhi_negative', 'hhi_time'}
        # 5 positive, 5 negative, symmetric magnitudes -> equal concentration
        assert stats['hhi_positive'] == pytest.approx(stats['hhi_negative'])


# --------------------------------------------------------------------------- #
class TestComputeDDTuW:
    def test_single_drawdown_known_value(self):
        idx = pd.date_range('2026-01-01', periods=5, freq='D')
        pnl = pd.Series([100, 110, 90, 105, 120], index=idx, dtype=float)
        dd, tuw = computeDD_TuW(pnl, dollars=False)
        assert len(dd) == 1
        assert dd.iloc[0] == pytest.approx(1 - 90 / 110)
        assert len(tuw) == 0  # only one drawdown -> no interval to measure

    def test_two_drawdowns_known_values_and_tuw(self):
        idx = pd.date_range('2026-01-01', periods=8, freq='D')
        pnl = pd.Series([100, 110, 90, 100, 115, 95, 110, 130], index=idx, dtype=float)
        dd, tuw = computeDD_TuW(pnl, dollars=False)
        assert len(dd) == 2
        assert dd.iloc[0] == pytest.approx(1 - 90 / 110)
        assert dd.iloc[1] == pytest.approx(1 - 95 / 115)
        assert len(tuw) == 1
        expected_years = 3 / 365.25  # day2 (idx1) to day5 (idx4) = 3 days
        assert tuw.iloc[0] == pytest.approx(expected_years, rel=1e-3)

    def test_dollars_mode_matches_pct_mode_relationship(self):
        idx = pd.date_range('2026-01-01', periods=5, freq='D')
        pnl = pd.Series([100, 110, 90, 105, 120], index=idx, dtype=float)
        dd_pct, _ = computeDD_TuW(pnl, dollars=False)
        dd_dollars, _ = computeDD_TuW(pnl, dollars=True)
        # dd_pct = dd_dollars / hwm  ->  110*dd_pct == dd_dollars for this single-hwm case
        assert dd_dollars.iloc[0] == pytest.approx(dd_pct.iloc[0] * 110)

    def test_monotonically_increasing_series_has_no_drawdown(self):
        idx = pd.date_range('2026-01-01', periods=5, freq='D')
        pnl = pd.Series([100, 110, 120, 130, 140], index=idx, dtype=float)
        dd, tuw = computeDD_TuW(pnl, dollars=False)
        assert len(dd) == 0
        assert len(tuw) == 0


# --------------------------------------------------------------------------- #
class TestProbabilisticSharpeRatio:
    def test_sr_hat_equals_benchmark_is_one_half(self):
        """Z(0) = 0.5 exactly, regardless of T/skew/kurtosis."""
        psr = probabilistic_sharpe_ratio(sr_hat=1.0, sr_benchmark=1.0, T=100, skew=0.3, kurtosis=4.0)
        assert psr == pytest.approx(0.5)

    def test_matches_manual_formula(self):
        sr_hat, sr_bench, T, skew, kurt = 1.5, 1.0, 200, 0.2, 4.5
        num = (sr_hat - sr_bench) * np.sqrt(T - 1)
        den = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4 * sr_hat ** 2)
        expected = norm.cdf(num / den)
        assert probabilistic_sharpe_ratio(sr_hat, sr_bench, T, skew, kurt) == pytest.approx(expected)

    def test_increases_with_sr_hat(self):
        low = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=100)
        high = probabilistic_sharpe_ratio(sr_hat=1.5, sr_benchmark=0., T=100)
        assert high > low

    def test_increases_with_longer_track_record(self):
        short = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=30)
        long_ = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=500)
        assert long_ > short

    def test_decreases_with_fatter_tails(self):
        thin = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=100, kurtosis=3.0)
        fat = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=100, kurtosis=10.0)
        assert fat < thin


# --------------------------------------------------------------------------- #
class TestExpectedMaxSharpe:
    def test_increases_with_n_trials(self):
        sr2 = expected_max_sharpe(var_sr_trials=0.01, N=2)
        sr10 = expected_max_sharpe(var_sr_trials=0.01, N=10)
        sr50 = expected_max_sharpe(var_sr_trials=0.01, N=50)
        assert sr2 < sr10 < sr50

    def test_increases_with_trial_variance(self):
        low_var = expected_max_sharpe(var_sr_trials=0.001, N=10)
        high_var = expected_max_sharpe(var_sr_trials=0.1, N=10)
        assert high_var > low_var

    def test_matches_manual_formula(self):
        var_sr, N = 0.02, 8
        z1 = norm.ppf(1 - 1.0 / N)
        z2 = norm.ppf(1 - 1.0 / (N * np.e))
        expected = np.sqrt(var_sr) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
        assert expected_max_sharpe(var_sr, N) == pytest.approx(expected)


class TestDeflatedSharpeRatio:
    def test_matches_psr_composed_with_expected_max_sharpe(self):
        sr_hat, var_sr, N, T = 0.8, 0.015, 5, 90
        sr_star = expected_max_sharpe(var_sr, N)
        expected = probabilistic_sharpe_ratio(sr_hat, sr_star, T)
        assert deflated_sharpe_ratio(sr_hat, var_sr, N, T) == pytest.approx(expected)

    def test_approaches_psr_zero_as_trial_variance_shrinks(self):
        """As V[{SR_n}] -> 0, SR* -> 0, so DSR -> PSR[0]."""
        dsr = deflated_sharpe_ratio(sr_hat=0.5, var_sr_trials=1e-10, N=2, T=100)
        psr_zero = probabilistic_sharpe_ratio(sr_hat=0.5, sr_benchmark=0., T=100)
        assert dsr == pytest.approx(psr_zero, abs=1e-4)

    def test_more_trials_deflates_more(self):
        """Holding sr_hat and trial variance fixed, more independent trials
        (larger N) raises SR* and therefore LOWERS the resulting DSR."""
        dsr_few = deflated_sharpe_ratio(sr_hat=0.5, var_sr_trials=0.02, N=2, T=100)
        dsr_many = deflated_sharpe_ratio(sr_hat=0.5, var_sr_trials=0.02, N=50, T=100)
        assert dsr_many < dsr_few
