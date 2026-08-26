"""
pipeline/orchestration/test_new_features_2026_08_25.py

Hand-traced TDD tests for the two new features added 2026-08-25:
compute_entropy_feature() (Ch18) and compute_structural_break_feature()
(Ch17, bounded-lookback adaptation). See features.py's own module
docstring and each function's LOAD-BEARING notes for the full design
rationale.

Per this project's TDD convention: values below are hand-traced /
independently reasoned about, not just asserted against whatever the
code happens to produce.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from features import compute_entropy_feature, compute_structural_break_feature  # noqa: E402


class TestComputeEntropyFeature:
    def test_returns_nan_for_warmup_period(self):
        idx = pd.date_range('2026-01-01', periods=30, freq='h')
        close = pd.Series(100 + np.arange(30) * 0.1, index=idx)
        result = compute_entropy_feature(close, window=20)
        # hand-traced: first `window`=20 bars must be NaN (not enough
        # history for a full rolling window yet)
        assert result.iloc[:20].isna().all()
        assert not result.iloc[20:].isna().all()

    def test_alternating_pattern_has_lower_entropy_than_random_walk(self):
        # hand-traced expectation: a perfectly alternating +/- price
        # pattern is maximally compressible (short non-redundant
        # substrings repeat immediately), so its LZ entropy-rate
        # estimate should be LOWER than a genuine random walk's.
        idx = pd.date_range('2026-01-01', periods=50, freq='h')
        prices_alt = [100 + (1 if i % 2 == 0 else -1) * 0.01 for i in range(50)]
        close_alt = pd.Series(prices_alt, index=idx)
        h_alt = compute_entropy_feature(close_alt, window=20).dropna().mean()

        rng = np.random.default_rng(0)
        prices_rand = 100 + np.cumsum(rng.standard_normal(50) * 0.1)
        close_rand = pd.Series(prices_rand, index=idx)
        h_rand = compute_entropy_feature(close_rand, window=20).dropna().mean()

        assert h_alt < h_rand

    def test_output_indexed_identically_to_input(self):
        idx = pd.date_range('2026-01-01', periods=30, freq='h')
        close = pd.Series(100 + np.arange(30) * 0.1, index=idx)
        result = compute_entropy_feature(close, window=20)
        assert result.index.equals(close.index)

    def test_too_few_nonzero_returns_in_window_stays_nan(self):
        # hand-traced: a flat (zero-return) window has < 4 non-zero
        # returns after binary_encode's own zero-drop rule -- should
        # stay NaN, not crash or feed konto() a near-empty message.
        idx = pd.date_range('2026-01-01', periods=25, freq='h')
        close = pd.Series([100.0] * 25, index=idx)  # perfectly flat
        result = compute_entropy_feature(close, window=20)
        assert result.iloc[20:].isna().all()


class TestComputeStructuralBreakFeature:
    def test_returns_nan_before_min_sample(self):
        idx = pd.date_range('2026-01-01', periods=20, freq='h')
        close = pd.Series(100 + np.arange(20) * 0.1, index=idx)
        result = compute_structural_break_feature(close, min_sample=3)
        # hand-traced: bars before min_sample=3 must be NaN
        assert result.iloc[:3].isna().all()

    def test_output_indexed_identically_to_input(self):
        idx = pd.date_range('2026-01-01', periods=20, freq='h')
        close = pd.Series(100 + np.arange(20) * 0.1, index=idx)
        result = compute_structural_break_feature(close)
        assert result.index.equals(close.index)

    def test_sustained_trend_produces_larger_stat_than_flat_noise(self):
        # hand-traced expectation: a sustained one-directional price
        # move should trip the CSW statistic (a real departure from a
        # flat reference level) more than pure noise around a constant
        # level does.
        idx = pd.date_range('2026-01-01', periods=60, freq='h')
        rng = np.random.default_rng(0)

        close_trend = pd.Series(100 + np.arange(60) * 0.05, index=idx)
        stat_trend = compute_structural_break_feature(close_trend).dropna().max()

        close_flat = pd.Series(100 + rng.standard_normal(60) * 0.01, index=idx)
        stat_flat = compute_structural_break_feature(close_flat).dropna().max()

        assert stat_trend > stat_flat

    def test_bounded_lookback_matches_unbounded_when_series_shorter_than_lookback(self):
        # hand-traced: when max_lookback exceeds the series length, the
        # bounded search range [max(0, t-max_lookback), t) reduces to
        # exactly [0, t) -- identical to an unbounded search. Confirms
        # the bounding logic doesn't change results when it isn't
        # actually needed, only when it is.
        idx = pd.date_range('2026-01-01', periods=15, freq='h')
        rng = np.random.default_rng(1)
        close = pd.Series(100 + np.cumsum(rng.standard_normal(15) * 0.05), index=idx)

        result_generous = compute_structural_break_feature(close, max_lookback=1000)
        result_default = compute_structural_break_feature(close, max_lookback=200)
        pd.testing.assert_series_equal(result_generous, result_default)
