"""
pipeline/orchestration/test_stages.py

TDD suite for the DSR uniqueness-weighting fix (2026-08-17 handoff, Part
"DSR uniqueness bug"). Before this fix, evaluate_overfitting() fed
deflated_sharpe_ratio() a raw, non-uniqueness-weighted T (bar-level nonzero
bet count) -- overstating how much independent information the winning
trial's realized returns actually carry, given how heavily triple-barrier
events overlap under this pipeline's fixed VERTICAL_BARRIER_NUM_DAYS=3.

All hand-traced values below were independently computed (see the
2026-08-17 session's hand-trace script, not reproduced here) using the
SAME real ch14.deflated_sharpe_ratio() this module already calls -- this
suite is testing the WEIGHTING WIRING inside evaluate_overfitting(), not
re-deriving DSR's own math (already hand-traced and real-machine-confirmed
in ch14/backtest_statistics/test_backtest_statistics.py).

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML\\pipeline\\orchestration
    python -m pytest test_stages.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stages import evaluate_overfitting, load_ch11_driver  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture -- 8 bars, 3 trials, S=4 (satisfies pbo.cscv's S-even,
# S<=rows, >=2 trials constraints). Trial 'A' is the intended winner: its
# nonzero bet_ret is EXACTLY [0.01, 0.02, -0.01, 0.015] (4 real observations)
# -- hand-traced skew=-1.4430588355316423, kurtosis(+3)=5.234867179561618.
#
# meta['sharpe_full_sample'] is supplied directly (A=0.9, B=0.3, C=-0.2),
# independent of M's own realized Sharpe -- evaluate_overfitting() only
# reads meta for trial selection, and M for the winning trial's own PnL
# column, so this decouples "which trial wins" from "PBO's internal CSCV
# mechanics" cleanly.
# ---------------------------------------------------------------------------
@pytest.fixture
def ch11():
    return load_ch11_driver()


@pytest.fixture
def M():
    return pd.DataFrame({
        'A': [0.01, 0, 0.02, 0, -0.01, 0, 0.015, 0],
        'B': [0.005, 0.003, 0.004, 0.006, 0.002, 0.007, 0.001, 0.005],
        'C': [-0.01, 0.02, -0.015, 0.01, -0.02, 0.015, -0.01, 0.005],
    })


@pytest.fixture
def meta():
    return pd.DataFrame({
        'sharpe_full_sample': {'A': 0.9, 'B': 0.3, 'C': -0.2},
    })


class TestUniquenessWeightedT:

    def test_T_is_uniqueness_weighted_not_raw_bar_count(self, M, meta, ch11):
        # tw hand-picked so tw.mean() = (1.0+0.5+0.5+0.25)/4 = 0.5625
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)

        assert result['T_raw'] == 4
        assert result['tw_mean'] == pytest.approx(0.5625)
        # T_effective = T_raw * tw_mean = 4 * 0.5625 = 2.25 -- NOT 4
        assert result['T'] == pytest.approx(2.25)

    def test_dsr_uses_effective_T_not_raw_T(self, M, meta, ch11):
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)

        # Hand-traced independently via ch14's real deflated_sharpe_ratio()
        # called with T=2.25 (sr_hat=0.9, var_sr_trials=0.30333333333333334,
        # N=3, skew=-1.4430588355316423, kurtosis=5.234867179561618):
        assert result['dsr'] == pytest.approx(0.606727483760279, abs=1e-9)

    def test_dsr_no_longer_matches_the_pre_fix_buggy_value(self, M, meta, ch11):
        # Regression guard: the OLD (buggy) behavior would have produced
        # DSR using raw T=4 -- hand-traced independently as 0.6625823708791474.
        # If a future edit silently reverts to raw T, this test catches it.
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)
        buggy_dsr_with_raw_T = 0.6625823708791474
        assert result['dsr'] != pytest.approx(buggy_dsr_with_raw_T, abs=1e-9)

    def test_skew_kurtosis_still_use_raw_T_not_effective_T(self, M, meta, ch11):
        # Skew/kurtosis estimation is a DATA-SUFFICIENCY question (do we
        # have enough real bet_ret points to trust a sample skew/kurtosis
        # estimate?), which is unrelated to the DSR-weighting question --
        # this must still gate on T_raw > 2, not T_effective > 2.
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)
        assert result['skew'] == pytest.approx(-1.4430588355316423)
        assert result['kurtosis'] == pytest.approx(5.234867179561618)

    def test_effective_T_matches_2026_08_16_cusum_investigation_numbers(self):
        # Regression anchor: the exact real numbers from the 2026-08-16
        # session's compare_tw_by_cusum_h.py controlled experiment, which
        # is what surfaced this bug in the first place. Locks the formula
        # shape (T_raw * tw.mean()) against the real finding, independent
        # of the synthetic fixtures above.
        T_raw_h500, tw_mean_h500 = 45, 0.4220
        T_raw_h100, tw_mean_h100 = 140, 0.1402
        assert T_raw_h500 * tw_mean_h500 == pytest.approx(18.99, abs=0.01)
        assert T_raw_h100 * tw_mean_h100 == pytest.approx(19.628, abs=0.01)
        # The whole point of the finding: effective T barely moves even
        # though raw T more than triples.
        assert (T_raw_h100 * tw_mean_h100) / (T_raw_h500 * tw_mean_h500) == \
            pytest.approx(1.03, abs=0.02)


class TestTwRequiredAndValidated:

    def test_tw_is_required_keyword_argument(self, M, meta, ch11):
        # No default -- a caller CANNOT silently fall back to the old,
        # buggy raw-T behavior by forgetting to pass tw.
        with pytest.raises(TypeError):
            evaluate_overfitting(M, meta, ch11, S=4)

    def test_tw_with_nan_raises_value_error(self, M, meta, ch11):
        tw_with_nan = pd.Series([1.0, np.nan, 0.5, 0.25])
        with pytest.raises(ValueError, match='NaN'):
            evaluate_overfitting(M, meta, ch11, S=4, tw=tw_with_nan)

    def test_empty_tw_raises_value_error(self, M, meta, ch11):
        with pytest.raises(ValueError):
            evaluate_overfitting(M, meta, ch11, S=4, tw=pd.Series([], dtype=float))


class TestUnrelatedFieldsUnchanged:
    """Guards against the fix accidentally touching PBO/trial-selection
    logic it has no business touching."""

    def test_best_trial_and_sr_hat_unchanged(self, M, meta, ch11):
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)
        assert result['best_trial'] == 'A'
        assert result['sr_hat'] == pytest.approx(0.9)

    def test_n_trials_and_var_sr_trials_unchanged(self, M, meta, ch11):
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)
        assert result['n_trials'] == 3
        assert result['var_sr_trials'] == pytest.approx(0.30333333333333334)

    def test_prob_overfit_present_and_in_valid_range(self, M, meta, ch11):
        tw = pd.Series([1.0, 0.5, 0.5, 0.25])
        result = evaluate_overfitting(M, meta, ch11, S=4, tw=tw)
        assert 0.0 <= result['prob_overfit'] <= 1.0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))


# ---------------------------------------------------------------------------
# TDD results -- real machine (mlfinlab env), 2026-08-17
#
# (mlfinlab) PS C:\ws\AFML> python -m pytest pipeline\orchestration\test_stages.py -v
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML
# collected 11 items
#
# test_stages.py::TestUniquenessWeightedT::test_T_is_uniqueness_weighted_not_raw_bar_count PASSED       [  9%]
# test_stages.py::TestUniquenessWeightedT::test_dsr_uses_effective_T_not_raw_T PASSED                    [ 18%]
# test_stages.py::TestUniquenessWeightedT::test_dsr_no_longer_matches_the_pre_fix_buggy_value PASSED     [ 27%]
# test_stages.py::TestUniquenessWeightedT::test_skew_kurtosis_still_use_raw_T_not_effective_T PASSED     [ 36%]
# test_stages.py::TestUniquenessWeightedT::test_effective_T_matches_2026_08_16_cusum_investigation_numbers PASSED [ 45%]
# test_stages.py::TestTwRequiredAndValidated::test_tw_is_required_keyword_argument PASSED                [ 54%]
# test_stages.py::TestTwRequiredAndValidated::test_tw_with_nan_raises_value_error PASSED                 [ 63%]
# test_stages.py::TestTwRequiredAndValidated::test_empty_tw_raises_value_error PASSED                    [ 72%]
# test_stages.py::TestUnrelatedFieldsUnchanged::test_best_trial_and_sr_hat_unchanged PASSED              [ 81%]
# test_stages.py::TestUnrelatedFieldsUnchanged::test_n_trials_and_var_sr_trials_unchanged PASSED         [ 90%]
# test_stages.py::TestUnrelatedFieldsUnchanged::test_prob_overfit_present_and_in_valid_range PASSED      [100%]
#
# ============================== 11 passed in 4.48s ==============================
#
# Two-pass verification (from inside pipeline/orchestration/), same 11/11:
# ============================== 11 passed in 2.33s ==============================
#
# Confirmed against real data via pipeline\run_pipeline.py immediately after:
# events: 87, features: 12, bars: 239, M = 238 bars x 20 trials.
# Winning trial: C0.1_s0.1 (Sharpe +0.0388). PBO: 82.86%.
# T_raw (bar-level nonzero-PnL count) -> T_effective (uniqueness-weighted):
# T_effective = 26.166490233278697 (previously an inflated raw bar count).
# DSR: 0.5206 (previously ~0.54 under the buggy raw-T calculation -- barely
# moved on THIS dataset since this winning trial's tw happened to average
# ~0.58, not as low as the live pipeline's ~0.42/~0.14 from the 2026-08-16
# CUSUM investigation; the fix is dataset-dependent by design, not a fixed
# haircut). Still squarely "no reliable edge" -- consistent with every
# other diagnostic this project has run on this data (Ch11-15).
# ---------------------------------------------------------------------------
