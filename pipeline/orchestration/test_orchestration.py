"""
pipeline/orchestration/test_orchestration.py

Tests for the pipeline orchestration layer. Unlike most chapters' tests,
much of this module wraps real classifier training on real data -- exact
predicted values are neither reproducible across sklearn versions nor the
right thing to assert on (a deliberate, documented departure from this
project's usual hand-traced-exact-value convention, same rationale as
Ch22's io_benchmark.py timing tests). Structural correctness (shapes,
index alignment, bounded ranges, deterministic pure-function behavior) is
tested instead. report.py's pure functions ARE hand-traceable and are
tested exactly.
"""
import os

import numpy as np
import pandas as pd
import pytest

from stages import (
    FEATURE_COLS, load_enriched_table, default_trials,
    run_trial_oos_pnl, assemble_pnl_matrix, evaluate_overfitting,
    latest_bet_signal,
)
from report import build_report, _confidence_band

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
INPUT_DATA = os.path.join(ROOT, 'input_data')


# ---------------------------------------------------------------------------
# load_enriched_table
# ---------------------------------------------------------------------------
class TestLoadEnrichedTable:

    def test_real_data_shapes_and_alignment(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        assert list(X.columns) == FEATURE_COLS
        assert len(X) == len(y) == len(w) == len(t1) == len(ret)
        assert X.index.equals(y.index)
        assert X.index.equals(w.index)
        assert X.index.equals(t1.index)
        assert X.index.equals(ret.index)
        assert X.index.is_monotonic_increasing

    def test_labels_are_plus_minus_one(self):
        _, y, _, _, _ = load_enriched_table(INPUT_DATA)
        assert set(y.unique()) <= {-1.0, 1.0}

    def test_t1_strictly_after_t0(self):
        X, _, _, t1, _ = load_enriched_table(INPUT_DATA)
        assert (t1.values > X.index.values).all()


# ---------------------------------------------------------------------------
# run_trial_oos_pnl / assemble_pnl_matrix
# ---------------------------------------------------------------------------
class TestAssemblePnlMatrix:

    def test_all_trials_share_identical_synchronous_index(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = default_trials()
        M, trial_probs = assemble_pnl_matrix(X, y, w, t1, ret, trials)

        # PBO's cscv() requires a true (T, N) matrix -- same rows for every
        # column. Verify every trial column covers the identical index.
        assert list(M.columns) == list(trials.keys())
        assert not M.isna().any().any(), "every observation should be " \
            "covered by exactly one PurgedKFold test fold, for every trial"
        for name in trials:
            assert trial_probs[name].index.equals(M.index)

    def test_pnl_matrix_row_count_matches_input(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = {'rf_shallow': default_trials()['rf_shallow']}
        M, _ = assemble_pnl_matrix(X, y, w, t1, ret, trials)
        assert M.shape == (len(X), 1)

    def test_probabilities_sum_to_one_per_row(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = {'rf_shallow': default_trials()['rf_shallow']}
        _, trial_probs = assemble_pnl_matrix(X, y, w, t1, ret, trials)
        row_sums = trial_probs['rf_shallow'].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# evaluate_overfitting
# ---------------------------------------------------------------------------
class TestEvaluateOverfitting:

    def test_returns_expected_keys_and_bounded_ranges(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = default_trials()
        M, _ = assemble_pnl_matrix(X, y, w, t1, ret, trials)
        result = evaluate_overfitting(M, S=8)

        for key in ('trial_sharpes', 'best_trial', 'sr_hat', 'prob_overfit',
                    'cscv_df', 'n_trials', 'var_sr_trials', 'T', 'dsr'):
            assert key in result

        assert result['best_trial'] in trials
        assert 0.0 <= result['prob_overfit'] <= 1.0
        assert result['n_trials'] == len(trials)
        assert result['T'] == M.shape[0]
        # DSR is a probability (probabilistic_sharpe_ratio's output range)
        assert 0.0 <= result['dsr'] <= 1.0

    def test_rejects_fewer_than_two_trials(self):
        # cscv() itself enforces this (ch11/backtest_dangers/pbo.py); verify
        # the orchestration layer surfaces the same real error, not a
        # silently wrong result.
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = {'rf_shallow': default_trials()['rf_shallow']}
        M, _ = assemble_pnl_matrix(X, y, w, t1, ret, trials)
        with pytest.raises(ValueError):
            evaluate_overfitting(M, S=8)


# ---------------------------------------------------------------------------
# latest_bet_signal
# ---------------------------------------------------------------------------
class TestLatestBetSignal:

    def test_signal_is_bounded_or_none(self):
        X, y, w, t1, ret = load_enriched_table(INPUT_DATA)
        trials = default_trials()
        M, trial_probs = assemble_pnl_matrix(X, y, w, t1, ret, trials)
        result = evaluate_overfitting(M, S=8)
        signal = latest_bet_signal(result['best_trial'], trial_probs, t1)
        assert signal is None or -1.0 <= signal <= 1.0

    def test_empty_probs_returns_none(self):
        empty_probs = {'trial': pd.DataFrame()}
        t1 = pd.Series(dtype='datetime64[ns]')
        assert latest_bet_signal('trial', empty_probs, t1) is None


# ---------------------------------------------------------------------------
# report.py -- pure functions, hand-traceable exactly
# ---------------------------------------------------------------------------
class TestConfidenceBand:

    def test_high_at_and_above_0_95(self):
        assert _confidence_band(0.95) == "high"
        assert _confidence_band(0.999) == "high"

    def test_moderate_between_0_5_and_0_95(self):
        assert _confidence_band(0.5) == "moderate"
        assert _confidence_band(0.94) == "moderate"

    def test_low_below_0_5(self):
        assert _confidence_band(0.49) == "low"
        assert _confidence_band(0.0) == "low"

    def test_nan_is_undetermined(self):
        assert _confidence_band(float('nan')) == \
            "undetermined (insufficient data for a deflated Sharpe estimate)"

    def test_none_is_undetermined(self):
        assert _confidence_band(None) == \
            "undetermined (insufficient data for a deflated Sharpe estimate)"


class TestBuildReport:

    def _fake_eval_result(self, T=87, n_trials=3, dsr=0.9995,
                           prob_overfit=0.0429):
        return {
            'sr_hat': 0.4949,
            'prob_overfit': prob_overfit,
            'dsr': dsr,
            'n_trials': n_trials,
            'best_trial': 'rf_shallow',
            'T': T,
        }

    def test_small_sample_triggers_warning(self):
        report = build_report(self._fake_eval_result(T=87, n_trials=3),
                               signal=0.65, asset_label='BTC/TUSD')
        assert 'SAMPLE SIZE WARNING' in report
        assert 'UNRELIABLE' in report

    def test_large_sample_does_not_trigger_warning(self):
        report = build_report(
            self._fake_eval_result(T=500, n_trials=10, dsr=0.97),
            signal=0.65, asset_label='BTC/TUSD',
        )
        assert 'SAMPLE SIZE WARNING' not in report
        assert 'high' in report

    def test_none_signal_handled(self):
        report = build_report(self._fake_eval_result(), signal=None,
                               asset_label='BTC/TUSD')
        assert 'No current bet-sizing signal' in report

    def test_negative_signal_reports_short(self):
        report = build_report(self._fake_eval_result(), signal=-0.4,
                               asset_label='BTC/TUSD')
        assert 'short' in report

    def test_report_never_issues_a_buy_sell_directive(self):
        # Scope guard: the report must present evidence, not a command.
        report = build_report(self._fake_eval_result(), signal=0.65,
                               asset_label='BTC/TUSD')
        assert 'NOT investment advice' in report
        lowered = report.lower()
        assert 'you should buy' not in lowered
        assert 'you should sell' not in lowered


# ---------------------------------------------------------------------------
# TDD results -- SANDBOX pre-check only, NOT YET real-machine confirmed.
# Sandbox: Python 3.12.3, pytest 9.1.1, pandas 3.0.2, numpy 2.4.4,
# scipy 1.17.1, scikit-learn 1.8.0 -- all NEWER than mlfinlab's pinned
# versions (pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2). Per project
# convention, this sandbox pass does not substitute for real-machine
# confirmation.
#
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 20 items
#
# TestLoadEnrichedTable::test_real_data_shapes_and_alignment PASSED        [  5%]
# TestLoadEnrichedTable::test_labels_are_plus_minus_one PASSED             [ 10%]
# TestLoadEnrichedTable::test_t1_strictly_after_t0 PASSED                  [ 15%]
# TestAssemblePnlMatrix::test_all_trials_share_identical_synchronous_index PASSED [ 20%]
# TestAssemblePnlMatrix::test_pnl_matrix_row_count_matches_input PASSED    [ 25%]
# TestAssemblePnlMatrix::test_probabilities_sum_to_one_per_row PASSED      [ 30%]
# TestEvaluateOverfitting::test_returns_expected_keys_and_bounded_ranges PASSED [ 35%]
# TestEvaluateOverfitting::test_rejects_fewer_than_two_trials PASSED       [ 40%]
# TestLatestBetSignal::test_signal_is_bounded_or_none PASSED               [ 45%]
# TestLatestBetSignal::test_empty_probs_returns_none PASSED                [ 50%]
# TestConfidenceBand::test_high_at_and_above_0_95 PASSED                   [ 55%]
# TestConfidenceBand::test_moderate_between_0_5_and_0_95 PASSED            [ 60%]
# TestConfidenceBand::test_low_below_0_5 PASSED                            [ 65%]
# TestConfidenceBand::test_nan_is_undetermined PASSED                      [ 70%]
# TestConfidenceBand::test_none_is_undetermined PASSED                     [ 75%]
# TestBuildReport::test_small_sample_triggers_warning PASSED               [ 80%]
# TestBuildReport::test_large_sample_does_not_trigger_warning PASSED       [ 85%]
# TestBuildReport::test_none_signal_handled PASSED                        [ 90%]
# TestBuildReport::test_negative_signal_reports_short PASSED               [ 95%]
# TestBuildReport::test_report_never_issues_a_buy_sell_directive PASSED    [100%]
#
# ======================== 20 passed, 3 warnings in 8.77s =========================
# (The 3 warnings are BaggingClassifier's small-bootstrap-sample warning,
# same harmless/expected warning already documented in
# ch07/cross_validation/purged_kfold.py's own TDD notes.)
#
# STILL NEEDED before this is real-machine confirmed:
#   conda activate mlfinlab
#   cd C:\ws\AFML
#   python -m pytest pipeline\orchestration\ -v
#   cd pipeline\orchestration
#   python -m pytest -v
# ---------------------------------------------------------------------------
