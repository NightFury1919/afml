"""
pipeline/orchestration/test_orchestration.py

Tests for the pipeline orchestration layer. Phase 1b reuses Ch11's real
trial-construction pipeline directly (SVC grid search, purged CV, bar-level
mark-to-market), so much of this module wraps real classifier training on
real data -- exact predicted values are neither reproducible across
sklearn versions nor the right thing to assert on (a deliberate, documented
departure from this project's usual hand-traced-exact-value convention,
same rationale as Ch22's io_benchmark.py timing tests). Structural
correctness (shapes, bounded ranges, deterministic pure-function behavior)
is tested instead. report.py's pure functions ARE hand-traceable and are
tested exactly.

Note: this file exercises Ch11's real SVC grid (20 trials x purged CV), so
the full suite takes noticeably longer to run than other chapters' test
suites -- expected, not a bug.
"""
import os

import pandas as pd
import pytest

from stages import (
    load_ch11_driver, run_real_trials, evaluate_overfitting,
    latest_bet_signal,
)
from report import build_report, _confidence_band

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
INPUT_DATA = os.path.join(ROOT, 'input_data')


@pytest.fixture(scope='module')
def ch11():
    return load_ch11_driver()


@pytest.fixture(scope='module')
def trials(ch11):
    return run_real_trials(ch11)


# ---------------------------------------------------------------------------
# run_real_trials (wraps Ch11's real part_c_build_trials)
# ---------------------------------------------------------------------------
class TestRunRealTrials:

    def test_trial_matrix_shape(self, trials):
        M, meta = trials
        assert M.shape[1] == 20  # 4 C values x 5 stepSize values
        assert len(meta) == 20

    def test_meta_has_expected_columns(self, trials):
        _, meta = trials
        for col in ('C', 'stepSize', 'pct_bars_in_market', 'sharpe_full_sample'):
            assert col in meta.columns

    def test_no_nan_in_pnl_matrix(self, trials):
        M, _ = trials
        assert not M.isna().any().any()


# ---------------------------------------------------------------------------
# evaluate_overfitting
# ---------------------------------------------------------------------------
class TestEvaluateOverfitting:

    def test_returns_expected_keys_and_bounded_ranges(self, ch11, trials):
        M, meta = trials
        result = evaluate_overfitting(M, meta, ch11, S=8)

        for key in ('trial_sharpes', 'best_trial', 'sr_hat', 'prob_overfit',
                    'cscv_df', 'n_trials', 'var_sr_trials', 'T', 'skew',
                    'kurtosis', 'dsr', 'meta'):
            assert key in result

        assert result['best_trial'] in meta.index
        assert 0.0 <= result['prob_overfit'] <= 1.0
        assert result['n_trials'] == 20
        assert result['T'] > 0
        # DSR is a probability (probabilistic_sharpe_ratio's output range)
        assert 0.0 <= result['dsr'] <= 1.0

    def test_reconciles_with_established_pbo(self, ch11, trials):
        # This project's established real-machine PBO on this exact trial
        # construction is ~0.83 (see pipeline/README.md). A wide tolerance
        # is used since CSCV's block-combination estimate has documented
        # run-to-run sensitivity even on identical data (see
        # ch11/backtest_dangers/pbo.py's own TDD notes) -- this test
        # guards against a REGRESSION back to Phase 1a's unreconciled
        # construction, not an exact value.
        M, meta = trials
        result = evaluate_overfitting(M, meta, ch11, S=8)
        assert result['prob_overfit'] > 0.5, (
            "PBO dropped well below this project's established ~0.83 "
            "finding on this same real trial construction -- check for a "
            "regression toward Phase 1a's weaker methodology"
        )


# ---------------------------------------------------------------------------
# latest_bet_signal
# ---------------------------------------------------------------------------
class TestLatestBetSignal:

    def test_signal_is_bounded_or_none(self, ch11, trials):
        M, meta = trials
        result = evaluate_overfitting(M, meta, ch11, S=8)
        signal = latest_bet_signal(result['best_trial'], meta, ch11, INPUT_DATA)
        assert signal is None or -1.0 <= signal <= 1.0


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

    def _fake_eval_result(self, T=119, n_trials=20, dsr=0.5445,
                           prob_overfit=0.8286, skew=0.0386, kurtosis=3.1186):
        return {
            'sr_hat': 0.0388,
            'prob_overfit': prob_overfit,
            'dsr': dsr,
            'n_trials': n_trials,
            'best_trial': 'C0.1_s0.1',
            'T': T,
            'skew': skew,
            'kurtosis': kurtosis,
        }

    def test_small_sample_triggers_warning(self):
        report = build_report(self._fake_eval_result(T=100, n_trials=3),
                               signal=0.10, asset_label='BTC/TUSD')
        assert 'SAMPLE SIZE WARNING' in report
        assert 'UNRELIABLE' in report

    def test_large_sample_does_not_trigger_warning(self):
        report = build_report(
            self._fake_eval_result(T=200, n_trials=20, dsr=0.97),
            signal=0.10, asset_label='BTC/TUSD',
        )
        assert 'SAMPLE SIZE WARNING' not in report
        assert 'high' in report

    def test_skew_kurtosis_shown_when_present(self):
        report = build_report(self._fake_eval_result(), signal=0.10,
                               asset_label='BTC/TUSD')
        assert 'skew=' in report
        assert 'kurtosis=' in report

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
        report = build_report(self._fake_eval_result(), signal=0.10,
                               asset_label='BTC/TUSD')
        assert 'NOT investment advice' in report
        lowered = report.lower()
        assert 'you should buy' not in lowered
        assert 'you should sell' not in lowered


class TestBuildReportRiskContext:
    """Phase 4 (2026-08-15): the optional risk-context section (Ch13 OTR,
    Ch15 strategy risk, rebuild.py's PT/SL). Uses build_report's SAME
    _fake_eval_result helper (inherited via a local copy, since these are
    two separate top-level classes in this file, matching this file's
    existing per-class-fixture style) and small, hand-built risk_context
    output dicts -- NOT re-testing risk_context.py's own computation
    (that's test_risk_context.py's job), only report.py's presentation
    logic over already-computed values."""

    def _fake_eval_result(self):
        return {
            'sr_hat': 0.0388, 'prob_overfit': 0.8286, 'dsr': 0.5445,
            'n_trials': 20, 'best_trial': 'C0.1_s0.1', 'T': 119,
            'skew': 0.0386, 'kurtosis': 3.1186,
        }

    def _fake_pt_sl_result(self):
        return {'pt_sl': [1, 1], 'latest_trgt': 0.015,
                'implied_pt_pct': 0.015, 'implied_sl_pct': 0.015}

    def test_no_risk_context_args_omits_section_entirely(self):
        # Backward-compat guard: run_pipeline.py's existing static-data
        # call doesn't pass these -- the section must not appear at all.
        report = build_report(self._fake_eval_result(), signal=0.10,
                               asset_label='BTC/TUSD')
        assert 'Risk Context' not in report

    def test_pt_sl_section_included_when_provided(self):
        report = build_report(
            self._fake_eval_result(), signal=0.10, asset_label='BTC/TUSD',
            pt_sl_result=self._fake_pt_sl_result(),
        )
        assert 'Risk Context' in report
        assert '+1.50%' in report
        assert '-1.50%' in report

    def test_otr_nonstationary_message(self):
        otr_result = {
            'phi_hat': 1.042, 'sigma_hat': 690.0, 'stationary': False,
            'half_life': float('nan'), 'n_opportunities': 87, 'best_node': None,
        }
        report = build_report(
            self._fake_eval_result(), signal=0.10, asset_label='BTC/TUSD',
            otr_result=otr_result,
        )
        assert 'NON-STATIONARY' in report
        assert 'random walk' in report
        assert 'phi_hat=1.0420' in report

    def test_otr_stationary_message_shows_best_node(self):
        otr_result = {
            'phi_hat': 0.5, 'sigma_hat': 10.0, 'stationary': True,
            'half_life': 4.2, 'n_opportunities': 50,
            'best_node': (5.0, 15.0, 1.2, 0.8, 1.5),
        }
        report = build_report(
            self._fake_eval_result(), signal=0.10, asset_label='BTC/TUSD',
            otr_result=otr_result,
        )
        assert 'STATIONARY' in report
        assert 'NON-STATIONARY' not in report
        assert 'Sharpe=1.5000' in report

    def test_strategy_risk_flags_too_risky_above_threshold(self):
        strategy_risk_result = {
            'p_fail': 0.45, 'freq_real': 366.5, 'p_bar': 0.55,
            'elapsed_years': 0.238, 'n_events': 87,
        }
        report = build_report(
            self._fake_eval_result(), signal=0.10, asset_label='BTC/TUSD',
            strategy_risk_result=strategy_risk_result,
        )
        assert 'TOO RISKY' in report

    def test_strategy_risk_within_threshold_when_low(self):
        strategy_risk_result = {
            'p_fail': 0.03, 'freq_real': 366.5, 'p_bar': 0.55,
            'elapsed_years': 0.238, 'n_events': 87,
        }
        report = build_report(
            self._fake_eval_result(), signal=0.10, asset_label='BTC/TUSD',
            strategy_risk_result=strategy_risk_result,
        )
        assert 'TOO RISKY' not in report
        assert 'within the book' in report


# ---------------------------------------------------------------------------
# TDD results -- REAL-MACHINE CONFIRMED 2026-08-15
# (mlfinlab env: Python 3.10.20, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2)
#
# Two-pass run (per project convention), all 23 tests in this file (17
# pre-existing + 6 new TestBuildReportRiskContext tests added 2026-08-15):
#   PASS 1 -- from repo root: 23 passed (part of a 32-item combined run
#     with test_risk_context.py, 10.84s)
#   PASS 2 -- from pipeline/orchestration/: 23 passed (part of the same
#     32-item combined run, 7.42s)
#
# This resolves the earlier "sandbox pre-check only" status this block
# used to carry -- the file (including Phase 1b's SVC-grid-dependent
# tests, which need a real ch11 run) is now real-machine confirmed, not
# just sandbox-verified. Original Phase 1b reconciliation numbers,
# preserved for history:
#   PBO = 0.8286 (established real-machine value: ~0.83)
#   DSR = 0.5445 (down from Phase 1a's unreconciled 0.9995)
#
# No bugs found in report.py/test_orchestration.py during this real-
# machine confirmation.
# ---------------------------------------------------------------------------
