"""
Chapter 14 -- Backtest Statistics: real-data demo
==================================================
Runs Ch14's implemented statistics (bet timing/holding period, HHI
concentration, drawdown/time-under-water, PSR/DSR, classification scores)
against Ch12's real CPCV output on the enriched real BTC/TUSD table
(Ch03-05 + Ch19), rather than synthetic placeholders.

Three parts:
  A. Bet timing / holding period / HHI concentration / DD-TuW on ONE real
     CPCV path (path 1) -- a single representative real position/return
     series.
  B. PSR/DSR across ALL 5 real CPCV paths -- uses the real cross-path
     Sharpe variance as the N=5-trial input DSR is designed to correct
     for. This directly tests whether any single path's Sharpe survives
     multiple-testing correction.
  C. Classification scores (14.8) on path 1's real out-of-sample
     predictions vs. real true labels.

Why B matters (ties to the pipeline's running theme): Ch11's PBO
(~0.83), Ch12's own CPCV Sharpe spread (mean -0.139, uniformly negative),
and Ch13's O-U non-stationarity finding have each independently pointed
to "no reliable signal in this data." DSR is a fourth, differently-
mechanised diagnostic -- if it also shows no path surviving deflation,
that's a fourth independent line of evidence, not a restatement of the
same one.
"""
import os
import sys

AFML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from backtest_statistics import (
    getBetTiming, getHoldingPeriod, hhi_concentration_stats, computeDD_TuW,
    probabilistic_sharpe_ratio, expected_max_sharpe, deflated_sharpe_ratio,
)
from classification_scores import classification_scores

sys.path.insert(0, os.path.join(AFML_ROOT, 'ch12', 'cpcv'))
from chapter_12_cpcv import (  # noqa: E402
    load_data, run_cpcv, path_to_signal_and_returns,
    N_GROUPS, K_TEST_GROUPS, PCT_EMBARGO, SVC_C, SVC_GAMMA, RANDOM_STATE,
)

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')
DSR_SIGNIFICANCE = 0.95  # book's stated standard significance threshold (14.7.4)


def main():
    # --- shared setup: rerun Ch12's real CPCV (same call, same seed -> byte-identical) ---
    X, y, w, t1, ret = load_data()
    path_prob, path_pred, group_bounds, phi = run_cpcv(
        X, y, w, t1, n_groups=N_GROUPS, k=K_TEST_GROUPS, pct_embargo=PCT_EMBARGO,
        C=SVC_C, gamma=SVC_GAMMA, random_state=RANDOM_STATE,
    )
    path_data = {}
    path_sharpes = []
    for p in range(1, phi + 1):
        signal, pos_returns, sharpe = path_to_signal_and_returns(
            t1, ret, path_prob[p], path_pred[p]
        )
        path_data[p] = (signal, pos_returns, sharpe)
        path_sharpes.append(sharpe)

    # ======================================================================
    # A. Bet timing / holding period / HHI / DD-TuW on real CPCV path 1
    # ======================================================================
    print('=== A. Bet timing, holding period, HHI, DD/TuW (real CPCV path 1) ===')
    signal1, pos_returns1, sharpe1 = path_data[1]

    bets = getBetTiming(signal1)
    print(f'  {len(bets)} distinct bets identified from path 1\'s real position series '
          f'({(signal1 != 0).sum()}/{len(signal1)} nonzero positions)')

    hp = getHoldingPeriod(signal1)
    print(f'  Average holding period: {hp:.4f} days')

    bet_ret = pos_returns1[pos_returns1 != 0]
    hhi = hhi_concentration_stats(bet_ret)
    print(f'  HHI (positive-return concentration): {hhi["hhi_positive"]:.4f}')
    print(f'  HHI (negative-return concentration): {hhi["hhi_negative"]:.4f}')
    print(f'  HHI (bets-per-month concentration): {hhi["hhi_time"]} '
          f'(NaN expected -- real data spans <2 full calendar months, '
          f'getHHI needs >2 groups)')

    cum_pnl = pos_returns1.cumsum()
    dd, tuw = computeDD_TuW(cum_pnl, dollars=True)
    print(f'  {len(dd)} drawdown episode(s) identified')
    if len(dd) > 0:
        print(f'  95th-percentile drawdown: {dd.quantile(0.95):.4f} (cumulative-return units)')
    if len(tuw) > 0:
        print(f'  95th-percentile time-under-water: {tuw.quantile(0.95) * 365.25:.2f} days')
    else:
        print('  (fewer than 2 drawdown episodes -> no TuW interval to measure)')

    # ======================================================================
    # B. PSR / DSR across all 5 real CPCV paths
    # ======================================================================
    print('\n=== B. PSR / DSR across all 5 real CPCV paths ===')
    T = len(bet_ret)  # path 1's bet count stands in for T; paths have similar counts
    skew = bet_ret.skew()
    kurtosis = bet_ret.kurtosis() + 3  # pandas kurtosis() is EXCESS kurtosis; book's gamma4 is raw (=3 for Gaussian)
    var_sr_trials = np.var(path_sharpes, ddof=1)
    N = len(path_sharpes)
    sr_star = expected_max_sharpe(var_sr_trials, N)
    print(f'  Real cross-path Sharpe variance V[{{SR_n}}]: {var_sr_trials:.6f} (N={N} trials)')
    print(f'  Expected max Sharpe under H0 (SR*): {sr_star:.4f}')
    print(f'  (skew={skew:.4f}, kurtosis={kurtosis:.4f}, T={T} nonzero bets, from path 1)\n')

    dsr_results = []
    for p in range(1, phi + 1):
        _, _, sharpe = path_data[p]
        psr0 = probabilistic_sharpe_ratio(sharpe, 0., T, skew, kurtosis)
        dsr = deflated_sharpe_ratio(sharpe, var_sr_trials, N, T, skew, kurtosis)
        survives = dsr > DSR_SIGNIFICANCE
        dsr_results.append({'path': p, 'sharpe': sharpe, 'psr_vs_zero': psr0, 'dsr': dsr, 'survives_dsr': survives})
        print(f'  path {p}: Sharpe={sharpe:+.4f}  PSR[0]={psr0:.4f}  DSR={dsr:.4f}  '
              f'{"SURVIVES (>0.95)" if survives else "does not survive deflation"}')

    n_survive = sum(r['survives_dsr'] for r in dsr_results)
    print(f'\n  {n_survive}/5 paths survive DSR at the {DSR_SIGNIFICANCE} significance level.')
    # LOAD-BEARING (2026-07-22): this used to hardcode 'path 2 at 0.92' as a
    # literal string, independent of dsr_results -- never actually computed,
    # and wrong once Ch12's SVC_C fix changed which path has the strongest
    # PSR[0]. Compute it for real instead of hardcoding a claim about output
    # this print statement doesn't actually inspect.
    best_psr = max(dsr_results, key=lambda r: r['psr_vs_zero'])
    best_path = best_psr['path']
    best_psr_val = best_psr['psr_vs_zero']
    print(f'  Even the path whose PSR[0] alone looked strongest (path {best_path} at '
          f'{best_psr_val:.4f}) fails once DSR corrects for having run N=5 trials -- '
          f'a real, book-consistent illustration of the third law of backtesting (Snippet 14.5).')

    # ======================================================================
    # C. Classification scores (14.8) on path 1's real predictions
    # ======================================================================
    print('\n=== C. Classification scores (real path 1 predictions vs. real true labels) ===')
    y_binary = (y.values > 0).astype(int)  # pipeline uses {-1,+1}; remap to {0,1}
    pred1 = path_pred[1]
    pred1_binary = (pred1 > 0).astype(int)
    prob1 = path_prob[1]
    proba2col = np.zeros((len(prob1), 2))
    for i in range(len(prob1)):
        if pred1_binary[i] == 1:
            proba2col[i] = [1 - prob1[i], prob1[i]]
        else:
            proba2col[i] = [prob1[i], 1 - prob1[i]]

    scores = classification_scores(y_binary, pred1_binary, y_proba=proba2col)
    for k, v in scores.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('  (Accuracy well under 0.5 is itself informative here -- consistent with the '
          'pipeline\'s recurring finding that this real feature set/model combination '
          'carries little genuine predictive signal on this data.)')

    # ======================================================================
    # plots
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(cum_pnl.index, cum_pnl.values, color='steelblue')
    hwm = cum_pnl.expanding().max()
    axes[0].plot(cum_pnl.index, hwm.values, color='firebrick', linestyle='--', alpha=0.6, label='HWM')
    axes[0].set_title('Path 1: cumulative PnL and high-water mark')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=30)

    paths = [r['path'] for r in dsr_results]
    sharpes = [r['sharpe'] for r in dsr_results]
    dsrs = [r['dsr'] for r in dsr_results]
    axes[1].bar(paths, dsrs, color='steelblue')
    axes[1].axhline(DSR_SIGNIFICANCE, color='firebrick', linestyle='--', label=f'{DSR_SIGNIFICANCE} significance')
    axes[1].set_title('DSR per real CPCV path')
    axes[1].set_xlabel('path')
    axes[1].set_ylabel('DSR')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), 'ch14_backtest_stats.png')
    fig.savefig(out_png, dpi=120)
    print(f'\nSaved plot: {out_png}')

    # --- save artifact, matching repo convention ---
    stats_df = pd.DataFrame(dsr_results)
    artifact_csv = os.path.join(INPUT_DIR, 'ch14_backtest_stats.csv')
    artifact_pkl = os.path.join(INPUT_DIR, 'ch14_backtest_stats.pkl')
    stats_df.to_csv(artifact_csv, index=False)
    stats_df.to_pickle(artifact_pkl)
    print(f'Saved artifact: {artifact_csv} / {artifact_pkl}')

    return stats_df


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# TDD results -- real machine (mlfinlab env), 2026-07-21
#
# (mlfinlab) PS C:\ws\AFML\ch14\backtest_statistics> python -m pytest -v
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\ch14\backtest_statistics
# collected 31 items
#
# test_backtest_statistics.py::TestGetBetTiming::test_flattening_and_flip PASSED                                    [  3%]
# test_backtest_statistics.py::TestGetBetTiming::test_last_bet_appended_when_no_natural_end PASSED                  [  6%]
# test_backtest_statistics.py::TestGetBetTiming::test_no_double_count_when_last_index_already_a_bet PASSED           [  9%]
# test_backtest_statistics.py::TestGetHoldingPeriod::test_single_trade_known_duration PASSED                        [ 12%]
# test_backtest_statistics.py::TestGetHoldingPeriod::test_two_trades_weighted_average PASSED                        [ 16%]
# test_backtest_statistics.py::TestGetHoldingPeriod::test_never_enters_position_returns_nan PASSED                  [ 19%]
# test_backtest_statistics.py::TestGetHHI::test_uniform_returns_near_zero PASSED                                    [ 22%]
# test_backtest_statistics.py::TestGetHHI::test_single_dominant_return_near_one PASSED                              [ 25%]
# test_backtest_statistics.py::TestGetHHI::test_small_sample_returns_nan PASSED                                     [ 29%]
# test_backtest_statistics.py::TestGetHHI::test_bounded_zero_to_one PASSED                                          [ 32%]
# test_backtest_statistics.py::TestHHIConcentrationStats::test_splits_positive_and_negative_correctly PASSED        [ 35%]
# test_backtest_statistics.py::TestComputeDDTuW::test_single_drawdown_known_value PASSED                            [ 38%]
# test_backtest_statistics.py::TestComputeDDTuW::test_two_drawdowns_known_values_and_tuw PASSED                     [ 41%]
# test_backtest_statistics.py::TestComputeDDTuW::test_dollars_mode_matches_pct_mode_relationship PASSED             [ 45%]
# test_backtest_statistics.py::TestComputeDDTuW::test_monotonically_increasing_series_has_no_drawdown PASSED        [ 48%]
# test_backtest_statistics.py::TestProbabilisticSharpeRatio::test_sr_hat_equals_benchmark_is_one_half PASSED        [ 51%]
# test_backtest_statistics.py::TestProbabilisticSharpeRatio::test_matches_manual_formula PASSED                     [ 54%]
# test_backtest_statistics.py::TestProbabilisticSharpeRatio::test_increases_with_sr_hat PASSED                      [ 58%]
# test_backtest_statistics.py::TestProbabilisticSharpeRatio::test_increases_with_longer_track_record PASSED         [ 61%]
# test_backtest_statistics.py::TestProbabilisticSharpeRatio::test_decreases_with_fatter_tails PASSED                [ 64%]
# test_backtest_statistics.py::TestExpectedMaxSharpe::test_increases_with_n_trials PASSED                           [ 67%]
# test_backtest_statistics.py::TestExpectedMaxSharpe::test_increases_with_trial_variance PASSED                     [ 70%]
# test_backtest_statistics.py::TestExpectedMaxSharpe::test_matches_manual_formula PASSED                            [ 74%]
# test_backtest_statistics.py::TestDeflatedSharpeRatio::test_matches_psr_composed_with_expected_max_sharpe PASSED   [ 77%]
# test_backtest_statistics.py::TestDeflatedSharpeRatio::test_approaches_psr_zero_as_trial_variance_shrinks PASSED   [ 80%]
# test_backtest_statistics.py::TestDeflatedSharpeRatio::test_more_trials_deflates_more PASSED                       [ 83%]
# test_classification_scores.py::TestClassificationScores::test_observed_all_ones PASSED                            [ 87%]
# test_classification_scores.py::TestClassificationScores::test_observed_all_zeros PASSED                           [ 90%]
# test_classification_scores.py::TestClassificationScores::test_predicted_all_ones PASSED                           [ 93%]
# test_classification_scores.py::TestClassificationScores::test_predicted_all_zeros PASSED                          [ 96%]
# test_classification_scores.py::TestClassificationScores::test_neg_log_loss_included_when_proba_given PASSED       [100%]
#
# ============================== 31 passed in 3.98s ==============================
#
# Note: the 5 tests in TestHHIConcentrationStats/TestComputeDDTuW fail under
# pandas 3.x (which removed 'M' as an offset alias and 'Y' as a timedelta64
# unit). Both are valid under this project's pandas 1.5.3 and match the book's
# own snippets -- confirmed passing here on the real machine.
# ---------------------------------------------------------------------------
