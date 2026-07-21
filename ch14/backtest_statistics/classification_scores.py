"""
Chapter 14, Section 14.8 -- Classification scores for meta-labeling.

Implemented directly from the book's own equations (accuracy, precision,
recall, F1) using raw confusion-matrix counts, rather than via sklearn's
built-in scorers. This matters because Table 14.1 (the four degenerate
cases of binary classification) defines precision/recall as NaN when
their denominator is zero -- e.g. recall is undefined, not 0, when there
are no actual positives to begin with. sklearn's zero_division= behavior
differs across the sklearn versions this project has touched (1.2.2 in
mlfinlab; not consistent about NaN vs 0 pre-1.3), so computing directly
from TP/TN/FP/FN sidesteps that version dependency and matches the book
exactly.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, log_loss


def classification_scores(y_true, y_pred, y_proba=None):
    """AFML 14.8: accuracy, precision, recall, F1 (book's own formulas,
    NaN where undefined per Table 14.1), plus negative log-loss if
    predicted-class probabilities are supplied.

    Parameters
    ----------
    y_true, y_pred : array-like of {0, 1}
        True and predicted meta-labels.
    y_proba : array-like of shape (n, 2), optional
        Predicted class probabilities (as from predict_proba), for
        negative log-loss (Ch09 Section 9.4).

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, tp, tn, fp, fn,
    and neg_log_loss if y_proba was given.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = np.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    scores = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }
    if y_proba is not None:
        scores['neg_log_loss'] = -log_loss(y_true, y_proba, labels=[0, 1])
    return scores


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
