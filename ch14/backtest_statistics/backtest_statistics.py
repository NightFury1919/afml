"""
Chapter 14 -- Backtest Statistics.

AFML snippets 14.1 (bet timing), 14.2 (holding period), 14.3 (HHI
concentration), 14.4 (drawdown / time-under-water), plus the PSR (14.7.2)
and DSR (14.7.3) formulas (no book snippet exists for these -- implemented
directly from the printed equations).

Book-fidelity fixes applied vs. the raw printed snippets:
- Snippet 14.2: `xrange` -> `range` (book is Python 2; xrange is a
  SyntaxError under Python 3).
- Snippet 14.3: `pd.TimeGrouper(freq='M')` -> `pd.Grouper(freq='M')`.
  TimeGrouper was deprecated in pandas 0.21 and removed entirely by the
  pandas version this project uses (1.5.3); Grouper is its direct,
  behavior-identical successor. Not a book bug -- API rename only.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329  # gamma constant, used in DSR (14.7.3)


# --------------------------------------------------------------------------- #
# 14.3 General characteristics -- bet timing and holding period
# --------------------------------------------------------------------------- #
def getBetTiming(tPos):
    """AFML Snippet 14.1: derive the timestamps at which bets take place
    (flattening or flipping events) from a series of target positions.

    Parameters
    ----------
    tPos : pd.Series
        Target position sized over time, indexed by timestamp (e.g. Ch10's
        getSignal output, or Ch12's per-path CPCV signal).

    Returns
    -------
    pd.DatetimeIndex of bet-ending timestamps.
    """
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)  # flattening
    df0 = tPos.iloc[1:] * tPos.iloc[:-1].values
    bets = bets.union(df0[df0 < 0].index).sort_values()  # tPos flips
    if tPos.index[-1] not in bets:
        bets = bets.append(tPos.index[-1:])  # last bet
    return bets


def getHoldingPeriod(tPos):
    """AFML Snippet 14.2: estimate the weighted-average holding period (in
    days) of a strategy, using the average-entry-time pairing algorithm.

    FIX: book snippet uses Python 2's `xrange`; replaced with `range`
    (SyntaxError under Python 3, not a semantic change).
    """
    hp, tEntry = pd.DataFrame(columns=['dT', 'w']), 0.
    pDiff, tDiff = tPos.diff(), (tPos.index - tPos.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, tPos.shape[0]):
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0:  # increased or unchanged
            if tPos.iloc[i] != 0:
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else:  # decreased
            if tPos.iloc[i] * tPos.iloc[i - 1] < 0:  # flip
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i]  # reset entry time
            else:
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum()
    else:
        hp = np.nan
    return hp


# --------------------------------------------------------------------------- #
# 14.5.1 Runs -- HHI concentration
# --------------------------------------------------------------------------- #
def getHHI(betRet):
    """AFML Snippet 14.3 (inner function): Herfindahl-Hirschman concentration
    index of a return series. 0 = uniform, 1 = a single dominant return."""
    if betRet.shape[0] <= 2:
        return np.nan
    wght = betRet / betRet.sum()
    hhi = (wght ** 2).sum()
    hhi = (hhi - betRet.shape[0] ** -1) / (1. - betRet.shape[0] ** -1)
    return hhi


def hhi_concentration_stats(ret):
    """AFML Snippet 14.3 (outer calls): HHI on positive returns, negative
    returns, and monthly bet count -- the three concentration diagnostics
    the book computes together.

    FIX: `pd.TimeGrouper(freq='M')` -> `pd.Grouper(freq='M')` (see module
    header).
    """
    rHHIPos = getHHI(ret[ret >= 0])
    rHHINeg = getHHI(ret[ret < 0])
    tHHI = getHHI(ret.groupby(pd.Grouper(freq='M')).count())
    return {'hhi_positive': rHHIPos, 'hhi_negative': rHHINeg, 'hhi_time': tHHI}


# --------------------------------------------------------------------------- #
# 14.5.2 Runs -- drawdown and time-under-water
# --------------------------------------------------------------------------- #
def computeDD_TuW(series, dollars=False):
    """AFML Snippet 14.4: derive the sequence of drawdowns and the
    time-under-water associated with each, from a cumulative PnL or
    return series."""
    df0 = series.to_frame('pnl')
    df0['hwm'] = series.expanding().max()
    df1 = df0.groupby('hwm').min().reset_index()
    df1.columns = ['hwm', 'min']
    df1.index = df0['hwm'].drop_duplicates(keep='first').index  # time of hwm
    df1 = df1[df1['hwm'] > df1['min']]  # hwm followed by a drawdown
    if dollars:
        dd = df1['hwm'] - df1['min']
    else:
        dd = 1 - df1['min'] / df1['hwm']
    tuw = ((df1.index[1:] - df1.index[:-1]) / np.timedelta64(1, 'Y')).values  # in years
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


# --------------------------------------------------------------------------- #
# 14.7.2 / 14.7.3 -- Probabilistic and Deflated Sharpe Ratio
# (no book snippet exists for either; implemented directly from the
# printed equations)
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(sr_hat, sr_benchmark, T, skew=0., kurtosis=3.):
    """AFML 14.7.2, eq. (no snippet number):
    PSR[SR*] = Z[ (SR_hat - SR*) * sqrt(T-1) / sqrt(1 - skew*SR_hat + (kurtosis-1)/4 * SR_hat^2) ]

    Parameters
    ----------
    sr_hat : observed (non-annualized) Sharpe ratio.
    sr_benchmark : benchmark SR* to test against (0. = "no skill").
    T : number of observed returns.
    skew, kurtosis : sample skewness/kurtosis of the return series
        (kurtosis=3. for Gaussian returns, matching the book's convention).
    """
    numerator = (sr_hat - sr_benchmark) * np.sqrt(T - 1)
    denominator = np.sqrt(1 - skew * sr_hat + (kurtosis - 1) / 4 * sr_hat ** 2)
    return norm.cdf(numerator / denominator)


def expected_max_sharpe(var_sr_trials, N):
    """AFML 14.7.3, eq. (no snippet number): SR*, the expected maximum
    Sharpe ratio across N independent trials under H0: true SR=0.

    SR* = sqrt(V[{SR_n}]) * ( (1-gamma)*Z^-1[1-1/N] + gamma*Z^-1[1-1/(N*e)] )
    """
    z1 = norm.ppf(1 - 1.0 / N)
    z2 = norm.ppf(1 - 1.0 / (N * np.e))
    return np.sqrt(var_sr_trials) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)


def deflated_sharpe_ratio(sr_hat, var_sr_trials, N, T, skew=0., kurtosis=3.):
    """AFML 14.7.3: DSR = PSR[SR*], where SR* is estimated (not user-set)
    from the variance and count of independent trials, correcting for
    selection bias under multiple testing."""
    sr_star = expected_max_sharpe(var_sr_trials, N)
    return probabilistic_sharpe_ratio(sr_hat, sr_star, T, skew, kurtosis)


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
