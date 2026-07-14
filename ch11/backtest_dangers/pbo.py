"""
Chapter 11 -- The Dangers of Backtesting
========================================
Combinatorially Symmetric Cross-Validation (CSCV) and the
Probability of Backtest Overfitting (PBO).

AFML Chapter 11 contains NO numbered code snippets. Section 11.6 instead
describes the CSCV procedure as a seven-step algorithm in prose (following
Bailey et al. [2017a]). This module implements that prose algorithm
step-for-step; the step numbers in the comments below map directly onto the
book's own numbering.

--- Why this exists (plain English) ---
Chapter 11's central claim is that a backtest is not a research tool. If you
try N strategy configurations against the same history and keep whichever
one backtests best, you have not discovered an edge -- you have selected the
configuration that best fits THIS history's noise. Because you only ever
report the winner, every backtested strategy is overfit to some degree.

CSCV puts a number on that. It asks: "when I pick the best strategy
in-sample, does it STAY good out-of-sample?" If the in-sample winner
routinely lands in the bottom half out-of-sample, the selection procedure
itself is overfitting, and PBO -- the probability of exactly that happening
-- will be high.

--- The mechanism ---
Unlike walk-forward (which has a single path through the data, and can
therefore be re-run until a false positive appears), CSCV chops the
performance history into S blocks and forms EVERY balanced way of splitting
them into half-train / half-test. That combinatorial symmetry is what
removes the single-path degree of freedom.

--- Relationship to Chapter 12's CPCV (deliberately NOT shared code) ---
Ch12's Combinatorial Purged CV looks superficially similar -- both enumerate
combinations of time blocks -- but they answer different questions and share
almost nothing operationally:

    CSCV (11.6)                       CPCV (12.4)
    ----------------------------      ----------------------------
    input: PnL of ALREADY-RUN trials  input: raw labelled observations
    trains no classifier              fits C(N,k) classifiers
    no purging / no embargo           purges + embargoes every split
    test block count fixed at S/2     test block count k (usually 2)
    output: PBO (one probability)     output: phi backtest PATHS
    "was my SELECTION overfit?"       "what is this strategy's Sharpe
                                       DISTRIBUTION?"

The only common ground is "partition into blocks, enumerate combinations" --
about ten lines of itertools. Factoring that out would cost a student an
indirection hop out of the chapter they are reading, for no real gain, so
each chapter stays self-contained.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def sharpe_ratio(pnl):
    """
    Performance metric R. The book's only requirement (11.6) is that the
    metric 'can be estimated on subsamples of each column' -- i.e. it must
    still make sense when computed on half the rows. The Sharpe ratio
    qualifies, under the IID-normal assumption the book explicitly names.

    Not annualised: PBO depends only on the RANKING of trials within a
    subsample, and annualising multiplies every column by the same constant,
    which cannot change a ranking.
    """
    pnl = np.asarray(pnl, dtype=float)
    sd = pnl.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return pnl.mean() / sd


def cscv(M, S=8, metric=sharpe_ratio):
    """
    Combinatorially Symmetric Cross-Validation -- AFML Section 11.6.

    Parameters
    ----------
    M : pd.DataFrame, shape (T, N)
        Performance (PnL) series. One column per trial / model configuration
        the researcher tried; one row per synchronous time observation.
        The book's two conditions: M is a true matrix (same rows for every
        column, observations synchronous across trials), and `metric` is
        estimable on subsamples of a column.
    S : int, even
        Number of disjoint row-blocks. Produces C(S, S/2) combinations.
        Book's example: S=16 -> C(16,8) = 12,870 combinations.
        (NOTE -- the printed book says "12,780"; that is a digit
        transposition. C(16,8) is 12,870. Flagged, not propagated.)
    metric : callable
        Maps a 1-D PnL array to a scalar performance statistic.

    Returns
    -------
    pd.DataFrame with one row per combination c:
        n_star   : the trial chosen as best IN-SAMPLE (argmax of R)
        r_is     : that trial's in-sample statistic
        r_oos    : that same trial's OUT-OF-SAMPLE statistic
        rank_oos : its rank among all N trials out-of-sample (1 = worst)
        w_bar    : relative rank, omega_bar in (0, 1)
        logit    : lambda_c = log(w_bar / (1 - w_bar))
    """
    if S % 2 != 0:
        raise ValueError('S must be even -- CSCV splits blocks into two '
                         'equal halves of S/2 (Section 11.6).')
    if S > M.shape[0]:
        raise ValueError('S cannot exceed the number of rows in M.')
    if M.shape[1] < 2:
        raise ValueError('CSCV needs at least 2 trials to rank.')

    trials = list(M.columns)
    N = len(trials)
    values = M.to_numpy(dtype=float)

    # --- Step 2: partition M across rows into S disjoint submatrices -------
    # LOAD-BEARING: the book specifies submatrices "of equal dimensions"
    # (T/S rows each), which strictly requires S | T. Real data rarely
    # obliges. np.array_split keeps every observation, at the cost of blocks
    # differing by at most one row. Trimming to a multiple of S instead would
    # silently discard tail observations -- worse. The near-equality is
    # harmless because every statistic is computed per-subsample.
    blocks = np.array_split(np.arange(values.shape[0]), S)

    # --- Step 3: all combinations of S/2 blocks ---------------------------
    half = S // 2
    rows = []
    for c in combinations(range(S), half):
        # Step 4.1: training set J = the S/2 blocks in this combination
        is_idx = np.concatenate([blocks[b] for b in c])
        # Step 4.2: testing set J-bar = the complement of J in M
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in c])

        # Step 4.3: R = vector of performance statistics on the training set
        R = np.array([metric(values[is_idx, j]) for j in range(N)])
        # Step 4.4: n* = argmax_n {R_n} -- the trial we WOULD have selected
        if np.all(np.isnan(R)):
            continue
        n_star = int(np.nanargmax(R))

        # Step 4.5: R-bar = performance statistics on the testing set
        R_bar = np.array([metric(values[oos_idx, j]) for j in range(N)])

        # Step 4.6: relative rank of R_bar[n*] within R_bar, omega in (0,1).
        # rank 1 = worst, N = best. Dividing by (N+1) keeps omega strictly
        # inside (0,1) -- so the logit never blows up -- and places the
        # median at exactly 0.5, which is what makes lambda = 0 the
        # "no better than a coin flip" point the book describes.
        rank = pd.Series(R_bar).rank(method='average').iloc[n_star]
        w_bar = rank / (N + 1.0)

        # Step 4.7: logit. lambda_c = 0 <=> the IS winner landed exactly at
        # the OOS median. lambda > 0 => IS/OOS consistency (low overfitting).
        # lambda < 0 => the IS winner UNDERPERFORMED the median OOS.
        rows.append({
            'n_star': trials[n_star],
            'r_is': R[n_star],
            'r_oos': R_bar[n_star],
            'rank_oos': rank,
            'w_bar': w_bar,
            'logit': np.log(w_bar / (1.0 - w_bar)),
        })

    return pd.DataFrame(rows)


def pbo(M, S=8, metric=sharpe_ratio):
    """
    Probability of Backtest Overfitting -- AFML Section 11.6, final step.

    PBO = integral of f(lambda) from -inf to 0
        = the share of combinations in which the strategy that looked best
          IN-SAMPLE performed WORSE THAN MEDIAN out-of-sample.

    Reading the number:
        PBO ~ 0.0  selection is reliable; the IS winner stays a winner.
        PBO ~ 0.5  selection is a coin flip -- the "best" backtest carries
                   no information about future rank. This is what you should
                   expect from a family of zero-edge strategies, and it is
                   the quantitative form of the chapter's warning.
        PBO ~ 1.0  selection is actively harmful -- picking the IS winner
                   reliably picks an OOS loser (a strong overfitting signal).

    Returns (pbo_value, cscv_results_dataframe).
    """
    res = cscv(M, S=S, metric=metric)
    if res.empty:
        raise ValueError('CSCV produced no valid combinations.')
    value = float((res['logit'] < 0).mean())
    return value, res


# ===========================================================================
# TDD results (test_pbo.py), embedded per project convention.
# Expected values hand-derived from Section 11.6's seven-step algorithm.
# ===========================================================================
#
# ============================= test session starts =============================
# test_pbo.py::test_sharpe_hand_traced PASSED                              [  6%]
# test_pbo.py::test_sharpe_zero_variance_is_nan PASSED                     [ 12%]
# test_pbo.py::test_sharpe_is_scale_invariant PASSED                       [ 18%]
# test_pbo.py::test_cscv_row_count_equals_S_choose_half PASSED             [ 25%]
# test_pbo.py::test_cscv_book_example_S16_is_12870_not_12780 PASSED        [ 31%]
# test_pbo.py::test_cscv_rejects_odd_S PASSED                              [ 37%]
# test_pbo.py::test_cscv_rejects_single_trial PASSED                       [ 43%]
# test_pbo.py::test_cscv_S_larger_than_rows_rejected PASSED                [ 50%]
# test_pbo.py::test_train_and_test_sets_are_complementary_halves PASSED    [ 56%]
# test_pbo.py::test_logit_is_zero_at_median_rank PASSED                    [ 62%]
# test_pbo.py::test_omega_strictly_inside_unit_interval PASSED             [ 68%]
# test_pbo.py::test_pbo_is_zero_when_one_strategy_dominates_everywhere PASSED [ 75%]
# test_pbo.py::test_pbo_is_one_when_edge_is_purely_time_localised PASSED   [ 81%]
# test_pbo.py::test_pbo_is_high_when_each_trial_fits_its_own_slice_of_history PASSED [ 87%]
# test_pbo.py::test_pbo_averages_near_half_for_pure_noise PASSED           [ 93%]
# test_pbo.py::test_pbo_returns_value_and_frame PASSED                     [100%]
# ============================== 16 passed in 2.38s ==============================
#
# Notes on two tests that caught real misconceptions:
#
#  * test_pbo_averages_near_half_for_pure_noise -- PBO on N zero-edge trials
#    centres on ~0.5, but a SINGLE draw ranges roughly 0.04-0.99 (measured over
#    40 seeds). Any individual PBO estimate is therefore imprecise; the test
#    asserts the MEAN over seeds, not one seed. Asserting a single draw would
#    have been flaky AND would have taught students that one PBO number is
#    precise. It is not.
#
#  * test_cscv_book_example_S16_is_12870_not_12780 -- BOOK ERRATUM guard.
#    Section 11.6 prints "12,780" combinations for S=16. C(16,8) = 12,870.
#    A digit transposition. Pinned in a test so it can never enter the code.
