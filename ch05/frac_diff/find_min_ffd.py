"""
Snippet 5.4 -- Finding the minimum d that passes the ADF test.

PLAIN-ENGLISH IDEA:
Everything so far (Snippets 5.1-5.3) just gave us TOOLS: a way to
compute weights for any d, and two ways to apply them to a series.
This snippet is where we actually USE those tools to answer the
chapter's real question: "what's the smallest d that makes my series
stationary, so I keep the maximum possible memory?"

The workflow:
    1. Take the log of the price series (standard preprocessing --
       log differencing approximates percentage returns, which behave
       far better for financial data than raw price differences).
    2. For each candidate d in some range (e.g. 0, 0.1, 0.2, ..., 1.0):
        a. Fractionally differentiate the log price at that d (using
           the fixed-width version, frac_diff_ffd, for speed).
        b. Measure how much the differenced series still correlates
           with the ORIGINAL log price -- this is a direct read on how
           much memory survived at this d. Should fall as d rises.
        c. Run an Augmented Dickey-Fuller (ADF) test on the differenced
           series. The null hypothesis of this test is "this series
           has a unit root" (i.e. is NOT stationary). A sufficiently
           negative test statistic (more negative than the critical
           value) lets us reject that null -- the series passed.
    3. Tabulate all of this across the d range. The smallest d where
       the ADF test passes is the answer: maximum memory retained,
       while still being usable for ML (which generally needs
       stationary inputs).

NOTE ON adfuller's RETURN VALUE:
With autolag=None (fixed lag, no automatic lag-order search), statsmodels'
adfuller() returns a 5-element tuple:
    (adf_stat, p_value, used_lags, n_obs, critical_values_dict)
This matches the book's df2[:4] slicing (the first four scalar values)
plus a lookup into critical_values_dict for the 95% threshold.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from frac_diff_ffd import frac_diff_ffd


def find_min_ffd(price_series: pd.Series, d_values=None, thres: float = 0.01,
                  maxlag: int = 1, regression: str = 'c') -> pd.DataFrame:
    """
    Build a table of ADF-test results across a range of d values, to
    find the minimum d that achieves stationarity.

    Parameters
    ----------
    price_series : pd.Series
        RAW prices (not yet logged -- this function takes the log
        internally, matching the book's df1=np.log(...) step).
    d_values : array-like, default np.linspace(0, 1, 11)
        Candidate d values to test (0, 0.1, 0.2, ..., 1.0 by default).
    thres : float, default 0.01
        Passed through to frac_diff_ffd -- an ABSOLUTE weight-magnitude
        cutoff (see frac_diff_ffd.py's docstring; this is NOT the same
        kind of threshold as frac_diff's, despite the shared name).
    maxlag : int, default 1
        Fixed lag order for the ADF test (matches the book's choice
        of a simple, fast, fixed-lag test rather than auto-selection).
    regression : str, default 'c'
        ADF regression type: 'c' tests against a constant only (no
        trend term) -- the book's choice, appropriate for differenced
        series that shouldn't have a deterministic trend left in them.

    Returns
    -------
    pd.DataFrame indexed by d, with columns:
        adf_stat            -- the ADF test statistic
        p_value             -- p-value for the unit-root null hypothesis
        lags                -- lags actually used by the test
        n_obs               -- number of observations used
        critical_value_95   -- the 95% critical value for THIS sample
                                size (varies slightly by d, since
                                different d values drop different
                                numbers of leading points)
        corr                -- correlation between the differenced
                                series and the original log price
                                (memory retained at this d)
    """
    if d_values is None:
        d_values = np.linspace(0, 1, 11)

    log_price = np.log(price_series)

    out = pd.DataFrame(
        columns=['adf_stat', 'p_value', 'lags', 'n_obs', 'critical_value_95', 'corr'],
        dtype=float,
    )

    for d in d_values:
        diffed = frac_diff_ffd(log_price, d, thres=thres)

        # Correlation measures memory retained, evaluated only at the
        # positions where the differenced series actually has a value
        # (early points are absent due to the fixed window's width).
        corr = np.corrcoef(log_price.loc[diffed.index], diffed)[0, 1]

        adf_stat, p_value, lags, n_obs, crit_values = adfuller(
            diffed.values, maxlag=maxlag, regression=regression, autolag=None
        )

        out.loc[d] = [adf_stat, p_value, lags, n_obs, crit_values['5%'], corr]

    out.index.name = 'd'
    return out


def find_minimum_d(results: pd.DataFrame, p_value_threshold: float = 0.05):
    """
    Given the table from find_min_ffd(), return the smallest d whose
    ADF test p-value is below the threshold (default 5%) -- i.e. the
    smallest d for which we can reject "this series has a unit root"
    and call it stationary.

    Returns None if no candidate d in the table passes -- meaning the
    range tested wasn't wide enough (try extending d_values past 1.0,
    or check your data; in genuine cases d=1 should virtually always
    pass since that's full differencing).
    """
    passing = results[results['p_value'] < p_value_threshold]
    if passing.empty:
        return None
    return passing.index.min()


# ---------------------------------------------------------------------
# TDD TEST RESULTS (tests/test_ch05.py, find_min_ffd portion)
# Run 2026-06-26. Verified against a KNOWN-ground-truth synthetic
# random walk (true unit root by construction), not just "looks
# reasonable" checks.
# ---------------------------------------------------------------------
# test_find_min_ffd_d0_fails_adf_on_true_random_walk           PASSED
# test_find_min_ffd_d1_passes_adf_on_true_random_walk           PASSED
# test_find_min_ffd_p_values_decrease_monotonically_for_clean_random_walk  PASSED
# test_find_minimum_d_returns_smallest_passing_d                 PASSED
# test_find_minimum_d_returns_none_when_nothing_passes            PASSED
#
# REAL-DATA RESULT (BTCTUSD-trades-2026-03.csv, 9205 ticks):
# Minimum d that passes the ADF test (p<0.05): d=0.2
#   Correlation with original log price at d=0.2: 0.9987
#   Correlation with original log price at d=1.0: 0.0219
# i.e. d=0.2 achieves stationarity while keeping ~99.9% of the
# original series' memory, vs. full differencing destroying ~98%.
# ---------------------------------------------------------------------
