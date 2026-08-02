"""
Chapter 17, Section 17.4.2: Supremum Augmented Dickey-Fuller (SADF)
=====================================================================

WHY THIS TEST EXISTS
---------------------
A standard unit-root test (plain ADF) asks one binary question about the
WHOLE sample: "does this series behave like a random walk, or like a
stationary/explosive process?" That's the wrong question for detecting
bubbles, because a bubble that inflates and then bursts can look, over
the full sample, statistically indistinguishable from ordinary noise --
the explosive run and the crash partially cancel each other out in a
single full-sample test.

SADF's fix: instead of running ONE ADF test on the whole sample, run MANY
ADF tests, each starting at a different (earlier) point t0 and always
ending at the current time t, then take the SUPREMUM (max) t-statistic
across all those starting points. If price behaved explosively for ANY
sub-window ending at t, at least one of those regressions will show it,
even if the full-sample regression wouldn't. Repeating this at every t
(the "outer loop", Sec 17.4.2, get_sadf below) produces a full SADF time
series -- it spikes during bubble-like runs and falls back down once the
bubble bursts (book's own Figure 17.1, E-mini S&P 500).

THE REGRESSION BEING FIT (Sec 17.4.2)
---------------------------------------
    delta_y_t = alpha + beta*y_{t-1} + sum_l(gamma_l * delta_y_{t-l}) + eps_t

testing H0: beta <= 0 (random walk / stationary) against H1: beta > 0
(explosive). The t-statistic on beta -- ADF_{t0,t} = beta_hat / se(beta_hat)
-- is what SADF takes the supremum of, across all valid start points t0.

Log prices, not raw prices, are used throughout (Sec 17.4.2.1): with log
prices, the price LEVEL conditions the MEAN of returns, not their
volatility, which is the correct regime for this test across decades of
data spanning very different price levels.

FUNCTION NAMES
---------------
Kept exactly as printed in the book (get_bsadf, getYX, lagDF, getBetas),
per this project's convention -- not rewritten to snake_case.
"""
import numpy as np
import pandas as pd


# =============================================================================
# Snippet 17.3: lagDF
# =============================================================================
def lagDF(df0, lags):
    """Apply a set of lags to every column of df0, returning one column per
    (original column, lag) pair, outer-joined on the index (so early rows
    that don't have a full set of lags come back as NaN -- callers dropna()
    afterward, see getYX).

    lags: either an int L (meaning "lags 0, 1, ..., L", via range(L+1) --
    note this INCLUDES lag 0, i.e. the unlagged series itself) or an
    explicit list/iterable of specific lag values to use.
    """
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(lags + 1)
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy(deep=True)
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        df1 = df1.join(df_, how='outer')
    return df1


# =============================================================================
# Snippet 17.2: getYX
# =============================================================================
def getYX(series, constant, lags):
    """Build the (y, x) arrays for the ADF regression
    delta_y_t = alpha + beta*y_{t-1} + sum_l(gamma_l*delta_y_{t-l}) + eps_t.

    Parameters
    ----------
    series : pd.DataFrame (single column) of log-prices, indexed by time.
        FIDELITY NOTE: the book's own docstring calls this "a pandas
        series", but the printed code's `series.values[..., 0]` 2D
        indexing only works if `series` is a single-column DataFrame --
        passing a genuine 1-D pd.Series crashes inside lagDF with an
        opaque `AttributeError: 'Series' object has no attribute
        'columns'` (confirmed by direct test). Rather than let students
        hit that confusing error, a bare Series passed in here is
        auto-converted to a single-column DataFrame -- LOAD-BEARING,
        do not remove: this is what makes the function's real, tested
        contract match its literal docstring.
    constant : {'nc', 'ct', 'ctt'}
        'nc'  -- no constant, no trend (just the level + lagged diffs)
        'ct'  -- constant + linear time trend
        'ctt' -- constant + linear AND quadratic time trend
    lags : int or list of int, passed through to lagDF.

    Returns
    -------
    y : ndarray, shape (n, 1) -- delta_y_t (the dependent variable)
    x : ndarray, shape (n, k) -- [y_{t-1}, delta_y_{t-1}, ..., delta_y_{t-L},
        (1 if constant != 'nc'), (trend if 'ct'/'ctt'), (trend**2 if 'ctt')]
    """
    if isinstance(series, pd.Series):
        series = series.to_frame()

    series_ = series.diff().dropna()
    x = lagDF(series_, lags).dropna()
    # Overwrite column 0 (which lagDF built as the UNLAGGED diff, i.e. lag 0)
    # with the actual LEVEL y_{t-1} -- the regression needs the level as its
    # beta regressor, not the contemporaneous diff, which isn't used at all.
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]  # lagged level
    y = series_.iloc[-x.shape[0]:].values

    # FIDELITY NOTE, fixed: as printed, x stays a pd.DataFrame when
    # constant=='nc' (the np.append branch below is what converts it to an
    # ndarray, and 'nc' skips that branch entirely) but becomes an ndarray
    # for 'ct'/'ctt' -- an inconsistent return TYPE depending on a
    # parameter value, confirmed by direct test (x[:, 0] raises
    # pandas.errors.InvalidIndexError for 'nc' where the same code works
    # fine for 'ct'/'ctt'). Downstream code happens to tolerate both types
    # (np.dot and positional slicing both work on a DataFrame here), so
    # this was never a runtime crash -- but a function silently returning
    # two different types based on an argument value is a real footgun for
    # any caller relying on x's type. Normalized to always return an
    # ndarray.
    x = np.asarray(x)

    if constant != 'nc':
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if constant[:2] == 'ct':
            trend = np.arange(x.shape[0]).reshape(-1, 1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            x = np.append(x, trend ** 2, axis=1)
    return y, x


# =============================================================================
# Snippet 17.4: getBetas
# =============================================================================
def getBetas(y, x):
    """OLS fit of y on x via the normal equations, returning both the point
    estimate (bMean) and its full variance-covariance matrix (bVar) --
    get_bsadf only ever needs bVar's [0,0] entry (beta's own variance), but
    the whole covariance matrix is returned exactly as printed, since a
    caller studying multi-lag specifications may want the rest of it.

    bVar's degrees-of-freedom divisor is (x.shape[0] - x.shape[1]) -- the
    number of observations minus the number of regressors, standard OLS.
    """
    xy = np.dot(x.T, y)
    xx = np.dot(x.T, x)
    xxinv = np.linalg.inv(xx)
    bMean = np.dot(xxinv, xy)
    err = y - np.dot(x, bMean)
    bVar = np.dot(err.T, err) / (x.shape[0] - x.shape[1]) * xxinv
    return bMean, bVar


# =============================================================================
# Snippet 17.1: get_bsadf (SADF's inner loop)
# =============================================================================
def get_bsadf(logP, minSL, constant, lags):
    """SADF's INNER loop: for a FIXED end point (the last row of logP),
    fit the ADF regression on every backward-expanding start point t0, and
    return the supremum ADF t-statistic across all of them --
    SADF_t = sup_{t0 in [1, t-tau]} { beta_hat_{t0,t} / se(beta_hat_{t0,t}) }.

    Parameters
    ----------
    logP : pd.DataFrame (single column) of log-prices, ending at the t we
        want SADF_t for. Only the LAST minSL..len(logP) rows are actually
        usable as regression windows -- see minSL below.
    minSL : int
        Minimum sample length (tau, book's notation) any individual
        regression window is allowed to use. Below this, an OLS fit
        on that window is either impossible or too noisy to trust.
    constant : {'nc', 'ct', 'ctt'} -- passed through to getYX.
    lags : int or list of int -- passed through to getYX.

    Returns
    -------
    dict with keys 'Time' (logP's last index value) and 'gsadf' (the
    supremum ADF statistic for that end point).

    REAL BOOK BUG, FIXED (confirmed by direct test, not just inspection):
    the book initializes `bsadf=None`, then does `if allADF[-1]>bsadf:
    bsadf=allADF[-1]` on the FIRST iteration. In Python 2, `float > None`
    was legal (None sorted below everything); in Python 3 this raises
    `TypeError: '>' not supported between instances of 'float' and
    'NoneType'` immediately, before the loop can ever produce a result --
    same category of Py2-vs-Py3 comparison-semantics trap as the Ch5
    tuple-assignment bug. Fixed by initializing bsadf = -np.inf, which
    preserves the original algorithm exactly (any real ADF statistic will
    correctly replace it on the first comparison) while being valid under
    Python 3's stricter comparison rules.
    """
    y, x = getYX(logP, constant=constant, lags=lags)
    startPoints = range(0, y.shape[0] + lags - minSL + 1)
    bsadf = -np.inf   # FIXED: book prints `None`, which crashes under Python 3
    allADF = []
    for start in startPoints:
        y_, x_ = y[start:], x[start:]
        bMean_, bStd_ = getBetas(y_, x_)
        bMean_, bStd_ = bMean_[0, 0], bStd_[0, 0] ** .5
        allADF.append(bMean_ / bStd_)
        if allADF[-1] > bsadf:
            bsadf = allADF[-1]
    out = {'Time': logP.index[-1], 'gsadf': bsadf}
    return out


# =============================================================================
# get_sadf: the OUTER loop (book's own description, Sec 17.4.2 -- "The outer
# loop (not shown here) repeats this calculation for an advancing t"). No
# code is printed for this in the book; written here directly from that
# description, not a fidelity-checked port of a snippet.
# =============================================================================
def get_sadf(logP, minSL, constant, lags):
    """Repeatedly call get_bsadf for every valid end point t = minSL+lags,
    ..., T, producing the full SADF_t time series described in Sec 17.4.2
    and plotted in the book's own Figure 17.1.

    Parameters
    ----------
    logP : pd.DataFrame (single column) of log-prices, full series.
    minSL, constant, lags : passed through to get_bsadf.

    Returns
    -------
    pd.Series of gsadf values, indexed by Time -- one entry per end point,
    starting at the earliest t for which a regression window of at least
    minSL observations exists.

    COMPUTATIONAL COST WARNING (book's own Sec 17.4.2.2): this is
    genuinely O(T^2) -- for each of ~T end points, get_bsadf itself loops
    over up to ~T start points. The book's own worked numbers for a
    ~350K-row E-mini dollar-bar series come out to ~242 PFLOPs for a full
    SADF series. On this project's real datasets (239 BTC bars, ~1750
    gold daily bars) this is comfortably fast, but this is NOT the
    algorithm to reach for on a long tick-level series without the
    parallelization strategies Ch20 discusses.
    """
    # +1 lag row is consumed by getYX's own diff+lag machinery before any
    # regression window can be formed, so the first usable end point is
    # minSL + lags rows into the series (matches get_bsadf's own
    # startPoints range logic for a single call).
    min_end = minSL + lags
    if len(logP) <= min_end:
        raise ValueError(
            f"logP has {len(logP)} rows, but minSL={minSL} + lags={lags} "
            f"requires at least {min_end + 1} rows to produce even one "
            f"SADF value."
        )
    results = []
    for t in range(min_end, len(logP)):
        out = get_bsadf(logP.iloc[:t + 1], minSL=minSL, constant=constant,
                         lags=lags)
        results.append(out)
    sadf = pd.DataFrame(results).set_index('Time')['gsadf']
    sadf.name = 'SADF'
    return sadf
