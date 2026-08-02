"""
Chapter 17, Section 17.3: CUSUM Tests
=======================================

Both tests below ask the same basic question -- "has the process's behavior
drifted away from what we'd expect under a stable, no-structural-break
null hypothesis?" -- but from two very different angles, and with very
different data requirements. Neither has printed book code (Sec 17.3.1
and 17.3.2 are formula-only), so everything here is written directly from
those formulas, then verified against hand-computed or synthetically
injected-break test cases -- there's no printed snippet to diff against
or "book bug" to find here.

17.3.1 -- Brown-Durbin-Evans (BDE) CUSUM on recursive residuals
-------------------------------------------------------------------
Needs a genuine PREDICTIVE regression: features x_t forecasting a target
y_t (y_t = beta'x_t + eps_t). Refits beta on expanding windows and tracks
whether the FORECAST ERRORS (using each window's beta to predict the very
next, still-unseen point) start drifting away from zero in a sustained
way. That sustained drift is the break signal: the relationship between
features and target has changed, not just the level of y itself.

17.3.2 -- Chu-Stinchcombe-White (CSW) CUSUM on levels
--------------------------------------------------------
Drops the feature/target regression setup entirely (H0: beta_t = 0, i.e.
"no forecastable change") and just asks whether the LOG-PRICE LEVEL has
drifted too far from some earlier reference level, relative to how
volatile the series has been in between. Needs only a single price
series, not a feature matrix.
"""
import numpy as np
import pandas as pd


# =============================================================================
# 17.3.1 -- Brown-Durbin-Evans CUSUM on recursive residuals
# =============================================================================
def get_bde_recursive_residuals(X, y, min_sample):
    """Fit y_t = beta'x_t + eps_t by OLS on expanding windows [1, k+1],
    [1, k+2], ..., [1, T] (book's own subsample notation), and compute the
    standardized ONE-STEP-AHEAD recursive residual at each step:

        omega_t = (y_t - beta_hat_{t-1}'x_t) / sqrt(f_t)
        f_t = sigma_hat_eps^2 * [1 + x_t'(X_{t-1}'X_{t-1})^-1 x_t]

    where beta_hat_{t-1} is fit using ONLY data through t-1 -- a genuine
    out-of-sample forecast of the still-unseen point t, not an in-sample
    residual.

    INDEXING NOTE (book gives no code, and its own prose notation is a
    little loose here -- the book writes both "the first estimate is
    beta_hat_{k+1}, fit on subsample [1,k+1]" AND "St sums omega_j from
    j=k+1", which would need omega_{k+1} to use beta_hat_k -- an estimate
    the book's own list of "T-k estimates (beta_hat_{k+1},...,beta_hat_T)"
    doesn't include). Adopted convention here, self-consistent and
    testable: `min_sample` is the size of the FIRST fit window (points
    0..min_sample-1); the first recursive residual is computed at index
    min_sample, using beta fit on points 0..min_sample-1 to forecast point
    min_sample. This is the same recursive-least-squares idea the book
    describes, just with an unambiguous starting index.

    Parameters
    ----------
    X : ndarray, shape (T, p) -- feature matrix, include a constant column
        yourself if you want an intercept (not added automatically).
    y : ndarray, shape (T,) -- target series, same length as X.
    min_sample : int -- size of the first fit window. Must exceed p (the
        number of regressors) or the first OLS fit is singular.

    Returns
    -------
    times : ndarray of int -- 0-indexed positions (into X/y) of each
        recursive residual, i.e. min_sample, min_sample+1, ..., T-1.
    omega : ndarray of float -- the standardized recursive residuals,
        same length as times.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    T, p = X.shape
    if min_sample <= p:
        raise ValueError(
            f"min_sample={min_sample} must exceed the number of "
            f"regressors ({p}) or the first OLS fit is singular."
        )
    times, omega = [], []
    for t in range(min_sample, T):
        X_prev, y_prev = X[:t], y[:t]
        XtX_inv = np.linalg.inv(X_prev.T @ X_prev)
        beta = XtX_inv @ (X_prev.T @ y_prev)
        dof = t - p
        sigma2 = np.sum((y_prev - X_prev @ beta) ** 2) / dof
        x_t = X[t]
        f_t = sigma2 * (1 + x_t @ XtX_inv @ x_t)
        times.append(t)
        omega.append((y[t] - x_t @ beta) / np.sqrt(f_t))
    return np.array(times), np.array(omega)


def get_bde_cusum(X, y, min_sample, index=None):
    """Full BDE CUSUM statistic S_t (Sec 17.3.1):

        S_t = sum_{j=k+1}^{t} omega_hat_j / sigma_hat_omega
        sigma_hat_omega^2 = (1/(T-k)) * sum_t (omega_hat_t - E[omega_hat_t])^2

    E[omega_hat_t] is unknown in practice; estimated here by the sample
    mean of the recursive residuals themselves (should be near zero under
    H0, but computed as an empirical mean rather than assumed to be
    exactly zero -- more honest about what's actually being estimated).

    Under H0 (beta constant throughout), the book states S_t ~
    N[0, t-k-1]. A conventional two-sided 5% band, +-1.96*sqrt(t-k-1), is
    included in the returned DataFrame as a convenience -- this is just
    the standard normal quantile applied to the book's OWN stated null
    distribution, not a separately printed book formula, and is flagged
    as such rather than presented as book content.

    Parameters
    ----------
    X, y, min_sample : passed through to get_bde_recursive_residuals.
    index : optional pd.Index (e.g. dates) to label the output by; if
        omitted, the raw 0-indexed positions from get_bde_recursive_residuals
        are used.

    Returns
    -------
    pd.DataFrame with columns ['omega', 'S', 'band_95'], indexed by
    `index` (or raw position if index is None).
    """
    times, omega = get_bde_recursive_residuals(X, y, min_sample)
    if len(omega) < 2:
        raise ValueError(
            "Fewer than 2 recursive residuals produced -- need a longer "
            "series or a smaller min_sample."
        )
    sigma_omega = np.std(omega - omega.mean(), ddof=0)
    if sigma_omega == 0:
        raise ValueError("Recursive residuals have zero variance -- "
                          "cannot standardize.")
    S = np.cumsum(omega) / sigma_omega
    # S_t's own construction starts summing from the first residual, so
    # "t - k - 1" in the book's stated N[0, t-k-1] corresponds to the
    # (1-indexed) count of residuals summed so far, minus 1.
    n_summed = np.arange(1, len(omega) + 1)
    band_95 = 1.96 * np.sqrt(np.maximum(n_summed - 1, 0))

    out = pd.DataFrame({'omega': omega, 'S': S, 'band_95': band_95})
    if index is not None:
        out.index = index[times]
    else:
        out.index = times
    return out


# =============================================================================
# 17.3.2 -- Chu-Stinchcombe-White CUSUM on levels
# =============================================================================
def _prep_log_series(series):
    """Accept pd.Series or single-column pd.DataFrame; return values+index."""
    if isinstance(series, pd.DataFrame):
        return series.iloc[:, 0].values, series.index
    return series.values, series.index


def get_csw_stat(logP, n_idx, t_idx):
    """Single Chu-Stinchcombe-White statistic S_{n,t} (Sec 17.3.2):

        S_{n,t} = (y_t - y_n) / (sigma_hat_t * sqrt(t - n))
        sigma_hat_t^2 = (1/(t-1)) * sum_{i=2}^{t} (Delta y_i)^2

    comparing the log-price at t against a REFERENCE level at an earlier
    point n < t, standardized by the series' own realized volatility
    through t. Under H0 (no drift, beta_t=0), S_{n,t} ~ N[0,1].

    Parameters
    ----------
    logP : pd.Series or single-column pd.DataFrame of log-prices.
    n_idx, t_idx : int, 0-indexed positions into logP with n_idx < t_idx.

    Returns
    -------
    float, or NaN if t_idx doesn't have enough history to estimate
    sigma_hat_t (needs at least 2 price observations, i.e. t_idx >= 1).
    """
    values, _ = _prep_log_series(logP)
    if t_idx <= n_idx:
        raise ValueError(f"t_idx ({t_idx}) must exceed n_idx ({n_idx}).")
    diffs = np.diff(values[:t_idx + 1])   # Delta y_2, ..., Delta y_t
    if len(diffs) < 2:
        return np.nan
    sigma_t2 = np.sum(diffs ** 2) / (len(diffs) - 1)
    if sigma_t2 <= 0:
        return np.nan
    sigma_t = np.sqrt(sigma_t2)
    return (values[t_idx] - values[n_idx]) / (sigma_t * np.sqrt(t_idx - n_idx))


def get_csw_critical_value(n_idx, t_idx, b_alpha=4.6):
    """Book's own time-dependent critical value (Sec 17.3.2):
    c_alpha[n,t] = sqrt(b_alpha + log(t-n)). b_0.05 = 4.6 is the book's
    own Monte-Carlo-derived constant for a one-sided 5% test."""
    return np.sqrt(b_alpha + np.log(t_idx - n_idx))


def get_csw_sup(logP, t_idx):
    """S_t = sup_{n in [1,t]} {S_{n,t}} (Sec 17.3.2) -- the book's own fix
    for the arbitrary-reference-level problem: instead of picking one
    fixed n, try every valid backward-shifting reference point and take
    the largest departure found.

    Returns
    -------
    dict with keys 'S' (the supremum statistic), 'n_star' (which
    reference index achieved it), and 'critical_value_95' (the book's
    c_alpha[n_star, t] at the winning n, for direct comparison against S).
    """
    best_S, best_n = -np.inf, None
    for n_idx in range(0, t_idx):
        S = get_csw_stat(logP, n_idx, t_idx)
        if np.isfinite(S) and S > best_S:
            best_S, best_n = S, n_idx
    if best_n is None:
        return {'S': np.nan, 'n_star': None, 'critical_value_95': np.nan}
    return {
        'S': best_S,
        'n_star': best_n,
        'critical_value_95': get_csw_critical_value(best_n, t_idx),
    }


def get_csw_cusum(logP, min_sample=3):
    """Outer loop (book gives no code for this either, but describes the
    same backward-shifting-window idea repeated across an advancing end
    point, matching SADF's own outer-loop shape in Sec 17.4.2): compute
    S_t = sup_n S_{n,t} for every valid end point t, producing a full CSW
    CUSUM time series rather than a single value.

    Parameters
    ----------
    logP : pd.Series or single-column pd.DataFrame of log-prices.
    min_sample : int -- smallest t_idx to start computing from (needs at
        least 2 prior returns for sigma_hat_t to be defined, i.e. t_idx>=2;
        default 3 leaves a little more room for a meaningful sup search).

    Returns
    -------
    pd.DataFrame with columns ['S', 'n_star', 'critical_value_95'],
    indexed by logP's own index.
    """
    values, idx = _prep_log_series(logP)
    T = len(values)
    if T <= min_sample:
        raise ValueError(
            f"logP has {T} rows, needs more than min_sample={min_sample}."
        )
    rows = []
    for t_idx in range(min_sample, T):
        rows.append(get_csw_sup(logP, t_idx))
    out = pd.DataFrame(rows)
    out.index = idx[min_sample:T]
    return out
