"""
Chapter 17, Section 17.4.1: Chow-Type Dickey-Fuller Test
===========================================================

WHY THIS TEST EXISTS
---------------------
This is the simplest of the book's explosiveness tests: it assumes there
is exactly ONE regime switch, from a random walk to an explosive process,
happening at some (unknown) break date. Consider

    y_t = { y_{t-1} + eps_t                 for t = 1, ..., tau*T   (random walk)
          { rho*y_{t-1} + eps_t, rho > 1     for t = tau*T+1, ..., T (explosive)

To test for this, we fit

    Delta y_t = delta * y_{t-1} * D_t[tau*] + eps_t

where D_t[tau*] is a dummy: 0 before the assumed break date tau*T, 1 from
it onward. If y_{t-1} only starts predicting positive drift AFTER the
break, delta will be positive and significant -- that's the signature of
a random-walk-to-explosive switch at that date.

FORMULA, NO PRINTED CODE
--------------------------
Unlike SADF (Sec 17.4.2, Snippets 17.1-17.4), the book gives NO code for
this test -- only the formulas above and DFC_tau* = delta_hat / se(delta_hat).
Everything below is written directly from those formulas, not ported from
a snippet, so there's no "book bug" to find here -- but it's tested
against hand-computed closed-form OLS-through-origin algebra instead, to
the same standard as a fidelity-checked port.

TWO VERSIONS
-------------
- get_dfc: the test at a SINGLE assumed break date tau* (DFC_tau*).
- get_sdfc: Andrews' [1993] generalization for an UNKNOWN break date --
  try every tau* in [tau0, 1-tau0] and take the supremum
  (SDFC = sup_{tau* in [tau0, 1-tau0]} {DFC_tau*}). tau0 trims some of the
  sample from both ends so each regime has enough observations to fit
  (the book flags this explicitly -- there must be enough zeros and
  enough ones in D_t[tau*]).

Reuses getBetas from sadf.py (same OLS machinery, no reason to duplicate
it) rather than reimplementing linear regression from scratch.
"""
import numpy as np
import pandas as pd

from sadf import getBetas


def _prep_series(series):
    """Accept either a pd.Series or a single-column pd.DataFrame of
    log-prices; return a 1-D ndarray of values and the index. Mirrors
    sadf.getYX's own Series-auto-conversion fix -- same interface
    forgiveness, same reasoning."""
    if isinstance(series, pd.DataFrame):
        idx = series.index
        values = series.iloc[:, 0].values
    else:
        idx = series.index
        values = series.values
    return values, idx


def get_dfc(logP, tau_star):
    """Chow-type DF statistic DFC_tau* for a SINGLE assumed break fraction.

    Parameters
    ----------
    logP : pd.Series or single-column pd.DataFrame of log-prices.
    tau_star : float in (0, 1) -- assumed break date, as a FRACTION of the
        sample length (the book's own tau* notation; internally converted
        to an integer break index k = round(tau_star * T)).

    Returns
    -------
    float -- DFC_tau* = delta_hat / se(delta_hat). NaN if tau_star places
    the break too close to either end for a regression to be fit (D_t
    would be all-zero or all-one, giving a singular x'x).
    """
    values, _ = _prep_series(logP)
    T = len(values)
    k = int(round(tau_star * T))
    if k < 1 or k >= T - 1:
        return np.nan

    dy = np.diff(values)          # Delta y_t, t = 1, ..., T-1
    y_lag = values[:-1]           # y_{t-1}, same length as dy

    # D_t[tau*]: 0 before the break (t < tau*T), 1 from it onward. dy[i]
    # corresponds to original time index i+1, so the break at index k in
    # the ORIGINAL series is index k-1 in this length-(T-1) array.
    D = np.zeros(T - 1)
    D[k - 1:] = 1.0

    x = (y_lag * D).reshape(-1, 1)
    y = dy.reshape(-1, 1)
    if np.sum(x ** 2) == 0:
        return np.nan

    bMean, bVar = getBetas(y, x)
    delta_hat = bMean[0, 0]
    se_delta = bVar[0, 0] ** .5
    return delta_hat / se_delta


def get_sdfc(logP, tau0=0.15, step=None):
    """Andrews' [1993] sup-DFC test: try every tau* in [tau0, 1-tau0] and
    return the supremum DFC_tau*, along with which tau* achieved it.

    Parameters
    ----------
    logP : pd.Series or single-column pd.DataFrame of log-prices.
    tau0 : float in (0, 0.5) -- trims tau0 of the sample from EACH end
        (book's own guidance: leave out enough of both ends that either
        regime has enough observations to fit). Andrews' original paper
        and common practice use values around 0.15.
    step : float or None -- grid spacing for tau*, as a fraction of the
        sample. Defaults to one candidate per feasible integer break
        index (matching the book's "T(1-2*tau0) values" -- i.e. every
        discrete time point in range, not an arbitrary continuous grid).

    Returns
    -------
    dict with keys 'sdfc' (the supremum DFC_tau* value) and
    'tau_star' (the break fraction that achieved it).
    """
    values, idx = _prep_series(logP)
    T = len(values)
    if step is None:
        # one grid point per feasible integer break index
        k_lo = max(1, int(np.ceil(tau0 * T)))
        k_hi = min(T - 2, int(np.floor((1 - tau0) * T)))
        if k_lo > k_hi:
            raise ValueError(
                f"tau0={tau0} leaves no feasible break points for a series "
                f"of length {T}. Use a smaller tau0 or a longer series."
            )
        candidate_ks = range(k_lo, k_hi + 1)
        candidate_taus = [k / T for k in candidate_ks]
    else:
        candidate_taus = np.arange(tau0, 1 - tau0 + 1e-12, step)

    best_dfc, best_tau = -np.inf, np.nan
    for tau_star in candidate_taus:
        dfc = get_dfc(logP, tau_star)
        if np.isfinite(dfc) and dfc > best_dfc:
            best_dfc, best_tau = dfc, tau_star

    return {'sdfc': best_dfc, 'tau_star': best_tau}
