"""
Chapter 19 -- Microstructural Features (AFML, Lopez de Prado).

Nine features spanning the three "generations" of microstructural theory
the book walks through, scoped to what's actually derivable from a raw
trade tape + dollar bars (no order book, no cancellations, no options data):

  First generation  (price sequences only):
    - tick_rule / tick_rule_accuracy        Sec 19.3.1
    - roll_measure                          Sec 19.3.2
    - parkinson_volatility                  Sec 19.3.3
    - corwin_schultz / becker_parkinson_sigma   Sec 19.3.4, Snippets 19.1/19.2

  Second generation (price + volume, strategic trade models):
    - kyle_lambda                           Sec 19.4.1
    - amihud_lambda                         Sec 19.4.2
    (Hasbrouck's Lambda is intentionally skipped: the book requires a Gibbs
    sampler for the Bayesian estimation and gives no snippet -- an OLS
    stand-in would be a real departure from the book's actual method, not
    a like-for-like port. Revisit if that becomes worth the machinery.)

  Third generation (sequential trade models):
    - vpin                                  Sec 19.5.2
    (PIN itself, Sec 19.5.1, is also skipped: it's a maximum-likelihood fit
    of a 3-component Poisson mixture with no snippet given, and VPIN is
    the book's own high-frequency, easier-to-estimate version of the same
    underlying idea -- see the paragraph introducing Sec 19.5.2.)

  Additional, theory-adjacent features (Sec 19.6):
    - round_number_frequency                Sec 19.6.1
    - serial_correlation_signed_flow        Sec 19.6.5
    (Cancellation rates, TWAP-detection, and options-market features,
    Secs 19.6.2-19.6.4, need data this pipeline doesn't have: order-book
    messages and options quotes.)

Data this module assumes you have, matching the real BTC/TUSD pipeline:
  - a raw trade tape with Price, Volume (base-asset qty), and a true
    aggressor-side Label (+1 buy-initiated, -1 sell-initiated) -- Binance's
    IsBuyerMaker flag gives us this directly, so the tick rule below is
    used as an ESTIMATE to compare against ground truth, not as our only
    way of knowing trade direction.
  - dollar bars (Open/High/Low/Close/Vwap) built with the same $10,000
    threshold used everywhere else in this pipeline (Ch02).
"""

import numpy as np
import pandas as pd


# =============================================================================
# 19.3.1 -- The Tick Rule
# =============================================================================
# --- Why ---
# Every trade has a buyer and a seller, but only one side "initiated" it
# (crossed the spread with a market order). Knowing which side was the
# aggressor is the raw ingredient for almost every other feature in this
# chapter (Kyle's Lambda, VPIN, serial correlation of order flow all need
# a signed trade series). The tick rule is a cheap way to *infer* that
# sign from price alone, for datasets that don't give you the true side.
#
# --- Math ---
#           |  1          if delta_p_t > 0
#   b_t  =  | -1          if delta_p_t < 0
#           |  b_(t-1)    if delta_p_t == 0
#
# with b_0 set arbitrarily to 1 (book's convention).
def tick_rule(prices, b0=1):
    """Infer trade-aggressor sign from a price series via the tick rule.

    Parameters
    ----------
    prices : array-like of float, trade-by-trade prices, in time order.
    b0 : int, the arbitrary sign assigned to the first trade (book: +1).

    Returns
    -------
    np.ndarray of int, same length as `prices`, each entry in {-1, 1}.
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n == 0:
        return np.array([], dtype=int)

    b = np.empty(n, dtype=int)
    b[0] = b0
    diffs = np.diff(prices)
    for t in range(1, n):
        d = diffs[t - 1]
        if d > 0:
            b[t] = 1
        elif d < 0:
            b[t] = -1
        else:
            b[t] = b[t - 1]
    return b


def tick_rule_accuracy(inferred_side, true_side):
    """Fraction of trades where the tick rule's inferred sign matches the
    true aggressor side. Sec 19.3.1 references studies (Aitken & Frino
    [1996]) on how accurate the tick rule tends to be in practice -- this
    lets us check that claim directly on real data instead of citing it.
    """
    inferred_side = np.asarray(inferred_side)
    true_side = np.asarray(true_side)
    if len(inferred_side) != len(true_side):
        raise ValueError("inferred_side and true_side must be the same length")
    if len(inferred_side) == 0:
        raise ValueError("cannot score accuracy on an empty series")
    return float(np.mean(inferred_side == true_side))


# =============================================================================
# 19.3.2 -- The Roll Model
# =============================================================================
# --- Why ---
# Roll's insight: even with nothing but a price series (no bid/ask, no
# volume), the bid-ask spread leaves a fingerprint in how prices bounce
# back and forth. A trade at the ask followed by one at the bid looks
# like a price drop that immediately reverses -- that bounce shows up as
# NEGATIVE serial covariance in price changes. The more negative that
# covariance, the wider the spread must be to produce it.
#
# --- Math (book's Sec 19.3.2) ---
#   Var[delta_p_t]              = 2c^2 + sigma_u^2
#   Cov[delta_p_t, delta_p_t-1] = -c^2
#   =>  c        = sqrt( max(0, -Cov[delta_p_t, delta_p_t-1]) )
#   =>  sigma_u^2 = Var[delta_p_t] + 2*Cov[delta_p_t, delta_p_t-1]
def roll_measure(prices):
    """Estimate Roll's effective half-spread `c` and the unobserved
    fundamental-price noise `sigma_u` from a price series.

    Returns
    -------
    dict with keys 'c' (effective half bid-ask spread), 'sigma_u'
    (fundamental price volatility net of microstructure noise),
    'cov_dp' and 'var_dp' (the underlying serial covariance/variance,
    kept for inspection/testing).
    """
    prices = np.asarray(prices, dtype=float)
    dp = np.diff(prices)
    if len(dp) < 2:
        raise ValueError("need at least 3 prices (2 price changes) to estimate Roll's model")

    var_dp = float(np.var(dp, ddof=1))
    cov_dp = float(np.cov(dp[:-1], dp[1:], ddof=1)[0, 1])

    c = float(np.sqrt(max(0.0, -cov_dp)))
    sigma_u2 = var_dp + 2 * cov_dp
    sigma_u = float(np.sqrt(max(0.0, sigma_u2)))

    return {"c": c, "sigma_u": sigma_u, "cov_dp": cov_dp, "var_dp": var_dp}


# =============================================================================
# 19.3.3 -- High-Low (Parkinson) Volatility Estimator
# =============================================================================
# --- Why ---
# A bar's closing price alone throws away information: two bars can close
# at the same price but one might have swung wildly intrabar while the
# other barely moved. High and low prices capture that swing, and
# Parkinson [1980] showed this gives a more accurate volatility estimate
# than close-to-close returns for the same amount of data.
#
# --- Math ---
#   E[ (1/T) * sum_t (log(H_t/L_t))^2 ]  = k1 * sigma_HL^2,   k1 = 4*log(2)
#   =>  sigma_HL = sqrt( mean( (log(H_t/L_t))^2 ) / k1 )
def parkinson_volatility(high, low):
    """Parkinson's high-low volatility estimate, sigma_HL, over the given
    bars (a single scalar summarizing the whole window passed in -- pass
    a rolling slice of bars to get a rolling estimate).
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if len(high) != len(low):
        raise ValueError("high and low must be the same length")
    if len(high) == 0:
        raise ValueError("need at least one bar")

    k1 = 4.0 * np.log(2.0)
    log_hl2 = np.log(high / low) ** 2
    sigma_hl = float(np.sqrt(np.mean(log_hl2) / k1))
    return sigma_hl


# =============================================================================
# 19.3.4 -- Corwin & Schultz Spread Estimator (Snippets 19.1 / 19.2)
# =============================================================================
# --- Why ---
# Corwin-Schultz sharpens the high-low idea into an actual bid-ask spread
# estimate (not just volatility), using the fact that a 2-bar high/low
# range grows with BOTH volatility and elapsed time, while a 1-bar range
# only grows with volatility -- comparing the two isolates the spread.
#
# --- Note on porting the book's snippet ---
# Snippet 19.1 is printed against a pre-2016 pandas API
# (`pd.stats.moments.rolling_sum` / `rolling_mean` / `rolling_max` /
# `rolling_min`), which was removed from pandas entirely years ago.
# Ported 1:1 in spirit to modern `.rolling().sum()/.mean()/.max()/.min()`
# -- same math, same window semantics -- this is the same kind of
# legitimate book-snippet modernization already applied in Ch09
# (`iid=` removed, `base_estimator=`->`estimator=`).
def get_beta(high, low, sl):
    """Snippet 19.1's getBeta: rolling 2-bar sum of squared log(H/L),
    averaged over a window of `sl` bars."""
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    hl = np.log(high.values / low.values) ** 2
    hl = pd.Series(hl, index=high.index)
    beta = hl.rolling(window=2).sum()
    beta = beta.rolling(window=sl).mean()
    return beta.dropna()


def get_gamma(high, low):
    """Snippet 19.1's getGamma: squared log-range of the max high / min low
    over each adjacent pair of bars."""
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    h2 = high.rolling(window=2).max()
    l2 = low.rolling(window=2).min()
    gamma = np.log(h2.values / l2.values) ** 2
    gamma = pd.Series(gamma, index=h2.index)
    return gamma.dropna()


def get_alpha(beta, gamma):
    """Snippet 19.1's getAlpha. Negative alphas are clipped to 0, per the
    book's own instruction (p.727 of Corwin & Schultz [2012])."""
    den = 3 - 2 * 2 ** 0.5
    alpha = (2 ** 0.5 - 1) * (beta ** 0.5) / den
    alpha = alpha - (gamma / den) ** 0.5
    alpha = alpha.copy()
    alpha[alpha < 0] = 0
    return alpha.dropna()


def corwin_schultz(high, low, sl=1):
    """Snippet 19.1's corwinSchultz: the estimated bid-ask spread S_t as a
    fraction of price, from high/low bar prices alone."""
    beta = get_beta(high, low, sl)
    gamma = get_gamma(high, low)
    alpha = get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread.name = "Spread"
    return spread


def becker_parkinson_sigma(beta, gamma):
    """Snippet 19.2's getSigma: the Becker-Parkinson volatility that falls
    out of the Corwin-Schultz derivation as a byproduct. Takes the same
    `beta`/`gamma` series produced by get_beta / get_gamma above."""
    k2 = (8.0 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** 0.5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma = sigma + (gamma / (k2 ** 2 * den)) ** 0.5
    sigma = sigma.copy()
    sigma[sigma < 0] = 0
    return sigma


# =============================================================================
# 19.4.1 -- Kyle's Lambda
# =============================================================================
# --- Why ---
# Kyle's model says: a market maker who can't tell informed traders from
# noise traders has to move the price in proportion to order-flow
# imbalance, just in case it's informed. That proportionality constant,
# lambda, IS the price-impact-per-unit-of-signed-volume -- i.e. a direct
# measure of illiquidity. High lambda = a little bit of signed volume
# moves price a lot = illiquid / adversely-selected market.
#
# --- Math ---
#   delta_p_t = lambda * (b_t * V_t) + epsilon_t
#
# Estimated per dollar-bar via a trade-level OLS regression of price
# changes on signed volume, using every trade inside that bar. (The book
# doesn't specify the estimation window explicitly; per-bar, using the
# trades that built that bar, is the natural choice here -- it's what
# lets us report one lambda per bar, matching Figure 19.1's histogram of
# many lambda estimates rather than a single value for the whole series.)
def kyle_lambda(delta_p, signed_volume):
    """Estimate a single Kyle's Lambda via OLS: delta_p ~ signed_volume
    (with intercept, for numerical stability -- lambda is the SLOPE).

    Parameters
    ----------
    delta_p : array-like, trade-to-trade price changes within a bar.
    signed_volume : array-like, b_t * V_t for the same trades (aligned).

    Returns
    -------
    float, the fitted lambda (slope). NaN if there isn't enough variation
    in signed_volume to identify a slope.
    """
    delta_p = np.asarray(delta_p, dtype=float)
    signed_volume = np.asarray(signed_volume, dtype=float)
    if len(delta_p) != len(signed_volume):
        raise ValueError("delta_p and signed_volume must be the same length")
    if len(delta_p) < 2:
        return np.nan
    if np.std(signed_volume) == 0:
        return np.nan

    X = np.column_stack([np.ones(len(signed_volume)), signed_volume])
    coeffs, *_ = np.linalg.lstsq(X, delta_p, rcond=None)
    return float(coeffs[1])


def kyle_lambda_by_bar(prices, signed_volumes, bar_ids, min_trades=3):
    """Apply kyle_lambda() bar-by-bar over a trade tape.

    Parameters
    ----------
    prices : array-like, trade-by-trade prices, in time order.
    signed_volumes : array-like, b_t * V_t for the same trades.
    bar_ids : array-like, the dollar-bar index each trade belongs to.
    min_trades : int, bars with fewer trades than this get NaN (not
        enough points to fit a meaningful regression).

    Returns
    -------
    pd.Series indexed by bar_id, one Kyle's Lambda estimate per bar.
    """
    df = pd.DataFrame({
        "price": np.asarray(prices, dtype=float),
        "signed_volume": np.asarray(signed_volumes, dtype=float),
        "bar_id": np.asarray(bar_ids),
    })
    out = {}
    for bar_id, grp in df.groupby("bar_id"):
        if len(grp) < min_trades:
            out[bar_id] = np.nan
            continue
        dp = np.diff(grp["price"].values)
        sv = grp["signed_volume"].values[1:]  # align: delta_p_t pairs with signed_volume_t
        out[bar_id] = kyle_lambda(dp, sv)
    return pd.Series(out).sort_index()


# =============================================================================
# 19.4.2 -- Amihud's Lambda
# =============================================================================
# --- Why ---
# Amihud's version of the same idea, but at the bar level instead of the
# trade level, and using dollar volume (not signed volume) against the
# ABSOLUTE size of the price move: "how many dollars did it take to move
# this bar's price by 1%?" is a blunter but much cheaper-to-compute proxy
# for the same illiquidity concept as Kyle's Lambda.
#
# --- Math ---
#   | delta_log(close_tau) | = lambda * sum_{t in bar tau} (p_t * V_t) + epsilon_tau
#
# Note: no intercept in the book's equation -- the regression is fit
# through the origin (zero dollar volume => zero expected price impact).
def amihud_lambda(bar_close, bar_dollar_volume):
    """Estimate a single Amihud's Lambda via OLS through the origin:
    |delta_log(close)| ~ dollar_volume, matching the book's equation
    (no intercept term).

    Parameters
    ----------
    bar_close : array-like, bar closing prices, in time order.
    bar_dollar_volume : array-like, total dollar volume traded in each
        bar (aligned so that bar_dollar_volume[i] is the volume that
        produced the move from bar_close[i-1] to bar_close[i]).

    Returns
    -------
    float, the fitted lambda.
    """
    bar_close = np.asarray(bar_close, dtype=float)
    bar_dollar_volume = np.asarray(bar_dollar_volume, dtype=float)
    if len(bar_close) != len(bar_dollar_volume):
        raise ValueError("bar_close and bar_dollar_volume must align 1:1")
    if len(bar_close) < 2:
        return np.nan

    abs_dlog = np.abs(np.diff(np.log(bar_close)))
    dollar_vol = bar_dollar_volume[1:]  # the volume that produced each move
    denom = np.dot(dollar_vol, dollar_vol)
    if denom == 0:
        return np.nan
    lam = float(np.dot(dollar_vol, abs_dlog) / denom)  # OLS-through-origin closed form
    return lam


# =============================================================================
# 19.5.2 -- VPIN (Volume-Synchronized Probability of Informed Trading)
# =============================================================================
# --- Why ---
# VPIN is PIN's practical cousin: instead of fitting an unobservable
# maximum-likelihood model (Sec 19.5.1), just measure how LOPSIDED buy vs.
# sell volume is across a rolling window of equal-sized (here, dollar)
# bars. Persistent one-sided volume is the observable symptom PIN was
# trying to explain in the first place.
#
# --- Math ---
#   VPIN_tau = sum_{i=tau-n+1}^{tau} |VB_i - VS_i|  /  sum_{i=tau-n+1}^{tau} (VB_i + VS_i)
def vpin(buy_volume, sell_volume, window):
    """Rolling VPIN over `window` bars.

    Parameters
    ----------
    buy_volume : array-like, buy-initiated volume per bar.
    sell_volume : array-like, sell-initiated volume per bar.
    window : int, number of bars n in the rolling window.

    Returns
    -------
    pd.Series, same length as inputs, NaN for the first (window-1) bars.
    """
    buy_volume = pd.Series(buy_volume, dtype=float).reset_index(drop=True)
    sell_volume = pd.Series(sell_volume, dtype=float).reset_index(drop=True)
    if len(buy_volume) != len(sell_volume):
        raise ValueError("buy_volume and sell_volume must be the same length")
    if window < 1:
        raise ValueError("window must be >= 1")

    imbalance = (buy_volume - sell_volume).abs()
    total = buy_volume + sell_volume
    numer = imbalance.rolling(window=window).sum()
    denom = total.rolling(window=window).sum()
    return numer / denom


# =============================================================================
# 19.6.1 -- Distribution of Order Sizes (round-number frequency)
# =============================================================================
# --- Why ---
# The book's finding (Easley et al. [2016]) is about DISCRETE contract
# counts (size 10 vs. size 9, etc.) -- human "GUI traders" click round
# button values, algorithmic traders don't. BTC quantities are continuous
# (e.g. 0.00148000 BTC), so "size 10 vs size 9" doesn't port literally.
# The adaptation used here: define a set of economically "round"
# quantities a human would plausibly type (0.001, 0.005, 0.01, 0.05, 0.1,
# 0.5, 1, 5, 10, ... BTC) and measure what fraction of real trade volume
# lands within a small tolerance of one of those levels, vs. what you'd
# expect if quantities were uniformly "un-round." This is a genuine
# judgment call adapting a discrete-market finding to a continuous asset,
# not a literal book formula -- flagged here and in the demo script.
DEFAULT_ROUND_LEVELS = (
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5,
    1.0, 5.0, 10.0, 50.0, 100.0,
)


def round_number_frequency(volumes, round_levels=DEFAULT_ROUND_LEVELS, rel_tol=1e-6):
    """Fraction of trades whose volume matches a "round" level within a
    small relative tolerance, plus the breakdown by level.

    Returns
    -------
    dict: {'round_fraction': float, 'by_level': dict{level: count}}
    """
    volumes = np.asarray(volumes, dtype=float)
    if len(volumes) == 0:
        raise ValueError("need at least one trade")

    is_round = np.zeros(len(volumes), dtype=bool)
    by_level = {}
    for level in round_levels:
        matches = np.isclose(volumes, level, rtol=rel_tol, atol=level * rel_tol)
        by_level[level] = int(matches.sum())
        is_round |= matches

    return {
        "round_fraction": float(is_round.mean()),
        "by_level": by_level,
    }


# =============================================================================
# 19.6.5 -- Serial Correlation of Signed Order Flow
# =============================================================================
# --- Why ---
# If order flow is positively autocorrelated (a buy tends to be followed
# by more buys), that's consistent with either informed traders acting on
# persistent information, or large orders being split into smaller
# pieces over time (Toth et al. [2011] argue the latter dominates on
# short timescales). Either way, persistence in {b_t} is informative.
#
# --- Math ---
#   corr( b_t, b_(t-lag) )   -- standard Pearson autocorrelation.
def serial_correlation_signed_flow(signed_flow, lag=1):
    """Lag-`lag` autocorrelation of a signed order-flow series (e.g. the
    true Label column, or b_t*V_t)."""
    signed_flow = pd.Series(signed_flow, dtype=float).reset_index(drop=True)
    if len(signed_flow) <= lag:
        raise ValueError("series too short for the requested lag")
    return float(signed_flow.autocorr(lag=lag))
