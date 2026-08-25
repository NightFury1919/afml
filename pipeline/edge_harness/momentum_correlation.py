"""
pipeline/edge_harness/momentum_correlation.py

Pure helper computing the momentum-edge analog of the OFI sweep's
raw_signal_corr metric.

*** LOAD-BEARING (2026-08-23): why lag-1 bar-return autocorrelation is
the right comparison metric ***
run_bar_aligned_edge_sweep.py's raw_signal_corr is
corrcoef(realized_bar_imbalance[:-1], next_bar_drift) -- a REALIZED,
observable, bar-level, one-step-ahead correlation between the injected
signal and forward price movement. It deliberately does NOT use the
hidden ground-truth edge_strength/target_directions, because what
matters for comparing detectability across signal TYPES is how strongly
the signal shows up in observable bar-level data, not how strongly it
was injected at the tick level (those two need not match 1:1 once
dollar-bar aggregation and averaging enter the picture).

The momentum analog: corrcoef(bar_return[:-1], bar_return[1:]) --
lag-1 autocorrelation of REALIZED bar close-to-close returns. This is
the natural momentum counterpart (current bar's return predicting the
next bar's return) computed the same way -- from realized, observable
bar-level prices, not from the hidden continuation_prob/target_directions
truth.
"""
import numpy as np


def bar_lag1_autocorr(close_prices):
    """corrcoef(bar_return[:-1], bar_return[1:]) on a close-price series.

    Parameters
    ----------
    close_prices : array-like of float, ordered by bar (chronological).
        Must have at least 3 elements (need at least 2 returns to
        correlate).

    Returns
    -------
    float. NaN if fewer than 2 returns are available, or if either lag
    slice has zero variance (corrcoef would divide by zero / return NaN
    itself -- this function returns NaN explicitly for a caller-visible
    signal, rather than propagating a silent RuntimeWarning).
    """
    prices = np.asarray(close_prices, dtype=float)
    if len(prices) < 3:
        return float('nan')
    returns = np.diff(prices) / prices[:-1]
    if len(returns) < 2:
        return float('nan')
    a, b = returns[:-1], returns[1:]
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
