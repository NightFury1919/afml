"""
AFML Chapter 21 -- Snippet 21.6: the static (myopic) optimal portfolio.

WHY this exists: it's the BASELINE the dynamic brute-force solution is
compared against. A static solution optimizes each horizon independently
(the classic unconstrained mean-variance / tangency-portfolio solution from
Chapter 16), completely ignoring the transaction costs of moving from one
horizon's weights to the next. The dynamic solution (brute_force.dyn_opt_port)
considers the whole trajectory jointly. Comparing their realized Sharpe
Ratios (both evaluated through the SAME transaction-cost-aware eval_sr) is
how the chapter demonstrates that accounting for path-dependent transaction
costs can beat a sequence of individually-optimal-looking decisions.

Translation notes (Python 2 -> Python 3):
- No `xrange` or dict-ordering issues in Snippet 21.6 itself.
- The Python 2 `print 'static SR:', sr_stat` statement (from the book's
  driver code, not this module) is converted to `print(...)` in the
  chapter driver script.
"""

import numpy as np


def stat_opt_portf(cov, a):
    """
    Snippet 21.6: Static (myopic) optimal portfolio -- the closed-form
    solution to the UNCONSTRAINED mean-variance optimization problem
    (maximize a'w subject to w'*cov*w = const), then rescaled to satisfy
    the chapter's full-investment constraint sum(|w_i|) = 1.

    w = cov^-1 @ a
    w /= (a' @ cov^-1 @ a)     # so that w' @ a == 1 before rescaling
    w /= sum(|w|)              # full-investment (leverage-normalized) rescale

    Parameters
    ----------
    cov : np.ndarray, shape (N, N)
        Covariance matrix for this horizon.
    a : np.ndarray, shape (N, 1)
        Forecasted mean returns for this horizon (the book reuses this
        function with a=mean, i.e. plugging expected returns in for the
        "expected excess return" vector of a tangency-portfolio solution).

    Returns
    -------
    np.ndarray, shape (N, 1)
        Static optimal weight vector, sum(|w|) == 1.
    """
    cov_inv = np.linalg.inv(cov)
    w = np.dot(cov_inv, a)
    w = w / np.dot(np.dot(a.T, cov_inv), a)  # np.dot(w.T, a) == 1 at this point
    w = w / np.abs(w).sum()                   # rescale for full investment
    return w


def stat_opt_trajectory(params):
    """
    Convenience wrapper (not a separate book snippet, just the loop from the
    book's driver code, Snippet 21.6's '#2) Static optimal portfolios' block,
    factored into a reusable function): apply stat_opt_portf independently
    at every horizon and stack the results into a trajectory matrix, so it
    has the same (N, H) shape brute_force.eval_t_costs/eval_sr expect.

    Parameters
    ----------
    params : list[dict]
        Length-H list of {'mean': (N,1), 'cov': (N,N), ...} dicts.

    Returns
    -------
    np.ndarray, shape (N, H)
    """
    w_stat = None
    for params_h in params:
        w_h = stat_opt_portf(cov=params_h['cov'], a=params_h['mean'])
        if w_stat is None:
            w_stat = w_h.copy()
        else:
            w_stat = np.append(w_stat, w_h, axis=1)
    return w_stat
