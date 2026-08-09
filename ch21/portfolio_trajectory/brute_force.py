"""
AFML Chapter 21 -- Brute Force and Quantum Computers
Snippets 21.1, 21.2, 21.3: pigeonhole partitions, signed weight vectors,
and trajectory evaluation (transaction costs, Sharpe Ratio, brute-force search).

WHY this chapter exists (plain English first):
A "trading trajectory" is a schedule of portfolio weights over multiple future
time horizons -- not just "what should I hold today" but "what should I hold
today, and next week, and the week after." Because transaction costs depend
on how much you CHANGE your weights between horizons, the optimal trajectory
can't be found by optimizing each horizon separately (that's the "static"
solution in static_solution.py) -- you have to jointly consider the whole
path. That joint problem isn't convex, so there's no closed-form solution.
This module implements the book's brute-force alternative: discretize the
space of possible weight vectors into a finite set, then exhaustively
evaluate every possible trajectory (every H-length sequence of weight
vectors) and keep the one with the best Sharpe Ratio. This is exactly the
kind of embarrassingly-parallel combinatorial search that (per the book's
motivation) quantum computers are theoretically suited to -- but the
digital-computer version in this chapter still works, it's just slow.

Translation notes (Python 2 -> Python 3), per project book-fidelity rule:
- Snippet 21.1 used `xrange`, which doesn't exist in Python 3. Replaced
  with the builtin `range` (Python 3's `range` is already lazy, like
  Python 2's `xrange` -- no generator wrapper needed).
- No other Python-2-isms found in Snippets 21.1-21.3; the combinatorial
  logic (pigeonhole partitioning, signed weight generation, trajectory
  evaluation) was verified against the book's stated formulas and is
  faithful to the printed snippets. No bugs analogous to Ch09's bagging-
  tuple-order bug were found here.
"""

from itertools import combinations_with_replacement, product

import numpy as np


def pigeon_hole(k, n):
    """
    Snippet 21.1: PARTITIONS OF k OBJECTS INTO n SLOTS.

    Generates every way to place k indistinguishable "units of capital"
    into n distinguishable "asset slots" -- i.e. every non-negative integer
    solution to x_1 + ... + x_n = k. Order of the SLOTS matters (slot 1
    getting 2 units and slot 2 getting 1 is different from the reverse),
    but the k units themselves are indistinguishable (this is the
    "stars and bars" combinatorial trick).

    Parameters
    ----------
    k : int
        Number of indistinguishable units of capital to allocate.
    n : int
        Number of asset slots to allocate them into.

    Yields
    ------
    list[int]
        Length-n list of non-negative integers summing to k.
    """
    for combo in combinations_with_replacement(range(n), k):
        partition = [0] * n
        for slot in combo:
            partition[slot] += 1
        yield partition


def get_all_weights(k, n):
    """
    Snippet 21.2: SET OMEGA OF ALL VECTORS ASSOCIATED WITH ALL PARTITIONS.

    Turns each capital partition from pigeon_hole() into a vector of
    ABSOLUTE weights (dividing by k so the weights sum to 1, satisfying
    the full-investment constraint), then attaches every possible
    combination of +/- signs (long or short each asset) to get the full
    set of feasible STATIC weight vectors at a single horizon.

    Parameters
    ----------
    k : int
        Number of capital units (must be a positive integer; k > 0).
    n : int
        Number of assets (n >= 1).

    Returns
    -------
    np.ndarray, shape (n, num_partitions * 2**n)
        Each column is one feasible signed weight vector; sum(abs(column)) == 1.
    """
    parts = pigeon_hole(k, n)
    weights = None
    for partition in parts:
        abs_weights = np.array(partition) / float(k)
        for signs in product([-1, 1], repeat=n):
            signed = (abs_weights * signs).reshape(-1, 1)
            if weights is None:
                weights = signed.copy()
            else:
                weights = np.append(weights, signed, axis=1)
    return weights


def eval_t_costs(w, params):
    """
    Snippet 21.3 (part 1): Compute per-horizon transaction costs for a trajectory.

    tau_1[w] = sum_n c_{n,1} * sqrt(|w_{n,1} - w*_n|)          (h=1, vs. initial allocation)
    tau_h[w] = sum_n c_{n,h} * sqrt(|w_{n,h} - w_{n,h-1}|)     (h>1, vs. previous horizon)

    The initial allocation w*_n is implicitly assumed to be all-zero (the
    book's own Snippet 21.3 doesn't take it as a parameter) -- i.e. the
    trajectory starts from an unallocated (all-cash) portfolio.

    Parameters
    ----------
    w : np.ndarray, shape (N, H)
        Trajectory: column h is the weight vector at horizon h.
    params : list[dict]
        Length-H list; params[h]['c'] is an (N,) array of per-asset cost factors.

    Returns
    -------
    np.ndarray, shape (H,)
        Transaction cost at each horizon.
    """
    tcost = np.zeros(w.shape[1])
    w_prev = np.zeros(shape=w.shape[0])
    for i in range(tcost.shape[0]):
        c = params[i]['c']
        tcost[i] = (c * abs(w[:, i] - w_prev) ** 0.5).sum()
        w_prev = w[:, i].copy()
    return tcost


def eval_sr(params, w, tcost):
    """
    Snippet 21.3 (part 2): Evaluate the Sharpe Ratio of a full trajectory.

    SR[r] = ( sum_h  mu_h' w_h - tau_h[w] )  /  sqrt( sum_h  w_h' V_h w_h )

    Parameters
    ----------
    params : list[dict]
        Length-H list; params[h]['mean'] is (N,1), params[h]['cov'] is (N,N).
    w : np.ndarray, shape (N, H)
        Trajectory.
    tcost : np.ndarray, shape (H,)
        Per-horizon transaction costs, from eval_t_costs.

    Returns
    -------
    float
        Sharpe Ratio of the trajectory.
    """
    mean, cov = 0.0, 0.0
    for h in range(w.shape[1]):
        params_h = params[h]
        mean += np.dot(w[:, h].T, params_h['mean'])[0] - tcost[h]
        cov += np.dot(w[:, h].T, np.dot(params_h['cov'], w[:, h]))
    sr = mean / cov ** 0.5
    return sr


def dyn_opt_port(params, k=None):
    """
    Snippet 21.3 (part 3): Brute-force search for the globally optimal trajectory.

    Enumerates EVERY feasible trajectory (every H-length sequence drawn from
    the feasible static weight set Omega) and keeps whichever one maximizes
    the Sharpe Ratio. This is the exhaustive, non-convex-optimization-free
    search described in Section 21.5 -- guaranteed globally optimal given the
    discretization, but combinatorially expensive: the number of trajectories
    evaluated is (num_partitions(k, n) * 2**n) ** H, where H = len(params).

    Parameters
    ----------
    params : list[dict]
        Length-H list of {'mean': (N,1), 'cov': (N,N), 'c': (N,)} dicts.
    k : int, optional
        Number of capital partition units. Defaults to N (matches the
        book's own numerical example in Snippet 21.7, which doesn't pass k
        explicitly). Note the book's Section 21.5.1 motivates the pigeonhole
        combinatorics assuming k > n, but the underlying algorithm
        (combinations_with_replacement) is valid for any k >= 0 -- the k > n
        assumption there is illustrative, not a code requirement.

    Returns
    -------
    np.ndarray, shape (N, H)
        The best trajectory found.
    """
    if k is None:
        k = params[0]['mean'].shape[0]
    n = params[0]['mean'].shape[0]
    w_all = get_all_weights(k, n)
    sr, w = None, None
    for combo in product(w_all.T, repeat=len(params)):
        w_ = np.array(combo).T
        tcost_ = eval_t_costs(w_, params)
        sr_ = eval_sr(params, w_, tcost_)
        if sr is None or sr < sr_:
            sr, w = sr_, w_.copy()
    return w


def count_trajectories(k, n, h):
    """
    Teaching utility (not in the book's snippets): the size of the brute-force
    search space, so students can see why this needs a quantum computer for
    realistic N/K/H before they accidentally run something that never finishes.

    |Omega| = C(k + n - 1, n - 1) * 2**n   (partitions x sign combinations)
    |Phi|   = |Omega| ** h                 (trajectories = H-fold Cartesian product)

    Parameters
    ----------
    k, n, h : int
        Capital units, number of assets, number of horizons.

    Returns
    -------
    dict
        {'num_partitions': int, 'omega_size': int, 'num_trajectories': int}
    """
    from math import comb
    num_partitions = comb(k + n - 1, n - 1)
    omega_size = num_partitions * (2 ** n)
    num_trajectories = omega_size ** h
    return {
        'num_partitions': num_partitions,
        'omega_size': omega_size,
        'num_trajectories': num_trajectories,
    }
