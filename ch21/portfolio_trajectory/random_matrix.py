"""
AFML Chapter 21 -- Snippets 21.4-21.5: random matrices of known rank, and
generating a synthetic multi-horizon parameter set (mu, V, c).

These are used for TDD (known-rank verification is a clean hand-checkable
property) and are also handy any time you want a quick synthetic sanity
check of the brute_force/static_solution modules before trusting them on
real data.

Translation notes (Python 2 -> Python 3):
- No `xrange`/print-statement issues in the printed snippets themselves.
- The book's Snippet 21.4/21.5 call the unseeded global `np.random.*`
  functions directly. Per this project's `random_state` convention
  (explicit `numpy.random.Generator`, threaded as a parameter, sklearn-style),
  both functions below take an optional `random_state` so tests are
  reproducible. When `random_state` is None, a fresh Generator is created
  each call (matches the book's own unseeded behavior for exploratory use).
"""

import numpy as np


def rnd_mat_with_rank(n_samples, n_cols, rank, sigma=0.0, hom_noise=True, random_state=None):
    """
    Snippet 21.4: Produce a random Gaussian matrix X of a given (known) rank.

    Useful whenever you want a synthetic covariance structure with a KNOWN,
    verifiable rank -- e.g. for Monte Carlo experiments or, here, for hand-
    checkable TDD tests (rank is something you can assert exactly).

    Parameters
    ----------
    n_samples : int
        Number of rows (observations).
    n_cols : int
        Number of columns (variables).
    rank : int
        Desired rank of the (noise-free) matrix; must be <= n_cols.
    sigma : float, default 0.0
        Standard deviation of additive noise. sigma=0.0 gives an exactly
        rank-`rank` matrix (up to floating point); sigma>0 perturbs it
        (and, with hom_noise=False, can push the numerical rank up to n_cols).
    hom_noise : bool, default True
        If True, homoscedastic noise (same sigma for every column). If
        False, heteroscedastic noise (a different sigma per column, drawn
        from [0.5*sigma, 1.5*sigma)).
    random_state : numpy.random.Generator, optional
        Explicit RNG for reproducibility (project convention). If None,
        a fresh, unseeded Generator is created (matches book behavior).

    Returns
    -------
    np.ndarray, shape (n_samples, n_cols)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    u, _, _ = np.linalg.svd(random_state.standard_normal((n_cols, n_cols)))
    x = np.dot(random_state.standard_normal((n_samples, rank)), u[:, :rank].T)
    if hom_noise:
        x += sigma * random_state.standard_normal((n_samples, n_cols))
    else:
        sigmas = sigma * (random_state.random(n_cols) + 0.5)
        x += random_state.standard_normal((n_samples, n_cols)) * sigmas
    return x


def gen_mean(size, random_state=None):
    """
    Snippet 21.5 (genMean helper): a random (size, 1) column vector of means,
    drawn from a standard Normal distribution.

    Parameters
    ----------
    size : int
        Number of assets.
    random_state : numpy.random.Generator, optional
        Explicit RNG for reproducibility. If None, a fresh Generator is used.

    Returns
    -------
    np.ndarray, shape (size, 1)
    """
    if random_state is None:
        random_state = np.random.default_rng()
    return random_state.standard_normal(size=(size, 1))


def gen_synthetic_params(size, horizon, random_state=None):
    """
    Snippet 21.5 (the '#1) Parameters' block): build a synthetic H-horizon
    params list, matching the shape brute_force.dyn_opt_port() expects.

    Each horizon gets its own independently-drawn (mean, cov, c) triple --
    this is what makes the dynamic (trajectory) problem genuinely different
    from a single-horizon static problem: the "best" portfolio can change
    from one horizon to the next, which is exactly what creates a
    transaction-cost trade-off worth solving jointly.

    Parameters
    ----------
    size : int
        Number of assets (N).
    horizon : int
        Number of horizons (H).
    random_state : numpy.random.Generator, optional
        Explicit RNG, threaded through every draw for full reproducibility.

    Returns
    -------
    list[dict]
        Length-H list of {'mean': (N,1), 'cov': (N,N), 'c': (N,)} dicts.
    """
    if random_state is None:
        random_state = np.random.default_rng()
    params = []
    for _ in range(horizon):
        x = rnd_mat_with_rank(1000, size, size, sigma=0.0, random_state=random_state)
        mean_ = gen_mean(size, random_state=random_state)
        cov_ = np.cov(x, rowvar=False)
        c_ = random_state.uniform(size=cov_.shape[0]) * np.diag(cov_) ** 0.5
        params.append({'mean': mean_, 'cov': cov_, 'c': c_})
    return params
