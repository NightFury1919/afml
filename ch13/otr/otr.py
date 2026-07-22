import numpy as np
from itertools import product

# Optimal Trading Rules (OTR) via synthetic backtesting -- AFML Chapter 13
#
# WHY this chapter exists (read this before the math):
# Chapters 11-12 both diagnosed the same disease from different angles: if you
# calibrate strategy parameters by testing many options against real history
# and picking the winner, you risk fitting noise, not signal. Chapter 13
# proposes a different cure -- instead of testing against real (or resampled
# real) history at all, fit a simple stochastic-process model to how your
# strategy's returns actually behave, then generate a huge number of
# SYNTHETIC paths from that model and optimize the trading rule against
# those. Because synthetic paths never touch any specific historical
# dataset, there's no particular set of datapoints to overfit to.
#
# The model used here is a discrete Ornstein-Uhlenbeck (O-U) process --
# a price that keeps getting pulled back toward some target level, with
# Gaussian noise layered on top. It's characterized by two parameters:
#   phi   -- how strongly/quickly the price reverts toward target
#            (phi close to 0: fast reversion; phi close to 1: slow,
#             approaching a random walk; phi outside (-1,1): NOT
#             stationary, no meaningful reversion at all)
#   sigma -- the size of the random shocks
#
# --- Pipeline ---
# 1. build_xy_from_opportunities  -> pool historical (deviation, next-
#                                     deviation) pairs across opportunities
#                                     (book eq. 13.6)
# 2. estimate_ou_params           -> OLS estimate of {phi, sigma} (13.7)
# 3. simulate_ou_path             -> one synthetic path under a candidate
#                                     profit-take/stop-loss rule (Snippet 13.2
#                                     inner loop)
# 4. batch                        -> sweep a whole mesh of (profit-take,
#                                     stop-loss) pairs, nIter synthetic paths
#                                     each, report each node's Sharpe ratio
#                                     (Snippet 13.2)
# 5. phi_to_half_life / half_life_to_phi -> convert between phi and the
#                                     process's half-life (Section 13.5.1)
#
# --- Book erratum fixed here ---
# Snippet 13.2 as printed uses Python-2 print-statement syntax
# (`print comb_[0],comb_[1],...` with no parentheses), which is a SyntaxError
# under Python 3. This is 2018-vintage code; harmless erratum, fixed below by
# using print(...) properly (and, better, returning results instead of
# printing them, so batch() is actually testable).


def build_xy_from_opportunities(paths, targets):
    """Pool (lagged deviation, next deviation) pairs across many trade paths.

    Implements AFML eq. 13.5/13.6: for every opportunity i, center its
    observed price path on that opportunity's target level, then stack every
    consecutive (deviation[t-1], deviation[t]) pair into X and Y respectively.
    Pooling across opportunities this way is what lets Step 1 estimate ONE
    {phi, sigma} pair characterizing the strategy's typical reversion
    behavior, from many separate historical trades.

    --- Inputs ---
    paths   : list of 1-D array-likes -- paths[i] is opportunity i's observed
              price (or return) series, P_{i,0}, P_{i,1}, ..., P_{i,T_i}.
    targets : list of floats, same length as paths -- targets[i] is
              opportunity i's target level E_0[P_{i,T_i}]. Passing 0.0 for a
              path that's already centered (e.g. already expressed as
              deviations, or centered on entry price by the caller) is fine.

    --- Output ---
    (X, Y) : 1-D numpy arrays of equal length, ready for estimate_ou_params.

    --- LOAD-BEARING note on real-data target choice (see chapter_13_otr.py) ---
    This function is deliberately agnostic about WHAT the target represents --
    that decision belongs to the caller. On our real BTC pipeline this
    matters a lot: Ch03's triple_barrier.py sets barriers long-only
    (side_=1 always), so the "natural" profit-taking target
    entry_price*(1+trgt) is only a genuine forecast for opportunities that
    actually resolved as wins. Using it for ALL opportunities (including
    bin=-1 losses, which drift AWAY from that always-long target) pools a
    converging process with a diverging one and produced phi_hat=1.027 on
    real data -- non-stationary. See chapter_13_otr.py for the full
    real-data investigation. Entry-price centering is the FINAL choice used
    there (2026-07-22 decision, not an interim placeholder) -- a deeper
    random-walk finding surfaced regardless of target choice, and is
    reported as this chapter's real, book-consistent result rather than
    something pending further work.
    """
    if len(paths) != len(targets):
        raise ValueError(
            f'paths and targets must have the same length, '
            f'got {len(paths)} and {len(targets)}'
        )
    X, Y = [], []
    for path, target in zip(paths, targets):
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            continue  # need at least one (t-1, t) pair
        dev = path - target
        X.extend(dev[:-1])
        Y.extend(dev[1:])
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def estimate_ou_params(X, Y):
    """OLS estimate of the O-U parameters {phi, sigma} -- AFML eq. 13.7.

    phi_hat   = cov[Y, X] / cov[X, X]   (the O-U autoregressive coefficient)
    xi_hat    = Y - phi_hat * X          (residuals)
    sigma_hat = sqrt(cov[xi_hat, xi_hat])

    Uses population covariance (divide by N, not N-1) to match cov[.,.] as
    used elsewhere in the book (e.g. Ch04's uniqueness weighting) -- the
    choice doesn't matter for phi_hat (numerator and denominator both scale
    by the same 1/N or 1/(N-1)), but it does very slightly affect sigma_hat.

    --- Inputs ---
    X, Y : 1-D array-likes of equal length (typically build_xy_from_opportunities's output)

    --- Output ---
    (phi_hat, sigma_hat) : floats
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError(f'X and Y must have the same length, got {len(X)} and {len(Y)}')
    if len(X) < 2:
        raise ValueError('need at least 2 (X, Y) pairs to estimate phi and sigma')

    cov_yx = np.mean(X * Y) - X.mean() * Y.mean()
    cov_xx = np.mean(X * X) - X.mean() ** 2
    if cov_xx == 0:
        raise ValueError('cov[X, X] == 0 -- X has no variance, phi is undefined')
    phi_hat = cov_yx / cov_xx

    resid = Y - phi_hat * X
    sigma_hat = np.sqrt(np.mean(resid * resid) - resid.mean() ** 2)

    return float(phi_hat), float(sigma_hat)


def phi_to_half_life(phi):
    """Convert phi to half-life tau -- AFML Section 13.5.1: tau = -log(2)/log(phi).

    Requires phi in (0, 1) for a meaningful, positive half-life (this is a
    stricter requirement than the O-U STATIONARITY condition phi in (-1, 1)
    quoted after eq. 13.4 -- a stationary process with phi in (-1, 0] doesn't
    have a real-valued half-life under this formula, since log(phi) is
    undefined or the ratio is negative).

    Returns np.nan (does NOT raise) for phi outside (0, 1) -- by design, so
    callers can report a non-stationary/degenerate phi_hat as a real finding
    (see chapter_13_otr.py's real-data result) rather than crash.
    """
    if not (0 < phi < 1):
        return float('nan')
    return -np.log(2) / np.log(phi)


def half_life_to_phi(tau):
    """Convert half-life tau to phi -- AFML Section 13.5.1: phi = 2**(-1/tau)."""
    if tau <= 0:
        raise ValueError(f'half-life must be positive, got {tau}')
    return 2 ** (-1.0 / tau)


def simulate_ou_path(phi, sigma, forecast, pt, sl, max_hp, seed=0.0, rng=None, random_state=None):
    """Simulate one synthetic O-U path under one candidate trading rule.

    Mirrors the inner loop of Snippet 13.2 exactly: starting from `seed`,
    repeatedly draw a new price under the O-U recursion (eq. 13.2) until the
    deviation from seed (cP) crosses the profit-take barrier (+pt), the
    stop-loss barrier (-sl), or the maximum holding period (max_hp) is
    reached -- whichever comes first. This max-holding-period exit is
    exactly the "vertical bar" from the triple-barrier method (Chapter 3).

    --- Inputs ---
    phi, sigma : floats -- O-U parameters (see estimate_ou_params)
    forecast   : float  -- the target level E_0[P_{i,T_i}], expressed in the
                 SAME centered coordinates as seed (book fixes P_{i,0}=0
                 without loss of generality; we keep that convention here --
                 forecast and seed are both "distance from wherever this
                 opportunity started")
    pt, sl     : floats, both >= 0 -- profit-take and stop-loss thresholds
                 (distance from seed; sl is a positive magnitude, the actual
                 stop-loss barrier sits at -sl)
    max_hp     : int -- maximum holding period (bars)
    seed       : float -- starting price (book: P_{i,0}; default 0.0)
    rng        : optional callable taking no arguments, returning one N(0,1)
                 draw. Injectable for deterministic testing (see test_otr.py's
                 _fixed_shocks). If provided, `random_state` is ignored.
    random_state : optional int (or None) -- seeds a fresh
                 numpy.random.default_rng for this call when `rng` is not
                 given. None (default) means non-reproducible (fresh OS
                 entropy each call), matching sklearn's random_state=None
                 convention used elsewhere in this pipeline (e.g. Ch09's
                 SVC). Pass an int for a reproducible single path.

    --- Output ---
    (cP, hp) : the exit deviation from seed, and the holding period (bars)
               at which the exit occurred.

    --- LOAD-BEARING note (bug fixed here) ---
    An earlier version defaulted `rng` to `lambda: gauss(0, 1)`, drawing from
    PYTHON'S BUILT-IN `random` module. That module is entirely separate from
    numpy's RNG state, so `np.random.seed(...)` calls elsewhere in the demo
    script (and in an earlier version of the book-validation TDD test) had
    ZERO effect on these draws -- real runs were silently non-reproducible
    despite code that looked seeded. Fixed by switching to an explicit,
    injectable numpy Generator (random_state convention, not global state).
    """
    if rng is None:
        rng = np.random.default_rng(random_state).normal

    p, hp = seed, 0
    while True:
        p = (1 - phi) * forecast + phi * p + sigma * rng()
        cP = p - seed
        hp += 1
        if cP > pt or cP < -sl or hp > max_hp:
            return cP, hp


def batch(coeffs, n_iter=100_000, max_hp=100,
          r_pt=None, r_sl=None, seed=0.0, rng=None, random_state=None):
    """Sweep a mesh of (profit-take, stop-loss) pairs -- AFML Snippet 13.2.

    For each node (pt, sl) in the Cartesian product of r_pt x r_sl, simulate
    n_iter independent synthetic O-U paths (simulate_ou_path) under that
    trading rule, and report the resulting Sharpe ratio (mean/std of the
    n_iter exit values) -- this is equation (13.1)'s SR_R, estimated via
    simulation rather than a closed form.

    --- Inputs ---
    coeffs : dict with keys 'forecast', 'hl' (half-life), 'sigma' -- mirrors
             the book's Snippet 13.1 `coeffs` dict exactly. phi is derived
             from 'hl' via half_life_to_phi.
    n_iter : int -- synthetic paths per mesh node (book default 1e5; tests
             and quick exploration use far fewer for speed)
    max_hp : int -- maximum holding period per path
    r_pt, r_sl : 1-D array-likes -- profit-take / stop-loss thresholds to
             sweep. Book default (Table 13.1 style, assumes sigma=1) is
             np.linspace(0, 10, 21) for both -- i.e. thresholds expressed as
             MULTIPLES of sigma. On real data with sigma far from 1 (e.g.
             our real BTC sigma_hat ~ 690), the caller MUST scale r_pt/r_sl
             by the real sigma (e.g. sigma_hat * np.linspace(0, 10, 21)) or
             every node will trigger on bar-to-bar noise. See
             chapter_13_otr.py for the real-data version of this scaling.
    seed   : float -- starting price for every simulated path (book: 0)
    rng    : optional callable, see simulate_ou_path. If provided, used
             as-is for every path in every node (caller's responsibility to
             manage its state); `random_state` is ignored.
    random_state : optional int (or None) -- when `rng` is not given, ONE
             numpy.random.default_rng(random_state) is created here and
             reused across every node and every path in the sweep (NOT
             reset per node -- resetting per node would make every mesh
             cell draw the identical shock sequence, which is wrong). None
             (default) means non-reproducible; pass an int for a
             reproducible full mesh.

    --- Output ---
    list of (pt, sl, mean, std, sharpe) tuples, one per mesh node, in the
    same product(r_pt, r_sl) order as the book's Snippet 13.2 (NOT a
    DataFrame -- kept as the book's own tuple-list structure; convert to a
    DataFrame/pivot for heat-mapping in the demo script).

    --- Book erratum fixed here ---
    Snippet 13.2 uses `print comb_[0],comb_[1],mean,std,mean/std` (Python-2
    print-statement syntax) inside the loop -- a SyntaxError under Python 3.
    Removed the print entirely (results are returned, not printed, which
    also makes this function testable) rather than just adding parentheses.
    """
    if r_pt is None:
        r_pt = np.linspace(0, 10, 21)
    if r_sl is None:
        r_sl = np.linspace(0, 10, 21)
    if rng is None:
        rng = np.random.default_rng(random_state).normal

    phi = half_life_to_phi(coeffs['hl'])
    sigma = coeffs['sigma']
    forecast = coeffs['forecast']

    output = []
    for pt, sl in product(r_pt, r_sl):
        exits = np.empty(int(n_iter))
        for i in range(int(n_iter)):
            cP, _hp = simulate_ou_path(
                phi, sigma, forecast, pt, sl, max_hp, seed=seed, rng=rng
            )
            exits[i] = cP
        mean, std = exits.mean(), exits.std()
        sharpe = mean / std if std > 0 else float('nan')
        output.append((pt, sl, mean, std, sharpe))
    return output


def best_node(results):
    """Given batch()'s output, return the (pt, sl, mean, std, sharpe) tuple
    with the highest Sharpe ratio -- AFML Step 5a: pick R* = argmax{SR_R}.
    NaN Sharpes (degenerate std==0 nodes) are ignored.
    """
    valid = [r for r in results if not np.isnan(r[4])]
    if not valid:
        raise ValueError('no valid (non-NaN Sharpe) nodes in results')
    return max(valid, key=lambda r: r[4])


# ---------------------------------------------------------------------------
# TDD results (test_otr.py), embedded per project convention.
# Expected values hand-derived from eq. 13.5-13.7 (estimation) and Snippet
# 13.2's recursion (simulation); see test_otr.py's docstrings/comments for
# the full hand traces.
# ============================================================================
# REAL-MACHINE CONFIRMED (Python 3.10.20 / pytest 9.0.3 / mlfinlab env) --
# 19 passed in 17.32s. Also verified reproducible: chapter_13_otr.py run
# twice back-to-back on the real machine post-fix produced byte-identical
# Part A/C output both times (Sharpe values, best-node coordinates, mesh
# range all matched exactly).
#
# ============================= test session starts ==============================
# test_otr.py::test_build_xy_matches_book_equation_13_6_toy_example PASSED  [  5%]
# test_otr.py::test_build_xy_skips_single_observation_paths PASSED         [ 10%]
# test_otr.py::test_build_xy_rejects_mismatched_lengths PASSED             [ 15%]
# test_otr.py::test_estimate_ou_params_hand_traced PASSED                  [ 21%]
# test_otr.py::test_estimate_ou_params_recovers_known_process PASSED       [ 26%]
# test_otr.py::test_estimate_ou_params_rejects_degenerate_x PASSED         [ 31%]
# test_otr.py::test_phi_to_half_life_known_value PASSED                    [ 36%]
# test_otr.py::test_half_life_to_phi_known_value PASSED                    [ 42%]
# test_otr.py::test_half_life_phi_round_trip PASSED                       [ 47%]
# test_otr.py::test_phi_to_half_life_returns_nan_for_non_stationary_phi PASSED [ 52%]
# test_otr.py::test_half_life_to_phi_rejects_nonpositive_half_life PASSED  [ 57%]
# test_otr.py::test_simulate_ou_path_hand_traced_exit_via_profit_take PASSED [ 63%]
# test_otr.py::test_simulate_ou_path_exits_via_stop_loss PASSED            [ 68%]
# test_otr.py::test_simulate_ou_path_exits_via_time_barrier PASSED         [ 73%]
# test_otr.py::test_simulate_ou_path_forecast_sign_symmetry PASSED         [ 78%]
# test_otr.py::test_batch_returns_expected_structure PASSED                [ 84%]
# test_otr.py::test_batch_book_reproduces_approx_sharpe_forecast0_hl5 PASSED [ 89%]
# test_otr.py::test_best_node_ignores_nan_sharpes PASSED                   [ 94%]
# test_otr.py::test_best_node_raises_if_all_nan PASSED                     [100%]
# ============================== 19 passed in 17.32s ===============================
#
# Notes on tests that pin real book claims:
#
#  * test_batch_book_reproduces_approx_sharpe_forecast0_hl5 -- BOOK VALIDATION.
#    Section 13.6.1 states Sharpe "reaching levels of around 3.2" for
#    {forecast=0, hl=5, sigma=1}. Our best node (reduced mesh/nIter for test
#    speed) landed at Sharpe ~3.3-4.0 depending on seed -- right ballpark,
#    same qualitative shape (narrow profit-take, wide stop-loss wins). The
#    full-resolution demo script run (Part A) got 4.01 and 12.99 against the
#    book's stated ~3.2 and ~12.0 for the two reproduced cases.
#
#  * test_simulate_ou_path_forecast_sign_symmetry -- pins the book's own
#    stated conjecture (Section 13.6.3): "Figure 13.6 resembles a rotated
#    photographic negative of Figure 13.16." Proven exactly (not just
#    approximately) by negating both forecast and every injected shock and
#    checking the resulting path is the exact negative at every step.
#
#  * test_phi_to_half_life_returns_nan_for_non_stationary_phi -- this
#    behavior is what let Part B of chapter_13_otr.py report the real
#    phi_hat=1.042 finding gracefully instead of crashing on log(negative)
#    or log(>1) edge cases.
#
# POST-REAL-MACHINE FIX: after the above suite passed on both sandbox and
# the real mlfinlab machine, a reproducibility bug was caught (not a
# correctness bug -- all 19 tests passed either way): simulate_ou_path's
# default rng drew from Python's built-in `random` module, so np.random.seed
# calls elsewhere (in this test file and in chapter_13_otr.py) had NO effect
# on the actual simulation randomness -- real runs were silently
# non-reproducible despite looking seeded. Fixed by switching to an
# explicit, injectable numpy Generator (random_state parameter, matching
# the sklearn random_state convention already used in Ch09's SVC). Suite
# re-run after the fix: still 19/19 passed, same count, same test names --
# only the internal seeding mechanism changed.
