"""
Chapter 16: Machine Learning Asset Allocation -- Out-of-Sample Monte Carlo
============================================================================

Implements AFML Snippet 16.5 (Appendix 16.A.4): the out-of-sample Monte
Carlo comparison of HRP, CLA (minimum-variance), and IVP allocations
(Section 16.6). This is deliberately synthetic, per this chapter's own
book-fidelity/real-data-policy decision (see ch16/README.md): the whole
point of this experiment is to inject KNOWN, controlled shocks (one
common, one idiosyncratic, each positive and negative) and observe how
each method responds -- that requires synthetic control, the same
category of exception already sanctioned for Ch08.

Why HRP wins here, in one sentence: CLA's minimum-variance objective
concentrates weight wherever in-sample covariance looks cheapest, which
is exactly where estimation error hides; HRP's tree structure limits how
far a single bad correlation estimate can propagate, so it's more robust
out-of-sample even though it isn't explicitly minimizing variance at all.

Python-2-to-3 fixes (documented, NOT book bugs -- language-version
issues only; core Monte Carlo logic unchanged from the printed snippet):
  1. `xrange` -> `range`, `print x` -> `print(x)`.
  2. Global `np.random.normal(...)` / `np.random.randint(...)` /
     `random.randint(...)` -> threaded seeded `numpy.random.Generator`,
     per project convention (see generateData/hrpMC `random_state` args).
     `random.randint(a, b)` is inclusive on both ends; the Generator
     equivalent is `rng.integers(a, b+1)` (exclusive upper by default).
  3. `r[func.__name__] = r[func.__name__].append(r_)` on an
     empty-initialized `pd.Series` -- the exact same fragile pattern
     already flagged as a project gotcha (`getQuasiDiag`'s
     `sortIx.append`, fixed the same way there). Fixed here by
     collecting each rebalance period's returns into a list and
     `pd.concat`-ing once per Monte Carlo iteration, rather than
     repeated in-loop `.append()`.
  4. `stats[func.__name__].loc[numIter] = ...` on an empty-initialized
     `pd.Series` -- same gotcha, same fix: accumulate into a
     dict-of-lists across iterations, build the DataFrame once at the
     end instead of assigning into a pre-declared empty Series row by
     row.
  5. `stats.to_csv('stats.csv')` -- hardcoded relative path, not
     portable. Replaced with an optional `output_csv_path` parameter
     (caller decides where to save, matching this project's
     `__file__`-derived-root convention for `.py` scripts).

Multiprocessing (added 2026-08-10, not in the printed book snippet):
each of the numIters Monte Carlo iterations is fully independent (its
own simulated path, its own walk-forward evaluation), so this is an
embarrassingly parallel workload. Reuses this project's existing AFML
multiprocessing engine (utils/multiprocess.py's process_jobs/
process_jobs_mp), mirroring the ALREADY-ESTABLISHED pattern in
ch04/sample_weights/monte_carlo.py's main_mc (Snippet 4.9): one job
dict per iteration, process_jobs (num_threads==1) or process_jobs_mp
(num_threads>1) -- not utils/multiprocess.py's separate mp_pandas_obj/
molecule-chunking path, which is designed for splitting a single large
pandas index rather than running N independent trials.

One adaptation beyond ch04's precedent: ch04's aux_mc relies on
UNSEEDED global np.random state per worker process for its randomness
(fine there since Chapter 4 predates this project's later seeded-
Generator convention). Here, each iteration gets its own INDEPENDENT
child numpy.random.SeedSequence, spawned up front from one base seed
via SeedSequence.spawn(numIters) -- numpy's documented pattern for
reproducible parallel streams. This is necessary, not just stylistic:
a single shared Generator object CANNOT safely cross process
boundaries -- each worker process gets its own pickled COPY of
whatever's passed to it, so "sharing" one Generator across workers
would silently produce CORRELATED (not independent) random streams
per worker, not a genuine parallel Monte Carlo. Consequence: hrpMC's
random_state parameter takes an int seed (or SeedSequence), not a
Generator -- unlike generateData's own random_state, which still takes
a Generator directly (unaffected, still fine for direct/sequential use
elsewhere). Given the SAME base seed, results are IDENTICAL regardless
of num_threads (verified in test_monte_carlo.py) -- num_threads only
changes wall-clock time, never the answer.

BLAS thread-capping: the env-var sets at the very top of this file
(before `import numpy`) cap OMP/MKL/OPENBLAS/NUMEXPR to 1 thread each,
mirroring this project's existing conftest.py pattern for heavy
sklearn estimators. Matters here because each worker PROCESS also
imports numpy fresh; without this, N worker processes each spawning
their own multi-threaded BLAS pool would oversubscribe the CPU (N
processes x M BLAS threads each, only N cores available) and could
easily run SLOWER than single-threaded despite the added parallelism.
Placed before `import numpy` specifically because MKL reads these vars
at first use, not dynamically afterward -- setting them post-import
would not reliably take effect. Harmless in the main process too (has
no effect if numpy's already loaded there by the time this module is
imported by the chapter driver).

Windows note (same constraint as ch04's main_mc and this project's
other process_jobs_mp/joblib callers): num_threads > 1 requires the
entry point to be behind `if __name__ == '__main__':` -- already true
of chapter_16_hrp.py's driver block.
"""
import os

# BLAS thread-capping MUST happen before numpy is imported anywhere in
# this process -- see module docstring above.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import sys
import warnings
import importlib.util

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


def _load_module_from_path(module_name, file_path):
    """Load a module by explicit file path into a UNIQUELY-named module
    object, sidestepping sys.path name collisions.

    Why this instead of `sys.path.insert(...)` + `from hrp import ...`:
    when this module is imported through the full chapter driver
    (chapter_16_hrp.py), ch16/ is already on sys.path so that
    `from hrp.hrp import ...` resolves the `hrp` PACKAGE (ch16/hrp/).
    If this module ALSO inserts ch16/hrp/ onto sys.path and does
    `from hrp import correlDist`, Python's import system can resolve
    the bare name `hrp` to either the package or the module depending
    on insertion order -- an ambiguous collision that broke exactly
    this way when this module was first wired into the chapter driver
    (see project handoff). Loading by explicit absolute path avoids the
    ambiguity entirely, matching this project's `__file__`-derived
    fully-qualified import convention (ch10/ch13 style).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HERE = os.path.dirname(os.path.abspath(__file__))
_hrp_mod = _load_module_from_path(
    'ch16_hrp_impl', os.path.join(_HERE, '..', 'hrp', 'hrp.py'))
_cla_mod = _load_module_from_path(
    'ch16_cla_impl', os.path.join(_HERE, '..', 'cla', 'cla.py'))

correlDist = _hrp_mod.correlDist
getIVP = _hrp_mod.getIVP
getQuasiDiag = _hrp_mod.getQuasiDiag
getRecBipart = _hrp_mod.getRecBipart
CLA = _cla_mod.CLA

# utils/multiprocess.py is a REAL package (utils/__init__.py exists at repo
# root) -- deliberately NOT loaded via _load_module_from_path's synthetic-
# name trick like hrp.py/cla.py above. process_jobs_mp's pool.map target
# (_job_wrapper, defined inside utils/multiprocess.py) must be resolvable
# by a genuinely importable dotted path in a freshly spawned WORKER
# process (Windows spawn re-imports fresh, it doesn't share the parent's
# in-memory module objects) -- a synthetic name like 'ch16_multiprocess_impl'
# has no real file/package anywhere any child process could `import` by
# that name. Standard sys.path + `from utils.multiprocess import ...` is
# the same pattern already proven working on the real Windows machine by
# ch04/ch08/ch09/ch20's own use of this exact module.
_AFML_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _AFML_ROOT not in sys.path:
    sys.path.insert(0, _AFML_ROOT)
from utils.multiprocess import process_jobs, process_jobs_mp  # noqa: E402


# =============================================================================
# Snippet 16.5: generateData (Monte Carlo version -- WITH injected shocks)
# =============================================================================
def generateData(nObs, sLength, size0, size1, mu0, sigma0, sigma1F,
                  random_state=None):
    """Simulate one Monte Carlo path of correlated returns, with two
    injected shocks: one COMMON (affects a base series and one of its
    correlated perturbations simultaneously) and one IDIOSYNCRATIC
    (affects a single base series only). Each shock type has one
    strongly negative and one strongly positive occurrence.

    NOT the same function as hrp.py's generateData (Snippet 16.4) --
    same name, different signature and purpose, exactly as the book
    itself redefines it in this Appendix. Kept in this separate module
    to avoid a same-name collision; callers importing both should alias
    one on import.

    Returns
    -------
    x : ndarray, shape (nObs, size0+size1)
    cols : list of int
        Base-column indices used to build each of the size1 perturbation
        series (same semantics as hrp.py's generateData).
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    # 1) generate random uncorrelated data
    x = rng.normal(mu0, sigma0, size=(nObs, size0))
    # 2) create correlation between the variables
    cols = rng.integers(0, size0, size=size1).tolist()
    y = x[:, cols] + rng.normal(0, sigma0 * sigma1F, size=(nObs, len(cols)))
    x = np.append(x, y, axis=1)
    # 3) add common random shock -- hits a base column (cols[0]) AND the
    # first perturbation column (index `size0`, i.e. y's first column)
    # simultaneously, at the SAME two random time points: one big
    # negative co-move (-50%/-50%), one big positive co-move (+200%/+200%).
    point = rng.integers(sLength, nObs - 1, size=2)
    x[np.ix_(point, [cols[0], size0])] = np.array([[-.5, -.5], [2, 2]])
    # 4) add specific (idiosyncratic) random shock -- hits only a single
    # base column (cols[-1]) at two OTHER random time points, unrelated
    # to any other series.
    point = rng.integers(sLength, nObs - 1, size=2)
    x[point, cols[-1]] = np.array([-.5, 2])
    return x, cols


# =============================================================================
# Snippet 16.5: getHRP -- thin wrapper matching the book's local re-import
# =============================================================================
def getHRP(cov, corr):
    """Construct the HRP allocation. Identical logic to hrp.py's own
    getHRP (Snippet 16.4) -- reused directly here (not duplicated) via
    the correlDist/getQuasiDiag/getRecBipart imports above, matching
    this project's "extend/reuse, don't duplicate" convention. Kept as
    its own top-level function (rather than just importing hrp.getHRP
    under this name) so hrpMC's `methods=[getIVP, getHRP, getCLA]` list
    comprehension gets the right `__name__` for column labeling, exactly
    matching the book's own local redefinition in this Appendix.
    """
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=sch.ClusterWarning,
            message='.*uncondensed distance matrix.*'
        )
        link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


# =============================================================================
# Snippet 16.5: getCLA -- CLA's minimum-variance portfolio
# =============================================================================
def getCLA(cov, **kargs):
    """Compute CLA's minimum-variance portfolio.

    Book's own comment: "mean=... # Not used by C[LA min-var] portf[olio]"
    -- since only the LAST turning point (lambda=0, the global min-var
    solution) is returned, the specific mean vector used to seed the
    turning-point search doesn't affect this particular output (mean
    only determines the ORDER turning points are visited, not the
    min-var endpoint itself). `**kargs` absorbs the `corr=` keyword that
    hrpMC passes uniformly to all three methods, matching getIVP's
    existing `**kargs` swallow in hrp.py.
    """
    mean = np.arange(cov.shape[0]).reshape(-1, 1).astype(float)
    lB = np.zeros(mean.shape)
    uB = np.ones(mean.shape)
    cla = CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


# =============================================================================
# Snippet 16.5 (adapted): _run_iteration -- ONE Monte Carlo trial
# =============================================================================
def _run_iteration(seed, nObs, size0, size1, mu0, sigma0, sigma1F, sLength,
                    rebal, pointers):
    """Run exactly one Monte Carlo iteration and return its cumulative
    out-of-sample return for each of the three methods. Top-level
    (module-scope) function, not a closure -- required so it (and the
    job dicts referencing it) can be pickled to worker processes.

    `seed` is this iteration's own independent numpy.random.SeedSequence
    (see hrpMC's docstring for why -- a shared Generator can't safely
    cross process boundaries).
    """
    rng = np.random.default_rng(seed)
    methods = [getIVP, getHRP, getCLA]
    x, cols = generateData(nObs, sLength, size0, size1, mu0, sigma0,
                            sigma1F, random_state=rng)
    # Fix #3 (see module docstring): accumulate each rebalance period's
    # returns into a list, pd.concat once -- NOT repeated Series.append().
    period_returns = {m.__name__: [] for m in methods}
    for pointer in pointers:
        x_ = x[pointer - sLength:pointer]
        cov_, corr_ = np.cov(x_, rowvar=False), np.corrcoef(x_, rowvar=False)
        x_oos = x[pointer:pointer + rebal]
        for func in methods:
            w_ = func(cov=cov_, corr=corr_)
            r_ = pd.Series(np.dot(x_oos, w_))
            period_returns[func.__name__].append(r_)
    result = {}
    for name, parts in period_returns.items():
        r_ = pd.concat(parts, ignore_index=True)
        p_ = (1 + r_).cumprod()
        result[name] = p_.iloc[-1] - 1
    return result


# =============================================================================
# Snippet 16.5: hrpMC -- the Monte Carlo driver
# =============================================================================
def hrpMC(numIters=10000, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2,
          sigma1F=.25, sLength=260, rebal=22, random_state=12345,
          num_threads=1, output_csv_path=None, verbose=True):
    """Out-of-sample Monte Carlo comparison of IVP, HRP, and CLA
    (Section 16.6). Book defaults kept exactly as printed (numIters=1e4,
    nObs=520 [2yrs daily], sLength=260 [1yr lookback], rebal=22
    [~monthly]) -- "chosen arbitrarily" per the book's own text, not
    re-tuned here.

    Each of numIters iterations: simulate one path (with shocks), walk
    forward through `pointers` = every `rebal` observations starting at
    `sLength`, computing each method's allocation from the trailing
    `sLength`-observation window and evaluating it on the NEXT `rebal`
    observations (genuinely out-of-sample, no look-ahead). Cumulative
    return over the full walk-forward is this iteration's result for
    each method; after all iterations, report std/var per method.

    Parameters
    ----------
    random_state : int or numpy.random.SeedSequence, default 12345
        Base seed. numIters independent child seeds are spawned from
        this up front (SeedSequence.spawn), one per iteration -- see
        module docstring for why this differs from generateData's own
        random_state (which still takes a Generator directly). Same
        base seed -> identical `stats` output regardless of num_threads.
    num_threads : int, default 1
        1 = sequential (utils.multiprocess.process_jobs). >1 = parallel
        across that many worker processes (process_jobs_mp). On
        Windows, num_threads>1 requires the caller to be behind
        `if __name__ == '__main__':`.

    Returns
    -------
    stats : pd.DataFrame, columns ['getIVP', 'getHRP', 'getCLA']
        One row per Monte Carlo iteration: that iteration's cumulative
        out-of-sample return for each method.
    summary : pd.DataFrame
        std, var, and variance-vs-HRP-ratio-minus-1 per method --
        the exact quantities quoted in the book's prose (sigma^2_CLA,
        sigma^2_IVP, sigma^2_HRP, and CLA/IVP's "% greater variance
        than HRP").
    """
    numIters = int(numIters)
    pointers = list(range(sLength, nObs, rebal))

    base_seed = (random_state if isinstance(random_state, np.random.SeedSequence)
                 else np.random.SeedSequence(random_state))
    child_seeds = base_seed.spawn(numIters)

    jobs = [{'seed': s, 'nObs': nObs, 'size0': size0, 'size1': size1,
             'mu0': mu0, 'sigma0': sigma0, 'sigma1F': sigma1F,
             'sLength': sLength, 'rebal': rebal, 'pointers': pointers}
            for s in child_seeds]

    if verbose:
        mode = 'single-threaded' if num_threads == 1 else f'{num_threads} parallel workers'
        print(f"Running {numIters} Monte Carlo iterations ({mode})...")

    if num_threads == 1:
        out = process_jobs(_run_iteration, jobs)
    else:
        out = process_jobs_mp(_run_iteration, jobs, num_threads)

    # 5) report results
    stats = pd.DataFrame(out, columns=['getIVP', 'getHRP', 'getCLA'])
    if output_csv_path is not None:
        stats.to_csv(output_csv_path)
    df0, df1 = stats.std(), stats.var()
    summary = pd.concat([df0, df1, df1 / df1['getHRP'] - 1], axis=1)
    summary.columns = ['std', 'var', 'var_vs_HRP_minus_1']
    if verbose:
        print(summary)
    return stats, summary
