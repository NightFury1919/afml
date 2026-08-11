# Chapter 16 — Machine Learning Asset Allocation: Hierarchical Risk Parity

Classical mean-variance optimization (and its numerically-stable cousin,
CLA) requires inverting the covariance matrix — treating every asset as a
potential substitute for every other. A small estimation error in any one
correlation can swing the whole solution ("Markowitz's curse", Sec 16.3).
**HRP never inverts the covariance matrix.** It replaces the complete graph
of pairwise relationships with a tree: cluster similar assets (Stage 1),
reorder the covariance matrix so similar assets sit together (Stage 2,
"quasi-diagonalization"), then allocate weight top-down by recursively
splitting each branch in inverse proportion to its variance (Stage 3). No
inversion, no positive-definiteness requirement — HRP works even on a
singular covariance matrix, exactly where CLA breaks down.

## What's implemented

| Topic | Book section | Snippet | Status |
|---|---|---|---|
| Inverse-variance portfolio (`getIVP`) | 16.4, Appx 16.A.2 | 16.4 (helper) | Implemented |
| Cluster variance (`getClusterVar`) | 16.4.3 | 16.4 (helper) | Implemented |
| Quasi-diagonalization (`getQuasiDiag`) | 16.4.2 | 16.2 / 16.4 | Implemented |
| Recursive bisection (`getRecBipart`) | 16.4.3 | 16.3 / 16.4 | Implemented |
| Correlation-distance metric (`correlDist`) | Appx 16.A.1 | 16.4 | Implemented |
| Full HRP driver (`getHRP`), tree clustering (16.1) | 16.4.1–16.4.3 | 16.1 / 16.4 | Implemented |
| Synthetic data generator (`generateData`) | 16.4 | 16.4 | Implemented |
| Correlation heatmap (`plotCorrMatrix`) | 16.4 | 16.4 | Implemented |
| Monte Carlo out-of-sample comparison (HRP vs CLA vs IVP) | 16.5 / 16.6 | 16.5 | Implemented, real-machine confirmed 2026-08-10 |

## Files

- `hrp/hrp.py` — Snippets 16.1–16.4: `getIVP`, `getClusterVar`,
  `getQuasiDiag`, `getRecBipart`, `correlDist`, `getHRP` (driver tying
  stages 1–3 together), `plotCorrMatrix`, `generateData`. Function/variable
  names kept exactly as printed in the book (not rewritten to snake_case),
  matching the Ch15 `probFailure`/`binHR` convention.
- `hrp/test_hrp.py` — 24 tests, several hand-traced against the book's own
  worked Examples 16.1–16.6 (independently re-derived before being
  embedded, not just read off scipy's output) — see below.
- `data_loader/continuous_futures.py` — project-specific real-data
  infrastructure (not a book snippet): builds one continuous,
  roll-gap-corrected front-month price series per commodity from raw
  per-contract files, reusing `ch02/multi_product/roll.py` (Snippets
  2.2/2.3) rather than duplicating it.
- `data_loader/test_continuous_futures.py` — 17 tests: synthetic
  regression tests that reproduce each of the three real bugs below in
  miniature, plus bounded sanity/regression guards run against the real
  six-commodity dataset.
- `chapter_16_hrp.py` / `.ipynb` (at `ch16/` root, per the Ch19-onward
  convention) — three-part demo: Part A, the book's own synthetic
  numerical example; Part B, the same three stages applied to six real
  commodity futures; Part C, the Section 16.5/16.6 Monte Carlo
  comparison (HRP vs CLA vs IVP), book-exact synthetic methodology.
- `cla/cla.py` — the Critical Line Algorithm, faithfully translated from
  Bailey & Lopez de Prado (2013)'s Appendix A.1 (the CLA reference the
  AFML book itself defers to for Section 16.5/16.6). NOT an AFML book
  snippet -- an external reference implementation, held to the same
  book-fidelity standard.
- `cla/test_cla.py` — 52 tests, hand-traced against the paper's own
  published Section 5 numerical example (Table 1 inputs, Table 2's 10
  turning points, and the getMaxSR/getMinVar figures quoted in the
  paper's prose) — real, independently-published numbers.
- `monte_carlo/monte_carlo.py` — Snippet 16.5 / Appendix 16.A.4: the
  `generateData` (shock-injecting version, distinct from `hrp.py`'s own
  `generateData`), `getHRP`/`getCLA` wrappers, and the `hrpMC` Monte
  Carlo driver.
- `monte_carlo/test_monte_carlo.py` — 14 tests: exact shock-injection
  values (deterministic given a seed), wrapper correctness against the
  underlying `hrp`/`cla` modules, and `hrpMC` end-to-end plumbing on a
  small, fast configuration (the book's own numIters=10000 run isn't
  independently hand-traceable — it's an emergent statistical property,
  not a closed-form value; see file docstring).

**Two subfolders, not one** — unlike prior chapters, `hrp/` (the book
algorithm) and `data_loader/` (this project's real-data infrastructure)
are logically distinct and versioned/tested separately, since the data
loader isn't itself a book snippet.

## Book-snippet fidelity notes (Py2→3 fixes, all in `hrp.py`)

- `getRecBipart`'s bisection used `len(i)/2` — true division in Python 3,
  which breaks list slicing outright (Python 2 relied on implicit integer
  division). Fixed to `len(i)//2`.
- `getQuasiDiag` used `pd.Series.append()` — deprecated in pandas 1.5.3,
  removed entirely in pandas 2.x. Replaced with `pd.concat`.
- `xrange` → `range`, `print x` → `print(x)`.
- **One real bug, not a Py2/3 issue:** the book's
  `w = pd.Series(1, index=sortIx)` creates an **integer-dtype** Series
  (since `1` is a Python int). The loop then does `w[...] *= alpha` with a
  float `alpha`, requiring an int64→float64 upcast on assignment — older
  pandas did this silently, current pandas raises `TypeError` outright.
  Fixed by initializing `w` as `1.0` (float) from the start.

Snippet 16.1's `sch.linkage(dist, 'single')` call is deliberately unusual:
passing the raw 2D distance matrix (not a condensed/squareform vector)
makes scipy compute Euclidean distance *between rows* internally — exactly
the book's "distance of distances" (d-tilde) step described in prose.
This triggers a `ClusterWarning` ("looks suspiciously like an uncondensed
distance matrix"), suppressed deliberately with a comment explaining it's
intentional, not silenced blindly.

`getQuasiDiag`/`getRecBipart` were hand-traced against the book's own
3-asset example independently before being embedded as tests: `correlDist`
reproduces the book's exact distance matrix (0.3873, 0.6325, 0.7746), the
distance-of-distances trick reproduces the book's d-tilde matrix (0.5659,
0.9747, 1.1225) and merge sequence, and the final `sortIx=[2,0,1]`
(0-indexed) / weights `17/37, 10/37, 10/37` were independently re-derived
by hand, not just read off scipy's output.

## Real-data infrastructure: three bugs found and fixed in `continuous_futures.py`

The existing `SP00–SP99` S&P 500 futures (already in this repo, used by
Ch02–Ch05) were rejected as the real-data vehicle for HRP — they're all one
underlying instrument at different maturities, so correlations sit near
1.0 by construction, not a meaningful test of a diversification algorithm.
Instead: **gold, crude oil, corn, live hogs, US T-bonds, British Pound**,
sourced from turtletrader.com (same source as the original SP files),
chosen for genuinely different macro drivers.

1. **Leading-zero date stripping.** `pd.read_csv` auto-infers the
   no-header `Date` column as int64, silently dropping leading zeros from
   any January date (`000104` → `104`), which then fails `%y%m%d` parsing
   outright. Confirmed to hit files across all six commodities. Fixed by
   forcing that column to load as a string first.
2. **Front-month selection vulnerable to single-day data gaps.** When the
   genuinely-dominant contract had one isolated missing row (a
   data-export gap, not a real trading halt), the day-by-day open-interest
   comparison would briefly and spuriously flip to a much less liquid,
   differently-priced contract for that one day, then flip back —
   producing two fake roll gaps bracketing one bad day. Confirmed this
   alone produced a spurious **+20% single-day "return" in T-bonds**
   (1999-12-09). A first attempt using a 5-day rolling mean of open
   interest did NOT fix this (nothing to smooth when the dominant
   contract's row doesn't exist that day at all). Real fix:
   forward-fill short gaps (≤3 business days) within each contract's own
   observed date range *before* the cross-contract comparison.
3. **GBP-specific unit-scale break.** Every British Pound contract from
   2000 onward quotes in plain decimal USD/GBP (~1.4–1.6), while every
   contract through 1999 quotes in "points" (price×100, ~120–235) — a
   genuine unit change by the data provider at the same point their
   export format changed. This alone produced a spurious **+9,796%
   single-day "return"** when a data gap briefly promoted a
   differently-scaled contract. Verified this break is GBP-specific
   (checked all 5 other commodities' old/new-format price-level ratios at
   the same boundary — none show a scale artifact). Fixed via an explicit
   `rescale_new_format_by=100` parameter, applied only to GBP.

## Real-data results (real-machine confirmed, 2026-07-31)

**Part A — synthetic numerical example** (book's own `generateData`,
`nObs=10000, size0=5, size1=5, sigma1=.25`, seed 12345): reproduces
bit-for-bit identically between sandbox and the real `mlfinlab` machine —
full cross-environment reproducibility, as expected from a seeded
`numpy.random.Generator`.

**Part B — six real commodity futures**, 1,749 aligned daily returns,
1996-01-03 to 2002-09-16 (the default `start='1996-01-01'` window — 1996–97
kept as a roll-history-stability buffer, not trimmed):

| Asset | HRP weight | IVP weight |
|---|---|---|
| GBP | 0.4090 | 0.4358 |
| T-bonds | 0.3471 | 0.3161 |
| Gold | 0.1309 | 0.1395 |
| Corn | 0.0575 | 0.0554 |
| Live hogs | 0.0341 | 0.0328 |
| Crude oil | 0.0215 | 0.0203 |

Top-2 concentration: HRP 0.756, IVP 0.752.

GBP and T-bonds dominate both allocations because they're by far the
lowest-volatility assets in this mix — a real, expected consequence of
inverse-variance-driven weighting on this particular asset mix, not a bug.
**HRP and IVP aren't very differentiated on this dataset, and that's a
legitimate, permanent finding worth stating plainly rather than chasing.**
HRP's edge over IVP shows up when there's meaningful correlation
*clustering* to exploit — pairs or groups of assets that move together.
Six genuinely diverse commodities (the whole point of rejecting the
SP00–SP99 approach) don't have much of that clustering structure, so HRP's
tree-based allocation ends up close to plain inverse-variance weighting
here. Consistent with this project's broader "report the real result"
precedent from Ch11–Ch15.

## Section 16.5/16.6 Monte Carlo comparison — implemented and real-machine confirmed, 2026-08-10

CLA reference implementation sourced from Bailey & Lopez de Prado (2013),
"An Open-Source Implementation of the Critical-Line Algorithm for
Portfolio Optimization" (Algorithms 6, 169-196) — the exact paper AFML
itself cites for CLA, since CLA is not printed in the book. Per this
project's book-fidelity convention extended to external referenced
material: built from Ethan's supplied copy of the paper (Appendix A.1's
full class), not reconstructed from memory.

**Scope decision (confirmed with Ethan, 2026-08-10):** this Monte Carlo
experiment is deliberately **synthetic-only**, no real-data supplement —
the same category of exception already sanctioned for Ch08, not a
cost-of-effort shortcut. The experiment's entire point is to inject
KNOWN, controlled shocks (one common, one idiosyncratic, each positive
and negative) and observe each method's response; that requires
synthetic control by construction. It's also not part of the core
dollar-bar/CUSUM/triple-barrier/classifier pipeline — HRP/CLA/IVP is a
separate, self-contained portfolio-allocation topic. Practically: the
real 6-commodity dataset is ~1,749 days (~7 years) vs. the book's
520-obs/2-year synthetic windows resampled 10,000 times — bootstrapping
one real 7-year history 10,000 times wouldn't add real information, just
resample the same limited stretch.

**Real bug found in the reference paper (Ethan sign-off obtained,
2026-08-10), not a Py2/3 issue:** `computeLambda`'s `c==0` branch does a
bare `return` (i.e. returns `None`), but every caller unpacks
`l, bi = self.computeLambda(...)` — that crashes with `TypeError` trying
to unpack `None`, in Python 2 as much as Python 3. Fixed by returning
`(None, None)` and having every caller explicitly skip `None`
candidates. See `cla/cla.py` module docstring and inline comments for
the full Py2/3 fix list (5 items, all standard language-version
translations) plus this one genuine reference-paper bug.

**Validated against the paper's own published numbers** (`cla/test_cla.py`,
52/52 passing in sandbox): all 10 of the paper's Table 2 turning points'
return/risk/lambda match exactly; `getMinVar()` risk=0.2052 and
`getMaxSR()`=(Sharpe 4.4535, risk 0.2274) match the paper's prose exactly;
an independent algebraic check (AFML Appendix 16.A.2: on a diagonal
covariance matrix, CLA's min-var solution must reduce to plain IVP)
passes. Five of the ten turning points show a harmless tied-lambda
degeneracy (documented in the test file) where portfolio-level
return/risk still match exactly but weight redistributes among three
specific near-tied assets — not a bug, a known Markowitz-CLA sensitivity.

**Book-scale run completed on the real machine, 2026-08-10 — results
closely match the book, both directionally AND in magnitude:**

| | This run | Book | Diff |
|---|---|---|---|
| σ²_CLA | 0.1152 | 0.1157 | 0.4% |
| σ²_IVP | 0.0940 | 0.0928 | 1.3% |
| σ²_HRP | 0.0689 | 0.0671 | 2.7% |
| CLA variance vs HRP | +67.2% | +72.5% | — |
| IVP variance vs HRP | +36.4% | +38.2% | — |

Every variance is within a few percent of the book's published figures,
and the ranking (CLA worst out-of-sample, IVP middle, HRP best) matches
exactly — despite this being a from-scratch CLA translation from an
external paper, a from-scratch Monte Carlo harness, and a different
random seed than the book used. The small residual gaps are consistent
with ordinary Monte Carlo sampling noise (the classic 1/sqrt(n)
standard-error scaling — quadrupling numIters would roughly halve this
gap, not eliminate CLA/IVP/HRP's relative ordering) plus the book not
disclosing its own seed, not evidence of implementation drift.

Ran considerably faster than initial estimates (~55min sandbox
extrapolation, ~38min projected from the small num_threads=4 smoke
test) — likely because per-job pool overhead amortizes much better at
10,000 iterations than it did at the 20-iteration smoke test used to
project that estimate. Exact wall-clock time not yet recorded here;
Ethan to confirm from his terminal timestamps if worth capturing.

Raw per-iteration results saved to `output/monte_carlo_stats.csv`
(10,000 rows, one per Monte Carlo trial, columns getIVP/getHRP/getCLA).

**Multiprocessing (Ethan's request, 2026-08-10):** each of the numIters
Monte Carlo iterations is fully independent, so this is an
embarrassingly parallel workload. `hrpMC` takes a `num_threads`
parameter, wired to this project's EXISTING AFML multiprocessing engine
(`utils/multiprocess.py`'s `process_jobs`/`process_jobs_mp`) rather than
a new dependency — mirroring the already-established, already-proven
pattern in `ch04/sample_weights/monte_carlo.py`'s `main_mc` (Snippet
4.9): one job per iteration, not the separate `mp_pandas_obj`/molecule-
chunking path (that's designed for splitting one large pandas index,
not running N independent trials). Chapter driver's Part C call uses
`num_threads=4`, this project's documented multiprocessing sweet spot
(CLAUDE.md: "reduced fan noise/system load preferred over the marginal
extra speed from 6 [cores]"). Real-machine smoke test (20 iterations,
before the full run): 2.28x speedup, results bit-identical to
num_threads=1 -- both the wiring AND the speed benefit confirmed on the
real machine, not just in sandbox.

One necessary adaptation beyond ch04's precedent: `hrpMC`'s
`random_state` parameter takes an **int seed** (or a
`numpy.random.SeedSequence`), not a `Generator` object — `numIters`
independent child seeds are spawned up front via `SeedSequence.spawn()`
(numpy's documented pattern for reproducible parallel streams). This is
necessary, not stylistic: a single `Generator` object can't safely be
shared across worker *processes* — each worker gets an independent
pickled *copy*, so "sharing" one `Generator` would silently produce
correlated, not independent, random streams per worker. Verified in
`test_monte_carlo.py` AND on the real machine (smoke test above): given
the same base seed, `stats` output is bit-identical whether
`num_threads=1` or `num_threads>1` — parallelism only changes wall-clock
time, never the result. (`generateData`'s own `random_state` is
unaffected by this — it still takes a `Generator` directly, for
direct/sequential callers.)

**BLAS thread-capping** (`OMP`/`MKL`/`OPENBLAS`/`NUMEXPR_NUM_THREADS=1`,
set at the very top of `monte_carlo.py`, before `numpy` is imported)
mirrors this project's existing `conftest.py` pattern for heavy sklearn
estimators — without it, N worker *processes* each spawning their own
multi-threaded BLAS pool would oversubscribe the CPU and could run
*slower* than sequential despite the added parallelism.

**Import-safety note for Windows `spawn`, CONFIRMED WORKING:** `hrp.py`/
`cla.py` are loaded via explicit `__file__`-derived paths (already
needed to avoid a package/module name collision, see earlier note), but
`utils/multiprocess.py` is deliberately imported the *normal* way
(`sys.path` + `from utils.multiprocess import ...`, matching ch04/08/
09/20's already-proven pattern) rather than the same file-path-loading
trick — `process_jobs_mp`'s internal pool-dispatch function needs to be
resolvable by a genuinely importable dotted path in a freshly spawned
worker process, which a synthetic module name wouldn't be. This could
only be verified LOGICALLY in sandbox (single-core, Linux `fork`, not
Windows `spawn`) — the real-machine smoke test and full 10,000-iteration
run both confirm this works correctly under actual Windows `spawn`.

## Verification

- `hrp/test_hrp.py`: 24/24, real-machine confirmed 2026-07-31, both from
  repo root and from inside `hrp/`.
- `data_loader/test_continuous_futures.py`: 17/17, real-machine confirmed
  2026-07-31, both from repo root and from inside `data_loader/`.
- `cla/test_cla.py`: **52/52, real-machine confirmed 2026-08-10**, both
  from repo root and from inside `cla/` (Python 3.10.20, mlfinlab env).
- `monte_carlo/test_monte_carlo.py`: **16/16, real-machine confirmed
  2026-08-10**, both from repo root and from inside `monte_carlo/`.
- `chapter_16_hrp.py`: **fully real-machine confirmed 2026-08-10** --
  Parts A, B, AND C all ran successfully end-to-end, including the full
  book-scale (numIters=10000, num_threads=4) Monte Carlo, with results
  closely matching the book (see table above). Part C's multiprocessing
  wiring (the one piece sandbox couldn't verify under real Windows
  `spawn`) confirmed working via both a small smoke test and the full
  run.
- `chapter_16_hrp.ipynb`: Parts A/B — 9/9 cells executed under the real
  `mlfinlab` kernel, `language_info.version=3.10.20` confirmed via
  `run_nb.py`. **Part C added 2026-08-10 (7 new cells: intro, small
  num_threads=1 demo, real-CSV-loading + book comparison, interpretation)
  — not yet Run All'd on the real machine.**

## Outstanding / next steps

- **Run the notebook:** `chapter_16_hrp.ipynb` now has a full Part C
  section (small live demo at num_threads=1 for Jupyter/Windows
  multiprocessing safety, plus the real book-scale result loaded from
  `output/monte_carlo_stats.csv`) — added 2026-08-10, **not yet
  Run All'd on the real machine.** Needs a fresh Run All under the
  `mlfinlab` kernel via `run_nb.py` (verify
  `language_info.version=3.10.20` afterward, not just "all cells ran").
- Delete the throwaway `ch16/test_part_c_small.py` smoke-test script
  before committing (not part of the chapter deliverable).
