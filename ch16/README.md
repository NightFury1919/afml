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
| Monte Carlo out-of-sample comparison (HRP vs CLA vs IVP) | 16.5 / 16.6 | 16.5 | **NOT implemented — blocked, see below** |

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
  commodity futures; Part C, a status note explaining why Section
  16.5/16.6's Monte Carlo comparison is deliberately not built yet.

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

## Section 16.5/16.6 Monte Carlo comparison — deliberately not built yet

The book's own methodology compares HRP against **both** IVP and CLA
out-of-sample via Monte Carlo simulation. CLA isn't printed in AFML itself
— the book cites Bailey & López de Prado's separate 2013 paper. Per this
project's book-fidelity convention, extended to external referenced
material: the actual reference implementation needs to come from Ethan,
not be reconstructed from memory or a web search. Building a two-way (HRP
vs IVP) Monte Carlo now would be premature — the book's own point is the
three-way comparison, and CLA is exactly the method HRP is positioned
against.

**Next steps, once CLA is available:**
1. Implement CLA faithfully from the real reference source.
2. Decide how to adapt Section 16.5's synthetic Monte Carlo methodology to
   real data — single historical run vs. block-bootstrap on the real
   6-asset returns (not yet decided).
3. Build the 3-way out-of-sample comparison.

## Verification

- `hrp/test_hrp.py`: 24/24, real-machine confirmed 2026-07-31, both from
  repo root and from inside `hrp/`.
- `data_loader/test_continuous_futures.py`: 17/17, real-machine confirmed
  2026-07-31, both from repo root and from inside `data_loader/`.
- `chapter_16_hrp.py`: real-machine run confirmed 2026-07-31 — Part A
  bit-for-bit identical to sandbox, Part B produces the real-data table
  above.
- `chapter_16_hrp.ipynb`: 9/9 cells executed under the real `mlfinlab`
  kernel, `language_info.version=3.10.20` confirmed via `run_nb.py`.

## Outstanding / next steps

- **CLA reference implementation** — still needed from Ethan before
  Section 16.6's Monte Carlo comparison can be built.
- Once CLA lands: the 3-way out-of-sample comparison described above.
