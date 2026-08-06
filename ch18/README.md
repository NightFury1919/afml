# Chapter 18 — Entropy Features

Price series convey information about supply and demand. This chapter estimates how
much information a price series actually carries, using tools from Shannon's
information theory, and turns that estimate into features an ML algorithm can learn
from.

## What's implemented

| Topic | Book section | Snippet(s) | Status |
|---|---|---|---|
| Plug-in (ML) entropy estimator (`plugIn`, `pmf1`) | 18.3 | 18.1 | Implemented |
| Lempel-Ziv dictionary (`lempelZiv_lib`) | 18.4 | 18.2 | Implemented |
| Longest-match length (`matchLength`) | 18.4 | 18.3 | Implemented |
| Kontoyiannis' LZ entropy estimate (`konto`) | 18.4 | 18.4 | Implemented |
| Encoding schemes: binary, quantile, sigma | 18.5 | formula-only | Implemented |
| Adverse-selection feature (order flow -> entropy -> CDF) | 18.8.4 | formula-only, workflow-only | Implemented |
| Portfolio concentration (Meucci) | 18.8.3 | formula-only | **Implemented 2026-08-06** — see below |
| Shannon entropy, redundancy, mutual information | 18.2 | formula-only | Conceptual only (see below) |
| Entropy of a Gaussian process | 18.6 | formula-only | Conceptual only |
| Generalized mean / effective number | 18.7 | formula-only | Conceptual only |
| Market efficiency, maximum entropy generation | 18.8.1, 18.8.2 | prose only, no formula | Discussed in notebook only |
| PIN (probability of informed trading) | 18.8.4 | formula-only | Not implemented (see Ch19's own PIN skip) |

**Scope decision (confirmed with Ethan, 2026-08-04):** sections that don't produce a
feature this project's single-asset BTC/TUSD pipeline actually consumes are covered as
discussion in the notebook, not implemented as tested code. Per this project's standing
rule, these are flagged as **revisitable, not permanently closed** — portfolio
concentration (18.8.3) was exactly this kind of item until Chapter 16 supplied its
missing prerequisite (a real multi-asset covariance matrix + real allocation vectors),
at which point it was implemented (2026-08-06, see below). 18.2/18.6/18.7 remain
conceptual for the same reason: they don't yet have a concrete real-data use in this
pipeline, not because they can't ever get one.

## Files

- `entropy_features/entropy_estimators.py` — Snippets 18.1–18.4, function names kept
  exactly as printed (`plugIn`, `pmf1`, `lempelZiv_lib`, `matchLength`, `konto`).
- `entropy_features/test_entropy_estimators.py` — 17 tests, hand-traced known values
  (see docstrings for full manual derivations).
- `entropy_features/encoding_schemes.py` — Sec 18.5, no printed snippet:
  `binary_encode`, `quantile_encode`, `sigma_encode`. This is the glue that turns
  continuous real returns into the discrete strings the estimators actually consume.
- `entropy_features/test_encoding_schemes.py` — 16 tests, hand-computed known values.
- `entropy_features/adverse_selection.py` — Sec 18.8.4, workflow-only (six prose
  steps, no code, no algorithm for the rolling-entropy mechanics — see "Judgment
  calls" below): `order_flow_imbalance`, `adverse_selection_feature`.
- `entropy_features/test_adverse_selection.py` — 12 tests, cross-checked against the
  already hand-traced `konto` estimator.
- `entropy_features/portfolio_concentration.py` — Sec 18.8.3, formula-only (four-step
  derivation, no printed snippet): `eigen_decomposition`, `factor_loadings`,
  `risk_contribution`, `portfolio_concentration`, and a `compute_portfolio_concentration`
  wrapper chaining all four. Takes a generic covariance matrix `V` and allocation
  vector `omega` — no dependency on Ch16's code; the chapter driver script is what
  wires it to Ch16's real output (see Part C below).
- `entropy_features/test_portfolio_concentration.py` — 21 tests, hand-traced known
  values (diagonal covariance matrices with known eigenvalues), plus property checks
  (theta sums to 1, eigenvector sign-invariance, asset-order permutation-invariance).
- `chapter_18_entropy_features.py` / `.ipynb` (at `ch18/` root, per the Ch19-onward
  convention) — three-part demo on real data:
  - **Part A** — all three encoding schemes + both estimators, run on the real
    248 BTC/TUSD bar-to-bar returns (same 249 real $10,000 dollar bars used
    throughout this pipeline).
  - **Part B** — the 18.8.4 adverse-selection workflow, run on Ch19's real
    `BuyVolume`/`SellVolume` per bar.
  - **Part C** — the 18.8.3 portfolio concentration formula, run against Chapter 16's
    real 6-commodity covariance matrix and HRP/IVP allocation vectors (imports
    `part_b_real_data` directly from `chapter_16_hrp.py` rather than duplicating
    Ch16's data-loading/HRP logic).

## Book-snippet fidelity notes (Py2→3 fixes, all in `entropy_estimators.py`)

- `xrange` → `range` in all four functions (Py2→3 syntax removal).
- **Real bug in `konto`, not just syntax:** the book's `len(msg)/2` is *true* division
  in Python 3 — a float, which breaks `range()`/`min()` against an int window. Fixed
  to `len(msg)//2` (floor division), same bug category as Ch16's `getRecBipart`.
- The `print konto(...)` statement at the bottom of Snippet 18.4 (Py2 print-statement
  syntax) was `__main__` demo code, not a function — omitted from the module; the same
  worked examples appear in the notebook instead.

## Real bug caught wiring Part C to Chapter 16 (not a book bug — a cross-chapter data bug)

`chapter_16_hrp.py`'s `part_b_real_data()` returns `hrp`/`ivp` sorted
**alphabetically** by asset name (its own `.sort_index()` calls), but `cov =
ret_df.cov()` keeps the `COMMODITIES` dict's **insertion order** (gold, crude_oil,
corn, live_hogs, tbonds, gbp — not alphabetical). `portfolio_concentration.py`'s
functions are plain-numpy / position-based, not label-aware, so naively pairing
`hrp.values` or `ivp.values` against `cov.values` would silently pair each weight
with the *wrong* asset's variance/covariance row. Verified this was a real,
material-magnitude risk (not rounding-level) with a synthetic mismatched-order
reproduction before touching real data. **Fix:** Part C explicitly does
`hrp.reindex(cov.index)` / `ivp.reindex(cov.index)`, with `assert`s guarding against
silent `NaN`s from a name mismatch, before calling
`compute_portfolio_concentration`.

## Judgment calls (formula-only / workflow-only sections)

- **Quantile encoding's train/test split (18.5.2):** the book describes fitting
  quantile boundaries on an in-sample period and applying them out-of-sample. Our
  real dataset (~87–88 events, 248 returns) is too small to split meaningfully — same
  tension as Ch13's O-U calibration. **Decision:** fit and apply boundaries on the
  full sample, documented as an in-sample limitation, not silently glossed over.
- **Sigma encoding's step size (18.5.3):** the book leaves `sigma`'s choice to the
  practitioner. Default: `std(returns)/4`, a data-driven default, not a book-specified
  value.
- **18.8.4's rolling-entropy mechanics:** the book's own step 6 asks for "the time
  series `{F[H[Xtau]]}`" but gives no algorithm for turning ONE entropy estimate into
  a time series. This project's own construction: quantize the full order-flow-
  imbalance series once, then run `konto` on a trailing `roll_window`-bar slice of the
  quantized message ending at each `tau`, producing a genuine rolling entropy series;
  take its empirical CDF (percentile rank) as the final feature. Documented explicitly
  in `adverse_selection.py`'s own docstring, not silently resolved.
  - Defaults: `n_quantiles=5`, `roll_window=30` (must be even — `konto`'s expanding-
    window mode requires `len(msg)%2==0`). With ~249 real bars this leaves ~220
    output points.
- **18.8.3's natural-log convention:** the book writes `H = 1 - (1/N)*e^{-sum(theta_i
  * log[theta_i])}`. "log" paired with `e^{...}` is natural log (ln) throughout this
  section and 18.7's generalized-mean special case, unlike 18.1–18.4's entropy-rate
  estimators, which are explicitly `log2` (bits). `portfolio_concentration.py` uses
  `np.log` (natural) accordingly — documented explicitly in the module docstring
  since the book's own "log" notation is ambiguous without that cross-reference.
- **18.8.3's `0 * log(0)` convention:** a component with zero risk contribution
  (`theta_i = 0`) uses the standard entropy limit `0 * log(0) := 0`, so degenerate
  (fully-concentrated) allocations don't raise or produce `NaN`.

## Real-data results (real-machine confirmed 2026-08-06)

**Part A — encoding schemes + estimators, 248 real BTC/TUSD bar-to-bar returns:**

| Encoding | Alphabet size | `plugIn` h (w=1) | Theoretical max | `konto` h | `konto` r |
|---|---|---|---|---|---|
| Binary | 2 | 0.9997 | 1.0000 | 0.8644 | 0.8877 |
| Quantile (n=5) | 5 | 2.3217 | 2.3219 | 1.6928 | 0.7872 |
| Sigma (step=std/4) | 23 | 3.7819 | 4.5236 | 2.4093 | 0.6971 |

**Real finding:** binary-encoded entropy (0.9997 bits) sits almost exactly at the
theoretical max of 1 bit — BTC's real up/down sequence reads as close to
indistinguishable from a coin flip. This **corroborates Ch13's `phi_hat≈1.03`
random-walk finding via a completely independent method** (information theory rather
than an O-U mean-reversion fit).

**Caveat** (matches the book's own Sec 18.5.2 warning): quantile encoding's entropy
reading (2.322 bits) sits almost exactly at `log2(5)=2.3219`, the theoretical MAX for
a 5-letter alphabet — a property of quantile encoding itself (it forces near-uniform
bin counts by construction), not independent evidence of high information content.
Binary and sigma encoding, which don't force uniformity, are the more trustworthy
readings here.

**Part B — 18.8.4 adverse-selection feature, real order flow, 249 real bars:**

220 real per-bar values, `roll_window=30`. Mean 0.5023, std 0.2893, range
[0.0045, 1.0000] — the feature swings across its full range with visible cyclical
structure over the real ~29-day window, not flat noise.

With only 220 real output points from a 249-bar/~29-day window, this feature carries
the same "heavy extrapolation from a short real window" caveat this project has
flagged before (Ch13's O-U calibration, Ch15's annualized bet frequency, Ch17 Part
C's BTC explosiveness contrast) — a real result, just a thinly-supported one.

**Part C — 18.8.3 portfolio concentration, Ch16's real 6-commodity covariance matrix
and HRP/IVP weights (1996–2002, 1749 aligned trading days):**

| Allocation | Portfolio concentration `H` | Top principal-component `theta` |
|---|---|---|
| HRP | 0.2821 | `crude_oil` 0.410 |
| IVP | 0.2695 | `crude_oil` 0.349 |

`H` is bounded in `[0, 1 - 1/N]` for `N=6` assets, i.e. `[0, 0.8333]` here — 0 means
risk spread perfectly evenly across principal components (maximally diversified); the
upper bound means all risk concentrated in a single principal component.

**Real finding:** HRP is slightly *more* risk-concentrated across principal
components (`H=0.2821`) than IVP (`H=0.2695`), even though HRP's top-2
*asset-weight* concentration is slightly *lower* than IVP's (0.756 vs 0.752, Ch16's
own metric). These are two different notions of concentration — asset-weight share
vs. how risk distributes across the covariance matrix's principal components. HRP
shifted weight toward the low-vol, low-correlation `gbp` and `tbonds`, but
`crude_oil`/`gold`/`corn` still dominate the largest eigenvalue direction, and HRP's
version of that dominance is marginally *more* skewed there (`crude_oil` theta=0.410
vs IVP's 0.349) than IVP's. Consistent with Ch16's own finding that HRP and IVP
converge for this genuinely low-correlation 6-asset set (correlations mostly under
0.15 in absolute value) — there's little cluster structure here for HRP's
correlation-awareness to actually exploit.

## Verification

- `entropy_features/test_entropy_estimators.py`: 17/17, real-machine confirmed
  2026-08-04, two-pass (repo root and from inside `entropy_features/`).
- `entropy_features/test_encoding_schemes.py`: 16/16, real-machine confirmed
  2026-08-04, two-pass.
- `entropy_features/test_adverse_selection.py`: 12/12, real-machine confirmed
  2026-08-04, two-pass.
- `entropy_features/test_portfolio_concentration.py`: 21/21, real-machine confirmed
  2026-08-06, two-pass (repo root and from inside `entropy_features/`).
- **66/66 total**, real-machine confirmed under `mlfinlab` (Python 3.10.20).
- `chapter_18_entropy_features.py` / `.ipynb`: real-machine confirmed 2026-08-06 —
  driver script run end-to-end (all pinned numbers, including Part C, reproduced
  exactly), notebook executed 7/7 cells clean under the `mlfinlab` kernel via
  `run_nb.py`.

## Outstanding

- 18.2, 18.6, 18.7: conceptual coverage only, flagged as revisitable per the standing
  project rule — not permanently closed. (18.8.3 was in this bucket too until
  2026-08-06, when Ch16 supplied its missing prerequisite.)
- None — chapter otherwise complete.
