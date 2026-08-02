# Chapter 17 — Structural Breaks

A standard unit-root test asks one binary question about a *whole*
sample: does this series behave like a random walk, or like a
stationary/explosive process? That's the wrong question for detecting
bubbles, regime changes, or a forecasting relationship that's quietly
stopped working, because a slow build-up and its eventual reversal can
average out to look like ordinary noise over the full sample. This
chapter implements three different ways of asking a sharper question —
*when*, if ever, did the process's behavior change? — each testing a
different kind of change and needing different data.

## What's implemented

| Test | Book section | Snippet(s) | Data need | Status |
|---|---|---|---|---|
| Brown-Durbin-Evans (BDE) CUSUM | 17.3.1 | formula-only | feature/target regression pair | Implemented |
| Chu-Stinchcombe-White (CSW) CUSUM | 17.3.2 | formula-only | single price series | Implemented |
| Chow-type DF, single break (`get_dfc`) | 17.4.1 | formula-only | single price series | Implemented |
| Chow-type DF, sup-search (`get_sdfc`, Andrews 1993) | 17.4.1 | formula-only | single price series | Implemented |
| SADF | 17.4.2 | 17.1–17.4 | single price series | Implemented |

Only SADF has printed book code (Snippets 17.1–17.4). CUSUM (17.3.1,
17.3.2) and Chow-DF (17.4.1) are formula-only in the book — there's no
snippet to diff a fix against, so book-fidelity for those three means
implementing the printed math carefully and verifying it against
hand-computed or synthetically injected-break test cases, with every
non-obvious implementation choice documented explicitly rather than
silently resolved. See the "Judgment calls" section below.

## Files

- `structural_breaks/sadf.py` — Snippets 17.1–17.4, function names kept
  exactly as printed (`get_bsadf`, `getYX`, `lagDF`, `getBetas`), per this
  project's convention:
  - `lagDF(df0, lags)` — builds one lagged column per (column, lag) pair.
  - `getYX(series, constant, lags)` — builds the (y, x) arrays for the
    ADF regression `Δy_t = α + β·y_{t-1} + Σ γ_l·Δy_{t-l} + ε_t`. Accepts
    either a `pd.Series` or single-column `pd.DataFrame` of log-prices.
  - `getBetas(y, x)` — OLS via the normal equations; returns both the
    point estimate and the full covariance matrix.
  - `get_bsadf(logP, minSL, constant, lags)` — SADF's **inner** loop: for
    one fixed end point, fits the ADF regression across every valid
    backward-expanding start point and returns the supremum t-statistic.
  - `get_sadf(logP, minSL, constant, lags)` — SADF's **outer** loop
    (book's own "not shown here"): repeats `get_bsadf` for every advancing
    end point, producing the full SADF time series from Figure 17.1. This
    project's own driver code, not a fidelity-checked port.
- `structural_breaks/chow_df.py` — Sec 17.4.1, no printed code:
  - `get_dfc(logP, tau_star)` — Chow-type DF statistic at a single assumed
    break fraction.
  - `get_sdfc(logP, tau0=0.15, step=None)` — Andrews' [1993] sup-search
    over `tau* ∈ [tau0, 1-tau0]`. Reuses `sadf.getBetas` rather than
    duplicating OLS.
- `structural_breaks/cusum.py` — Sec 17.3, no printed code:
  - `get_bde_recursive_residuals(X, y, min_sample)` — expanding-window OLS
    with genuine one-step-ahead (out-of-sample) recursive residuals.
  - `get_bde_cusum(X, y, min_sample, index=None)` — full BDE CUSUM
    statistic `S_t`, plus a conventional ±1.96·√(t−k−1) 95% band derived
    from the book's own stated null distribution (flagged in the
    docstring as a convenience, not separately printed book content).
  - `get_csw_stat`, `get_csw_critical_value`, `get_csw_sup`, and the outer
    loop `get_csw_cusum(logP, min_sample=3)` — the CSW test comparing the
    log-price level at `t` against every earlier reference point `n`,
    taking the supremum departure and comparing it to the book's own
    time-dependent critical value `c_α[n,t] = √(b_α + log(t−n))`.
- `structural_breaks/test_sadf.py` (19 tests), `test_chow_df.py` (9
  tests), `test_cusum.py` (17 tests) — 45 total.
- `structural_breaks/conftest.py` — BLAS thread cap (mirrors
  Ch08/09/12/13/19), needed here too since `get_bsadf`/`get_sadf` run many
  small `np.linalg.inv` calls in tight loops.
- `chapter_17_structural_breaks.py` / `.ipynb` (at `ch17/` root, per the
  Ch19-onward convention) — three-part demo on real data:
  - **Part A** — BDE CUSUM on Ch19's real 12-feature enriched table
    predicting Ch3's real triple-barrier returns.
  - **Part B** — CSW CUSUM, Chow-DF, and SADF on gold's real continuous
    price series (reuses `ch16/data_loader/continuous_futures.py`).
    These explosiveness tests are built to catch slow-building,
    multi-period bubbles, so they need genuine duration — gold's
    ~6.7-year daily series is the right structural match.
  - **Part C** — Chow-DF and SADF re-run on BTC's real ~29-day dollar-bar
    window, as a secondary contrast: does a short window give these
    tests enough to work with at all?

## Judgment calls (formula-only sections, no printed code to check against)

- **`getYX`'s Series/DataFrame handling** (`sadf.py`): the book's
  docstring calls the input "a pandas series," but the printed indexing
  (`series.values[..., 0]`) only works on a single-column DataFrame — a
  bare `pd.Series` crashes with an opaque `AttributeError` (confirmed by
  direct test). Auto-converts a Series to a single-column DataFrame so
  the function's real, tested contract matches its docstring.
- **`getYX`'s return type** (`sadf.py`): as printed, `x` stays a
  `pd.DataFrame` when `constant='nc'` but becomes an `ndarray` for
  `'ct'`/`'ctt'` — an inconsistent type depending on an argument value
  (confirmed by direct test: `x[:, 0]` raises `InvalidIndexError` only
  for `'nc'`). Normalized to always return `ndarray`.
- **`get_bsadf`'s `bsadf=None` initialization** (`sadf.py`): legal under
  Python 2's looser comparison semantics, but `float > None` raises
  `TypeError` immediately under Python 3 (confirmed by direct test, same
  category as the Ch5 tuple-assignment bug). Fixed to `bsadf = -np.inf`,
  which preserves the original algorithm exactly.
- **BDE CUSUM's indexing** (`cusum.py`): the book's own prose is
  internally loose here — it says the first RLS estimate is
  `beta_hat_{k+1}`, fit on subsample `[1,k+1]`, but also says `S_t` sums
  `omega_j` starting at `j=k+1`, which would need `beta_hat_k` — an
  estimate not in the book's own listed set of `T-k` estimates. Adopted
  a self-consistent, testable convention instead of silently picking an
  interpretation: `min_sample` is the size of the first fit window; the
  first recursive residual is computed by forecasting the point right
  after that window, using only data through the window's end. Documented
  explicitly in the docstring rather than resolved silently.
- **BDE CUSUM's `S_t` sums raw (non-demeaned) residuals**: confirmed by
  direct comparison against the book's own formula that this is genuinely
  what's printed — only the *standard deviation* denominator uses a
  demeaned variance, not the cumulative sum itself. Not "fixed" toward a
  fully-demeaned running sum, since that would depart from the book's
  actual stated formula, not correct a bug in it.

## Real-data results (real-machine confirmed, 2026-08-01)

**Part A — BDE CUSUM**, Ch19's enriched features (standardized)
predicting Ch3's real returns, 87 real aligned events, 69 recursive
residuals (`min_sample=18`, 13 regressors incl. constant):

- Final `S`: −5.263, band at that point: ±16.163
- `S` **does** cross the 95% band for 23 of 68 points, from
  **2026-03-12 to 2026-03-20**, then reverts inside the band by month's
  end.
- Framed alongside this pipeline's broader "no stable exploitable
  signal" finding (Ch11 PBO, Ch12 CPCV, Ch13 O-U, Ch14 DSR, Ch15 P[fail])
  rather than as contradicting it — a temporary, reverting drift in an
  already-weak relationship isn't a newly tradeable signal.

**Part B — gold, 1996-01-03 to 2002-10-01, 1,760 real daily bars:**

| Test | Result |
|---|---|
| SDFC (Chow-DF, sup-search) | 0.173 at τ\*=0.0511 (1996-05-08) |
| Max SADF | 3.599 at 1997-02-11 |
| Max CSW `S` | 13.127 at **1999-09-28** (reference point 1999-09-24, critical value 2.301) |

The CSW CUSUM signal lands almost exactly on the **Washington Agreement
on Gold (Sept 26, 1999)** — a real, documented event, independently
flagged during Ch16's own data-hygiene pass as gold's largest real
historical spike in this window. Genuine cross-chapter corroboration:
the test detects a real, known event using only 4 days of reference
window, not a data artifact.

**Part C — BTC secondary contrast**, 239 real dollar bars, ~29-day
window: max SADF = 2.518, SDFC = 0.321 (vs. gold's 3.599 / 0.173). Not
directly comparable in absolute terms (different assets, different vol
regimes) — the point is qualitative: BTC's numbers are estimated from
under a month of history vs. gold's ~6.7 years, so they carry the same
"heavy extrapolation from a short real window" caveat this project has
flagged before (Ch13's O-U calibration, Ch15's annualized bet frequency).
A real result, just a thinly-supported one.

## Runtime note

SADF and CSW CUSUM's own sup-search are both genuinely **O(T²)** (book's
own Sec 17.4.2.2 complexity warning). On gold's 1,760-bar real series:
SADF ≈ 70s, CSW CUSUM ≈ 105s on the real machine — not a hang, expected.
Chow-DF and Part A/C are all well under a few seconds.

## Verification

- `structural_breaks/test_sadf.py`: 19/19
- `structural_breaks/test_chow_df.py`: 9/9
- `structural_breaks/test_cusum.py`: 17/17
- **45/45 total**, real-machine confirmed 2026-07-31, two-pass (repo
  root and from inside `structural_breaks/`). One expected
  `RuntimeWarning` (`invalid value encountered in double_scalars`) from a
  deliberately-tested degenerate near-zero-variance case — not a bug.
- `chapter_17_structural_breaks.py` / `.ipynb`: real-machine confirmed
  2026-08-01. Notebook: 9/9 cells, `kernelspec.name=mlfinlab`,
  `language_info.version=3.10.20` confirmed via `run_nb.py`. All three
  headline real numbers (13.127, 3.599, 0.173) confirmed present in the
  executed notebook, not just "all cells ran."

## Outstanding

None — chapter complete.
