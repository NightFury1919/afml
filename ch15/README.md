# Chapter 15 — Understanding Strategy Risk

Models a strategy's bet outcomes as a binomial process (profit-taking vs.
stop-loss) and asks a sharper question than portfolio risk: given a
precision (win rate) and betting frequency, what's the probability that
**the strategy itself** fails to sustain a target Sharpe ratio? Strategy
risk is not portfolio risk — a strategy can hold low-volatility positions
and still be very likely to underperform its target if the edge is thin.

## What's implemented

| Topic | Book section | Snippet | Status |
|---|---|---|---|
| Symmetric payouts (theta[p,n], implied precision) | 15.2 | 15.1 (Monte Carlo verification) | Implemented |
| Asymmetric payouts (binSR, binHR, binFreq) | 15.3 | 15.2 (SymPy, not ported — see below), 15.3, 15.4 | Implemented |
| Probability of strategy failure (mixGaussians, probFailure) | 15.4, 15.4.1 | 15.5 | Implemented |

Snippet 15.2 (SymPy symbolic factoring of the variance) is a one-off
console demonstration, not a reusable function — not ported as a module
function; the closed-form results it derives are the ones implemented in
`asymmetric.py`.

## Files

- `strategy_risk/symmetric.py` — Sec 15.2: `sharpe_ratio_symmetric`,
  `implied_precision_symmetric`, `simulate_symmetric_sharpe` (Snippet 15.1,
  ported to Python 3 / a seeded `numpy.random.Generator`).
- `strategy_risk/asymmetric.py` — Sec 15.3: `binHR`, `binFreq`, `binSR`,
  kept with the book's own function names. File order matches the book's
  own printed order (Snippet 15.3, then Snippet 15.4).
- `strategy_risk/algorithm.py` — Sec 15.4/15.4.1: `mixGaussians`,
  `probFailure` (Snippet 15.5), including the confirmed `norm.cdf` fix
  (see below).
- `strategy_risk/test_symmetric.py`, `test_asymmetric.py`,
  `test_algorithm.py` — 34-test TDD suite, split by topic to match the
  implementation. Every test uses a known value: the book's own worked
  examples (theta=1.173, implied precision .72, p_theta*=0=2/3, p=0.6336
  for weekly bets, n=396 for Sharpe=2 at p=.55), hand-traced algebra, or a
  seeded-Generator regression pin.
- `strategy_risk/conftest.py` — BLAS thread cap, matches Ch08/09/12/13/19.
- `chapter_15_strategy_risk.py` / `.ipynb` (at `ch15/` root, per the
  Ch19-onward convention) — three-part demo: symmetric payouts, asymmetric
  payouts, and a real-data strategy-risk analysis on Ch3's actual 88
  BTC/TUSD triple-barrier bet outcomes.

## Book-snippet fidelity notes

- **Snippet 15.1** used Python 2's `xrange`/`print` statement and the
  legacy global `np.random` state. Ported to Python 3, vectorized with
  `numpy.random.Generator`, per this project's random_state convention
  (a shared Generator threaded through calls, not reset per iteration).
- **Snippet 15.5**'s `mixGaussians` likewise used the legacy global
  `np.random` state — ported to `numpy.random.Generator`.
- Two pure paste/OCR artifacts caught and resolved against the screenshots
  (not real book issues): Snippet 15.3's `b**2-4*a*c` came through with an
  en-dash instead of a minus sign in a first text paste; Snippet 15.4's
  `binFreq`/`binSR` came through with `binSR`'s return line reordered
  above its own `def` line (a page-column OCR glitch). Both resolved using
  the actual page screenshots as the canonical source.

## Genuine judgment calls / fixes (read before trusting these functions blindly)

1. **Confirmed real book bug in Snippet 15.5, fixed (Ethan sign-off,
   2026-07-23).** The printed line
   `risk=ss.norm.cdf(thresP,p,p*(1-p))` passes `p*(1-p)` — a **variance**
   (Sec 15.4.1's own approximation is `f[p] ~ N[p_bar, p_bar(1-p_bar)]`,
   and `p_bar(1-p_bar)` is literally the Bernoulli variance formula) — as
   `scipy.stats.norm.cdf`'s `scale` argument, which scipy documents as a
   **standard deviation**. Fixed to pass `sqrt(p*(1-p))`. Same category as
   Ch5's tuple-assignment bug and Ch9's bagging-tuple-order bug — printed
   AFML code isn't assumed bug-free. `test_algorithm.py` has a dedicated
   test proving the fix actually changes the numeric result on identical
   seeded data (0.4871 fixed vs. 0.4728 as literally printed), so this
   isn't a cosmetic rewrite.
2. **`binHR`'s negative-discriminant guard, added.** The book's own
   `p=(-b+sqrt(b**2-4*a*c))/(2*a)` takes only the "+" root. If the
   discriminant is negative, Python's `(-1)**.5` silently returns a
   **complex number** rather than raising — the same category of
   language-semantics trap as the Ch5 tuple-assignment gotcha in
   `CLAUDE.md`. Added an explicit `ValueError`. Verified symbolically
   (via SymPy): `disc = tSR^2*(pt-sl)^2*[tSR^2*(pt-sl)^2 - 4*freq*pt*sl]`
   — for the book's normal usage (`sl<0<pt`) the bracket can never go
   negative, so this guard is inert on every worked example and on this
   chapter's real-data usage; it only fires in the pathological case where
   `sl` and `pt` share a sign.
3. **`binFreq`'s "extraneous solution" caveat, explained not just
   flagged.** Found while writing tests: squaring the equation to isolate
   `freq` loses the sign of expected profit. Below the break-even
   precision (`p = -sl/(pt-sl)`, the same `p_theta*=0` special case from
   `binHR`), a positive `tSR` request has no valid solution, and `binFreq`
   correctly returns `None` — this is the book's own documented caveat
   working as intended, not a bug. `test_asymmetric.py` has dedicated
   tests for both the extraneous case and the valid case just above the
   threshold.

## Real-data results (genuine, from `chapter_15_strategy_risk.py`, sandbox run — see reproducibility note)

Applied `probFailure` to Ch3's actual 88 real BTC/TUSD triple-barrier bet
outcomes (`input_data/ch03_events.csv`'s `ret` column — real realized
returns from real trades, not a synthetic mixture):

- Realized precision `p_bar = 0.5568` (49 of 88 bets positive), mean
  winning return 0.0337, mean losing return -0.0325.
- Annualized frequency `n = T/y = 1101.8` bets/year — extrapolated from
  only ~29 real days of trade data (Ch2's single-month BTC/TUSD tape), so
  this multiplies the apparent bet rate by ~12.6x. Flagged clearly in both
  the script's output and the notebook — treat "annualized" figures as a
  genuine but heavily-extrapolated real-data result, not a full-year track
  record.
- Realized (empirical) annualized Sharpe on this series: **4.12** — looks
  striking, but is inflated by the same short-window extrapolation
  (`sqrt(freq)` scales it up ~3.5x vs. a true full-year sample).
- **`P[fail]` at tSR=0.5, 1.0, 2.0: 0.4527, 0.4587, 0.4707** — at every
  target Sharpe tested, far above the book's own "disregard if >.05" rule
  of thumb.
- **This corroborates, via an independent method, the same "no reliable
  exploitable signal in this feature set/model" finding already
  established elsewhere in this pipeline**: Ch11's PBO (~0.83), Ch12's
  CPCV (all 5 paths negative), Ch13's non-stationary O-U calibration, and
  Ch14's DSR (0/5 paths survive). Strategy risk here isn't a quirk of this
  chapter's method — it's consistent with everything else this pipeline
  has found on this real data.

## Outstanding / next steps

- ~~Real-machine pytest confirmation~~ **Done (2026-07-24).** All 34 tests
  pass under the real `mlfinlab` env (Python 3.10.20, pytest 9.0.3) —
  identical pass count and warnings to the sandbox run, nothing here was
  environment-sensitive. TDD comment blocks in `symmetric.py`,
  `asymmetric.py`, `algorithm.py`, and `chapter_15_strategy_risk.py`
  updated from "SANDBOX CONFIRMED" to "REAL-MACHINE CONFIRMED" with the
  real pytest output embedded.
- ~~Notebook real-machine confirmation~~ **Done (2026-07-24).**
  `chapter_15_strategy_risk.ipynb` genuinely re-run under the `mlfinlab`
  kernel and saved (`kernelspec.name=mlfinlab`, `language_info.version=
  3.10.20`, both cross-checked, not just kernelspec taken at face value).
  Every real-data number — including the seeded Monte Carlo cross-check
  (seed=2026), which only matches if the actual computation ran — is
  byte-identical to the sandbox run. No errors, both figures rendered.
- **Chapter 15 is complete.** Implementation, tests, chapter script,
  notebook, and README all real-machine confirmed.
