# Chapter 19 — Microstructural Features

Implements 9 of AFML Chapter 19's features, scoped to what's actually
derivable from a raw trade tape + dollar bars (no order book, no
cancellations, no options data). Built out of turn (ahead of Ch14) as a
deliberate feature-enrichment effort: three earlier chapters (Ch08/09's
degenerate feature importance and weak CV scores, Ch12's near-zero CPCV
Sharpe distribution, and Ch13's non-stationary O-U calibration) all
independently pointed at the same root cause — the real training table has
essentially one feature (`fracdiff`). This chapter exists to give it more.

## What's implemented, and what's deliberately skipped

| Feature | Book section | Status |
|---|---|---|
| Tick rule + accuracy vs. true side | 19.3.1 | Implemented |
| Roll model (`c`, `sigma_u`) | 19.3.2 | Implemented |
| Parkinson high-low volatility | 19.3.3 | Implemented |
| Corwin-Schultz spread + Becker-Parkinson sigma | 19.3.4, Snippets 19.1/19.2 | Implemented |
| Kyle's Lambda | 19.4.1 | Implemented |
| Amihud's Lambda | 19.4.2 | Implemented |
| Hasbrouck's Lambda | 19.4.3 | **Skipped** — book requires a Gibbs sampler for Bayesian estimation, no snippet given; an OLS stand-in would be a real departure from the book's method, not a port |
| PIN | 19.5.1 | **Skipped** — MLE fit of a 3-component Poisson mixture, no snippet; VPIN is the book's own easier-to-estimate high-frequency version of the same idea |
| VPIN | 19.5.2 | Implemented |
| Round-number order-size frequency | 19.6.1 | Implemented, **adapted** (see below) |
| Cancellation rates / TWAP detection / options features | 19.6.2–19.6.4 | **Skipped** — need order-book messages / options quotes we don't have |
| Serial correlation of signed order flow | 19.6.5 | Implemented |

## Files

- `microstructural_features.py` — all 9 features, single flat module (Ethan's
  choice, matching Ch11–13's single-file convention). Each feature has a
  plain-English "why" + the book's own math in a comment block before the code.
- `test_microstructural_features.py` — 41-test TDD suite. Every test uses a
  known value: either hand-traced by working the book's formula on a tiny
  example, or cross-validated against an independent reference (numpy's
  `polyfit` for Kyle's Lambda, a from-scratch re-derivation of Snippet 19.1
  for Corwin-Schultz, pandas' own `.autocorr()`, closed-form OLS-through-
  origin algebra for Amihud's Lambda).
- `chapter_19_microstructural_features.py` / `.ipynb` (at `ch19/` root, per
  the updated hybrid folder convention) — three-part real-data demo:
  - Part A: rebuilds the real $10,000 dollar bars from raw trades, tagging
    every trade with its bar id.
  - Part B: computes each feature once over the whole real series, reports
    genuine headline numbers.
  - Part C: builds a full 249-bar feature table and saves it to
    `input_data/ch19_microstructural_features.{csv,pkl}`.
- `conftest.py` — BLAS thread cap, matches Ch08/09/12/13.

## Book-snippet fidelity notes

- **Snippet 19.1/19.2 modernization:** the book's printed code uses a
  pre-2016 pandas API (`pd.stats.moments.rolling_sum` etc.) that no longer
  exists. Ported 1:1 in spirit to `.rolling().sum()/.mean()/.max()/.min()` —
  same math, same window semantics. Same category of legitimate
  modernization as Ch09's `iid=` removal.
- Kyle's Lambda, Amihud's Lambda, and VPIN are specified as equations, not
  printed code, in the book — implemented directly from those equations
  (Secs 19.4.1, 19.4.2, 19.5.2), same bar as every other chapter's
  book-fidelity rule.

## Genuine judgment calls / adaptations (read before trusting these features blindly)

1. **Round-number frequency (19.6.1) is adapted, not ported.** The book's
   finding is about *discrete* equity contract counts (size 10 vs. size 9).
   BTC quantities are continuous, so this module instead checks whether a
   trade's volume lands near a set of plausible "round" levels
   (0.001, 0.01, 0.1, 1.0 BTC, etc.). **Real-data finding:** the most
   frequently matched level (0.0001 BTC) is plausibly just Binance's minimum
   order-size increment (lot-size grid), not evidence of human "round-number"
   psychology — a different phenomenon from what Easley et al. [2016]
   describe for GUI traders. Flagged in the demo script's own output, not
   silently presented as a clean reproduction.
2. **Roll's model applied to bar closes, not trade-to-trade prices.** Roll's
   model was derived for tick-to-tick bounce; dollar bars aren't
   uniformly-spaced ticks. Kept anyway because bar closes are the only price
   series consistent with the rest of the pipeline (fracdiff, triple-barrier
   events, etc. are all bar-indexed) — but it's a real adaptation, not a
   literal application of the book's assumptions.
3. **Kyle's Lambda per bar produces some negative estimates** (real result:
   see below) — the book's model requires lambda>0 as a second-order
   condition, so a negative fit isn't "wrong data," it's the regression
   failing to identify a stable slope on a small number of trades per bar
   (median ~37). Kept as-is (reported, not clipped or discarded) since
   silently forcing positivity would hide a real small-sample limitation.

## Real-data results (genuine, from `chapter_19_microstructural_features.py`, sandbox run — see reproducibility note)

- **Tick rule accuracy**: 66.2% against Binance's true `IsBuyerMaker` side —
  notably lower than the >85% typically cited for equities. Plausibly BTC's
  fine price granularity and high rate of same-price consecutive trades
  (which the tick rule can only resolve by carrying forward the previous
  sign, compounding errors).
- **Roll model**: effective half-spread `c ≈ $172.60`, `sigma_u ≈ $625.87`
  on ~$67k BTC bar closes.
- **Corwin-Schultz**: mean spread 0.26% of price, but 57.7% of bars have
  `alpha < 0` and get clipped to a 0 spread estimate — the model finds no
  measurable spread over half the time on this data.
- **Kyle's Lambda**: 233 of 249 bars had enough trades (>=5) to fit; ~mean
  1467, but a wide range including negative values (see judgment call #3
  above).
- **VPIN** (10-bar window): mean ≈ 0.53 — elevated relative to typical
  published VPIN values (~0.1–0.3), plausibly a small-window/small-bar-count
  artifact rather than a genuine informed-trading signal.
- **Serial correlation of true signed order flow**: lag-1 = 0.45, lag-5 =
  0.21 — positive and decaying, consistent with the book's own discussion
  of order-splitting over short horizons (Toth et al. [2011]).

## Outstanding / next steps

- ~~Real-machine confirmation~~ **Done (2026-07-16).** All 41 tests pass
  under the real `mlfinlab` env (Python 3.10.20, pytest 9.0.3, pandas 1.5.3,
  numpy 1.23.5, 0.88s). Real-data headline numbers are byte-identical to the
  sandbox run (tick rule accuracy 0.6618, Kyle's Lambda range
  [-10787.4, 22233.8], VPIN mean 0.5256, etc.) — nothing here was
  environment-sensitive.
- **Not yet merged into Ch07's training table.** This chapter produces a
  bar-indexed 249-row feature table
  (`input_data/ch19_microstructural_features.{csv,pkl}`); joining it onto
  the 88-event training table (aligning each triple-barrier event to its
  bar) and re-running Ch08/09/12/13 against the enriched table is the
  deliberate next step, not done in this chapter's delivery.
- A handful of `RuntimeWarning: invalid value encountered in divide`
  warnings appeared during Part C in the sandbox run (from
  `pandas.Series.autocorr()` hitting a zero-variance window and correctly
  returning NaN — expected, not a bug). They did not appear in the real
  machine's captured output; likely just a difference in how the terminal
  captured/displayed warnings rather than a real behavioral difference,
  since the underlying `serial_corr_signed_flow` NaN counts (206/249,
  matching the sandbox run) are identical either way.
