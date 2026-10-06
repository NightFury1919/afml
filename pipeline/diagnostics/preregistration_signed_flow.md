# Pre-registration: signed trade-flow imbalance and the next 4-hour return

Written 2026-10-03, before the test statistic was computed on any real window.
Commit this file before running `signed_flow_predictability.py`.

## Hypothesis (one, two-sided)

Signed trade-flow imbalance in one 4-hour block carries information about the
log return of the next 4-hour block.

Why it might hold: aggressive one-sided market orders move price. If that
pressure persists, the next block continues the move. If it overshoots, the next
block reverses. The test is two-sided, so it counts either direction.

## Data (nothing new is downloaded)

The 8 independent 30-day Kraken XBTUSD windows from
`briefing_kraken_8window_resolution.md`, as listed in `WINDOW_DIRS` in
`signed_flow_predictability.py`. The script stops if any two windows overlap.

## Construction (frozen)

- Blocks: 4-hour blocks on UTC clock boundaries. The first and last block of each
  window are dropped (partial). Blocks with fewer than 50 trades are dropped.
- Imbalance = (buy-initiated volume - sell-initiated volume) / total volume.
  Buy-initiated means `IsBuyerMaker == False`. That mapping is an assumption noted
  in `ingestion_kraken.py`; if it is backwards, only the sign of rho flips.
- Next return = log(close of next block / close of this block), where close is
  the last trade price in the block.
- Pairs use only consecutive blocks inside one window. No pair crosses a window.
- The imbalance uses only trades inside its own block, so nothing looks ahead.

## Statistic and decision rule (frozen)

- Pooled Spearman correlation rho over all pairs, with
  t = rho * sqrt((n - 2) / (1 - rho^2)).
- Evidence of predictability only if both hold: |t| >= 1.96, and rho has the
  same sign in at least 6 of the 8 windows.
- Otherwise: no evidence of predictability.

## Expectations stated in advance

- About 178 pairs per window, about 1,400 pooled (one real window checked for
  counts only, no statistics).
- Smallest correlation detectable at |t| = 1.96 is about 0.052. With 80% power
  it is about 0.074. A correlation that size explains under 1% of return variance.
- On pure noise the full rule fires about 4% of the time (2,000 simulated runs).
- A pass is a lead, not a result. It still has to survive costs, and it has to
  replicate on the new windows (9 and later) from the overnight download under
  the same rules before any further step.

## Process rules

- One hypothesis, one run. The script refuses to overwrite its results file.
- No other flow variants (block length, horizon, volume weighting, thresholds)
  under this registration. Any variant needs a new pre-registration and counts as
  an additional trial in the deflation.
- Report n, rho, t, the per-window rhos and the sign count, whatever they are.
