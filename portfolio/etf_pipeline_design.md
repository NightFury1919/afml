# ETF pipeline design (decisions as of 2026-10-04)

Agreed with Ethan on 2026-10-04. Change a decision here before changing code.

## Universe
- 48 ranked ETFs plus the SH hedge (see `universe.txt`). SH is pulled but never
  ranked or labeled (`panel_data.drop_hedge`).
- An ETF joins the panel on a date only when all five features exist for it.
- A date needs at least 10 eligible ETFs (`min_assets = 10`).

## Labels (`etf_labels.py`)
- 1 if the ETF's forward return over h trading days beats the cross-sectional
  median on the same date, else 0. Ties with the median get 0.
- h = 21 trading days (about one month). Judgment call for costs, not a tested
  result. The signal is refreshed daily.
- t1 = the date h trading days ahead, for purging and uniqueness.
- Uniqueness: per-ETF (`panel_uniqueness.py`) as a first approximation. The
  shuffle null measures whether that inflates false positives.

## Cross-validation (`panel_purged_cv.py`)
- Folds are blocks of dates; purge by date across all ETFs; embargo after the test block.
- Embargo = 252 dates (the longest feature lookback, mom_12_1). Decided 2026-10-04.
  It costs about 5% of the history.

## Features (`etf_features.py`), five fixed, no lookback tuning
| Feature | Definition | Why someone would pay for it |
|---|---|---|
| mom_12_1 | return from t-252 to t-21 | winners keep winning for months |
| mom_3m | return over 63 days | medium-horizon continuation |
| ret_5d | return over 5 days | short-term overreaction fades |
| vol_60 | realized volatility, 60 days | low-risk assets are often under-priced |
| trend_200 | price / 200-day average - 1 | trend filter |

Returns, momentum and trend are divided by the ETF's trailing 60-day volatility,
then every feature is ranked across the eligible ETFs each date, scaled to (0, 1).

## Positive control (to build)
1. Scoring-machinery power: reuse `positive_control_exposure_power.py` (iid PnL
   with a planted z). It tests PBO, DSR and the ramp, not the model.
2. End-to-end: plant a known signal into the real daily ETF returns, so features
   and labels are recomputed from the planted path. Plant sizes: gross active
   Sharpe of about 0.3, 0.5, 0.75, 1.0, 1.5 for a long-only top-ranked portfolio
   against the equal-weight universe.
3. Record: detection rate (confidence above the null's 95th percentile), the
   share of the planted Sharpe captured after costs, and the exposure the ramp gives.
4. Shuffle null: same pipeline on labels shuffled across the cross-section.
5. PROPOSED pass bar (not yet agreed): detects a planted 0.75 at 50% power or
   better, with false positives at 5% or less.

## Trials
Every model configuration counts as a trial in the DSR and PBO deflation.
Declare the grid in a pre-registration before running. The time-decay c values
count too.

## Build order and status
- [x] panel data loader, features, labels (this commit)
- [x] panel purged CV by date across all ETFs (`panel_purged_cv.py`, 19 tests, mutation-checked)
- [ ] positive-control harness and shuffle null
- [ ] effective breadth (Meucci, from correlation eigenvalues)
- [ ] portfolio construction (HRP x signal tilt x confidence scalar, long-only)

## Still open
- Success benchmark: equal-weight universe, or SPY?
- Model class and the declared trial grid.
