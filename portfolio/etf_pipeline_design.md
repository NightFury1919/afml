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

## Positive control (`positive_control_etf.py`, pre-registered in `preregistration_etf_positive_control.md`)
1. Scoring-machinery power (reuse `positive_control_exposure_power.py`, iid PnL with a planted z)
   tests PBO, DSR and the ramp, not the model.
2. End-to-end: block-bootstrap the real 48-ETF returns (no-edge world), plant a tilt on the
   `mom_12_1` rank at nominal IC 0 / 0.02 / 0.03 / 0.05 / 0.075 / 0.10, recompute features and
   labels, run panel purged CV with a logistic regression, and measure the IC t-stat.
3. Detection = t above the 95th percentile of the IC = 0 worlds. Also reported: false-positive
   rate of t >= 1.96, realized planted IC, long-only top-quintile active IR, capture ratio.
4. Decision bars (fixed before the run): primary = power >= 50% at IC 0.05 with false positives 2% to 8%;
   sensitivity, reported only = the same rule at IC 0.03 and 0.075.

## Trials
Every model configuration counts as a trial in the DSR and PBO deflation.
Declare the grid in a pre-registration before running. The time-decay c values
count too.

## Build order and status
- [x] panel data loader, features, labels (this commit)
- [x] panel purged CV by date across all ETFs (`panel_purged_cv.py`, 19 tests, mutation-checked)
- [x] positive-control harness (`positive_control_etf.py`, 16 tests, mutation-checked); logistic regression on the five ranked features
- [ ] full positive-control run: ready (500 worlds x 6 IC levels)
- [ ] lever what-ifs, AFTER the baseline positive control: long-short vs long-only (TC), horizon (e.g. 5 / 21 / 63 days), fewer features; same harness, compare what each can detect
- [x] effective breadth (`effective_breadth.py`, 12 tests): ENB 16.5 vs median (9.0 on raw returns); implied IC needed 0.049 (TC=1) / 0.098 (TC=0.5) at 50% power
- [ ] portfolio construction (HRP x signal tilt x confidence scalar, long-only)

## Still open
- Success benchmark: equal-weight universe, or SPY?
- Model class and the declared trial grid.
