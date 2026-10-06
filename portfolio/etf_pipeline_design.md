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

## Within-class ranking (decided in principle 2026-10-06; boss approval pending)
- Rank features and label every ETF against its own class instead of one pool of 48. Classes
  (`etf_classes.py`): US equity 20, International developed 7, Emerging markets 6, Bonds 8,
  Commodities 4, Currencies 3. SH has no class.
- Why: effective breadth 16.5 (one pool) -> 27.2 (4 classes) / 30.2 (6 classes), no new ETFs.
  Estimated IC needed for 50% power falls from about 0.049 to about 0.036 to 0.038 (formula, to be
  confirmed by the positive control).
- Costs: the model no longer chooses stocks vs bonds (fixed class weights, to be decided), and the
  3- and 4-member classes give noisy rankings. With 3 members only one ETF can beat the median, so
  the Currencies label mean is exactly 1/3 (real data); odd-sized classes sit slightly below 0.5.
- Built: `compute_features`, `make_labels` and `build_dataset` take `groups=` (default None keeps the
  pooled behaviour). Real data: 283,570 rows vs 284,712 pooled; each class usable from 1999-12-22 (US,
  Intl developed), 2001-06-25 (EM), 2003-07-30 (Bonds), 2007-05-01 (Commodities), 2008-02-29 (Currencies).
- Next: harness support for grouped ranking, then a pre-registered comparison of pooled vs within-class
  power in the same worlds.

## Trials
Every model configuration counts as a trial in the DSR and PBO deflation.
Declare the grid in a pre-registration before running. The time-decay c values
count too.

## Build order and status
- [x] panel data loader, features, labels (this commit)
- [x] panel purged CV by date across all ETFs (`panel_purged_cv.py`, 19 tests, mutation-checked)
- [x] positive-control harness (`positive_control_etf.py`, 16 tests, mutation-checked); logistic regression on the five ranked features
- [x] positive-control v1 run (500 worlds x 6 IC levels): primary bar NOT met; null contaminated (false positives 69%) by static differences between ETF average returns
- [x] positive-control v1.1, demeaned worlds: null still NOT clean (false positives 26%, model null IC 0.028; oracle IC now clean at 0.002). Leftover IC is carried mostly by vol_60 (rank IC credits median/skew differences that survive mean removal). Details at the bottom of `positive_control_etf.py`
- [ ] v1.2 proposal (needs a pre-registration): label-level design. Null = permute forward returns across ETFs within each date (exact no-edge for any statistic); plant = add a tilt to the forward returns; same CV, model and statistic. Optional: a mean-based statistic (top-quintile active return) alongside rank IC
- [ ] possible follow-up: walk-forward (past-only) CV variant, if v1.1 is not clean
- [ ] lever what-ifs, AFTER the baseline positive control: long-short vs long-only (TC), horizon (e.g. 5 / 21 / 63 days), fewer features; same harness, compare what each can detect
- [x] effective breadth (`effective_breadth.py`, 12 tests): ENB 16.5 vs median (9.0 on raw returns); implied IC needed 0.049 (TC=1) / 0.098 (TC=0.5) at 50% power
- [x] rebalance step (`rebalance.py`, 19 tests): targets vs current holdings -> sells first, then buys; whole-share rounding bug fixed in `sizing.py`
- [ ] portfolio construction (HRP x signal tilt x confidence scalar, long-only): produces the target weights the rebalance step consumes

## Still open
- Success benchmark: equal-weight universe, or SPY?
- Model class and the declared trial grid.
