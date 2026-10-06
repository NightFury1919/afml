# Pre-registration: ETF positive control v1.1 (demeaned no-edge world)

Written 2026-10-04, after the v1 result and before any v1.1 run. Commit this file
before running `positive_control_etf.py --demean`.

## Why this run exists

v1 did not meet its primary bar: at nominal IC = 0 the fixed rule t >= 1.96 fired in
69.4% of worlds (needed 2% to 8%), and the model's mean IC in no-edge worlds was 0.057.
Diagnostics (40 fresh worlds, development only) showed the cause: the block bootstrap
keeps each ETF's own average return, those averages differ widely (annualized -2.2% to
+15.8%), and the mom_12_1 rank had mean IC +0.020 in the "no-edge" worlds with the
averages kept and -0.003 with them removed. So v1's no-edge world was not truly no-edge.
This run checks the harness. It is not a test of the strategy.

## The single change

Each ETF's average daily log return (over the 4,725-day sample) is subtracted before the
bootstrap. Volatility, fat tails, volatility clustering and the real correlation between
ETFs are unchanged. Everything else is identical to v1: the same 500 seeds (so each world
uses the same blocks as in v1), the same IC levels (0, 0.02, 0.03, 0.05, 0.075, 0.10),
the same planting (a = IC / sqrt(21)), pipeline, statistic and detection rule.

## Reported, whatever the values

The same table as v1, plus a paired comparison of v1 and v1.1 at nominal IC = 0: mean
model IC, mean oracle IC, mean t, and the false-positive rate of t >= 1.96.

## Harness-validity criteria (the main question of this run)

The null is clean only if ALL hold at nominal IC = 0:

1. The false-positive rate of t >= 1.96 is between 2% and 8%.
2. The mean model IC is within +/-0.01 of zero.
3. The mean oracle IC (the mom_12_1 rank itself) is within +/-0.01 of zero.

## Detection bars, evaluated only if the null is clean

The v1 roles stay: the primary bar is at least 50% power at nominal IC = 0.05 together
with a false-positive rate of 2% to 8%; the 0.03 and 0.075 power levels are reported as
sensitivity, not used for the decision. The v1 outcome ("not met") stays on record
unchanged. This run is a second look after a documented fix to the harness, not a
replacement for it.

## What each outcome means

- **Clean null:** the demeaned world becomes the reference power curve and a fixed
  threshold near t = 1.96 becomes usable. Next: the lever what-ifs (long/short, horizon,
  fewer features) and a plan for the real-data test.
- **Not clean:** another source of inflation remains. Candidates, in the order I would
  check them: k-fold CV training on later periods (a walk-forward variant would be a new
  pre-registration), the independence assumption behind the t-statistic, and the model
  finding structure in no-edge data. Nothing downstream is built until that is resolved.

## Known limits

- Removing each ETF's average also removes any real static premium from the world. The
  question here is whether time-varying or ranking edges are detectable, not whether
  static premia exist.
- The mean is removed over the full sample, so it uses hindsight. That is fine for
  building a no-edge world; it would not be fine inside a trading signal.
- All v1 limits stand: one planted edge shape, a linear model, uniform sample weights,
  no costs, no shorting, no DSR or PBO, and a t-statistic that assumes independent
  spaced-date ICs.

## Process rules

- One run, these settings, 500 worlds. Results go to
  `positive_control_etf_demeaned_results.csv`, separate from v1.
- No changes after seeing results. Any variant needs a new pre-registration and counts
  as an additional trial.
