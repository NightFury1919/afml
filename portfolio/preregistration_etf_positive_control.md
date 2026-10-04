# Pre-registration: ETF pipeline positive control (v1 baseline)

Written 2026-10-04, before the full run. Commit this file before running `positive_control_etf.py`.

## Question

If a real edge of known size exists in this ETF universe, how often does the whole
pipeline find it, and how often does it "find" one when there is none?

## Frozen design

- **World:** moving-block bootstrap (block length 63 days) of the real 48-ETF daily
  log returns, 2007-12-20 to 2026-10-02 (4,725 days, every ETF present). Whole
  cross-sections are resampled together, so fat tails, volatility clustering and the
  real cross-ETF correlation stay. Real long-horizon momentum is destroyed, so
  nominal IC = 0 is a true no-edge world.
- **Planted edge:** the cross-sectional rank of `mom_12_1`, computed on the unplanted
  prices and centered to unit variance. Each ETF's next-day return gets
  `+ a * sigma_i * z`, with `a = IC / sqrt(21)`. Features and labels are recomputed
  from the planted prices.
- **Nominal IC levels (21-day horizon):** 0 (null), 0.02, 0.03, 0.05, 0.075, 0.10.
- **Replicates:** 500 worlds, seeds 20261004 onward; every IC level is planted into
  the same world (common random numbers).
- **Pipeline:** 5 ranked features, relative 21-day labels, panel purged CV (5 folds,
  252-date embargo), logistic regression (C = 1, uniform sample weights), out-of-fold
  probabilities for every row.
- **Statistic:** cross-sectional Spearman IC between the out-of-fold probability and
  the realized 21-day excess return, on dates 21 apart, then t = mean / (sd / sqrt(n)).
- **Detection rule:** t above the 95th percentile of the t-stats from the nominal
  IC = 0 worlds in the same run. Power = share of planted worlds above it.

## Reported, whatever the values

Power at every IC level, the IC = 0 false-positive rate of the fixed rule t >= 1.96,
mean model IC, mean realized IC of the planted score, long-only top-quintile active IR
for the model and for the oracle, and the capture ratio.

## Decision bars (fixed here, before the run)

The run reports power at every IC level, so one run answers all three candidate bars.
To keep the choice honest, the roles are fixed now:

- **Primary bar (used for the decision):** the pipeline detects the planted edge in at
  least 50% of worlds at nominal IC = 0.05, and the t >= 1.96 false-positive rate at
  IC = 0 is between 2% and 8%.
- **Sensitivity bars (reported, not used for the decision):** the same 50% power rule
  at nominal IC = 0.03 (a typical edge) and at 0.075 (a more lenient bar).
- No other bar may be chosen after seeing results.

Why 0.05 is primary: the effective-breadth arithmetic said an IC of about 0.049 is
needed at full signal capture (TC = 1), and a few hundredths is the range commonly
cited for single signals.

## What each outcome means

- **Pass:** the pipeline can certify an edge of that size in this universe. Next:
  the lever what-ifs (long/short, horizon, fewer features) to see how far the
  detectable size can move, then portfolio construction.
- **Fail:** the pipeline cannot certify an edge of that size here. Next: the lever
  what-ifs decide what to restructure before building anything else.

## Known limits (stated now so they are not discovered later)

- Realized IC of the planted score runs about 0.86 to 0.98 of nominal, because the
  score is not perfectly persistent over 21 days. The realized value is reported.
- One planted edge shape (a momentum-type rank tilt). A real edge may look different.
- Linear model, uniform sample weights, no transaction costs, no shorting, no DSR or
  PBO (one declared model counts as one trial).
- The t-statistic assumes the spaced-date ICs are independent.
- The block bootstrap resamples one history; it is not a new market.

## Process rules

- One run, these settings. The script appends per replicate and can resume, but the
  reported result is the full 500.
- No changes to settings after seeing results. Any variant (sample weights, time decay,
  features, horizon, model) needs a new pre-registration and counts as an additional
  trial.
