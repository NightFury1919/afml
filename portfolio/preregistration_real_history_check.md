# Pre-registration: real-history check, "does the model add anything beyond volatility?"

Status: **draft for approval. Nothing below has been run on the real data.** Freeze it by committing it before
running `python real_history_check.py`. The script refuses to run twice (it will not overwrite its results).

## Why this check
The first dry-run plan from the frozen baseline (`plans/plan_2026-10-08.json`) had a rank correlation of 0.92 with
60-day volatility, and 7 of its 10 picks were among the 10 most volatile ETFs. Earlier positive controls also showed
the rank statistic crediting volatility in no-edge worlds. Before any paper money, test on the REAL history whether
the model beats a plain volatility sort, and whether the book's remedy removes the tilt.

The book (AFML 3.2 and 3.3) calls a label threshold that ignores volatility "a very common error" and scales by a
rolling exponentially weighted standard deviation (Snippet 3.1). Section 4.6 weights samples by absolute return, which
would push the other way here, so it is not used.

## Data
`prices_daily_asof_2026-10-08.csv`, the 48 ranked ETFs (SH excluded), all dates with five features and a resolved
21-day label (about 285,000 rows). Features, eligibility and labels are the frozen ones (`etf_pipeline_design.md`).

## The three arms
1. **current**: logistic regression (C = 1) on the frozen label (forward 21-day return beats the cross-sectional median).
2. **vol_scaled**: the same model and features; the label compares forward return divided by (daily volatility x sqrt(21)),
   with the volatility known at the label date. Ties with the median get 0, as before.
3. **vol_rule**: no model. Score = the vol_60 rank. Holds the most volatile fifth.

Out-of-fold scores come from `PanelPurgedKFold`, 5 folds, 252-date embargo, as in the positive control. Nothing is tuned.
Number of trials: 2 fitted variants and 1 rule. No other variant will be run on this data under this registration; any
further variant needs its own registration and counts as another trial.

**Deviation from Snippet 3.1, stated up front.** The repo's `get_daily_vol` (faithful to the book) uses a calendar-day
lookup for intraday bars. On daily bars it lands 1 to 3 bars back depending on the weekday and drops the first two dates
(checked on an 8-bar example). The check therefore applies the same estimator (exponentially weighted standard deviation,
span 100) to plain one-bar returns, with min_periods = 100. It uses only data up to each date (tested).

## Statistics (raw forward returns, dates 21 apart)
- **ir**: annualized information ratio of "hold the top fifth by score" minus the equal-weight universe.
- **mean_ic**: mean out-of-fold Spearman IC against raw 21-day excess return.
- **vol_tilt**: mean over dates of the Spearman correlation between the score and the vol_60 rank.
- **mean_vol_rank_of_picks**: average vol_60 rank (0 to 1) of the top fifth.
- First-half and second-half ir, for stability (descriptive only).
- Paired differences in ir between arms: moving-block bootstrap, block = 4 periods, 5,000 resamples, seed 20261009,
  90% interval, with the same resampled blocks for both arms.

## Pass rules (frozen at commit)
1. **A model arm "adds beyond volatility"** if the lower end of the 90% interval for (arm minus vol_rule) in ir is above 0
   AND the arm's vol_tilt is at most 0.7.
2. **"Scaling reduces the tilt"** if vol_tilt(current) minus vol_tilt(vol_scaled) is at least 0.4.

Calibration of the second threshold (sandbox, synthetic independent-returns worlds with no planted edge, 16 worlds,
48 ETFs x 4,700 dates, seeds 100 to 115): the difference in tilt between the two label designs had a spread of 0.20 and
a maximum of 0.44; 19% of worlds exceeded 0.2 by chance, which is why 0.2 was rejected and 0.4 chosen (about two spreads).
The tilt itself ranged from -0.9 to +0.7 across those worlds, so the 0.7 cap only separates a pure volatility sort from a
partial one; the interval condition in rule 1 does the main work. These worlds are not the real data, so treat the
calibration as indicative.

## What each outcome would mean (a plan, not a result)
| Outcome | Next step |
|---|---|
| Neither arm adds beyond volatility | The baseline is a volatility sort. Do not paper trade it. Redesign under a new registration. |
| current adds, vol_scaled does not | Keep the frozen label. |
| vol_scaled adds, current does not | Use the vol-scaled label in the next protocol version. |
| Both add | Keep the frozen label unless scaling also reduces the tilt (rule 2); then prefer vol_scaled. |

## Power and honesty
About 18 years of data gives roughly 210 non-overlapping periods, so each arm's ir has a standard error near 0.25 and a
paired difference near 0.3 (in no-edge synthetic worlds the ir spread was about 0.3). Only a large gap will pass rule 1.
Failing to pass means "not shown", not "proved useless".

## Limits
Survivorship: the 48 ETFs are today's survivors. K-fold purged CV trains on later periods too (standard AFML CV, not
walk-forward). No costs, no shorting, no time decay. Overlapping 21-day labels sampled daily in training.

## Reporting
All arms, intervals, half-split numbers and both rule outcomes are reported whether or not they pass, and recorded as
comments at the bottom of `real_history_check.py` after the run.


## Outcome (2026-10-09)

Run once on `prices_daily_asof_2026-10-08.csv` after this file was committed. 320 non-overlapping periods.

| Arm | ir | mean IC | vol tilt | vol rank of picks | ir first / second half |
|---|---|---|---|---|---|
| current | 0.237 | 0.073 | 0.885 | 0.847 | 0.279 / 0.185 |
| vol_scaled | -0.031 | 0.010 | -0.339 | 0.330 | 0.039 / -0.121 |
| vol_rule | 0.129 | 0.072 | 1.000 | 0.895 | 0.123 / 0.144 |

Paired difference in ir (90% interval): current minus vol_rule [-0.09, +0.28]; vol_scaled minus vol_rule
[-0.67, +0.34]; vol_scaled minus current [-0.73, +0.21].

Rules: current adds beyond volatility **False**; vol_scaled adds beyond volatility **False**; scaling reduces tilt **True**.

Outcome table row: **neither arm adds beyond volatility.** The frozen baseline is not distinguishable from a
volatility sort, so it should not be paper traded as it stands; a redesign needs a new registration.

Observations (not part of the rules): the model's mean IC equals the volatility rule's, so all of its rank IC is the
volatility signal; the vol-scaled label over-corrected into a low-volatility tilt and earned nothing; the run had 320
periods, more than the roughly 210 assumed above, because early dates have 10 or more ETFs. This is "not shown", not
"proved useless": each arm's ir has a standard error near 0.2.
