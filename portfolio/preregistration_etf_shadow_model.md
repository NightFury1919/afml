# Pre-registration: ETF forward shadow model (stage 1)

Status: **draft for approval. Nothing below has been built or run.** Freeze it by committing it BEFORE writing or running
any shadow-scoring code, and confirm the commit landed before the first run (never chain a run after a commit in the
same command block; see `handoff_2026-10-09_etf_pipeline.md`).

## Why this exists
Two real-history checks (`preregistration_real_history_check.md`, `..._wc.md`) found no demonstrated predictive edge in
the five frozen features: all six pre-registered rules were False, and the best arm (`wc_current`, ir 0.18, 90% interval
[-0.13, +0.53]) cannot be told from zero. Four fitted variants have now been tried on this one history, so **no further
variant may be tried on it**. A new model needs held-out data. The only held-out data that exist are the days that have
not happened yet. This registration therefore freezes two already-tested models and scores them forward, one day at a
time, without trading on them. "Shadow" means: scores are recorded, nothing is ordered from them.

## What stage 1 is, and is not
- IS: a daily scoring step that produces, for each of the 48 ranked ETFs, a score from each frozen arm, logged with the
  inputs needed to audit it, and a forward record of the pre-registered statistics.
- IS NOT: an exposure ramp. The confidence-exposure ramp (`apply_confidence_exposure.py`, confidence = DSR x (1 - PBO),
  zero point 0.5363) was calibrated on the crypto pipeline. An ETF version needs its own trial set and its own null
  calibration (the positive-control harness is the tool for that). That is **stage 2, a separate registration**.
  Until then the live portfolio's `exposure` stays a fixed config value.
- IS NOT: a change to the live paper account. The live non-predictive allocation (protocol v2) runs unchanged.

## Forward boundary (the held-out rule)
- First shadow scoring date: **2026-10-16** (as-of date; uses prices through that close). Its 21-day label resolves
  2026-11-16.
- Nothing dated on or before 2026-10-16 is ever used for evaluation. History is used for fitting only.
- Each daily score uses only prices up to and including its as-of date, and a model fitted only on labeled rows whose
  label end date `t1` is on or before that as-of date (no label that had not yet resolved).

## Arms (frozen)
| Arm | What it is | Already tested on history? |
|---|---|---|
| `wc_current` | all five within-class features, logistic regression C = 1 | yes (ir 0.18) |
| `wc_no_vol` | the same without `vol_60` | yes (ir -0.02) |
| `wc_vol_rule` | no model; score = within-class `vol_60` rank (reference) | yes (ir 0.03) |

Forward trials: 2 fitted arms. No other arm runs under this registration. Time decay is **not** an arm here (see
Decisions); adding any arm later needs its own registration and counts as another trial.

## Model, features, labels (all frozen, nothing tuned)
- Universe: the 48 ranked ETFs of `universe.txt` (SH excluded); classes from `etf_classes.CLASS_OF`.
- Features: `etf_features` defaults, each ranked within its own class per date (same as the within-class check).
- Label: forward 21-day return above the class median (ties 0), `etf_labels.make_labels` with `groups=CLASS_OF`.
- Fit: expanding window, all labeled rows available at the as-of date, unweighted. Refit every 21 trading days (the first
  scoring day and every 21st after it); between refits the fitted model is reused to score new days.
- A class needs at least 3 eligible ETFs on a date and a date at least 10 in total, as in the history checks.

## Daily scoring log (the audit trail)
One file per as-of date, `shadow/scores_asof_<YYYY-MM-DD>.csv`, **never overwritten**, with one row per ETF and columns
`asset, class, wc_current, wc_no_vol, wc_vol_rule, vol_60_rank`, plus a header record of: as-of date, model fit end date,
last refit date, the price snapshot file name and its SHA-256, and the code version (git commit). Scores are never
recomputed after the fact; a restated price history changes later scores, not logged ones. Adjusted closes are revised
when ETFs pay dividends, so the logged snapshot hash is what makes a score reproducible.

## Statistics (same definitions as the within-class check)
Evaluated only on forward periods whose label has resolved, on as-of dates **21 trading days apart** starting 2026-10-16
(non-overlapping), using raw forward returns:
- `ir`: annualized information ratio of the class-neutral active return (top fifth of each class by score, ceil, at least
  1, minus that class's equal-weight mean; classes weighted by their number of ETFs on that date).
- `mean_ic`: mean over (date, class) of the Spearman correlation between score and excess return versus the class median.
- `vol_tilt`: mean over (date, class) of the Spearman correlation between the score and the within-class `vol_60` rank.

## Reporting and decision rules
- **Monthly, descriptive only**: every statistic above for every arm, number of resolved periods, no pass/fail, no
  intervals (too few periods). No decision is taken from these.
- **Formal evaluation at 36 resolved non-overlapping periods** (about 3 years of forward data), using the within-class
  check's frozen rules and the same moving-block bootstrap (block 4, 5,000 resamples, seed 20261009, 90% level):
  1. `adds_beyond_volatility`: lower end of the paired 90% interval for (arm minus `wc_vol_rule`) in ir above 0 AND the
     arm's `vol_tilt` at most 0.7.
  2. `positive_alone`: lower end of the arm's standalone 90% interval for ir above 0.
  3. `edge_found`: both hold. Two fitted arms are tested, so an arm needs `edge_found` and the other arm's result is
     reported beside it; a pass in one arm of two is described as such, not as a clean win.
- No early stopping on good results, and no change of arms, refit schedule or rules after the freeze.

## Power and honesty
The history gave about 320 periods and an ir standard error near 0.2. Forward, a period arrives every 21 trading days,
so 12 per year, and the standard error of an annualized ir is roughly 1 / sqrt(years): about 1.0 after one year and 0.6
after three. **A forward test of these arms can only confirm a very large edge (an ir near 1 or more).** Over this
semester it can confirm nothing; at best it shows that the pipeline runs and starts an honest record. Failing to pass
means "not shown", not "proved useless". If a decision is needed sooner than three years, the route is more breadth or
more history (for example a much larger stock universe, which would be a new registration), not a shorter test of this one.

## Limits
Survivorship (48 ETFs that exist today; later closures are not modelled); adjusted-close revisions; no costs, no
shorting, no time decay; classes with 3 or 4 ETFs (Commodities, Currencies) give coarse picks; overlapping 21-day labels
are used in training; two data vendors' differences are not tested (Yahoo only).

## Not in stage 1 (and where each goes)
| Item | Where |
|---|---|
| Exposure ramp (confidence to exposure) | stage 2, separate registration, after an ETF null calibration |
| Time decay (AFML 4.7, `panel_time_decay`) as an arm | separate registration (a new trial) |
| A model-tilted portfolio with real orders | not before stage 2 and a formal evaluation |
| Larger stock universe | new registration; breadth is the real power lever |

## Decisions for the author before freezing
1. Time decay: the notes say it must be part of the ETF pipeline. This draft keeps it out of stage 1 so the two arms stay
   identical to the ones already tested, and registers a decay arm separately. Say if you want it as a third arm now (it
   costs power and adds a trial).
2. Refit every 21 trading days and an expanding window: proposed, not tested; change before the freeze if you prefer.
3. Formal evaluation at 36 periods: proposed. A shorter horizon is possible but the rules would be uninformative.
4. Whether to also log a descriptive model-tilted portfolio on the scoreboard (class budgets from the live rule, equal
   weights across each class's top fifth). Not in this draft, to keep the statistics unambiguous.

## Outcome
(Empty until the first monthly report and the formal evaluation.)
