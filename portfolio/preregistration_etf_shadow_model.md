# Pre-registration: ETF forward shadow model (stage 1)

Status: **FINAL. Frozen by the commit that adds this file.** Nothing below has been built or run. After the freeze, the
text above the "Amendments" heading is never edited; any change is a dated amendment at the bottom and, if it adds a
model or changes a rule, counts as a new trial. Confirm the freeze commit landed before writing or running any
shadow-scoring code (never chain a run after a commit in the same command block).

## Why this exists
Two real-history checks (`preregistration_real_history_check.md`, `..._wc.md`) found no demonstrated predictive edge in
the five frozen features: all six pre-registered rules were False, and the best arm (`wc_current`, ir 0.18, 90% interval
[-0.13, +0.53]) cannot be told from zero. Four fitted variants have now been tried on this one history, so **no further
variant may be tried on it**. A new model needs held-out data, and the only held-out data are the days that have not
happened yet. This registration scores three fitted models forward, one day at a time, without trading on them.
"Shadow" means: scores are recorded and a hypothetical portfolio is tracked; nothing is ordered from them.

## Timeline and what this can deliver
The author has about two months of the fall 2026 semester and about one year before graduating. A period resolves every
21 trading days, so:

| Checkpoint | Resolved periods (approx.) | What it is |
|---|---|---|
| Semester report (end of fall 2026) | 2 to 3 | Descriptive only. Shows the pipeline runs and starts the record. No verdict. |
| Graduation report (about Oct 2027) | about 12 | Descriptive, intervals shown but explicitly not decisive. |
| Formal evaluation | 36 (about Oct 2029) | The only pass/fail. Likely run by a successor, using the handoff document. |

**Within the author's timeline this registration cannot deliver a verdict about skill.** Its deliverable is a working
pipeline and an unedited forward record. A faster verdict needs more breadth (many more assets), which is a separate
registration.

## What stage 1 is, and is not
- IS: a daily scoring step producing a score per ETF from each frozen arm, a logged audit trail, a descriptive
  hypothetical portfolio line per arm, and the pre-registered statistics.
- IS NOT: an exposure ramp. The confidence-exposure ramp (`apply_confidence_exposure.py`, confidence = DSR x (1 - PBO),
  zero point 0.5363) was calibrated on crypto. An ETF version needs its own null calibration. That is **stage 2, a
  separate registration**. Until then the live portfolio's `exposure` stays a fixed config value.
- IS NOT: a change to the live paper account. The live non-predictive allocation (protocol v2) runs unchanged.
- IS NOT: paper trading of the hypothetical portfolio. Mirroring it to a separate Alpaca paper profile is a possible
  later step and needs its own decision once the execution layer is proven.

## Forward boundary (the held-out rule)
- First shadow scoring date: **2026-10-16** (as-of date; uses prices through that close). Its 21-day label resolves
  2026-11-16.
- Nothing dated on or before 2026-10-16 is ever used for evaluation. History is used for fitting only.
- Each score uses only prices up to and including its as-of date, and a model fitted only on labeled rows whose label
  end date `t1` is on or before that as-of date (no label that had not yet resolved).

## Arms (frozen)
| Arm | What it is | Tested on history? |
|---|---|---|
| `wc_current` | all five within-class features, logistic regression C = 1, unweighted | yes (ir 0.18) |
| `wc_no_vol` | the same without `vol_60` | yes (ir -0.02) |
| `wc_book` | `wc_current` plus the AFML Chapter 4 treatment: average-uniqueness sample weights (Sections 4.3 and 4.4, `panel_uniqueness`) multiplied by time-decay factors (Section 4.7, Snippet 4.11, `panel_time_decay`) with **c = 0.5** (the oldest data gets half the weight of the newest) | **no** |
| `wc_vol_rule` | no model; score = within-class `vol_60` rank (reference only, not a trial) | yes (ir 0.03) |

Forward trials: **3 fitted arms**. `wc_book` weights are scaled to average 1 so that C = 1 means the same strength of
regularisation as in the unweighted arms. The value c = 0.5 is declared in advance and is **not tuned**; it sits in the
middle of the book's own examples (c in {1, .75, .5, 0, -.25, -.5}) and has no data behind it. `wc_book` has no history
to compare against, and because it adds weights and decay together, a difference from `wc_current` cannot be attributed
to one piece. No other arm runs under this registration.

## Model, features, labels (all frozen, nothing tuned)
- Universe: the 48 ranked ETFs of `universe.txt` (SH excluded); classes from `etf_classes.CLASS_OF`.
- Features: `etf_features` defaults, each ranked within its own class per date.
- Label: forward 21-day return above the class median (ties 0), `etf_labels.make_labels` with `groups=CLASS_OF`.
- Fit: expanding window, all labeled rows available at the as-of date.
- **Refit schedule: on the first scoring day (2026-10-16) and every 21st trading day counted from it.** The schedule is
  defined by trading-day count, so a missed run or a holiday cannot shift it. Between refits the fitted model is reused
  to score new days. Because the evaluation dates (below) are exactly the refit dates, every evaluated score comes from
  a freshly fitted model.
- A class needs at least 3 eligible ETFs on a date and a date at least 10 in total, as in the history checks.

## Daily scoring log (the audit trail)
One file per as-of date, `shadow/scores_asof_<YYYY-MM-DD>.csv`, **never overwritten**, one row per ETF with columns
`asset, class, wc_current, wc_no_vol, wc_book, wc_vol_rule, vol_60_rank`, plus a header record of: as-of date, model fit
end date, last refit date, the price snapshot file name and its SHA-256, and the code version (git commit). Scores are
never recomputed after the fact; a restated price history changes later scores, not logged ones. Adjusted closes are
revised when ETFs pay dividends, so the logged snapshot hash is what makes a score reproducible.

## Statistics (same definitions as the within-class check)
Evaluated only on forward periods whose label has resolved, on as-of dates **21 trading days apart** starting
2026-10-16 (non-overlapping), using raw forward returns:
- `ir`: annualized information ratio of the class-neutral active return (top fifth of each class by score, ceil, at least
  1, minus that class's equal-weight mean; classes weighted by their number of ETFs on that date).
- `mean_ic`: mean over (date, class) of the Spearman correlation between score and excess return versus the class median.
- `vol_tilt`: mean over (date, class) of the Spearman correlation between the score and the within-class `vol_60` rank.

## Hypothetical portfolio lines (descriptive, no verdict, no orders)
For each fitted arm, on each evaluation date: in each class hold the top fifth of ETFs by score (ceil, at least 1),
equal-weighted within the class; classes weighted by their number of ETFs on that date; fully invested, long-only, held
for 21 trading days, no costs. Reported beside two reference lines: the same class weights with equal weight inside each
class (the "class baseline"), and SPY. The arm's active return in `ir` is exactly this line minus the class baseline, so
this adds no new statistic and no new trial. Every report that shows these lines must say that two or three periods are
mostly luck.

## Reporting and decision rules
- **Monthly, descriptive only**: every statistic and line above for every arm, number of resolved periods, no pass/fail.
- **Semester and graduation reports**: as in the Timeline table. Intervals at the 90% level may be shown but carry no
  decision.
- **Formal evaluation at 36 resolved non-overlapping periods**, with the same moving-block bootstrap (block 4) but a
  stricter level because three arms are tested: each arm uses a one-sided error of 0.05 / 3, about 0.0167, i.e. a
  **96.7% interval** (lower end at the 1.67th percentile), **20,000 resamples**, seed 20261009. For each fitted arm:
  1. `adds_beyond_volatility`: lower end of the paired interval for (arm minus `wc_vol_rule`) in ir above 0 AND the
     arm's `vol_tilt` at most 0.7.
  2. `positive_alone`: lower end of the arm's standalone interval for ir above 0.
  3. `edge_found`: both hold.
  Results for all three arms are always reported together; a pass in one arm of three is described that way, not as a
  clean win. The stricter level makes a real but modest edge harder to confirm; that cost is accepted.
- No early stopping on good results, and no change of arms, `c`, refit schedule or rules after the freeze.

## Power and honesty
The history gave about 320 periods and an ir standard error near 0.2. Forward, a period arrives every 21 trading days,
so 12 per year, and the standard error of an annualized ir is roughly 1 / sqrt(years): about 1.0 after one year and
0.6 after three. **A forward test of these arms can only confirm a very large edge (an ir near 1 or more).** Failing to
pass means "not shown", not "proved useless". With the stricter level the required edge is a little larger still.

## Limits
Survivorship (48 ETFs that exist today; later closures are not modelled); adjusted-close revisions; no costs, no
shorting; classes with 3 or 4 ETFs (Commodities, Currencies) give coarse picks; overlapping 21-day labels are used in
training in all arms; Yahoo is the only price source.
`wc_book` uniqueness is computed within each ETF from its own overlapping 21-day windows (`panel_uniqueness`). The label
is relative to the class median, so labels on different ETFs in a class share information, which that function's own
documentation says it does not handle. The time-overlap effect dominates, so this is accepted as an approximation for
stage 1 and stated here rather than hidden.

## Not in stage 1 (and where each goes)
| Item | Where |
|---|---|
| Exposure ramp (confidence to exposure) | stage 2, separate registration, after an ETF null calibration |
| Paper-account mirror of the hypothetical portfolio | later decision, after the execution layer is proven |
| Other AFML items not yet in the ETF pipeline | decided after the book-to-pipeline map; each needs its own registration |
| Larger stock universe | new registration; breadth is the real power lever |

## Amendments
Dated amendments only, never edits above this line.

- 2026-10-17, replacement of the committed text. Commit `7e4d9df` ("Freeze pre-registration") contained the earlier
  DRAFT of this file (two fitted arms, 90% interval, open decisions listed) because the draft, not this final text, was
  the copy saved to disk. The final text above is what the author approved. It replaces the draft in the next commit.
  No shadow-scoring code had been written or run, and no forward data had been looked at, between the two commits, so
  nothing was chosen after seeing results. The differences are: a third fitted arm (`wc_book`), the 96.7% level with
  20,000 resamples, the refit schedule, the hypothetical portfolio lines, and the timeline table.

## Outcome
(Empty until the first monthly report, the semester and graduation reports, and the formal evaluation.)
