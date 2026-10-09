# Pre-registration: within-class real-history check

Status: **draft for approval. Nothing below has been run on the real data.** Freeze it by committing it before running
`python real_history_check_wc.py`. The script refuses to run twice (it will not overwrite its results).

## Why this check
`preregistration_real_history_check.md` found the pooled baseline indistinguishable from a volatility sort (tilt 0.885,
paired interval for current minus vol_rule [-0.09, +0.28]). Pooled ranking mostly sorts classes against each other
(bonds against emerging-market stocks), and class volatility differences are huge. The positive control (v1.2) showed
that ranking inside each of the 6 asset classes is a cleaner, better-powered test (power 0.87 vs 0.29 at nominal IC 0.03,
null false-positive rate 10% vs 26%). This repeats the real-history check with everything built inside each class.

## Data and design (nothing tuned)
`prices_daily_asof_2026-10-08.csv`, the 48 ranked ETFs (SH excluded), classes from `etf_classes.CLASS_OF`
(US equity 20, International developed 7, Emerging markets 6, Bonds 8, Commodities 4, Currencies 3).
Features: the five frozen features, each ranked within its own class per date. Label: forward 21-day return above the
CLASS median (ties get 0). A class needs at least 3 eligible ETFs on a date; a date needs at least 10 in total.
Model: logistic regression, C = 1. Out-of-fold scores from `PanelPurgedKFold`, 5 folds, 252-date embargo.

## The three arms
1. **wc_current**: all five within-class features.
2. **wc_no_vol**: the same without `vol_60`. It cannot use volatility directly; momentum and trend features can still
   carry some of it, which is why the tilt is measured rather than assumed.
3. **wc_vol_rule**: no model. Score = the within-class `vol_60` rank.

## Statistics (raw forward returns, dates 21 apart)
- **ir**: annualized information ratio of the class-neutral active return. In each class hold the top fifth by score
  (ceil, at least 1) minus that class's equal-weight mean; classes weighted by their number of ETFs that date. A class-wide
  move cancels, so only selection inside classes is credited. (With these weights it equals "top picks minus the whole
  universe".) Class weights proportional to class size are a placeholder: the real class budgets are a boss decision.
- **mean_ic**: mean over (date, class) of the out-of-fold Spearman IC against excess return versus the class median.
- **vol_tilt**: mean over (date, class) of the Spearman correlation between the score and the within-class `vol_60` rank.
- **mean_vol_rank_of_picks**, first-half and second-half ir (descriptive only).
- Intervals: moving-block bootstrap, block = 4 periods, 5,000 resamples, seed 20261009, 90% level. Paired intervals
  resample the same blocks for both arms; each arm also gets a standalone interval for its own ir.

## Pass rules (frozen at commit), applied to each fitted arm
1. **adds_beyond_volatility**: lower end of the paired 90% interval for (arm minus wc_vol_rule) in ir is above 0 AND the
   arm's vol_tilt is at most 0.7.
2. **positive_alone**: lower end of the arm's standalone 90% interval for ir is above 0.
3. **edge_found**: rules 1 and 2 both hold.

The standalone rule is added because beating a volatility sort is not enough: the volatility rule's own ir was only 0.13
in the pooled check, so an arm could beat it and still be indistinguishable from zero.

## Number of trials, stated honestly
This registration runs 2 fitted variants and 1 rule. Together with the pooled check (2 fitted variants), 4 fitted
variants will have been tried on this one history. The data are reused, so a pass here carries forking-path risk and would
call for a harder test (a held-out period, or more data), not for trading real money. No other variant will be run under
this registration; any further variant needs its own registration and counts as another trial.

## What each outcome would mean (a plan, not a result)
| Outcome | Next step |
|---|---|
| Neither fitted arm has `edge_found` | There is no demonstrated predictive edge in these features on this history. For the challenge, switch to a transparent non-predictive strategy (class-diversified allocation with the boss's risk limits; the volatility sort is not preferred, since it is risk, not alpha). |
| `wc_no_vol` has `edge_found`, `wc_current` does not | Use the no-vol variant as the next candidate; test it on data not used so far. |
| `wc_current` has `edge_found` | Candidate; but check that it differs from the vol rule (tilt) before believing it, and test on new data. |
| Both | Prefer `wc_no_vol` (less volatility exposure), then test on new data. |

## Power and honesty
About 320 non-overlapping periods give each arm an ir standard error near 0.2, so a standalone pass needs an ir of roughly
0.33 or more, and a paired pass a gap of similar size. Failing to pass means "not shown", not "proved useless".

## Limits
Survivorship (the 48 ETFs are today's survivors); k-fold purged CV trains on later periods too (not walk-forward); no
costs, no shorting, no time decay; overlapping 21-day labels sampled daily in training; classes with 3 or 4 ETFs
(Commodities, Currencies) contribute coarse picks (one ETF per class each date).

## Reporting
All arms, intervals, half-split numbers and every rule outcome are reported whether or not they pass, and recorded as
comments at the bottom of `real_history_check_wc.py` and in an Outcome section here after the run.
