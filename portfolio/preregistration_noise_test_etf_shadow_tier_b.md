# Pre-registration: noise test of the real shadow pipeline (Tier B)

Registered 2026-10-17, before any run. Files: `noise_test_tier_b.py`, `test_noise_test_tier_b.py`. Companion to
`preregistration_noise_test_etf_shadow.md` (Tier A, which tested the evaluation rule only).

## Question
Does the actual daily pipeline (`shadow_scoring.run_shadow`: expanding-window refit every 21 days, three fitted arms with
the AFML time-decay and uniqueness weights, the volatility rule, the log files) produce skill from pure noise? A leak, a
look-ahead, or a bug that rewards noise would show up as an arm earning a steady positive ir or IC in worlds where
nothing can be predicted.

## What is run
`python noise_test_tier_b.py --prices prices_daily_asof_2026-10-16.csv --workers 4`, once, mlfinlab env.
- 8 worlds, seeds 20261018 to 20261025. Each world: the real 48-ETF daily log returns over the days when all 48
  exist (since 2007-12-20), each ETF's average removed (no ETF better than another), resampled in moving blocks of
  63 whole days, turned into prices. 1,260 days of history, then 36 scoring dates 21 trading days apart, then 21 more
  days so the last period resolves.
- On each scoring date `run_shadow` is called with the live settings (first scoring date = the first of the 36, a refit
  every 21 days, so every scoring date is a refit day). Nothing about the model, features, weights or windows is changed.
- The 36 logs give the registered statistics: class-neutral active returns, ir, mean IC, vol_tilt, and
  `shadow_evaluate.decide_formal` (Tier A's code).
- The history is 5 years, not the live 30+, to keep one world near 20 to 30 minutes (each world refits 36 times).

## Pass criteria (frozen now)
1. **Integrity, every world:** all 36 scoring dates logged, 48 ETFs scored on each, `refit_today` true and `late_fit`
   false on each, 36 resolved periods, no error.
2. **No skill from noise:** for each of the three fitted arms, the mean over the 8 worlds of ir, and of mean IC, is within
   2 standard errors of zero (standard error = sd across worlds / sqrt(8)).
Both must hold. The run **passes** only if both do.

## Reported, no threshold
Per-world ir, mean IC, vol_tilt for each arm; how many worlds triggered `edge_found` per arm; the realized vol_tilt of
each arm against Tier A's assumed 0.5 / 0.2 / 0.5. If the real tilts differ a lot from those, Tier A's calibration
assumed the wrong arms and should be rerun with the measured values (a dated follow-up, not an edit of the result).

## What this can and cannot show
- Eight worlds can expose gross faults only: with 8 worlds the standard error of a mean ir is about 0.3 to 0.4, so a
  small leak would not be seen. A pass means "no gross leak", not "no leak".
- Tier A measured the false-positive rate of the rule; this does not.
- A failure (integrity or skill from noise) stops everything: the forward log and the paper-account plan are not
  trusted until the cause is found, fixed by a dated amendment, and the test is rerun as a new registered run.

## Outcome (appended 2026-10-18 after the registered run; nothing above this line was changed)
Run on Ethan's machine (mlfinlab env, `prices_daily_asof_2026-10-16.csv`, 8 worlds, seeds 20261018 to 20261025; results in
`noise_test_tier_b/summary.csv`). It finished in well under the estimated hour.

**Criterion 1, integrity: MET** in all 8 worlds (36 of 36 dates logged, 48 ETFs scored every day, `refit_today` true,
no late fits, 36 resolved periods, no errors).
**Criterion 2, no skill from noise: MET.** Mean over worlds (standard error), all within 2 SE of zero:

| arm | mean ir (SE) | mean IC (SE) |
|---|---|---|
| wc_current | 0.115 (0.230) | 0.008 (0.020) |
| wc_no_vol | 0.054 (0.208) | 0.018 (0.012) |
| wc_book | 0.200 (0.263) | 0.007 (0.020) |

**The run passes.** No gross leak or bug that turns noise into skill was found. Eight worlds cannot detect a small one.
`edge_found` fired in 0 of 8 worlds for every arm. One arm-world (wc_current, world 2, ir 1.32) passed `positive_alone`
alone, 1 of 24 arm-worlds (4%), in line with Tier A's single-condition rate.

Reported, no threshold: **vol_tilt is far less stable than Tier A assumed.** Tier A fixed the tilts at 0.5 / 0.2 / 0.5.
In these noise worlds (5-year histories) the realized tilts were:
- wc_current: 0.41, 0.81, 0.03, -0.92, 0.52, 0.68, -0.74, -0.27 (mean about 0.0, sign flips)
- wc_book: 0.44, 0.78, 0.06, -0.82, 0.48, 0.61, -0.65, -0.31 (tracks wc_current closely)
- wc_no_vol: 0.02, -0.19, 0.05, -0.03, -0.12, -0.06, -0.01, -0.08 (near zero, as designed)
Two consequences, neither acted on here:
1. The registered limit is `vol_tilt <= 0.7` (signed). A tilt of -0.9 means the arm is buying the LOWEST-volatility ETFs,
   which is a volatility sort in the opposite direction, yet it passes the limit. The cap probably should be on the
   absolute value. Changing it would be a dated amendment to `preregistration_etf_shadow_model.md` made before 36 forward
   periods exist, with Tier A rerun using tilts drawn from this spread. That is a decision for Ethan; nothing was changed.
2. Training windows of 5 years make the volatility coefficient flip sign in noise. The live model trains on 30+ years, so
   its tilt should be steadier; the nightly logs record `vol_60_rank` so this can be watched.
