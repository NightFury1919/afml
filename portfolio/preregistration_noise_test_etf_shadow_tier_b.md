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
