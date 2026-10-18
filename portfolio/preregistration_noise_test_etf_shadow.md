# Pre-registration: noise test of the formal evaluation rule (Tier A)

Registered 2026-10-17, before any run. Files: `shadow_evaluate.py`, `noise_test_tier_a.py`, and their tests.

## Question
The ETF forward shadow model (see `preregistration_etf_shadow_model.md`) will be judged once, at 36 resolved periods, by
a fixed rule: for each of three fitted arms, `edge_found` needs (1) the lower end of the paired 96.7% interval for
ir(arm) minus ir(`wc_vol_rule`) above 0 with `vol_tilt` at most 0.7, and (2) the lower end of the arm's own 96.7%
interval above 0. **When there is no edge at all, how often does that rule still say "edge found"?** And how large must a
real edge be before the rule finds it?

## What is run
`python noise_test_tier_a.py --prices prices_daily_asof_2026-10-16.csv --workers 4`, once, in the mlfinlab env.
- 7 values of beta (0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30) x 2,000 worlds each. World `i` of every beta uses seed
  20261017 + i.
- One world = 36 independent forward periods for the 48 ETFs in their 6 classes. Returns are fat-tailed (Student-t, 5 df),
  scaled by each ETF's real 21-day volatility (last five years of the price file), with a class-wide shock that cancels
  in the class-neutral statistic. Beta 0 = scores independent of returns = **no edge**. Beta above 0 plants a
  within-class signal that the three arms see only partly (q = 0.5).
- Arms: vol tilts 0.5 (`wc_current`), 0.2 (`wc_no_vol`), 0.5 (`wc_book`); shared noise between arms 0.9, between
  `wc_current` and `wc_book` 0.98; `wc_vol_rule` = within-class volatility rank. Full definitions are in the file's
  docstring.
- Each world goes through `shadow_evaluate.decide_formal` (96.7% level, 20,000 resamples, block 4, seed 20261009), the
  code the real evaluation will use. Its vectorised resampling is tested equal to the loop in the earlier checks.

## Pass criteria (frozen now)
1. **Calibration at beta 0:** the family-wise rate (at least one of the three arms has `edge_found`) is **at most 0.06**,
   and each arm's own `edge_found` rate is at most 0.04.
2. **Reported, no threshold:** power per beta (per arm and any-arm) with the realized mean ir of `wc_current`; the
   smallest realized ir at which any-arm power reaches 0.8 is quoted as the minimum edge the rule can confirm.

## If criterion 1 fails
The rule is too easily fooled (block-bootstrap intervals on 36 periods can be too narrow). The forward test has not
produced a single resolved period, so the rule can still be changed. That change would be a dated amendment to
`preregistration_etf_shadow_model.md` before 36 periods exist, citing this result; it would not be a silent edit.

## What this does not test
- It tests the **statistic's calibration**, not the model or the daily pipeline (Tier B does that on a few worlds).
- Periods are independent here; real active returns may be serially correlated, which would make the true false-positive
  rate higher than measured. Volatility ranks are static; real ones drift.
- The planted signal is generic, not the real features. Power numbers say how big an edge the rule needs, not what the
  model has.
