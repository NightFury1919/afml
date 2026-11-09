# Pre-registration: trend filter and group cap on the live momentum rule, "do either of these help C50 against SPY?"

Status: FROZEN

Not frozen. Nothing below has been run on real prices. The code (`momentum_variants.py`, `run_momentum_variants.py`) and its tests
exist and pass on made-up prices only. The runner refuses to run unless the line above says FROZEN, the SHA-256 under Data matches
the price file, the Groups file SHA-256 matches the group table, and no results exist yet (it runs once). Freeze means: change
`Status: FROZEN` to `Status: FROZEN`, commit and push this file and the group table, and only then run. Never chain the run after the
commit in the same command block. After freezing, the rules, the variants, the group table and the price file are fixed; any
change needs a new registration and counts as another trial.

## Why this check
The live momentum account (C50: top 3 ETFs from the ladder signal at 50% of the account, the rest in SPY; bought 2026-11-03) lost
about 5% in its first four trading days, mostly from EWT and EWY, two Asian equity funds that fell together. The boss asked on
2026-10-27 for "trend/momentum following". The ladder test (frozen 2026-10-31) checked only the ranking version, and the stop-loss
test (frozen 2026-11-08, commit 8cf1641) found that a volatility stop did not beat SPY and did not add value over C50.
This registration tests two other changes to the same rule, each aimed at a flaw seen in the live week:
1. **Trend filter** (trial 10): do not hold a top-ranked fund that is below its own 200-day average.
2. **Group cap** (trial 11): do not hold two top-ranked funds from the same region or asset class.

**Honesty about the order of events.** Both ideas came after the ladder result (no variant beat SPY, C50 included), after the stop
result and after a live loss. This is a third look at the same history, so a pass here is weak evidence, and the trial count below is
part of how to read it. The result for C50 is already known (about 10.1% a year versus SPY 9.4%; the 90% interval for the yearly gap
was -2.4 to +3.6 points). The reason for each variant (a trend exit, a diversification rule) was chosen from general practice and the
live week's holdings, not from tuning on the 25-year results.

## Data
`prices_daily_asof_2026-10-30.csv` (adjusted closes), the same file as the ladder and stop registrations, so every variant and the C50
baseline are compared on identical prices. Days after 2026-10-30, including the live loss, are NOT in this data.
SHA-256: dff1a011fee8a8f662d94d166f58c6ddd685a9d5ac819407e3970f1d2160b30c (checked on Ethan's file 2026-11-08).
Universe, eligibility and the 273-day signal rule: exactly as in the ladder registration (every ETF in the file except SH and SPY, so
SHY is both rankable and the parking fund; point in time; an ETF is eligible on a date only if it has a price on that date and 273
trading days earlier).

## The group table (for the cap)
Every ETF in the universe is assigned to one group (a region or asset class) in the file `portfolio/preregistrations/momentum_groups.json`,
written before any run and frozen with this file. The runner refuses to run if any ETF is missing from the table or if the file's hash
differs from the one below.
Groups file SHA-256: 6e8e70085285c0f66f1bb8ebdbd8d2b1ce7afd3ecd80e32b1ce99807e68c9df0
The table is built from the fund's name and mandate, not from its returns. It is not tuned.

## The variants (two, plus the baseline)
**Baseline C50 (already defined, not a new trial):** top 3 by the ladder signal (return from 273 to 21 trading days back), equal weight,
sleeve 50% of the account, the rest SPY; signals recomputed every 21 trading days; trades at the next day's close. Rerun inside this
registration so all three use the same code and dates.
**Trend filter (trial 10).** Same as C50, but at each rebalance a pick is bought only if its adjusted close on the signal day is
strictly ABOVE the average of its last 200 closes (that day included). A pick that fails is not replaced by the next ranked fund;
its slice of the sleeve goes to SHY until the next rebalance (0% cash on dates when SHY has no price, before 2002-07-30). The
check is made at rebalances only (no daily exit). A missing price in the 200-day window counts as failing.
**Group cap (trial 11).** Same as C50, but walk down the ranking (highest score first, ties by ticker) and take a fund only if no fund
from its group has been taken yet, until 3 funds are chosen. If fewer than 3 groups have eligible funds, the sleeve is split equally
among the funds taken. Weights, costs and rebalance as in C50.
**Not run under this registration:** a combined trend-plus-cap rule, other averages (such as 100 or 150 days), other cap sizes, a
daily exit, or any tuned setting. Each would be another trial and needs its own registration.

**Costs.** 5 basis points of traded value each way, on every leg (including the swap into SHY and out of it at the next rebalance).
Also reported with zero cost. The main result is the 5 basis point run.

**Number of trials.** The ladder used 8 variants on this data and the stop test 1 (trial 9). These are trials 10 and 11, so a pass by
either is read as weak evidence. Because two variants are tested, a pass by one of them alone is weaker than the same pass in a
single test; the report says so and does not select a "best of two."

## Account
Nothing changes in the live momentum account under this registration. If a variant passes and is later run live, that is a separate
decision at a scheduled rebalance (about 2026-12-02 is the next), human-confirmed; the nightly job never places orders.

## Evaluation
**Windows, ranges, statistics:** exactly as in the ladder and stop registrations (every rolling 35-trading-day window from the first
date with at least 20 eligible ETFs to the end; windows from 2016-01-04 reported separately; outcome percentiles, end-below-floor,
touch-floor at -10% and -50%, annualized return, volatility, worst fall, windows ahead of SPY).
**Descriptive only (no pass rule):** for the trend filter, how many rebalances parked at least one pick and how many picks were
parked; for the cap, how many rebalances had a changed pick; turnover for all three.
**Beats SPY?** Non-overlapping 21-day periods, paired difference (variant minus SPY), moving-block bootstrap (block 4 periods, 5,000
resamples, seed 20261109, 90% interval, same resampled blocks for every comparison). The statistic is the mean period difference x 12.
Computed for each variant minus SPY (primary) and each variant minus C50 (secondary).

## Pass rules (to be frozen at commit), applied to each variant separately
1. **Risk-eligible:** the -10% touch-floor rate is at most 2x SPY's own rate over the same dates AND the worst 7-week window is no
   worse than -50% (the same rule as the ladder and the stop test).
2. **Beats SPY** if the lower end of the 90% interval for the annualized difference against SPY is above 0.
3. **Adds value over C50** if the lower end of the 90% interval for the variant minus C50 is above 0 (secondary).
4. **Recommended for a live trial** only if risk-eligible AND beats SPY. A variant that lowers risk but does not beat SPY is reported
   as "lower risk, edge not shown", not as a recommendation.

## What each outcome would mean (a plan, not a result)
| Outcome (per variant) | Next step |
|---|---|
| Risk-eligible, beats SPY | Offer it to the boss as a live paper trial at a clean rebalance, human-confirmed. Weak evidence (third look, trial 10 or 11, two variants). |
| Risk-eligible, adds value over C50, does not beat SPY | Report "helps against C50, SPY is still the better holding". Not a recommendation. |
| Lower risk only, or no change | Report plainly; not shown to help. Keep the live account as it is. |
| Worse than C50 | Report it with the descriptive counts; do not tune the setting on this data. |

## Power and honesty
Yearly return gaps have 90% intervals about +/-3 points wide over 25 years, so a real edge of one or two points would not be
detected. A trend filter acts on the left tail, which is rare. Failing to pass means "not shown", not "proved useless". The 35-day
windows overlap heavily, so their percentages describe the history, not independent samples. A passing result from either variant
would be one of two tries on a history already used ten times; the interval does not account for that.

## Limits
- Third look at the same history (see the order-of-events note); the C50 result is already known.
- Survivorship and hindsight in the universe, no taxes, no shorting, no leverage, no individual stocks, as in the ladder.
- The group table is a judgment (which funds are the same bet). Funds can fall in the same group in one period and different ones in another.
- Fills at the close at the stated cost; SHY parking assumes the swap fills at the same close; cash-only buying power may delay it.
- Adjusted closes include dividends. Past 7-week outcomes are a history, not a forecast.

## Items to settle before freezing
1. The group table (`momentum_groups.json`): Claude drafts from the 47 tickers Ethan pastes; Ethan approves it; the file's SHA-256 is
   written into this file; both are committed together.
2. Ethan's approval of the definitions above (trend: 200-day average, strict greater-than, rebalance only, parked in SHY; cap: one fund
   per group). Ethan agreed to the definitions in chat on 2026-11-08; the table is the remaining judgment.
3. Recheck the SHA-256 of `prices_daily_asof_2026-10-30.csv` (done 2026-11-08; the runner checks it again).
4. Boss's answer on whether he meant trend-following with an exit rule (not a blocker for freezing).
5. Code: written and tested on made-up prices; rerun the tests on Windows before freezing.

## Reporting
Both variants, C50 and SPY, both date ranges, with and without costs, and the descriptive statistics are reported whether or not they
pass, and recorded as comments at the bottom of `momentum_variants.py` after the run.
