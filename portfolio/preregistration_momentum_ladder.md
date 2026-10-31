# Pre-registration: momentum risk ladder on ETFs, "how much extra risk buys how much extra upside, against SPY?"

Status: **DRAFT, not frozen. Nothing below has been run.** Items marked **OPEN** need Ethan's confirmation before this
file is committed. To freeze: fill in the SHA-256 line under Data, change this line to exactly `Status: FROZEN`, then commit
this file together with the price file. The runner `run_momentum_ladder.py` (built and tested 2026-10-31, not run on real
prices) refuses to run unless the line says FROZEN and the SHA-256 matches the price file, and it refuses to run twice.

## Why this check
The boss wants strategies that can grow the $1,000 much faster than the market, with 2x, 5x or 10x mentioned as goals,
and wants to decide how aggressive to be. SPY alone turned $1,000 into about $1,012 over the last seven weeks
(prices through 2026-10-27). Across every 7-week window since 2016, SPY's median was +2.8%, its 95th percentile +9.7%,
its best +28.6%, and 0% of windows ended at 2x. So the goals sit far outside what the market has done.
The semester ends about 2026-12-19 (a Saturday; last trading day Friday 2026-12-18). From 2026-10-29 that is about
36 trading days, close to the 35-day window used below, so the 7-week table is the right yardstick. A new paper account
started in early November would have fewer, roughly 30 to 34, and at most one or two 21-day rebalances.
This check asks a narrower, answerable question: if we add a momentum sleeve on top of SPY, how does the range of
7-week outcomes change as the sleeve gets bigger, and at what size does the chance of hitting the $900 floor become too high?

## Data
The newest `prices_daily_asof_*.csv` at freeze (2026-10-30 or later; adjusted closes).
SHA-256: FILL_IN_AT_FREEZE
Universe: every ETF in the file
except **SH** (inverse fund) and **SPY** (it is the core holding and the benchmark). Point in time: an ETF is eligible on a
date only if it has a price on that date and 273 trading days earlier. No ETF is added or dropped by hindsight.
**Not tested:** a theme sleeve (for example semiconductors or AI). The file has no such ETF, and choosing one now would
be picking a winner with hindsight.

## The strategies (8 variants plus the benchmark)
**Signal (one definition, not tuned).** Momentum = total return from the close 273 trading days ago to the close 21 trading
days ago (the usual "12 months, skipping the latest month"). Rank eligible ETFs; ties broken by ticker name.

**Sleeves.**
- **D (diversified):** hold the top 10 by momentum, equal weight.
- **C (concentrated):** hold the top 3 by momentum, equal weight.

**Rungs (sleeve share of the account; the rest is SPY).** 10%, 25%, 50%, 100%. That gives D10, D25, D50, D100, C10, C25, C50, C100.
**Benchmark:** 100% SPY (Account B).

**Timing.** Signals and weights are recomputed every 21 trading days on the close. To mirror how the nightly plan runs, trades
happen at the next day's close, so the first return earned is two days after the signal date. Between rebalances weights drift.
At each rebalance the sleeve/SPY split is reset to the rung's target.

**Costs.** 5 basis points (0.05%) of traded value each way. Results are also reported with zero cost.

Number of trials: 8 variants on one signal. No other variant (other lookback, other number of holdings, a cash rule when
momentum is negative, leverage) will be run on this data under this registration; any further variant needs its own
registration and counts as another trial.

## Account
If a rung is later run live, it runs in a **new paper account funded with $1,000**, the same size as Accounts A and B, so
all three compare in dollars as well as percentages. The boss answered on 2026-10-31: he will accept losing $500 (50%) to win big, and his
second goal is to beat the S&P 500. So the live equity floor for this account is **$500** (set `--equity-floor 500` and
check that the daily report banner shows it). The -10% level is kept only as a comparison yardstick against SPY. The $2,000 AFML2K account is not used. Alpaca support confirmed that accounts under $2,000
have cash buying power only (no margin, no shorting), so a rebalance that buys with the proceeds of same-day sales should
be watched for rejected orders (sell first, buy once the cash has settled in the account's buying power).

## Evaluation
**Windows.** Every rolling 35-trading-day window (7 weeks), started on each day, from the **first date at least 20 ETFs
are eligible** (set by this rule, reported in the output) to the end of the data. Also reported separately: windows
starting 2016-01-04 or later, to compare directly with the SPY numbers above.

**Ranges.** The rules below use the full range (first 20-ETF date to the end of the data) with 5 bps costs. The
2016-01-04 range and the zero-cost versions are reported for comparison only.

**Statistics, per strategy and for SPY.**
- Window outcome: worst, 5th percentile, median, 95th percentile, best.
- Share of windows ending above 0%, +10%, +20%, +50% and +100%.
- **End-below-floor:** share of windows ending below -10% ($900).
- **Touch-floor:** share of windows in which the account was at or below -10% at ANY close inside the window.
  Also reported: the same at the boss's hard limit of -50% ($500).
- Full-period annualized return, volatility, worst fall (peak to trough).
- Also reported: how many windows ended better than SPY over the same dates, and the average amount.

**Beats SPY?** Using non-overlapping 21-day periods, the paired difference (strategy minus SPY, per period) gets a
moving-block bootstrap (block = 4 periods, 5,000 resamples, seed 20261028, 90% interval, same resampled blocks for both).
The statistic is the mean period difference x 12 (a simple annualization, not a compounded one).

## Pass rules (frozen at commit)
1. **A strategy is risk-eligible** if its -10% touch-floor rate is at most **2x SPY's own rate** over the same dates
   (OPEN, Ethan to confirm: SPY's rate since 2016-01-04 was 6.6%, so the limit would be about 13%; it is recomputed on the
   full range of the frozen file) AND its worst 7-week window is no worse than **-50%** (the boss's stated limit).
   The 2x rule is deliberately stricter than the boss's $500 tolerance: it keeps the pick from being a coin-flip on
   -10% drops. A looser rule can be chosen later, but only as a separately registered variant.
2. **A strategy "beats SPY"** if the lower end of the 90% interval for the annualized return difference is above 0.
3. **The recommended rung** for each sleeve is the largest rung that is risk-eligible AND has a median 7-week window
   above SPY's median. If none, the recommendation is "no rung".

## What each outcome would mean (a plan, not a result)
| Outcome | Next step |
|---|---|
| A rung is risk-eligible, beats SPY and has a higher median | Offer it to the boss as a third paper account, same tooling, same safeguards as Accounts A and B. |
| Risk-eligible but does not beat SPY (interval spans 0) | Not shown. Report to the boss that more risk did not buy a demonstrable edge; SPY stays the better holding. |
| Higher rungs raise upside but fail the risk rule | Show the table so the boss can choose a different loss limit knowingly. No rung is recommended. |
| No variant reaches 2x in any window | Expected. Report plainly: the 2x to 10x goals need a different kind of bet than momentum on ETFs. |

## Power and honesty
About 25 years gives roughly 300 non-overlapping 21-day periods. Return differences between momentum and SPY are
noisy, so the interval in rule 2 will be wide and only a large gap will pass. Failing to pass means "not shown", not
"proved useless". The 35-day windows overlap heavily, so their percentages describe the history, not independent samples.

## Limits
- **Survivorship and hindsight in the universe:** the ETFs are ones that exist today, and many launched after 1999. This
  understates the risk of ETFs that closed, and the early years have fewer ETFs to choose from.
- No taxes, no shorting, no leverage, no options, no individual stocks, no crypto. Individual stocks wait for a
  survivor-free data source (**OPEN:** the Biola library question).
- The ETFs in the file are broad, mostly diversified funds, so even the concentrated sleeve (3 ETFs) is far less
  concentrated than a few single stocks. It cannot reproduce Nvidia-style outcomes.
- Adjusted closes include dividends. Trades are assumed to fill at the close at the stated cost.
- Past 7-week outcomes are a history, not a forecast.

## OPEN items before freezing
1. ~~The biggest drop the boss accepts.~~ Settled 2026-10-31: -50% ($500). Still OPEN: keep the 2x-SPY touch rule in rule 1?
2. ~~Which account this strategy would replace.~~ Settled 2026-10-28: a NEW paper account starting at $1,000 (the $2,000 AFML2K account is left alone).
   Semester end is settled: about 2026-12-19. See "Account" below.
3. Confirm the SPY comparison start and the 20-ETF start rule are acceptable.
4. Fill in the SHA-256 of the price file (`Get-FileHash <file> -Algorithm SHA256`) and set `Status: FROZEN` at commit.

## Reporting
All 8 variants, the SPY benchmark, both date ranges, with and without costs, are reported whether or not they pass, and
recorded as comments at the bottom of `momentum_ladder.py` after the run.
