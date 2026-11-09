# Pre-registration: stop-loss on the live momentum rule, "does a volatility-based stop help C50 against SPY?"

Status: FROZEN

Not frozen. Nothing below has been run on real prices. The code (`momentum_stop.py`, `run_momentum_stop.py`) and its tests exist and
pass on made-up prices only. The runner refuses to run unless the line above says FROZEN and the SHA-256 under Data matches the price file, and must
refuse to run twice. Freeze means: change `Status: DRAFT` to `Status: FROZEN`, commit and push this file, and only then run. Never
chain the run after the commit in the same command block. After freezing, the rules, the variant and the price file are fixed; any
change needs a new registration and counts as another trial.

## Why this check
The live momentum account (C50: top 3 ETFs from the ladder signal at 50% of the account, the rest in SPY; started 2026-11-02, bought
2026-11-03) had equity of $949.83 on 2026-11-06, about -5% in four trading days. From the weights in the daily report, roughly two
thirds of the loss came from EWT and EWY, two Asian equity funds that fell together. The account holds through falls by design:
it re-ranks every 21 trading days and has no exit rule between rebalances.
The boss asked on 2026-10-27 for "trend/momentum following". The ladder test (frozen 2026-10-31) checked only the ranking version.
This check asks one narrower question: if each fund in the sleeve carried a stop set from its own volatility, would the history
have been better than C50 without a stop, and better than SPY?

**Honesty about the order of events.** This question was raised after the ladder result (no variant beat SPY, C50 included) and after
a live loss. It is a second look at the same history, so a pass here is weaker evidence than a pass on fresh data, and the trial
count below is part of how to read it. The result for C50 without a stop is already known from the ladder run (about 10.1% a year
versus SPY 9.4%, 90% interval for the yearly gap -2.4 to +3.6 points).

## Data
`prices_daily_asof_2026-10-30.csv` (adjusted closes), the same file as the ladder registration, so the stop variant and the
C50 baseline are compared on identical prices. Days after 2026-10-30, including the live loss, are NOT in this data.
SHA-256: dff1a011fee8a8f662d94d166f58c6ddd685a9d5ac819407e3970f1d2160b30c (recorded in the ladder registration; recheck on the file
before freezing).
Universe, eligibility and the 273-day signal rule: exactly as in the ladder registration (every ETF in the file except SH and SPY, so
SHY is both rankable and the parking fund;
point in time; an ETF is eligible on a date only if it has a price on that date and 273 trading days earlier).

## The variant (one, plus the baselines)
**C50s** = the ladder's C50 rule (signal: return from 273 to 21 trading days back; top 3, equal weight; sleeve 50% of the account;
the rest SPY; signals recomputed every 21 trading days; trades at the next day's close) **plus a stop on each of the 3 sleeve funds.**
**Baselines (already defined, not new trials):** C50 without the stop (ladder result, rerun inside this registration so both use the
same code path and dates) and 100% SPY.

**The stop (one definition, not tuned).**
- **Barrier, fixed at entry:** `barrier = entry_price x (1 - 1.5 x sigma_d x sqrt(21))`, where `entry_price` is the close at which the
  fund was bought (the next close after the signal) and `sigma_d` is the Chapter 3 `get_daily_vol` of the fund's daily returns
  computed with `span0 = 100` (the repo's default) on that fund's own price history, using data up to and including the entry date (no later data).
  If the volatility is missing or zero the fund gets no stop.
  **Known quirk, kept on purpose:** the book's snippet looks back to the bar before "one day earlier", so on daily bars most of its
  returns span two days (Mondays one day). Its volatility reads about 1.34 times the true one-day volatility (measured on made-up
  data). The stop is therefore about 2.0 true daily volatilities times sqrt(21) wide, not 1.5. The function is used unchanged (book
  fidelity, one fixed distance, nothing tuned); the test suite records this ratio. Causality was checked: past values do not change
  when future prices change. The factor 1.5 is fixed here and will not be varied.
  The barrier does not move up as the price rises (it is not a trailing stop). There is no profit-taking barrier; the
  scheduled rebalance is the time limit.
- **Barrier reset:** at every scheduled rebalance every barrier is cleared and set again from the new entry price and the volatility at
  that date, including for a fund that is held over (bought again).
- **Trigger:** checked at every close after the entry close. If a fund's close is at or below its barrier, the stop is triggered.
- **Execution:** the fund is sold at the NEXT close, at that close's actual price (not the barrier price), so a gap below the
  barrier is a real loss in the test. If the next day is a scheduled rebalance, the rebalance takes over and no separate stop sale is made
  (counted as `superseded`). A trigger on the last day of data has no sale.
- **Proceeds:** bought into **SHY** at the same close and held until the next scheduled rebalance. SHY is in the price file; its first price is
  2002-07-30. If SHY has no price on that date (before then), the proceeds earn 0% as cash. The sleeve share stays 50% of the account while the money is parked.
- **No re-entry before the next scheduled rebalance** for a stopped fund. At the rebalance the signal is recomputed from scratch,
  so a stopped fund can be bought again if it ranks in the top 3.
- **Applies to the 3 sleeve funds only.** The SPY half has no stop.
- **Rebalance:** every 21 trading days as in the ladder; parked SHY is sold and the account is reset to 50% sleeve / 50% SPY.

**Costs.** 5 basis points of traded value each way, on every leg (selling the fund and buying SHY are two legs; selling into cash is
one leg; plus the rebalance trades). Also reported with zero cost. The main result is the 5 basis point run.

**Number of trials:** one new variant (C50s) on this data and signal. The ladder registration already used 8 variants on this
data, so this is the 9th trial (Ethan confirmed 2026-11-08 that no other variant has been registered on this data since 2026-10-31). No other stop distance, lookback, holdings count or parking rule will be run under this
registration; any other variant needs its own registration and counts as another trial.

## Account
Nothing changes in the live momentum account under this registration. If C50s passes and is later run live, that is a separate
decision at a scheduled rebalance (about 2026-12-02 is the next), and the nightly job would only FLAG a triggered stop in the
plan; selling stays a human-confirmed step (`execute_plan.py` is dry-run by default). Alpaca accounts under $2,000 have cash
buying power only, so the sell and the SHY buy may not both fill on the same day in practice; the test assumes they do.

## Evaluation
**Windows, ranges, statistics:** exactly as in the ladder registration (every rolling 35-trading-day window from the first date with
at least 20 eligible ETFs to the end of the data; windows from 2016-01-04 also reported separately; the same outcome
percentiles, end-below-floor, touch-floor at -10% and -50%, annualized return, volatility, worst fall, and windows ahead of SPY).
**Stop-specific, descriptive only (no pass rule):**
- number of stops triggered, and the share of 21-day periods with at least one;
- among stopped funds, the share whose price 21 trading days after the stop sale was HIGHER than the sale price, and the average
  change (this shows how often a stop locked in a loss the fund then recovered);
- turnover and total cost paid, compared with C50.

**Beats SPY?** As in the ladder: non-overlapping 21-day periods, paired difference (strategy minus SPY, per period), moving-block
bootstrap (block = 4 periods, 5,000 resamples, seed 20261108, 90% interval, same resampled blocks for all comparisons). The
statistic is the mean period difference x 12. Computed for C50s minus SPY (primary) and C50s minus C50 (secondary).

## Pass rules (to be frozen at commit)
1. **Risk-eligible:** the -10% touch-floor rate is at most 2x SPY's own rate over the same dates AND the worst 7-week window is no
   worse than -50% (the same rule as the ladder).
2. **C50s "beats SPY"** if the lower end of the 90% interval for the annualized difference against SPY is above 0.
3. **The stop "adds value"** if the lower end of the 90% interval for C50s minus C50 is above 0. This is a secondary result.
4. **Recommended for a live trial** only if C50s is risk-eligible AND beats SPY (rule 2). A stop that lowers risk but does not beat
   SPY is reported as "lower risk, edge not shown", not as a recommendation.

## What each outcome would mean (a plan, not a result)
| Outcome | Next step |
|---|---|
| Risk-eligible, beats SPY | Offer C50s to the boss as a live paper trial at a clean rebalance date, with human-confirmed sells. Treat as weak evidence (second look, at least 9 trials). |
| Risk-eligible, adds value over C50, does not beat SPY | Report "the stop helps, SPY is still the better holding". Not a recommendation. |
| Lower risk but not risk-eligible, or no change | Report plainly; the stop is not shown to help. Keep the live account as it is. |
| The stop worsens results (sells then rebounds) | Report it with the share of stops that rebounded; do not tune the distance on this data. |

## Power and honesty
Return differences between a stop variant and its own baseline are small and noisy; about 25 years gives roughly 300
non-overlapping 21-day periods, so a wide interval is expected and failing to pass means "not shown", not "proved useless".
A stop acts on the left tail, which is rare, so its effect on averages may be too small to detect even if it reduces the worst
falls. AFML Chapter 13 notes that no stop distance is optimal when the price follows a random walk and that rules tuned on one
history overfit; this is why the distance is fixed in advance and only one is run.
The 35-day windows overlap heavily, so their percentages describe the history, not independent samples.

## Limits
- Second look at the same history as the ladder (see the order-of-events note above); the C50 baseline result is already known.
- Survivorship and hindsight in the universe, no taxes, no shorting, no leverage, no individual stocks, as in the ladder.
- Fills at the close at the stated cost; the stop sells at the next close, but a real order could fill at a different price.
- SHY parking assumes the swap fills at the same close; cash-only buying power may delay the buy in practice.
- Adjusted closes include dividends. Past 7-week outcomes are a history, not a forecast.

## Items to settle before freezing
1. ~~Defaults~~ Approved by Ethan 2026-11-08: barrier factor 1.5 and sqrt(21) scaling; fixed barrier, reset at each rebalance; sell at the
   next close at its actual price; proceeds to SHY (0% cash before 2002-07-30); no re-entry before the next rebalance; stop on the
   sleeve funds only; one variant only.
2. ~~Volatility function~~ Approved: the book's `get_daily_vol` (`ch03\labeling\triple_barrier.py`, `span0 = 100`) unchanged, quirk and all.
3. ~~SHY in the price file~~ Done: first price 2002-07-30.
4. ~~Trial number~~ Settled: 9th.
5. Recheck the SHA-256 of `prices_daily_asof_2026-10-30.csv` (the runner also checks it).
6. Boss's answer on whether he meant trend-following with an exit rule (message drafted 2026-11-08, not confirmed sent). Not a blocker
   for freezing; if his answer changes the question, this registration may no longer be the one he wants.
7. ~~Code~~ Written and tested on made-up prices (32 pass on Ethan's machine, mlfinlab env).

## Reporting
C50s, C50 and SPY, both date ranges, with and without costs, and the stop-specific descriptive statistics are reported whether or
not they pass, and recorded as comments at the bottom of the new module after the run.
