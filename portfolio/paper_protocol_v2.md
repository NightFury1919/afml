# Paper-trading protocol v2 (dry run): S&P 500 sleeve against the QF sleeve

Status: **draft settings for the dry-run runner `paper_runner_v2.py`. No orders are sent by this version.**
v2 replaces the model-based top-10 selection of v1 (`paper_protocol_v1.md`). v1 stays in the repo unchanged.
The settings live in `PROTOCOL` in `paper_runner_v2.py`; each plan file records the settings it was made under.

## Why v2
The pooled and within-class real-history checks showed no demonstrated predictive edge, so there is no model in v2.
On 2026-10-13 the boss said to split the investment and use the S&P 500 as the baseline to compete against the QF
strategy. The account is two sleeves.

| Sleeve | Share of equity | What it holds |
|---|---|---|
| SPY (baseline) | `spy_share` | SPY only, bought and held |
| QF | `1 - spy_share` | the 48 ETFs, spread by a class-budget rule |

SPY can appear in both sleeves, because it is also one of the 48 US-equity ETFs. The plan reports the combined
weight, and the scoreboard keeps the sleeves separate.

## Settings
**Every value below is a placeholder, not a finding. None was tuned on results.**

| Setting | Default | Whose decision | Note |
|---|---|---|---|
| `spy_share` | 0.5 | boss | split ratio not stated on 2026-10-13 |
| `qf_rule` | `size` | Ethan | `size` = plain equal weight over all ETFs (neutral); `equal`, `inverse_vol`, `hrp` also wired in |
| `max_weight` | none | boss | per-ETF cap, **as a share of the QF sleeve** (not of the account); applied after exposure |
| `exposure` | 1.0 | Ethan | QF sleeve only; the rest of that sleeve stays cash |
| `rebalance_every` | 1 | Ethan | trading days; 1 = daily, 21 = monthly |
| `window` | 252 | convention | trailing days for `inverse_vol` and `hrp`, and the "full history" test for every rule |
| `mode`, `min_notional` | fractional, $1 | | Alpaca fractional orders, as in v1 |
| Universe | the 48 ETFs; SH never held | frozen design | an ETF joins once it has `window + 1` prices |
| Long/short | long-only | Ethan | shorting needs the $2,000 margin floor |

Boss decisions still open: split ratio, what "compete" means (final dollars or drawdown), whether the $10k goal and
cut points apply to the total or the QF sleeve, drawdown tolerance, the per-ETF cap and loss limits.

## What the plan is
`plans/plan_v2_<last price date>.json`: sleeve dollars, QF weights, combined weights, the orders it would place, the
cash left and the settings used. One per price date, never overwritten.

## What the scoreboard is
`scoreboard(prices, start, protocol, equity)`: a replay of both sleeves from a start date, computed from the price
file alone. Weights on each day use only prices up to that day and earn the next day's return. Columns: dollars,
cumulative return, the QF-minus-SPY difference, and drawdown for each sleeve. It reports both final dollars and
drawdown so either reading of "compete" can be scored without changing the code.

Limits stated up front:
- No trading costs or spreads, and no interest on cash.
- Prices are adjusted closes, which can be revised after the fact (dividends). A replay is recomputed from the newest
  snapshot, so old rows can move slightly. The saved plan files are the record of what was decided each day.
- **Do not use a replay over past dates to choose a rule, a cap or a split.** That is fitting on one history, which
  the project has already done four times. Use the scoreboard to watch the dry run go forward from a start date.

## Not in this version
The predictive model and its scores, time decay, trading costs, shorting, reading holdings from Alpaca, sending orders.
Optional later: log the old model's daily scores beside the plan as a free forward record.
