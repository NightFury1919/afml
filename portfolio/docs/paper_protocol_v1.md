# Paper-trading protocol v1 (dry run)

Status: **draft settings for the dry-run runner `paper_runner.py`. No orders are sent by this version.**
The settings live in `PROTOCOL` in `paper_runner.py`; each plan file records the settings it was made under.

## Settings (defaults, none of them tuned on results)
| Setting | Value | Why |
|---|---|---|
| Universe | the 48 ranked ETFs; SH is never scored or bought | frozen design, `etf_pipeline_design.md` |
| Features and labels | the five fixed features, relative 21-day label | frozen design |
| Model | logistic regression, C = 1, refit from scratch each run on every resolved label | baseline used in the positive controls |
| Selection | top 20% by score (10 of 48), equal weight | same top-quintile statistic as the positive control; a default, not a finding |
| Position cap | none | 10 equal positions are already 10% each |
| Exposure | 1.0 (fully invested) | the confidence ramp and zero-exposure point are not decided |
| Cadence | daily | matches the "daily rebalancing" assignment; the horizon is 21 days, so most days change little |
| Order type | fractional dollar amounts, minimum order $1 | Alpaca fractional orders |
| Ranking | pooled 48 | within-class waits for the v1.2 comparison and the class budgets |

## Not in this version
Time decay, trading costs, shorting, sample weights, DSR and PBO, reading holdings from Alpaca, sending orders.

## Open decisions (these change the settings above)
- Daily vs 21-day rebalancing (boss).
- Pooled vs within-class ranking, and the class budgets (v1.2 result, then boss).
- Exposure ramp and zero-exposure point (confidence prototype).
- Long-only vs long/short (needs the $2,000 margin floor).
- Account holder and starting amount (boss).

## What a "plan" is
`plans/plan_<last price date>.json`: the scores, weights, the orders the runner would place, the cash left, and the settings used. A plan is written once per price date and never overwritten.
