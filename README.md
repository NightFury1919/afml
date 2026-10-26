# AFML: *Advances in Financial Machine Learning* in Python, plus a live paper-trading test

A teaching-oriented implementation of the methods in Marcos López de Prado's *Advances in Financial Machine Learning*,
and a paper-trading system (Alpaca) that puts those methods through a pre-registered forward test.

**Everything here uses simulated (paper) money. Nothing in this repository is investment advice, and no result below
is evidence of a trading edge.**

## What is in the repository

| Part | What it is |
|---|---|
| `ch02` to `ch09`, `ch16` | One folder per book chapter: implementation, pytest tests with hand-worked expected values, a script and a notebook that run on real data, and a README. Covers bars, labeling, sample weights, fractional differentiation, ensembles, purged cross-validation, feature importance, hyper-parameter tuning, and hierarchical risk parity. |
| `portfolio/` | The paper-trading system: price updates and health checks, a rules-based planner, a scoreboard, guarded order execution, and nightly reports. |
| `utils/` | Shared helpers (for example multiprocessing). |

Each chapter follows the same order: plain-English idea, then the math, then the code, then plots.

## The paper-trading system

Two Alpaca **paper** accounts run side by side for comparison only:

- **Account A:** a 44-ETF portfolio. Weights come from an inverse-volatility rule inside asset classes, with a 10% cap per
  position, 252-day window, monthly rebalance (21 trading days). Four ETFs are excluded (three that issue K-1 tax forms,
  and SPY).
- **Account B:** SPY only, the baseline.

The pipeline is deliberately split so that a mistake cannot place an order by accident:

1. `daily_run.py` checks the price data, makes the plan, updates the scoreboard and writes a report. It **never sends orders**.
2. `execute_plan.py` places orders only when a person runs it, after pre-trade checks.

Safeguards, each added after a real problem or a review of what could go wrong, and each covered by tests:

- Stale-plan guard: refuses a saved plan that no longer matches the configuration.
- Holdings check: refuses to trade if the account does not match what the plan assumes (within $1 of tolerance).
- Equity floor: refuses to place orders if account equity is below a set loss floor ($900 on a $1,000 account).
- Sells first, with a wait for fills, then buys. If cash falls short by more than 2%, execution stops instead of guessing.
- An order ledger is written before the first order is sent, and buys are polled to a final status afterwards.
- Plan files are never overwritten.

### Testing

The paper-trading modules have 212 automated tests (`daily_run`, `execute_plan`, `paper_runner_v2`), written before the
code and with known expected values. The tests are also checked by deliberately breaking the code ("mutation checks") to
confirm the tests notice. Full `pytest -v` output is embedded at the bottom of each module.

## How to judge whether this works

The forward test is set up so the result can be trusted whichever way it goes:

- **Frozen in advance.** The portfolio rules were fixed before the test started (26 Oct 2026). Changing them mid-test
  would be recorded as a new trial, not hidden.
- **Honest yardsticks.** SPY ("just buy the market") and SHY (a cash-like Treasury ETF) are shown beside every result.
- **Risk first.** `performance_page.py` reports volatility, worst fall from a peak, and distance from the loss floor next
  to return, and it refuses to print an annualized return until there are 126 trading days of history.
- **Controls.** A positive control (a planted signal the pipeline must find) and negative controls (noise it must not
  find anything in) check that the testing machinery itself is sound.
- **A scheduled evaluation.** A frozen model is scored out of sample at a date set in advance (2029), with the number of
  trials counted so the result can be adjusted for multiple testing (deflated Sharpe ratio, Chapter 14 of the book).

## What the data shows so far

**Live paper test:** started 26 Oct 2026, so there is no meaningful history yet. Any number from the first weeks is noise.

**Historical replay** (the same rules run on past prices, 4 Jan 2016 to 19 Oct 2026). The rules and the ETF list were
chosen with hindsight, and there are no trading costs or taxes, so this shows how the rules behaved, not what they will do:

| | Portfolio (44 ETFs) | SPY | SHY |
|---|---|---|---|
| Annualized return | +6.3% | +15.1% | +1.6% |
| Volatility | 8.4% | 17.7% | 1.5% |
| Worst fall from a peak | 20.6% | 33.7% | 5.7% |

The portfolio earned less than SPY with less volatility and a smaller worst fall. It does not beat the market, and this
project does not claim it does. The aim is a lower-risk portfolio that still grows, tested honestly.

## Run it

Requires Python 3.10 (developed on 3.10.20, pandas 1.5.3, numpy 1.23.5). Use a dedicated conda environment.

```powershell
conda activate mlfinlab
cd portfolio
pytest                                    # run the tests
python daily_run.py --help                # health check, plan, scoreboard, report (no orders)
python performance_page.py --folder . --config accounts\etf_only\run_config.json --out performance --replay-start 2016-01-04
```

Alpaca keys are read from the environment variables `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` and are never stored in
the repository. Use paper keys.

## Limitations

- Paper fills are not real fills: no slippage, and no partial fills as a real account would see.
- One history, with a universe picked in hindsight. Results from it are optimistic by construction.
- No tax modelling.
- Not a recommendation to trade anything.

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
