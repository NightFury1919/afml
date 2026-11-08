# portfolio/ — the paper-trading system and its pre-registered tests

Everything here uses simulated (paper) money. Nothing here is investment advice, and no result is evidence of a trading edge.

Run all the tests (about 90 seconds): `python -m pytest portfolio -q`

## Where things are

| Folder or file | What it holds |
|---|---|
| `*.py` (this folder) | The code, grouped by job in the next section. Scripts are run from this folder, for example `python daily_run.py`. |
| `tests/` | One `test_<module>.py` for each module, with hand-worked expected values. `conftest.py` makes the modules importable. |
| `preregistrations/` | The rules written down and committed **before** each test was run. Never edit a frozen one; add a dated amendment. |
| `results/` | What each pre-registered test produced (CSV, text, JSON). Each registered run refuses to run twice. |
| `docs/` | Design notes and protocols (`paper_protocol_v2.md` is the current one). |
| `accounts/` | One folder per paper account (`etf_only`, `spy_only`, `momentum`): its config, plans, orders and run logs. |
| `shadow/` | The shadow model: saved models and daily score logs. It never trades. |
| `plans/`, `orders/`, `runs/` | Plans, order ledgers and run reports from the first (single-account) setup. |
| `prices_daily_asof_<date>.csv` | Dated price snapshots. A new file is written each time; old ones are kept so any result can be reproduced. |
| `universe.txt`, `run_config.json` | The ETF list and the default account settings. |
| `nightly_all.bat`, `nightly_run.bat`, `*.ps1` | Windows launchers for the nightly job and for switching account keys. |

## The code, by job

**Get data**
`pull_prices.py` (ETF prices) · `update_prices.py` (append new days safely) · `pull_crypto_daily.py` (Kraken) ·
`pull_binance_daily.py` (Binance) · `stock_data_pull.py` (S&P 500 stocks, free data, coverage report) ·
`check_fractionable.py` (can every ETF be bought in dollar amounts?) · `panel_data.py` (load and reshape the price panel)

**Build the ETF model pieces** (pre-registered, no tuning)
`etf_classes.py` · `etf_features.py` · `etf_labels.py` · `panel_uniqueness.py` · `panel_time_decay.py` · `panel_purged_cv.py`
(these three adapt AFML Chapters 4 and 7 to a panel of many assets)

**Shadow model (forward test, no trading)**
`shadow_scoring.py` (daily scores) · `shadow_evaluate.py` (the formal rule, due about Oct 2029) · `nightly_run.py` (prices, then scores)

**Pre-registered checks that are already run**
`positive_control_etf.py` (can the pipeline find a planted edge?) · `noise_test_tier_a.py`, `noise_test_tier_b.py` (does it invent
an edge from noise?) · `real_history_check.py`, `real_history_check_wc.py` (does the model beat "hold the most volatile ETFs"?) ·
`crypto_family.py` (three-horizon crypto test) · `crypto_family_audit_AB.py`, `crypto_family_audit_C.py` (after-the-fact audits, not pre-registered) ·
`momentum_ladder.py` + `run_momentum_ladder.py` (momentum variants against SPY)

**Weights and orders**
`class_budgets.py` · `hrp_allocation.py` · `show_class_budgets.py` · `show_hrp.py` · `effective_breadth.py` · `report_effective_breadth.py` ·
`portfolio_construction.py` (scores to target weights) · `rebalance.py` (weights and holdings to orders) · `sizing.py` (whole shares)

**Run the paper accounts**
`daily_run.py` (check data, plan, scoreboard, report — **never sends orders**) · `execute_plan.py` (places orders only when a person
runs it, after checks) · `momentum_plan.py` + `momentum_daily.py` (the momentum account) · `paper_runner_v2.py` (dry-run planner; `paper_runner.py` is v1)

**Report and check**
`performance_page.py` (honest summary with risk next to return) · `spy_since.py` · `spy_window_stats.py` ·
`check_account.py`, `check_orders.py`, `paper_order_check.py` (early one-off Alpaca checks)

## Why the code is not in sub-folders yet

Scripts here import each other by plain name and look for their data next to themselves, and the nightly launchers call them by name.
Moving them into packages is a separate step that needs the imports and launchers changed together.
