# AFML: *Advances in Financial Machine Learning* in Python

A teaching-oriented implementation of the methods in Marcos López de Prado's *Advances in Financial Machine Learning*
(Wiley, 2018), one folder per chapter, plus a paper-trading system that puts those methods through a pre-registered
forward test.

**Everything here uses simulated (paper) money. Nothing in this repository is investment advice, and no result below
is evidence of a trading edge.**

## Start here

1. Pick a chapter from the table below.
2. Open that chapter's `README.md`, then its notebook (`chapter_N_*.ipynb`). Every chapter goes in the same order:
   plain-English idea, then the math, then the code, then plots.
3. Run the chapter's tests to see the code reproduce the book's hand-worked examples.

Chapters use real data wherever possible (S&P 500 futures, Bitcoin trades, and others). Synthetic data appears only in
unit tests where the right answer is known in advance.

## Set up

- Python 3.10 in a **dedicated conda environment** (developed on Python 3.10.20, pandas 1.5.3, numpy 1.23.5). Do not
  use a newer base environment: files saved with one numpy version may not load in another.
- Install what a chapter needs from its own `requirements.txt`, for example `pip install -r ch03/requirements.txt`.
- Scripts find their own folders automatically. Notebooks cannot, so each notebook's first code cell has one line,
  `AFML_ROOT = r'C:\ws\AFML'`. Edit that line to wherever you cloned the repository.

## Chapter map

| Folder | Book chapter | Main package inside |
|---|---|---|
| `ch02` | Financial Data Structures | `bars/`, `multi_product/` |
| `ch03` | Labeling | `labeling/` |
| `ch04` | Sample Weights | `sample_weights/` |
| `ch05` | Fractionally Differentiated Features | `frac_diff/` |
| `ch06` | Ensemble Methods | `ensemble/` |
| `ch07` | Cross-Validation in Finance | `cross_validation/` |
| `ch08` | Feature Importance | `feature_importance/` |
| `ch09` | Hyper-Parameter Tuning with Cross-Validation | `hyper_parameter_tuning/` |
| `ch10` | Bet Sizing | `bet_sizing/` |
| `ch11` | The Dangers of Backtesting | `backtest_dangers/` |
| `ch12` | Backtesting through Cross-Validation | `cpcv/` |
| `ch13` | Backtesting on Synthetic Data | `otr/` |
| `ch14` | Backtest Statistics | `backtest_statistics/` |
| `ch15` | Understanding Strategy Risk | `strategy_risk/` |
| `ch16` | Machine Learning Asset Allocation | `cla/`, `hrp/`, `monte_carlo/`, `data_loader/` |
| `ch17` | Structural Breaks | `structural_breaks/` |
| `ch18` | Entropy Features | `entropy_features/` |
| `ch19` | Microstructural Features | `microstructural_features/` |
| `ch20` | Multiprocessing and Vectorization | `multiprocessing_engine/` |
| `ch21` | Brute Force and Quantum Computers | `portfolio_trajectory/` |
| `ch22` | High-Performance Computing supplements | `hpc_supplements/` |

## Other folders

| Folder | What it is |
|---|---|
| `input_data/` | Datasets shared by several chapters. They are kept here so no chapter copies them. |
| `utils/` | Shared helpers, such as the multiprocessing engine. |
| `pipeline/` | A research pipeline for crypto data (ingestion, diagnostics, calibration, edge tests). |
| `portfolio/` | The paper-trading system and its forward test (see below). It has its own `README.md`. |
| `ch02ts/`, `ch03ts/` | TypeScript versions of the chapter 2 and 3 code. |
| `archive/` | Retired scripts and old run logs, kept for the record. Nothing depends on them. |

## Run the tests

From the repository root, run one chapter at a time. Use `python -m pytest`, not bare `pytest`:

```powershell
conda activate mlfinlab
python -m pytest ch03 -q
```

Run chapters separately rather than in one command, because pytest can stop with an "import file mismatch" error when
several chapters share test file names. As of November 2026, every chapter from `ch02` to `ch22` passes this way, 719
tests in all (3 are skipped by design: a multithreading check that is turned off on Windows). `ch09` is the slowest, at
about 1.5 minutes. Each chapter's own `README.md` has more detail.

The paper-trading system has its own suite, also run from the repository root:

```powershell
python -m pytest portfolio -q
```

This runs 848 tests (847 pass, 1 is skipped by design) in about 2.5 minutes. The tests were written before the code,
with known expected values, and were also checked by deliberately breaking the code ("mutation checks") to confirm the
tests notice.

## The paper-trading test (`portfolio/`)

Three Alpaca **paper** accounts run side by side for comparison only:

- **Account A:** a 44-ETF portfolio. Weights come from an inverse-volatility rule inside asset classes, with a 10% cap
  per position and a monthly rebalance (21 trading days).
- **Account B:** SPY only, the baseline.
- **Momentum account** (`accounts/momentum`): an experiment with half the account in the top 3 ranked funds and the
  rest in SPY. It has no demonstrated edge. In the ETF history tests, none of 8 momentum variants beat SPY (about 9.4% a year versus 9.6% to
  10.4%).

The pipeline is split so a mistake cannot place an order by accident. `daily_run.py` checks the price data, makes the
plan, updates the scoreboard and writes a report, and it **never sends orders**. `execute_plan.py` places orders only
when a person runs it, after pre-trade checks (stale-plan guard, holdings check, equity floor, sells before buys, and an
order ledger written before the first order goes out).

### How to judge whether it works

- **Frozen in advance.** The shadow model (an ETF scoring model that does not trade) was pre-registered on 16 Oct 2026,
  with its refit schedule and evaluation rule written down before any result existed. The live portfolio's rules were
  fixed before its start on 26 Oct 2026. Changes are made only through dated amendments, and logs are never
  overwritten.
- **Honest yardsticks.** SPY and SHY (a cash-like Treasury ETF) are shown beside every result.
- **Risk first.** Reports show volatility, worst fall from a peak, and distance from the loss floor next to return, and
  refuse to print an annualized return until there are 126 trading days of history.
- **Controls.** Noise tests check that the testing machinery does not invent an edge where none exists. A positive
  control checks that it can find a planted one.
- **A scheduled evaluation.** The formal check of the shadow model needs 36 resolved 21-trading-day periods, so it
  happens around October 2029. The rule is a pre-registered bootstrap interval (96.7%, 20,000 resamples). Anything
  short of a pass means "edge not shown".

### What the data shows so far

**Live paper test:** started 26 Oct 2026, so there is no meaningful history yet. Any number from the first weeks is
noise.

**Historical replay** (the same rules run on past prices, 4 Jan 2016 to 19 Oct 2026). The rules and the ETF list were
chosen with hindsight, and there are no trading costs or taxes, so this shows how the rules behaved, not what they will
do:

| | Portfolio (44 ETFs) | SPY | SHY |
|---|---|---|---|
| Annualized return | +6.3% | +15.1% | +1.6% |
| Volatility | 8.4% | 17.7% | 1.5% |
| Worst fall from a peak | 20.6% | 33.7% | 5.7% |

The portfolio earned less than SPY with less volatility and a smaller worst fall. It does not beat the market, and this
project does not claim it does. The aim is a lower-risk portfolio that still grows, tested honestly.

### Run it

```powershell
conda activate mlfinlab
cd portfolio
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
