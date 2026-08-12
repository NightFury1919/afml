# Pipeline — Capstone Integration Layer

## What this is

Unlike the chapter folders, this isn't a book chapter — it's a capstone
orchestration layer chaining real, already-tested code from across the repo
into one continuous flow: real feature table → multi-trial purged
cross-validation → PBO/DSR overfitting diagnostics → bet-sizing signal →
plain-English evidence report. The end goal (per project scope, 2026-08-12)
is a tool a college trading club can point at real market data and get an
honest, statistically rigorous assessment back — not a black-box buy/sell
signal.

**No new AFML formula is implemented here.** Every calculation delegates to
existing, real-machine-confirmed chapter modules:

| Stage | Real module |
|---|---|
| Cross-validation | `ch07/cross_validation/purged_kfold.py` (`PurgedKFold`) |
| Bet sizing | `ch10/bet_sizing/bet_sizing.py` (`getSignal`) |
| Overfitting probability | `ch11/backtest_dangers/pbo.py` (`pbo`, `sharpe_ratio`) |
| Deflated Sharpe | `ch14/backtest_statistics/backtest_statistics.py` (`deflated_sharpe_ratio`) |

This file (`orchestration/stages.py`) is pure glue, plus one genuinely new
piece: `orchestration/report.py`, which synthesizes those real statistics
into a plain-English writeup.

## Structure

```
pipeline/
├── run_pipeline.py              driver script, real data
├── README.md
├── requirements.txt
└── orchestration/
    ├── __init__.py
    ├── stages.py                 orchestration: load → CV → PBO/DSR → signal
    ├── report.py                 new: plain-English evidence report
    └── test_orchestration.py     20 tests (structural, not exact-value)
```

No paired notebook yet — will be added once Phase 1's design is validated
against real-machine results.

## Phase 1 scope (this delivery)

Wires the orchestration + report layer to the **existing static** real
March 2026 BTC/TUSD artifacts (`ch07_training_table_enriched.csv`,
`ch03_events.csv` — both already real-machine confirmed by Ch04/05/19).
Phase 2 will replace `stages.load_enriched_table()` with a live Binance
pull feeding the same downstream bar/feature/label pipeline (Ch02–05,
Ch17–19) — everything past that one loading function is already asset- and
data-source-agnostic.

## Design: why 3 trials, why PurgedKFold reused across all of them

`orchestration/stages.default_trials()` defines 3 classifier configurations
(2 RandomForest depths + Ch07's own BaggingClassifier-with-avgU
convention) — a small, honest trial set, not a hyperparameter fishing
expedition. Every trial is run through **the same PurgedKFold configuration**
(identical `n_splits`/`t1`/`pctEmbargo`, which is deterministic — no
randomness in fold assignment), so every trial's stitched out-of-sample PnL
series covers an identical set of timestamps. This is a hard requirement of
Ch11's `pbo()`/`cscv()`, which needs a true `(T, N)` matrix with synchronous
rows across trial columns (see that module's own docstring).

## ⚠️ Known limitation — read before trusting any report this produces

**The first real run of this pipeline produced a DSR of 0.9995 and "high
confidence" of edge — and this should NOT be trusted.** It directly
contradicts this project's own convergent, honestly-established finding of
NO exploitable signal on this same BTC/TUSD dataset, verified independently
five separate ways (Ch11 PBO≈0.83, Ch12 CPCV all-negative, Ch13 O-U≈random
walk, Ch14 DSR 0/5 paths survive, Ch15 P[fail]≈0.45–0.47).

The likely cause: with only **T=87 observations** and **3 trials**, the
PBO/DSR estimate here is far too small a sample to be a reliable estimator.
This project's own `ch11/backtest_dangers/pbo.py` test suite documents this
exact failure mode directly: a single PBO draw for a genuinely zero-edge
strategy can range **~4%–99%** purely from sampling noise (measured over 40
seeds in that chapter's own tests). A "great-looking" 4% PBO on a small
sample is not evidence of edge — it's within the expected noise band for
*no* edge at all.

`report.build_report()` now surfaces this explicitly: when `T < 250` or
`n_trials < 5`, the report emits a `SAMPLE SIZE WARNING` and reports
confidence as `UNRELIABLE` rather than a false `high`/`moderate`/`low`
verdict. **These thresholds are a starting heuristic, not a validated
statistical cutoff** — worth revisiting once real-machine results come
back, and before this is ever put in front of an actual trading club
member.

### What Phase 1b should address before this is trustworthy
- A meaningfully larger real out-of-sample count than 87 events (this
  dataset's own size limit — likely needs either a longer real history or
  coarser events, a genuine open design question, not yet resolved)
- More trial configurations (5+) so DSR's multiple-testing correction has
  real information to work with
- Reconciling this orchestration's simplified `pnl = ret * pred` proxy
  against Ch14's actual, more rigorous backtest logic (which is why Ch14's
  own DSR came out near 0, not near 1) — the discrepancy between the two
  needs to be understood before either is trusted for a live report

## Real-data confirmation status

**Sandbox pre-check only, NOT YET real-machine confirmed.** Sandbox:
Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1, scikit-learn
1.8.0 — all newer than mlfinlab's pinned versions. 20/20 tests passed in
sandbox; pipeline ran end-to-end without error. Real-machine confirmation
(mlfinlab conda env) still needed — see commands below.

```powershell
conda activate mlfinlab
cd C:\ws\AFML
python -m pytest pipeline\orchestration\ -v
cd pipeline\orchestration
python -m pytest -v
cd ..\..
python pipeline\run_pipeline.py
```

## Known limitations / deferred items

- No paired notebook yet.
- The `SAMPLE SIZE WARNING` thresholds (`T < 250`, `n_trials < 5`) are
  provisional and not statistically derived — a design decision to revisit.
- Live Binance ingestion (Phase 2) not yet started.
- The report deliberately never outputs a buy/sell directive — see
  `report.build_report()`'s docstring and
  `test_report_never_issues_a_buy_sell_directive`.
