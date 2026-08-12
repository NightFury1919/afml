# Pipeline — Capstone Integration Layer

## What this is

Unlike the chapter folders, this isn't a book chapter — it's a capstone
orchestration layer chaining real, already-tested code from across the repo
into one continuous flow: real trial construction → PBO/DSR overfitting
diagnostics → bet-sizing signal → plain-English evidence report. The end
goal (per project scope, 2026-08-12) is a tool a college trading club can
point at real market data and get an honest, statistically rigorous
assessment back — not a black-box buy/sell signal.

**No new AFML formula is implemented here.** Every calculation delegates to
existing, real-machine-confirmed chapter modules:

| Stage | Real module |
|---|---|
| Trial construction (20-config SVC grid, purged CV, bar-level PnL) | `ch11/chapter_11_backtest_dangers.py` (`part_c_build_trials`, `out_of_sample_probs`) |
| Overfitting probability | `ch11/backtest_dangers/pbo.py` (`pbo`, `sharpe_ratio`) |
| Bet sizing | `ch10/bet_sizing/bet_sizing.py` (`getSignal`) |
| Deflated Sharpe | `ch14/backtest_statistics/backtest_statistics.py` (`deflated_sharpe_ratio`) |

`orchestration/stages.py` is pure glue, plus one genuinely new piece:
`orchestration/report.py`, which synthesizes those real statistics into a
plain-English writeup.

## Structure

```
pipeline/
├── run_pipeline.py              driver script, real data
├── README.md
├── requirements.txt
└── orchestration/
    ├── stages.py                 orchestration: real trials → PBO/DSR → signal
    ├── report.py                 new: plain-English evidence report
    └── test_orchestration.py     17 tests (structural + one regression guard)
```

No `__init__.py` in `orchestration/` — deliberate, matches every other
chapter's test-containing folder (e.g. `ch07/cross_validation`,
`ch11/backtest_dangers`). Adding one broke two-pass pytest collection from
the repo root (see Phase 1 fix history below).

No paired notebook yet — will be added once the design is fully settled.

## Phase 1 scope

Wires the orchestration + report layer to the **existing static** real
March 2026 BTC/TUSD artifacts. Phase 2 will replace the static-artifact
load with a live Binance pull feeding the same downstream bar/feature/label
pipeline (Ch02–05, Ch17–19).

## Phase 1a → Phase 1b: a real methodology bug, found and fixed

**Phase 1a's first draft was wrong, and it's worth recording why.** It used
3 ad-hoc classifiers on 87 raw event-level `ret * pred` pairs (always
full-size, no confidence weighting) and produced a **DSR of 0.9995** with
a false "high confidence" verdict — directly contradicting this project's
own established, convergent finding of **no exploitable signal** on this
same BTC/TUSD dataset, independently verified five separate ways (Ch11
PBO≈0.83, Ch12 CPCV all-negative, Ch13 O-U≈random walk, Ch14 DSR 0/5 paths
survive, Ch15 P[fail]≈0.45–0.47).

Root cause, found by reading Ch11's and Ch14's actual driver scripts rather
than guessing: Phase 1a's construction had (1) too few trials — 3, vs.
Ch11's real 20-configuration `SVC(C) × getSignal(stepSize)` grid — (2) too
few effective observations — 87 raw events, vs. Ch11's 238 bar-level
mark-to-market points — and (3) a naive full-size-every-call PnL proxy
instead of Ch10's real discretized `getSignal` position, plus a silent
Gaussian `skew=0, kurtosis=3` assumption in the DSR call instead of the
winning trial's real (fatter-tailed) return distribution.

**Phase 1b fixes this by reusing Ch11's own real, established
trial-construction function directly** (`part_c_build_trials`,
`out_of_sample_probs`) rather than re-deriving a parallel, weaker version.
Real sandbox result after the fix:

| Metric | Phase 1a (wrong) | Phase 1b (reconciled) | This project's established finding |
|---|---|---|---|
| PBO | not computed the same way | **82.86%** (sandbox) / **82.86%** (real-machine) | ~0.83 |
| DSR | 0.9995 (false high confidence) | **0.5445** (borderline, correctly flagged unreliable at this sample size) | 0/5 CPCV paths survive |
| Latest signal | +0.65 (long) | **+0.00 (flat)** | consistent with no edge |

Phase 1b's PBO now matches the established real number almost exactly, and
DSR/signal are both consistent with "no reliable edge" rather than
contradicting it. This is the intended state of Phase 1 going forward —
**Phase 1a's approach should not be resurrected.**

## ⚠️ Still worth caution

`report.build_report()` still surfaces an explicit `SAMPLE SIZE WARNING`
(`T < 150` or `n_trials < 10` by default) rather than a false
`high`/`moderate`/`low` verdict — this project's own `pbo.py` test suite
documents that a single PBO/DSR draw can range **~4%–99%** for a genuinely
zero-edge strategy purely from sampling noise, and this dataset's ~238-bar
ceiling means T will likely always sit in a cautious range for this
particular asset/period. **These thresholds are a starting heuristic, not
a validated statistical cutoff.**

## Real-data confirmation status

**Real-machine confirmed** (mlfinlab conda env, 2026-08-12): 20/20 tests
passed (Phase 1a's test suite; Phase 1b's rewritten 17-test suite is
sandbox pre-checked but not yet real-machine confirmed — see below).
Sandbox pre-check: Python 3.12.3, pandas 3.0.2, numpy 2.4.4, scipy 1.17.1,
scikit-learn 1.8.0 — all newer than mlfinlab's pinned versions.

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
- The `SAMPLE SIZE WARNING` thresholds (`T < 150`, `n_trials < 10`) are
  provisional and not statistically derived — a design decision to revisit.
- Live Binance ingestion (Phase 2) not yet started.
- The report deliberately never outputs a buy/sell directive — see
  `report.build_report()`'s docstring and
  `test_report_never_issues_a_buy_sell_directive`.
- Phase 1b's test suite (17 tests) is sandbox pre-checked only as of this
  writing — needs real-machine confirmation before being treated as final.
