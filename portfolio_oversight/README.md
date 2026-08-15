# portfolio_oversight/

**This directory is NOT AFML content.** It lives at the repo root,
sibling to `ch01`-`ch22` and `pipeline/`, specifically so it can never be
confused with the book-fidelity codebase.

## Why this exists

AFML Ch1 Sec 1.3.1.6 names a "Portfolio Oversight" production station --
with an embargo -> paper trading -> graduation -> re-allocation ->
decommission lifecycle -- but deliberately never gives it an algorithm
(the book's own stated reasoning: "it would be unreasonable for a book
to reveal specific investment strategies"). This directory is this
project's own attempt to fill that named-but-unspecified gap, for
teaching purposes, added 2026-08-15 as a clearly separated follow-on to
Part 5's real-AFML-content work (Ch13 OTR / Ch15 strategy risk / PT-SL,
see `pipeline/orchestration/risk_context.py`).

## What's here

`oversight.py` -- three small, independent functions plus a presentation
wrapper:
- `suggest_capital_allocation` -- turns Ch10's abstract `[-1, +1]`
  `getSignal` confidence scale into a suggested dollar figure, given a
  configurable paper-capital constant.
- `check_circuit_breaker` -- an informational "would a cautious operator
  pause here?" flag from this run's own PBO / Ch15 P[fail].
- `classify_lifecycle_stage` -- a single-run heuristic classification
  into EMBARGO / PAPER_TRADING / GRADUATION_CANDIDATE, using the SAME
  DSR breakpoints as `report.py`'s own confidence bands (0.5, 0.95) for
  internal consistency, not because those numbers come from the book.
- `build_oversight_section` -- plain-English, banner-labeled text
  concatenated onto `report.py`'s output by `run_pipeline_live.py` --
  never merged into `report.py` itself.

## What's deliberately NOT here

- **No real capital, no real order execution.** This entire project only
  ever reads market data (the Binance.US key is read-only). Every
  suggestion here is informational text in a report, never an action.
- **No persisted state.** A "real" circuit breaker or lifecycle tracker
  would need to remember things across multiple live runs (running
  drawdown, time-in-stage). That's new infrastructure this project
  doesn't have yet -- deliberately deferred (decided 2026-08-15), not an
  oversight. Every function here classifies fresh from each run's own
  numbers and forgets everything when the run ends.
- **No buy/sell directive**, matching `report.py`'s own scope discipline
  -- see `test_oversight.py`'s explicit guard test for this.

## Testing

Same TDD convention as the rest of this project (hand-traced expected
values, two-pass pytest confirmation) -- see `test_oversight.py`. There
is no book snippet to trace against here (there's nothing to trace --
see above), so expected values are hand-derived directly from this
module's own documented formulas/thresholds instead.
