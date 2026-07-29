# Chapter 14 — Backtest Statistics

## What's in this chapter
Chapter 14 is mostly a catalogue of the statistics used to report and judge a
backtest. Four sections have real printed code snippets:

- **14.1** Bet timing (`getBetTiming`) — derives the timestamps at which a bet
  ends (flattening or flipping), from a target-position series.
- **14.2** Holding period (`getHoldingPeriod`) — weighted-average holding
  period in days, via the book's average-entry-time pairing algorithm.
- **14.3** HHI concentration (`getHHI`, `hhi_concentration_stats`) — how
  concentrated positive/negative returns (and bet timing) are, on a 0-1 scale.
- **14.4** Drawdown / time-under-water (`computeDD_TuW`) — the sequence of
  drawdowns from each high-water mark, and how long each took to recover from.

Two sections have formulas but **no book snippet** — implemented directly from
the printed equations:

- **14.7.2** Probabilistic Sharpe Ratio (`probabilistic_sharpe_ratio`)
- **14.7.3** Deflated Sharpe Ratio (`deflated_sharpe_ratio`,
  `expected_max_sharpe`)

One section (**14.8**, classification scores) is standard metrics
reinterpreted for meta-labeling, implemented as a small dedicated module
(`classification_scores.py`) rather than relying on sklearn's built-in
scorers, since Table 14.1's degenerate cases require precision/recall to be
`NaN` (not `0`) in specific cases — sklearn's `zero_division=` behavior isn't
consistent about this across the sklearn versions this project has used.

## Explicitly out of scope (recorded in project memory, 2026-07-20)
**Implementation shortfall (14.6)** and **attribution (14.9)** are
deliberately not implemented. Both are prose-only sections with no code
snippets, and both require data this pipeline doesn't produce:

- 14.6 needs real broker fees / slippage / fill-price data. Binance trade
  data isn't the same as our own order fills.
- 14.9 needs a multi-asset portfolio with disjoint risk-class buckets
  (duration, credit, sector, etc.) to attribute PnL across. This pipeline is
  single-asset (BTC/TUSD).

This is a **deferred-pending-data** decision, not a permanent exclusion like
Ch08's synthetic-data policy — revisit once the necessary data exists.

## Book-fidelity fixes applied
- **Snippet 14.2**: `xrange` → `range` (book is Python 2; `xrange` is a
  `SyntaxError` under Python 3). Not a semantic change.
- **Snippet 14.3**: `pd.TimeGrouper(freq='M')` → `pd.Grouper(freq='M')`.
  `TimeGrouper` was deprecated in pandas 0.21 and is fully removed by pandas
  1.5.3 (this project's version). `Grouper` is its direct, behavior-identical
  successor — an API rename, not a book bug.

## Real-data source
Rather than reaching back to Ch10's older, pre-Ch19-enrichment `getSignal`
run, this chapter's demo (`chapter_14_backtest_statistics.py`) re-runs Ch12's
real CPCV pipeline directly (same `random_state`, confirmed deterministic) and
consumes:

- **One CPCV path's real position/return series** (path 1) for bet timing,
  holding period, HHI, and drawdown/TuW — a genuine real position-over-time
  series, consistent with the rest of the post-enrichment pipeline.
- **All 5 real CPCV paths' Sharpe ratios** as the N=5 trial set for DSR — a
  genuinely good fit, since DSR's whole design is "correct a Sharpe estimate
  for how many trials you actually ran," and Ch12 gives us 5 real trials with
  a real cross-path variance.
- **Path 1's real out-of-sample predictions vs. real true labels** for
  Section 14.8's classification-scores demo.

## Real-data finding
**0 of 5 real CPCV paths survive DSR at the 0.95 significance level** — even
path 4, whose PSR against a naive zero benchmark alone looked strongest
(0.4465), fails once DSR corrects for having genuinely run N=5 trials.

*(Corrected 2026-07-22: this previously named "path 2 ... 0.92" -- a figure
that was hardcoded as a literal string in `chapter_14_backtest_statistics.py`'s
print statement rather than computed from the actual DSR results, and was
never true of any committed run. The script now computes the
strongest-PSR path dynamically; see that file's 2026-07-22 LOAD-BEARING note.)*

This is a fourth, differently-mechanised real-data diagnostic pointing the
same direction as Ch11's PBO (~0.83), Ch12's own CPCV Sharpe spread (mean
−0.139, uniformly negative across all 5 paths), and Ch13's O-U
non-stationarity finding: **this real feature set/model combination shows no
backtest result that survives rigorous statistical scrutiny.** Not a code
bug — a real, book-consistent result (this is exactly the kind of
overfitting DSR exists to catch — Snippet 14.5, the "third law of
backtesting").

## Running the tests
```powershell
conda activate mlfinlab
cd ch14\backtest_statistics
python -m pytest -v
```
31 tests total (26 in `test_backtest_statistics.py`, 5 in
`test_classification_scores.py`), all hand-traced against known values or the
book's own Table 14.1 -- no real-data dependency in the unit tests themselves
(real data is exercised only in the chapter script/notebook, per convention).

## Running the real-data demo
```powershell
conda activate mlfinlab
python ch14\backtest_statistics\chapter_14_backtest_statistics.py
```
