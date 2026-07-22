# Chapter 12 — Backtesting through Cross-Validation (CPCV)

## A note before anything else

Unlike every other chapter so far, **Chapter 12 has no printed code
snippets.** Sections 12.1–12.5 are motivation, math, and an
algorithm-in-prose (the numbered steps in 12.4.2), illustrated by
Figures 12.1/12.2. There's nothing to diff implementation code against —
this module is built directly from that specification.

The one place the book is genuinely ambiguous is exactly which split
feeds which backtest path. The book's own prose, though, spells out the
*full* group/split composition of path 1 and path 2 for its N=6, k=2
example. Those 12 (group, split) data points are used as ground truth —
`test_cpcv.py::TestPathAssignment::test_reproduces_book_path1_and_path2`
reproduces them exactly. See `cpcv.py`'s module docstring for the
derivation.

## What CPCV does, in one paragraph

A walk-forward or plain cross-validation backtest gives you exactly one
historical performance curve — one Sharpe ratio, with a lot of luck
baked into that one particular sequence of outcomes. CPCV instead
partitions the data into N groups and tests *every* combinatorially
possible way of holding out k groups at once, purging/embargoing around
each held-out group. Because every group ends up in the test set of the
same number of splits, those partial out-of-sample forecasts reassemble
into φ[N,k] = k·C(N,k)/N complete, non-overlapping backtest **paths** —
each one uses every observation exactly once, built from a different
combination of trained models. Instead of one Sharpe ratio, you get a
distribution of them.

## Design decisions (per project chat, July 2026)

- **N=6, k=2** — mirrors the book's own Fig 12.1/12.2 worked example
  exactly: 15 splits, 5 paths. On the real 88-event dataset this gives
  five ~14–18-event groups (`[14, 14, 14, 14, 14, 18]`), 4/6 groups
  (~67%) of the data used to train each split.
- **Purging/embargo**: Ch07 `PurgedKFold`'s exact formula (Snippet 7.3),
  generalized from one contiguous test block to k simultaneous,
  possibly non-adjacent test groups — a training observation must be
  "safe" (per Ch07's leading/trailing formula) with respect to *every*
  held-out group at once, not just one. Verified to reduce to Ch07's
  original single-block output exactly when k=1.
- **Classifier**: Ch09's real winning SVC (`C=100, gamma=0.1`),
  `probability=True`, `random_state=0` (determinism — see Ch09/Ch10
  handoffs), `n_jobs=1` (Windows joblib/loky + `SVC(probability=True)`
  crash risk).
- **Per-path Sharpe**: each path's prob/pred forecasts are fed through
  Ch10's real `getSignal` (same `stepSize=0.01`) to get discretized bet
  sizes, multiplied by `ch03_events.csv`'s real per-event `ret` (raw
  entry→exit price return), then `mean/std` of that position-return
  series. Not annualized — the real dataset spans about two weeks of
  March 2026 trades, so an annualization factor would be more noise than
  signal; this reports the per-bet Sharpe, same spirit as Ch11's PBO
  comparison.

## Real-data result (87-row enriched BTC/TUSD training table; 88 real triple-barrier events -- one dropped for still being inside a Ch19 rolling-window warmup, see `load_data()`'s docstring)

| path | Sharpe |
|------|--------|
| 1 | -0.1952 |
| 2 | -0.2075 |
| 3 | -0.0998 |
| 4 | -0.0146 |
| 5 | -0.1780 |

Distribution: mean **-0.139**, std 0.081, range [-0.208, -0.015].

**Single-path baseline** (Ch10-style plain `PurgedKFold`, `n_splits=4` --
the one number a walk-forward or plain-CV backtest would have reported):
Sharpe = **-0.0115**.

**(Corrected 2026-07-22.)** The table and paragraph below replace two
earlier, inconsistent versions of this section. `SVC_C` was `100.0` --
Ch09's *pre*-Ch19-enrichment grid-search winner -- from Ch12's original
commit straight through the Ch19-enrichment commit (97a5101), which fixed
the StandardScaler bug but never migrated this constant (the same class of
bug Ch10/Ch11 had and were fixed for). An earlier version of the table
above, with mean approx -0.009, was what that stale run actually produced --
but the prose paragraph that followed it in this file claimed "mean Sharpe
+0.067, 1 negative/3 positive" regardless, directly contradicting the table
above it. That number was never produced by any committed run of this
chapter; it appears to have been a transcription error introduced during
the 2026-07-21 audit and never reconciled against the table it sat next to.

With `SVC_C` corrected to **0.01** (Ch09's real post-enrichment winner, the
same value Ch10/Ch11 use), the result sharpens rather than reverses: **all 5
CPCV paths are negative**, in a materially tighter distribution than either
prior version showed. Where the earlier (fabricated) narrative was "genuine
disagreement in sign across paths reveals uncertainty," the corrected result
tells a different but equally CPCV-relevant story: the single-path baseline
alone looks close to breakeven (-0.0115), but all 5 resampled paths agree the
real performance is meaningfully negative -- exactly the kind of consistent
signal a single lucky/unlucky path can mask, and exactly what CPCV exists to
surface.

Section 12.5 variance-reduction check: average pairwise path-return
correlation rho_bar approx **-0.041** (still close to zero -- paths remain close to
independent, as CPCV intends), giving an implied variance of the CPCV mean
Sharpe (0.0011) well below any single path's own variance (0.0066).

## Files

- `cpcv.py` — partitioning, split enumeration, path-assignment algorithm,
  generalized purge/embargo, classifier fit/predict, and the `run_cpcv`
  orchestrator.
- `test_cpcv.py` — 17-test TDD suite (golden book-reproduction test,
  k=1 regression test against Ch07 `PurgedKFold`, synthetic-data
  end-to-end checks).
- `chapter_12_cpcv.py` — real-data demo script (this README's results).
- `chapter_12_cpcv.ipynb` — paired notebook.
- `conftest.py` — BLAS/MKL thread cap.

## Real-machine confirmation

Confirmed on the real `mlfinlab` env (Python 3.10.20 / pandas 1.5.3 /
sklearn 1.2.2), July 2026:
- `chapter_12_cpcv.py`'s real-data output matched the sandbox run exactly
  (all 5 path Sharpes, ρ̄, and the variance-reduction numbers above,
  same precision).
- `test_cpcv.py`: **17/17 pass**.

**Gotcha hit along the way (worth knowing for any future chapter with a
module and its containing folder sharing a name, like `cpcv.py` inside
`ch12\cpcv\`):** bare `pytest test_cpcv.py` failed with
`ImportError: cannot import name 'partition_groups' from 'cpcv'` —
pytest's rootdir-insertion walked up from `test_cpcv.py` through
`__init__.py`-containing ancestor folders and, because `ch12\__init__.py`
was missing, resolved `cpcv` to the *package* `ch12\cpcv\` (empty
`__init__.py`) instead of the *module* `ch12\cpcv\cpcv.py`. Fixed by
adding the missing `ch12\__init__.py`. **Standing convention going
forward: invoke tests as `python -m pytest`, not bare `pytest`** — `-m`
puts the current directory on `sys.path` first and sidesteps this
ambiguity regardless of `__init__.py` placement.
