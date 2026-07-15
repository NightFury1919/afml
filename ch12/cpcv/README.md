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

## Real-data result (88-row BTC/TUSD triple-barrier table)

| path | Sharpe |
|------|--------|
| 1 | −0.0380 |
| 2 | −0.1738 |
| 3 | +0.2770 |
| 4 | +0.0237 |
| 5 | −0.1343 |

Distribution: mean −0.009, std 0.178, range [−0.174, +0.277].

**Single-path baseline** (Ch10-style plain `PurgedKFold`, `n_splits=4` —
the one number a walk-forward or plain-CV backtest would have reported):
Sharpe = **−0.1254**.

This is the chapter's point, made concretely on real data: the
single-path baseline alone would read as "this strategy loses money."
The 5-path CPCV distribution tells a more honest story — genuine
disagreement in *sign* across paths (2 negative, 2 slightly positive, 1
near-zero), consistent with the "thin feature set" theme flagged since
Ch08/Ch09 (the real training table has only one feature, `fracdiff`).
The evidence points to a strategy with close-to-zero real skill and high
estimation uncertainty, not a confidently bad one — a conclusion the
single path alone couldn't support.

Section 12.5 variance-reduction check: average pairwise path-return
correlation ρ̄ ≈ 0.052 (paths are close to independent, as CPCV intends),
giving an implied variance of the CPCV mean Sharpe well below any single
path's own variance.

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
