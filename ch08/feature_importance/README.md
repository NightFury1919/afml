# Chapter 8 — Feature Importance

Teaching implementation of AFML (López de Prado) snippets **8.2–8.10**, run on
the book's synthetic dataset whose columns are a known answer key
(`I_*` informative, `R_*` redundant, `N_*` noise).

## What's here

| File | Purpose |
|---|---|
| `feature_importance.py` | Implementation: `getTestData`, `featImpMDI`, `featImpMDA`, `auxFeatImpSFI`, `get_eVec`/`orthoFeats`, `featPCA_rank_corr`, `featImportance`, `testFunc`, `plotFeatImportance`. TDD results embedded at the bottom. |
| `test_feature_importance.py` | 17-test pytest suite. |
| `chapter_8_feature_importance.py` | Runnable demo (`__file__`-derived root) — MDI/MDA/SFI tables + plots, I/R/N summary, orthogonal features + weighted τ, saves a summary CSV/PKL. |
| `chapter_8_feature_importance.ipynb` | Paired notebook: plain-English → math → code → plots. |

## The four methods

- **MDI** (8.2) — in-sample tree impurity. Fast, but noise never scores zero and it suffers substitution effects.
- **MDA** (8.3) — out-of-sample permutation over **purged CV** (consumes Chapter 7). Pushes noise ≈0; still suffers substitution effects.
- **SFI** (8.4) — out-of-sample, one feature alone. Immune to substitution effects, blind to joint effects.
- **Orthogonal features** (8.5) + **weighted Kendall's τ** (8.6) — decorrelate with PCA, then check the supervised ranking against the unsupervised PCA ranking.

## Cross-chapter dependency

Imports `PurgedKFold` and `cvScore` from `ch07/cross_validation/purged_kfold.py`.
The import resolves the repo root two directories up from this module; adjust
the `os.pardir` hops if your `ch07` package lives elsewhere. The book's Ch08
calls a `cvScore(cvGen=…, cv=…)` signature; this repo's `cvScore` takes
`n_splits=/t1=/pctEmbargo=` and builds `PurgedKFold` internally. Because
`PurgedKFold` uses `shuffle=False`, rebuilding it from the same arguments yields
identical folds — so Ch08 threads those args through instead of a shared
`cvGen`, and **Chapter 7 is left untouched**.

## Fixes applied vs. the raw book snippets

1. **`getTestData` (8.7):** `pd.DatetimeIndex(periods=,freq=,end=)` is not a
   valid modern-pandas constructor and `pd.datetime.today()` was removed →
   `pd.date_range(end=datetime.today(), periods=, freq=BDay())`; `xrange`→`range`.
2. **`featImpMDA` (8.3):** `np.random.shuffle(X1_[j].values)` is a **silent
   no-op** on pandas 1.5.3 and a read-only error on ≥2.0 — MDA would report
   ~zero importance for everything. Fixed by reassignment:
   `X1_[j] = np.random.permutation(X1_[j].values)`.
3. **`featImportance` (8.8):** `base_estimator=` → `estimator=` (renamed in
   sklearn 1.2, removed in 1.4).
4. **`testFunc` (8.9):** `izip`→`zip`, `print` statement → `print()`, added
   `itertools.product`, `out[['…']]` double-bracket fix.
5. **`plotFeatImportance` (8.10):** `str(tag)` so an int default doesn't crash
   the title; returns an `Axes` for inline display.

## ⚠ LOAD-BEARING: `max_samples` is a float on purpose

`featImportance` builds `BaggingClassifier(max_samples=1.)` as a **float**.

- On **sklearn 1.2.2** (this repo's env): float `max_samples` = fraction of the
  fold's **row count** → `int(1.0 × n_rows)` → correct.
- On **newer sklearn**: float `max_samples` = fraction of **summed sample
  weight**. Since `getTestData` sets `w = 1/n` (weights sum to 1),
  `max_samples=1.` collapses to **one bootstrap row per tree** — all importances
  go to noise, `oob ≈ 0.5`.

**Run everything under `mlfinlab` (Python 3.10.20 / sklearn 1.2.2).** Do not
change the float to silence a newer-sklearn warning without also rescaling the
weights.

## Running

```bash
# from the repo root, mlfinlab env active
python ch08/feature_importance/chapter_8_feature_importance.py
# tests
pytest ch08/feature_importance/test_feature_importance.py -v
```

The demo uses the book's canonical size (40 features, 10k samples, 1000 trees,
10-fold CV). Dial `N_ESTIMATORS` down in the script for a faster exploratory run
— the I/R/N pattern is already clear by ~200 trees.

## Performance / troubleshooting

**If tests or the demo run many times slower than expected** (minutes per test,
SFI seemingly stuck) on a multi-core Windows/conda machine, the cause is almost
always **BLAS/OpenMP thread oversubscription**: fitting many small trees makes
every tiny matrix op spawn a thread pool across all cores, and they thrash.
Counterintuitively this is *worse* on more cores.

Fixes (either works; the first is automatic for tests):

- Tests: `conftest.py` caps BLAS to one thread before numpy imports. The demo
  script does the same at its top.
- Anything else (notebook, interactive): set the caps in your shell *before*
  launching, so numpy reads them at import:
  ```
  # PowerShell
  $env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1
  ```
  In a notebook, put this in the very first cell, before `import numpy`:
  ```python
  import os
  for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
      os.environ[v] = '1'
  ```

This pairs correctly with `n_jobs=4`: joblib parallelizes across features/trees
at the process level while each worker runs single-threaded BLAS — full core
usage, no contention.

**See per-test timings:** `pytest … --durations=0` (all) or `--durations=10`
(slowest ten). The SFI tests and `testFunc` are the heaviest (one CV per
feature); the rest are sub-second.
