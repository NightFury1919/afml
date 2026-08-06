# Chapter 9 — Hyper-Parameter Tuning with Cross-Validation

Implements AFML snippets 9.1–9.4 and wires them onto this repo's **purged**
cross-validator (Ch07 `PurgedKFold`), so hyper-parameter search on
overlapping-label financial data does not leak the way a plain `GridSearchCV`
over a random `KFold` would.

## Files

| File | What it is |
|---|---|
| `hyper_parameter_tuning.py` | Core module: `clfHyperFit`, `MyPipeline`, `logUniform_gen`/`logUniform`, `_pick_scoring`. |
| `test_hyper_parameter_tuning.py` | 17-test TDD suite. |
| `chapter_9_hyper_parameter_tuning.py` | Runnable demo (logUniform → synthetic → real), deferred plotting, saves the stats artifact. |
| `chapter_9_hyper_parameter_tuning.ipynb` | Paired teaching notebook (plain-English → math → code → plots). |
| `conftest.py` | Caps BLAS/OpenMP threads to 1 before numpy import. |

## The three pieces

- **`clfHyperFit`** (9.1/9.2) builds a `PurgedKFold` inner CV and runs
  `GridSearchCV` (`rndSearchIter=0`) or `RandomizedSearchCV` (`rndSearchIter>0`),
  returning the fitted best pipeline. Optionally bags the winner. Scoring is
  auto-selected: **F1** for meta-labels {0,1}, else **neg_log_loss**.
- **`MyPipeline`** (9.2) is a `Pipeline` subclass whose *only* job is to let a
  **bare** `sample_weight=` kwarg reach the final estimator. A plain `Pipeline`
  handles `svc__sample_weight` fit-params inside `GridSearchCV` fine — but
  `BaggingClassifier` forwards a bare `sample_weight`, which a plain `Pipeline`
  rejects. `MyPipeline` rewrites it to `{final_step}__sample_weight`.
- **`logUniform_gen`** (9.3/9.4) is a log-uniform `rv_continuous` — the correct
  prior for scale hyper-parameters (SVC's `C`, `gamma`), where every order of
  magnitude should be equally likely.

## Running

```powershell
conda activate mlfinlab          # Python 3.10.20 / sklearn 1.2.2
cd C:\ws\AFML
python ch09\chapter_9_hyper_parameter_tuning.py
python -m pytest ch09\hyper_parameter_tuning\test_hyper_parameter_tuning.py -v
```

The notebook hardcodes `AFML_ROOT = r'C:\ws\AFML'` (edit if your root differs),
per this repo's notebook path convention. The `.py` scripts derive the root from
`__file__`.

### Windows + SVC threading note

The demo and notebook run the SVC searches at **`n_jobs=1`**. `SVC(probability=True)`
hard-crashes (native, no Python traceback) inside joblib/loky spawned worker
processes on Windows — libsvm's internal probability-CV isn't spawn-safe.
Tree-based chapters (Ch04/Ch08) parallelize fine at `n_jobs=4`; SVC-with-probability
does not. The searches here are small, so serial is plenty fast. (BLAS/OpenMP
threads are still capped to 1 via `conftest.py`, same as Ch08.)

## Cross-chapter imports

Resolves the repo root two directories up and imports:

```python
from ch07.cross_validation.purged_kfold import PurgedKFold
from ch08.feature_importance.feature_importance import getTestData
```

No `__init__.py` files are required anywhere in this chain -- `ch07/`,
`ch07/cross_validation/`, `ch08/`, `ch08/feature_importance/`, `ch09/`, and
`ch09/hyper_parameter_tuning/` are all bare subfolders (implicit namespace
packages). The `__file__`-derived `sys.path` insert plus a fully-qualified
import is what makes this work from any cwd -- no `__init__.py` needed.
(An earlier version of this README incorrectly claimed `__init__.py` files
were required and present; they were never present under ch07/ch08, and
ch09's own two `__init__.py` files -- vestigial, unused by the
fully-qualified import style -- were removed during the chapter-root
layout migration for consistency.) Sanity check:

```powershell
python -c "from ch07.cross_validation.purged_kfold import PurgedKFold; print('ok')"
```

## Fixes applied vs. the raw book snippets (verified on sklearn 1.2.2)

1. **`iid=False` removed** from `GridSearchCV`/`RandomizedSearchCV` — deprecated
   in 0.22, removed in 0.24; a hard `TypeError` on 1.2.2.
2. **`base_estimator=` → `estimator=`** on `BaggingClassifier` — renamed in 1.2,
   removed in 1.4.
3. **Bagging attribute path** resolved from the winning pipeline's own `.steps`.
4. **Python-3 `None > 0` guard.** The book default `bagging=[0, None, 1.]` with
   `if bagging[1] > 0:` raises `TypeError` on Python 3; guarded as
   `bagging[1] is not None and bagging[1] > 0`.
5. **Guarded bagging sample-weight lookup** — no bare `KeyError` when the caller
   didn't pass weights.
6. **scipy docstring `%` caveat (NOT a book bug — a caveat from our
   enhancement).** The book gives `logUniform_gen` a one-line *comment*, so it
   never hits this. We added a *docstring* (better for students/IDE tooltips),
   and an `rv_continuous` *subclass* docstring must contain **no percent sign** —
   scipy treats it as a printf-style template at construction, so a stray `%`
   raises at import. Our docstring is kept percent-free with a regression test
   guarding it. If you'd rather match the book exactly, replace the docstring
   with the book's comment and drop `test_loguniform_docstring_has_no_percent_sign`.

## LOAD-BEARING note — float `max_samples` (carried from Ch08)

On **sklearn 1.2.2** a float `max_samples` for `BaggingClassifier` is a fraction
of the fold's **row count**. Newer sklearn reinterprets it as a fraction of the
**summed sample weight**. With Ch04-style weights this changes how many bootstrap
rows each tree sees. If you port the bagging path to a newer sklearn, revisit
that number.

## Notes / departures (flagged for the boss)

- Demonstrated on **both** synthetic `getTestData` (meta-labels {0,1} → F1) and
  the real BTC/TUSD table (labels {−1,+1} → neg_log_loss). Since 2026-07-21 this
  loads Ch19's real **12-feature enriched table** (fracdiff + 11 microstructural
  features, 87 events) by default, falling back to the original single-feature
  table only if the enriched artifact isn't present. Real winning
  hyperparameters on the enriched table: grid search `C=0.01, gamma=0.1`;
  randomized search `C≈1.223, gamma≈0.175` (both `neg_log_loss`-scored) — see
  `input_data/ch09_hyperparameter_tuning_stats.csv`.
- A matched log-spaced grid runs alongside the randomized search purely so the
  grid-vs-randomized comparison is fair (not from the book).
- `_pick_scoring` was factored out of `clfHyperFit` (identical behaviour) only to
  make the scoring branch directly unit-testable.

## Artifact

`chapter_9_hyper_parameter_tuning.py` writes
`input_data/ch09_hyperparameter_tuning_stats.{csv,pkl}` (best C/gamma per
demo × search, plus the logUniform KS statistic in `.attrs`). Record artifact —
no downstream chapter consumes a ch09 pkl.
