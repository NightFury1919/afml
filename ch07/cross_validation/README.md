# Chapter 7: Cross-Validation in Finance

Implements AFML snippets 7.3 (`PurgedKFold`) and 7.4 (`cvScore`) and applies
them to the real 88-row BTC/TUSD triple-barrier training table assembled
from Chapters 3-5.

## Why this chapter exists

Standard k-fold cross-validation assumes observations are IID. Financial
triple-barrier labels span an *interval* `[t0, t1]` rather than a single
point in time, and neighboring observations' label intervals overlap.
A naive random k-fold split lets overlapping-in-time observations land on
both sides of the train/test boundary, leaking information and producing
an optimistic CV score you can't trust.

`PurgedKFold` fixes this with two mechanisms:

1. **Purging** -- drop training observations whose label interval overlaps
   the test set's time range.
2. **Embargo** -- drop a further slice of training observations immediately
   after the test set, to account for serial correlation past the strict
   label window.

## Files

| File | Purpose |
|---|---|
| `purged_kfold.py` | `PurgedKFold` and `cvScore` implementation |
| `test_purged_kfold.py` | 16-test TDD suite (pytest) |
| `chapter_7_cross_validation.py` | Pipeline-artifact script -- run directly to reproduce results |
| `chapter_7_cross_validation.ipynb` | Paired notebook, plain-English -> math -> code walkthrough |

## Fixes applied vs. the raw AFML book snippets

The book's snippets 7.3/7.4 have a few rough edges that surface once you
run them for real:

1. **`pctEmbargo` default is `0.`, not `None`.** `int(n * None)` crashes on
   the default in some implementations. `0.` is a valid, meaningful
   default (no embargo) that composes correctly with `int(n * pctEmbargo)`.

2. **Positional indexing uses `.iloc` explicitly.** The original snippet's
   `self.t1[test_indices]` relies on pandas' *deprecated* positional-
   fallback behavior for `Series.__getitem__` on a non-integer (datetime)
   index. That fallback still works on pandas 1.5.3 (this repo's pinned
   version) with a `FutureWarning`, but pandas >= 2.0 removed it entirely,
   turning it into a hard `KeyError`. `.iloc[test_indices]` is correct on
   every pandas version.

3. **`split()` enforces index alignment rather than assuming it.** X, y,
   `sample_weight`, and `t1` must share an identical, identically-ordered
   index before iterating -- previously this was only checked with a bare
   assertion (or not at all), and misalignment silently produced a
   wrong-but-plausible CV score.

## Sanity-checked against sklearn 1.2.2 specifically

`PurgedKFold` subclasses sklearn's private `_BaseKFold` and overrides
`split()` entirely. Because `_BaseKFold` is private (leading underscore),
sklearn makes no backward-compatibility promise on it, so this was verified
directly against the real sklearn 1.2.2 wheel (this repo's pinned version)
rather than assumed:

- `_BaseKFold.__init__` is `@abstractmethod`, signature
  `(self, n_splits, *, shuffle, random_state)` -- `PurgedKFold` overriding
  `__init__` satisfies the contract.
- The inherited `shuffle=False` + `random_state != None` validation
  correctly raises.
- `get_n_splits()` is inherited unchanged.
- No version drift found on any of this between sklearn 1.2.2 and current
  releases.

## A `BaggingClassifier(max_samples=avgU)` note

On sklearn 1.2.2, `max_samples` (a float) is computed as
`int(max_samples * n_train_rows)` -- a fraction of the training fold's row
count, **independent of `sample_weight`**. (Later sklearn releases added an
alternative "frequency semantics" mode that scales by
`sample_weight.sum()` instead when weights are passed -- not relevant here,
since this repo is pinned to 1.2.2.)

Because purging shrinks each fold's training set well below the full
88 rows, `max_samples=0.2288` (calibrated from Ch04's `avgU` against the
*full* dataset) draws a genuinely small number of rows per tree inside CV
(9-14 rows per fold on this dataset) -- worth knowing rather than assuming
it always means "~20 rows."

## Usage

```python
from purged_kfold import PurgedKFold, cvScore
from sklearn.ensemble import RandomForestClassifier

scores = cvScore(
    RandomForestClassifier(class_weight='balanced_subsample'),
    X, y, sample_weight=w, t1=t1,
    scoring='accuracy', n_splits=4, pctEmbargo=0.12,
)
```

Run `python chapter_7_cross_validation.py` to reproduce the full pipeline
on real data, or open the paired notebook for the full walkthrough.

## Results on this dataset (n_splits=4, pctEmbargo=0.12)

| Fold | train_n | test_n | RF accuracy | Bagging neg_log_loss |
|---|---|---|---|---|
| 0 | 52 | 22 | 0.459 | -0.813 |
| 1 | 43 | 22 | 0.589 | -0.661 |
| 2 | 42 | 22 | 0.716 | -0.686 |
| 3 | 63 | 22 | 0.422 | -0.843 |

Mean RF accuracy: 0.547 (std 0.116). Mean bagging neg log loss: -0.751
(std 0.079). High fold-to-fold variance is expected at n=88 with purged
CV -- each fold's test set is only 22 rows.
