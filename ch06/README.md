# Chapter 6 -- Ensemble Methods

Implements AFML Snippets 6.1-6.2: bagging classifier accuracy
calculation and three ways of setting up a random forest for financial
ML, incorporating Chapter 4's sequential bootstrap and average
uniqueness machinery.

## Contents

```
ch06/
├── ensemble/
│   ├── __init__.py
│   └── bagging_accuracy.py   Snippet 6.1 -- P[X > N/k] calculation
├── tests/
│   └── test_ch06.py          10 tests, all passing
├── chapter_6_ensemble.py      example script (Snippets 6.1 + 6.2)
├── chapter_6_ensemble.ipynb   notebook walkthrough
├── README.md
└── requirements.txt
```

## Key concepts

**Snippet 6.1 -- bagging accuracy:** If each of N independent
classifiers is correct with probability p, the ensemble beats random
guessing when p > 1/k (where k is the number of classes). Below that
threshold, more classifiers makes things worse, not better. The
function computes P[X > N/k] -- the necessary-condition probability
for a correct ensemble prediction.

**Snippet 6.2 -- three RF setups:** Three ways to instantiate a random
forest, each incorporating `avgU` (average uniqueness from Chapter 4)
as `max_samples` to address the non-IID overlap problem in financial
labels. Note: classifiers are configured here but NOT fitted -- fitting
happens in Chapter 7 once cross-validation is in place.

## Book errata in Snippet 6.1

The printed code uses Python 2 syntax throughout:
- `from scipy.misc import comb` → use `scipy.special.comb`
- `xrange(...)` → use `range(...)`
- `print p, 1-p_` → use `print(p, 1-p_)`

## sklearn compatibility notes for Snippet 6.2

Tested against sklearn 1.2.2 (Ethan's mlfinlab environment):
- `base_estimator=` (deprecated 1.2, removed 1.4) → use `estimator=`
- `max_features='auto'` (deprecated 1.1, removed 1.3) → use `max_features='sqrt'`

## Running

```bash
# Tests
cd ch06/tests && pytest test_ch06.py -v

# Example script
cd ch06 && python chapter_6_ensemble.py
```
