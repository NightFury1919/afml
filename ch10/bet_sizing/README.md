# Chapter 10 — Bet Sizing

Implements AFML Snippets 10.1–10.4 (López de Prado, *Advances in
Financial Machine Learning*): translating classifier probabilities into
bet sizes, averaging concurrently active bets, discretizing bet size to
prevent overtrading, and dynamic position sizing against a limit price.

## Files
- `bet_sizing.py` — implementation of Snippets 10.1–10.4
- `test_bet_sizing.py` — TDD suite, 13 tests, known expected values
  hand-traced from the book's own formulas (Section 10.3's z/m
  derivation, Snippet 10.4's calibration math) — not shape checks
- `conftest.py` — BLAS/MKL thread cap for the test session
- `chapter_10_bet_sizing.py` / `chapter_10_bet_sizing.ipynb` — pending
  real-data run (see below)

## Functions
| Snippet | Function(s) | Purpose |
|---|---|---|
| 10.1 | `getSignal` | probability → bet size (one-vs-rest OvR), meta-labeling side, averaging, discretizing |
| 10.2 | `avgActiveSignals`, `mpAvgActiveSignals` | average signal among bets concurrently open |
| 10.3 | `discreteSignal` | round to stepSize, cap at ±1 |
| 10.4 | `betSize`, `getTPos`, `invPrice`, `limitPrice`, `getW` | dynamic position sizing and limit-price calculation |

## Book-fidelity notes
- `getSignal`/`avgActiveSignals` dispatch through this repo's
  `utils/multiprocess.py:mp_pandas_obj` (snake_case reimplementation of
  the book's `mpPandasObj`, established Ch04+). Only the dispatch call
  name changes — molecule handling and kwarg passing are unchanged.
- `limitPrice` used Python 2's `xrange` in the printed snippet, which
  doesn't exist in Python 3.10. Replaced with `range` — no semantic
  change.
- All other lines match the printed snippets 10.1–10.4 as pasted.
  Two-class and one-vs-rest z-statistic derivations were checked
  against the book's Section 10.3 text and match exactly.

## Known edge case (documented, not a bug)
If `avgActiveSignals` is called on an empty `signals` frame,
`mp_pandas_obj`'s empty-index guard returns `pd.DataFrame()` rather than
`pd.Series()`. This doesn't surface in the real pipeline because
`getSignal`'s own `prob.shape[0]==0` guard short-circuits first — but
it's exercised directly in `test_bet_sizing.py` so the behavior is
pinned rather than assumed.

## Real-data pipeline decision (July 2026)
No fitted Ch09 classifier exists on disk — only its winning hyperparameters
were persisted (`ch09_hyperparameter_tuning_stats.{csv,pkl}`). Rather than
wait or reconstruct, `chapter_10_bet_sizing.py`/`.ipynb` refit an SVC with
Ch09's real winning hyperparameters (`C=100, gamma=0.1`, the real-data
grid-search winner on `neg_log_loss`) on the real Ch07 training table,
**out-of-sample via Ch07's `PurgedKFold`** (not a plain in-sample refit,
which would be lookahead bias for a bet-sizing signal). If a future chapter
ever does persist a fitted classifier to disk, switch to using that object
directly instead of this refit.

## Status
- [x] TDD suite passing (sandbox: Python 3.12.3 / pandas 3.0.2 / scipy
  1.17.1 / numpy 2.4.4)
- [x] TDD suite confirmed under real `mlfinlab` env (Python 3.10.20 /
  pandas 1.5.3 / numpy 1.23.5 / sklearn 1.2.2) — see project chat, July 2026
- [x] Real-data run: real Ch07 training table, out-of-sample SVC
  probabilities (PurgedKFold, Ch09's real winning hyperparameters), fed
  through `getSignal` end-to-end — confirmed on both sandbox and real
  machine
- [x] `chapter_10_bet_sizing.py` example script
- [x] `chapter_10_bet_sizing.ipynb` paired notebook (executed, real
  outputs + plot embedded)

## Bug found on the real-machine run (fixed)
Running `chapter_10_bet_sizing.py` on the real `mlfinlab` env surfaced a
real determinism bug: `SVC(probability=True)` without a pinned
`random_state` fits an internal randomized 5-fold CV (Platt scaling) to
calibrate `predict_proba`. Left unset, `predict`/`predict_proba` are
non-deterministic run-to-run, and this dataset is small and thin enough
(single feature, 88 rows) that sklearn 1.2.2 (real machine) and a later
sklearn (sandbox) landed on **entirely different winning classes** for
`pred` — not just noisy probabilities, a flipped majority class. Fixed by
pinning `random_state=0` in `out_of_sample_probs`; verified reproducible
(identical `pred` distribution across repeated runs) after the fix, on
both environments.

Also fixed: two `pd.Series()` calls in `bet_sizing.py` (`getSignal`'s
empty-input guard, `mpAvgActiveSignals`'s accumulator) now specify
`dtype=float` explicitly — silences a pandas `FutureWarning` about
empty-Series default dtype, same category as the `xrange`→`range` fix
(forward-compat, not a math change).
