# Chapter 13 — Optimal Trading Rules (Backtesting on Synthetic Data)

Implements AFML Chapter 13: instead of calibrating a profit-take/stop-loss
trading rule by backtesting against real (or resampled real) history, fit a
discrete Ornstein-Uhlenbeck process to the strategy's own historical
opportunities, then optimize the trading rule against a large number of
**synthetic** paths generated from that fitted process. Because synthetic
paths never touch any specific historical dataset, there's no particular set
of datapoints to overfit to.

**Book-fidelity note:** unlike Ch12, this chapter *does* have printed code
(Snippets 13.1/13.2). One erratum was found and fixed: the book's Snippet
13.2 uses Python-2 print-statement syntax (`print comb_[0],comb_[1],...`,
no parentheses), a `SyntaxError` under Python 3. Fixed by returning results
instead of printing them (also makes `batch()` testable).

## Files

- `otr.py` — core implementation:
  - `build_xy_from_opportunities` / `estimate_ou_params` — Step 1 (eq. 13.5–13.7)
  - `phi_to_half_life` / `half_life_to_phi` — Section 13.5.1 conversions
  - `simulate_ou_path` — one synthetic path under one trading rule (Snippet 13.2 inner loop)
  - `batch` — sweep a mesh of (profit-take, stop-loss) pairs (Snippet 13.2)
  - `best_node` — Step 5a: pick R* = argmax{Sharpe}
- `test_otr.py` — 19-test TDD suite, including a genuine book-magnitude
  validation test (reproduces the ~3.2 and ~12.0 Sharpe figures the book
  states for its own {forecast=0,hl=5} and {forecast=5,hl=5} examples).
- `chapter_13_otr.py` / `chapter_13_otr.ipynb` — three-part demo:
  - Part A: reproduces two of the book's own synthetic heat-maps
  - Part B: calibrates {phi, sigma} from real Ch10 BTC/TUSD opportunities
  - Part C: applies the real calibration to a real mesh sweep

## Real-data result (important — read before trusting any OTR output on this data)

Calibrating on real BTC/TUSD opportunities gives **phi_hat ≈ 1.04** —
**non-stationary** (violates the O-U requirement phi ∈ (-1,1) from eq. 13.4).
This holds regardless of how the target level is defined (tried both the
book-literal profit-taking level and a simpler entry-price centering — see
the `LOAD-BEARING` comment in `chapter_13_otr.py`'s Part B for the full
investigation). The likely cause: raw BTC bar-level prices, over these short
(~12-bar) trade windows, behave close to a random walk — which Section
13.6.1 itself identifies as the case where "there are no recognizable areas
where performance can be maximized." Part C's mesh confirms this: Sharpe
ratios across the entire real mesh are flat and near zero (range roughly
±0.09), unlike Part A's book-reproduction cases (Sharpe ~3-13 at the optimal
node). **This is a genuine, book-consistent finding, not an implementation
bug** — treated as the chapter's real result rather than forced into a false
"optimal" trading rule. **(2026-07-22.)** This is no longer an isolated finding: Ch11's PBO (~0.83), Ch12's CPCV (all 5 real paths negative), and Ch14's DSR (0/5 paths survive at 0.95) are three independently-mechanised diagnostics on this same real-data pipeline, all corroborating "no exploitable signal in this feature set/model combination." Per-side calibration was considered as an alternative and rejected — see the `DECISION` comment in `chapter_13_otr.py`'s Part B.

## Reproducibility note

`simulate_ou_path`/`batch` take an explicit `random_state` parameter (sklearn
convention, matching Ch09's SVC) rather than relying on global RNG state.
An earlier version defaulted to Python's built-in `random.gauss`, which made
`np.random.seed(...)` calls elsewhere silently ineffective — real runs
looked seeded but weren't. Caught after real-machine confirmation; fixed,
re-tested (19/19 still pass), and both `chapter_13_otr.py`/`.ipynb` now pass
`random_state` explicitly so their real-data mesh outputs are genuinely
reproducible from a fresh run.

## Real-machine notes

`otr.py`/`test_otr.py` have no sklearn/pandas dependency in the core module
(only `numpy`), so this chapter is lighter-weight than Ch09-Ch12. `batch()`
is a pure-Python loop per the book's own Snippet 13.2 (the book itself notes
this "can be parallelized... we leave that task as an exercise") — the demo
script uses reduced mesh/nIter for runtime; scale up `n_iter` and mesh
resolution for a closer match to the book's 20×20 × 100,000 if desired.

Run tests with `python -m pytest` (not bare `pytest`), per the Ch12
pytest-rootdir convention.
