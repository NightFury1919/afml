# Chapter 21 — Brute Force and Quantum Computers

## What this chapter is

Unlike most AFML chapters, Ch21 isn't a formula applied to the repo's
feature/label pipeline. It's a **combinatorial optimization** chapter: the
book shows how a dynamic portfolio-trajectory problem (optimal weights
across multiple future horizons, accounting for path-dependent transaction
costs) can be discretized into a finite search space and solved by brute
force — motivating why quantum computers (via qubit superposition) are
theoretically suited to this class of NP-hard problem. §21.1–21.2 are pure
motivation; no code in this chapter requires or simulates quantum hardware
— the numerical example in §21.6 runs the exhaustive search sequentially on
an ordinary computer, same as this chapter's implementation.

## Structure

```
ch21/
├── chapter_21_brute_force.py       driver script, real data
├── chapter_21_brute_force.ipynb    notebook walkthrough, real data
├── README.md
├── requirements.txt
└── portfolio_trajectory/
    ├── __init__.py
    ├── brute_force.py              Snippets 21.1, 21.2, 21.3
    ├── random_matrix.py            Snippets 21.4, 21.5
    ├── static_solution.py          Snippet 21.6
    ├── data_prep.py                real-data pipeline (NOT in the book)
    ├── test_portfolio_trajectory.py   25 tests (brute_force/random_matrix/static_solution)
    └── test_data_prep.py              18 tests (16 unit + 2 real-data integration)
```

## Book-fidelity notes (Python 2 → Python 3 translation)

Per the project's book-fidelity rule, all code below was built from Ethan's
actual printed Snippets 21.1–21.7, not from memory. No bugs analogous to
Ch09's bagging-tuple-order bug were found — the combinatorial logic
(pigeonhole partitioning, signed weight generation, trajectory evaluation)
was verified faithful to the book's stated formulas. Translation changes:

- `xrange` (Snippet 21.1) → `range` (Python 3's `range` is already lazy).
- `print 'static SR:', sr_stat` / `print 'dynamic SR:', sr_dyn` (Python 2
  statements in the book's driver code) → `print(...)` calls.
- The book's Snippets 21.4/21.5 call the unseeded global `np.random.*`
  directly. Per this project's `random_state` convention, both
  `rnd_mat_with_rank` and `gen_synthetic_params` take an explicit
  `numpy.random.Generator`, threaded through every draw, so tests are
  reproducible. Functionally identical to the book when left as `None`.
- `dyn_opt_port`'s default `k = params[0]['mean'].shape[0]` sets K=N by
  default, which appears to undercut §21.5.1's stated "K > N" assumption
  for the pigeonhole partition-count formula. On inspection this is *not*
  a bug: `combinations_with_replacement` (the actual algorithm) is valid
  for any K ≥ 0 — the book's K > N framing is illustrative (it makes the
  partition count more interesting), not a code requirement. Documented
  here rather than silently "fixed."

## The real-data extension (not in the book)

The book hands you `params` (μ_h, V_h, c_h per horizon) as a given and
demonstrates them only on synthetic random matrices (§21.6.1). Per this
project's real-data-first policy, `data_prep.py` builds those parameters
from real continuous-futures history instead:

- **Assets**: gold, crude oil, US T-bonds (`input_data/gold`,
  `input_data/crude oil`, `input_data/US-T bonds` — continuous futures
  data originally sourced from turtletrader.com, first used anywhere in
  this repo by this chapter). N=3 was chosen to keep the brute-force
  search tractable — trajectory count grows as `(partitions(K,N) ×
  2^N)^H`, so N stays small deliberately, not as a data limitation.
- **Front-month selection**: the raw files are one file per individual
  futures contract, and several contracts' histories overlap (a contract
  can trade for years before its expiry — e.g. `GC02Z` trades from 1998
  through its Dec 2002 expiry). There's no fixed expiry calendar in the
  raw data, so `select_front_month` picks, for each date, whichever
  contract had the highest trading volume that day — a standard real
  roll-selection rule.
- **Raw file format**: two incompatible formats coexist in the *same*
  commodity folder, not split cleanly by era (`GC75F.txt` from 1975 is
  headerless with 6-digit `YYMMDD` dates; `GC02Z.txt`, whose data starts
  in 1998, has a quoted header and `MM/DD/YYYY` dates). `load_contract_file`
  sniffs the format per file and parses accordingly.
- **Roll adjustment**: reuses `ch02/multi_product/roll.py`'s actual
  `non_negative_rolled_prices` (imported via the project's `__file__`-derived
  path pattern, not duplicated) — this is that function's first real
  exercise on non-S&P data anywhere in the repo. It was written in Ch02 but
  never previously run end-to-end on these commodities.
- **μ_h, V_h, c_h construction** (a genuine design decision, since the book
  doesn't specify one): non-overlapping trailing windows of real daily
  returns near the end of the aligned history (60 trading days per horizon
  by default), sample mean/covariance per window, and c_h proxied as
  `cost_scale × that window's realized daily volatility per asset` (costlier
  to trade a more volatile/illiquid asset — a standard proxy).

### A known real-data quirk, documented rather than silently patched

The raw `crude oil` data has one clearly erroneous tick on 1991-01-17 (front
contract `CL91H` drops from $12.99 to $3.33 to $1.69 over three days — not
a real WTI price level for that period; likely a feed error in this vintage
dataset, not something introduced by this chapter's parsing). It sits deep
in history and is **not** included in the default horizon windows (which
use the most recent ~120 trading days, ending Oct 2002), so it doesn't
affect the chapter's real-data results — but a `test_data_prep.py` reader
extending the lookback window substantially into 1991 should be aware of it.

## Real-machine confirmation

**Fully confirmed 2026-08-09, mlfinlab conda env** (Python 3.10.20, pytest
9.0.3, `C:\ws\AFML`, Windows): **43/43 tests passed**, two-pass (repo root +
inside `portfolio_trajectory/`), including the 2 real-data integration
tests run against the actual commodity files (not skipped). Driver script
and notebook (Run All, mlfinlab kernel) both confirmed real-machine, with
identical results: static SR `0.2986251686981421`, dynamic SR
`0.30803461924049597`. No blockers, no outstanding cosmetic items.

Minor, harmless discrepancy noted for the record: the full-history
`describe()` stats for crude oil/US T-bonds differ from the sandbox
pre-check in the 4th-5th decimal (e.g. crude oil mean `0.000495` real
machine vs `0.000475` sandbox), traced to a pandas/numpy version difference
in front-month tie-breaking on same-volume days. Does not touch the Part D
comparison, which matches exactly.

```powershell
conda activate mlfinlab
cd C:\ws\AFML
python -m pytest ch21\portfolio_trajectory\ -v
cd ch21\portfolio_trajectory
python -m pytest -v
cd ..\..
python ch21\chapter_21_brute_force.py
```

## Real-data result (Part D of the driver script/notebook)

Gold + crude oil + US T-bonds, aligned daily returns 1983-03-31 to
2002-10-01 (4,872 trading days), K=4, H=2 horizons (60-trading-day windows
ending 2002-10-01), 14,400 trajectories evaluated:

| Solution | Sharpe Ratio |
|---|---|
| Static (myopic, per-horizon optimal) | 0.2986 |
| Dynamic (brute-force trajectory) | 0.3080 |

The dynamic trajectory search beat the static solution by ~0.0094 SR on
this real window — i.e., jointly accounting for the transaction cost of
moving between horizons' weights found a better path than optimizing each
horizon in isolation, which is exactly the effect this chapter sets out to
demonstrate.

**Caveat worth stating plainly**: this is one 2-horizon window on one
3-asset universe with a coarse K=4 discretization — a demonstration that
the method *can* find a genuinely different, better-scoring trajectory,
not a claim that dynamic trajectory optimization reliably beats static
optimization in general. Different K values did not move the result
monotonically (K=3 → SR=0.3076, K=4 → SR=0.3080, K=5 → SR=0.3031) since
different K values produce genuinely different, non-nested discretizations
of the feasible weight set — worth flagging as a teaching point on its own.

## Why quantum computing, concretely (not implemented here)

This chapter's brute-force search is only tractable because this repo's
real-data run keeps the search space small on purpose: K=4 discretization
levels, H=2 horizons, N=3 assets → 14,400 trajectories, enumerated
sequentially on an ordinary CPU in the run above. The number of candidate
trajectories in this class of problem scales combinatorially with the
number of assets, horizons, and discretization levels — coarser grids or
more assets/horizons blow the search space up exponentially, the same way
`k` holding levels over `N` time steps in the book's simpler framing yields
`k^N` possible trading rules (e.g. 5 levels over 10 steps ≈ 9.8 million
sequences; over 50 steps, ≈ 8.9 × 10^34 — more than the number of atoms in
a human body). Classical brute force, however parallelized (see Ch20's
`utils/multiprocess.py`), still evaluates candidates at a fixed rate; more
cores does not change the exponent.

This is the concrete reason §21.1-21.2 point to quantum computing:
algorithms like Grover's search offer a **quadratic** speedup over
classical exhaustive search — roughly `sqrt(M)` steps instead of `M` for a
search space of size `M` (e.g. `sqrt(9.8 million) ≈ 3,130`, a ~3,000x
speedup at that scale). That pushes the size of tractable brute-force
problems out considerably, but it is a quadratic speedup, not a change from
exponential to polynomial — `sqrt(k^N) = k^(N/2)` is still exponential in
`N`, just with the exponent effectively halved. A large enough N still
breaks even a quantum-Grover approach eventually; quantum computing extends
the horizon of what's brute-forceable, it doesn't dissolve the underlying
combinatorial-optimization problem. This repo's real-data run above sits
comfortably in the classical-brute-force regime specifically because K, H,
and N were kept small enough that 14,400 trajectories was a tractable
sequential search, not because quantum acceleration was applied.

## Known limitations / deferred items

- No blockers. Chapter is complete and real-machine-pending (sandbox
  confirmed, mlfinlab confirmation is the standing final gate per project
  convention).
- The front-month selection rule (highest daily volume) is a reasonable,
  standard proxy but not the only valid roll methodology (calendar-based
  and open-interest-based rules are common alternatives) — worth a note if
  this data prep is ever reused by a later chapter.
- `cost_scale=0.02` is a chosen constant, not calibrated to real
  transaction-cost data for these instruments — it's a proxy, documented
  as such, not a fitted parameter.
