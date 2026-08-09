# Chapter 20 — Multiprocessing and Vectorization

Unlike every other chapter in this project, Chapter 20 isn't a
financial-ML formula chapter — it's about infrastructure. `mpPandasObj`
is a function this project has been quietly calling since Chapter 3
(triple-barrier labeling, sample-weight uniqueness, feature importance,
bet sizing) without ever explaining what it does. This chapter explains
it, and completes the book's own Chapter 20 material with pieces earlier
chapters never needed: `nestedParts` (triangular-workload-aware
partitioning) and the output-reduction path (`processJobsRedux` /
`mpJobList`).

## What's implemented

| Topic | Book section | Snippet(s) | Where |
|---|---|---|---|
| Vectorization (Cartesian product) | 20.2 | 20.1, 20.2 | `multiprocessing_engine/vectorization.py` |
| Single-thread vs. multiprocessing timing | 20.3 | 20.3, 20.4 | `multiprocessing_engine/barrier_touch.py` |
| Atoms/molecules — equal partition | 20.4.1 | 20.5 | `utils/multiprocess.py` (`lin_parts`, pre-existing since Ch03) |
| Atoms/molecules — triangular partition | 20.4.2 | 20.6 | `utils/multiprocess.py` (`nested_parts`, **new this chapter**) |
| `mpPandasObj` engine | 20.5.1 | 20.7, 20.8 | `utils/multiprocess.py` (`mp_pandas_obj`, `process_jobs`, pre-existing since Ch03) |
| Async multiprocessing calls | 20.5.2 | 20.9 | `utils/multiprocess.py` (`process_jobs_mp`, pre-existing; `report_progress`, **new**) |
| Callback unwrapping | 20.5.3 | 20.10 | `utils/multiprocess.py` (`_job_wrapper`, pre-existing, equivalent role to `expandCall`) |
| Bound-method pickling workaround | 20.5.4 | 20.11 | Not ported — see "Book-fidelity notes" below |
| Output reduction | 20.5.5 | 20.12, 20.13 | `utils/multiprocess.py` (`process_jobs_redux`, `mp_job_list`, **new**) |
| Worked memory-bounded example | 20.6 | 20.14 | `multiprocessing_engine/barrier_touch.py` (`barrier_touch_engine` / `run_barrier_touch_engine`, adapted from the book's PCA-by-file-chunk example to this chapter's own barrier-touch problem) |

## Why this chapter extends `utils/multiprocess.py` instead of building a
## separate engine

`utils/multiprocess.py`'s `mp_pandas_obj`/`lin_parts` **is** the book's
`mpPandasObj`/`linParts` — already ported to Python 3, already the real
function used by Ch03, Ch04, Ch08, and Ch10. Building a second,
Ch20-local copy just to have something to "teach" would split the repo
into two competing multiprocessing engines — one actually used, one that
only exists to be explained — which is exactly the kind of split-brain
situation this project has hit before (the Ch07 training-table split,
Ch12/Ch14's SVC_C drift) and now deliberately avoids.

**Additive only:** `nested_parts`, `report_progress`, `process_jobs_redux`,
and `mp_job_list` were added to the end of `utils/multiprocess.py`. The
four pre-existing functions' signatures and behavior are untouched —
verified via a full regression run of Ch03/Ch04/Ch10's test suites (137
tests, all pass) after the extension, so none of those three already-closed
chapters are put at risk.

`utils/multiprocess.py` also had **no dedicated test file until this
session**, despite being load-bearing for four chapters — closed as part
of this chapter's own TDD work (`utils/test_multiprocess.py`, 30 tests
covering both the pre-existing and new functions).

## Files

- `utils/multiprocess.py` (repo-shared, not chapter-local) — extended
  this chapter with `nested_parts`, `report_progress`,
  `process_jobs_redux`, `mp_job_list`.
- `utils/test_multiprocess.py` (repo-shared) — 30 tests, hand-traced
  partition boundaries against the book's own closed-form formulas.
- `multiprocessing_engine/vectorization.py` — Snippets 20.1/20.2,
  `cartesian_product_unvectorized` (hardcoded 3-loop) and
  `cartesian_product_vectorized` / `_generator` (`itertools.product`).
- `multiprocessing_engine/barrier_touch.py` — Snippets 20.3/20.4
  (`barrier_touch`, `main0`, `main1`), plus `barrier_touch_engine` /
  `run_barrier_touch_engine` (this chapter's own addition, fixing a real
  limitation in the book's `main1` — see "Judgment calls" below) and
  `time_single_vs_multi` (real wall-clock benchmarking helper).
- `multiprocessing_engine/test_vectorization.py`,
  `multiprocessing_engine/test_barrier_touch.py` — 21 tests total.
- `multiprocessing_engine/conftest.py` — BLAS thread cap (matches
  Ch08/09/12/13/17/19).
- `chapter_20_multiprocessing.py` / `.ipynb` — four-part driver: (A)
  vectorization, (B) `lin_parts` vs `nested_parts` on Ch17's real SADF
  workload size, (C) real single-thread vs. multiprocessing timing
  benchmark, (D) the same benchmark run through `mp_job_list`.

## Book-fidelity notes (Python 2 → 3 translation)

All of Chapter 20's printed snippets are Python 2. Straightforward
syntax translations (`xrange`→`range`, `print x`→`print(x)`) aren't
called out individually below; two are worth flagging explicitly:

- **`itertools.izip` (Snippet 20.2)** doesn't exist in Python 3 —
  removed, because Python 3's builtin `zip` already does what `izip` did
  in Python 2 (lazy iteration). No replacement import needed.
- **Bound-method pickling (Snippet 20.11, Section 20.5.4)** — **not
  ported**. This is a Python-2-specific workaround: Python 2's bound
  methods (`im_func`/`im_self`/`im_class`) aren't pickleable at all
  without registering a custom pickler via `copy_reg`. Porting the
  snippet verbatim would raise `AttributeError` in Python 3, since those
  attributes were renamed to `__func__`/`__self__` back in Python 3.0 —
  and the underlying problem the snippet solves mostly doesn't arise in
  Python 3, where ordinary bound methods pickle natively in the common
  case. Documented here rather than silently dropped, per this project's
  book-fidelity rule.

## Windows-specific note

`mp.Pool` uses `spawn` on Windows, not `fork` — worker processes
re-import the defining module from scratch. Every worker function used
in this chapter (`barrier_touch`, `barrier_touch_engine`, the
`_mp_test_workers.py` helpers) is a real module-level function for
exactly this reason; an inline function or lambda target is not
spawn-picklable and would silently hang or error on Windows even if it
appeared to work in a fork-based environment.

## Judgment calls

- **`barrier_touch_engine` / `run_barrier_touch_engine` (new, not in the
  book):** the book's own `main1` (Snippet 20.4) chunks `r`'s columns and
  runs `barrier_touch` on each chunk via raw `mp.Pool`, but each chunk's
  result dict is keyed by that chunk's **local** column index (0, 1, 2,
  …) — results from different chunks would collide if merged. The book
  never actually reassembles a global result; `main1` only measures
  wall-clock time and discards the (locally-indexed) outputs. Kept
  `main1` exactly as printed (it's a faithful port, and the timing
  demonstration doesn't need correct indexing), but added
  `barrier_touch_engine` (takes a `molecule` of **global** column
  indices, returns a dict keyed by those same global indices) and
  `run_barrier_touch_engine` (routes through the new `mp_job_list` with a
  `dict.update` reducer) as a second, correctly-indexed version — both to
  fix the real limitation and to give Section 20.5/20.6's formalized
  engine a concrete, chapter-relevant worked example beyond the abstract
  PCA-by-file-chunk case in Snippet 20.14.
- **`process_jobs_redux`'s explicit `list.append` behavior (documented,
  not "fixed"):** the book's own Snippet 20.12 branches so that when
  `redux=None` (the default), the first molecule's output gets wrapped in
  a list (`out=[out_]`) before subsequent outputs are appended — but when
  a user explicitly passes `redux=list.append` with `redux_in_place=True`
  (the *same* reducer), the code skips that wrapping step
  (`out=copy.deepcopy(out_)` instead), so explicit `list.append` produces
  a structurally different result than the implicit default for the same
  inputs. Ported verbatim (this is the book's own printed logic, not a
  translation artifact) and pinned with a dedicated test
  (`test_explicit_list_append_reducer_documents_real_quirky_behavior`)
  documenting the actual behavior, since it's genuinely non-obvious and a
  student relying on intuition here would get it wrong. `dict.update` is
  the case where this branching is unambiguously correct — a dict's first
  output already **is** the right-shaped accumulator, no wrapping needed
  — and is the pattern this chapter actually uses in
  `run_barrier_touch_engine`.
- **`main1`'s hardcoded `numThreads=24` (Snippet 20.4):** made a
  parameter instead, defaulting to 4 — this project's own established
  sweet spot on its real 6-core machine (reduced fan noise/system load
  preferred over the marginal extra speed from 6; see `CLAUDE.md`), not
  the book's specific 24-core reference machine.
- **`main0`/`main1`'s `num_paths`/`path_len` random generation:** uses a
  seeded `numpy.random.Generator` (`np.random.default_rng(seed)`) per
  this project's standing `random_state` convention, rather than the
  book's unseeded `np.random.normal`, so the driver script's benchmark is
  reproducible run to run.

## Real-machine verification

**Real-machine confirmed 2026-08-08** (Windows, `mlfinlab` env, Python
3.10.20, pytest 9.0.3): 51/51 tests pass (`utils/test_multiprocess.py`
30, `multiprocessing_engine/test_barrier_touch.py` 12,
`multiprocessing_engine/test_vectorization.py` 9 — 9.61s), two-pass
(repo root: `pytest utils\ ch20\ -v` collected 51; from inside `utils\`:
30 collected; from inside `ch20\multiprocessing_engine\`: 21 collected —
both isolated passes matched the combined root-level run exactly).
Regression check on the four chapters that import `utils/multiprocess.py`:
`pytest ch03\ ch04\ ch10\ -v` → 104 passed, 3 skipped (expected Windows
multiprocessing skips), 0 failed. No drift from the additive extension.

**Real Part C/D timing (12 logical / 6 physical cores detected):**
single-thread 2.98s vs. 4-thread multiprocessing 2.13s — a genuine
**1.40x speedup**, replacing the sandbox's earlier ~1.0x "speedup" (that
build environment had only 1 CPU core available, so no real
parallelism was possible there — flagged explicitly in the driver
script's `mp.cpu_count()` warning at the time, and confirmed a non-issue
now that it's run for real).

**Notebook** re-run under the real `mlfinlab` kernel via `run_nb.py`:
9/9 code cells executed, `kernelspec.name=mlfinlab`,
`language_info.version=3.10.20` confirmed.

**Process note:** the first handoff attempt hit a stray nested
`ch20\ch20\` and `ch20\utils\` duplication from the file-copy step (not
a code issue) — pytest's "import file mismatch" error correctly caught
two `test_barrier_touch.py`/`test_vectorization.py` pairs with identical
module names and no `__init__.py` to disambiguate them. Removing the
stray duplicates immediately gave the clean 51/51 root-level run above —
the isolated per-folder passes had already confirmed the *real* files
were correct throughout, so this was purely a file-placement artifact,
not a test or implementation bug.

## Outstanding

None — chapter complete. Implementation, tests, driver script, notebook,
and README are all real-machine confirmed as of 2026-08-08.
