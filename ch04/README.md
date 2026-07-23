# Chapter 4 — Sample Weights

Implements AFML Chapter 4: techniques for correcting the non-IID nature of
financial labels caused by overlapping outcomes.

## What's in this chapter

Financial labels (from Chapter 3's triple-barrier method) often overlap in
time — two events can both depend on the same stretch of price history. This
violates the independence assumption most ML algorithms rely on. Chapter 4
addresses this with three complementary weighting schemes:

| Section | Concept | Files |
|---|---|---|
| 4.5 | Average uniqueness & sequential bootstrap | `co_events.py`, `uniqueness.py`, `indicator_matrix.py`, `avg_uniqueness_matrix.py`, `sequential_bootstrap.py` |
| 4.5.4 | Monte Carlo validation | `monte_carlo.py`, `real_data_bootstrap_comparison.py` |
| 4.6 | Sample weights by return attribution | `return_attribution.py` |
| 4.7 | Time decay factors | `time_decay.py` |

## Folder structure

```
ch04/
├── sample_weights/                          ← all implementation code + tests
│   ├── __init__.py
│   ├── co_events.py                         Snippet 4.1 — mp_num_co_events
│   ├── uniqueness.py                        Snippet 4.2 — mp_sample_tw, get_average_uniqueness
│   ├── indicator_matrix.py                  Snippet 4.3 — get_ind_matrix
│   ├── avg_uniqueness_matrix.py             Snippet 4.4 — get_avg_uniqueness
│   ├── sequential_bootstrap.py              Snippet 4.5 — seq_bootstrap
│   ├── example_sequential_bootstrap.py      Snippet 4.6 — runnable demo
│   ├── monte_carlo.py                       Snippets 4.7-4.9 — get_rnd_t1, aux_mc, main_mc
│   ├── return_attribution.py                Snippet 4.10 — mp_sample_w, get_sample_weights
│   ├── time_decay.py                        Snippet 4.11 — get_time_decay
│   ├── real_data_bootstrap_comparison.py    companion: bootstrap comparison on real events
│   └── test_ch04.py                         44 TDD tests, all numeric values verified
├── chapter_4_ntrials_fix.ipynb              notebook with explanations + inline graphs
├── chapter_4_ntrials_fix.py                 standalone script, same content, popup graphs
│                                             (renamed from ch04_timed_v2.py, 2026-07 layout
│                                             refactor -- this README previously described
│                                             chapter_4_sample_weights.{ipynb,py}, which never
│                                             matched what was actually on disk; corrected here)
├── README.md                                this file
└── requirements.txt
```

## Running the tests

```bash
conda activate mlfinlab
cd C:\ws\AFML
pytest ch04/sample_weights/test_ch04.py -v
```

## Running the notebook / example script

Both use real BTC tick data from `ch02/input_data/` and real triple-barrier
events built the same way as Chapter 3.

```bash
# Notebook (inline plots)
jupyter notebook ch04/chapter_4_ntrials_fix.ipynb

# Standalone script (popup plot windows)
python ch04/chapter_4_ntrials_fix.py
```

## Key design notes

- **Real data first.** Wherever feasible, code operates on real BTC tick
  data and real triple-barrier events rather than synthetic placeholders —
  this project is built for students to learn from.
- **Synthetic data is used only where the book's method requires it** — the
  Monte Carlo experiment (Section 4.5.4) needs many independent random trials
  to validate a general statistical claim, so `monte_carlo.py` generates
  synthetic overlap scenarios for that specific purpose. A companion file,
  `real_data_bootstrap_comparison.py`, runs the same comparison directly on
  real events for a more practically grounded result.
- **Runtime caps.** `compare_bootstrap_on_real_events()` subsamples real
  events down to a small, contiguous block (default 12) before bootstrapping
  — `seq_bootstrap`'s cost grows roughly quadratically with event count, so
  this keeps the real-data demo running in a few seconds rather than minutes.
- **Multiprocessing.** All `mp_*` functions follow the same `mp_pandas_obj`
  pattern established in Chapter 3 (see `utils/multiprocess.py`) — pass
  `num_threads > 1` to parallelize across CPU cores. On Windows, multi-threaded
  calls must be wrapped in `if __name__ == '__main__':`.
