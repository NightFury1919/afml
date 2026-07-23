# Chapter 3 — Labeling

Implements AFML Chapter 3: methods for assigning labels to financial
observations, used as the target variable for supervised ML models.

## What's in this chapter

Raw price series don't come with labels — there's no built-in answer to
"was this a good trade?" This chapter covers three labeling approaches of
increasing sophistication:

| Section | Concept | Files |
|---|---|---|
| 3.2 | Fixed-time horizon labeling | `returns.py` |
| 3.3–3.5 | Triple barrier method | `triple_barrier.py` |
| 3.6–3.9 | Meta-labeling | `meta_labeling.py` |

## Folder structure

```
ch03/
├── labeling/                             ← all implementation code + tests
│   ├── __init__.py
│   ├── returns.py                        Section 3.2 — fixed_time_horizon
│   ├── triple_barrier.py                 Sections 3.3-3.5 — get_daily_vol,
│   │                                      add_vertical_barrier, apply_pt_sl_on_t1,
│   │                                      get_events, get_bins
│   ├── meta_labeling.py                  Sections 3.6-3.9 — get_events_meta,
│   │                                      get_bins_meta, drop_labels
│   └── test_ch03.py                      38 TDD tests, all numeric values verified
│                                          (includes multithreading tests, skipped on
│                                          Windows due to multiprocessing constraints)
├── chapter_3_labeling.ipynb               notebook with explanations + inline graphs
├── examples_chapter_3_labeling.py         standalone script, same content, popup graphs
├── README.md                              this file
└── requirements.txt
```

## Running the tests

```bash
conda activate mlfinlab
cd C:\ws\AFML
pytest ch03/labeling/test_ch03.py -v
```

To verify multithreading manually (skipped automatically in pytest on Windows):

```bash
python -c "
import sys; sys.path.insert(0, '.')
from ch03.tests.test_ch03 import run_threading_tests
run_threading_tests()
"
```

## Running the notebook / example script

Both use real BTC tick data from `ch02/input_data/`, converted to dollar
bars, filtered through CUSUM, then labeled with all three methods above.

```bash
# Notebook (inline plots)
jupyter notebook ch03/chapter_3_labeling.ipynb

# Standalone script (popup plot windows)
python ch03/examples_chapter_3_labeling.py
```

## Key design notes

- **Real data first.** All labeling methods run against real BTC dollar
  bars and real CUSUM-filtered events, not synthetic placeholders.
- **Multiprocessing.** `get_events` and `get_events_meta` use the shared
  `mp_pandas_obj` utility (`utils/multiprocess.py`) to parallelize barrier
  checking across CPU cores via `num_threads`. Defaults to single-threaded
  (`num_threads=1`) for safety/debugging; on Windows, multi-threaded calls
  must be wrapped in `if __name__ == '__main__':`.
- **Triple barrier vs fixed-time horizon.** The triple barrier method
  accounts for the PATH price took (not just the endpoint), which is more
  realistic — a trade that dipped sharply before recovering would be
  correctly labeled a loss (stop-out), where fixed-time horizon would miss
  this entirely.
