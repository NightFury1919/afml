# Chapter 2 — Financial Data Structures

Implements AFML Chapter 2: turning a raw stream of individual trades into
the sampled "bars" every later chapter builds on, plus three techniques
for combining multiple related instruments into one clean series.

## What's in this chapter

Raw trade data arrives at an uneven, activity-dependent pace — there's no
such thing as "one trade per second." Chapter 2 covers two families of
bar construction, plus multi-product series handling:

| Section | Concept | Files |
|---|---|---|
| 2.3.1 | Standard bars (tick, volume, dollar) | `standard_bars.py` |
| 2.3.2 | Information-driven bars (tick/volume imbalance, tick/volume run) | `imbalance_bars.py`, `run_bars.py` |
| 2.4.1 | The ETF trick | `etf_trick.py` |
| 2.4.2 | PCA weights | `pca_weights.py` |
| 2.4.3 | Roll gaps / rolled series | `roll.py` |
| 2.5.2 | CUSUM filter (Snippet 2.4) | `filters.py` |

## Folder structure

```
ch02/
├── bars/                                    ← standard + information-driven bars
│   ├── __init__.py
│   ├── standard_bars.py                     Sec 2.3.1 — tick/volume/dollar bars
│   ├── imbalance_bars.py                    Sec 2.3.2 — tick/volume imbalance bars
│   ├── run_bars.py                          Sec 2.3.2 — tick/volume run bars
│   ├── filters.py                           Sec 2.5.2 (Snippet 2.4) — CUSUM filter
│   ├── utils.py                             shared helpers (e.g. ewma)
│   └── test_ch02.py                         66 TDD tests, all numeric values verified
├── multi_product/                           ← combining multiple instruments
│   ├── __init__.py
│   ├── roll.py                              Sec 2.4.3 — roll_gaps, rolled series
│   ├── etf_trick.py                         Sec 2.4.1 — etf_trick
│   └── pca_weights.py                       Sec 2.4.2 (Snippet 2.1) — pca_weights
├── output_data/                             saved bar-construction outputs
├── chapter_2_bars.ipynb                     notebook, real BTC/TUSD tick data
├── chapter_2_bars_example.ipynb             companion notebook, same real data
├── chapter_2_multiproduct_example.ipynb     notebook, real S&P 500 futures data
├── examples_chapter_2_realdata.py           standalone script (bars), popup graphs
├── examples_chapter_2_multiproduct_realdata.py  standalone script (multi-product)
├── README.md                                this file
└── requirements.txt
```

## Running the tests

```bash
conda activate mlfinlab
cd C:\ws\AFML
pytest ch02/bars/test_ch02.py -v
```

## Running the notebooks / example scripts

Bars use real BTC/TUSD tick data (`input_data/BTCTUSD-trades-2026-03.csv`,
this project's shared dataset). Multi-product examples use the real
SP00–SP99 S&P 500 futures files (`input_data/SP*.txt`).

```bash
# Notebooks (inline plots)
jupyter notebook ch02/chapter_2_bars.ipynb
jupyter notebook ch02/chapter_2_multiproduct_example.ipynb

# Standalone scripts (popup plot windows)
python ch02/examples_chapter_2_realdata.py
python ch02/examples_chapter_2_multiproduct_realdata.py
```

Note: `examples_chapter_2_realdata.py` builds dollar bars at
`thresh=50000` for its own illustration, while the downstream pipeline
(Ch03 labeling onward) standardizes on `thresh=10000`. That divergence is
deliberate (commented at the call site) — not a dependency issue, but the
most common source of confusion when re-running this chapter in isolation.

## Key design notes

- **Real data first.** Both bar construction and multi-product handling
  run against real data — real Binance BTC/TUSD ticks for bars, real
  S&P 500 futures contracts for the roll/ETF-trick/PCA-weights examples.
- **Information-driven bars sample adaptively.** Imbalance and run bars
  close based on an EWMA-adaptive expected threshold, so they sample more
  frequently during informed-trading activity and less frequently during
  quiet periods — unlike standard bars, which sample at a fixed threshold
  regardless of market activity.
- **Shared dataset convention.** `BTCTUSD-trades-2026-03.csv` and the
  `SP*.txt` futures files live in the project's root `input_data/` folder
  (used by Ch02 onward), not duplicated into this chapter's own folder —
  per the project's shared-artifact convention.
