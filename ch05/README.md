# Chapter 5 -- Fractionally Differentiated Features

Implements AFML Snippets 5.1-5.4: fractional differencing, both the
expanding-window and fixed-width-window versions, plus the ADF-based
search for the minimum `d` that achieves stationarity.

## Contents

```
ch05/
├── frac_diff/
│   ├── __init__.py
│   ├── get_weights.py        Snippet 5.1 -- fixed-count weight generation
│   ├── frac_diff.py           Snippet 5.2 -- expanding-window application
│   ├── get_weights_ffd.py     Snippet 5.3 (part 1) -- threshold-stopped weights
│   ├── frac_diff_ffd.py       Snippet 5.3 (part 2) -- fixed-width application
│   │                          (VECTORIZED -- see module docstring; not a
│   │                          book snippet detail, our own perf rewrite)
│   ├── find_min_ffd.py        Snippet 5.4 -- ADF-based search across d values
│   └── calibration.py         OUR OWN utility (not a book snippet) -- finds
│                              a fair absolute thres for frac_diff_ffd that
│                              matches frac_diff's relative thres semantics
├── tests/
│   └── test_ch05.py           29 tests, all passing
├── input_data/
│   └── BTCTUSD-trades-2026-03.csv   real Binance BTC/TUSD tick data
├── chapter_5_frac_diff.py      example script, runs the full pipeline on
│                              real data
├── chapter_5_frac_diff.ipynb   notebook walkthrough, executed end-to-end
├── README.md
└── requirements.txt
```

## Key concept

Raw prices have full memory but aren't stationary. Plain returns are
stationary but destroy almost all memory (only yesterday matters).
Fractional differencing is a dial between the two: a parameter `d`
between 0 and 1 lets you find the *minimum* differencing needed to
achieve stationarity, maximizing how much memory survives.

## Real-data result (BTCTUSD-trades-2026-03.csv, 9,205 ticks)

Minimum `d` that passes the ADF test (p < 0.05): **d = 0.2**
- Correlation with original log price at d=0.2: **0.9987**
- Correlation with original log price at d=1.0 (full differencing): **0.0219**

i.e. on real BTC data, fractional differencing at d=0.2 keeps ~99.9% of
the original series' memory while still achieving stationarity -- versus
full differencing, which throws away ~98% of that memory for no extra
stationarity benefit beyond d=0.2.

## Two traps worth remembering

1. **`thres` means different things in different functions.** In
   `frac_diff`, it's a *relative* weight-loss fraction (0.01 = tolerate
   losing 1% of total weight mass). In `frac_diff_ffd`, it's an
   *absolute* weight magnitude cutoff (0.01 = stop once a weight's size
   drops below 0.01). The book's own Snippet 5.4 demo passes the same
   literal number to both -- this is a real trap, not a contrived one.
   See the notebook for a worked real-data example of how badly this
   can diverge for slowly-decaying (low) `d` values.

2. **Calibrating the thresholds to "mean the same thing" does NOT make
   `frac_diff` and `frac_diff_ffd` agree.** They're structurally
   different algorithms -- one's window keeps growing through the whole
   series, the other's is permanently fixed. This isn't a bug to fix; the
   book recommends the fixed-width version specifically because applying
   a consistent operator at every point is arguably better for downstream
   ML, independent of the (very real, ~490x) speed benefit.

## A likely erratum in the book's printed code

Snippet 5.3's line `w,width,df=getWeights_FFD(d,thres),len(w)-1,{}` is a
single tuple assignment. Python evaluates the entire right-hand side
before assigning anything, so `len(w)-1` sees whatever `w` was *before*
this line ran, not the array just computed on the same line. Implemented
here as three separate statements instead.

## Performance note

`frac_diff_ffd` was rewritten from the book's row-by-row Python loop to
a vectorized numpy sliding-window dot product. Verified identical output
to the literal loop version (floating-point precision, ~1e-15 difference)
on real data. Result: ~2.8s -> ~0.06s per call on the 9,205-tick BTC
series (~490x faster). `frac_diff` (the expanding-window version) is
deliberately left as the simple loop -- it's meant as the "naive, slow"
baseline that motivates the fixed-width snippet's existence in the first
place, which is the book's own documented optimization story for this
chapter.

## Running

```bash
# Tests
cd ch05/tests && pytest test_ch05.py -v

# Example script (real data)
cd ch05 && python chapter_5_frac_diff.py

# Notebook
jupyter notebook chapter_5_frac_diff.ipynb
```
