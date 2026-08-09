# Chapter 22 — High-Performance Computational Intelligence and Forecasting Technologies

**Guest chapter, Kesheng Wu & Horst D. Simon (Lawrence Berkeley National
Laboratory).** This README *is* the chapter deliverable — see "Why no
implementation module" below.

## What this chapter is

Unlike every other chapter in this repo, Ch22 is not a formula chapter. It's
a narrative account of LBNL's CIFT (Computational Intelligence and
Forecasting Technologies) project: an argument for applying high-performance
computing (HPC) infrastructure — built for scientific simulation — to
financial streaming-data problems, illustrated with six real case studies.
**There are no printed code snippets anywhere in this chapter.**

### 22.1–22.3: Motivation and background

The chapter's origin story: the SEC/CFTC took **five months** to investigate
the May 6, 2010 Flash Crash (Dow dropped ~10% in minutes, some stocks traded
at $0.01, Apple briefly quoted at $100,000/share), and cited the ~20TB of
data involved as the reason for the delay. LBNL's argument: HPC centers like
NERSC routinely process hundreds of terabytes in minutes, so financial data
volume shouldn't be a bottleneck at all — it's a tooling gap, not a
fundamental limit. CIFT was founded to close that gap.

The chapter draws a specific distinction between two computing paradigms:
- **Cloud computing** — built for *data-parallel* throughput (many
  independent objects processed concurrently), virtualized for multi-tenancy,
  optimized for cost/flexibility over latency.
- **HPC computing** — built for *tightly-coupled* real-time simulation,
  non-virtualized (avoiding the overhead virtualization imposes — the
  chapter cites a benchmark where a scientific application ran **53x slower**
  on a commercial cloud than on an HPC system, almost entirely from
  networking virtualization overhead), and historically 3–7x cheaper for
  DOE-scale workloads than commercial cloud equivalents.

Streaming financial data — arriving continuously, needing real-time
response, not easily parallelized across independent objects — fits neither
paradigm perfectly, but the chapter argues HPC's tools are the better
starting point.

### 22.4–22.5: HPC hardware and software

- **Hardware**: modern HPC and cloud systems use mostly the same commodity
  components (CPUs, GPUs, InfiniBand); the real differences are in storage
  architecture (concentrated/global-filesystem in HPC vs. distributed in
  cloud) and the absence of a virtualization layer in HPC.
- **MPI (Message Passing Interface)**: the standard inter-process
  communication protocol for parallel HPC computing — point-to-point and
  collective operations, portable across languages via a shared Language
  Independent Specification.
- **HDF5 (Hierarchical Data Format 5)**: the dominant HPC array-storage
  library — datasets (arrays + metadata) organized into hierarchical groups,
  with built-in compression/indexing that the chapter credits for a
  **21x** read speedup over ASCII text files on 10 years of S&P 500 data.
- **In situ / ADIOS**: processing data *as it streams* rather than writing
  it to disk first and analyzing later — motivated by CPU speed (Moore's
  Law) badly outpacing disk I/O speed over the last few decades.

### 22.6: Use cases

Six case studies, each pairing an HPC tool with a real streaming-data
problem: supernova detection (PTF telescope image classification, 3.8%
misclassification rate), fusion plasma blob tracking (KSTAR reactor,
near-real-time collaborative analysis via ADIOS's ICEE transport engine),
intraday peak electricity usage forecasting (comparing gradient tree
boosting against a purpose-built white-box model, LTAP, for utility
baseline modeling), the 2010 Flash Crash analysis itself (HDF5-accelerated
computation of VPIN and a Herfindahl-Hirschman-Index-based fragmentation
measure — **720x** speedup over the original implementation), and
non-uniform FFT analysis of natural gas futures trading data (finding a
sharp once-per-minute Fourier component consistent with TWAP algorithmic
trading).

## Why no implementation module

Every prior chapter in this repo follows the book-fidelity rule: get the
actual printed snippet(s), then implement, test, and validate against real
data. **Chapter 22 has no printed snippets to work from** — it's prose,
architecture diagrams, and benchmark charts (Figures 22.1–22.10), not
formulas or code listings. Building a "chapter deliverable" here would mean
inventing an implementation the book never specifies, which runs directly
against this project's core rule (never reconstruct from memory or fill in
gaps the source material doesn't actually contain).

Two things specifically worth noting:

- **VPIN** is one of the chapter's two named early-warning indicators
  (§22.6.4), but Ch22 doesn't define it — it just cites Easley, López de
  Prado & O'Hara (2011), the same source already implemented in this repo
  at **Chapter 19, §19.5.2** (`vpin()` in the microstructural features
  module). No duplication needed; if you want to see VPIN in action, that's
  the place.
- **HHI** (Herfindahl-Hirschman Index of market fragmentation), the
  chapter's other named indicator, is referenced only as "a variant of the
  Herfindahl-Hirschman Index" (§22.6.4) — the specific variant used is never
  spelled out. Per the book-fidelity rule, we don't guess at an unstated
  formula and implement it as if it were "the chapter's HHI."

## Optional teaching supplements (not book snippets)

Two of the chapter's ideas *are* directly applicable to data already in this
repo, so — clearly separated from the core chapter above — we built small,
real, single-machine supplements illustrating them:

```
ch22/
├── README.md                              this file (the actual chapter deliverable)
├── chapter_22_hpc_supplements.py          supplement driver script, real data
├── chapter_22_hpc_supplements.ipynb       supplement notebook, real data
├── requirements.txt
└── hpc_supplements/
    ├── __init__.py
    ├── io_benchmark.py                    Part 1: CSV vs. Parquet (§22.6.4 analog)
    ├── nufft_analysis.py                  Part 2: non-uniform FFT (§22.6.6)
    └── test_hpc_supplements.py            16 tests
```

### Part 1 — I/O format benchmark (echoes §22.6.4)

We don't have HDF5 or a real HPC cluster, but the underlying lesson —
row-based text formats force you to read and parse everything, column-
oriented binary formats let you skip what you don't need — translates
directly to a CSV-vs-Parquet comparison anyone with pandas can run. Using
our real BTC/TUSD trade data (replicated 200x for a measurable benchmark,
matching the book's own "replicated the data 10 times" approach in
§22.6.4 when their own real dataset was too small to show a difference):

| | CSV | Parquet |
|---|---|---|
| Write | 5.39s | 0.24s |
| Full-file read | 0.86s | 0.04s (**23.47x faster**) |
| File size | 99.2 MB | 11.4 MB (**8.68x smaller**) |
| Single-column ('price') read | 0.63s | 0.02s (**39.54x faster**) |

*(Real-machine confirmed, mlfinlab env. These exact timing numbers will
vary run-to-run and machine-to-machine — wall-clock benchmarks aren't
reproducible the way this project's other hand-traced values are; a
sandbox pre-check on different hardware ranged ~3.7x-6.4x read speedup.
The **direction and rough magnitude** of the effect — Parquet reads
meaningfully faster, especially for single-column access — is the
reproducible, real finding; the file-size ratio is closer to exactly
reproducible since it depends mostly on the data and compression, not
the clock.)*

The single-column result is the closest single-machine analogy to the
book's own HDF5-indexing finding (Figure 22.8: 16.95s → 4.59s, a 3.7x
speedup from indexing alone) — ours is more dramatic here because Parquet's
columnar layout skips entire unread column blocks, while CSV must scan
every row regardless of how many columns you actually want.

### Part 2 — Non-uniform Fourier transform (§22.6.6)

Trade arrival times aren't evenly spaced, which is exactly the situation
this technique is for. Implemented directly from the math (a direct
non-uniform discrete Fourier transform — no snippet exists to translate,
and our dataset, 9,205 points, is small enough that the un-approximated
direct sum is trivially fast). Applied to real BTC/TUSD price returns and
trade sizes:

- Price-return spectrum: mild peak at ~2.98 cycles/day.
- Trade-size spectrum: mild peak at ~0.51 cycles/day.
- **Honest result**: neither peak is a sharp standout the way the book's
  natural gas example was (theirs was >10x stronger than neighboring
  frequencies). Our real dataset spans ~300 trades/day over one month —
  far sparser than the book's year-long, higher-frequency futures data — so
  a once-per-minute TWAP-style signature (like the book found) isn't
  realistically resolvable here. This is a genuine data-density limitation,
  reported honestly rather than oversold, consistent with this project's
  practice of reporting real findings even when they're a null or weak
  result.

## Real-machine confirmation

**Fully confirmed 2026-08-09, mlfinlab conda env** (Python 3.10.20, pytest
9.0.3, pyarrow 14.0.2, `C:\ws\AFML`, Windows): **16/16 tests passed**,
two-pass (repo root + inside `hpc_supplements/`). Driver script confirmed
real-machine with genuine output.

One real setup snag worth recording: the first real-machine run failed 3/16
tests with a parquet-engine `ImportError` — `pyarrow` was listed in
`requirements.txt` but not yet actually installed into the `mlfinlab` env
(a gap in the handoff, not the code — flagged and fixed by explicitly
running `pip install pyarrow==14.0.2` rather than just documenting the
dependency in a file). All 16 passed on re-run.

Part 1 (I/O benchmark) results, real machine, 1,841,000 replicated rows:

| | CSV | Parquet |
|---|---|---|
| Write | 5.39s | 0.24s |
| Full-file read | 0.86s | 0.04s (**23.47x faster**) |
| File size | 99.2 MB | 11.4 MB (**8.68x smaller**) |
| Single-column ('price') read | 0.63s | 0.02s (**39.54x faster**) |

*(As expected, these exact multipliers differ from the sandbox pre-check
(3.7x-6.4x range) — wall-clock timing is inherently non-deterministic and
machine-dependent. The direction and rough magnitude of the effect is the
reproducible finding, not the exact number.)*

Part 2 (non-uniform FFT) results are **bit-for-bit identical** to the
sandbox pre-check, as expected for deterministic math: price-return
spectrum peak at 2.980 cycles/day (magnitude 0.3281), trade-size spectrum
peak at 0.505 cycles/day (magnitude 8.9660).

```powershell
conda activate mlfinlab
pip install pyarrow==14.0.2
cd C:\ws\AFML
python -m pytest ch22\hpc_supplements\ -v
cd ch22\hpc_supplements
python -m pytest -v
cd ..\..
python ch22\chapter_22_hpc_supplements.py
```

Note: `io_benchmark.py`'s tests check structural correctness (round-trip
fidelity, non-negative durations) rather than exact timing values, since
wall-clock timings are inherently non-deterministic — a deliberate,
documented departure from every other chapter's hand-traced-exact-value
convention. `nufft_analysis.py`'s tests use hand-traced exact values as
usual, since it's pure math.

## Known limitations / deferred items

- No blockers. This chapter is complete as scoped (README core deliverable
  + two optional real-data supplements), confirmed with Ethan before
  building anything, given how much it departs from every other chapter's
  pattern.
- The I/O benchmark is single-machine and format-comparative only — it does
  not and cannot reproduce the book's actual multi-hundred-core HPC cluster
  numbers (Figures 22.8, real speedups up to 720x). It illustrates the same
  underlying principle at a scale anyone can actually run.
- The non-uniform FFT supplement's real result is a genuine null/weak
  finding (no sharp periodic signature), which is reported as such rather
  than adjusted to look more like the book's own denser-data result.
