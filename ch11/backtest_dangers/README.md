# Chapter 11 — The Dangers of Backtesting

CSCV (Combinatorially Symmetric Cross-Validation) and PBO (Probability of
Backtest Overfitting), from *Advances in Financial Machine Learning*, Ch. 11.

---

## What makes this chapter different

**AFML Chapter 11 contains no numbered code snippets.** Sections 11.2–11.5 are
an *argument*, not an algorithm. Only §11.6 describes something implementable —
and it does so in prose, as seven numbered steps (after Bailey et al. [2017a]).

So `pbo.py` implements that **prose algorithm**, step-for-step, with the comment
numbering mapped directly onto the book's own. There is no snippet to diff
against; the seven steps *are* the source text.

The chapter in one line:

> **A backtest is not a research tool.** Its purpose is to *discard* bad models,
> not to improve them.

---

## Files

| file | what it is |
|---|---|
| `pbo.py` | `cscv()`, `pbo()`, `sharpe_ratio()` — §11.6, seven steps |
| `test_pbo.py` | 16-test TDD suite, hand-derived expected values |
| `chapter_11_backtest_dangers.py` | real-data demo, 4 parts |
| `chapter_11_backtest_dangers.ipynb` | paired notebook (plain English → math → code → plots) |
| `conftest.py` | BLAS thread cap = 1 (mirrors ch08/ch09) |

**Run:**

```powershell
conda activate mlfinlab
cd C:\ws\AFML\ch11\backtest_dangers
pytest test_pbo.py -v
python chapter_11_backtest_dangers.py
```

---

## The result on our real data

We built **20 real strategy trials** from the Ch10 bet-sizing pipeline on real
BTC/TUSD data — sweeping SVC `C` ∈ {0.1, 1, 10, 100} × `stepSize` ∈ {0.01, 0.02,
0.05, 0.10, 0.20} — then ran CSCV with `S=8` (70 combinations).

| | |
|---|---|
| **PBO** | **0.7286** |
| median logit λ | −0.9163 |
| mean **IS** Sharpe of the selected trial | **+0.0232** |
| mean **OOS** Sharpe of that same trial | **−0.0187** |
| mean OOS rank of the IS winner | 6.84 / 20 |
| distinct trials ever crowned "best" | 10 of 20 |

**PBO = 0.73 means selection is worse than a coin flip — it is actively harmful.**
The configuration that wins in-sample lands in the bottom third out-of-sample.
That IS→OOS sign flip (+0.023 → −0.019) is Figure 11.1's "strong and persistent
performance decay", reproduced on our own data.

### This is an honest result, not a bug

The strategy family rests on a **single feature** (`fracdiff`) over 88 events — a
limitation flagged since Ch08 (degenerate feature importance) and Ch09 (thin
tuning). A one-feature model has no real edge, and CSCV says so, loudly.

**The correct response is *not* to tune `stepSize` until PBO improves.** That
would be performing the chapter's own sin on the overfitting detector itself.
The correct response is to go back and **enrich the feature set**.

---

## Design notes

### Why this does *not* share code with Ch12's CPCV

Both enumerate combinations of time blocks, so they look like siblings. They
aren't:

| | **CSCV** (§11.6) | **CPCV** (§12.4) |
|---|---|---|
| input | PnL of **already-run** trials | raw labelled observations |
| trains a classifier? | **no** | **yes** — fits C(N,k) of them |
| purging / embargo? | **no** | **yes**, on every split |
| test blocks | fixed at S/2 | k (usually 2) |
| output | **PBO** — one probability | **φ backtest paths** → Sharpe distribution |
| question | "was my *selection* overfit?" | "what is this strategy's Sharpe *distribution*?" |

The only shared ground is "partition into blocks, enumerate combinations" — about
ten lines of `itertools`. Factoring that into `utils/` would cost a student an
indirection hop out of the chapter they're reading, for no real gain. Each
chapter stays self-contained.

### Book erratum

§11.6 states that `S=16` yields **12,780** combinations. It is **12,870**:
C(16,8) = 12,870. A digit transposition. Pinned by
`test_cscv_book_example_S16_is_12870_not_12780` so it can never enter the code.

### Deviation from the book: block sizes

§11.6 specifies submatrices "of equal dimensions" (T/S rows each), which strictly
requires `S | T`. Real data rarely obliges (here T=238, S=8). We use
`np.array_split`, so blocks differ by at most one row and **no observation is
discarded**. Trimming to a multiple of S instead would silently throw away tail
data — worse. Flagged in a `LOAD-BEARING` comment in `pbo.py`.

### PBO is imprecise — say so out loud

Measured over 40 seeds, PBO on pure zero-edge noise **averages ~0.53 but any
single draw ranges 0.04–0.99.** A single PBO number is a noisy estimate. The test
suite asserts the mean over seeds precisely so this doesn't get taught as a
precise statistic.

### `random_state=0` on SVC is load-bearing

`SVC(probability=True)` runs an internal randomised Platt-scaling CV. Without a
pinned seed it flips the predicted class run-to-run on this small dataset (a real
bug caught in Ch10). Also `n_jobs`/`numThreads = 1`: SVC-with-probability is not
spawn-safe under joblib/loky on Windows.

---

## Where §11.5's recommendations get cashed out

| recommendation | chapter |
|---|---|
| Simulate **scenarios**, not history | **Ch12** — CPCV |
| **Record every trial**, deflate the Sharpe by the number of trials | **Ch14** — Deflated Sharpe Ratio |
| Apply **bagging** | Ch6 |
| Model whole asset classes, not single securities | Ch8 |
| Don't backtest until research is complete | Ch1–10 |

> *"Backtesting while researching is like drinking and driving.
> Do not research under the influence of a backtest."*
> — Marcos López de Prado, Second Law of Backtesting
