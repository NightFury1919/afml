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
BTC/TUSD data — sweeping SVC `C` ∈ {0.01, 0.1, 1, 10} × `stepSize` ∈ {0.01, 0.02,
0.05, 0.10, 0.20} — then ran CSCV with `S=8` (70 combinations).

Since 2026-07-21 these trials run on the **12-feature enriched training table**
(87 events: `fracdiff` + Ch19's 11 microstructural features), not the original
single-feature table. The `C` grid was shifted down by one decade to match —
Ch09's tuned optimum on the enriched data is `C=0.01`, below the old grid's
floor. It was **swapped**, not extended, deliberately: the trial count is an
input to PBO, so holding it at 20 keeps the new number comparable to the old.

| | before (1 feature) | **after (12 features)** |
|---|---|---|
| **PBO** | 0.7286 | **0.8286** |
| median logit λ | −0.9163 | **−1.2993** |
| mean **IS** Sharpe of the selected trial | +0.0232 | **+0.0882** |
| mean **OOS** Sharpe of that same trial | −0.0187 | **+0.0086** |
| mean OOS rank of the IS winner (1 = worst) | 6.84 / 20 | **6.12 / 20** |
| distinct trials ever crowned "best" | 10 of 20 | **11 of 20** |

**PBO = 0.83 means selection is worse than a coin flip — it is actively harmful.**
The configuration that wins in-sample lands in the bottom third out-of-sample,
and it keeps under 10% of its in-sample Sharpe when it gets there
(+0.0882 → +0.0086). That is Figure 11.1's "strong and persistent performance
decay", reproduced on our own data.

### This is an honest result, not a bug — and the follow-up is the lesson

The first version of this chapter ran on a **single feature** (`fracdiff`) over
88 events — a limitation flagged since Ch08 (degenerate feature importance) and
Ch09 (thin tuning). PBO came out at 0.73, and this README concluded: a
one-feature model has no real edge, so **go and enrich the feature set**.

That is exactly what happened. Ch19 was pulled forward out of book order to
build 11 microstructural features, and on 2026-07-21 this chapter was re-run on
the enriched table.

**PBO got worse: 0.73 → 0.83.**

That deserves sitting with, because the naive reading of the numbers says the
opposite. Every individual trial improved — full-sample Sharpes moved from
mostly negative to almost entirely positive, and the in-sample winner's Sharpe
nearly quadrupled (+0.0232 → +0.0882). Anyone reading only the Sharpe column
would report a large improvement. But the winner's *out-of-sample* rank fell
(6.84 → 6.12 of 20) and PBO rose 10 points: the selection procedure became less
trustworthy at the same time as the backtest became more flattering.

Eleven extra features on 87 observations bought **capacity to fit noise**, not
edge. In-sample performance cannot distinguish the two — which is the entire
reason CSCV exists.

So the honest conclusion is stronger than the original one, not weaker. It is
not "this strategy family is thin"; it is "this strategy family has no edge that
survives selection, and adding features made that harder to see, not easier."

**The correct response is still *not* to tune `stepSize`, or the feature set,
until PBO improves.** That would be performing the chapter's own sin on the
overfitting detector itself — and this re-run is a concrete demonstration of how
easily that sin would have gone unnoticed. Section 11.4: a backtest's purpose is
to discard bad models, not to improve them.

This is now one of four independent diagnostics on the same enriched pipeline
that agree: Ch11's PBO (0.83), Ch12's CPCV Sharpe spread, Ch13's O-U
non-stationarity, and Ch14's DSR (0 of 5 paths survive at 0.95). Before this
migration Ch11 was still running on the old table, so that agreement was being
claimed across two different datasets. It is now genuine.

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
