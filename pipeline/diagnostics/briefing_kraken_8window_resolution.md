# Briefing: Kraken live PBO anomaly resolved via 8-window replication

**Status: CLOSED. Supersedes the earlier draft ("trending-regime PBO
artifact" / 2026-09-10 morning), which was written mid-investigation
before the real resolution below.**

## One-line summary

Wiring Ch12 CPCV into the live pipeline surfaced an anomalously low PBO
reading on one Kraken BTC/USD window (0.0087-0.0498 across most
`target_bars`, vs. the established ~0.78-0.83 static-dataset baseline).
Six single- and two-window diagnostics run across that day each found a
plausible-looking explanation -- a feature artifact, a trending-regime
effect, an exchange-specific construction bug -- and each was
contradicted by the next independent check. The anomaly was resolved not
by a cleverer diagnostic, but by gathering enough real independent data
to answer the question properly: **8 non-overlapping 30-day Kraken
windows spanning ~7 months (Feb-Sep 2026) show a mean PBO of 0.4876,
statistically indistinguishable from this project's own calibrated null
distribution (mean 0.4925, std 0.2051 at S=12; z=-0.07, p=0.95).** The
single-window anomaly was noise. This is the same real, book-consistent
"no exploitable edge" conclusion the static dataset, three prior
Binance.US live runs, and the meanshift investigation all reached
independently -- now confirmed a fifth way, and for the first time with
genuine multi-window statistical power behind it rather than a single
snapshot.

## The investigation, in order (2026-09-10)

1. **Trigger**: Ch12 CPCV wired into the live pipeline (per Ethan's
   decision to add Ch08/Ch12), first live run on a fresh Kraken 720h
   snapshot ("window 1") showed PBO 0.0087-0.0498 across
   `target_bars`=2000-5000 -- dramatically below the established
   baseline.
2. **Feature ablation** (drop `entropy_rate`/`structural_break_stat`,
   the two not-yet-live-validated Ch17/18 features): ruled out. PBO
   stayed low or got more extreme without them.
3. **Trending-regime hypothesis**: window 1 had a real 21% BTC rally
   (log-close R²=0.63). Detrending (excess returns) confirmed this
   explained `target_bars`=1000 cleanly (PBO 0.26->0.45, winner's
   Sharpe went negative) but NOT `target_bars`=2000-5000 (detrended
   winner stayed positive, 40-75% of raw Sharpe retained).
4. **Window 2 replication** (flat period, no trend): null-consistent,
   but uninformative for the trend hypothesis specifically since there
   was no trend to test against.
5. **Binance.US comparison** (fresh pull, comparable ~20% trend, same
   calendar period): mixed, not a clean replication -- 2/5
   `target_bars` configs flipped to negative detrended Sharpe, one came
   back fully null.
6. **DSR-vs-null-baseline gap** (via the newly-parallelized
   `calibrate_kraken_detection_power.py`, Ch20's `mp_job_list` wired in
   for a ~4x speedup on this step): both Kraken windows showed
   positive gaps regardless of trend; Binance was mixed. Looked like an
   exchange-specific effect.
7. **Cross-sectional trial-Sharpe skew** (built specifically to test
   #6): flatly contradicted it. Window 1 skew +0.44, window 2 skew
   **-0.42** (opposite sign), Binance +0.84 (largest of the three). No
   Kraken-vs-Binance split survived.

Six checks, five different candidate explanations, none surviving
independent re-testing. Diagnosed as a sample-size problem, not an
analysis problem: T_effective in the ~50-650 range, 20 non-independent
trials per run (shared return series, overlapping hyperparameter grid),
and 5 `target_bars` values per window that are resamplings of one
underlying dataset, not independent replications.

## The resolution: 8 independent windows

Captured via `capture_kraken_windows_chain.py` (new script), chaining
backward from window 2 with each window's boundary computed upfront
from a single fixed real anchor timestamp (not from a prior window's
actual downloaded data), avoiding the sequential-dependency design that
would otherwise have made this impractically slow. One real bug found
and fixed along the way: `ingestion_kraken.py` only checked HTTP 429 for
rate-limiting; a concurrent-pull test revealed Kraken's public endpoint
can also return `{"error": ["EGeneral:Too many requests"]}` on a normal
HTTP response, which fell through the old retry logic as an
unconditional raise. Fixed to route that error text through the same
backoff-retry path. Concurrency itself (tested at `--max-concurrent 2`)
triggered this on the very first attempt; all 8 windows were ultimately
captured sequentially (`--max-concurrent 1`), ~30-46 minutes each.

`pool_kraken_pbo_across_windows.py` (new script) compares two views
deliberately: a naive pooled 40-row view (overstates independence, since
the 5 `target_bars` rows per window are correlated resamplings) and the
real answer -- the per-window mean PBO, one number per window, 8
genuinely independent values:

| window | mean PBO across target_bars |
|---|---|
| window1 | 0.068 |
| window2 | 0.673 |
| window3 | 0.604 |
| window4 | 0.698 |
| window5 | 0.284 |
| window6 | 0.652 |
| window7 | 0.344 |
| window8 | 0.576 |

Mean of these 8 = 0.4876. Null distribution (S=12, this project's
standard S, from the already-calibrated `pbo_precision_calibration.csv`)
= mean 0.4925, std 0.2051. **z = -0.07, two-sided p = 0.95.** The 8
window means themselves range widely (0.068 to 0.698) -- exactly the
scatter pure noise around a ~0.49 center produces, not a population
clustered near zero the way a real persistent edge would look.

(Housekeeping: windows 3-8's CSVs each contain the same 5 rows twice,
from a duplicated batch invocation -- confirmed harmless, since
`calibrate_kraken_target_bars.py` is fully deterministic on frozen local
data with no randomness anywhere in its path, so the duplicate rows are
bit-identical and don't affect the per-window mean. Worth deleting the
extras before committing, not worth re-running.)

## Bottom line

No exploitable edge in live Kraken BTC/USD data across ~7 months of real
history, tested with genuine multi-window statistical power. This is the
fifth independent method (after static-dataset PBO/CPCV/DSR, the entropy
corroboration, and now this) converging on the same real finding this
project has held since the original static-dataset investigation.
Recommend folding this into the next boss briefing alongside the TAO
null and the meanshift/BTC-replication results -- four items now ready
together.

## Infrastructure produced along the way (reusable beyond this specific finding)

- Ch08 (MDA/SFI feature importance) and Ch12 (CPCV) wired into the live
  pipeline.
- `calibrate_cpcv_groups.py` -- two-phase N/k feasibility+Sharpe sweep
  for live CPCV, with `--drop=` (feature ablation) and `--detrend`
  flags.
- `calibrate_pbo_detrended.py` -- side-by-side raw vs. trend-adjusted
  PBO/DSR on the same real trial grid.
- `diagnose_window_trend_bias.py` -- window trend + naive-directional-
  baseline diagnostic.
- `diagnose_trial_sharpe_skew.py` -- cross-sectional skew/kurtosis of
  the 20 trial-level Sharpes (previously never saved anywhere).
- `capture_binance_snapshot.py` -- Binance.US live snapshot capture,
  matching the Kraken snapshot schema.
- `capture_kraken_windows_chain.py` -- multi-window, upfront-boundary
  Kraken capture (the tool that actually resolved this).
- `pool_kraken_pbo_across_windows.py` -- multi-window PBO pooling
  against the calibrated null distribution.
- `calibrate_kraken_detection_power.py` -- parallelized via Ch20's real
  `mp_job_list` engine (~4x speedup), the first live use of Ch20 in this
  project's pipeline.
- Real bug fixes: `ingestion_kraken.py`'s JSON-body rate-limit gap;
  `calibrate_pbo_detrended.py`'s near-zero-denominator ratio display.

## Still open / next session

1. Fix the Sept 8 backlog's CSV-overwrite bug (`measure_real_btc_roll_c_
   signal_sweep.py`'s `--out-csv` argument) -- still not done, blocking
   that session's commit.
2. Commit ALL of today's + Sept 8's work as clearly separated units,
   following the established explicit-filename discipline.
3. Delete the duplicate rows in `kraken_target_bars_calibration_window
   {3-8}.csv` before committing.
4. Ch09 log-uniform prototype (`prototype_ch09_loguniform_trials.py`) --
   drafted but never actually run against the static baseline; still an
   open "try it and see" item from earlier today.
5. Boss briefing now has four ready items: TAO null, meanshift/BTC
   replication, this 8-window Kraken resolution, and (pending #1-2)
   Sept 8's window-1/window-2 BTC replication work.
