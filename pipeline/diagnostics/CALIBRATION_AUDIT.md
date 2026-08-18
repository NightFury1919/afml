# Calibration Audit â€” Set Constants Across the Live Pipeline
**Compiled:** 2026-08-16, from the real committed source (not from memory)

## Purpose

A catalog of every hardcoded/calibrated constant feeding the live pipeline's
headline numbers (PBO, DSR, OTR stationarity, signal), sorted by **how much
epistemic weight each one can bear**. The goal is not to "fix" every entry â€”
some are legitimate modeling choices with no correct answer to find. The
goal is to know, for each one, *why* it has the value it has, and whether
that's ever been checked against real sensitivity.

## Tier definitions

- **Tier 1 â€” Book-sourced.** The book's own literal snippet default or most
  common worked-example value. Faithful by construction; NOT independently
  validated for BTC/USDT.
- **Tier 2 â€” Empirically anchored.** Derived from real data via a stated,
  re-checkable procedure.
- **Tier 3 â€” Trial-and-error / undocumented provenance.** No book citation,
  no data-anchored derivation on record.
- **Tier 4 â€” Explicitly invented, self-labeled as such in the code already.**
  The healthiest tier â€” nobody is pretending these are more than judgment
  calls.
- **Modeling choice (MC)** â€” not a calibration question at all. Changing the
  value tests a *different strategy*, it doesn't get you "closer to truth."

## The audit

| Constant | Value | File | Tier | Notes |
|---|---|---|---|---|
| `cv` (PurgedKFold splits) | 3 | ch09 `clfHyperFit` (Snippet 9.3) | 1 | Book's own literal default |
| `pctEmbargo` default | 0. | ch07 `PurgedKFold.__init__` | 1 | Book's own literal default (opt-in) |
| `pt_sl` | `[1, 1]` | `rebuild.py` `PT_SL` | 1 / MC | Book's most common worked example; also a real strategy-horizon choice, not just a calibration |
| `target_bars` | 250 | `rebuild.py` `compute_dynamic_threshold` | 2 | Explicitly matched to static dataset's real measured density (249 bars / ~9,205 trades) â€” re-checkable |
| `CUSUM_H` | 500 (fixed $) | `rebuild.py` | 3 | **Confirmed trial-and-error** (2026-08-16 conversation). Sensitivity-tested 2026-08-16: swinging 500â†’100 didn't change the "no edge" finding, but DID nearly triple raw event count while barely moving *effective* (uniqueness-weighted) sample size â€” see DSR note below |
| `MIN_RET` | 0.005 | `rebuild.py` | 3 / MC | No documented derivation found. Book leaves this to the user (transaction-cost dependent) â€” likely a genuine MC, not a calibration gap |
| `VERTICAL_BARRIER_NUM_DAYS` | 3 | `rebuild.py` | MC | Defines the holding-period being tested. Not "wrong" at 3 â€” it's a choice. **But this is the mechanism behind the DSR/uniqueness problem below â€” worth revisiting jointly with CUSUM_H, not in isolation** |
| `DAILY_VOL_SPAN0` | 100 | `rebuild.py` | 1 | Matches Ch03's own Snippet 3.1 example span |
| `ROLL_WINDOW` | 20 | `features.py` | 3 | "Carried over unchanged from Ch19's established calibration" â€” no derivation on record |
| `VPIN_WINDOW` | 10 | `features.py` | 3 | Same as above |
| `FFD_THRES` | 0.01 | `features.py` | 1 | **Corrected 2026-08-18**: matches the book's own Snippet 5.4 worked example (`fracDiff_FFD(df1,d,thres=.01)` in `plotMinFFD()`), which produced Fig 5.5's ES1 stationarity result at d=0.35. Snippet 5.3's function signature default is a different value (`1e-5`) â€” that's not what the book's worked example actually used. |
| `mesh_n_iter` / `mesh_points` | 2000 / 8 | `risk_context.py` `compute_otr_finding` | 1 | "Mirror chapter_13_otr.py's real Part C exactly" â€” inherited from an established chapter default |
| `random_state` (OTR mesh) | 7 | `risk_context.py` | 4 | Arbitrary seed, doesn't affect the finding's validity, only reproducibility |
| `S` (CSCV splits for PBO) | 12 (was 8) | `stages.py` `evaluate_overfitting` default | **2 (2026-08-18)** | **Corrected 2026-08-18**: empirically derived via calibrate_pbo_precision.py's null-hypothesis Monte Carlo -- std_pbo improves 0.2174->0.2051 at S=8->12 (real T=237/N=20 scale), diminishing returns past S~10-12, real cost ~1.3s/call. See "PBO Precision Calibration Findings" section below. |
| **DSR's `T`** | raw `n_events` (not uniqueness-weighted) | `stages.py` `evaluate_overfitting` | **3 â€” flagged as a likely bug, not just an uncalibrated constant** | **2026-08-16 finding: `rebuild_result['tw']` (Ch04's real average uniqueness) is computed but never fed into `deflated_sharpe_ratio()`. Confirmed via controlled same-pull comparison: raw T went 45â†’140 (CUSUM_H 500â†’100) while uniqueness-weighted effective T stayed ~19.0â†’19.6. DSR swung 0.59â†’0.89 on data that carried almost no additional real information.** |
| `min_reliable_T` | 150 | `report.py` `build_report` | 4 | Already self-documented: "a heuristic, not a statistically derived cutoff" |
| `min_reliable_trials` | 10 | `report.py` `build_report` | 4 | Same â€” explicit heuristic |
| DSR confidence bands | 0.5 / 0.95 | `report.py` `_confidence_band` | 4 | Round numbers, not book-derived |
| `DEFAULT_MAX_POSITION_FRACTION` | 0.10 | `oversight.py` | 4 | Self-labeled "arbitrary, invented judgment call" |
| Circuit breaker thresholds | PBO>0.5, P[fail]>0.5 | `oversight.py` | 4 | Self-labeled "simple, round, and arbitrary" |
| `PAPER_CAPITAL_USD` | $10,000 | `run_pipeline_live.py` | 4 | Self-labeled arbitrary |
| `LOOKBACK_HOURS` | 720 | `run_pipeline_live.py` | 2 | Live-confirmed minimum for `get_daily_vol()` to have prior bars â€” empirically anchored, not arbitrary |

## What this tells us, honestly

- **Most Tier-1/Tier-2 entries are fine as-is** â€” they're either the book's
  own stated defaults or have a real, checkable derivation. Not the priority.
- **The DSR uniqueness gap is the highest-priority finding of this whole
  audit** â€” it's not "we don't know if 500 is right," it's "the DSR formula
  is being fed a number (T) that's provably wrong given data this project
  already computes." This affects every DSR value ever reported by this
  pipeline, not just today's experiment.
- **Tier 3 entries that never got sensitivity-tested** (`ROLL_WINDOW`,
  `VPIN_WINDOW`, `S`) are the next-priority audit targets â€” not because
  they're necessarily wrong, but because nobody has checked whether the
  findings are robust to them, the way we just checked CUSUM_H.
  `FFD_THRES` was reclassified to Tier 1 on 2026-08-18 (it matches the
  book's own Snippet 5.4 value) but is still included in this session's
  sweep alongside the genuine Tier 3 entries, since tier and sensitivity
  are separate questions -- book-sourced doesn't mean untested on THIS
  dataset.
- **`MIN_RET` and `VERTICAL_BARRIER_NUM_DAYS` are probably modeling choices,
  not calibration gaps** â€” but `VERTICAL_BARRIER_NUM_DAYS` is directly
  implicated in the DSR/uniqueness mechanism, so it can't be cleanly
  separated from that fix.

## Suggested next-session priority order

1. **Fix DSR's `T`** to be uniqueness-weighted (or at minimum, report both
   raw and effective T so the reader can judge) â€” `stages.py`'s
   `evaluate_overfitting`, with its own TDD suite given how load-bearing
   DSR is.
2. Re-run the existing live pipeline once fixed, to see whether prior
   reported DSR values (0.37â€“0.55 range) move, and by how much.
3. Sensitivity-scan `ROLL_WINDOW`, `VPIN_WINDOW`, `FFD_THRES`, `S` the same
   way CUSUM_H was scanned 2026-08-16 â€” cheap, reuses the same
   monkeypatch-and-compare pattern already built.
4. Revisit `VERTICAL_BARRIER_NUM_DAYS` jointly with the DSR fix, since it's
   the mechanism (not CUSUM_H) actually driving the overlap problem.


## Sensitivity Sweep Findings (2026-08-18)

Full sweep of the Tier-3 constants flagged in the "Suggested next-session
priority order" above (`ROLL_WINDOW`, `VPIN_WINDOW`, `FFD_THRES`, `S`),
run against ONE frozen live-data snapshot (99,365 raw trades, 238 bars,
41 events -- see `pipeline/diagnostics/sensitivity_snapshot_2026-08-18/`
and `sensitivity_scan.csv` for the full artifacts) so every comparison is
against the same underlying market data, not confounded by live drift.

| Constant | Value | T_eff | DSR | PBO | n_events |
|---|---|---|---|---|---|
| baseline | default | 50.13 | 0.5308 | 0.1571 | 40 |
| S | 4 | 50.13 | 0.5308 | 0.3333 | 40 |
| S | 8 (=baseline) | 50.13 | 0.5308 | 0.1571 | 40 |
| ROLL_WINDOW | 10 | 21.87 | 0.4663 | 0.5000 | 41 |
| ROLL_WINDOW | 40 | 40.34 | 0.6274 | 0.7429 | 38 |
| VPIN_WINDOW | 5 | 62.10 | 0.5254 | 0.7571 | 40 |
| VPIN_WINDOW | 20 | 49.00 | 0.5191 | 0.5143 | 40 |
| FFD_THRES | 1e-5 | -- | -- | FAILED (see note below) | -- |
| FFD_THRES | 0.05 | 46.76 | 0.5366 | 0.2000 | 40 |

**DSR stayed stable, PBO did not.** Across every constant and value
tested, DSR stayed in a narrow band (0.4663-0.6274) -- never leaving "no
reliable edge" territory. PBO ranged from 0.1571 to 0.7571, a ~5x swing,
purely from Tier-3 constants that were never previously validated.
`S=4` vs `S=8` alone -- same trained models, nothing else changed -- moved
PBO from 0.157 to 0.333, consistent with CSCV's combinatorial split count
(`C(4,2)=6` vs `C(8,4)=70`) making PBO noisier at lower `S`. `ROLL_WINDOW`/
`VPIN_WINDOW` swings also shrink `n_events` (38-41 vs baseline 40),
suggesting part of PBO's instability is the same small-sample fragility
DSR had before the 2026-08-17 uniqueness-weighting fix -- except PBO has
never received an equivalent correction. **Flagged as a candidate for
next-session priority: is PBO's own estimator reliable at this pipeline's
sample sizes, the same question DSR already had to answer.**

**FFD_THRES=1e-5 failure is a genuine finding, not a script bug.**
`get_weights_ffd(d=0.1, thres=1e-5)` requires ~4,075 weights before the
series crosses that threshold; `get_weights_ffd(d=0.1, thres=0.01)` only
needs ~7. Against this pipeline's ~238-bar live series, a window of 4,075
leaves ZERO valid output rows once `frac_diff_ffd()` drops the warmup
period, so `adfuller()` receives an empty array. This is not a bug in
Ch05's `find_min_ffd.py`/`frac_diff_ffd.py` (both real-machine-confirmed,
TDD-tested chapter deliverables) -- it's a real constraint: `FFD_THRES`'s
practical floor is data-length-dependent, and `1e-5` (the function's own
literal signature default, never actually used in the book's own Snippet
5.4 worked example) is well past that floor for a series this short. This
reinforces the FFD_THRES Tier reclassification above: `0.01` isn't just
what the book's own worked example used, it's close to necessary for a
live pull of this size.

**Caveat:** all of the above is from ONE frozen snapshot on ONE day's
live pull (41 events). Real, but small-sample -- worth treating as a
first data point, not a settled conclusion, especially for the PBO
volatility finding.

## PBO Precision Calibration Findings (2026-08-18)

Resolves the "candidate for next-session priority" flag raised earlier
today in "Sensitivity Sweep Findings" -- is PBO's own estimator reliable
at this pipeline's sample sizes, the same question DSR already had to
answer on 2026-08-17?

Method: `pipeline/diagnostics/calibrate_pbo_precision.py`, null-hypothesis
Monte Carlo mirroring `calibrate_min_reliable_T.py`'s structure exactly --
N=20 zero-true-edge trials (matching C_GRID x STEP_GRID), T=237 i.i.d.
standard-normal bar-level PnL (this pipeline's real bar count), fed
through the REAL `ch11.backtest_dangers.pbo.pbo()` -- not a re-derivation.
S grid capped at 12 (S=16's 12,870 combinations were ruled out on
runtime grounds during today's sensitivity-sweep planning); n_reps=300
(vs DSR's 20,000) for the same reason -- a documented precision-vs-
runtime tradeoff, not free of its own Monte Carlo noise.

| S | n_combinations | mean_pbo | std_pbo | p05_p95_width | elapsed |
|---|---|---|---|---|---|
| 4 | 6 | 0.4994 | 0.2666 | 0.8333 | 3.0s |
| 6 | 20 | 0.5035 | 0.2246 | 0.7000 | 7.2s |
| 8 | 70 | 0.4940 | 0.2174 | 0.7000 | 26.9s |
| 10 | 252 | 0.4954 | 0.2066 | 0.6669 | 95.5s |
| 12 | 924 | 0.4925 | 0.2051 | 0.6451 | 386.4s |

**PBO is unbiased at every S tested** (mean_pbo stays within 0.008 of 0.5
throughout) -- mirrors DSR's own 2026-08-17 finding that small samples
don't produce falsely confident readings. Choosing S is therefore purely
a PRECISION question, not a bias correction.

**PBO's noise floor is large and does NOT shrink with T the way DSR's
does.** std_pbo at the best-tested S (12) is 0.2051 -- still far wider
than DSR's own asymptotic floor (std_dsr~0.157 at T=1000, per
calibrate_min_reliable_T.py). This is a structural difference between the
two metrics, not a bug in either: DSR is a continuous statistic with a
sqrt(T)-type precision improvement; PBO is literally a fraction of
C(S,S/2) discrete combinations, bounded by how many combinations exist
and by N (trial count), independent of T.

**Diminishing returns kick in around S=10-12.** std_pbo drops 18.5% from
S=4->8, but only 5% from S=8->10, and under 1% from S=10->12 -- despite
S=12 costing over 4x the runtime of S=10. This grid was capped at S=12 for
runtime reasons, NOT because precision is proven to plateau there --
whether it improves further past S=12 remains untested.

**2026-08-18's sensitivity sweep is RESOLVED as noise, not signal.** The
sweep's own observed PBO range (0.1571-0.7571, width 0.60) is NARROWER
than the pure-null p05-p95 width at this pipeline's actual S=8 (0.70).
The sweep's PBO swings cannot be distinguished from S=8's own known
sampling noise -- there is no evidence in that data that ROLL_WINDOW,
VPIN_WINDOW, or S itself were driving the swing, only that PBO is this
noisy at this scale regardless.

**Action taken:** `stages.py`'s `evaluate_overfitting()` default and both
`run_pipeline.py`/`run_pipeline_live.py`'s explicit calls changed S: 8 ->
12 (real cost ~1.3s/call, trivial next to live-run model training time).
`evaluate_overfitting()` now returns `S` in its result dict so
`report.py` can reference the actual value used rather than hardcoding
one (avoiding the exact silent-drift failure mode found and fixed in
`portfolio_oversight/oversight.py`'s `min_reliable_T` earlier today).
`report.py` now carries an unconditional PBO precision caveat (not gated
on small_sample, since PBO's noise persists even at this pipeline's real
T=237/N=20 scale) alongside a corrected small-sample warning that no
longer conflates PBO's S/N-driven noise with DSR's T-driven noise.

**Caveat:** S grid capped at 12 for runtime reasons; whether further
runtime investment (larger S, more reps, possibly overnight) would show
continued improvement or a genuine plateau is not yet known.
