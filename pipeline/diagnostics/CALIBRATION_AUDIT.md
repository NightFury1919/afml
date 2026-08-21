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

## Detection Power Calibration Findings (2026-08-19)

Resolves a real question Ethan raised directly this session: this pipeline's
live runs keep reading "no exploitable edge" (DSR sub-0.5 on every run so
far, PBO/CPCV/O-U all pointing the same direction per the standing
cross-method finding above) -- is that because there genuinely is no edge,
or because a real-but-small edge would be invisible at this pipeline's
actual sample size regardless? This is a different question from anything
answered so far: 2026-08-17's fix confirmed DSR's `T` is now computed
correctly (uniqueness-weighted); 2026-08-18's `calibrate_min_reliable_T.py`
confirmed DSR is *unbiased* under the null at this pipeline's T range.
Neither of those addresses whether DSR would actually detect a real edge if
one existed at this scale.

Method: `pipeline/diagnostics/calibrate_detection_power.py`, using the REAL
`ch14.backtest_statistics.deflated_sharpe_ratio()` -- not a re-derivation.
N=20 simulated trials (matching this project's real C_GRID x STEP_GRID),
one given a real injected population Sharpe (`true_sharpe` in {0, 0.05,
0.10, 0.15, 0.20, 0.30}), the rest genuine zero-edge, at T in this
project's real observed live range (50-80) plus canonical reference points
up to T=1000. 20,000 reps per (T, true_sharpe, regime) cell. Two regimes:
Part 1 Gaussian (cross-checks clean against `calibrate_min_reliable_T.py`'s
null: P[DSR>0.5]~0.47 at true_sharpe=0 across all T, see table below);
Part 2 fat-tailed (Bernoulli-jump mixture tuned to the neighborhood of
2026-08-19's real observed live skew/kurtosis -- documented as an
approximate match in the script's own docstring, not exact). Real observed
runtime: ~1 hour on real hardware (Part 2's heavier per-rep skew/kurtosis
calls) -- see script docstring for the full note.

### Finding 1: DSR is measurably miscalibrated under fat tails -- via a real, verified mechanism

Real computed null-hypothesis false-positive rate (P[DSR>0.5 | true edge=0])
at this pipeline's actual T range:

| T | Gaussian null | Fat-tailed null |
|---|---|---|
| 30 | 0.483 | 0.803 |
| 50 | 0.472 | 0.742 |
| 55 | 0.473 | 0.731 |
| 60 | 0.472 | 0.714 |
| 66 | 0.472 | 0.705 |
| 70 | 0.471 | 0.697 |
| 80 | 0.470 | 0.693 |
| 100 | 0.471 | 0.657 |
| 200 | 0.471 | 0.598 |
| 1000 | 0.469 | 0.499 |

Under the Gaussian regime, DSR is correctly calibrated at every T (~0.47-0.48,
matching `calibrate_min_reliable_T.py`'s prior finding). Under the
fat-tailed regime -- the one that actually resembles this project's real
live returns -- DSR is biased **toward false positives** at this pipeline's
real T range (0.69-0.80 vs the correct ~0.50), converging back to correct
calibration only around T=1000.

**Mechanism, directly verified (not assumed):** DSR selects the best-of-20
trial by realized Sharpe, then feeds that *same selected trial's own*
realized skew/kurtosis back into the deflation formula -- exactly mirroring
`stages.py`'s real convention. Under jump-risk conditions, selecting for
the best realized Sharpe implicitly selects for trials that, by chance,
avoided the jump/tail events in their own finite sample. Confirmed directly
in the script's own diagnostics: at T=66, the selected trial averaged ~1.1
jump events against an unconditional expectation of ~3.3. Since DSR is fed
that trial's own (understated) sample tail risk, its own selection-bias
correction is distorted by the very selection process it exists to correct
for -- specifically under fat-tailed/jump conditions.

**Practical read for this project:** every real live DSR reading so far
has come in below 0.5 (see live_run_log.csv, six real runs 2026-08-19:
0.4193, 0.4703, 0.4503, 0.2136, plus prior sessions) despite this measured
upward bias. A metric biased toward saying yes, still consistently saying
no, is a *stronger* null result than an unbiased metric saying no would be
-- this strengthens, not weakens, the standing "no exploitable edge"
finding.

### Finding 2: detection power is genuinely weak at this project's real sample size

Real computed gap between the (inflated) fat-tailed null baseline and each
true_sharpe column, at this project's real T range:

| T | gap @ true_sharpe=0.05 | gap @ 0.15 | gap @ 0.30 |
|---|---|---|---|
| 50 | 0.014 | 0.049 | 0.125 |
| 55 | 0.016 | 0.053 | 0.139 |
| 60 | 0.017 | 0.058 | 0.160 |
| 66 | 0.016 | 0.067 | 0.174 |
| 70 | 0.016 | 0.070 | 0.186 |
| 80 | 0.019 | 0.075 | 0.203 |
| 100 | 0.021 | 0.100 | 0.261 |
| 200 | 0.036 | 0.194 | 0.385 |
| 1000 | 0.146 | 0.496 | 0.501 |

This project's real live best-trial Sharpes have run 0.03-0.07 (see
live_run_log.csv) -- squarely in the `true_sharpe=0.05` column. At this
project's actual T range (50-80), the detection gap there is essentially
nothing (0.014-0.019) -- DSR's readout would look almost identical whether
a real edge of this size existed or not. Even a hypothetical edge 2-4x
larger (true_sharpe=0.15) only opens a gap of 0.05-0.08 at this same T
range. Meaningful discrimination only emerges around **T=200-1000 -- 3 to
20x more independent (uniqueness-weighted) observations than this pipeline
currently gets per live pull.**

### What this changes about the standing "no exploitable edge" finding

Not overturned -- the cross-method convergence documented above (PBO,
CPCV, O-U non-stationarity, DSR all independently pointing the same
direction) remains real and, per Finding 1, arguably understated its own
strength. But the finding needs a more honest frame going forward: this
is not "we ruled out an edge of the size actually observed in live
trading" -- it's "no method available at this project's current sample
size could reliably rule an edge of that size in or out, and every method
tried still came back negative anyway." Report/handoff language describing
the null result should reflect this distinction rather than implying a
stronger detection claim than the pipeline's real sample size supports.

### Suggested next-session priority addition

5. **Raise T_effective toward the 200+ range** where detection power
   becomes meaningful, per Finding 2 above. Candidate levers, none yet
   explored: longer `LOOKBACK_HOURS`, `CUSUM_H` recalibration (more events
   per pull, at the cost of more overlap/lower `tw` -- the same tradeoff
   already documented under `CUSUM_H`'s Tier-3 entry above), or
   accumulating multiple live pulls into a combined historical dataset
   over time (`live_run_log.csv`'s accumulating structure could eventually
   support this).

**Caveat:** the fat-tailed regime is a documented approximate match to
2026-08-19's observed live skew/kurtosis, not an exact distributional fit
(see script docstring) -- worth revisiting if live skew/kurtosis drifts
further from that observed range. Full 132-row (66 cells x 2 regimes)
result grid in `detection_power_calibration.csv`.
## CUSUM_H Staleness Audit (2026-08-21)

First real measurement of the long-deferred cross-asset staleness question
(open since the pivot from the static March 2026 BTC/TUSD dataset to live
BTC/USDT data): is `CUSUM_H=500` -- a flat DOLLAR threshold, self-flagged
in `rebuild.py`'s own docstring as an unresolved open question -- still
calibrated to the current data, and in which direction is it off?

Method: `pipeline/diagnostics/audit_cusum_h_staleness.py`. Runs the real
March static baseline AND one fresh 720h (30-day) live BTCUSDT pull
through the SAME real `rebuild.build_bars_and_labels(target_bars=250)`
and `ch02_filters.cusum_filter()` -- not a raw-trade-level proxy --
measuring the bar-to-bar CLOSE price diff distribution `CUSUM_H` is
actually applied against, on both.

### Results

| | March static | Live (30-day pull, 2026-08-21) | Ratio |
|---|---|---|---|
| n_bars | 230 | 241 | ~1.0x |
| bar close price mean | $69,670.50 | $67,000.14 | -3.8% |
| bar close price std (window dispersion) | $2,692.81 | $4,785.18 | 1.78x |
| bar-to-bar close diff std (what CUSUM_H accumulates) | $674.59 | $422.01 | **0.626x** |
| CUSUM events at h=500 | 98 (42.6% of bars) | 73 (30.3% of bars) | -- |

### Finding: `target_bars` self-correction is confirmed working; `CUSUM_H=500` is measurably too HIGH for the current regime, not too low

`target_bars=250`'s dynamic threshold produced bar counts within 5% of
each other (230 vs 241) despite the live pull having ~12x the raw trade
count of the March static data (113,367 vs 9,205) -- direct real-data
confirmation that this lever is genuinely self-correcting, as the 2026-08-17
staleness categorization predicted from reading the source alone.

The counterintuitive part: the live window's mean price level is barely
different from March's (-3.8%), and its overall price DISPERSION is much
larger (1.78x -- this 30-day window caught the current rally off a lower
base). A naive "price is higher/more volatile now, so a fixed $ threshold
fires more easily" story would predict a HIGHER CUSUM event rate today.
The opposite happened: bar-to-bar move size (the quantity `CUSUM_H`
actually accumulates against) is SMALLER now (0.626x), and the event rate
dropped from 42.6% to 30.3% of bars. Window-level dispersion and
adjacent-bar move size are not the same thing, and only the latter is what
`CUSUM_H` sees -- this is the real mechanism, not a data artifact.

**`h~313` would restore March's relative firing rate on today's data** (a
MEASUREMENT, not a new calibration decision -- see script's own closing
caveat). `CUSUM_H=500` is running ~38% higher than that today, meaning the
live pipeline is generating FEWER CUSUM events, and therefore fewer
triple-barrier bets, than the March calibration intended for a
comparably-sized bar series.

### Caveats

- **Single pull, single day.** This measures one 30-day live window as of
  2026-08-21, not a distribution across multiple pulls. BTC has been
  unusually volatile this specific week (real, independently-verified:
  ~+24% in the week leading up to this measurement) -- `h~313` should be
  treated as a rough current estimate, not a stable target, until repeated
  on a later day shows whether it holds or was itself a product of this
  week's unusual regime.
- This does NOT change the standing "no exploitable edge" finding --
  CUSUM_H's effect on event count/uniqueness was already shown (2026-08-16,
  2026-08-20) to be a T_effective/DSR consideration, not a signal-quality
  one. This section only establishes that the flat h=500 IS measurably
  stale, and in which direction, closing the "is it stale" half of the
  question this project has deferred since 2026-08-16's original CUSUM
  investigation. The "what should replace it" half (an h-per-day-volatility
  redesign, per `rebuild.py`'s own KNOWN OPEN QUESTION) remains a genuine,
  undesigned next step -- not attempted here.
- Interacts directly with the 2026-08-20 T_effective lever sweep's finding
  that LOWERING `CUSUM_H` (500->250) made T_effective WORSE (-45%, via
  `tw_mean` collapse from label overlap). That sweep tested a lower h for a
  different reason (more raw events) and found it net-negative for
  T_effective. This section's `h~313` is a much smaller reduction than that
  sweep's h=250, in a different direction of reasoning (staleness
  correction, not event-count maximization) -- whether a modest reduction
  toward ~313 avoids the uniqueness-collapse mechanism that made h=250 net
  negative is untested and should be checked before treating h~313 as
  actionable, not just descriptive.
## T_effective Lever Sweep Findings (2026-08-20)

Real follow-on to the prior section's 5th priority item: which candidate
lever(s) actually raise T_effective toward the 200-1000 range where
DSR's detection power becomes meaningful?

Method: `pipeline/diagnostics/calibrate_t_effective_levers.py`, run
against ONE frozen raw-trades snapshot (`t_effective_snapshot_2026-08-20`,
102,994 raw trades) so every config is compared against the same
underlying market data, not confounded by live drift -- same discipline
as the 2026-08-18 Tier-3 sweep. Unlike that sweep, all three levers here
(`target_bars`, `CUSUM_H`, `VERTICAL_BARRIER_NUM_DAYS`) sit upstream of
`build_bars_and_labels()`, so each config re-ran the FULL chain (rebuild
-> enrich -> stage -> Ch11's real 20-configuration SVC grid -> evaluate),
not just a downstream patch. `target_bars` was passed directly (a real
function parameter); `CUSUM_H`/`VERTICAL_BARRIER_NUM_DAYS` were
monkeypatched-and-restored as `rebuild.py` module globals -- confirmed
safe to do this way (unlike `features.py`'s ROLL_WINDOW/VPIN_WINDOW/
FFD_THRES gotcha) by reading the real source: both are referenced as bare
module-global names inside `build_bars_and_labels()`'s body, evaluated at
call time, not bound as default-argument values at def time.

MECHANISM CORRECTION to the prior section's lever list: `LOOKBACK_HOURS`
was NOT included in this sweep. Tracing the real source shows
`run_pipeline_live.py` always calls `build_bars_and_labels(raw_trades)`
with `target_bars`'s default (250) -- `compute_dynamic_threshold()`
rescales the dollar-bar threshold to hit ~`target_bars` bars regardless
of how much history was pulled. A longer live pull would NOT increase
bar count (and therefore not `T_raw`) under the pipeline's current
design. `target_bars` is the real, direct lever for "more bars";
`LOOKBACK_HOURS` was a red herring for this specific question (it
remains genuinely load-bearing elsewhere -- see its own Tier-2 entry
above -- for having enough prior history for `get_daily_vol()`).

### Results

| config | T_raw | tw_mean | T_effective | vs baseline | DSR | PBO |
|---|---|---|---|---|---|---|
| baseline | 198 | 0.3791 | 75.07 | -- | 0.5158 | 0.3950 |
| target_bars=500 | 361 | 0.3682 | 132.90 | **+77%** | 0.6692 | 0.3885 |
| CUSUM_H=250 | 197 | 0.2086 | 41.10 | **-45%** | 0.5871 | 0.2294 |
| vertical_barrier=1 day | 124 | 0.5234 | 64.90 | **-14%** | 0.3174 | 0.5195 |

Full sweep output (n_bars, n_events, n_events_enriched, best_sharpe, all
four configs) in `pipeline/diagnostics/t_effective_lever_sweep.csv`.

### Finding: only `target_bars` raises T_effective -- the other two backfire, each for a real, distinct reason

**`target_bars=500` (250->500) is the one working lever tested,** raising
T_effective 75.07->132.90 (+77%) -- real, meaningful progress toward the
200+ range. Both `T_raw` (198->361) and `tw_mean` (0.3791->0.3682, only a
small decline) moved favorably: doubling the bar-count target roughly
doubled the number of realized bet opportunities without proportionally
increasing label overlap.

**`CUSUM_H=250` (500->250) makes T_effective WORSE (-45%), not neutral.**
Raw triple-barrier events nearly doubled (48->97) exactly as expected
from a lower threshold -- but `tw_mean` collapsed 45% (0.3791->0.2086):
packing more events into the same bar window means far more overlapping
labels. The uniqueness collapse overwhelms the event-count gain. This
CONFIRMS and SHARPENS 2026-08-16's original CUSUM_H finding (which only
tested the extreme case, 500->100, and found effective T "barely moved")
-- at this more moderate perturbation, the net effect is clearly
negative, not merely flat. **CUSUM_H reduction should be considered a
net-negative lever for T_effective, not a candidate worth pursuing
further.**

**`VERTICAL_BARRIER_NUM_DAYS=1` (3->1) makes T_effective WORSE (-14%),
despite its own predicted mechanism working exactly as expected.**
`tw_mean` rose 38% (0.3791->0.5234) -- a shorter holding period really
does reduce triple-barrier label overlap, confirming the mechanism this
lever was chosen to test. But `T_raw` dropped even more (198->124): a
shorter horizon gives the winning trial's signal fewer bars with an open
position to realize a bet on in the first place. The mechanism was real;
the net effect on T_effective was still negative because the drop in bet
opportunities outweighed the uniqueness gain. **Shortening the vertical
barrier is not a viable T_effective lever on its own** -- though this
doesn't rule out combining a shorter horizon with something that
independently increases bet frequency (untested here).

### Caution on the target_bars=500 config's DSR reading

DSR rose 0.5158->0.6692 in the target_bars=500 config -- worth being
explicit that this should NOT be read as "found an edge." Interpolating
this session's own `detection_power_calibration.csv` (fat-tailed null,
T=100 -> 0.6566, T=150 -> 0.61615) puts the null-hypothesis false-positive
baseline at T=132.90 around **~0.630**. DSR=0.6692 is only ~0.04 above
that inflated null baseline -- consistent with noise at this T, not
evidence of a real signal emerging. The value of `target_bars=500` here
is purely about raising T_effective toward the range where detection
power eventually becomes meaningful (200-1000, per the prior section) --
not as a standalone result suggesting an edge was found.

### Suggested next-session priority addition

6. **Push `target_bars` further** (e.g. 750, 1000) on a fresh frozen
   snapshot, since it's the only lever confirmed to help, to see whether
   the ~1.8x T_effective gain from 250->500 continues scaling roughly
   linearly or starts to plateau (more bars per pull also means a smaller
   dynamic dollar-bar threshold -- worth checking bar quality/degeneracy
   doesn't break down at higher target_bars values). Also worth testing
   whether combining a higher `target_bars` with the untested
   `VERTICAL_BARRIER_NUM_DAYS` recovers that lever's real uniqueness gain
   without its T_raw cost.

**Caveat:** single frozen snapshot (one day's pull, 102,994 raw trades) --
real, but small-sample, same caveat as every prior sensitivity sweep in
this document. Each config here is ALSO a real, single SVC trial-grid
run (not a Monte-Carlo-averaged result the way the detection-power
calibration was) -- some of the observed DSR/PBO movement could reflect
which specific 20-trial grid happened to win under each config, not a
stable underlying effect. Worth treating this table as a real first data
point on the T_effective mechanism, not a final answer on DSR/PBO's
response to it.
## CUSUM_H Staleness Correction vs. T_effective (2026-08-21)

Direct follow-on closing the open question this same day's "CUSUM_H
Staleness Audit" section flagged in its own caveats: does the
staleness-motivated correction (h~313, a much smaller reduction than the
2026-08-20 sweep's h=250) avoid the tw_mean/uniqueness-collapse mechanism
that made h=250 net-negative for T_effective, or does any reduction below
500 trigger it?

Method: `pipeline/diagnostics/calibrate_cusum_h_correction.py`, reusing
`calibrate_t_effective_levers.py`'s real `_run_one_config()` directly (same
monkeypatch-and-restore pattern, same full rebuild -> enrich -> stage ->
Ch11 trials -> evaluate chain -- no reimplementation). Run against ONE
freshly frozen snapshot (`t_effective_snapshot_2026-08-21`, 113,497 raw
trades -- NOT the deleted 2026-08-20 snapshot, to avoid silently mixing
today's staleness finding with three-day-old market data). Three configs:
baseline (h=500), h=313 (this session's staleness-corrected value), and
h=375 (a bracketing midpoint, to see whether the effect is graded or a
step change specific to the extreme h=250 case already tested).

### Results

| config | n_events | T_raw | tw_mean | T_effective | vs baseline | DSR | PBO |
|---|---|---|---|---|---|---|---|
| baseline (h=500) | 61 | 199 | 0.3122 | 62.14 | -- | 0.9617 | 0.5952 |
| h=375 (bracket) | 84 | 192 | 0.2350 | 45.11 | **-27.4%** | 0.7711 | 0.0974 |
| h=313 (staleness-corrected) | 97 | 200 | 0.2024 | 40.47 | **-34.9%** | 0.7370 | 0.7175 |

Full sweep output in `pipeline/diagnostics/cusum_h_correction_calibration.csv`.

### Finding: NO -- the uniqueness-collapse mechanism is graded across the whole range, not specific to the extreme h=250 case

`T_raw` stays essentially flat across ALL FOUR values now tested (500, 375,
313, 250 -- 199/192/200/197 respectively, from this section plus the
2026-08-20 sweep), while `tw_mean` falls monotonically as `h` decreases
(0.3122 -> 0.2350 -> 0.2024, continuing the same trend that reached 0.2086
at h=250). More events packed into the same bar window collapses average
uniqueness at EVERY reduction tested, not just the extreme one -- this
answers the open question directly: **h~313 does NOT avoid the mechanism.
Any CUSUM_H reduction in the 250-375 range tested so far costs T_effective,
roughly in proportion to how far h drops from 500.**

This creates a real, stated tension rather than a clean fix: the staleness
audit (a real price-regime measurement, independent of T_effective) says
CUSUM_H should come down toward ~313 to match March's calibration intent
on current data. But every reduction tested actively works against this
project's separate, higher-priority goal (per the Detection Power Findings)
of RAISING T_effective toward 200-1000. **Staleness-correcting CUSUM_H
downward and raising T_effective cannot both be solved by moving this one
lever -- they pull in opposite directions.** Addressing the staleness
finding without a T_effective cost would require either the deliberate
h-per-day-volatility redesign rebuild.py's own docstring already flags
(not a simple new fixed number), or combining a CUSUM_H reduction with the
one lever independently shown to help T_effective (target_bars=500, 08-20
finding, +77%) to see whether the combined net effect clears the cost --
untested, a real next step if this thread continues.

DSR/PBO swung substantially across all three configs here (DSR
0.96->0.77->0.74, PBO 0.60->0.10->0.72) -- per this document's own
Detection Power and PBO Precision findings, neither is reliable at
T_effective=40-62 (DSR detection power is weak below T~200; PBO's noise
floor doesn't shrink with T), so none of this movement should be read as
evidence of edge one way or the other, consistent with every other finding
in this document.

**Caveat:** single frozen snapshot, single SVC trial-grid per config --
same caveats as the 2026-08-20 sweep this extends. `T_raw` values across
all four h values compared here come from two DIFFERENT snapshots taken on
different days (102,994 raw trades on 08-20 vs 113,497 on 08-21) -- the
near-identical T_raw values (197-200) across both days and all four h
values is itself informative (suggests target_bars=250's bar count, and
therefore roughly how many CUSUM events survive per bar-window regardless
of h in this range, is fairly stable day-to-day), but it means this
section's h=375/313 rows are not from the exact same underlying data as
the 08-20 sweep's h=250/baseline rows -- a same-day four-way comparison
would be a cleaner test if this needs to be revisited.
## Combined Lever: target_bars=500 + CUSUM_H=313 (2026-08-21)

Closes the exact open question the prior section flagged as untested: does
combining `target_bars=500` (the one lever independently shown to HELP
T_effective, +77% on 2026-08-20) with `CUSUM_H=313` (the staleness-
corrected value, which alone COSTS T_effective -34.9%) net out ahead of
baseline?

Method: `pipeline/diagnostics/calibrate_combined_lever.py`, reusing
`calibrate_t_effective_levers.py`'s real `_run_one_config()` directly.
Run against the SAME snapshot as the prior section
(`t_effective_snapshot_2026-08-21`, 113,497 raw trades) -- deliberately
avoiding the cross-day confound that section's own caveat flagged. Two new
configs (`target_bars_500` alone on today's data, and the combined config)
plus the baseline/`cusum_h_313`-alone rows already real-machine-confirmed
today in `cusum_h_correction_calibration.csv`, giving a clean same-snapshot
four-way comparison.

### Results

| config | T_raw | tw_mean | T_effective | vs baseline | DSR | PBO |
|---|---|---|---|---|---|---|
| baseline (tb=250, h=500) | 199 | 0.3122 | 62.14 | -- | 0.9617 | 0.5952 |
| cusum_h_313 alone | 200 | 0.2024 | 40.47 | -34.9% | 0.7370 | 0.7175 |
| target_bars=500 alone | 416 | 0.3150 | 131.05 | **+111%** | 0.7733 | 0.2846 |
| **combined (tb=500 + h=313)** | 419 | 0.2090 | **87.57** | **+40.9%** | 0.8316 | 0.2825 |

Full sweep output in `pipeline/diagnostics/combined_lever_calibration.csv`.

### Finding: YES -- the combination nets out ahead of baseline; both goals are satisfiable together, with a real (not free) trade-off

`target_bars=500`'s ~2.1x T_raw gain (199->416) is large enough to absorb
`CUSUM_H=313`'s proportional tw_mean collapse (0.3150->0.2090, -33.7% --
nearly identical to the -35.1% collapse h=313 caused alone, 0.3122->0.2024)
and still land at T_effective=87.57, a real +40.9% gain over baseline.
**Staleness-correcting CUSUM_H and raising T_effective ARE jointly
achievable via this combined config** -- the prior section's stated
tension is resolvable, not fundamental, once `target_bars` is raised
alongside the CUSUM_H correction rather than left at 250.

Honest caveat on the trade-off: relative to `target_bars=500` ALONE
(131.05), adding the staleness correction gives back roughly a third of
that gain (87.57, -33.2%). This is not "both maximized simultaneously" --
it's a real, quantified cost of also correcting CUSUM_H's staleness, not
a free combination. Whether that trade is worth taking depends on how much
weight the staleness finding itself deserves versus maximizing T_effective
alone; this section only establishes that the combination is a real,
available option, not which config should ultimately be adopted.

Incidental cross-day replication: `target_bars=500` alone landed at
T_effective=131.05 today vs 132.90 on 2026-08-20 (different snapshot,
different day, same config) -- within 1.4% of each other, a real (if
small-sample) sign that this specific config's effect is stable day to
day, not a one-off artifact of either day's particular market data.

DSR/PBO for the combined config (0.8316 / 0.2825) sit closer to the
target_bars=500-alone row's values than to cusum_h_313-alone's -- but per
this document's standing caveats, none of these are individually reliable
at T_effective=88-131 (still well below the ~200 threshold where DSR's
detection power becomes meaningful), so this is noted for completeness,
not as an edge signal.

**No further action taken here** -- this section establishes the combined
config as a real, available candidate; deciding whether to adopt it as the
pipeline's new default (vs. keeping target_bars=500 alone, vs. keeping the
current baseline pending the more principled h-per-day-volatility redesign)
is a genuine next-session decision, not made in this diagnostic session.
## target_bars=750 Scaling and Combined-Lever Interaction (2026-08-21)

Direct follow-on to two threads converging today: the still-open "push
target_bars further (750, 1000)" item (deferred since 2026-08-20), and the
prior section's Combined Lever finding at target_bars=500. Tests
target_bars=750 both alone (does the target_bars->T_effective relationship
keep scaling past 500, or plateau?) and combined with CUSUM_H=313 (does a
bigger target_bars base absorb the staleness correction's cost more
easily than target_bars=500 did?).

Method: `pipeline/diagnostics/calibrate_target_bars_750.py`, same
`_run_one_config()` reuse, same snapshot (`t_effective_snapshot_2026-08-21`)
as every other CUSUM_H/target_bars result today.

### Results

| config | target_bars | T_raw | tw_mean | T_effective |
|---|---|---|---|---|
| baseline (tb=250) | 250 | 199 | 0.3122 | 62.14 |
| target_bars=500 alone | 500 | 416 | 0.3150 | 131.05 |
| **target_bars=750 alone** | 750 | 563 | 0.3206 | **180.48** |
| combined tb=500 + h=313 | 500 | 419 | 0.2090 | 87.57 |
| combined tb=750 + h=313 | 750 | 527 | 0.1994 | 105.07 |

Full sweep output in `pipeline/diagnostics/target_bars_750_calibration.csv`.

### Finding 1: target_bars scaling is starting to plateau past 500, not still linear

250->500 (2.0x target_bars) produced a slightly SUPER-linear T_effective
gain (2.11x: 62.14->131.05). 500->750 (1.5x target_bars) produced a clearly
SUB-linear gain (1.38x: 131.05->180.48). `tw_mean` stayed essentially flat
across all three (0.3122/0.3150/0.3206 -- no real degradation), so the
plateau isn't a uniqueness-collapse story the way CUSUM_H reduction was --
`T_raw`'s own growth rate is slowing (199->416 is +109%, but 416->563 is
only +35% despite the same +50% step in target_bars this time vs. the
prior step's +100%), consistent with the pulled window's raw trade count
(113,497) starting to become a binding constraint on how many additional
bars a larger target_bars can extract. **target_bars=750 alone reaches
T_effective=180.48 -- the closest this project has come to the ~200
threshold** where the Detection Power Findings section says DSR's
detection power starts to become meaningful.

### Finding 2: the combined config's RETAINED FRACTION after adding h=313 got WORSE at tb=750, not better -- opposite of the pre-registered hypothesis

|  | combined T_eff | alone T_eff | fraction retained |
|---|---|---|---|
| tb=500 + h=313 | 87.57 | 131.05 | 66.8% |
| tb=750 + h=313 | 105.07 | 180.48 | **58.2%** |

Going in, the natural guess was that a bigger target_bars base would
absorb CUSUM_H=313's uniqueness cost more easily (more raw events to
"cushion" the collapse). The opposite happened: `tw_mean` UNDER the h=313
correction stays essentially flat as target_bars rises (0.2024 ->
0.2090 -> 0.1994 -- no real trend, note the 750 value is even slightly
LOWER than the 500 value), while `tw_mean` WITHOUT the correction keeps a
slight upward drift (0.3122 -> 0.3150 -> 0.3206). The correction's
uniqueness cost doesn't shrink relative to target_bars; if anything the
uncorrected config's uniqueness holds up marginally better at scale,
widening the relative gap rather than closing it.

**What still holds, in absolute terms:** tb=750+h=313 (105.07) beats
tb=500+h=313 (87.57) -- pushing target_bars further continues to help the
combined config in absolute T_effective, just at a worse retained-fraction
rate than the 500 step showed. Anyone treating "retained fraction" as the
metric to optimize would conclude bigger target_bars makes the staleness
correction relatively MORE costly, not less -- but anyone tracking the
combined config's raw T_effective would still want to push target_bars
higher. Which framing matters depends on what's actually being optimized
for; this section reports both rather than picking one.

**Caveat:** same single-snapshot, single-SVC-grid-per-config limitations
as every T_effective sweep in this document. target_bars=1000 (the other
value flagged in the deferred "push further" item) remains untested --
whether the plateau observed here (500->750) continues, flattens further,
or the retained-fraction trend reverses again is unknown past this point.
## target_bars=1000 -- Plateau Confirmed, Retained-Fraction Trend Reversed (2026-08-21)

Closes both open items the prior section's caveat flagged. Same method,
same snapshot (`t_effective_snapshot_2026-08-21`), same
`_run_one_config()` reuse -- `pipeline/diagnostics/calibrate_target_bars_1000.py`.

### Results (complete four-point picture)

| target_bars | T_raw (alone) | T_effective (alone) | T_raw (combined, h=313) | T_effective (combined) | fraction retained |
|---|---|---|---|---|---|
| 250 (baseline) | 199 | 62.14 | 200 | 40.47 | 65.1% |
| 500 | 416 | 131.05 | 419 | 87.57 | 66.8% |
| 750 | 563 | 180.48 | 527 | 105.07 | 58.2% |
| **1000** | 536 | **182.86** | 704 | **145.52** | **79.6%** |

Full sweep output in `pipeline/diagnostics/target_bars_1000_calibration.csv`.

### Finding 1 (confirmed, strengthened): target_bars alone has genuinely plateaued, not just decelerating

750->1000 (1.33x target_bars) produced essentially ZERO T_effective gain
(180.48->182.86, +1.3%) -- the flattest step yet, well past "decelerating"
into "converged." `target_bars=1000` alone still doesn't clear the ~200
threshold; it's plateaued at roughly the same ceiling as 750, not still
climbing toward it. Pushing target_bars beyond 1000 on this snapshot looks
unlikely to help further, though that itself is untested past this point.

### Finding 2 (REVERSED, not extended): the "retained fraction worsens with scale" read from the prior section does not hold at 1000 -- explicitly correcting that framing

The prior section's two-point read (500->750: 66.8%->58.2%) is NOT a
stable trend -- the third point reverses it (750->1000: 58.2%->79.6%,
the HIGHEST of all three). This is a real methodological lesson worth
stating plainly: two points suggested a monotonic decline; a third point
broke it. The prior section's own framing ("bigger target_bars base makes
the correction relatively more costly") should be read as describing what
happened at exactly two tested values, not a general mechanism -- it does
NOT extrapolate to 1000.

The actual driver, on inspection: retained fraction depends heavily on the
RATIO of raw event counts between the combined and alone configs at each
target_bars level, and that ratio is not stable across target_bars values
-- roughly even at 500 (T_raw 419 vs 416), lower for combined at 750
(527 vs 563), then notably HIGHER for combined at 1000 (704 vs 536 --
h=313 produced 176 raw triple-barrier events vs baseline h=500's 103 at
this particular bar count/threshold). This ratio appears to depend on how
CUSUM event timing interacts with the specific dollar-bar structure at
each target_bars level -- a regime-dependent interaction, not a clean
function of target_bars itself. No stable predictive rule for this ratio
is established here; it should be treated as noisy across target_bars
values, not modeled as a trend.

### Practical result: the combined config keeps improving even though the alone config has flattened

Combined T_effective climbs cleanly across all three points tested
(87.57 -> 105.07 -> **145.52**), unlike the alone config's plateau after
750. **target_bars=1000 + CUSUM_H=313 is the strongest T_effective result
(combined or alone-with-staleness-correction) found in this entire day's
work.** If the practical goal is the best available T_effective WITH the
staleness correction in place (rather than target_bars alone, which
plateaus and never incorporates the correction), this is the candidate
config this session's evidence points to.

DSR/PBO at tb=1000 alone (0.5029 / 0.7294) are the weakest of the three
target_bars-alone points -- DSR near the pure-null 0.5 mark, PBO high --
but per this document's standing caveats, T_effective=182.86 is still
below the ~200 reliability threshold, so this reads as more noise, not a
meaningful signal shift.

**No further target_bars values tested.** This closes the target_bars
scaling question as scoped (250/500/750/1000, alone and combined) for
today's session. Whether target_bars=1000+h=313 should become the
pipeline's actual default remains an explicit next-session decision, same
status as the prior section's combined-config candidate.