# Briefing draft: a trending-regime artifact in live PBO, not a new edge

**Status: DRAFT — based on tb1000 only. Pending confirmation from the
tb2000-5000 trend/bias check and the detrended PBO/DSR re-run before this
goes in front of the boss as a settled finding.**

## One-line summary

A CPCV/PBO sweep against live Kraken BTC data (Sept 8 snapshot) produced
PBO readings of 0.0087-0.0498 across five `target_bars` values —
dramatically lower than this project's established ~0.78-0.83 "no edge"
baseline (three Binance.US live runs + the static dataset). Investigating
why turned up a specific, mechanistic explanation: this window's BTC
price trended 21.5% over 30 days (log-close R²=0.63), and the winning
trial's actual Sharpe (+0.0122) was less than a quarter of what a
trivial always-long baseline would have scored (+0.0578) on the same
bars, despite the winning trial being 95.9% long itself. That pattern —
low PBO, positive CPCV path Sharpes, but the "winning" model actually
underperforming naive buy-and-hold — is the signature of shared
directional exposure inflating rank-stability across all 20 trial
configs, not genuine feature-driven skill.

## Why this is worth reporting even in draft form

Two ablation/diagnostic passes already ruled out the more mundane
explanations before this one was identified:

1. **Not the new features.** Dropping `entropy_rate` and
   `structural_break_stat` (the two features flagged as not yet
   validated on live Kraken data) left PBO just as low, or lower —
   0.0087 → 0.0130 at tb1000. Whatever's happening isn't coming from
   those two.
2. **Not (yet ruled out as) a Kraken-specific artifact** — worth noting
   the established ~0.78-0.83 baseline is entirely Binance.US; this is
   the first time CPCV/PBO has been run against live Kraken data at all.
   Still an open question, separate from the trend finding below.

If the trend explanation holds up across the full sweep, this becomes a
genuine methodological finding independent of whether BTC has an edge:
**PBO/CPCV as currently applied to live windows has no check for whether
the window itself trended**, and a strong trend can produce a
low-PBO reading that looks identical to a real detected edge from the
metrics alone. That is a blind spot worth fixing (see Next steps),
regardless of what this specific window turns out to mean for BTC.

## Evidence so far (tb1000 only)

| Metric | Value |
|---|---|
| PBO (raw) | 0.0087-0.0498 across tb1000-5000 |
| Established live/static baseline | ~0.78-0.83 |
| Window total return (tb1000) | +21.5% |
| log(close) OLS R² (tb1000) | 0.6286 |
| Winning trial Sharpe (tb1000) | +0.0122 |
| Naive always-long Sharpe (tb1000) | +0.0578 |
| Winning trial % long (tb1000) | 95.9% |
| Ablation (drop entropy/structural-break) | PBO unchanged/worse — rules this out |

## What's NOT yet confirmed

- Whether tb2000-5000 show the same winner-underperforms-naive-long
  pattern (script run, results pending).
- The actual detrended PBO/DSR numbers (script written, not yet run) —
  tb1000's trend/bias numbers are suggestive but the direct
  excess-return recomputation is the real test.
- Whether this is BTC-window-specific or would also appear on a
  Binance.US live pull over a similarly trending period — i.e. whether
  it's really about the trend, or about Kraken specifically.
- No claim here about window 2 (the second Kraken snapshot from the
  Sept 8 session) — deliberately not run yet, since replicating a
  possible artifact doesn't distinguish it from replicating a real
  effect.

## Recommended next steps before this joins the briefing queue for real

1. Run `diagnose_window_trend_bias.py` on tb2000-5000 (same window) —
   confirm the naive-long-beats-winner pattern holds throughout, not
   just at tb1000.
2. Run `calibrate_pbo_detrended.py` — get the actual excess-return
   PBO/DSR numbers, not just the indirect trend/bias evidence.
3. Only then decide whether window 2 replication or a Binance.US
   comparison is the more informative next step.
4. If confirmed, consider whether `evaluate_overfitting()` itself should
   gain a trend-check warning (mirroring its existing T<150/n_trials<10
   warnings) — flagging when a window's log-close R² is high enough that
   PBO should be read with this caveat attached, even outside this
   specific investigation.
