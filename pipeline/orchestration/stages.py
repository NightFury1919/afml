"""
pipeline/orchestration/stages.py

Orchestration layer chaining real, already-tested AFML chapter modules into
one continuous flow: real bar-level mark-to-market trial construction ->
PBO / DSR overfitting diagnostics -> bet-sizing signal.

PHASE 1b (2026-08-12): this module was rewritten to reuse Ch11's own real,
established trial-construction pipeline (chapter_11_backtest_dangers.py's
part_c_build_trials / out_of_sample_probs) directly, rather than building a
parallel, weaker version. Phase 1a's first draft used 3 ad-hoc classifiers
on 87 raw event-level `ret * pred` pairs and produced a DSR of 0.9995 --
directly contradicting this project's established PBO~0.83 / DSR-does-not-
survive findings (Ch11-15). Root cause: too few trials (3 vs Ch11's real
20), too few effective observations (87 events vs Ch11's 238 bar-level
mark-to-market points), and a naive full-size-every-call PnL proxy instead
of Ch10's real discretized getSignal position. Reusing Ch11's own function
directly (not re-deriving equivalent logic) reconciles PBO to ~0.83 (matches
the established real-machine number almost exactly) and drops DSR to ~0.54
-- a genuinely different, more honest, and reconciled picture. See the
project README for the full writeup.

This module implements NO new AFML formula. Every calculation below
delegates to existing, real-machine-confirmed chapter code:
  - ch11/chapter_11_backtest_dangers.py  (part_c_build_trials,
    out_of_sample_probs -- the REAL, established trial-construction logic)
  - ch11/backtest_dangers/pbo.py         (pbo, sharpe_ratio)
  - ch10/bet_sizing/bet_sizing.py        (getSignal)
  - ch14/backtest_statistics/backtest_statistics.py (deflated_sharpe_ratio)
This file is pure glue: run Ch11's real trial grid, hand its output to
Ch11's real pbo(), compute real skew/kurtosis-corrected DSR on the winning
trial (Ch14's own convention), and surface the winning trial's live signal.

Phase 1 scope: operates on the EXISTING static March 2026 BTC/TUSD real
artifacts. Phase 2 will swap the data-loading step for a live pull; the
downstream Ch11-based trial-construction logic is already asset-and-data-
source agnostic (it operates on whatever real events/bars are loaded).
"""
import importlib.util
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))

for rel in (
    os.path.join('ch10', 'bet_sizing'),
    os.path.join('ch14', 'backtest_statistics'),
):
    p = os.path.join(ROOT, rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from bet_sizing import getSignal                        # ch10, real module
from backtest_statistics import deflated_sharpe_ratio    # ch14, real module


def load_ch11_driver():
    """Dynamically load ch11/chapter_11_backtest_dangers.py by file path
    (NOT reimplemented here) so this orchestration layer reuses the exact
    real, established trial-construction and PBO logic behind this
    project's published PBO~0.83 finding. The module computes its own ROOT
    from its own __file__, so this works regardless of caller cwd."""
    path = os.path.join(ROOT, 'ch11', 'chapter_11_backtest_dangers.py')
    spec = importlib.util.spec_from_file_location('ch11_driver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_real_trials(ch11):
    """Reuse Ch11's real part_c_build_trials(): a genuine 20-configuration
    SVC(C) x getSignal(stepSize) grid, purged-CV out-of-sample probabilities,
    bar-level (not event-level) mark-to-market PnL with a load-bearing
    .shift(1) to avoid lookahead. Returns (M, meta): M is a (bars, 20)
    PnL matrix, meta carries each trial's (C, stepSize, full-sample Sharpe).
    """
    M, meta = ch11.part_c_build_trials()
    return M, meta


def evaluate_overfitting(M, meta, ch11, S=8, *, tw):
    """Wraps Ch11's real pbo() and Ch14's real deflated_sharpe_ratio(),
    with real skew/kurtosis computed from the winning trial's actual
    nonzero-bet return distribution (Ch14's own convention -- pandas'
    kurtosis() is EXCESS kurtosis, +3 converts to the book's raw gamma_4).

    *** LOAD-BEARING (2026-08-17): DSR's T is uniqueness-weighted, not a
    raw bar count ***
    Bug found in the 2026-08-16 session's CUSUM investigation: T used to be
    len(bet_ret) -- the count of nonzero-PnL bars in the winning trial's
    bar-level mark-to-market series -- fed straight into
    deflated_sharpe_ratio() as if every bar were an independent
    observation. It is not: triple-barrier events overlap heavily under
    this pipeline's fixed VERTICAL_BARRIER_NUM_DAYS=3, which is exactly
    what Ch04's average uniqueness (tw) measures. A controlled real-data
    experiment (compare_tw_by_cusum_h.py, 2026-08-16) confirmed this
    precisely: tripling the raw CUSUM event count (45->140, via
    CUSUM_H=500->100) left the uniqueness-weighted effective sample size
    essentially flat (19.0->19.6), because the extra events were almost
    entirely overlapping restatements of the same underlying price moves,
    not new independent information.

    `tw` is therefore now a REQUIRED, keyword-only argument -- there is no
    default, so no caller can silently regress to the old, inflated-T
    behavior by forgetting to pass it. T_effective = T_raw * tw.mean().

    KNOWN SIMPLIFICATION (documented, not hidden): tw.mean() is the
    average uniqueness across the WHOLE event population feeding the
    trial grid, not filtered down to the specific bars/events behind the
    winning trial's own signal (Ch11's driver -- deliberately never
    edited directly, see this module's own header -- doesn't expose
    per-trial event indices). This matches the population-level approach
    the 2026-08-16 diagnostic itself used.

    Parameters
    ----------
    tw : pd.Series, Ch04's average-uniqueness output
        (get_average_uniqueness()), already reindexed by the CALLER to the
        event population that fed this run's trial grid (static:
        input_data/ch04_weights.csv's 'tw' column reindexed to
        ch07_training_table_enriched.csv's index; live: rebuild_result[
        'tw'] reindexed to enriched_result['enriched_events'].index --
        same reindex pattern live_staging.py already uses for 'w'). Must
        contain no NaNs and must be non-empty.
    """
    prob_overfit, cscv_df = ch11.pbo(M, S=S)

    trial_sharpes = meta['sharpe_full_sample']
    best_trial = trial_sharpes.idxmax()
    sr_hat = trial_sharpes[best_trial]

    bet_ret = M[best_trial][M[best_trial] != 0]
    T_raw = len(bet_ret)
    if T_raw > 2:
        skew = float(bet_ret.skew())
        kurtosis = float(bet_ret.kurtosis()) + 3.0
    else:
        skew, kurtosis = 0.0, 3.0  # too few realized bets to estimate; fall
                                    # back to the Gaussian assumption rather
                                    # than a division-by-near-zero estimate
                                    # -- gated on T_raw, a DATA-SUFFICIENCY
                                    # question, deliberately NOT T_effective

    if len(tw) == 0:
        raise ValueError(
            'tw is empty -- cannot compute a uniqueness-weighted T. Check '
            'the caller\'s reindexing of tw to the trial-grid event '
            'population.'
        )
    if tw.isna().any():
        raise ValueError(
            'tw contains NaN -- an event in the trial-grid population has '
            'no matching uniqueness value. This should be impossible if '
            'tw was reindexed correctly by the caller; investigate before '
            'evaluating overfitting on it.'
        )
    tw_mean = float(tw.mean())
    T_effective = T_raw * tw_mean

    n_trials = M.shape[1]
    var_sr_trials = float(trial_sharpes.var(ddof=1))
    dsr = deflated_sharpe_ratio(sr_hat, var_sr_trials, n_trials, T_effective, skew, kurtosis)

    return {
        'trial_sharpes': trial_sharpes,
        'best_trial': best_trial,
        'sr_hat': sr_hat,
        'prob_overfit': prob_overfit,
        'cscv_df': cscv_df,
        'n_trials': n_trials,
        'var_sr_trials': var_sr_trials,
        'T': T_effective,
        'T_raw': T_raw,
        'tw_mean': tw_mean,
        'skew': skew,
        'kurtosis': kurtosis,
        'dsr': dsr,
        'meta': meta,
    }


def latest_bet_signal(best_trial, meta, ch11, input_data_dir):
    """Recompute the winning trial's real out-of-sample signal series --
    Ch11's own out_of_sample_probs() feeding Ch10's real getSignal(), the
    same two real functions the winning trial's PnL column was built from
    -- and return its most recent value. Returns None if unavailable."""
    C = meta.loc[best_trial, 'C']
    step = meta.loc[best_trial, 'stepSize']

    enriched_path = os.path.join(input_data_dir, 'ch07_training_table_enriched.csv')
    events = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
    X, y, w, t1 = events[feature_cols], events['bin'], events['w'], events['t1']

    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)
    if prob.empty:
        return None
    ev = events.loc[prob.index]

    signal = getSignal(ev, step, prob, pred, numClasses=2, numThreads=1)
    if signal.empty:
        return None
    latest_val = signal.iloc[-1]
    return float(latest_val) if pd.notna(latest_val) else None
def run_live_trials(ch11, live_input_dir, live_here_dir):
    """Runs Ch11's real part_c_build_trials() against LIVE data instead of
    the static March 2026 dataset, WITHOUT modifying chapter_11_backtest_
    dangers.py at all. That file has no parameters for this -- it hard-
    loads two filenames from its own module-level INPUT constant.

    *** LOAD-BEARING (2026-08-14): monkeypatches ch11's INPUT and HERE
    module globals rather than editing the file ***
    Two reasons this is monkeypatched, not parameterized in the source:
    (1) chapter_11_backtest_dangers.py is a real-machine-confirmed
    TEACHING deliverable (Ch11's README, notebook, and committed PNGs all
    describe ITS OWN exact static-data run) -- adding live-data plumbing
    to it would blur what that chapter's own confirmed output means.
    (2) part_c_build_trials() has a side effect beyond its return value:
    it SAVES PNG PLOTS to HERE (ch11_trial_sharpes.png etc). Without also
    patching HERE, every live pipeline run would silently overwrite Ch11's
    own committed teaching plots with live-run byproducts. Both INPUT and
    HERE are patched here, and restored in a finally block so a live run
    can never leave the ch11 module object in a state that would corrupt
    a SUBSEQUENT static-data call in the same Python process.
    """
    original_input, original_here = ch11.INPUT, ch11.HERE
    try:
        ch11.INPUT = live_input_dir
        ch11.HERE = live_here_dir
        os.makedirs(live_here_dir, exist_ok=True)
        M, meta = ch11.part_c_build_trials()
    finally:
        ch11.INPUT = original_input
        ch11.HERE = original_here
    return M, meta

# ---------------------------------------------------------------------------
# TDD results -- real machine (mlfinlab env), 2026-08-17
# (see pipeline/orchestration/test_stages.py for the full suite and its
# own embedded pytest output; summarized here per this project's .py
# TDD-embed convention)
#
# (mlfinlab) PS C:\ws\AFML> python -m pytest pipeline\orchestration\test_stages.py -v
# ============================== 11 passed in 4.48s ==============================
# Two-pass (from inside pipeline/orchestration/): 11 passed in 2.33s
#
# Confirmed against real data via pipeline\run_pipeline.py immediately
# after: T (uniqueness-weighted, was a raw bar count) = 26.166490233278697,
# DSR = 0.5206 -- still squarely "no reliable edge," consistent with every
# other diagnostic this project has run on this dataset (Ch11-15).
# ---------------------------------------------------------------------------
