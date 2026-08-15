"""
pipeline/run_pipeline_positive_control.py

Positive-control counterpart to run_pipeline_live.py. Instead of pulling
real trades (ingestion.pull_recent_trades), generates SYNTHETIC trades
carrying a DELIBERATE, known momentum edge (positive_control_data.
generate_momentum_trades) and runs them through the EXACT SAME unmodified
chain: rebuild -> features -> live_staging -> Ch11's real trial
construction (via stages.run_live_trials) -> evaluate_overfitting.

Why this exists (see 2026-08-14 handoff, Part 6): three straight real
BTC/USDT live runs (plus the static baseline) all report "no reliable
edge". Negative-only testing on real data can't distinguish "the market
really has no edge" from "the pipeline silently suppresses any signal" --
those look identical from the outside. A LOW PBO / HIGH DSR result here,
on data with a KNOWN engineered edge, is the missing positive evidence
that the machinery works.

Also runs the two cheap supporting checks from the same handoff section:
  1. Feature degeneracy -- flags any near-zero-variance column in the
     enriched feature table (a silently-constant feature wouldn't crash
     anything, just quietly contribute nothing).
  2. Class balance / prediction spread -- flags a badly imbalanced `bin`
     label or a winning trial whose predictions never vary.

*** LOAD-BEARING (2026-08-15): sandbox-verified only, NOT yet a real-
machine (mlfinlab env) two-pass confirmation ***
This driver (and positive_control_data.py / test_positive_control_data.py)
were developed and dry-run end-to-end in Claude's own sandbox against a
fresh pull of the real repo (numpy 2.4/pandas 3.0/sklearn 1.8 -- NOT
mlfinlab's pinned numpy 1.23.5/pandas 1.5.3/sklearn 1.2.2), because that
sandbox can't build the older pinned versions on its Python 3.12. The
dry run's numbers (see module-level RESULT note below) are real evidence
the LOGIC works, not a substitute for this project's own real-machine
two-pass pytest convention -- run test_positive_control_data.py yourself
in mlfinlab before treating it as confirmed, same as every other chapter.

*** DRY-RUN RESULT (2026-08-15, sandbox, n_trades=6000, random_state=42,
default calibration) ***
244 bars, 200 triple-barrier events, 188/200 (94%) survived feature
enrichment, fracdiff_d=1.0 (see note below). PBO=0.0286, DSR=0.8540 --
sharply different from every real-data run (PBO 0.78-0.83, DSR 0.37-0.55).
This is exactly the missing positive-control evidence: the pipeline DOES
detect a genuine edge when one is deliberately engineered to exist, which
is what makes the real-data "no edge" findings credible rather than a
silent-failure artifact.

Two things the dry run also surfaced, worth knowing before you run this
for real (not blocking, both explained in positive_control_data.py's own
LOAD-BEARING notes):
  - round_number_fraction comes back EXACTLY constant (std=0) -- a direct
    consequence of holding Volume constant across all synthetic trades
    (a deliberate design choice, see positive_control_data.py). This is
    the feature-degeneracy check (below) correctly catching something
    real, not a bug to silently ignore.
  - fracdiff_d landed at 1.0 (the top of Ch05's search range) -- a
    strongly trending/near-unit-root price path (which strong engineered
    momentum naturally produces) needs closer to full differencing to
    become stationary. Expected given the calibration, not an error.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
PC_STAGING_DIR = os.path.join(HERE, 'positive_control_staging_data')
PC_HERE_DIR = os.path.join(HERE, 'positive_control_output')

sys.path.insert(0, os.path.join(HERE, 'orchestration'))

from positive_control_data import generate_momentum_trades  # noqa: E402
from rebuild import build_bars_and_labels                    # noqa: E402
from features import build_enriched_events                    # noqa: E402
from live_staging import stage_live_training_tables            # noqa: E402
from stages import (                                          # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

N_TRADES = 6000            # see positive_control_data.py's own docstring
                            # for why this, at default calibration, lands
                            # around ~250 bars / ~200 events (comparable
                            # order of magnitude to the real live runs)
CONTINUATION_PROB = 0.85   # deliberately far stronger than any real
                            # market's momentum -- see module docstring
RANDOM_STATE = 42          # fixed for a reproducible positive control;
                            # change/loop this to check the finding isn't
                            # a lucky single seed (see main()'s note)


def check_feature_degeneracy(feature_table, std_threshold=1e-10):
    """Flags any column in the enriched feature table with near-zero
    variance -- a silently-constant feature wouldn't crash the SVC, it
    would just quietly contribute nothing, which could mask a weaker real
    edge without ever raising an error. Returns a list of degenerate
    column names (empty if none)."""
    stds = feature_table.std(numeric_only=True)
    degenerate = list(stds[stds < std_threshold].index)
    return degenerate


def check_class_balance_and_predictions(enriched_events, ch11, staging_dir, C):
    """Two cheap sanity checks from the 2026-08-14 handoff (Part 6):
      (a) is the `bin` label itself badly imbalanced?
      (b) does the winning trial's out-of-sample prediction actually vary,
          or does it just predict one class every single time (which
          would trivially inflate an apparent 'edge' with zero real
          skill)?
    Reuses ch11.out_of_sample_probs() directly (the same real function
    stages.latest_bet_signal() calls) rather than re-deriving equivalent
    logic.
    """
    bin_counts = enriched_events['bin'].value_counts()
    bin_balance_ratio = bin_counts.min() / bin_counts.max()

    import pandas as pd
    enriched_path = os.path.join(staging_dir, 'ch07_training_table_enriched.csv')
    events = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
    X, y, w, t1 = events[feature_cols], events['bin'], events['w'], events['t1']

    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)
    pred_counts = pred.value_counts() if len(pred) else pd.Series(dtype=int)
    pred_is_degenerate = len(pred_counts) < 2

    return {
        'bin_counts': bin_counts.to_dict(),
        'bin_balance_ratio': float(bin_balance_ratio),
        'pred_counts': pred_counts.to_dict(),
        'pred_is_degenerate': pred_is_degenerate,
    }


def main():
    print(f'Generating {N_TRADES} synthetic trades with a deliberate '
          f'momentum edge (continuation_prob={CONTINUATION_PROB})...')
    synth = generate_momentum_trades(
        N_TRADES, continuation_prob=CONTINUATION_PROB, random_state=RANDOM_STATE,
    )
    raw_trades = synth['raw_trades']

    rebuild_result = build_bars_and_labels(raw_trades)
    print(f"  {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} triple-barrier events, "
          f"threshold=${rebuild_result['threshold']:,.2f}")

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  {enriched_result['n_events_after']}/"
          f"{enriched_result['n_events_before']} events survived feature "
          f"enrichment (fracdiff d={enriched_result['fracdiff_d']})")

    degenerate_cols = check_feature_degeneracy(enriched_result['feature_table'])
    if degenerate_cols:
        print(f"  [FEATURE DEGENERACY] near-zero-variance columns: {degenerate_cols}")
    else:
        print("  [FEATURE DEGENERACY] none found -- all features have real variance")

    staged = stage_live_training_tables(
        rebuild_result, enriched_result, PC_STAGING_DIR,
    )
    print(f"  staged {staged['n_events']} enriched events to "
          f"{staged['enriched_csv_path']}")

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, PC_STAGING_DIR, PC_HERE_DIR)

    eval_result = evaluate_overfitting(M, meta, ch11, S=8)
    best_trial = eval_result['best_trial']
    best_C = meta.loc[best_trial, 'C']

    balance_check = check_class_balance_and_predictions(
        enriched_result['enriched_events'], ch11, PC_STAGING_DIR, best_C,
    )
    print(f"  [CLASS BALANCE] bin counts: {balance_check['bin_counts']} "
          f"(min/max ratio: {balance_check['bin_balance_ratio']:.3f})")
    if balance_check['pred_is_degenerate']:
        print(f"  [PREDICTION SPREAD] WARNING -- winning trial's predictions "
              f"never vary: {balance_check['pred_counts']}")
    else:
        print(f"  [PREDICTION SPREAD] winning trial's predictions vary normally: "
              f"{balance_check['pred_counts']}")

    print()
    print('=' * 70)
    print('POSITIVE CONTROL RESULT')
    print('=' * 70)
    print(f"  PBO: {eval_result['prob_overfit']:.4f}")
    print(f"  DSR: {eval_result['dsr']:.4f}")
    print(f"  Best trial ({best_trial}): Sharpe = {eval_result['sr_hat']:.4f}")
    print()
    print('  Compare against this project\'s real-data runs: PBO ~0.78-0.83, '
          'DSR ~0.37-0.55 (static baseline + 2 live runs, 2026-08-13/14).')
    if eval_result['prob_overfit'] < 0.3 and eval_result['dsr'] > 0.6:
        print('  Low PBO / high DSR on data with a KNOWN edge -- the pipeline '
              'CAN detect a real edge. This supports treating the real-data '
              '"no edge" finding as a genuine market result, not a silent '
              'pipeline failure.')
    else:
        print('  PBO/DSR did NOT come back clearly favorable on data with a '
              'KNOWN edge -- investigate before trusting the real-data "no '
              'edge" finding. Check the feature-degeneracy and prediction-'
              'spread warnings above first.')


if __name__ == '__main__':
    main()
