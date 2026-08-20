"""
pipeline/diagnostics/calibrate_t_effective_levers.py

Scoped 2026-08-19 follow-on to the detection-power finding (see
CALIBRATION_AUDIT.md's "Detection Power Calibration Findings" section):
meaningful DSR detection power only emerges around T_effective=200-1000,
3-20x more than this pipeline's real observed range (50-80). This script
tests three candidate levers, ONE AT A TIME against the SAME frozen raw
trades snapshot (see capture_t_effective_snapshot.py), to see which
actually move T_effective = T_raw * tw_mean and by how much:

  1. target_bars (rebuild.py's build_bars_and_labels() parameter,
     currently 250) -- more dollar bars from the same pull.
  2. CUSUM_H (rebuild.py module constant, currently 500) -- fewer/more
     triple-barrier events. NOTE: 2026-08-16's CUSUM_H sweep (500->100)
     found raw event count nearly tripled while EFFECTIVE (uniqueness-
     weighted) T barely moved -- this script re-tests a more moderate
     value (250) specifically to see whether that finding holds at a
     smaller perturbation, not just the extreme one already tested.
  3. VERTICAL_BARRIER_NUM_DAYS (rebuild.py module constant, currently 3)
     -- shorter label horizon reduces triple-barrier overlap, which is
     what tw_mean (average uniqueness) actually measures.

MECHANISM NOTE (traced 2026-08-19, corrects CALIBRATION_AUDIT.md's
original lever list): LOOKBACK_HOURS is NOT tested here. run_pipeline_live.py
always calls build_bars_and_labels(raw_trades) with target_bars' default
(250) -- compute_dynamic_threshold() rescales the dollar-bar threshold to
hit ~target_bars bars regardless of how much history was pulled, so a
longer live pull would NOT increase bar count (and therefore likely not
T_raw) under the pipeline's current design. Raising target_bars directly
is the real lever for "more bars"; LOOKBACK_HOURS is a red herring for
this specific question (though it remains genuinely load-bearing
elsewhere, per its own Tier-2 entry in CALIBRATION_AUDIT.md, for having
enough prior history for get_daily_vol()).

MONKEYPATCH-AND-RESTORE (never touching rebuild.py itself): CUSUM_H and
VERTICAL_BARRIER_NUM_DAYS are referenced as bare module-global names
INSIDE build_bars_and_labels()'s function body (confirmed by reading the
real source, 2026-08-19) -- NOT bound as default-argument values the way
features.py's ROLL_WINDOW/VPIN_WINDOW/FFD_THRES are. This means ordinary
module-attribute monkeypatching (rebuild_module.CUSUM_H = X) works
correctly here and is picked up by build_bars_and_labels() at call time --
unlike features.py's gotcha, where monkeypatching silently did nothing
because the values were already bound at function-definition time.
target_bars needs no monkeypatching at all -- it's a real, explicit
keyword parameter of build_bars_and_labels().

COST NOTE: unlike the 2026-08-18 sweep (which only needed to patch
downstream of a frozen rebuild_result), every config here re-runs the
FULL chain -- rebuild -> enrich -> stage -> Ch11's real 20-configuration
SVC(C) x getSignal(stepSize) grid -- since events/labels/tw all change
fundamentally upstream of build_bars_and_labels(). 4 configs (baseline +
3 one-at-a-time alternates) = 4 full SVC grid searches, n_jobs=1 per
this project's established Windows/loky SVC(probability=True) constraint.

Run (after capture_t_effective_snapshot.py has produced a snapshot dir):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_t_effective_levers.py <snapshot_dir>
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

import rebuild as rebuild_module                     # noqa: E402
from rebuild import build_bars_and_labels             # noqa: E402
from features import build_enriched_events            # noqa: E402
from live_staging import stage_live_training_tables    # noqa: E402
from stages import (                                  # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

OUTPUT_CSV = os.path.join(HERE, 't_effective_lever_sweep.csv')
SWEEP_COLUMNS = [
    'config', 'target_bars', 'cusum_h', 'vertical_barrier_num_days',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

# baseline real values, per rebuild.py's own module constants
BASELINE_TARGET_BARS = 250
BASELINE_CUSUM_H = rebuild_module.CUSUM_H
BASELINE_VERTICAL_BARRIER_NUM_DAYS = rebuild_module.VERTICAL_BARRIER_NUM_DAYS

CONFIGS = [
    ('baseline', {}),
    ('target_bars_500', {'target_bars': 500}),
    ('cusum_h_250', {'CUSUM_H': 250}),
    ('vertical_barrier_1day', {'VERTICAL_BARRIER_NUM_DAYS': 1}),
]


def _run_one_config(raw_trades, config_name, overrides, work_root):
    """Runs the FULL rebuild -> enrich -> stage -> Ch11 trials -> evaluate
    chain for one lever config, monkeypatching-and-restoring rebuild.py's
    module constants around the single build_bars_and_labels() call that
    needs them. Returns a dict matching SWEEP_COLUMNS (minus 'config',
    added by the caller)."""
    target_bars = overrides.get('target_bars', BASELINE_TARGET_BARS)

    original_cusum_h = rebuild_module.CUSUM_H
    original_vbnd = rebuild_module.VERTICAL_BARRIER_NUM_DAYS
    try:
        rebuild_module.CUSUM_H = overrides.get('CUSUM_H', original_cusum_h)
        rebuild_module.VERTICAL_BARRIER_NUM_DAYS = overrides.get(
            'VERTICAL_BARRIER_NUM_DAYS', original_vbnd
        )

        rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
        print(f"  [{config_name}] {len(rebuild_result['bars'])} bars, "
              f"{len(rebuild_result['events'])} events, "
              f"threshold=${rebuild_result['threshold']:,.2f}")
    finally:
        rebuild_module.CUSUM_H = original_cusum_h
        rebuild_module.VERTICAL_BARRIER_NUM_DAYS = original_vbnd

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  [{config_name}] {enriched_result['n_events_after']}/"
          f"{enriched_result['n_events_before']} events survived enrichment")

    staging_dir = os.path.join(work_root, config_name, 'staging')
    here_dir = os.path.join(work_root, config_name, 'ch11_here')
    staged = stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)

    # *** same LOAD-BEARING reindex as run_pipeline_live.py (2026-08-17) ***
    tw_aligned = rebuild_result['tw'].reindex(
        enriched_result['enriched_events'].index
    )
    if tw_aligned.isna().any():
        raise ValueError(
            f"[{config_name}] tw has NaN after reindexing to the enriched "
            "event index -- see run_pipeline_live.py's identical guard."
        )

    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)

    return {
        'target_bars': target_bars,
        'cusum_h': overrides.get('CUSUM_H', BASELINE_CUSUM_H),
        'vertical_barrier_num_days': overrides.get(
            'VERTICAL_BARRIER_NUM_DAYS', BASELINE_VERTICAL_BARRIER_NUM_DAYS
        ),
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'n_events_enriched': enriched_result['n_events_after'],
        'T_raw': eval_result['T_raw'],
        'tw_mean': eval_result['tw_mean'],
        'T_effective': eval_result['T'],
        'best_sharpe': eval_result['sr_hat'],
        'pbo': eval_result['prob_overfit'],
        'dsr': eval_result['dsr'],
        'notes': '',
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python calibrate_t_effective_levers.py <snapshot_dir>\n'
            'Run capture_t_effective_snapshot.py first to produce one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    work_root = os.path.join(HERE, 't_effective_sweep_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for config_name, overrides in CONFIGS:
        print(f'\n=== Running config: {config_name} ({overrides or "baseline"}) ===')
        row = _run_one_config(raw_trades, config_name, overrides, work_root)
        row['config'] = config_name
        rows.append(row)
        print(f"  [{config_name}] T_raw={row['T_raw']}, tw_mean={row['tw_mean']:.4f}, "
              f"T_effective={row['T_effective']:.2f}, DSR={row['dsr']:.4f}, "
              f"PBO={row['pbo']:.4f}")

    file_exists = os.path.exists(OUTPUT_CSV)
    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
    print(f'\nResults appended to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()