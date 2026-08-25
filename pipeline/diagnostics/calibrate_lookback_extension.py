"""
pipeline/diagnostics/calibrate_lookback_extension.py

Tests whether extending LOOKBACK_HOURS -- combined with raising
target_bars further -- can push T_effective past the ~180 plateau
target_bars alone hit on a 720h pull (CALIBRATION_AUDIT.md's
"target_bars=1000" section), toward the ~200-1000 range where DSR's
detection power becomes meaningful (2026-08-19 Detection Power
Calibration Findings). See capture_lookback_extension_snapshot.py's
module docstring for the full mechanism/reasoning.

Reuses calibrate_t_effective_levers.py's `_run_one_config()` pattern
directly (same monkeypatch-and-restore discipline for CUSUM_H/
VERTICAL_BARRIER_NUM_DAYS, though neither is varied here -- both stay
at their current production values, 313/3, for every config; only
target_bars and the raw_trades WINDOW are varied). Every config re-runs
the full rebuild -> enrich -> stage -> Ch11 trials -> evaluate chain,
since target_bars and window length are both upstream of
build_bars_and_labels().

3 windows (720h / 1440h / 2160h, sliced from ONE frozen pull -- see
capture script) x 3 target_bars values (1000 / 1500 / 2000) = 9 configs,
9 full SVC(C) x getSignal(stepSize) grid searches. This is a real-time-
cost sweep (n_jobs=1 per this project's established Windows/loky SVC
constraint) -- budget accordingly, likely comparable to or longer than
the 2026-08-21 four-point target_bars sweep given the larger raw trade
counts involved, especially at the 2160h/target_bars=2000 corner.

Run (after capture_lookback_extension_snapshot.py has produced a
snapshot dir):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_lookback_extension.py <snapshot_dir>
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

OUTPUT_CSV = os.path.join(HERE, 'lookback_extension_calibration.csv')
SWEEP_COLUMNS = [
    'config', 'lookback_hours', 'target_bars', 'n_raw_trades_in_window',
    'n_bars', 'n_events', 'n_events_enriched', 'T_raw', 'tw_mean',
    'T_effective', 'best_sharpe', 'pbo', 'dsr', 'notes',
]

# Production values, held FIXED across every config in this sweep --
# only lookback window and target_bars vary here.
CUSUM_H = rebuild_module.CUSUM_H                          # 313
VERTICAL_BARRIER_NUM_DAYS = rebuild_module.VERTICAL_BARRIER_NUM_DAYS  # 3

LOOKBACK_HOURS_GRID = [720, 1440, 2160]
TARGET_BARS_GRID = [1000, 1500, 2000]


def _slice_window(raw_trades, lookback_hours):
    """Timestamp is microseconds since epoch (see rebuild.py's
    preprocess_raw_trades: pd.to_datetime(raw['Timestamp'], unit='us')).
    Slices the most recent `lookback_hours` of the frozen pull -- since
    Binance's historicalTrades paged backward from the most recent trade
    when this snapshot was captured, this reproduces exactly what a
    live pull of that shorter lookback_hours would have returned at
    capture time."""
    max_ts = raw_trades['Timestamp'].max()
    cutoff = max_ts - int(lookback_hours * 3600 * 1e6)
    window = raw_trades[raw_trades['Timestamp'] >= cutoff].reset_index(drop=True)
    return window


def _run_one_config(raw_trades_window, config_name, target_bars, work_root):
    """Mirrors calibrate_t_effective_levers.py's _run_one_config() --
    CUSUM_H/VERTICAL_BARRIER_NUM_DAYS held at their current module
    values via the same monkeypatch-and-restore pattern (a no-op here
    since neither is overridden, but kept for consistency with that
    script and as a safety net against this module's globals having
    drifted from rebuild.py's own current values between runs)."""
    original_cusum_h = rebuild_module.CUSUM_H
    original_vbnd = rebuild_module.VERTICAL_BARRIER_NUM_DAYS
    try:
        rebuild_module.CUSUM_H = CUSUM_H
        rebuild_module.VERTICAL_BARRIER_NUM_DAYS = VERTICAL_BARRIER_NUM_DAYS

        rebuild_result = build_bars_and_labels(raw_trades_window, target_bars=target_bars)
        print(f"  [{config_name}] {len(rebuild_result['bars'])} bars, "
              f"{len(rebuild_result['events'])} events, "
              f"threshold=${rebuild_result['threshold']:,.2f}")
    finally:
        rebuild_module.CUSUM_H = original_cusum_h
        rebuild_module.VERTICAL_BARRIER_NUM_DAYS = original_vbnd

    enriched_result = build_enriched_events(
        raw_trades_window, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  [{config_name}] {enriched_result['n_events_after']}/"
          f"{enriched_result['n_events_before']} events survived enrichment")

    staging_dir = os.path.join(work_root, config_name, 'staging')
    here_dir = os.path.join(work_root, config_name, 'ch11_here')
    staged = stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)

    tw_aligned = rebuild_result['tw'].reindex(
        enriched_result['enriched_events'].index
    )
    if tw_aligned.isna().any():
        raise ValueError(
            f"[{config_name}] tw has NaN after reindexing to the enriched "
            "event index."
        )

    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)

    return {
        'target_bars': target_bars,
        'n_raw_trades_in_window': len(raw_trades_window),
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
            'Usage: python calibrate_lookback_extension.py <snapshot_dir>\n'
            'Run capture_lookback_extension_snapshot.py first to produce one.'
        )
    snapshot_dir = sys.argv[1]
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades_full = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades_full)} raw trades '
          f'from {snapshot_dir}')

    work_root = os.path.join(HERE, 'lookback_extension_sweep_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for lookback_hours in LOOKBACK_HOURS_GRID:
        window = _slice_window(raw_trades_full, lookback_hours)
        print(f'\n### lookback_hours={lookback_hours} '
              f'({len(window)} raw trades in window) ###')
        for target_bars in TARGET_BARS_GRID:
            config_name = f'lb{lookback_hours}_tb{target_bars}'
            print(f'\n=== Running config: {config_name} ===')
            row = _run_one_config(window, config_name, target_bars, work_root)
            row['config'] = config_name
            row['lookback_hours'] = lookback_hours
            rows.append(row)
            print(f"  [{config_name}] T_raw={row['T_raw']}, "
                  f"tw_mean={row['tw_mean']:.4f}, "
                  f"T_effective={row['T_effective']:.2f}, "
                  f"DSR={row['dsr']:.4f}, PBO={row['pbo']:.4f}")

    file_exists = os.path.exists(OUTPUT_CSV)
    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(OUTPUT_CSV, mode='a', header=not file_exists, index=False)
    print(f'\nResults appended to {OUTPUT_CSV}')

    print('\n' + '=' * 78)
    print('SUMMARY: T_effective by lookback_hours (rows) x target_bars (cols)')
    print('=' * 78)
    pivot = df.pivot(index='lookback_hours', columns='target_bars', values='T_effective')
    print(pivot.round(2).to_string())
    print("""
INTERPRETATION:
  - If T_effective keeps climbing as BOTH lookback_hours and target_bars
    increase (not just target_bars alone, which plateaued at ~180 on a
    720h pull), that confirms the "raw trade count was the real binding
    constraint" hypothesis -- a longer pull genuinely unlocks further
    target_bars scaling.
  - If T_effective plateaus again even at 2160h (e.g. target_bars=2000
    on the 2160h row looks no better than target_bars=1000 on the same
    row), the constraint isn't raw trade count after all -- something
    else is capping it (worth a fresh trace rather than assuming lookback
    alone is the answer).
  - Watch tw_mean too (in the CSV, not this pivot) -- if a longer window
    or higher target_bars starts degrading average uniqueness the way
    CUSUM_H reduction did, that would be a cost worth weighing against
    any T_effective gain, same as the CUSUM_H/target_bars trade-off
    already documented in CALIBRATION_AUDIT.md.
""")


if __name__ == '__main__':
    main()
