"""
pipeline/diagnostics/calibrate_kraken_target_bars.py

Tests how far target_bars can scale on the real 720h Kraken snapshot
(kraken_snapshot_720h_2026-08-25/, 1,673,952 raw trades -- confirmed
2026-08-25) before hitting the same "running out of raw trades to
subdivide" plateau target_bars alone hit on Binance.US's 720h pull
(~180 T_effective ceiling, per CALIBRATION_AUDIT.md's "target_bars=1000"
section). This snapshot has ~4.9x the raw trades of the entire 90-day
(2160h) Binance.US pull from earlier the same session, in a quarter of
the calendar window -- the working hypothesis is a materially higher
plateau, reached without needing anywhere near as long a window (which
also reduces, though doesn't eliminate, the single-window regime-
dependency concern that 90-day Binance pull surfaced).

*** LOAD-BEARING (2026-08-25): production constants (CUSUM_H=313,
VERTICAL_BARRIER_NUM_DAYS=3, MIN_RET=0.005, etc.) reused UNCHANGED as a
first pass, NOT yet re-validated for Kraken's own price/volume
character ***
Same approach the lookback-extension sweep took on Binance.US data.
These constants were calibrated against Binance.US specifically (see
CALIBRATION_AUDIT.md's "CUSUM_H Staleness Audit" section) -- Kraken's
trade-size distribution, tick frequency, and bar-forming dynamics could
differ enough that these aren't automatically well-calibrated here too.
This script's results are a first look at whether Kraken data is
WORTH pursuing further, not a validated final calibration -- if the
answer is yes, re-deriving CUSUM_H/MIN_RET/etc. specifically for Kraken
(mirroring the original Binance.US staleness-audit methodology) is real,
separate follow-up work.

Reuses rebuild.py/features.py/live_staging.py/stages.py exactly as
every other calibration script in this project does -- target_bars is
the only thing varied here.

Run (after capture_kraken_snapshot.py --hours 720 has produced the
snapshot):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_kraken_target_bars.py <snapshot_dir>
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from rebuild import build_bars_and_labels              # noqa: E402
from features import build_enriched_events              # noqa: E402
from live_staging import stage_live_training_tables      # noqa: E402
from stages import (                                    # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

OUTPUT_CSV = os.path.join(HERE, 'kraken_target_bars_calibration.csv')
SWEEP_COLUMNS = [
    'target_bars', 'n_bars', 'n_events', 'n_events_enriched', 'T_raw',
    'tw_mean', 'T_effective', 'best_sharpe', 'pbo', 'dsr', 'skew',
    'kurtosis', 'notes',
]

TARGET_BARS_GRID = [1000, 2000, 3000, 4000, 5000]


def _run_one_config(raw_trades, target_bars, work_root):
    config_name = f'tb{target_bars}'

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    print(f"  [{config_name}] {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} events, "
          f"threshold=${rebuild_result['threshold']:,.2f}")

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  [{config_name}] {enriched_result['n_events_after']}/"
          f"{enriched_result['n_events_before']} events survived enrichment")

    staging_dir = os.path.join(work_root, config_name, 'staging')
    here_dir = os.path.join(work_root, config_name, 'ch11_here')
    stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

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
        'n_bars': len(rebuild_result['bars']),
        'n_events': len(rebuild_result['events']),
        'n_events_enriched': enriched_result['n_events_after'],
        'T_raw': eval_result['T_raw'],
        'tw_mean': eval_result['tw_mean'],
        'T_effective': eval_result['T'],
        'best_sharpe': eval_result['sr_hat'],
        'pbo': eval_result['prob_overfit'],
        'dsr': eval_result['dsr'],
        'skew': eval_result['skew'],
        'kurtosis': eval_result['kurtosis'],
        'notes': '',
    }


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            'Usage: python calibrate_kraken_target_bars.py <snapshot_dir> [output_csv]\n'
            'Run capture_kraken_snapshot.py --hours 720 first to produce a snapshot.\n'
            '[output_csv] is optional -- defaults to kraken_target_bars_calibration.csv.\n'
            'ADDED 2026-08-25: pass a distinct output_csv when running on a SECOND '
            'window (e.g. for a replication check) so results don\'t mix with the '
            'first window\'s in the same file.'
        )
    snapshot_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else OUTPUT_CSV
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen Kraken snapshot: {len(raw_trades)} raw trades '
          f'from {snapshot_dir}')
    print(f'Writing results to: {output_csv}')

    work_root = os.path.join(HERE, 'kraken_target_bars_sweep_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for target_bars in TARGET_BARS_GRID:
        print(f'\n=== target_bars={target_bars} ===')
        try:
            row = _run_one_config(raw_trades, target_bars, work_root)
            rows.append(row)
            print(f"  [tb{target_bars}] T_raw={row['T_raw']}, "
                  f"tw_mean={row['tw_mean']:.4f}, "
                  f"T_effective={row['T_effective']:.2f}, "
                  f"DSR={row['dsr']:.4f}, PBO={row['pbo']:.4f}")
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            rows.append({
                'target_bars': target_bars, 'n_bars': None, 'n_events': None,
                'n_events_enriched': None, 'T_raw': None, 'tw_mean': None,
                'T_effective': None, 'best_sharpe': None, 'pbo': None,
                'dsr': None, 'skew': None, 'kurtosis': None,
                'notes': f'{type(e).__name__}: {e}',
            })

    file_exists = os.path.exists(output_csv)
    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
    print(f'\nResults appended to {output_csv}')

    print('\n' + '=' * 66)
    print('SUMMARY: T_effective by target_bars')
    print('=' * 66)
    print(df[['target_bars', 'T_raw', 'tw_mean', 'T_effective', 'dsr', 'pbo']]
          .to_string(index=False))
    print("""
INTERPRETATION:
  - Compare against Binance.US's own target_bars=1000 plateau (~180-183
    T_effective, CALIBRATION_AUDIT.md's "target_bars=1000" section, same
    720h calendar window). If Kraken's plateau sits meaningfully higher
    -- and/or is reached at a HIGHER target_bars value before flattening
    -- that's real evidence this venue's density genuinely helps, not
    just in raw trade count but in what it buys downstream.
  - Watch tw_mean too -- if it degrades faster here than it did on
    Binance.US as target_bars rises, that's a real cost worth weighing,
    same trade-off already documented for CUSUM_H reduction.
  - Remember: CUSUM_H/VERTICAL_BARRIER_NUM_DAYS/MIN_RET are Binance.US-
    calibrated constants, reused here unvalidated (see module
    docstring). A strong result here is grounds to invest in a proper
    Kraken-specific re-calibration, not a signal that one is unnecessary.
""")


if __name__ == '__main__':
    main()
