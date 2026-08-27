"""
pipeline/diagnostics/calibrate_tao_target_bars.py

TAO counterpart to calibrate_xrp_target_bars.py -- same real chain
(rebuild.py/features.py/live_staging.py/stages.py), same target_bars
grid, same sweep structure. The one real difference: CUSUM_H is
monkeypatched to TAO's own calibrated value for the duration of this
sweep, NOT left at the BTC-calibrated production default (313) or
XRP's own value (0.0113).

*** LOAD-BEARING (2026-08-26): CUSUM_H=1.71 for TAO, NOT the BTC or
XRP values ***
Derived via calibrate_tao_cusum_h.py (real measurement: TAO/BTC
bar-to-bar diff std ratio 0.00746281x, applied to BTC's current
CUSUM_H=313, giving a ratio-scaled starting candidate of 2.335860) and
adjusted via sanity_check_tao_cusum_h.py on the REAL 720h TAO snapshot
already captured (221,947 trades, kraken_snapshot_taousd_720h_2026-08-26):
the ratio-scaled candidate (2.335860) under-fired at 22.0% (LOW verdict,
below BTC's own 25-35% reference range), so it was scaled down by the
ratio of observed-to-target event rate (0.220/0.30) to 1.71, which
PASSED at 30.5% -- comfortably inside BTC's own real reference range.
This is the first of tonight's asset-specific CUSUM_H derivations that
needed a manual iteration beyond the pure ratio-scaling step (XRP's
0.0113 passed on the ratio-scaled value directly) -- flagged here so
the extra tuning step isn't lost.

STILL UNVALIDATED for TAO as of this sweep: MIN_RET (0.005) and
VERTICAL_BARRIER_NUM_DAYS (3) are reused unchanged from the BTC
production defaults, same "first look, not a validated final
calibration" status both prior assets carried at this same stage.

SEPARATE, REAL RISK flagged repeatedly through tonight's TAO
calibration chain and worth restating here: TAO's real observed trade
density (308.3 trades/hour on the same 720h snapshot this sweep reuses)
is well below BTC's (~1,900-4,100/hr) or XRP's (~1,079.5/hr) --
roughly a third of XRP's own density, which was itself the lowest of
the three assets built so far. A passing CUSUM_H event rate does NOT
resolve this -- n_bars/T_effective viability at higher target_bars
values is a genuinely open question this sweep is the first real test
of.

Run (reuses the snapshot already captured by capture_kraken_snapshot.py
--pair TAOUSD --hours 720 -- no need to re-pull):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_tao_target_bars.py <snapshot_dir>
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

import rebuild as rebuild_module                        # noqa: E402
from rebuild import build_bars_and_labels                # noqa: E402
from features import build_enriched_events                # noqa: E402
from live_staging import stage_live_training_tables        # noqa: E402
from stages import (                                      # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

OUTPUT_CSV = os.path.join(HERE, 'tao_target_bars_calibration.csv')
SWEEP_COLUMNS = [
    'target_bars', 'n_bars', 'n_events', 'n_events_enriched', 'T_raw',
    'tw_mean', 'T_effective', 'best_sharpe', 'pbo', 'dsr', 'skew',
    'kurtosis', 'notes',
]

TARGET_BARS_GRID = [1000, 2000, 3000, 4000, 5000]
TAO_CUSUM_H = 1.71  # see module docstring


def _run_one_config(raw_trades, target_bars, work_root):
    config_name = f'tb{target_bars}'

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    print(f"  [{config_name}] {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} events, "
          f"threshold=${rebuild_result['threshold']:,.6f}")

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
            'Usage: python calibrate_tao_target_bars.py <snapshot_dir> [output_csv]\n'
            'Run capture_kraken_snapshot.py --pair TAOUSD --hours 720 first '
            '(already done tonight -- see module docstring for the path).'
        )
    snapshot_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else OUTPUT_CSV
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen TAO snapshot: {len(raw_trades)} raw trades '
          f'from {snapshot_dir}')
    print(f'Writing results to: {output_csv}')
    print(f'Using TAO-calibrated CUSUM_H={TAO_CUSUM_H} for this entire '
          f'sweep (monkeypatched, restored on exit)')

    work_root = os.path.join(HERE, 'tao_target_bars_sweep_work')
    os.makedirs(work_root, exist_ok=True)

    original_cusum_h = rebuild_module.CUSUM_H
    rows = []
    try:
        rebuild_module.CUSUM_H = TAO_CUSUM_H
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
    finally:
        rebuild_module.CUSUM_H = original_cusum_h

    file_exists = os.path.exists(output_csv)
    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
    print(f'\nResults appended to {output_csv}')

    print('\n' + '=' * 66)
    print('SUMMARY: T_effective by target_bars (TAO)')
    print('=' * 66)
    print(df[['target_bars', 'T_raw', 'tw_mean', 'T_effective', 'dsr', 'pbo']]
          .to_string(index=False))
    print("""
INTERPRETATION:
  - Compare against BTC (Kraken, target_bars=5000, T_effective=758.19,
    no plateau) and XRP (target_bars=5000, T_effective=625.37 on window
    1, 1481.35 on window 2 -- neither replicated as a real edge).
  - Remember: only CUSUM_H has been re-derived and sanity-checked for
    TAO specifically. MIN_RET and VERTICAL_BARRIER_NUM_DAYS are still
    reused unchanged from BTC's production defaults.
  - TAO's real trade density (308.3/hour) is well below both BTC's and
    XRP's -- if T_effective plateaus early or n_events stays thin at
    higher target_bars, that's the density disadvantage flagged
    throughout tonight's TAO calibration chain showing up concretely,
    not a bug.
  - This sweep alone does NOT answer whether there's an edge -- read
    these DSR/PBO numbers with the SAME caution as BTC's and XRP's:
    promising numbers alone are not a finding until interpreted against
    a proper detection-power calibration AND an independent
    second-window replication check. Both failed to hold for XRP on
    its very first attempt tonight -- treat a promising TAO reading
    with at least that much skepticism, not less.
""")


if __name__ == '__main__':
    main()
