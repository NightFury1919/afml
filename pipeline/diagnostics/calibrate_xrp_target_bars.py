"""
pipeline/diagnostics/calibrate_xrp_target_bars.py

XRP counterpart to calibrate_kraken_target_bars.py -- same real chain
(rebuild.py/features.py/live_staging.py/stages.py), same target_bars
grid, same sweep structure. The one real difference: CUSUM_H is
monkeypatched to XRP's own calibrated value for the duration of this
sweep, NOT left at the BTC-calibrated production default (313).

*** LOAD-BEARING (2026-08-25): CUSUM_H=0.0113 for XRP, NOT the BTC
production default ***
Derived via calibrate_xrp_cusum_h.py (real measurement: XRP/BTC bar-to-
bar diff std ratio 0.00003613x, applied to BTC's current CUSUM_H=313)
and confirmed via sanity_check_xrp_cusum_h.py on a REAL 720h XRP pull
(776,755 trades): 31.8% CUSUM event rate, comfortably inside BTC's own
real reference range (25-35%) -- PASS. Reusing BTC's CUSUM_H=313
unchanged on XRP data would either produce zero CUSUM events (XRP's
bar-to-bar moves are tiny fractions of a dollar vs. BTC's ~$70k price
level) or crash build_bars_and_labels() outright -- confirmed directly,
not assumed.

STILL UNVALIDATED for XRP as of this sweep: MIN_RET (0.005) and
VERTICAL_BARRIER_NUM_DAYS (3) are reused unchanged from the BTC
production defaults. MIN_RET is a fractional/percentage threshold, not
a dollar one, so it's less likely to need re-derivation the way
CUSUM_H did -- but this has NOT been independently confirmed for XRP's
own volatility character, only assumed reasonable. Same "first look,
not a validated final calibration" status calibrate_kraken_target_bars.py
carried for Kraken BTC data relative to Binance.US.

Run (after capture_kraken_snapshot.py --pair XRPUSD --hours 720 has
produced the snapshot):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_xrp_target_bars.py <snapshot_dir>
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

OUTPUT_CSV = os.path.join(HERE, 'xrp_target_bars_calibration.csv')
SWEEP_COLUMNS = [
    'target_bars', 'n_bars', 'n_events', 'n_events_enriched', 'T_raw',
    'tw_mean', 'T_effective', 'best_sharpe', 'pbo', 'dsr', 'skew',
    'kurtosis', 'notes',
]

TARGET_BARS_GRID = [1000, 2000, 3000, 4000, 5000]
XRP_CUSUM_H = 0.0113  # see module docstring


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
            'Usage: python calibrate_xrp_target_bars.py <snapshot_dir> [output_csv]\n'
            'Run capture_kraken_snapshot.py --pair XRPUSD --hours 720 first.'
        )
    snapshot_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else OUTPUT_CSV
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen XRP snapshot: {len(raw_trades)} raw trades '
          f'from {snapshot_dir}')
    print(f'Writing results to: {output_csv}')
    print(f'Using XRP-calibrated CUSUM_H={XRP_CUSUM_H} for this entire '
          f'sweep (monkeypatched, restored on exit)')

    work_root = os.path.join(HERE, 'xrp_target_bars_sweep_work')
    os.makedirs(work_root, exist_ok=True)

    original_cusum_h = rebuild_module.CUSUM_H
    rows = []
    try:
        rebuild_module.CUSUM_H = XRP_CUSUM_H
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
    print('SUMMARY: T_effective by target_bars (XRP)')
    print('=' * 66)
    print(df[['target_bars', 'T_raw', 'tw_mean', 'T_effective', 'dsr', 'pbo']]
          .to_string(index=False))
    print("""
INTERPRETATION:
  - Compare against tonight's real BTC results on the SAME real chain:
    Binance.US target_bars=1000 plateau (~180 T_effective), Kraken BTC
    target_bars=5000 (T_effective=758.19, no plateau reached).
  - Remember: only CUSUM_H has been re-derived and sanity-checked for
    XRP specifically. MIN_RET and VERTICAL_BARRIER_NUM_DAYS are still
    reused unchanged from BTC's production defaults -- a real result
    here is grounds to validate those too, not a signal they're
    already fine.
  - This sweep alone does NOT answer whether XRP's scouted negative
    autocorrelation (scout_alternative_kraken_assets.py) survives
    proper triple-barrier labeling and PBO/DSR scrutiny -- that's what
    these DSR/PBO numbers are for. Read them with the SAME caution as
    tonight's BTC Kraken sweep: promising numbers alone are not a
    finding until interpreted against a proper detection-power
    calibration (see calibrate_kraken_detection_power.py -- an XRP
    version of that script is a real, separate next step if this
    sweep looks promising).
""")


if __name__ == '__main__':
    main()
