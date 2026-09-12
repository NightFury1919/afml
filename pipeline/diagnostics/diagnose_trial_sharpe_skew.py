"""
pipeline/diagnostics/diagnose_trial_sharpe_skew.py

The 2026-09-10 trending-regime investigation's detection-power readback
(calibrate_kraken_detection_power.py) turned up something that reframes
the whole day: BOTH Kraken windows (window 1, trending; window 2, flat)
show a similar pattern of positive DSR-vs-null-baseline gaps, while
Binance.US (also trending, comparable to window 1) is genuinely mixed
with two negative gaps. That lines up with EXCHANGE, not TREND -- the
opposite of what the day's trend/detrend work was testing for.

One concrete difference already visible in that same readback: the
fat-tailed jump-mixture calibration's TARGET SKEW (of the winning
trial's own bet-level RETURNS, used to feed deflated_sharpe_ratio) is
0.46 (window 1), 0.37 (window 2), 0.08 (Binance) -- both Kraken windows
show real positive skew there; Binance doesn't.

*** THIS IS A DIFFERENT OBJECT THAN THAT SKEW NUMBER, DELIBERATELY ***
calibrate_kraken_target_bars.py's CSV 'skew'/'kurtosis' columns are the
TIME-SERIES skew/kurtosis of ONE trial's (the winner's) bet-level return
stream -- what DSR itself needs. This script instead computes the
CROSS-SECTIONAL skew/kurtosis of the 20 TRIAL-LEVEL SHARPE RATIOS
(meta['sharpe_full_sample'] -- one number per trial, 20 trials) at each
target_bars. Nothing computes or saves this today; it's only ever been
visible as 20 printed numbers per run, never compared as a distribution
in its own right. If Kraken shows a consistent cross-sectional skew
here regardless of trend, and Binance doesn't, that's a direct,
market-behavior-independent signal pointing at bar/label CONSTRUCTION
(the Binance.US-calibrated MIN_RET/VERTICAL_BARRIER_NUM_DAYS/dynamic-
threshold constants, reused on Kraken unvalidated -- see
calibrate_kraken_target_bars.py's own docstring) rather than at BTC's
real trading behavior on either exchange.

Reuses run_live_trials/load_ch11_driver exactly as every other script
today has -- no new trial-construction logic, just a new measurement
taken on trial output that already existed.

Run (once per snapshot):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\diagnose_trial_sharpe_skew.py <snapshot_dir> [output_csv]
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ROOT)
sys.path.insert(0, ORCH)

from rebuild import build_bars_and_labels                # noqa: E402
from features import build_enriched_events                 # noqa: E402
from live_staging import stage_live_training_tables         # noqa: E402
from stages import load_ch11_driver, run_live_trials         # noqa: E402

TARGET_BARS_GRID = [1000, 2000, 3000, 4000, 5000]   # matches calibrate_kraken_target_bars.py
OUTPUT_CSV = os.path.join(HERE, 'trial_sharpe_skew_calibration.csv')
SWEEP_COLUMNS = [
    'target_bars', 'n_trials', 'n_events',
    'trial_sharpe_mean', 'trial_sharpe_std',
    'trial_sharpe_skew', 'trial_sharpe_kurtosis',
    'trial_sharpe_min', 'trial_sharpe_max', 'notes',
]


def _run_one_config(raw_trades, target_bars, work_root):
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    config_name = f'tb{target_bars}'
    staging_dir = os.path.join(work_root, config_name, 'staging')
    here_dir = os.path.join(work_root, config_name, 'ch11_here')
    stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)
    sharpes = meta['sharpe_full_sample'].values

    return {
        'target_bars': target_bars,
        'n_trials': len(sharpes),
        'n_events': enriched_result['n_events_after'],
        'trial_sharpe_mean': float(np.mean(sharpes)),
        'trial_sharpe_std': float(np.std(sharpes, ddof=1)),
        'trial_sharpe_skew': float(stats.skew(sharpes)),
        'trial_sharpe_kurtosis': float(stats.kurtosis(sharpes, fisher=False)),
        'trial_sharpe_min': float(np.min(sharpes)),
        'trial_sharpe_max': float(np.max(sharpes)),
        'notes': '',
    }


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            'Usage: python diagnose_trial_sharpe_skew.py <snapshot_dir> [output_csv]'
        )
    snapshot_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) == 3 else OUTPUT_CSV
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')
    print(f'Writing results to: {output_csv}')

    work_root = os.path.join(HERE, 'trial_sharpe_skew_work')
    os.makedirs(work_root, exist_ok=True)

    rows = []
    for target_bars in TARGET_BARS_GRID:
        print(f'\n=== target_bars={target_bars} ===')
        try:
            row = _run_one_config(raw_trades, target_bars, work_root)
            rows.append(row)
            print(f"  n_trials={row['n_trials']}, n_events={row['n_events']}")
            print(f"  trial-level Sharpe: mean={row['trial_sharpe_mean']:.4f}, "
                  f"std={row['trial_sharpe_std']:.4f}, "
                  f"skew={row['trial_sharpe_skew']:+.4f}, "
                  f"kurtosis={row['trial_sharpe_kurtosis']:.4f}, "
                  f"range=[{row['trial_sharpe_min']:.4f}, {row['trial_sharpe_max']:.4f}]")
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            rows.append({
                'target_bars': target_bars, 'n_trials': None, 'n_events': None,
                'trial_sharpe_mean': None, 'trial_sharpe_std': None,
                'trial_sharpe_skew': None, 'trial_sharpe_kurtosis': None,
                'trial_sharpe_min': None, 'trial_sharpe_max': None,
                'notes': f'{type(e).__name__}: {e}',
            })

    df = pd.DataFrame(rows)[SWEEP_COLUMNS]
    df.to_csv(output_csv, index=False)

    print('\n' + '=' * 78)
    print('SUMMARY: cross-sectional skew/kurtosis of the 20 trial-level Sharpes')
    print('=' * 78)
    print(df[['target_bars', 'n_trials', 'trial_sharpe_mean', 'trial_sharpe_std',
              'trial_sharpe_skew', 'trial_sharpe_kurtosis']].to_string(index=False))
    mean_skew = df['trial_sharpe_skew'].mean()
    print(f'\nMean trial_sharpe_skew across target_bars: {mean_skew:+.4f}')
    print('\nCompare this mean across snapshots (window1, window2, Binance). If both '
          'Kraken windows land at a similarly positive value regardless of target_bars '
          'while Binance sits near zero, that points at bar/label CONSTRUCTION '
          '(exchange-specific), not at BTC market behavior or trend -- the opposite of '
          "today's earlier trend/detrend conclusion. Written to " + output_csv)


if __name__ == '__main__':
    main()
