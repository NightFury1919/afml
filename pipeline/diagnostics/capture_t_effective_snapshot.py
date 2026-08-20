"""
pipeline/diagnostics/capture_t_effective_snapshot.py

Captures ONE frozen raw_trades snapshot so calibrate_t_effective_levers.py
can compare target_bars/CUSUM_H/VERTICAL_BARRIER_NUM_DAYS against the SAME
underlying market data -- otherwise live data's own drift between pulls
would confound the comparison with the lever actually being tested.

Unlike capture_sensitivity_snapshot.py (2026-08-18), this snapshot freezes
ONLY raw_trades, not rebuild.py's downstream outputs -- target_bars,
CUSUM_H, and VERTICAL_BARRIER_NUM_DAYS are all upstream of/inside
build_bars_and_labels() itself, so every lever value requires re-running
the full rebuild chain from raw trades, not just the constants
sensitivity_scan.py's sweep (ROLL_WINDOW/VPIN_WINDOW/FFD_THRES/S) could
get away with patching downstream of a frozen rebuild_result.

Run ONCE before calibrate_t_effective_levers.py.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\\diagnostics\\capture_t_effective_snapshot.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion import pull_recent_trades           # noqa: E402

LOOKBACK_HOURS = 720   # same as run_pipeline_live.py -- LIVE-CONFIRMED
                        # minimum for get_daily_vol() to have prior bars

SNAPSHOT_DIR = os.path.join(
    HERE, f't_effective_snapshot_{date.today().isoformat()}'
)


def main():
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    if os.path.exists(SNAPSHOT_DIR):
        raise SystemExit(
            f'{SNAPSHOT_DIR} already exists -- refusing to overwrite a '
            'snapshot the T_effective sweep may already be using. Delete '
            'it manually first if you really want a fresh capture today.'
        )
    os.makedirs(SNAPSHOT_DIR)

    print(f'Pulling last {LOOKBACK_HOURS}h of BTCUSDT trades from Binance.US...')
    raw_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    raw_trades.to_parquet(os.path.join(SNAPSHOT_DIR, 'raw_trades.parquet'))

    print(f'\nSnapshot frozen to {SNAPSHOT_DIR}')
    print('This exact raw_trades will be reused for EVERY lever value in the sweep.')
    print('Do not re-run this script mid-sweep -- later lever values would end up')
    print('compared against different underlying market data than earlier ones.')


if __name__ == '__main__':
    main()