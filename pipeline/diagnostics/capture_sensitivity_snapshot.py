"""
pipeline/diagnostics/capture_sensitivity_snapshot.py

Captures ONE frozen live-data snapshot (raw trades + rebuild.py's
bars/close/threshold/events/w/tw) to disk, so the Tier-3 constant
sensitivity sweep (ROLL_WINDOW, VPIN_WINDOW, FFD_THRES, S) can compare
alternate constant values against the SAME underlying market data --
otherwise live data's own drift between pulls would confound the
comparison with the constant actually being tested.

Run ONCE per sweep session, before sensitivity_scan.py. Does NOT touch
run_pipeline_live.py's own live_staging_data/ or latest_live_report.txt --
writes only to its own dated snapshot folder under pipeline/diagnostics/.

Usage
-----
    conda activate mlfinlab
    cd C:\ws\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\diagnostics\capture_sensitivity_snapshot.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion import pull_recent_trades           # noqa: E402
from rebuild import build_bars_and_labels           # noqa: E402

LOOKBACK_HOURS = 720   # same as run_pipeline_live.py -- LIVE-CONFIRMED
                        # minimum for get_daily_vol() to have prior bars

SNAPSHOT_DIR = os.path.join(
    HERE, f'sensitivity_snapshot_{date.today().isoformat()}'
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
            'snapshot that a sweep may already be using. Delete it '
            'manually first if you really want a fresh capture today.'
        )
    os.makedirs(SNAPSHOT_DIR)

    print(f'Pulling last {LOOKBACK_HOURS}h of BTCUSDT trades from Binance.US...')
    raw_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    rebuild_result = build_bars_and_labels(raw_trades)
    print(f"  {len(rebuild_result['bars'])} bars, "
          f"{len(rebuild_result['events'])} triple-barrier events, "
          f"threshold=${rebuild_result['threshold']:,.2f}")

    raw_trades.to_parquet(os.path.join(SNAPSHOT_DIR, 'raw_trades.parquet'))
    rebuild_result['bars'].to_parquet(os.path.join(SNAPSHOT_DIR, 'bars.parquet'))
    rebuild_result['close'].to_frame('close').to_parquet(
        os.path.join(SNAPSHOT_DIR, 'close.parquet'))
    rebuild_result['events'].to_parquet(os.path.join(SNAPSHOT_DIR, 'events.parquet'))
    rebuild_result['w'].to_frame('w').to_parquet(
        os.path.join(SNAPSHOT_DIR, 'w.parquet'))
    rebuild_result['tw'].to_frame('tw').to_parquet(
        os.path.join(SNAPSHOT_DIR, 'tw.parquet'))
    with open(os.path.join(SNAPSHOT_DIR, 'threshold.txt'), 'w') as f:
        f.write(repr(float(rebuild_result['threshold'])))

    print(f'\nSnapshot frozen to {SNAPSHOT_DIR}')
    print('This exact dataset will be reused for EVERY constant in the sweep.')
    print('Do not re-run this script mid-sweep -- later constants would end up')
    print('compared against different underlying market data than earlier ones.')


if __name__ == '__main__':
    main()