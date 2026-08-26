"""
pipeline/diagnostics/capture_xrp_snapshot_window2.py

XRP counterpart to capture_kraken_snapshot_window2.py -- same real
mechanism (ingestion_kraken.pull_recent_trades_kraken()'s end_time_unix
parameter, anchored to window 1's earliest trade so the two windows
share zero raw trades), applied to XRPUSD instead of XBTUSD.

Motivation: tonight's real XRP detection-power calibration
(calibrate_xrp_detection_power.py, run against xrp_target_bars_
calibration.csv) found a DSR-vs-null-baseline gap pattern that is
positive at 4 of 5 T_effective values but NOT monotonic or stable --
implied edge size swings 0.11 -> 0.15 -> 0.06 -> 0.07 (and negative at
the smallest T) rather than holding roughly constant the way a genuine
fixed edge should. This is structurally the same shape Kraken BTC's
window-1 result showed before its own second-window replication check
(capture_kraken_snapshot_window2.py) killed it as noise. All 5 XRP
readings came from ONE shared 720h window, so -- same reasoning as the
Kraken BTC precedent -- this script pulls a genuinely independent
second window (same length, same pair, non-overlapping calendar time)
so the whole target_bars -> detection_power pipeline can be re-run on
it and compared. This is the specific check the 2026-08-25/26 handoff
flags as NOT yet built for XRP; this script builds it.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_xrp_snapshot_window2.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

PAIR = 'XRPUSD'
LOOKBACK_HOURS = 720
WINDOW1_SNAPSHOT_DIR = os.path.join(HERE, 'kraken_snapshot_xrpusd_720h_2026-08-25')

SNAPSHOT_DIR = os.path.join(
    HERE, f'kraken_snapshot_xrpusd_720h_window2_{date.today().isoformat()}'
)


def main():
    import pandas as pd

    window1_path = os.path.join(WINDOW1_SNAPSHOT_DIR, 'raw_trades.parquet')
    if not os.path.exists(window1_path):
        raise SystemExit(
            f'{window1_path} not found -- window 1 must exist first, so '
            'window 2 can be anchored to its exact start time.'
        )
    window1 = pd.read_parquet(window1_path)
    # window1's earliest Timestamp (microseconds since epoch) is where
    # window 2 should END -- guarantees zero overlap between the two
    # windows, same reasoning as the Kraken BTC version of this script.
    window1_start_us = int(window1['Timestamp'].min())
    end_time_unix = window1_start_us / 1_000_000.0

    print(f'Window 1 starts at {pd.to_datetime(window1_start_us, unit="us")} '
          f'-- window 2 will end exactly there (zero overlap).')

    if os.path.exists(SNAPSHOT_DIR):
        raise SystemExit(
            f'{SNAPSHOT_DIR} already exists -- refusing to overwrite. '
            'Delete it manually first if you really want a fresh capture.'
        )
    os.makedirs(SNAPSHOT_DIR)

    # Rate estimate from the real window-1 XRP capture (~1,079.5
    # trades/hour, confirmed 2026-08-25/26 handoff) -- not Kraken BTC's
    # ~2,300-4,105/hour, which would badly under-estimate max_calls if
    # reused here by mistake.
    est_trades = 1080 * LOOKBACK_HOURS
    est_calls = int(est_trades / 1000) + 10
    max_calls = max(est_calls * 2, 600)
    print(f'Pulling {LOOKBACK_HOURS}h of {PAIR} trades ending '
          f'{pd.to_datetime(end_time_unix, unit="s")}...')
    print(f'  Estimated ~{est_trades:,} trades, ~{est_calls:,} calls, '
          f'max_calls={max_calls:,} -- budget similar time to window 1\'s '
          'capture.')

    raw_trades = pull_recent_trades_kraken(
        PAIR, LOOKBACK_HOURS, max_calls=max_calls, end_time_unix=end_time_unix,
    )
    print(f'  {len(raw_trades)} raw trades pulled')

    out_path = os.path.join(SNAPSHOT_DIR, 'raw_trades.parquet')
    raw_trades.to_parquet(out_path)

    span_hours = (
        raw_trades['Timestamp'].max() - raw_trades['Timestamp'].min()
    ) / 1_000_000 / 3600.0
    actual_rate = len(raw_trades) / span_hours if span_hours > 0 else float('nan')
    latest_ts = raw_trades['Timestamp'].max()
    overlap_check = latest_ts <= window1_start_us
    print(f'\nSnapshot frozen to {SNAPSHOT_DIR}')
    print(f'  Actual span: {span_hours:.2f}h, actual rate: '
          f'{actual_rate:.1f} trades/hour')
    print(f'  Zero-overlap check (window 2\'s latest trade <= window 1\'s '
          f'earliest trade): {"PASS" if overlap_check else "FAIL -- investigate!"}')
    print('\nNext: run calibrate_xrp_target_bars.py on this snapshot with '
          'a distinct output CSV, e.g.:')
    print(f'  python pipeline\\diagnostics\\calibrate_xrp_target_bars.py '
          f'{SNAPSHOT_DIR} pipeline\\diagnostics\\xrp_target_bars_window2.csv')
    print('\nThen re-run calibrate_xrp_detection_power.py against that '
          'window 2 CSV (it already accepts <sweep_csv> [output_csv] args '
          'for exactly this), e.g.:')
    print('  python pipeline\\diagnostics\\calibrate_xrp_detection_power.py '
          'pipeline\\diagnostics\\xrp_target_bars_window2.csv '
          'pipeline\\diagnostics\\xrp_detection_power_window2.csv')


if __name__ == '__main__':
    main()
