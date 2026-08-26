"""
pipeline/diagnostics/capture_kraken_snapshot_window2.py

Captures a SECOND, non-overlapping 720h Kraken window -- specifically
the 720h immediately BEFORE the window already captured by
capture_kraken_snapshot.py --hours 720 (kraken_snapshot_720h_2026-08-25/,
covering roughly the most recent 30 days as of 2026-08-25).

Motivation: today's real Kraken target_bars sweep found every one of 5
DSR readings sitting clearly above their fat-tailed null baseline
(gaps +0.06 to +0.18) -- but the implied edge size shrinks steadily as
T_effective rises (roughly 0.20 -> 0.07), which looks more like a
residual selection-bias artifact than one stable, genuine edge. All 5
readings came from ONE shared 720h window, so this pattern could easily
be a property of that one window's particular history rather than
anything general. This script pulls a genuinely independent second
window (same length, same pair, non-overlapping calendar time) so the
whole target_bars -> detection_power pipeline can be re-run on it and
compared -- the closest thing to an out-of-sample replication check
available without waiting for new calendar time to pass.

Uses ingestion_kraken.pull_recent_trades_kraken()'s new end_time_unix
parameter (added 2026-08-25 specifically for this) -- window 2 ends
exactly where window 1 begins, so the two windows share zero raw
trades.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_kraken_snapshot_window2.py
"""
import os
import sys
import time
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

PAIR = 'XBTUSD'
LOOKBACK_HOURS = 720
WINDOW1_SNAPSHOT_DIR = os.path.join(HERE, 'kraken_snapshot_720h_2026-08-25')

SNAPSHOT_DIR = os.path.join(
    HERE, f'kraken_snapshot_720h_window2_{date.today().isoformat()}'
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
    # windows, rather than relying on two separately-computed "time.time()
    # minus N hours" calls that could drift apart by however long window
    # 1's own capture took to run.
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

    est_trades = 2300 * LOOKBACK_HOURS  # rough, based on window 1's
                                          # observed ~2325-2331 trades/hour
    est_calls = int(est_trades / 1000) + 10
    max_calls = max(est_calls * 2, 600)
    print(f'Pulling {LOOKBACK_HOURS}h of {PAIR} trades ending '
          f'{pd.to_datetime(end_time_unix, unit="s")}...')
    print(f'  Estimated ~{est_trades:,} trades, ~{est_calls:,} calls, '
          f'max_calls={max_calls:,} -- budget similar time to window 1\'s '
          'capture (roughly 45-60+ minutes).')

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
    print('\nNext: run calibrate_kraken_target_bars.py on this snapshot with '
          'a distinct output CSV, e.g.:')
    print(f'  python pipeline\\diagnostics\\calibrate_kraken_target_bars.py '
          f'{SNAPSHOT_DIR} pipeline\\diagnostics\\kraken_target_bars_window2.csv')


if __name__ == '__main__':
    main()
