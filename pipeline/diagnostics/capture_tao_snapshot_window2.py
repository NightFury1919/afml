"""
pipeline/diagnostics/capture_tao_snapshot_window2.py

TAO counterpart to capture_xrp_snapshot_window2.py (which itself mirrored
capture_kraken_snapshot_window2.py's real mechanism -- ingestion_kraken.
pull_recent_trades_kraken()'s end_time_unix parameter, anchored to
window 1's earliest trade so the two windows share zero raw trades),
applied to TAOUSD instead of XRPUSD/XBTUSD.

Motivation: 2026-08-27's real TAO detection-power calibration
(calibrate_tao_detection_power.py, run against tao_target_bars_
calibration.csv) found every observed-DSR-vs-fat-tailed-null gap
NEGATIVE across all 5 T_effective values, and growing MORE negative as
T rises (-0.0149 at T=102.99 down to -0.1303 at T=355.01) -- the
opposite direction a real, underpowered edge would move as more
effective evidence accumulates, so on its face this argues against a
hidden edge rather than for one. The same calibration's Part 4 vs Part
5 comparison also found a genuine sample-size bottleneck (ideal-
conditions minimum-detectable-edge floor of 0.05 vs. TAO-realistic
floors of 0.15-0.30 across the real T range), so a real small edge
still cannot be ruled out on power grounds alone. The fat-tailed
regime's calibration match was also the loosest of any asset so far
(achieved skew=-0.0036 against TAO's real target skew=+0.3683, missing
sign and magnitude), so the null baseline itself is of uncertain
quality. None of that is resolved by a single window. All 5 TAO
readings came from ONE shared 720h window
(kraken_snapshot_taousd_720h_2026-08-26) -- same reasoning as the
Kraken BTC and XRP precedents -- this script pulls a genuinely
independent second window (same length, same pair, non-overlapping
calendar time) so the whole target_bars -> detection_power pipeline can
be re-run on it and compared. This is the specific check queued as the
2026-08-26 handoff's next item after this calibration; this script
builds it.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_tao_snapshot_window2.py
"""
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

PAIR = 'TAOUSD'
LOOKBACK_HOURS = 720
WINDOW1_SNAPSHOT_DIR = os.path.join(HERE, 'kraken_snapshot_taousd_720h_2026-08-26')

SNAPSHOT_DIR = os.path.join(
    HERE, f'kraken_snapshot_taousd_720h_window2_{date.today().isoformat()}'
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
    # windows, same reasoning as the Kraken BTC/XRP versions of this
    # script.
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

    # Rate estimate from the real window-1 TAO capture (~308.3
    # trades/hour, confirmed 2026-08-26 handoff) -- notably lower than
    # both Kraken BTC's ~2,300-4,105/hour and XRP's ~1,079.5/hour, which
    # would badly OVER-estimate max_calls if reused here by mistake
    # (harmless -- extra unused call budget -- but worth getting right
    # for an accurate runtime estimate below).
    est_trades = 308 * LOOKBACK_HOURS
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
    print('\nNext: run calibrate_tao_target_bars.py on this snapshot with '
          'a distinct output CSV, e.g.:')
    print(f'  python pipeline\\diagnostics\\calibrate_tao_target_bars.py '
          f'{SNAPSHOT_DIR} pipeline\\diagnostics\\tao_target_bars_window2.csv')
    print('\nThen re-run calibrate_tao_detection_power.py against that '
          'window 2 CSV (it already accepts <sweep_csv> [output_csv] args '
          'for exactly this), e.g.:')
    print('  python pipeline\\diagnostics\\calibrate_tao_detection_power.py '
          'pipeline\\diagnostics\\tao_target_bars_window2.csv '
          'pipeline\\diagnostics\\tao_detection_power_window2.csv')


if __name__ == '__main__':
    main()
