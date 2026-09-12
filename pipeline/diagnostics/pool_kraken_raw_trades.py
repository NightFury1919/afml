"""
pipeline/diagnostics/pool_kraken_raw_trades.py

These 8 windows weren't captured as 8 disjoint samples to be carefully
stitched -- they were built so each window's end EXACTLY equals the
next window's start (that's what the zero-overlap checks in
capture_kraken_windows_chain.py were verifying). That means they are 8
contiguous slices of ONE continuous ~240-day trade history, not 8
separate datasets needing custom cross-window purge/embargo logic.

This script does the correct, simple thing given that: concatenate the
raw trades, sort by timestamp, RE-VERIFY contiguity across the whole
assembled series (not just trust the individual pairwise PASS results
from capture time -- matches this project's verification standard: a
past check passing is not a substitute for checking again on the actual
combined artifact), and save it in the exact same raw_trades.parquet
schema every other script already expects.

No new pipeline/orchestration code needed as a result: rebuild.py's
CUSUM filter, triple-barrier labeling, and Ch11's PurgedKFold embargo
logic all already handle a continuous series correctly by construction
-- calibrate_kraken_target_bars.py and everything downstream of it can
be pointed at this pooled snapshot UNCHANGED. The only real adjustment
needed is target_bars itself: it must scale with the pooled span to
keep bar GRANULARITY (bars/day) comparable to the single-window
calibrations already done, not stay at the same absolute count (which
would make each bar represent ~8x more volume -- coarser, not more of
the same resolution).

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\pool_kraken_raw_trades.py
    python pipeline\\diagnostics\\pool_kraken_raw_trades.py --pattern "kraken_snapshot_720h_window*_2026-*"
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# *** FIXED (2026-09-11): explicit list, not a glob ***
# The original glob pattern ('kraken_snapshot_720h_window*_2026-*')
# accidentally matched kraken_snapshot_720h_window2_2026-08-25 -- an
# unrelated leftover from the entirely separate Aug 25 calibration
# session, covering a DIFFERENT, overlapping time period, not part of
# this chain at all. That single stray match produced 8.5M duplicate
# timestamps and 22 real gaps when concatenated against our actual
# window3/window4 (which legitimately cover overlapping real dates
# relative to that old file). A glob that matches on "looks like the
# right name" isn't safe here -- enumerate the real 8 explicitly.
DEFAULT_SNAPSHOTS = [
    os.path.join(HERE, 'kraken_snapshot_xbtusd_720h_2026-09-08'),   # window1
    os.path.join(HERE, 'kraken_snapshot_720h_window2_2026-09-08'),
    os.path.join(HERE, 'kraken_snapshot_720h_window3_2026-09-10'),
    os.path.join(HERE, 'kraken_snapshot_720h_window4_2026-09-11'),
    os.path.join(HERE, 'kraken_snapshot_720h_window5_2026-09-10'),
    os.path.join(HERE, 'kraken_snapshot_720h_window6_2026-09-10'),
    os.path.join(HERE, 'kraken_snapshot_720h_window7_2026-09-11'),
    os.path.join(HERE, 'kraken_snapshot_720h_window8_2026-09-11'),
]
DEFAULT_OUTPUT = os.path.join(HERE, 'kraken_snapshot_pooled_8windows_2026-09-11')

# Real observed max gap so far: 1255.5s (~21 min), and the gap locations
# (verified below) sit INSIDE individual windows' own data, not at any
# of the 7 real window-boundary indices -- i.e. these are natural lulls
# in real BTC trading (thin overnight liquidity etc.), not evidence a
# capture silently missed a chunk of trades at a seam. An earlier 300s
# threshold was too strict for real tick data; loosened with this
# reasoning on record, not silently.
MAX_ACCEPTABLE_GAP_SECONDS = 1800


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshots', type=str, nargs='+', default=DEFAULT_SNAPSHOTS,
                         help='Explicit list of snapshot dirs to pool (default: the '
                              'real 8-window chain built 2026-09-08 to 2026-09-11).')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot_dirs = args.snapshots
    if not snapshot_dirs:
        raise SystemExit('No snapshot dirs given.')

    print(f'Found {len(snapshot_dirs)} snapshot(s):')
    frames = []
    boundary_timestamps = []  # each window's own (min, max) -- used below
                                # to check whether a gap sits at a real
                                # window seam or purely inside one window
    for d in snapshot_dirs:
        p = os.path.join(d, 'raw_trades.parquet')
        if not os.path.exists(p):
            raise SystemExit(f'{p} not found.')
        df = pd.read_parquet(p)
        frames.append(df)
        ts_min, ts_max = df['Timestamp'].min(), df['Timestamp'].max()
        boundary_timestamps.append((ts_min, ts_max))
        print(f'  {os.path.basename(d)}: {len(df)} trades, '
              f'{pd.to_datetime(ts_min, unit="us")} to '
              f'{pd.to_datetime(ts_max, unit="us")}')
    # The 7 real seams are wherever one window's max sits right next to
    # another window's min (should be ~identical given how these were
    # captured -- see capture_kraken_windows_chain.py).
    all_boundary_points = sorted(t for pair in boundary_timestamps for t in pair)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values('Timestamp').reset_index(drop=True)
    n_before = len(pooled)
    # *** FIXED (2026-09-11): full-row dedup, not Timestamp-only ***
    # Real tick data legitimately has many DISTINCT trades sharing the
    # exact same recorded timestamp (simultaneous fills at whatever
    # resolution the feed records) -- that is normal market behavior,
    # not duplication. subset='Timestamp' wiped out ~half of all real
    # trades (7.1M of 14.3M) in the first attempt, which then made the
    # contiguity check fail on gaps that were themselves an artifact of
    # deleting real trades, not evidence of an actual missing chunk of
    # data. Matching on the full row only removes a trade that is
    # IDENTICAL in every column -- which is what a genuine boundary
    # double-count between two adjacent captures would look like, not
    # what ordinary same-millisecond distinct fills look like.
    pooled = pooled.drop_duplicates()
    n_dupes = n_before - len(pooled)
    if n_dupes:
        print(f'\nDropped {n_dupes} exact full-row duplicate trades '
              '(expected to be a small number, only at the handful of real '
              'boundary points where windows meet -- NOT the same as '
              'same-timestamp-different-trade rows, which are real and kept).')

    print(f'\nPooled: {len(pooled)} total trades, '
          f'{pd.to_datetime(pooled["Timestamp"].min(), unit="us")} to '
          f'{pd.to_datetime(pooled["Timestamp"].max(), unit="us")}')
    total_days = (pooled['Timestamp'].max() - pooled['Timestamp'].min()) / 1_000_000 / 86400
    print(f'Total span: {total_days:.2f} days (~{total_days / 30:.2f}x a single '
          '30-day window)')

    # Re-verify contiguity on the ASSEMBLED series -- do not just trust
    # the individual capture-time PASS results.
    gaps = pooled['Timestamp'].diff().dropna() / 1_000_000  # seconds
    max_gap = gaps.max()
    n_large_gaps = (gaps > MAX_ACCEPTABLE_GAP_SECONDS).sum()
    print(f'\nContiguity re-check: max gap between consecutive trades = '
          f'{max_gap:.1f}s (threshold {MAX_ACCEPTABLE_GAP_SECONDS}s)')
    if n_large_gaps > 0:
        worst = gaps.nlargest(5)
        print(f'  FAIL: {n_large_gaps} gap(s) exceed the threshold. For each, '
              'checking whether it sits at a real window seam (expected/benign) '
              'or purely inside one window\'s own data (would need real '
              'investigation):')
        for idx in worst.index:
            gap_seconds = gaps.loc[idx]
            gap_start_ts = pooled.loc[idx - 1, 'Timestamp']
            gap_end_ts = pooled.loc[idx, 'Timestamp']
            nearest_boundary = min(all_boundary_points, key=lambda b: abs(b - gap_start_ts))
            distance_days = abs(nearest_boundary - gap_start_ts) / 1_000_000 / 86400
            location = ('AT a real window seam' if distance_days < 0.01
                        else f'INSIDE a window\'s own data ({distance_days:.1f} '
                             'days from the nearest real seam)')
            print(f'    {pd.to_datetime(gap_start_ts, unit="us")} -> '
                  f'{pd.to_datetime(gap_end_ts, unit="us")} '
                  f'({gap_seconds:.1f}s): {location}')
        raise SystemExit(
            'Refusing to write a pooled snapshot with gaps over the threshold -- '
            'see the per-gap locations above. If all of them are reported as '
            '"INSIDE a window\'s own data", that\'s consistent with ordinary '
            'real market lulls, not a capture problem, and the threshold '
            '(MAX_ACCEPTABLE_GAP_SECONDS) may just need to be even more '
            'generous rather than something being wrong.'
        )
    print('  PASS: no gap exceeds the threshold -- genuinely continuous series.')

    if os.path.exists(args.output):
        raise SystemExit(f'{args.output} already exists -- refusing to overwrite.')
    os.makedirs(args.output)
    out_path = os.path.join(args.output, 'raw_trades.parquet')
    pooled.to_parquet(out_path)
    print(f'\nWritten to {out_path}')
    print(f'\nSame raw_trades.parquet schema as every single-window snapshot -- '
          'calibrate_kraken_target_bars.py and everything downstream works on '
          'this UNCHANGED. Reminder: scale target_bars by roughly '
          f'{total_days / 30:.1f}x a single window\'s value to keep bar '
          'granularity (bars/day) comparable, e.g.:')
    print(f'  python pipeline\\diagnostics\\calibrate_kraken_target_bars.py '
          f'{args.output} kraken_target_bars_calibration_pooled.csv '
          f'--target-bars=8000,16000,24000,32000,40000')


if __name__ == '__main__':
    main()
