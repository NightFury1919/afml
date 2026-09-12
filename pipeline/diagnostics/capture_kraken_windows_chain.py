"""
pipeline/diagnostics/capture_kraken_windows_chain.py

v2 (2026-09-10): switched from a sequential chain (each window's end
time read from the PREVIOUS window's actual downloaded data) to
concurrent pulls, after establishing this is I/O-bound work against a
shared external API -- a different bottleneck class from the CPU-bound
Monte Carlo work Ch20's mp_job_list was wired into earlier today.
Uses a THREAD pool (concurrent.futures.ThreadPoolExecutor), not
mp_job_list/multiprocessing -- the right tool for I/O-bound waiting on
network calls, where processes would just add pickling overhead for no
benefit.

*** WHY THIS NEEDED A DESIGN CHANGE, NOT JUST A THREAD POOL BOLTED ON ***
The original version computed window N+1's end_time_unix FROM window
N's actual downloaded earliest-trade timestamp -- deliberately, to
avoid clock drift between two separately-timed `time.time()` calls.
That is exactly the dependency that forced strict sequencing. Fixed by
computing ALL window boundaries upfront via pure arithmetic from ONE
already-known, fixed real anchor (the earliest window already on disk,
e.g. window 2's actual earliest trade timestamp) -- boundaries are
`anchor_start - k*lookback_hours*3600` for k=1,2,3..., no live data
needed to compute any of them. This removes the interdependency
entirely, which is the real prerequisite for concurrency (not just
"wrap it in a thread pool").

*** CAUTION LEVEL: real, not performative ***
ingestion_kraken.py's own LOAD-BEARING note explains Kraken's public
trades endpoint isn't subject to the same tiered per-IP counter system
its authenticated endpoints are -- but nothing in this project has ever
tested concurrent multi-thousand-call pulls against it. Default
concurrency here is 2, not 6-8, deliberately: this is a real first test
of untested behavior, not an assumed-safe optimization. Each thread's
pull_recent_trades_kraken() call already has its own 429 backoff-retry
built in (session=None per call, no shared state between threads) --
if you see repeated 429s in the interleaved output, STOP, drop back to
sequential (--max-concurrent 1), and reconsider before raising
concurrency further.

Zero-overlap is verified per window AFTER all pulls complete, comparing
each window's actual latest trade against its own REQUESTED end
boundary (not another window's actual data, since with concurrent pulls
there's no guaranteed completion order to compare against).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_kraken_windows_chain.py --num-windows 2 --max-concurrent 2
    python pipeline\\diagnostics\\capture_kraken_windows_chain.py --num-windows 6 --max-concurrent 2
    python pipeline\\diagnostics\\capture_kraken_windows_chain.py --num-windows 1 --max-concurrent 1  (old sequential-equivalent, single window)
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

PAIR = 'XBTUSD'
LOOKBACK_HOURS = 720
DEFAULT_ANCHOR = os.path.join(HERE, 'kraken_snapshot_720h_window2_2026-09-08')
DEFAULT_NUM_WINDOWS = 2
DEFAULT_MAX_CONCURRENT = 2

# Real observed rates so far (both real pulls, density varies a lot
# window to window, NOT a fixed constant): window 1 ~3154/hour, window 2
# ~1886/hour. Averaged for a rough pre-pull estimate only.
ESTIMATED_RATE_PER_HOUR = 2500


def _pull_one_window(end_time_unix, window_label):
    """Runs in a worker thread. Pulls one window at a precomputed,
    already-fixed end_time_unix -- no dependency on any other window's
    result. Returns (window_label, snapshot_dir, raw_trades, elapsed_min)
    or raises."""
    snapshot_dir = os.path.join(
        HERE, f'kraken_snapshot_720h_{window_label}_{date.today().isoformat()}'
    )
    if os.path.exists(snapshot_dir):
        raise FileExistsError(
            f'{snapshot_dir} already exists -- refusing to overwrite.'
        )
    os.makedirs(snapshot_dir)

    est_trades = ESTIMATED_RATE_PER_HOUR * LOOKBACK_HOURS
    est_calls = int(est_trades / 1000) + 10
    max_calls = max(est_calls * 2, 600)
    print(f'[{window_label}] starting: ending {pd.to_datetime(end_time_unix, unit="s")}, '
          f'~{est_trades:,} est. trades, max_calls={max_calls:,}')

    t0 = time.time()
    raw_trades = pull_recent_trades_kraken(
        PAIR, LOOKBACK_HOURS, max_calls=max_calls, end_time_unix=end_time_unix,
    )
    elapsed_min = (time.time() - t0) / 60.0
    print(f'[{window_label}] done: {len(raw_trades)} trades in {elapsed_min:.1f} min')

    out_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    raw_trades.to_parquet(out_path)
    return window_label, snapshot_dir, raw_trades, elapsed_min, end_time_unix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anchor', type=str, default=DEFAULT_ANCHOR,
                         help='Snapshot dir whose EARLIEST trade anchors all new '
                              'window boundaries (default: window 2).')
    parser.add_argument('--num-windows', type=int, default=DEFAULT_NUM_WINDOWS,
                         help=f'Additional windows to capture (default {DEFAULT_NUM_WINDOWS}).')
    parser.add_argument('--max-concurrent', type=int, default=DEFAULT_MAX_CONCURRENT,
                         help=f'Threads running simultaneously (default '
                              f'{DEFAULT_MAX_CONCURRENT} -- this is genuinely untested '
                              f'territory, see module docstring; do not jump straight '
                              f'to a high value).')
    parser.add_argument('--start-index', type=int, default=3,
                         help='Label index for the first new window (default 3).')
    args = parser.parse_args()

    anchor_path = os.path.join(args.anchor, 'raw_trades.parquet')
    if not os.path.exists(anchor_path):
        raise SystemExit(f'{anchor_path} not found -- check --anchor.')
    anchor_start_us = int(pd.read_parquet(anchor_path)['Timestamp'].min())

    # All boundaries computed NOW, from ONE fixed real anchor -- no
    # dependency on any window's actual pulled data. See module
    # docstring for why this is the real prerequisite for concurrency.
    # Boundary depends on the ABSOLUTE window label (start_index + i),
    # not the loop position -- window3 is offset 0 from the anchor
    # (ends exactly at window2's start), window4 is offset 1*720h
    # further back, etc. This makes re-running with a different
    # --start-index to resume/continue the chain safe: each label always
    # maps to the same real time period regardless of how many windows
    # are requested in a given invocation. (Bug found before it could
    # cause a silent duplicate pull: the first version keyed the offset
    # to the loop index alone, so --start-index 4 would have recomputed
    # window3's exact boundary and mislabeled it "window4".)
    boundaries = []
    for i in range(args.num_windows):
        offset_windows = (args.start_index + i) - 3
        boundaries.append(
            anchor_start_us / 1_000_000.0 - offset_windows * LOOKBACK_HOURS * 3600
        )
    labels = [f'window{args.start_index + i}' for i in range(args.num_windows)]

    print(f'Anchor: {args.anchor} (earliest trade '
          f'{pd.to_datetime(anchor_start_us, unit="us")})')
    print(f'{args.num_windows} new window(s) planned, boundaries computed upfront:')
    for label, b in zip(labels, boundaries):
        print(f'  {label}: ends {pd.to_datetime(b, unit="s")}')
    print(f'max_concurrent={args.max_concurrent} -- see module docstring\'s caution '
          f'note before raising this.')
    confirm = input('\nProceed? [y/N] ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        return

    results = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
        futures = {
            pool.submit(_pull_one_window, b, label): label
            for label, b in zip(labels, boundaries)
        }
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                _, snapshot_dir, raw_trades, elapsed_min, requested_end = fut.result()
                results[label] = (snapshot_dir, raw_trades, elapsed_min, requested_end)
            except Exception as e:
                print(f'[{label}] FAILED: {type(e).__name__}: {e}')
                failures[label] = e

    print('\n' + '=' * 70)
    print('RESULTS')
    print('=' * 70)
    for label in labels:
        if label not in results:
            print(f'  {label}: FAILED -- {failures.get(label)}')
            continue
        snapshot_dir, raw_trades, elapsed_min, requested_end = results[label]
        span_hours = (
            raw_trades['Timestamp'].max() - raw_trades['Timestamp'].min()
        ) / 1_000_000 / 3600.0
        actual_rate = len(raw_trades) / span_hours if span_hours > 0 else float('nan')
        latest_ts = raw_trades['Timestamp'].max()
        requested_end_us = int(requested_end * 1_000_000)
        overlap_ok = latest_ts <= requested_end_us
        print(f'  {label}: {snapshot_dir}')
        print(f'    {len(raw_trades)} trades, {elapsed_min:.1f} min, '
              f'{span_hours:.2f}h span, {actual_rate:.1f} trades/hour')
        print(f'    Zero-overlap check (latest trade <= requested end boundary): '
              f'{"PASS" if overlap_ok else "FAIL -- investigate!"}')

    n_ok = len(results)
    n_failed = len(failures)
    print(f'\n{n_ok}/{args.num_windows} windows captured successfully'
          + (f', {n_failed} FAILED' if n_failed else ''))
    if n_failed:
        print('Re-run with --start-index set past the successful ones to retry '
              'just the failures (check which labels succeeded above first).')

    if results:
        print('\nNext: run the target_bars sweep on each new window, e.g.:')
        for label, (snapshot_dir, *_rest) in results.items():
            print(f'  python pipeline\\diagnostics\\calibrate_kraken_target_bars.py '
                  f'{snapshot_dir} kraken_target_bars_calibration_{label}.csv')


if __name__ == '__main__':
    main()
