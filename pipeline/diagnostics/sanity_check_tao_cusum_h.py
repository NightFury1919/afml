"""
pipeline/diagnostics/sanity_check_tao_cusum_h.py

TAO counterpart to sanity_check_xrp_cusum_h.py -- Step 1 of
calibrate_tao_cusum_h.py's own stated "before trusting it" checklist:
actually run ch02's REAL cusum_filter() at the candidate TAO CUSUM_H
on a real TAO bar series, and confirm a reasonable event rate results
(not zero -- the whole reason a fresh derivation was needed -- and not
every bar, which would mean the threshold is too LOW).

*** DIFFERENCE vs the XRP version (deliberate): reuses the ALREADY-
CAPTURED 720h snapshot instead of re-pulling ***
capture_kraken_snapshot.py --pair TAOUSD --hours 720 already ran
tonight (kraken_snapshot_taousd_720h_2026-08-26/raw_trades.parquet,
221,947 real trades, confirmed 308.3 trades/hour). Re-pulling fresh
data here the way the XRP version did would waste another 720h pull
for no real benefit -- same frozen-snapshot discipline this project
already uses elsewhere (capture_lookback_extension_snapshot.py,
capture_kraken_snapshot.py's own docstring: "one pull, saved once,
reused by whatever calibration/sweep work comes next, rather than
re-pulling live data mid-analysis"). Takes --snapshot_dir instead of
pulling live.

Inherits the XRP version's own real, load-bearing lesson: CALIBRATION_
HOURS must be >= this project's established LOOKBACK_HOURS minimum
(720) for triple-barrier labeling to have enough forward data to
resolve events -- a 24h window fails downstream of CUSUM even if the
candidate H itself is fine. Using the 720h snapshot here satisfies
that automatically.

Reference point: BTC's own real event rate at CUSUM_H=313 (measured
2026-08-25, audit_kraken_cusum_h_staleness.py): ~28.7-29.4% of bars.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\sanity_check_tao_cusum_h.py
    python pipeline\\diagnostics\\sanity_check_tao_cusum_h.py --h 0.0006
    python pipeline\\diagnostics\\sanity_check_tao_cusum_h.py --snapshot_dir <path>
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ORCH)
sys.path.insert(0, ROOT)

import rebuild as rebuild_module                          # noqa: E402
from rebuild import build_bars_and_labels                 # noqa: E402

TARGET_BARS = 1000
DEFAULT_SNAPSHOT_DIR = os.path.join(HERE, 'kraken_snapshot_taousd_720h_2026-08-26')

# DEFAULT_CANDIDATE_H is intentionally None here -- unlike the XRP
# version, no calibrate_tao_cusum_h.py run has produced a real number
# yet as this script is written. Pass --h explicitly with whatever
# calibrate_tao_cusum_h.py's real output prints as
# 'candidate_tao_cusum_h', or read it from tao_cusum_h_calibration.csv.
DEFAULT_CANDIDATE_H = None

# Reference: BTC's own real event rate range at its current CUSUM_H=313,
# measured 2026-08-25 (audit_kraken_cusum_h_staleness.py): 28.7%
# (Binance.US) to 29.4% (Kraken) of bars.
BTC_REFERENCE_EVENT_RATE_RANGE = (0.25, 0.35)  # a reasonable band around
                                                  # the real 28.7-29.4%
                                                  # measurement, not an
                                                  # exact target to hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=float, default=DEFAULT_CANDIDATE_H,
                         required=(DEFAULT_CANDIDATE_H is None),
                         help='Candidate CUSUM_H to sanity-check -- read '
                              'from calibrate_tao_cusum_h.py\'s real output '
                              '(tao_cusum_h_calibration.csv, '
                              'candidate_tao_cusum_h column).')
    parser.add_argument('--snapshot_dir', type=str, default=DEFAULT_SNAPSHOT_DIR,
                         help='Frozen snapshot directory with raw_trades.'
                              'parquet. Defaults to tonight\'s real 720h '
                              'TAOUSD capture.')
    args = parser.parse_args()

    raw_trades_path = os.path.join(args.snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir? '
                          'Run capture_kraken_snapshot.py --pair TAOUSD '
                          '--hours 720 first.')
    tao_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen TAO snapshot: {len(tao_trades)} raw trades '
          f'from {args.snapshot_dir}')

    original_h = rebuild_module.CUSUM_H
    try:
        rebuild_module.CUSUM_H = args.h
        print(f'\nRunning build_bars_and_labels() with CUSUM_H={args.h} '
              f'(monkeypatched, restored after this check)...')
        result = build_bars_and_labels(tao_trades, target_bars=TARGET_BARS)
    finally:
        rebuild_module.CUSUM_H = original_h

    n_bars = len(result['bars'])
    n_events = len(result['events'])
    event_rate = n_events / n_bars if n_bars > 0 else float('nan')

    print(f'\n=== Result ===')
    print(f'  n_bars: {n_bars}')
    print(f'  n_events (triple-barrier, post-labeling): {n_events}')
    print(f'  event rate: {event_rate:.1%} of bars')
    print(f'  BTC reference range: {BTC_REFERENCE_EVENT_RATE_RANGE[0]:.0%}-'
          f'{BTC_REFERENCE_EVENT_RATE_RANGE[1]:.0%} of bars')

    lo, hi = BTC_REFERENCE_EVENT_RATE_RANGE
    if lo <= event_rate <= hi:
        verdict = 'PASS -- comparable to BTC\'s own real event rate range'
    elif event_rate < lo:
        verdict = ('LOW -- threshold may still be too HIGH for TAO '
                    '(under-firing); consider a smaller candidate H')
    else:
        verdict = ('HIGH -- threshold may be too LOW for TAO '
                    '(over-firing); consider a larger candidate H')
    print(f'  verdict: {verdict}')

    print("""
This is one 720h sample at one candidate H -- a real, useful first
check, not a fully validated calibration. If PASS, this H is a
reasonable starting point for the real TAO target_bars sweep. If LOW
or HIGH, re-run this script with a manually adjusted --h before
proceeding further (e.g. python sanity_check_tao_cusum_h.py --h 0.001).

Also worth weighing regardless of verdict: TAO's real trade density
(308.3/hour on this same 720h snapshot) is well below BTC's or XRP's --
a real, separate risk to n_bars/T_effective viability that a passing
CUSUM_H event rate does NOT resolve on its own.
""")

    out_path = os.path.join(HERE, 'tao_cusum_h_sanity_check.csv')
    pd.DataFrame([{
        'candidate_h': args.h, 'n_bars': n_bars, 'n_events': n_events,
        'event_rate': event_rate, 'verdict': verdict,
    }]).to_csv(out_path, mode='a',
               header=not os.path.exists(out_path), index=False)
    print(f'Result appended to {out_path}')


if __name__ == '__main__':
    main()
