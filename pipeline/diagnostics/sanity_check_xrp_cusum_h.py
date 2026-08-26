"""
pipeline/diagnostics/sanity_check_xrp_cusum_h.py

Step 1 of calibrate_xrp_cusum_h.py's own stated "before trusting it"
checklist: actually run ch02's REAL cusum_filter() at the candidate
XRP CUSUM_H (measured 2026-08-25: ~0.0113) on a real XRP bar series,
and confirm a reasonable event rate results -- not zero (the whole
reason a fresh derivation was needed in the first place), and not
every bar (which would mean the threshold is now too LOW, over-firing
the opposite direction).

*** LOAD-BEARING (2026-08-25): CALIBRATION_HOURS raised 24 -> 720,
real bug found and fixed before this script's first real use ***
The first real-machine run of this script (24h window) failed with
"Triple-barrier labeling produced zero events" -- NOT a CUSUM_H
problem (the CUSUM step itself succeeded silently at 24h, meaning the
candidate H DID produce real events -- the failure happened one step
later, in triple-barrier labeling). Root cause: this project's own
established minimum, LOOKBACK_HOURS=720 (see CALIBRATION_AUDIT.md's
Tier-2 entry, "live-confirmed minimum for get_daily_vol() to have
prior bars"), exists specifically because VERTICAL_BARRIER_NUM_DAYS=3
needs 72+ hours of FORWARD data to resolve each event, plus enough
history for get_daily_vol()'s volatility estimate to stabilize. A 24h
window is far short of that -- most/all CUSUM events near a 24h pull's
end would have no future data to resolve against, regardless of
whether CUSUM_H itself is right. This script was written picking a
short window for speed without checking that pre-existing constraint
first -- a real design mistake, corrected here rather than silently
retried.

Reference point: BTC's own real event rate at CUSUM_H=313 (measured
tonight, audit_kraken_cusum_h_staleness.py): ~28.7-29.4% of bars.

If this sanity check passes (a comparable, non-degenerate event rate),
the candidate H is a reasonable STARTING point for the real
target_bars sweep -- not yet a fully validated calibration (see
calibrate_xrp_cusum_h.py's own remaining caveats: single short-sample
diff_std measurement, worth a second day's check eventually).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\sanity_check_xrp_cusum_h.py
    python pipeline\\diagnostics\\sanity_check_xrp_cusum_h.py --h 0.0113
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

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402
import rebuild as rebuild_module                          # noqa: E402
from rebuild import build_bars_and_labels                 # noqa: E402

CALIBRATION_HOURS = 720.0  # matches this project's own established
                             # LOOKBACK_HOURS minimum -- see this
                             # module's own LOAD-BEARING note on why
                             # a shorter window (24h) failed
TARGET_BARS = 1000
DEFAULT_CANDIDATE_H = 0.0113  # rounded from calibrate_xrp_cusum_h.py's
                                # real measured 0.011310 (2026-08-25)

# Reference: BTC's own real event rate range at its current CUSUM_H=313,
# measured tonight (audit_kraken_cusum_h_staleness.py): 28.7% (Binance.US)
# to 29.4% (Kraken) of bars.
BTC_REFERENCE_EVENT_RATE_RANGE = (0.25, 0.35)  # a reasonable band around
                                                  # the real 28.7-29.4%
                                                  # measurement, not an
                                                  # exact target to hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=float, default=DEFAULT_CANDIDATE_H,
                         help='Candidate CUSUM_H to sanity-check.')
    args = parser.parse_args()

    print(f'Pulling {CALIBRATION_HOURS}h of fresh XRPUSD trades from Kraken...')
    print('  (720h at XRP\'s real observed density (~1,600/hour) is a '
          'genuinely large pull -- ~1,150+ calls, budget 20-30+ minutes, '
          'same caution as every other large capture tonight)')
    est_calls = int(1650 * CALIBRATION_HOURS / 1000) + 50  # generous buffer
    max_calls = max(est_calls * 2, 600)
    xrp_trades = pull_recent_trades_kraken('XRPUSD', CALIBRATION_HOURS,
                                            max_calls=max_calls)
    print(f'  {len(xrp_trades)} raw trades pulled')

    original_h = rebuild_module.CUSUM_H
    try:
        rebuild_module.CUSUM_H = args.h
        print(f'\nRunning build_bars_and_labels() with CUSUM_H={args.h} '
              f'(monkeypatched, restored after this check)...')
        result = build_bars_and_labels(xrp_trades, target_bars=TARGET_BARS)
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
        verdict = ('LOW -- threshold may still be too HIGH for XRP '
                    '(under-firing); consider a smaller candidate H')
    else:
        verdict = ('HIGH -- threshold may be too LOW for XRP '
                    '(over-firing); consider a larger candidate H')
    print(f'  verdict: {verdict}')

    print("""
This is one 24h sample at one candidate H -- a real, useful first check,
not a fully validated calibration. If PASS, this H is a reasonable
starting point for the real XRP target_bars sweep. If LOW or HIGH,
re-run this script with a manually adjusted --h before proceeding
further (e.g. python sanity_check_xrp_cusum_h.py --h 0.02).
""")

    out_path = os.path.join(HERE, 'xrp_cusum_h_sanity_check.csv')
    pd.DataFrame([{
        'candidate_h': args.h, 'n_bars': n_bars, 'n_events': n_events,
        'event_rate': event_rate, 'verdict': verdict,
    }]).to_csv(out_path, mode='a',
               header=not os.path.exists(out_path), index=False)
    print(f'Result appended to {out_path}')


if __name__ == '__main__':
    main()
