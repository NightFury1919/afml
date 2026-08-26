"""
pipeline/diagnostics/capture_kraken_snapshot.py

First large-scale, real-machine test of ingestion_kraken.py's
pull_recent_trades_kraken() -- following the same incremental
verification discipline as everything else this session (2h -> 24h
both clean per verify_kraken_pull.py's real output). This script is the
next real step up, NOT yet a full 720h pull -- defaults to 168h (7
days) as an intermediate real-machine check, since this is genuinely
new code running at a call-count scale (hundreds of sequential HTTP
calls) far beyond anything tested so far (24h only needed ~99 calls).

Once a 168h pull completes cleanly, re-run with --hours 720 for the
full-scale pull -- at Kraken's real observed density (~4,105
trades/hour, confirmed 2026-08-25), that's ~2,955 calls and likely
2.95M+ trades, comfortably MORE raw data than the entire 90-day/2160h
Binance.US pull from earlier today (343,038 trades) -- in roughly a
third of the calendar window, which also meaningfully reduces (though
doesn't eliminate) the single-window regime-dependency concern that
pull surfaced.

Same frozen-snapshot discipline as capture_lookback_extension_snapshot.py
-- one pull, saved once, reused by whatever calibration/sweep work comes
next, rather than re-pulling live data mid-analysis.

*** WIDENED (2026-08-25, later same session): --pair argument added ***
Originally BTC-only (PAIR hardcoded to 'XBTUSD'). Ethan's cross-asset
scouting (scout_alternative_kraken_assets.py) found XRP/SOL/ETH all
showing meaningfully negative lag-1 autocorrelation at reasonable
density, held up across a 6h and a 24h pull -- worth building the real
pipeline for, starting with XRP (strongest, most consistent signal +
best density). --pair defaults to 'XBTUSD' so every existing call site
(and every command in tonight's own handoff notes) still works
unchanged.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\capture_kraken_snapshot.py
    python pipeline\\diagnostics\\capture_kraken_snapshot.py --hours 720
    python pipeline\\diagnostics\\capture_kraken_snapshot.py --pair XRPUSD --hours 720
"""
import argparse
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

DEFAULT_PAIR = 'XBTUSD'
# Real-machine-confirmed rates, 2026-08-25 -- used ONLY to produce a
# rough pre-pull time/call estimate for the user (see main()); actual
# max_calls always includes a real safety margin on top of whichever
# rate applies, so an inaccurate estimate for a non-BTC pair costs
# nothing beyond a less-precise printed guess.
KNOWN_RATES_PER_HOUR = {
    'XBTUSD': 4105,    # verify_kraken_pull.py's 24h output
    'ETHUSD': 1145,    # scout_alternative_kraken_assets.py's 24h output
    'SOLUSD': 1415,    # scout_alternative_kraken_assets.py's 24h output
    'XRPUSD': 1619,    # scout_alternative_kraken_assets.py's 24h output
}
FALLBACK_RATE_PER_HOUR = 1000  # conservative guess for any other pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hours', type=float, default=168.0,
        help='Lookback window. Default 168h (7 days) is the recommended '
             'NEXT step after the 2h/24h verification pulls -- not yet '
             'the full 720h target. Re-run with --hours 720 once this '
             'completes cleanly.',
    )
    parser.add_argument(
        '--pair', type=str, default=DEFAULT_PAIR,
        help=f'Kraken pair alias, e.g. XBTUSD (default), XRPUSD, SOLUSD, '
             f'ETHUSD. Snapshot directory name includes the pair so '
             f'different assets\' snapshots never collide.',
    )
    args = parser.parse_args()
    pair = args.pair

    rate_per_hour = KNOWN_RATES_PER_HOUR.get(pair, FALLBACK_RATE_PER_HOUR)
    if pair not in KNOWN_RATES_PER_HOUR:
        print(f'NOTE: no real-machine-confirmed rate on file for {pair!r} -- '
              f'using a conservative fallback estimate ({FALLBACK_RATE_PER_HOUR}'
              f'/hour) for the pre-pull time estimate below. This only '
              f'affects the ESTIMATE printed before pulling, not correctness '
              f'-- max_calls still includes a real safety margin.')

    est_trades = rate_per_hour * args.hours
    est_calls = int(est_trades / 1000) + 10  # +10 buffer for safety
    max_calls = max(est_calls * 2, 600)  # 2x safety margin over the estimate

    est_minutes = (est_calls * 1.0) / 60.0  # sleep_seconds=1.0 default

    print('=' * 70)
    print(f'CAPTURING Kraken snapshot: pair={pair}, hours={args.hours}')
    print('=' * 70)
    print(f'  Estimated trades: ~{est_trades:,.0f}')
    print(f'  Estimated calls: ~{est_calls:,} (max_calls set to {max_calls:,})')
    print(f'  Estimated MINIMUM runtime: ~{est_minutes:.0f} minutes '
          f'(sleep_seconds alone -- actual will be longer with real '
          f'request latency; budget generously, same caution as this '
          f'project\'s other large-pull diagnostics)')
    if args.hours >= 500:
        print('\n  This is a LARGE pull. Recommended: confirm a smaller '
              '--hours value (e.g. 168) completes cleanly first if you '
              'have not already.')

    confirm = input('\nProceed? [y/N] ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        return

    snapshot_dir = os.path.join(
        HERE, f'kraken_snapshot_{pair.lower()}_{int(args.hours)}h_{date.today().isoformat()}'
    )
    if os.path.exists(snapshot_dir):
        raise SystemExit(
            f'{snapshot_dir} already exists -- refusing to overwrite. '
            'Delete it manually first if you really want a fresh capture.'
        )
    os.makedirs(snapshot_dir)

    print(f'\nPulling {args.hours}h of {pair} trades from Kraken '
          f'(this will take a while -- see estimate above)...')
    raw_trades = pull_recent_trades_kraken(
        pair, args.hours, max_calls=max_calls,
    )
    print(f'  {len(raw_trades)} raw trades pulled')

    out_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    raw_trades.to_parquet(out_path)

    span_hours = (
        raw_trades['Timestamp'].max() - raw_trades['Timestamp'].min()
    ) / 1_000_000 / 3600.0
    actual_rate = len(raw_trades) / span_hours if span_hours > 0 else float('nan')

    print(f'\nSnapshot frozen to {snapshot_dir}')
    print(f'  Actual span: {span_hours:.2f}h, actual rate: '
          f'{actual_rate:.1f} trades/hour')
    print('\nThis raw_trades.parquet can be fed through rebuild.py\'s '
          'build_bars_and_labels() exactly like a Binance.US pull -- '
          'same schema, zero downstream code changes needed.')


if __name__ == '__main__':
    main()
