"""
pipeline/diagnostics/capture_binance_snapshot.py

Needed for step 2 of the trending-regime-PBO investigation (2026-09-10):
window 1 (Kraken, Aug~9-Sep~8) showed a persistent low-PBO anomaly that
survived trend-adjustment; window 2 (Kraken, ~30 days earlier) showed
none, but window 2 also had essentially zero trend (log-close R^2=0.0018
vs window 1's 0.63) -- so it doesn't cleanly distinguish "window 1 was a
fluke" from "the model has real but trend-conditional skill". A flat
Binance.US window would have the same problem. This snapshot needs to
cover a window that ALSO trended, so the exchange (Kraken vs Binance.US)
is the only thing that differs from window 1's real test.

*** DESIGN NOTE: exact calendar match isn't possible, and isn't needed ***
Binance's ingestion.py.pull_recent_trades() only supports "last N hours
from right now" (paginates backward from the most recent trade) -- unlike
Kraken's pull_recent_trades_kraken(), it has no end_time_unix parameter
for an arbitrary historical window (see ingestion.py's real signature).
So this cannot be pointed at window 1's exact Aug~9-Sep~8 dates. It
doesn't need to be: BTC/USD is heavily arbitraged across major exchanges,
so a pull taken now (Sep 10) covering the last ~720h will overlap
window 1's period by ~28 of its 30 days and should show close to the
same real ~21% trend on Binance.US data, independent of Kraken. Check
the trend with diagnose_window_trend_bias.py on this snapshot before
concluding anything -- if it did NOT trend, this snapshot answers a
different question than intended and a wider --hours pull (or checking
for the existing ~90-day/2160h Binance.US pull mentioned in
capture_kraken_snapshot.py's own docstring, 343,038 trades, if it's
still on disk) is the better path.

Requires a free, read-only Binance.US API key (Binance.US -> API
Management) -- read from the BINANCE_API_KEY environment variable, never
a CLI argument, so it never lands in shell history or a captured log.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    set BINANCE_API_KEY=your_read_only_key_here
    python pipeline\\diagnostics\\capture_binance_snapshot.py
    python pipeline\\diagnostics\\capture_binance_snapshot.py --hours 720
"""
import argparse
import os
import sys
from datetime import date

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion import pull_recent_trades   # noqa: E402

DEFAULT_SYMBOL = 'BTCUSDT'   # BTC/TUSD is not listed on Binance.US -- see
                              # ingestion.py's own LOAD-BEARING note
# ~158.8/hour, backed out from capture_kraken_snapshot.py's own docstring
# mention of a prior real 90-day/2160h Binance.US pull (343,038 trades) --
# NOT independently re-measured here, only used for the pre-pull time
# estimate below (an inaccurate estimate costs nothing beyond a less
# precise printed guess, same caveat as the Kraken capture script).
KNOWN_RATE_PER_HOUR = {'BTCUSDT': 158.8}
FALLBACK_RATE_PER_HOUR = 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hours', type=float, default=720.0,
        help='Lookback window from right now (see module docstring for '
             'why this cannot target window 1\'s exact historical dates). '
             'Default 720h (30 days) to overlap window 1\'s period as '
             'closely as this API allows.',
    )
    parser.add_argument(
        '--symbol', type=str, default=DEFAULT_SYMBOL,
        help=f'Binance.US symbol, default {DEFAULT_SYMBOL}.',
    )
    args = parser.parse_args()
    symbol = args.symbol

    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY environment variable not set. Get a free '
            'read-only key at binance.us -> API Management, then:\n'
            '  set BINANCE_API_KEY=your_key_here\n'
            'and re-run this script (same session).'
        )

    rate_per_hour = KNOWN_RATE_PER_HOUR.get(symbol, FALLBACK_RATE_PER_HOUR)
    if symbol not in KNOWN_RATE_PER_HOUR:
        print(f'NOTE: no real-machine-confirmed rate on file for {symbol!r} -- '
              f'using a conservative fallback estimate ({FALLBACK_RATE_PER_HOUR}'
              f'/hour) for the pre-pull time estimate below.')

    est_trades = rate_per_hour * args.hours
    est_calls = int(est_trades / 1000) + 10
    max_calls = max(est_calls * 2, 600)
    est_minutes = (est_calls * 0.25) / 60.0  # sleep_seconds=0.25 default

    print('=' * 70)
    print(f'CAPTURING Binance.US snapshot: symbol={symbol}, hours={args.hours}')
    print('=' * 70)
    print(f'  Estimated trades: ~{est_trades:,.0f}')
    print(f'  Estimated calls: ~{est_calls:,} (max_calls set to {max_calls:,})')
    print(f'  Estimated MINIMUM runtime: ~{est_minutes:.0f} minutes')
    print('\n  IMPORTANT: this covers the last --hours from RIGHT NOW, not '
          "window 1's exact historical dates -- see module docstring. "
          'Run diagnose_window_trend_bias.py on the result FIRST to '
          'confirm it actually trended before treating this as a fair '
          "comparison against window 1's finding.")

    confirm = input('\nProceed? [y/N] ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        return

    snapshot_dir = os.path.join(
        HERE, f'binance_snapshot_{symbol.lower()}_{int(args.hours)}h_{date.today().isoformat()}'
    )
    if os.path.exists(snapshot_dir):
        raise SystemExit(
            f'{snapshot_dir} already exists -- refusing to overwrite. '
            'Delete it manually first if you really want a fresh capture.'
        )
    os.makedirs(snapshot_dir)

    print(f'\nPulling {args.hours}h of {symbol} trades from Binance.US '
          f'(this will take a while -- see estimate above)...')
    raw_trades = pull_recent_trades(
        symbol, args.hours, api_key, max_calls=max_calls,
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
    print('\nSame raw_trades.parquet schema as the Kraken snapshots -- '
          'diagnose_window_trend_bias.py / calibrate_pbo_detrended.py / '
          'calibrate_cpcv_groups.py all work on this unchanged. Check the '
          'trend FIRST before drawing any exchange comparison from it.')


if __name__ == '__main__':
    main()
