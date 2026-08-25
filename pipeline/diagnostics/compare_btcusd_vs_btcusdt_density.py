"""
pipeline/diagnostics/compare_btcusd_vs_btcusdt_density.py

Ethan's question (2026-08-25 session, continued): ingestion.py's own
2026-08-13 LOAD-BEARING note documents that BTCUSDT was deliberately
chosen over BTCUSD -- not for density reasons, but to stay consistent
with the static March baseline's stablecoin-quoted character (USDT
mirrors TUSD; BTCUSD's fiat quote "would change the pair's underlying
price/volume character more than swapping one stablecoin peg for
another does"). That same note also LIVE-CONFIRMED BTCUSDT trades thin
(~194 trades/hour over a 24h pull on 2026-08-13) -- exactly the
constraint today's lookback-extension work has been fighting.

This script does NOT decide to switch pairs. It's a cheap, short (24h),
side-by-side density pull on BOTH symbols, so that decision -- if it's
made at all -- is made on real comparative numbers rather than a guess.
Reuses ingestion.pull_recent_trades() directly, unmodified -- symbol is
already a real, generic parameter, no code changes needed anywhere else
in the pipeline to test this.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\compare_btcusd_vs_btcusdt_density.py
    python pipeline\\diagnostics\\compare_btcusd_vs_btcusdt_density.py --hours 48
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion import pull_recent_trades           # noqa: E402

SYMBOLS = ['BTCUSDT', 'BTCUSD']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=24.0,
                         help='Short lookback window for a quick density '
                              'comparison -- deliberately small, this is '
                              'a scouting pull, not a real snapshot.')
    args = parser.parse_args()

    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    print('=' * 66)
    print(f'DENSITY COMPARISON: last {args.hours}h, BTCUSDT vs. BTCUSD')
    print('=' * 66)

    results = {}
    for symbol in SYMBOLS:
        print(f'\nPulling {symbol}...')
        try:
            raw_trades = pull_recent_trades(symbol, args.hours, api_key)
            n = len(raw_trades)
            rate = n / args.hours
            price_min = raw_trades['Price'].min()
            price_max = raw_trades['Price'].max()
            results[symbol] = {
                'n_trades': n, 'trades_per_hour': rate,
                'price_min': price_min, 'price_max': price_max,
            }
            print(f'  {n} trades ({rate:.1f} trades/hour), '
                  f'price range ${price_min:,.2f}-${price_max:,.2f}')
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            results[symbol] = None

    print('\n' + '=' * 66)
    print('SUMMARY')
    print('=' * 66)
    for symbol, r in results.items():
        if r is None:
            print(f'  {symbol}: FAILED (see above)')
        else:
            print(f'  {symbol}: {r["trades_per_hour"]:.1f} trades/hour')

    if results.get('BTCUSDT') and results.get('BTCUSD'):
        ratio = results['BTCUSD']['trades_per_hour'] / results['BTCUSDT']['trades_per_hour']
        print(f'\n  BTCUSD is {ratio:.2f}x the density of BTCUSDT on this pull.')
        print("""
INTERPRETATION:
  - If BTCUSD is meaningfully denser (ratio well above 1), it's a real
    candidate worth a full design discussion -- remember this trades off
    against the 2026-08-13 rationale for choosing USDT (stablecoin-quote
    consistency with the static baseline), not a free upgrade.
  - If the ratio is close to 1 or BTCUSD is thinner, this settles the
    question cheaply -- no reason to reopen the 2026-08-13 decision.
  - Either way, this is ONE short 24h scouting pull, not a calibrated
    comparison -- confirm with a second pull at a different time of day
    before treating the ratio as stable, same caution as every other
    single-pull measurement in this project's diagnostics.
""")


if __name__ == '__main__':
    main()
