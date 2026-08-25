"""
pipeline/diagnostics/compare_exchange_density.py

Follow-on to today's BTCUSD-vs-BTCUSDT check (settled: BTCUSDT denser,
no reason to switch pairs within Binance.US). Separate question: is
Binance.US ITSELF the constraint? Its own daily spot volume (~$20-25M)
is roughly three orders of magnitude below other US-compliant venues
(Coinbase ~tens of billions/day, Kraken similarly large) -- flagged as
an open item in CALIBRATION_AUDIT.md's "OFI Null Confirmed Real..."
section. This script is a cheap, short scouting check on two
alternatives, BEFORE any decision to actually switch data sources
(which would be a real design decision -- see that section).

NOT integrated with ingestion.py / rebuild.py -- deliberately
standalone. Each exchange has its own trade schema and its own public
API; this script normalizes just enough (trade count + time span) to
answer "is this meaningfully denser," not to become a drop-in
replacement for pull_recent_trades().

Both exchanges' public trade endpoints are used AS DOCUMENTED as of
2026-08-25 (checked via web search this session, since exchange APIs
change and older references online are frequently stale -- in
particular, Coinbase's old "Coinbase Pro" / plain "Exchange" REST API
is being sunset in favor of "Advanced Trade"; this script uses the
current Advanced Trade public endpoint, not the deprecated one):

  Coinbase (Advanced Trade, public, no auth required for this specific
  endpoint despite the API generally requiring auth elsewhere):
    GET https://api.coinbase.com/api/v3/brokerage/market/products/
        {product_id}/ticker?limit=N
    Returns the most recent N trades (max observed in docs: unclear
    upper bound, tries 1000 first, falls back to 500 then 100 on a
    4xx in case the limit is capped lower than expected).

  Kraken (public, no auth required):
    GET https://api.kraken.com/0/public/Trades?pair=XBTUSD
    Returns roughly the last 1000 trades by default when no `since` is
    given.

Neither call requires an API key -- unlike Binance.US, no
BINANCE_API_KEY-equivalent setup needed to run this.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\compare_exchange_density.py
"""
import sys

import pandas as pd
import requests

COINBASE_URL_TEMPLATE = (
    'https://api.coinbase.com/api/v3/brokerage/market/products/'
    '{product_id}/ticker'
)
COINBASE_PRODUCT_ID = 'BTC-USD'
COINBASE_LIMIT_FALLBACKS = [1000, 500, 100]

KRAKEN_URL = 'https://api.kraken.com/0/public/Trades'
KRAKEN_PAIR = 'XBTUSD'


def fetch_coinbase():
    last_error = None
    for limit in COINBASE_LIMIT_FALLBACKS:
        url = COINBASE_URL_TEMPLATE.format(product_id=COINBASE_PRODUCT_ID)
        try:
            resp = requests.get(url, params={'limit': limit}, timeout=10)
        except requests.RequestException as e:
            last_error = f'request failed: {e}'
            continue
        if resp.status_code != 200:
            last_error = (f'HTTP {resp.status_code} at limit={limit}: '
                           f'{resp.text[:300]}')
            continue
        data = resp.json()
        trades = data.get('trades', [])
        if not trades:
            last_error = f'empty trades list at limit={limit}: {data}'
            continue
        times = pd.to_datetime([t['time'] for t in trades], utc=True)
        return {
            'n_trades': len(trades),
            'span_start': times.min(),
            'span_end': times.max(),
            'limit_used': limit,
        }
    raise RuntimeError(f'Coinbase fetch failed at all fallback limits. '
                        f'Last error: {last_error}')


def fetch_kraken():
    try:
        resp = requests.get(KRAKEN_URL, params={'pair': KRAKEN_PAIR}, timeout=10)
    except requests.RequestException as e:
        raise RuntimeError(f'Kraken request failed: {e}')
    if resp.status_code != 200:
        raise RuntimeError(f'Kraken HTTP {resp.status_code}: {resp.text[:300]}')
    data = resp.json()
    if data.get('error'):
        raise RuntimeError(f'Kraken API returned errors: {data["error"]}')
    result = data.get('result', {})
    pair_keys = [k for k in result.keys() if k != 'last']
    if not pair_keys:
        raise RuntimeError(f'No pair data in Kraken response: {data}')
    trades = result[pair_keys[0]]
    if not trades:
        raise RuntimeError(f'Empty trades list from Kraken: {data}')
    # third element of each trade array is a UNIX timestamp in seconds
    # (float) -- see support.kraken.com's public Trades examples
    times = pd.to_datetime([t[2] for t in trades], unit='s', utc=True)
    return {
        'n_trades': len(trades),
        'span_start': times.min(),
        'span_end': times.max(),
        'pair_key_used': pair_keys[0],
    }


def summarize(name, result):
    if result is None:
        print(f'  {name}: FAILED (see error above)')
        return None
    span_hours = (result['span_end'] - result['span_start']).total_seconds() / 3600.0
    if span_hours <= 0:
        print(f'  {name}: {result["n_trades"]} trades, span too short to '
              f'compute a rate (all trades within the same second)')
        return None
    rate = result['n_trades'] / span_hours
    print(f'  {name}: {result["n_trades"]} trades over {span_hours:.3f}h '
          f'-> {rate:.1f} trades/hour')
    return rate


def main():
    print('=' * 66)
    print('EXCHANGE DENSITY SCOUTING: Coinbase vs. Kraken (BTC/USD)')
    print('=' * 66)
    print('\nFor reference, from today\'s earlier check on this same venue:')
    print('  Binance.US BTCUSDT: 512.2 trades/hour (24h pull)')
    print('  Binance.US BTCUSD:  365.2 trades/hour (24h pull)')

    print('\nFetching Coinbase (BTC-USD, Advanced Trade public API)...')
    try:
        cb_result = fetch_coinbase()
        print(f"  got {cb_result['n_trades']} trades "
              f"(limit={cb_result['limit_used']}), "
              f"span {cb_result['span_start']} to {cb_result['span_end']}")
    except RuntimeError as e:
        print(f'  FAILED: {e}')
        cb_result = None

    print('\nFetching Kraken (XBTUSD, public Trades API)...')
    try:
        kr_result = fetch_kraken()
        print(f"  got {kr_result['n_trades']} trades "
              f"(pair key: {kr_result['pair_key_used']}), "
              f"span {kr_result['span_start']} to {kr_result['span_end']}")
    except RuntimeError as e:
        print(f'  FAILED: {e}')
        kr_result = None

    print('\n' + '=' * 66)
    print('SUMMARY (implied rate from however much history the most')
    print('recent N trades happened to span -- NOT a fixed 24h window,')
    print('unlike the Binance.US numbers above -- so treat these as a')
    print('rough scouting signal, not a precise apples-to-apples rate)')
    print('=' * 66)
    summarize('Coinbase BTC-USD', cb_result)
    summarize('Kraken XBTUSD', kr_result)

    print("""
INTERPRETATION:
  - This is a SINGLE scouting call per exchange, not a calibrated
    density measurement -- unlike the Binance.US BTCUSD/BTCUSDT check,
    which pulled a fixed 24h window, these return "however many trades
    happened in whatever time the last N trades cover," which varies
    with how liquid the moment is right now. Treat differences of 2x+
    as meaningful; differences under that as noise worth a second look
    before acting on.
  - If either comes back meaningfully denser than Binance.US's ~365-512
    trades/hour, that's worth a real design conversation (per
    CALIBRATION_AUDIT.md's standing note) -- NOT a reason to switch
    today. A real switch would need: confirming historical-pull
    pagination actually works the way ingestion.py's Binance.US
    integration does (this script only checked the most-recent-N
    snapshot, not deep history), understanding each exchange's own
    schema differences, and deciding whether re-validating this
    project's dollar-bar/CUSUM/triple-barrier calibration on a new
    venue's price/volume character is worth the cost -- same category
    of decision as the 2026-08-13 BTCUSDT-vs-BTCUSD choice itself.
""")


if __name__ == '__main__':
    main()
