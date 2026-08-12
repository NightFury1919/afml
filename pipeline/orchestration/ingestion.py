"""
pipeline/orchestration/ingestion.py

Pulls recent raw trades for a symbol from Binance's public REST API, in the
EXACT schema this project's existing raw-trade CSV already uses
(TradeID/Price/Volume/QuoteVolume/Timestamp/IsBuyerMaker/IsBestMatch) --
Binance's own /api/v3/historicalTrades response uses these same fields
(id/price/qty/quoteQty/time/isBuyerMaker/isBestMatch), so this is a direct
field rename, not a reinterpretation.

*** NOT YET LIVE-TESTED ***
This environment's network allowlist does not include api.binance.com, so
this module has been reviewed against Binance's public API documentation
but has NOT been run against the real endpoint. Test this on your machine
(which has normal internet access) before trusting it -- see
pipeline/README.md's Phase 2 section.

Why historicalTrades (not the simpler /api/v3/trades):
/api/v3/trades only returns the most recent <=1000 trades with no time-
range control -- for a liquid pair like BTC/TUSD, 1000 trades may cover
well under an hour, nowhere near "pull the last N hours". /api/v3/
historicalTrades supports paging backward via fromId across an arbitrary
time range, at the cost of requiring a free, read-only Binance API key
(X-MBX-APIKEY header only -- no secret, no signature, no trading
permission needed).
"""
import time

import pandas as pd
import requests

BINANCE_BASE_URL = 'https://api.binance.com'
RAW_TRADE_COLUMNS = [
    'TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
    'IsBuyerMaker', 'IsBestMatch',
]


def _fetch_page(symbol, api_key, from_id=None, limit=1000, session=None):
    """One page of /api/v3/historicalTrades. Returns a list of raw dicts
    (Binance's own field names: id/price/qty/quoteQty/time/isBuyerMaker/
    isBestMatch) -- NOT yet renamed to this project's schema."""
    sess = session or requests
    params = {'symbol': symbol, 'limit': limit}
    if from_id is not None:
        params['fromId'] = from_id
    headers = {'X-MBX-APIKEY': api_key}
    resp = sess.get(
        f'{BINANCE_BASE_URL}/api/v3/historicalTrades',
        params=params, headers=headers, timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _latest_trade_id(symbol, session=None):
    """/api/v3/trades (no key needed) to find the current latest TradeID,
    the starting point for paging backward via historicalTrades."""
    sess = session or requests
    resp = sess.get(
        f'{BINANCE_BASE_URL}/api/v3/trades',
        params={'symbol': symbol, 'limit': 1}, timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise ValueError(f'No trades returned for symbol {symbol!r}')
    return data[0]['id']


def pull_recent_trades(symbol, lookback_hours, api_key, limit_per_call=1000,
                        max_calls=500, sleep_seconds=0.25, session=None):
    """
    Pull raw trades for `symbol` covering approximately the last
    `lookback_hours` hours, paging backward from the most recent trade via
    Binance's /api/v3/historicalTrades.

    Parameters
    ----------
    symbol : str, e.g. 'BTCTUSD'
    lookback_hours : float
    api_key : str, required (see module docstring -- free, read-only)
    limit_per_call : int, max 1000 per Binance's API limit
    max_calls : int, hard stop to avoid an unbounded pull / rate-limit ban
        on an illiquid pair or an accidentally huge lookback_hours
    sleep_seconds : float, delay between calls to stay well under Binance's
        published rate limits (a real IP ban is disruptive for an entire
        trading club sharing a network -- err conservative)
    session : optional requests.Session, for testing / connection reuse

    Returns
    -------
    pd.DataFrame, columns = RAW_TRADE_COLUMNS, sorted ascending by
    Timestamp -- a drop-in replacement for this project's existing
    header=None, names=RAW_TRADE_COLUMNS static CSV format.

    Raises
    ------
    ValueError if no trades are returned, or the pull hits max_calls
    before covering the requested lookback window (the caller should
    treat this as "increase max_calls or shrink lookback_hours", not
    silently truncate a request for N hours of data).
    """
    if api_key is None:
        raise ValueError(
            'A Binance API key is required (historicalTrades needs the '
            'X-MBX-APIKEY header, even though no trading permission or '
            'secret is needed). Get a free read-only key at binance.com '
            '-> API Management.'
        )

    cutoff_ms = int((time.time() - lookback_hours * 3600) * 1000)

    latest_id = _latest_trade_id(symbol, session=session)
    from_id = max(0, latest_id - limit_per_call + 1)

    pages = []
    for call_num in range(max_calls):
        page = _fetch_page(symbol, api_key, from_id=from_id,
                            limit=limit_per_call, session=session)
        if not page:
            break
        pages.append(page)

        oldest_time_ms = page[0]['time']
        if oldest_time_ms <= cutoff_ms:
            break

        from_id = max(0, from_id - limit_per_call)
        if from_id == 0 and call_num > 0:
            break  # reached the start of this symbol's trade history
        time.sleep(sleep_seconds)
    else:
        raise ValueError(
            f'Hit max_calls={max_calls} before covering '
            f'{lookback_hours} hours of {symbol} history -- increase '
            'max_calls or shrink lookback_hours.'
        )

    raw = [trade for page in pages for trade in page]
    if not raw:
        raise ValueError(f'No trades returned for symbol {symbol!r}')

    df = pd.DataFrame(raw)
    df = df.rename(columns={
        'id': 'TradeID', 'price': 'Price', 'qty': 'Volume',
        'quoteQty': 'QuoteVolume', 'time': 'Timestamp',
        'isBuyerMaker': 'IsBuyerMaker', 'isBestMatch': 'IsBestMatch',
    })[RAW_TRADE_COLUMNS]

    df['Price'] = df['Price'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['QuoteVolume'] = df['QuoteVolume'].astype(float)
    df['Timestamp'] = df['Timestamp'].astype('int64') * 1000  # ms -> us,
        # matching this project's existing raw CSV, which is in microseconds
        # (pd.to_datetime(..., unit='us') is used throughout the repo)

    df = df.drop_duplicates(subset='TradeID').sort_values('Timestamp')
    df = df[df['Timestamp'] >= cutoff_ms * 1000].reset_index(drop=True)
    return df
