"""
pipeline/orchestration/ingestion.py

Pulls recent raw trades for a symbol from Binance.US's public REST API, in
the EXACT schema this project's existing raw-trade CSV already uses
(TradeID/Price/Volume/QuoteVolume/Timestamp/IsBuyerMaker/IsBestMatch) --
Binance's own /api/v3/historicalTrades response uses these same fields
(id/price/qty/quoteQty/time/isBuyerMaker/isBestMatch), so this is a direct
field rename, not a reinterpretation.

*** LOAD-BEARING (2026-08-13): api.binance.us, not api.binance.com ***
binance.com account creation is unavailable for this project's operator
(US residency routes to Binance.US, a separate registered entity/API).
Binance.US mirrors binance.com's REST surface 1:1 for this endpoint --
same path, same response schema, same X-MBX-APIKEY-only auth -- so only
the base URL changes here, not the request/response handling below.
LIVE-CONFIRMED 2026-08-13: api.binance.com returns 451 (unavailable for
legal reasons) from this operator's location; api.binance.us works.

*** LOAD-BEARING (2026-08-13): BTCUSDT, not BTCTUSD ***
BTC/TUSD (this project's static-data baseline pair) is not listed on
Binance.US. Of the available alternatives, BTCUSDT was chosen over
BTCUSD because USDT is a stablecoin quote asset like TUSD was -- BTCUSD
(a fiat quote) would change the pair's underlying price/volume character
more than swapping one stablecoin peg for another does. This does NOT
mean the static baseline's exact calibration (dollar-bar $10k threshold,
CUSUM h=500, etc.) transfers unchanged -- Phase 2a's dynamic-threshold
rebuild exists specifically because a live pull's price/volume level
will differ from the static BTC/TUSD data's. Re-validate downstream
event/bar counts against a live BTCUSDT pull before trusting them.
LIVE-CONFIRMED 2026-08-13: BTCUSDT on Binance.US trades far thinner than
the static baseline (~194 trades/hour observed over a 24h pull, vs. the
static dataset's much higher density) -- expect longer lookback_hours to
be needed for a comparable bar count.

*** LOAD-BEARING (2026-08-13): millisecond-timestamp collision handling ***
Binance's raw trade `time` field is millisecond-resolution, not
microsecond. On a real 24h live pull, 343 of 4,661 trades (7.4%) shared
an identical millisecond timestamp with at least one other trade (likely
bursty bot/market-maker activity at this venue's lower liquidity) --
confirmed via a real ValueError downstream in rebuild.py's
build_bars_and_labels() (`cannot reindex a non-unique index`), traced to
duplicate Date values in Ch02's dollar_bars() output when two bars
completed on colliding timestamps. Fixed here, not in Ch02/03, because
this is an artifact of Binance's timestamp resolution meeting this
venue's trade-arrival pattern, not a Ch02/03 bug -- those chapters were
never exercised against timestamp collisions on the static dataset.
Within each colliding millisecond group, trades are given consecutive
microsecond offsets (0, 1, 2, ...) in TradeID order (Binance guarantees
TradeID is sequential/unique, so it's a real ordering signal even when
exact sub-millisecond timing isn't resolvable). This makes Timestamp
strictly increasing and unique WITHOUT fabricating real sub-millisecond
precision -- the added microseconds are a synthetic tiebreaker for
ordering only, not a measured timing signal. Rejected alternative:
dropping duplicate-timestamp trades instead -- discards real trade data
(7.4% of the pull) just to dodge the crash; disambiguating is strictly
better.

*** NOT YET LIVE-TESTED ***
This environment's network allowlist does not include api.binance.us, so
this module has been reviewed against Binance's public API documentation
but has NOT been run against the real endpoint from within this sandbox.
It HAS been live-tested by the project operator on their own machine (see
LIVE-CONFIRMED notes above) -- see pipeline/README.md's Phase 2 section
for the full status.

Why historicalTrades (not the simpler /api/v3/trades):
/api/v3/trades only returns the most recent <=1000 trades with no time-
range control -- for a liquid pair like BTC/USDT, 1000 trades may cover
well under an hour, nowhere near "pull the last N hours". /api/v3/
historicalTrades supports paging backward via fromId across an arbitrary
time range, at the cost of requiring a free, read-only Binance API key
(X-MBX-APIKEY header only -- no secret, no signature, no trading
permission needed).
"""
import time

import pandas as pd
import requests

BINANCE_BASE_URL = 'https://api.binance.us'
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


def _disambiguate_timestamps(df):
    """Make Timestamp strictly increasing and unique when multiple trades
    share the same millisecond (see module-level LOAD-BEARING note on
    millisecond-timestamp collisions). Within each group of trades sharing
    one millisecond-derived microsecond value, assigns consecutive +0, +1,
    +2, ... microsecond offsets in TradeID order. This is a synthetic
    ordering tiebreaker, NOT a claim of real sub-millisecond measurement
    precision -- callers needing genuine sub-millisecond timing should not
    rely on this column for that purpose.

    Must be called BEFORE the final ascending sort/dedup in
    pull_recent_trades, since it relies on grouping by the raw (pre-nudge)
    Timestamp value.
    """
    df = df.sort_values(['Timestamp', 'TradeID']).reset_index(drop=True)
    offsets = df.groupby('Timestamp').cumcount()
    df['Timestamp'] = df['Timestamp'] + offsets.astype('int64')
    return df


def pull_recent_trades(symbol, lookback_hours, api_key, limit_per_call=1000,
                        max_calls=500, sleep_seconds=0.25, session=None):
    """
    Pull raw trades for `symbol` covering approximately the last
    `lookback_hours` hours, paging backward from the most recent trade via
    Binance's /api/v3/historicalTrades.

    Parameters
    ----------
    symbol : str, e.g. 'BTCUSDT' (BTC/TUSD is not listed on Binance.US --
        see module-level LOAD-BEARING note)
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
            'secret is needed). Get a free read-only key at binance.us '
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

    df = df.drop_duplicates(subset='TradeID')
    df = _disambiguate_timestamps(df)
    df = df[df['Timestamp'] >= cutoff_ms * 1000].reset_index(drop=True)
    return df
