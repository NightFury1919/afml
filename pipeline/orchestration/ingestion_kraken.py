"""
pipeline/orchestration/ingestion_kraken.py

Kraken counterpart to ingestion.py's pull_recent_trades() -- SEPARATE
module, ingestion.py itself is untouched, so the existing Binance.US
flow (run_pipeline_live.py, every diagnostic sweep that pulls live data)
is completely unaffected by this addition.

Motivation (2026-08-25 session): scouting comparison found Kraken's
public BTC/USD trade feed running meaningfully denser than Binance.US's
BTCUSDT (compare_exchange_density.py's initial scout: 3,676 vs. 512
trades/hour; verify_kraken_pull.py's real, confirmed 24h pull: 4,104.7
vs. 512 trades/hour -- roughly 8x) -- CALIBRATION_AUDIT.md's standing
open item on whether Binance.US itself is the real density constraint,
now with a concrete, real-machine-confirmed candidate.

Returns the SAME schema as ingestion.py's pull_recent_trades()
(RAW_TRADE_COLUMNS: TradeID/Price/Volume/QuoteVolume/Timestamp/
IsBuyerMaker/IsBestMatch), so downstream code (rebuild.py, features.py,
everything built on top of them) needs ZERO changes to accept Kraken
data -- they only ever depend on this schema, never on which exchange
produced it.

*** LOAD-BEARING (2026-08-25): buy/sell -> IsBuyerMaker mapping is an
INFERENCE, not yet real-machine confirmed ***
Kraken's trade tuple's 4th field is <buy/sell>, documented as denoting
which side the trade represents -- the standard convention across
exchange public trade feeds (also true of Binance's own isBuyerMaker,
just phrased from the opposite side) is that this flags the TAKER
(aggressor) side: 's' = sell-initiated (the taker sold, meaning the
BUYER was passively resting as the maker), 'b' = buy-initiated (the
taker bought, meaning the SELLER was the maker). Binance's isBuyerMaker
directly asks "was the buyer the maker" -- so Kraken 's' should
correspond to isBuyerMaker=True, and Kraken 'b' to isBuyerMaker=False.
This has NOT been independently confirmed against Kraken's own trade
history / chart UI -- verify_kraken_pull.py's real-machine output should
be sanity-checked (does the buy/sell split look plausible -- roughly
balanced, not wildly skewed in a way that suggests a sign error) before
trusting any downstream tick-rule-dependent feature computed from this
mapping.

*** LOAD-BEARING (2026-08-25): `since` parameter precision ambiguity,
resolved by NEVER reformatting Kraken's own returned cursor ***
Two official-looking Kraken doc sources disagree on `since`'s expected
precision -- the current OpenAPI reference page's own example shows
plain 10-digit UNIX SECONDS ('1616663618'), while the response schema's
own `last` field example is a 19-digit NANOSECOND value
('1688671969993150842') meant to be used verbatim as the next `since`.
Rather than resolve this by guessing, this module only ever constructs
the FIRST call's `since` itself (as whole seconds, matching the current
official example format) -- every subsequent page uses the exact `last`
string Kraken returned, unmodified. This sidesteps the ambiguity
entirely rather than risking a silent unit-conversion bug.

TradeID: Kraken's current API DOES include a real numeric trade_id as
the tuple's 7th element (confirmed via docs.kraken.com, 2026-08-25 --
older examples/support articles online show only 6 fields without one;
this project uses the current, confirmed 7-field schema) -- used
directly as this schema's TradeID, same dedup role Binance's own
TradeID plays.

RATE LIMITING: Kraken's public market-data endpoints are not subject to
the same tiered counter system its AUTHENTICATED endpoints are (that
system, referenced in Kraken's docs, applies to private trading/ledger
calls) -- but sleep_seconds defaults conservatively anyway, same "an IP
ban is disruptive for a shared network" reasoning ingestion.py's own
Binance.US pull already uses.

*** LOAD-BEARING (2026-08-25): 429 retry-with-backoff added after
real-machine density confirmation ***
24h/2h verification pulls (verify_kraken_pull.py) both completed
without any rate-limit response -- but those were ~25-100 calls each.
Kraken's real observed density here (~4,105 trades/hour, confirmed
real-machine 2026-08-25) means a full 720h pull needs ~2,955 calls
(vs. Binance.US's ~500-600 for a comparable window) -- a MUCH longer
sequential call chain than anything tested so far. Added explicit HTTP
429 detection with backoff-and-retry (not just letting
requests.raise_for_status() throw immediately) since a transient
rate-limit hit partway through a multi-thousand-call pull should not
discard all progress already made in `pages`.

NOT YET WIRED into run_pipeline_live.py or any production flow --
deliberately standalone pending real-machine verification via
verify_kraken_pull.py (small pull first) before this is trusted for a
full-scale 720h+ pull, per this project's real-data-first, verify-
before-trusting convention.
"""
import time

import pandas as pd
import requests

RAW_TRADE_COLUMNS = [
    'TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
    'IsBuyerMaker', 'IsBestMatch',
]

KRAKEN_TRADES_URL = 'https://api.kraken.com/0/public/Trades'


def pull_recent_trades_kraken(pair, lookback_hours, limit_per_call=1000,
                               max_calls=500, sleep_seconds=1.0, session=None,
                               max_429_retries=5, retry_backoff_seconds=5.0):
    """
    Pull raw trades for `pair` covering approximately the last
    `lookback_hours` hours from Kraken's public /0/public/Trades
    endpoint, paging FORWARD from `since` (the OPPOSITE direction from
    ingestion.py's Binance.US pull, which pages backward from the most
    recent trade) until catching up to the present.

    Parameters
    ----------
    pair : str, e.g. 'XBTUSD' (Kraken's alias for BTC/USD; the response
        itself keys results under Kraken's internal name, e.g.
        'XXBTZUSD' -- this function reads whichever key comes back
        rather than assuming the alias echoes back unchanged)
    lookback_hours : float
    limit_per_call : int, max 1000 per Kraken's documented `count` cap
    max_calls : int, hard stop to avoid an unbounded pull. Kraken's
        density here is far higher than Binance.US's -- a 720h pull
        needs ~2,955 calls at limit_per_call=1000, not the ~500-600 a
        comparable Binance.US pull needs. Callers targeting a large
        lookback_hours MUST raise this explicitly (this function's own
        default, 500, is NOT auto-scaled to lookback_hours -- same
        precedent as ingestion.py's Binance.US pull, where the caller
        passes an explicit override rather than the shared default
        silently guessing at an appropriate value).
    sleep_seconds : float, delay between calls
    session : optional requests.Session
    max_429_retries : int, retries on a rate-limit response before
        giving up on that page (progress from all prior pages is kept)
    retry_backoff_seconds : float, sleep duration before retrying a
        429'd call

    Returns
    -------
    pd.DataFrame, columns = RAW_TRADE_COLUMNS, sorted ascending by
    Timestamp -- schema-identical to ingestion.py's
    pull_recent_trades() output.

    Raises
    ------
    ValueError if Kraken returns an API-level error, if no trades are
    returned, or if the pull hits max_calls before catching up to the
    present (same "increase max_calls or shrink lookback_hours"
    guidance as ingestion.py's own equivalent guard).
    """
    session = session or requests.Session()

    since_cursor = str(int(time.time() - lookback_hours * 3600))  # first
        # call only -- whole seconds, matching the current official
        # docs.kraken.com example format (see module LOAD-BEARING note)
    now_buffer_sec = 5.0  # "caught up to present" tolerance

    pages = []
    for call_num in range(max_calls):
        for retry_num in range(max_429_retries + 1):
            resp = session.get(
                KRAKEN_TRADES_URL,
                params={'pair': pair, 'since': since_cursor, 'count': limit_per_call},
                timeout=10,
            )
            if resp.status_code == 429:
                if retry_num == max_429_retries:
                    raise ValueError(
                        f'Kraken rate-limited (HTTP 429) {max_429_retries} '
                        f'times in a row at call {call_num} -- giving up. '
                        f'{len(pages)} pages ({sum(len(p) for p in pages)} '
                        'trades) already collected are lost since this '
                        'function has no partial-result return path -- '
                        'consider raising retry_backoff_seconds or lowering '
                        'the pull scope if this recurs.'
                    )
                time.sleep(retry_backoff_seconds * (retry_num + 1))
                continue
            resp.raise_for_status()
            break
        data = resp.json()
        if data.get('error'):
            raise ValueError(f'Kraken API returned errors: {data["error"]}')

        result = data.get('result', {})
        pair_keys = [k for k in result.keys() if k != 'last']
        if not pair_keys:
            raise ValueError(f'No pair data in Kraken response: {data}')
        page = result[pair_keys[0]]

        if not page:
            break
        pages.append(page)

        newest_trade_time_sec = float(page[-1][2])
        if newest_trade_time_sec >= time.time() - now_buffer_sec:
            break  # caught up to present

        # Reuse Kraken's own cursor VERBATIM -- see module LOAD-BEARING
        # note on why this is never reformatted or reparsed as an int.
        since_cursor = result['last']
        time.sleep(sleep_seconds)
    else:
        raise ValueError(
            f'Hit max_calls={max_calls} before catching up to the '
            f'present for pair {pair!r} -- increase max_calls or shrink '
            'lookback_hours.'
        )

    raw = [trade for page in pages for trade in page]
    if not raw:
        raise ValueError(f'No trades returned for pair {pair!r}')

    df = pd.DataFrame(raw, columns=[
        'Price', 'Volume', 'TimestampSec', 'Side', 'OrderType', 'Misc', 'TradeID',
    ])
    df['Price'] = df['Price'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['QuoteVolume'] = df['Price'] * df['Volume']
    df['Timestamp'] = ((df['TimestampSec'].astype(float) * 1000).round().astype('int64')) * 1000
        # LOAD-BEARING (2026-08-25, found real-machine via the target_bars
        # sweep crashing with the same error class as the 2026-08-21
        # duplicate-timestamp bug): rounds to MILLISECOND precision first
        # (matching Binance's native resolution), THEN multiplies by 1000
        # to reach microseconds -- NOT a direct round(seconds*1e6). A
        # direct microsecond round gives genuine, ungridded values with no
        # guaranteed gap between distinct real timestamps (two real trades
        # can land 1-2 microseconds apart); ingestion.py's
        # _disambiguate_timestamps() (reused via rebuild.py's
        # preprocess_raw_trades(), see that module's own LOAD-BEARING
        # note) adds +0,+1,+2,... offsets to duplicate groups, which is
        # only collision-safe when every DISTINCT timestamp is guaranteed
        # >=1000us from its neighbor -- exactly what multiplying a
        # millisecond-rounded value by 1000 provides (same property
        # Binance's native ms-resolution data already has). The direct-
        # microsecond version caused real ValueErrors ("index must be
        # monotonic increasing", "operands could not be broadcast
        # together") at every target_bars value tested on the 720h
        # snapshot -- offset collisions creating fresh duplicate/
        # out-of-order Timestamp values downstream. Sacrifices genuine
        # sub-millisecond timing precision from Kraken's raw data, which
        # does not matter for this pipeline's dollar-bar aggregation;
        # TradeID (Kraken's real, sequential exchange-assigned ID) is the
        # disambiguation/sort tiebreaker either way, so correct trade
        # ORDER is preserved regardless of this precision reduction.
    df['IsBuyerMaker'] = df['Side'] == 's'  # see module LOAD-BEARING note
    df['IsBestMatch'] = True  # Kraken has no equivalent flag; same stub
        # precedent already used by positive_control_data.py /
        # synthetic_trade_adapter.py for this always-unused-downstream column
    df['TradeID'] = df['TradeID'].astype('int64')

    df = df[RAW_TRADE_COLUMNS]
    df = df.drop_duplicates(subset='TradeID')
    df = df.sort_values('Timestamp').reset_index(drop=True)

    cutoff_us = int((time.time() - lookback_hours * 3600) * 1_000_000)
    df = df[df['Timestamp'] >= cutoff_us].reset_index(drop=True)
    return df
