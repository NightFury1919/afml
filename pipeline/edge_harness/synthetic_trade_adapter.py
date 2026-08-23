"""
synthetic_trade_adapter.py

generate_synthetic_trades() produces a 4-column DataFrame (Price, Volume,
Timestamp-as-datetime64, IsBuyerMaker) -- a clean, minimal shape for the
generator's own TDD. The REAL pipeline's raw_trades schema (confirmed by
reading ingestion.py directly, 2026-08-22) is 7 columns: TradeID, Price,
Volume, QuoteVolume, Timestamp (int64 MICROSECONDS since epoch -- not
datetime64), IsBuyerMaker, IsBestMatch. This adapter bridges the two.

LOAD-BEARING (2026-08-22): Timestamp conversion. rebuild.py's
preprocess_raw_trades() calls pd.to_datetime(raw['Timestamp'], unit='us')
-- it expects raw int64 microseconds, not datetime64. Converting a
datetime64 column to int64 nanoseconds-since-epoch via .astype('int64')
and then //1000 gives microseconds. Confirmed against ingestion.py's own
real conversion (df['Timestamp'].astype('int64') * 1000 going the OTHER
way, ms->us) -- this adapter performs the inverse direction correctly
(datetime64[ns] -> us), not just an assumed inverse.

LOAD-BEARING (2026-08-22): TradeID assignment. _disambiguate_timestamps()
(reused by rebuild.py's preprocess_raw_trades()) sorts by
['Timestamp', 'TradeID'] and uses TradeID as the within-timestamp tie-
break order. generate_synthetic_trades()'s output is already strictly
time-ordered by construction (cumulative exponential inter-arrivals), so
assigning TradeID = sequential row order (0, 1, 2, ...) exactly preserves
that ordering -- consistent with Binance's own real TradeID semantics
(sequential, unique) that _disambiguate_timestamps() relies on.

IsBestMatch is stubbed to True for every row -- confirmed by reading both
rebuild.py and features.py that neither actually consumes this column
(it's carried in ingestion.py's schema for parity with Binance's raw API
response, not used downstream in this project's pipeline).
"""
import numpy as np
import pandas as pd


def synthetic_to_raw_trades_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert generate_synthetic_trades()'s output (Price, Volume,
    Timestamp[datetime64], IsBuyerMaker) into the exact raw_trades schema
    rebuild.py/features.py expect (TradeID, Price, Volume, QuoteVolume,
    Timestamp[int64 microseconds], IsBuyerMaker, IsBestMatch).

    Parameters
    ----------
    df : pd.DataFrame from generate_synthetic_trades(), NOT already
        containing TradeID/QuoteVolume/IsBestMatch columns.

    Returns
    -------
    pd.DataFrame, columns = ingestion.RAW_TRADE_COLUMNS exactly,
    ready to pass directly to rebuild.build_bars_and_labels() and
    features.build_enriched_events().
    """
    if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
        raise TypeError(
            f"Expected df['Timestamp'] to be datetime64, got "
            f"{df['Timestamp'].dtype}. This adapter is meant to convert "
            "generate_synthetic_trades()'s raw output, which always "
            "returns datetime64 Timestamps -- if this fires, check "
            "whether df has already been converted once (double "
            "conversion would silently corrupt the timestamps)."
        )

    out = pd.DataFrame({
        'TradeID': np.arange(len(df), dtype='int64'),
        'Price': df['Price'].values,
        'Volume': df['Volume'].values,
        'QuoteVolume': (df['Price'] * df['Volume']).values,
        'Timestamp': (df['Timestamp'].values.astype('int64') // 1000),
        'IsBuyerMaker': df['IsBuyerMaker'].values,
        'IsBestMatch': True,
    })
    return out
