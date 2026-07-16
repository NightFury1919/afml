"""
ch19/build_enriched_training_table.py

Joins Ch19's 249-bar microstructural feature table onto the real 88-event
Ch03-05 training data, producing an ENRICHED training table with 12
features (fracdiff + Ch19's 11) instead of 1.

Why a new artifact instead of overwriting ch07_training_table.{csv,pkl}
------------------------------------------------------------------------
Ch08's synthetic-only design and any other consumer that expects the
original single-feature table shouldn't silently change underneath them.
This writes a SEPARATE artifact -- ch07_training_table_enriched.{csv,pkl}
-- and Ch09/Ch12 are updated to load that instead. The original
ch07_training_table.{csv,pkl} is untouched.

Alignment
---------
Ch19's feature table is indexed by bar_id (0-248). Ch03's 88 events are
indexed by timestamp. Both timestamps are dollar-bar CLOSE times built
with the identical $10,000 threshold (verified: all 88 event timestamps
match a real bar-close timestamp exactly -- see project chat, no fuzzy/
nearest-match logic needed). This script rebuilds that same bar_id<->Date
mapping from the raw trade tape (same logic as chapter_19's Part A) to
join on.

The one event whose bar (bar_id=15) is still inside Ch19's 20-bar rolling-
window warmup period (roll_c/roll_sigma_u/parkinson_vol_20bar/
amihud_lambda_20bar all undefined there) is DROPPED, matching the same
warmup-drop convention Ch05 already uses for fracdiff's own FFD window
(that's why ch05_features.csv has 239 rows instead of 249) -- not a new
policy, just applying the existing one.

Path convention: this .py derives its root via __file__.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
INPUT_DATA = os.path.join(ROOT, 'input_data')

import numpy as np
import pandas as pd

DOLLAR_BAR_THRESHOLD = 10000.0


def build_bar_id_to_date():
    """Rebuild the bar_id -> bar-close-Date mapping from raw trades (same
    $10,000-threshold logic as chapter_19's Part A)."""
    trades_path = os.path.join(INPUT_DATA, 'BTCTUSD-trades-2026-03.csv')
    raw = pd.read_csv(trades_path, header=None,
                       names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                              'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])
    raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')

    cumm_dollar, bar_id, bar_ids = 0.0, 0, []
    for price, volume in zip(raw['Price'], raw['Volume']):
        cumm_dollar += price * volume
        bar_ids.append(bar_id)
        if cumm_dollar >= DOLLAR_BAR_THRESHOLD:
            bar_id += 1
            cumm_dollar = 0.0
    raw['bar_id'] = bar_ids

    n_complete_bars = raw['bar_id'].max()
    raw = raw[raw['bar_id'] < n_complete_bars]
    return raw.groupby('bar_id')['Date'].last()


def main():
    print("Rebuilding bar_id <-> Date mapping from raw trades...")
    bar_close_date = build_bar_id_to_date()
    date_to_bar_id = pd.Series(bar_close_date.index, index=bar_close_date.values)

    ch03 = pd.read_csv(os.path.join(INPUT_DATA, 'ch03_events.csv'), index_col=0, parse_dates=[0, 1])
    ch04 = pd.read_csv(os.path.join(INPUT_DATA, 'ch04_weights.csv'), index_col=0, parse_dates=[0])
    ch05 = pd.read_csv(os.path.join(INPUT_DATA, 'ch05_features.csv'), index_col=0, parse_dates=[0])
    ch19 = pd.read_csv(os.path.join(INPUT_DATA, 'ch19_microstructural_features.csv'), index_col=0)

    assert ch03.index.isin(bar_close_date.values).all(), \
        'every Ch03 event must land exactly on a real bar-close timestamp'

    event_bar_ids = ch03.index.map(date_to_bar_id)
    ch19_at_events = ch19.loc[event_bar_ids].copy()
    ch19_at_events.index = ch03.index

    fracdiff = ch05.loc[ch03.index][['fracdiff']]

    table = pd.concat([fracdiff, ch19_at_events], axis=1)
    table['bin'] = ch03['bin']
    table['w'] = ch04['w']
    table['t1'] = ch03['t1']

    n_before = len(table)
    incomplete_rows = table[table.drop(columns=['bin', 'w', 't1']).isna().any(axis=1)]
    if len(incomplete_rows):
        print(f"Dropping {len(incomplete_rows)} event(s) still inside a rolling-window "
              f"warmup period (same convention Ch05 already uses for fracdiff's FFD "
              f"warmup):")
        print(incomplete_rows.index.tolist())
    table = table.dropna()
    print(f"Enriched training table: {n_before} -> {len(table)} events, "
          f"{table.shape[1] - 3} features (fracdiff + {ch19.shape[1]} Ch19 features)")

    csv_out = os.path.join(INPUT_DATA, 'ch07_training_table_enriched.csv')
    pkl_out = os.path.join(INPUT_DATA, 'ch07_training_table_enriched.pkl')
    table.to_csv(csv_out)
    table.to_pickle(pkl_out)
    print(f"Saved: {csv_out}")
    print(f"Saved: {pkl_out}")
    print(f"\nColumns: {list(table.columns)}")
    return table


if __name__ == '__main__':
    main()
