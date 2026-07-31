import os
import sys
import pandas as pd
import numpy as np

# Hybrid path convention (per CLAUDE.md): .py scripts derive AFML_ROOT from
# __file__. This file lives at ch16/data_loader/, so AFML_ROOT is two hops up.
AFML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# roll.py (Snippet 2.2/2.3) lives in ch02/multi_product/ -- reused directly
# rather than duplicated, since it's already real-machine-confirmed there.
sys.path.insert(0, os.path.join(AFML_ROOT, 'ch02'))
from multi_product.roll import roll_gaps, get_rolled_series, non_negative_rolled_prices

DATA_DIR = os.path.join(AFML_ROOT, 'input_data')

def load_contract(path, rescale_new_format_by=None):
    """Dual-format loader (header/no-header). Forces Date to load as
    string first to avoid pandas stripping leading zeros from YYMMDD
    dates in January (e.g. '000104' -> 104), which corrupts any contract
    trading through a January -- confirmed to hit multiple files across
    every one of these six commodities.

    rescale_new_format_by: if set, multiplies OHLC by this factor for
    files in the newer (header/quoted) format. ONLY needed for British
    Pound: confirmed every BP contract from 2000 onward quotes price in
    plain decimal USD/GBP (~1.4-1.6), while every contract through 1999
    quotes in "points" (price x100, ~120-235) -- a genuine unit change by
    the data provider at the same point their export format changed, not
    present in the other five commodities (verified: gold/crude
    oil/corn/live hogs/tbonds show no such break -- their old/new format
    price-level ratios are consistent with ordinary market moves, not a
    100x scale artifact). Without this correction, a single day where the
    dominant old-format contract has a data gap can cause the front-month
    selector to briefly pick a new-format contract, producing a fake
    ~100x roll gap (confirmed: this produced a spurious +9796% single-day
    'return' in GBP before this fix)."""
    first = pd.read_csv(path, nrows=1, header=None)
    has_header = not str(first.iloc[0, 0]).isdigit()
    if has_header:
        df = pd.read_csv(path, dtype={0: str})
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
        df['Date'] = pd.to_datetime(df['Date'])
        if rescale_new_format_by is not None:
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] *= rescale_new_format_by
    else:
        df = pd.read_csv(path, header=None, dtype={0: str},
                          names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
        df['Date'] = pd.to_datetime(df['Date'], format='%y%m%d')
    return df


def build_front_month_series(folder, prefix, start='1996-01-01', end='2002-10-01',
                              rescale_new_format_by=None):
    """Load every contract file for one commodity, tag each row with its
    contract, and for each date keep only the row from whichever
    currently-listed contract has the HIGHEST open interest (the
    conventional definition of "front month" -- the most liquid/most
    heavily-traded contract at that moment, not necessarily the
    nearest-to-expiry one)."""
    files = [f for f in os.listdir(folder) if f.upper().startswith(prefix)]
    pieces = []
    for f in files:
        df = load_contract(os.path.join(folder, f), rescale_new_format_by=rescale_new_format_by)
        df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
        if len(df) == 0:
            continue
        df['Instrument'] = f.rsplit('.', 1)[0]
        pieces.append(df)
    # Forward-fill short (<=3 business day) gaps WITHIN each contract's own
    # date range before comparing open interest across contracts.
    #
    # Root cause this fixes: individual contract files sometimes have a
    # single isolated missing row (a data-export gap, not a real trading
    # halt or the contract actually ceasing to trade) even while that
    # contract remains the genuinely dominant one on the surrounding days.
    # A same-day comparison only ever considers contracts that HAVE a row
    # that date -- so the true front-month contract's one missing day
    # causes selection to briefly and spuriously flip to a much less
    # liquid (and often differently priced) contract for exactly that one
    # day, then flip back -- producing TWO fake roll gaps bracketing one
    # bad day (confirmed: this produced a spurious +20% single-day
    # T-bonds "return" before this fix). Forward-filling within each
    # contract's own observed range (not extrapolating before its first
    # or after its last real row) closes these gaps at the source.
    filled_pieces = []
    for instrument, df in all_contracts_by_instrument(pieces):
        df = df.set_index('Date').sort_index()
        full_range = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(full_range).ffill(limit=3)
        df = df.dropna(subset=['Close'])          # drop any gap too long to fill
        df['Instrument'] = instrument
        df.index.name = 'Date'
        filled_pieces.append(df.reset_index())

    all_contracts = pd.concat(filled_pieces, ignore_index=True)

    # For each date, keep the row from whichever contract has the highest
    # open interest (front month = most liquid contract at that moment).
    all_contracts = all_contracts.sort_values(['Date', 'OpenInt'])
    front_month = all_contracts.drop_duplicates(subset='Date', keep='last')
    front_month = front_month.sort_values('Date').reset_index(drop=True)
    front_month = front_month.set_index('Date')
    return front_month


def all_contracts_by_instrument(pieces):
    """Yield (instrument_name, df) for each per-contract DataFrame in
    `pieces`, reading the name back out of the 'Instrument' column each
    piece was already tagged with."""
    for df in pieces:
        yield df['Instrument'].iloc[0], df


def build_continuous_price(folder, prefix, start='1996-01-01', end='2002-10-01',
                            rescale_new_format_by=None):
    """Stitch front-month contracts into one continuous, roll-gap-corrected
    $1-indexed price series, reusing ch02's roll.py (Snippet 2.2/2.3)."""
    front_month = build_front_month_series(folder, prefix, start, end,
                                            rescale_new_format_by=rescale_new_format_by)
    dictio = {'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}
    non_neg = non_negative_rolled_prices(front_month, dictio=dictio, match_end=True)
    return non_neg, front_month


# rescale_new_format_by=100 is ONLY needed for gbp (see load_contract's
# docstring) -- confirmed the other five commodities have no such break.
COMMODITIES = {
    'gold': ('GC', DATA_DIR + '/gold', None),
    'crude_oil': ('CL', DATA_DIR + '/crude oil', None),
    'corn': ('C', DATA_DIR + '/corn', None),
    'live_hogs': ('LH', DATA_DIR + '/live hogs', None),
    'tbonds': ('US', DATA_DIR + '/US-T bonds', None),
    'gbp': ('BP', DATA_DIR + '/British Pound', 100),
}

if __name__ == '__main__':
    results = {}
    for name, (prefix, folder, rescale) in COMMODITIES.items():
        non_neg, front_month = build_continuous_price(folder, prefix, rescale_new_format_by=rescale)
        n_rolls = front_month['Instrument'].nunique()
        print(f"{name:12s}: {len(non_neg):5d} bars, {n_rolls:3d} contracts used, "
              f"date range {non_neg.index.min().date()} to {non_neg.index.max().date()}")
        results[name] = non_neg
