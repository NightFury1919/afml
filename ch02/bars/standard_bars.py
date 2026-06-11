import numpy as np
import pandas as pd

# Standard Bars — AFML Chapter 2, Section 2.3.1
# All four standard bar types sample at fixed thresholds rather than
# adapting to market activity. They serve as a baseline comparison
# for the information-driven bars in Section 2.3.2.
# VWAP (Volume Weighted Average Price) is computed in all bar types:
#   VWAP = Σ(price * volume) / Σ(volume)
#
# --- What is a "bar"? ---
# In financial data, raw trades arrive thousands of times per second.
# A "bar" is a summary of all trades in a window, storing:
#   Open  — the FIRST price in the window
#   High  — the HIGHEST price in the window
#   Low   — the LOWEST price in the window
#   Close — the LAST price in the window
#   VWAP  — the AVERAGE price, weighted by how much was traded at each level
#
# These are the same OHLC bars you see on any stock chart. The four types
# below differ only in HOW they decide when to close a bar and open a new one.
#
# --- Why not always use time bars? ---
# Time bars (e.g. 1-minute candles) have a serious problem: during a quiet
# overnight period, a 1-minute bar might contain 0 trades; during a news event,
# it might contain 10,000 trades. Both bars look identical in shape but represent
# radically different amounts of economic activity. The other bar types fix this.


def time_bars(df, freq='W'):
    # Time Bars — sample at fixed time intervals.
    # Each bar represents one period (e.g. weekly) regardless of trading activity.
    # The book notes time bars oversample quiet periods and undersample active ones.
    #
    # --- How it works ---
    # pandas resample() groups all rows that fall within the same calendar period
    # (e.g. the same week) and then aggregates them:
    #   'Price'  → take the LAST price in the period (= closing price)
    #   'Volume' → SUM all volume in the period
    #
    # freq examples:  'D' = daily,  'W' = weekly,  'ME' = month-end,  'H' = hourly
    #
    # This is the simplest bar type — and the one most traditional finance
    # software defaults to — but the book argues it is statistically inferior
    # to the volume/dollar bars below.

    df = df.copy()                          # avoid modifying the original dataframe
    df['Date'] = pd.to_datetime(df['Date']) # ensure Date column is a datetime object
    df.set_index('Date', inplace=True)      # resample() requires the datetime to be the index

    bars = df.resample(freq).agg({
        'Price':  'last',   # closing price = last trade price in the window
        'Volume': 'sum'     # total volume traded during the window
    }).dropna()             # drop any empty periods (e.g. weekends with no trades)

    return bars.reset_index()   # move Date back from index to a regular column


def tick_bars(df, thresh):
    # Tick Bars — sample every N trades regardless of price or volume.
    # Addresses the limitation of time bars by sampling based on trading activity.
    # A bar closes when len(collector) >= thresh (i.e. N trades have occurred).
    #
    # --- How it works ---
    # We iterate trade by trade, appending each price to a running list (collector).
    # When the list reaches `thresh` entries, we close the bar, record its OHLC + VWAP,
    # and start a fresh list for the next bar.
    #
    # --- Why is this better than time bars? ---
    # Every tick bar contains exactly the same NUMBER of trades. During active
    # markets, bars close faster (more bars per hour). During quiet markets,
    # bars close slower. This means each bar represents the same "amount of
    # trading activity", not the same amount of calendar time.
    #
    # --- Remaining limitation ---
    # Tick bars ignore HOW BIG each trade is. A single institutional order for
    # 10,000 shares and 10,000 retail orders for 1 share each are treated
    # identically. Volume bars (below) fix this.

    bars = []           # will collect tuples of (date, index, open, low, high, close, vwap)
    collector = []      # accumulates prices within the current bar
    cumm_vol = 0        # running total of volume within current bar
    vol_price = 0       # running total of (price × volume), used to compute VWAP

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        collector.append(price)         # add this trade's price to the current bar
        cumm_vol  += volume             # accumulate volume
        vol_price += price * volume     # accumulate price*volume for VWAP numerator

        if len(collector) >= thresh:    # have we collected enough ticks to close a bar?
            open_p  = collector[0]          # first price in this bar
            high_p  = np.max(collector)     # highest price seen in this bar
            low_p   = np.min(collector)     # lowest price seen in this bar
            close_p = collector[-1]         # last price in this bar

            # VWAP = Σ(price × volume) / Σ(volume)
            # This is the average price, weighted by how much was traded at each level.
            # A trade of 1000 units at $100 matters more to the average than 1 unit at $99.
            vwap = vol_price / cumm_vol

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # Reset everything — start a fresh bar
            collector = []
            cumm_vol  = 0
            vol_price = 0

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)


def volume_bars(df, thresh):
    # Volume Bars — sample every time a fixed amount of the asset is traded.
    # Improves on tick bars by accounting for trade size differences.
    # A bar closes when cumm_vol >= thresh (i.e. N units have been traded).
    #
    # --- How it works ---
    # Instead of counting the NUMBER of trades, we count the TOTAL SIZE of trades.
    # Each trade adds its volume to a running counter. When that counter crosses
    # `thresh`, the bar closes and the counter resets.
    #
    # --- Why this matters ---
    # One massive institutional trade might push through more "real" market
    # information than thousands of tiny retail orders. By sampling every
    # fixed amount of volume, each bar reflects the same "quantity of asset
    # changing hands", regardless of how many individual trades it took.
    #
    # --- Remaining limitation ---
    # Volume bars don't account for changes in PRICE LEVEL over time.
    # If Bitcoin goes from $10,000 to $100,000, the same number of coins
    # represents 10× more economic activity. Dollar bars (below) fix this.

    cumm_vol  = 0       # running total of volume within current bar
    vol_price = 0       # running total of (price × volume) for VWAP
    collector = []      # accumulates prices within the current bar
    bars = []

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        cumm_vol  += volume             # add this trade's volume to the running total
        vol_price += price * volume     # accumulate for VWAP
        collector.append(price)

        if cumm_vol >= thresh:          # has enough volume traded to close a bar?
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol  # VWAP = Σ(p*v) / Σ(v)

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # Reset for the next bar
            cumm_vol  = 0
            vol_price = 0
            collector = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)


def dollar_bars(df, thresh):
    # Dollar Bars — sample every time a fixed dollar value is traded.
    # Most robust to price-level changes and corporate actions.
    # A bar closes when cumm_dollar >= thresh (i.e. $N has been exchanged).
    # The book notes dollar bars are the most useful standard bar type
    # and uses them throughout subsequent chapters.
    #
    # --- How it works ---
    # For each trade, "dollar value" = price × volume (how many dollars changed hands).
    # We accumulate these dollar values. Once the running total crosses `thresh`,
    # the bar closes and we reset.
    #
    # --- Why dollar bars are the best standard bar type ---
    # Consider a stock that splits 2-for-1: overnight, each share is worth half
    # as much and volume doubles. Volume bars would produce twice as many bars
    # after the split with no change in underlying dollar activity. Dollar bars
    # are unaffected — each bar still represents $thresh of real economic activity.
    #
    # Similarly, a stock rising from $10 to $100 means each share represents
    # 10× more money. Dollar bars automatically adjust sampling frequency
    # to remain consistent in dollar terms throughout.
    #
    # --- Example ---
    # thresh = $50,000. If one trade is 10 BTC at $5,000/BTC = $50,000 → bar closes.
    # If trades are 1 BTC at $5,000 each → bar closes after exactly 10 trades.

    cumm_dollar = 0     # running total of dollar value within current bar
    cumm_vol    = 0     # running total of volume (needed for VWAP denominator)
    vol_price   = 0     # running total of (price × volume) = same as cumm_dollar here
    collector   = []
    bars        = []

    for i, (price, volume, date) in enumerate(zip(df['Price'], df['Volume'], df['Date'])):
        dollar       = price * volume   # dollar value of this single trade
        cumm_dollar += dollar           # accumulate dollar value
        cumm_vol    += volume           # accumulate volume
        vol_price   += dollar           # vol_price == cumm_dollar for dollar bars
        collector.append(price)

        if cumm_dollar >= thresh:       # has enough dollar value traded to close a bar?
            open_p  = collector[0]
            high_p  = np.max(collector)
            low_p   = np.min(collector)
            close_p = collector[-1]
            vwap    = vol_price / cumm_vol  # VWAP = Σ(p*v) / Σ(v)

            bars.append((date, i, open_p, low_p, high_p, close_p, vwap))

            # Reset for the next bar
            cumm_dollar = 0
            cumm_vol    = 0
            vol_price   = 0
            collector   = []

    cols = ['Date', 'Index', 'Open', 'Low', 'High', 'Close', 'Vwap']
    return pd.DataFrame(bars, columns=cols)
