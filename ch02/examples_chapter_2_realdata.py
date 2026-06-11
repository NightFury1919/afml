import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# sys.path.insert(0, ...) adds the current directory to Python's module search path.
# This lets us write `import bars` instead of a full path, regardless of where
# the script is run from. Must be done BEFORE the `import bars` line below.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bars   # our local bars package (bars/__init__.py exposes all bar functions)

# =============================================================================
# Load Real Binance Tick Data
# =============================================================================
# This script demonstrates every bar type from AFML Chapter 2 using real
# Bitcoin tick data downloaded from Binance's public trade history API.
#
# Each row of the raw CSV is ONE INDIVIDUAL TRADE — not a bar, not a candle.
# Fields: TradeID, Price, Volume, QuoteVolume, Timestamp, IsBuyerMaker, IsBestMatch
#
# Volume      = how many BTC changed hands in this trade
# QuoteVolume = how many USD changed hands (= Price × Volume)
# Timestamp   = microseconds since Unix epoch (Jan 1 1970)
# IsBuyerMaker= True  if the SELLER was aggressive (market sell order hit a resting bid)
#             = False if the BUYER was aggressive (market buy order hit a resting ask)
# IsBestMatch = whether this was executed at the best available price (always True here)

base = os.path.dirname(os.path.abspath(__file__))   # absolute path to this script's folder
data_path = os.path.join(base, 'input_data', 'BTCTUSD-trades-2026-03.csv')

print("Loading data...")
raw = pd.read_csv(data_path, header=None,
                  names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                         'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])

# --- Convert timestamp ---
# Binance timestamps are in MICROseconds (millionths of a second).
# pd.to_datetime(..., unit='us') converts to a standard Python datetime.
# Example: 1709251200000000 us → 2026-03-01 00:00:00
raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')

# --- Assign trade direction label ---
# IsBuyerMaker=True  → the buyer was the passive side (resting limit order)
#                       → the SELLER was aggressive → price went DOWN → label = -1
# IsBuyerMaker=False → the buyer was aggressive (market order) → price went UP → label = +1
#
# This is equivalent to the Tick Rule output but more accurate — Binance tells us
# directly who initiated the trade, so we don't need to infer from price changes.
raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)

# --- Build the working dataframe ---
# We keep only the four columns our bar functions expect:
#   Date   : timestamp of the trade
#   Price  : execution price in USD
#   Volume : BTC traded
#   Label  : +1 (buy) or -1 (sell)
df = raw[['Date', 'Price', 'Volume', 'Label']].copy()

# Dollar value of each trade = Price × Volume (useful for dollar bar accounting)
df['Dollar'] = df['Price'] * df['Volume']

# Compute Δp_t — price change from previous tick (needed by some bar functions)
df = bars.delta(df)     # adds a 'Delta' column: Delta[i] = Price[i] - Price[i-1]

print(f"Loaded {len(df)} trades")
print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
print()

# =============================================================================
# Standard Bars
# =============================================================================
# These four bar types all sample at FIXED thresholds.
# The thresholds below are chosen to produce a manageable number of bars
# for visualisation — in practice you'd tune them based on your data.

print("Generating standard bars...")

# Daily bars — one bar per calendar day regardless of trading activity
time_bar   = bars.time_bars(df, freq='D')

# Tick bars — close every 100 trades
# 100 trades is relatively small for a liquid BTC market (thousands of trades/min)
tick_bar   = bars.tick_bars(df, thresh=100)

# Volume bars — close every 1 BTC traded
# Approximately equalises the "economic weight" of each bar
volume_bar = bars.volume_bars(df, thresh=1.0)

# Dollar bars — close every $50,000 of BTC traded
# Most robust to BTC's price changes over the month
dollar_bar = bars.dollar_bars(df, thresh=50000)

print(f"Time bars:   {len(time_bar)} bars")
print(f"Tick bars:   {len(tick_bar)} bars")
print(f"Volume bars: {len(volume_bar)} bars")
print(f"Dollar bars: {len(dollar_bar)} bars")

# =============================================================================
# Information-Driven Bars
# =============================================================================
# These four bar types adapt their sampling frequency to the INFORMATION CONTENT
# of the order flow, rather than a fixed threshold.
#
# expected_num_ticks_init sets the warmup length — we need this many ticks before
# the first bar can close. Higher values → more stable initial estimates but
# a longer period before the first bar appears.

print("\nGenerating information-driven bars...")

# Tick Imbalance Bars — close when cumulative signed tick count exceeds expectation
# High expected_num_ticks_init=500 → conservative, needs 500 ticks to warm up
tick_imbalance_bar = bars.tick_imbalance_bars(df, expected_num_ticks_init=500)

# Volume Imbalance Bars — same idea, but each tick is weighted by its trade volume
# Lower warmup (50) → bars start forming sooner
imbalance_bar      = bars.volume_imbalance_bars(df, expected_num_ticks_init=50)

# Tick Run Bars — close when the MAX of buy-tick-run or sell-tick-run exceeds expectation
run_bar            = bars.tick_run_bars(df, expected_num_ticks_init=50)

# Volume Run Bars — same idea, accumulating volume on each side instead of tick counts
volume_run_bar     = bars.volume_run_bars(df, expected_num_ticks_init=10)
print(f"Volume run bar expected imbalance debug: {len(volume_run_bar)} bars")

print(f"Tick imbalance bars:  {len(tick_imbalance_bar)} bars")
print(f"Volume imbalance bars:{len(imbalance_bar)} bars")
print(f"Tick run bars:        {len(run_bar)} bars")
print(f"Volume run bars:      {len(volume_run_bar)} bars")

# =============================================================================
# CUSUM Filter
# =============================================================================
# The CUSUM filter is a "pre-filter" that identifies dates where the price has
# moved significantly (by at least h dollars) away from its level at the last event.
# The output is a list of event dates — not bars — typically used to label
# training samples for a machine learning model.
#
# h = 500 means: fire an event only when cumulative price drift exceeds $500.
# This prevents generating hundreds of signals during a slow sideways grind.

print("\nApplying CUSUM filter...")
h = 500     # threshold in dollars — tune based on typical BTC move size
events = bars.cusum_filter(df, h)
print(f"CUSUM filter fired {len(events)} events")

# =============================================================================
# Plot 1: Standard Bar Types
# =============================================================================
# A 2×2 grid showing all four standard bar types on the same dataset.
# Each subplot plots PRICE (or VWAP) on the y-axis and TIME on the x-axis.
# Notice how the bars look similar in shape but differ in frequency:
#   - Time bars: exactly 1 per day regardless of activity
#   - Tick/Volume/Dollar bars: denser during volatile periods, sparser during quiet ones

sns.set_style("whitegrid")     # clean background with subtle grid lines
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))   # 2 rows × 2 columns of subplots
fig1.suptitle("Chapter 2 — Standard Bar Types — Real Bitcoin Trade Data (March 2026)",
              fontsize=13, y=0.98)

# Each tuple: (axis, x-values, y-values, color, title)
# Time bars use 'Price' (last price of the day); others use 'Vwap' (more representative)
standard_plots = [
    (axes1[0,0], time_bar['Date'],   time_bar['Price'],   'brown',  "Time Bars (Daily)"),
    (axes1[0,1], tick_bar['Date'],   tick_bar['Vwap'],    'purple', "Tick Bars (every 100 trades)"),
    (axes1[1,0], volume_bar['Date'], volume_bar['Vwap'],  'green',  "Volume Bars (every 1 BTC)"),
    (axes1[1,1], dollar_bar['Date'], dollar_bar['Vwap'],  'red',    "Dollar Bars (every $50,000)"),
]

for ax, x, y, color, title in standard_plots:
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=45)  # rotate dates so they don't overlap

plt.subplots_adjust(hspace=0.6, wspace=0.3)    # vertical and horizontal spacing between subplots
plt.show()

# =============================================================================
# Plot 2: Information-Driven Bar Types
# =============================================================================
# Same structure as Plot 1 but for the adaptive bar types.
# Notice that these bars will cluster together during volatile/imbalanced periods
# and spread out during calm, balanced periods — unlike the fixed-threshold bars above.

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle("Chapter 2 — Information-Driven Bar Types — Real Bitcoin Trade Data (March 2026)",
              fontsize=13, y=0.98)

info_plots = [
    (axes2[0,0], tick_imbalance_bar['Date'], tick_imbalance_bar['Close'], 'blue',    "Tick Imbalance Bars"),
    (axes2[0,1], imbalance_bar['Date'],      imbalance_bar['Vwap'],       'teal',    "Volume Imbalance Bars"),
    (axes2[1,0], run_bar['Date'],            run_bar['Vwap'],             'orange',  "Tick Run Bars"),
    (axes2[1,1], volume_run_bar['Date'],     volume_run_bar['Vwap'],      'magenta', "Volume Run Bars"),
]

for ax, x, y, color, title in info_plots:
    ax.plot(x, y, color=color)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=45)

plt.subplots_adjust(hspace=0.6, wspace=0.3)
plt.show()
