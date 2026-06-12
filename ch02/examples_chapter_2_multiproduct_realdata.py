import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
# Adds this script's folder to Python's module search path so that
# `import multi_product as mp` resolves to the local multi_product/ package.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multi_product as mp   # our local package: etf_trick, pca_weights, roll functions

# =============================================================================
# Load S&P 500 Futures Data
# =============================================================================
# This script demonstrates the three multi-product techniques from AFML Chapter 2,
# Section 2.4, using real historical S&P 500 futures data.
#
# Contract naming convention:
#   SP = S&P 500 futures
#   98 = year 1998
#   H/M/U/Z = quarterly expiry month code:
#     H = March, M = June, U = September, Z = December
#
# Each .txt file contains daily OHLCV data for one contract:
#   Date, Open, High, Low, Close, Volume, OpenInt (open interest)

base = os.path.dirname(os.path.abspath(__file__))   # folder containing this script

def load_contract(filename):
    # Load one futures contract from a CSV/TXT file.
    # The files come in two slightly different formats:
    #   - Some have a header row with column names
    #   - Some start directly with data (date in YYMMDD integer format)
    # We auto-detect which format by checking if the first cell is a digit.
    path = os.path.join(base, 'input_data', filename)

    first = pd.read_csv(path, nrows=1, header=None)     # peek at just the first row
    has_header = not str(first.iloc[0, 0]).isdigit()    # if first cell isn't a number → header row

    if has_header:
        # File has column names already — just rename them to a consistent set
        df = pd.read_csv(path)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        # No header — dates are stored as integers like 980103 (= Jan 3, 1998)
        df = pd.read_csv(path, header=None,
                         names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OpenInt'])
        df['Date'] = pd.to_datetime(df['Date'], format='%y%m%d')  # parse YYMMDD format

    return df

# =============================================================================
# Section 2.4.2 & 2.4.1: ETF Trick + PCA Weights
# =============================================================================
# We use three contracts from 1998 as our instrument basket:
#   SP98H (March 1998), SP98M (June 1998), SP98U (September 1998)
# All three trade simultaneously — on any given date, all three are alive in the market.

print("Loading contracts for ETF Trick and PCA Weights...")
sp98h = load_contract('SP98H.txt')
sp98m = load_contract('SP98M.txt')
sp98u = load_contract('SP98U.txt')

# --- Align dates across all three contracts ---
# Not every contract trades on every date (e.g. around expiry). We keep only
# dates where ALL THREE contracts have data so our DataFrames are the same length.
dates = sp98h['Date']
common_dates = dates[dates.isin(sp98m['Date']) & dates.isin(sp98u['Date'])]
# isin() returns True for each date that appears in the other contract too.
# Chaining with & ensures a date must appear in ALL three.

sp98h_aligned = sp98h[sp98h['Date'].isin(common_dates)].set_index('Date')
sp98m_aligned = sp98m[sp98m['Date'].isin(common_dates)].set_index('Date')
sp98u_aligned = sp98u[sp98u['Date'].isin(common_dates)].set_index('Date')

# --- Build multi-instrument DataFrames (one column per instrument) ---
# Each DataFrame has shape (T × 3): T common trading days × 3 contracts.
instruments = ['SP98H', 'SP98M', 'SP98U']

close_prices = pd.DataFrame({
    'SP98H': sp98h_aligned['Close'],
    'SP98M': sp98m_aligned['Close'],
    'SP98U': sp98u_aligned['Close']
}).dropna()     # drop any remaining NaN rows (shouldn't be any after alignment above)

open_prices = pd.DataFrame({
    'SP98H': sp98h_aligned['Open'],
    'SP98M': sp98m_aligned['Open'],
    'SP98U': sp98u_aligned['Open']
}).dropna()

# --- Volume data ---
# Real traded volume for each contract, used to compute tradeable volume v_t.
# This tells us how many contracts were traded each day — used instead of
# close prices as a proxy, giving a much more realistic liquidity estimate.
volumes = pd.DataFrame({
    'SP98H': sp98h_aligned['Volume'],
    'SP98M': sp98m_aligned['Volume'],
    'SP98U': sp98u_aligned['Volume']
}).dropna()

# Align open_prices to exactly the same rows as close_prices
common_idx  = close_prices.index
common_idx  = common_idx[common_idx >= '1998-01-01']

close_prices = close_prices.loc[common_idx]   # ← filter close_prices too
open_prices  = open_prices.loc[common_idx]
volumes      = volumes.loc[common_idx]        # ← filter volumes too
n_bars       = len(close_prices)              # ← now correct

print(f"Common bars across all 3 contracts: {n_bars}")
print(f"Date range: {common_idx[0]} to {common_idx[-1]}")

# --- Point values ---
# S&P 500 futures: each index point is worth $250.
# So if the S&P moves from 1000 to 1001, one contract gains $250.
# We store this as a (T × 3) DataFrame so the ETF Trick can index it by date and instrument.
point_values = pd.DataFrame(
    np.ones((n_bars, 3)) * 250,     # $250 per point for all three contracts
    index=common_idx, columns=instruments
)

# --- Dividends ---
# Futures contracts do not pay dividends directly (they price them in via the
# cost-of-carry). We set dividends to zero for simplicity.
dividends = pd.DataFrame(
    np.zeros((n_bars, 3)),
    index=common_idx, columns=instruments
)

# --- Allocation weights ---
# Equal weight: 1/3 allocated to each of the three contracts.
# Weights are stored as a (T × 3) DataFrame so they can vary over time if needed.
alloc_weights = pd.DataFrame(
    np.ones((n_bars, 3)) / 3,       # 33.3% in each contract
    index=common_idx, columns=instruments
)

# --- Transaction costs ---
# 1 basis point = 0.0001 = 0.01% per trade, per instrument.
# This is a typical round-trip cost for liquid futures.
trans_costs = pd.Series(1e-4, index=instruments)

# --- Rebalance dates ---
# We rebalance every 63 trading days ≈ once per quarter.
# [::63] takes every 63rd element of the index (quarterly spacing).
rebalance_dates = list(common_idx[::63])

# =============================================================================
# Section 2.4.2: PCA Weights
# =============================================================================
# Goal: find portfolio weights such that risk is spread equally across the
# three principal components of the return covariance matrix.
#
# For three highly correlated contracts (all track the same S&P 500 index),
# most of the variance is concentrated in the first principal component
# (= the "market" direction where all three move together).
# The third component (= a spread, where some go up and some go down) has
# almost no variance — it's nearly a risk-free position.
#
# With risk_dist = [1/3, 1/3, 1/3], PCA Weights spreads risk equally:
# 1/3 of total risk comes from the market direction,
# 1/3 from the intermediate spread, 1/3 from the near-zero spread.
# Achieving equal risk in the near-zero component requires VERY large weights.

print("\n--- PCA Weights ---")
returns  = close_prices.pct_change().dropna()   # daily % returns: (p_t - p_{t-1}) / p_{t-1}
cov      = returns.cov().values                 # 3×3 covariance matrix of returns
risk_dist = np.ones(3) / 3                      # equal risk: 1/3 to each principal component
weights  = mp.pca_weights(cov, risk_dist=risk_dist, risk_target=1.0)

print("PCA Weights:")
for i, w in zip(instruments, weights.flatten()):
    print(f"  {i}: {w:.4f}")
# Expect large absolute values on some instruments — this is correct.
# The third principal component (lowest risk) needs very large offsetting positions
# to contribute a meaningful fraction of total portfolio risk.

# =============================================================================
# Section 2.4.1: ETF Trick
# =============================================================================
# Combines our three contracts into a single $1 portfolio value series K_t.
# K_t grows or shrinks each day by the P&L of the basket.

print("\n--- ETF Trick ---")
result = mp.etf_trick(
    open_prices     = open_prices,
    close_prices    = close_prices,
    alloc_weights   = alloc_weights,
    point_values    = point_values,
    dividends       = dividends,
    rebalance_dates = rebalance_dates,
    trans_costs     = trans_costs,
    volumes         = volumes       # real traded volume for liquidity calculation
)
print(result.head(10))
print("""
--- ETF Trick Output Notes ---
K:              Portfolio value starting at $1. Rises and falls with daily P&L across all three contracts.
rebalance_cost: Transaction cost paid on rebalance days only (~every 63 bars). Zero on all other days.
bid_ask_cost:   Implicit spread cost of holding positions, computed every bar. Small (~0.0001) because
                S&P 500 futures are highly liquid with tight spreads.
volume:         How many units of the virtual ETF could be traded today, limited by the least liquid
                instrument. Large values reflect that our $1 portfolio is tiny relative to market
                liquidity — the market could absorb far more. In a real deployment with a larger
                portfolio, this number would shrink proportionally.
""")

# =============================================================================
# Section 2.4.3: Single Future Roll (Roll Gap Correction)
# =============================================================================
# For a SINGLE continuous futures series we need to stitch together multiple
# contracts in order. Here we use the front-month contract (the nearest expiry)
# for each quarterly period, which is the most liquid and most commonly tracked.
#
# Each entry in front_month_contracts is:
#   (contract name, active start date, active end date)
# The dates are approximate — the contract is considered "front month" between
# its listing date and roughly 2-3 days before its expiry date.

print("\n--- Single Future Roll ---")

front_month_contracts = [
    ('SP98H', '1998-01-01', '1998-03-20'),
    ('SP98M', '1998-03-21', '1998-06-19'),
    ('SP98U', '1998-06-20', '1998-09-18'),
    ('SP98Z', '1998-09-19', '1998-12-18'),
    ('SP99H', '1998-12-19', '1999-03-19'),
    ('SP99M', '1999-03-20', '1999-06-18'),
    ('SP99U', '1999-06-19', '1999-09-17'),
    ('SP99Z', '1999-09-18', '1999-12-17'),
    ('SP00H', '1999-12-18', '2000-03-16'),
    ('SP00M', '2000-03-17', '2000-06-15'),
    ('SP00U', '2000-06-16', '2000-09-14'),
    ('SP00Z', '2000-09-15', '2000-12-14'),
]

# --- Stitch together the front-month contracts into one long series ---
pieces = []
for contract, start, end in front_month_contracts:
    df = load_contract(f'{contract}.txt')
    # Keep only the rows where this contract was the front month
    df = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()
    df['Instrument'] = contract     # tag each row with its contract name (needed by roll_gaps)
    pieces.append(df)

# pd.concat stacks all the contract slices vertically.
# sort_values ensures chronological order (some files may not be sorted).
# reset_index gives a clean 0,1,2,... integer index.
futures_series = pd.concat(pieces).sort_values('Date').reset_index(drop=True)
futures_series = futures_series.set_index('Date')   # set date as index for roll functions

print(f"Stitched futures series: {len(futures_series)} bars")
print(f"Date range: {futures_series.index[0]} to {futures_series.index[-1]}")
print(f"Contracts used: {futures_series['Instrument'].unique().tolist()}")

# --- Apply roll gap correction ---
# dictio maps the column names our roll functions expect to the actual column names in the data
dictio = {'Instrument': 'Instrument', 'Open': 'Open', 'Close': 'Close'}

# get_rolled_series subtracts cumulative gaps from Open and Close prices,
# producing a smooth series with no artificial jumps at roll dates.
rolled  = mp.get_rolled_series(futures_series, dictio=dictio, match_end=True)

# non_negative_rolled_prices converts to a $1 investment series via cumulative returns,
# ensuring the series is always positive even if adjusted prices would go negative.
non_neg = mp.non_negative_rolled_prices(futures_series, dictio=dictio, match_end=True)

print("\nRolled series (first 10 rows):")
print(rolled[["Open", "Close", "Volume", "Instrument"]].head(10))
print("\nNon-negative rolled prices (first 10 rows):")
print(non_neg[['Close', 'Returns', 'rPrices']].head(10))

# =============================================================================
# Figure 1: Price Series — three panels
# =============================================================================
# Panel 1: ETF Trick portfolio value K_t over time
# Panel 2: Raw vs. rolled S&P 500 futures prices (to see the gap correction)
# Panel 3: Non-negative compounded return series

fig1, axes1 = plt.subplots(3, 1, figsize=(14, 14))
fig1.suptitle("Chapter 2.4 — Multi-Product Series (Real S&P 500 Futures Data)",
              fontsize=14, y=0.98)

# Panel 1 — ETF Trick: how $1 invested in our equal-weight basket of 3 futures grew
axes1[0].plot(result.index, result['K'], color='blue')
axes1[0].set_title("ETF Trick: $1 Investment Value Across 3 S&P 500 Futures Contracts (Section 2.4.1)")
axes1[0].set_xlabel("Date")
axes1[0].set_ylabel("Portfolio Value ($)")
axes1[0].tick_params(axis='x', rotation=45)

# Panel 2 — Roll correction: red = raw prices (with artificial jumps), green = smoothed
axes1[1].plot(futures_series.index, futures_series['Close'],
              color='red', label='Raw prices', linewidth=0.8)
axes1[1].plot(rolled.index, rolled['Close'],
              color='green', label='Rolled prices', linewidth=0.8)
axes1[1].set_title("Single Future Roll: Raw vs Rolled S&P 500 Futures Prices (Section 2.4.3)")
axes1[1].set_xlabel("Date")
axes1[1].set_ylabel("Price")
axes1[1].legend(loc='upper left')
axes1[1].tick_params(axis='x', rotation=45)

# Panel 3 — Non-negative series: always positive, starts at 1.0
axes1[2].plot(non_neg.index, non_neg['rPrices'], color='purple', linewidth=0.8)
axes1[2].set_title("Single Future Roll: $1 Investment Value After Roll Correction (Section 2.4.3)")
axes1[2].set_xlabel("Date")
axes1[2].set_ylabel("Value ($)")
axes1[2].tick_params(axis='x', rotation=45)

plt.subplots_adjust(hspace=0.9, top=0.91, bottom=0.12)
plt.show()

# =============================================================================
# Figure 2: PCA Weights bar chart
# =============================================================================
# Shows the portfolio weight assigned to each of the three contracts.
# Green bars = long positions (positive weight), Red bars = short positions (negative weight).
# The large magnitudes are expected — equal-risk allocation to the near-zero
# spread component requires offsetting long/short positions of large size.

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle("Chapter 2.4 — PCA Weights: Risk-Balanced Allocation\nAcross 3 S&P 500 Futures Contracts (Section 2.4.2)",
              fontsize=12, y=0.98)
ax2.set_title("")   # clear subplot title; everything is in suptitle

weight_values = weights.flatten()                               # convert (3,1) array to (3,) array
colors = ['green' if w > 0 else 'red' for w in weight_values]  # positive=green, negative=red
bars_plot = ax2.bar(instruments, weight_values, color=colors, width=0.5)
ax2.axhline(y=0, color='black', linewidth=0.8)  # horizontal line at zero for reference
ax2.set_xlabel("Instrument")
ax2.set_ylabel("Weight")

# --- Label each bar with its weight value ---
# If the bar is very tall (|w| > 500), put the label INSIDE the bar (white text).
# If the bar is short, put the label just ABOVE or BELOW the bar (black text).
for i, w in enumerate(weight_values):
    if abs(w) > 500:
        y_pos = w / 2                           # centre of the bar
        ax2.text(i, y_pos, f'{w:.2f}', ha='center', va='center',
                 fontsize=11, color='white', fontweight='bold')
    else:
        offset = 200 if w >= 0 else -400        # nudge label above/below the bar
        ax2.text(i, w + offset, f'{w:.2f}', ha='center', va='center',
                 fontsize=11, color='black', fontweight='bold')

# Add strategy explanation text box
explanation = (
    "Strategy: Relative Value\n"
    "This portfolio does not bet on the S&P 500 going up or down.\n"
    "It bets on June (SP98M) outperforming March and September.\n\n"
    "Why long/short? The PCA formula found that all three contracts\n"
    "move together — so the only independent risk comes from the\n"
    "spread between them. The eigenvector for the smallest risk\n"
    "component has a positive sign for June and negative signs for\n"
    "March and September, so the math naturally produces this\n"
    "long/short structure to extract that spread.\n\n"
    "Risk profile: Low market risk (positions mostly cancel out).\n"
    "Sensitive to small pricing differences between contracts."
)
ax2.text(
    0.5, -0.28, explanation,
    transform=ax2.transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='center',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4F6FA', edgecolor='#1B2A4A', linewidth=1.2)
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.38)
plt.show()

# =============================================================================
# TDD TEST RESULTS — Chapter 2
# pytest ch02/tests/test_ch02.py -v
# Run date: 2026-06-12  |  Python 3.10.20  |  pytest 9.0.3
# =============================================================================
# TestEwma::test_empty_array_returns_zero                         PASSED
# TestEwma::test_single_element_returns_that_element              PASSED
# TestEwma::test_three_element_window_three                       PASSED
# TestEwma::test_five_elements_window_two                         PASSED
# TestEwma::test_larger_window_smoother                           PASSED
# TestDelta::test_first_row_is_zero                               PASSED
# TestDelta::test_price_changes_correct                           PASSED
# TestDelta::test_output_length_matches_input                     PASSED
# TestDelta::test_does_not_modify_price_column                    PASSED
# TestTickRule::test_uptick_gives_positive_one                    PASSED
# TestTickRule::test_downtick_gives_negative_one                  PASSED
# TestTickRule::test_flat_price_carries_forward                   PASSED
# TestTickRule::test_full_sequence                                 PASSED
# TestTickRule::test_default_first_label_is_one                   PASSED
# TestEstimateBuySellProbs::test_three_buys_one_sell              PASSED
# TestEstimateBuySellProbs::test_probs_sum_to_one                 PASSED
# TestEstimateBuySellProbs::test_all_buys                         PASSED  (bug fixed)
# TestCusumFilter::test_fires_on_upward_drift                     PASSED
# TestCusumFilter::test_event_dates_correct                       PASSED
# TestCusumFilter::test_no_events_when_drift_below_threshold      PASSED
# TestCusumFilter::test_returns_datetimeindex                     PASSED
# TestCusumFilter::test_downward_drift_also_fires                 PASSED
# TestTickBars::test_correct_number_of_bars                       PASSED
# TestTickBars::test_ohlc_values_first_bar                        PASSED
# TestTickBars::test_vwap_equal_volumes                           PASSED
# TestTickBars::test_vwap_unequal_volumes                         PASSED
# TestTickBars::test_incomplete_last_bar_not_included             PASSED
# TestVolumeBars::test_closes_on_volume_threshold                 PASSED
# TestVolumeBars::test_ohlc_values                                PASSED
# TestVolumeBars::test_large_single_trade_closes_bar              PASSED
# TestDollarBars::test_closes_on_dollar_threshold                 PASSED
# TestDollarBars::test_ohlcv_values                               PASSED
# TestDollarBars::test_single_large_trade_closes_bar              PASSED
# TestTickImbalanceBars::test_produces_bars                       PASSED
# TestTickImbalanceBars::test_first_bar_ohlc                      PASSED
# TestTickImbalanceBars::test_open_lte_high                       PASSED
# TestTickImbalanceBars::test_close_within_high_low               PASSED
# TestVolumeImbalanceBars::test_produces_bars                     PASSED
# TestVolumeImbalanceBars::test_has_vwap_column                   PASSED
# TestVolumeImbalanceBars::test_high_gte_low                      PASSED
# TestTickRunBars::test_correct_number_of_bars                    PASSED
# TestTickRunBars::test_first_bar_ohlcv                           PASSED
# TestTickRunBars::test_high_gte_open_and_close                   PASSED
# TestVolumeRunBars::test_correct_number_of_bars                  PASSED
# TestVolumeRunBars::test_first_bar_ohlcv                         PASSED
# TestVolumeRunBars::test_has_vwap                                PASSED
# TestPcaWeights::test_output_shape                               PASSED
# TestPcaWeights::test_min_variance_weights                       PASSED
# TestPcaWeights::test_equal_risk_weights                         PASSED
# TestPcaWeights::test_identity_cov_min_var                       PASSED
# TestPcaWeights::test_risk_target_scales_weights                 PASSED
# TestRollGaps::test_gaps_backward_correct                        PASSED
# TestRollGaps::test_last_gap_is_zero_when_match_end              PASSED
# TestRollGaps::test_forward_roll_first_gap_is_zero               PASSED
# TestGetRolledSeries::test_rolled_close_values                   PASSED
# TestGetRolledSeries::test_no_jump_at_roll_date                  PASSED
# TestGetRolledSeries::test_does_not_modify_original              PASSED
# TestNonNegativeRolledPrices::test_rprices_always_positive       PASSED
# TestNonNegativeRolledPrices::test_rprices_known_values          PASSED
# TestNonNegativeRolledPrices::test_first_rprices_is_nan          PASSED
# TestNonNegativeRolledPrices::test_has_returns_column            PASSED
# TestEtfTrick::test_starts_at_one_dollar                         PASSED
# TestEtfTrick::test_k_series_values                              PASSED
# TestEtfTrick::test_output_columns                               PASSED
# TestEtfTrick::test_output_length                                PASSED
# TestEtfTrick::test_no_costs_when_zero_trans_costs               PASSED
# -----------------------------------------------------------------------------
# 66 passed in 1.71s
# Note: test_all_buys revealed a real bug in estimate_buy_sell_probs —
#       function crashed when all ticks had the same label. Fixed with
#       a one-label guard in ch02/bars/utils.py.
# =============================================================================
