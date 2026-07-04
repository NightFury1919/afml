import sys
import os

# Add both ch02 and ch03 to path
ch02 = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ch02'))
ch03 = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ch02)
sys.path.insert(0, ch03)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import bars      # from ch02
import labeling  # from ch03

# =============================================================================
# Load Raw Tick Data and Generate Dollar Bars
# =============================================================================
afml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(afml_root, 'input_data', 'BTCTUSD-trades-2026-03.csv')

print("Loading tick data...")
raw = pd.read_csv(data_path, header=None,
                  names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                         'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])

raw['Date']  = pd.to_datetime(raw['Timestamp'], unit='us')
raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)

df = raw[['Date', 'Price', 'Volume', 'Label']].copy()
df['Dollar'] = df['Price'] * df['Volume']
df = bars.delta(df)

print(f"Loaded {len(df):,} ticks")
print(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")

# =============================================================================
# Dollar Bars
# =============================================================================
print("\nGenerating dollar bars...")
dollar_bars = bars.dollar_bars(df, thresh=10000)
dollar_bars = dollar_bars.set_index('Date')
print(f"Dollar bars: {len(dollar_bars)} bars")

close = dollar_bars['Close']

# =============================================================================
# CUSUM Filter
# =============================================================================
print("\nApplying CUSUM filter...")
cusum_df = pd.DataFrame({'Date': close.index, 'Price': close.values})
events   = bars.cusum_filter(cusum_df, h=500)
print(f"CUSUM events: {len(events)}")

# =============================================================================
# Daily Volatility (Section 3.1)
# =============================================================================
print("\nComputing daily volatility...")
daily_vol = labeling.get_daily_vol(close, span0=100)
print(f"Mean volatility: {daily_vol.mean()*100:.2f}%")

# --- Plot 1: Daily Volatility ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily_vol.index, daily_vol * 100, color='purple', linewidth=0.8)
ax.set_title("Daily Volatility Estimate (EWMA, span=100) — Dollar Bar Closes", fontsize=12)
ax.set_xlabel("Date")
ax.set_ylabel("Volatility (%)")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 3.2 — Fixed-Time Horizon Labeling
# =============================================================================
print("\nApplying fixed-time horizon labeling...")
fth_labels = labeling.fixed_time_horizon(close=close, events=events, h=5, threshold=0.01)
print(f"FTH labels: {len(fth_labels)}")
print(fth_labels.value_counts().sort_index())

# --- Plot 2: Fixed-Time Horizon ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fixed-Time Horizon Labeling (h=5 bars, τ=1%)", fontsize=12)

counts = fth_labels.value_counts().sort_index()
axes[0].bar(['-1 (Loss)', '0 (Neutral)', '+1 (Win)'],
            [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)],
            color=['red', 'grey', 'green'])
axes[0].set_title("Label Counts")
axes[0].set_ylabel("Number of observations")

axes[1].scatter(fth_labels.index, fth_labels.values,
                c=fth_labels.map({-1: 'red', 0: 'grey', 1: 'green'}),
                alpha=0.6, s=20)
axes[1].set_title("Labels Over Time")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Label")
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 3.3-3.5 — Triple Barrier Labeling
# =============================================================================
print("\nApplying triple barrier labeling...")
t1        = labeling.add_vertical_barrier(close, events, num_days=3)
tb_events = labeling.get_events(
    close=close, t_events=events, pt_sl=[1, 1],
    trgt=daily_vol, min_ret=0.005, t1=t1
)
tb_labels = labeling.get_bins(tb_events, close)
print(f"Triple barrier labels: {len(tb_labels)}")
print(tb_labels['bin'].value_counts().sort_index())

# --- Plot 3: Triple Barrier ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Triple Barrier Labeling (ptSl=[1,1], num_days=3)", fontsize=12)

counts = tb_labels['bin'].value_counts().sort_index()
axes[0].bar(['-1 (Loss)', '0 (Neutral)', '+1 (Win)'],
            [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)],
            color=['red', 'grey', 'green'])
axes[0].set_title("Label Counts")
axes[0].set_ylabel("Number of observations")

axes[1].scatter(tb_labels.index, tb_labels['bin'].values,
                c=tb_labels['bin'].map({-1: 'red', 0: 'grey', 1: 'green'}),
                alpha=0.6, s=20)
axes[1].set_title("Labels Over Time")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Label")
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# Comparison — FTH vs Triple Barrier
# =============================================================================
fth_counts = fth_labels.value_counts().sort_index()
tb_counts  = tb_labels['bin'].value_counts().sort_index()
comparison = pd.DataFrame({
    'Fixed-Time Horizon': fth_counts,
    'Triple Barrier':     tb_counts
}).fillna(0).astype(int)
comparison.index = ['-1 (Loss)', '0 (Neutral)', '+1 (Win)']
print("\nLabel distribution comparison:")
print(comparison)

# =============================================================================
# Section 3.6-3.7 — Meta-Labeling
# =============================================================================
print("\nApplying meta-labeling...")
ma20 = close.rolling(20).mean()
primary_side = pd.Series(
    np.where(close.reindex(events, method='bfill') > ma20.reindex(events, method='bfill'), 1, -1),
    index=events
)
print(f"Primary model: {(primary_side==1).sum()} long, {(primary_side==-1).sum()} short")

t1          = labeling.add_vertical_barrier(close, events, num_days=3)
meta_events = labeling.get_events_meta(
    close=close, t_events=events, pt_sl=[1, 1],
    trgt=daily_vol, min_ret=0.005, t1=t1, side=primary_side
)
meta_labels = labeling.get_bins_meta(meta_events, close)
print(f"Meta-labels: {len(meta_labels)}")
print(meta_labels['bin'].value_counts().sort_index())
print(f"Primary model accuracy: {meta_labels['bin'].mean()*100:.1f}%")

# --- Plot 4: Meta-Labeling ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Meta-Labeling: Was the Primary Model Right?", fontsize=12)

counts = meta_labels['bin'].value_counts().sort_index()
axes[0].bar(['0 (Wrong)', '1 (Right)'],
            [counts.get(0, 0), counts.get(1, 0)],
            color=['red', 'green'])
axes[0].set_title("Primary Model Accuracy")
axes[0].set_ylabel("Number of observations")

correct   = meta_labels[meta_labels['bin'] == 1]['ret']
incorrect = meta_labels[meta_labels['bin'] == 0]['ret']
axes[1].hist(correct * 100,   bins=15, alpha=0.6, color='green', label='Correct (bin=1)')
axes[1].hist(incorrect * 100, bins=15, alpha=0.6, color='red',   label='Wrong (bin=0)')
axes[1].set_title("Return Distribution by Label")
axes[1].set_xlabel("Return (%)")
axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.show()

# =============================================================================
# Section 3.9 — Drop Labels
# =============================================================================
print("\nDropping rare labels...")
tb_labels_clean = labeling.drop_labels(tb_labels.copy(), min_pct=0.10)
print(f"Before: {len(tb_labels)} observations")
print(f"After:  {len(tb_labels_clean)} observations")
print(tb_labels_clean['bin'].value_counts().sort_index())

# =============================================================================
# TDD TEST RESULTS — Chapter 3
# pytest ch03/tests/test_ch03.py -v
# Run date: 2026-06-12  |  Python 3.10.20  |  pytest 9.0.3
# =============================================================================
# TestFixedTimeHorizon::test_labels_known_values                  PASSED
# TestFixedTimeHorizon::test_output_is_series                     PASSED
# TestFixedTimeHorizon::test_labels_in_valid_set                  PASSED
# TestFixedTimeHorizon::test_skips_events_too_close_to_end        PASSED
# TestFixedTimeHorizon::test_positive_return_above_threshold      PASSED
# TestFixedTimeHorizon::test_negative_return_below_threshold      PASSED
# TestFixedTimeHorizon::test_neutral_label_within_threshold       PASSED
# TestGetDailyVol::test_returns_series                            PASSED
# TestGetDailyVol::test_first_value_is_nan                        PASSED
# TestGetDailyVol::test_known_values                              PASSED
# TestGetDailyVol::test_all_non_nan_values_positive               PASSED
# TestGetDailyVol::test_larger_span_smoother                      PASSED
# TestAddVerticalBarrier::test_returns_series                     PASSED
# TestAddVerticalBarrier::test_known_vertical_barrier_dates       PASSED
# TestAddVerticalBarrier::test_barrier_dates_after_event_dates    PASSED
# TestAddVerticalBarrier::test_events_beyond_data_excluded        PASSED
# TestGetEvents::test_returns_dataframe                           PASSED
# TestGetEvents::test_has_t1_and_trgt_columns                     PASSED
# TestGetEvents::test_known_t1_value                              PASSED
# TestGetEvents::test_min_ret_filters_events                      PASSED
# TestGetBins::test_returns_dataframe                             PASSED
# TestGetBins::test_has_ret_and_bin_columns                       PASSED
# TestGetBins::test_bin_values_in_valid_set                       PASSED
# TestGetBins::test_known_bin_and_ret                             PASSED
# TestGetBins::test_bin_sign_matches_ret                          PASSED
# TestGetEventsMeta::test_no_side_behaves_like_get_events         PASSED
# TestGetEventsMeta::test_with_side_includes_side_column          PASSED
# TestGetEventsMeta::test_without_side_drops_side_column          PASSED
# TestGetBinsMeta::test_no_side_bin_values_correct                PASSED
# TestGetBinsMeta::test_no_side_bins_in_valid_set                 PASSED
# TestGetBinsMeta::test_with_side_bins_binary                     PASSED
# TestGetBinsMeta::test_with_side_known_bin                       PASSED
# TestGetBinsMeta::test_correct_primary_model_gives_bin_one       PASSED
# TestDropLabels::test_removes_rare_label                         PASSED
# TestDropLabels::test_stops_at_two_labels                        PASSED
# TestDropLabels::test_known_drop_result                          PASSED
# TestDropLabels::test_returns_dataframe                          PASSED
# TestDropLabels::test_no_drop_when_all_above_threshold           PASSED
# -----------------------------------------------------------------------------
# 38 passed in 1.03s
# =============================================================================

print(tb_events.columns.tolist())
print(tb_events.head())
print(tb_labels.columns.tolist())
print(tb_labels.head())

# --- Save Chapter 3 outputs for downstream chapters ---
input_data_dir = os.path.join(afml_root, 'input_data')
ch03_events = pd.concat([tb_events, tb_labels], axis=1)
# LOAD-BEARING — do not remove. The outer concat above keeps all 94 tb_events
# rows; the 6 events whose barrier never resolved (t1 NaN, vertical deadline
# past the data end) come through with NaN in bin/ret. This dropna is the ONLY
# thing preventing those 6 NaN-label rows from entering the Ch07 training table.
# Deleting it silently poisons downstream chapters. 94 -> 88 by design.
ch03_events = ch03_events.dropna(subset=['t1', 'bin'])  # drop unresolved events
ch03_events.to_pickle(os.path.join(input_data_dir, 'ch03_events.pkl'))
ch03_events.to_csv(os.path.join(input_data_dir, 'ch03_events.csv'))
print(f"\nSaved ch03_events: {ch03_events.shape}, columns: {ch03_events.columns.tolist()}")

print(tb_events.shape, tb_labels.shape)
print(ch03_events.isna().sum())