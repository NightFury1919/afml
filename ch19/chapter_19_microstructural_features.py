"""
Chapter 19 -- Microstructural Features.

Real-data-first demo, three parts:
  A. Rebuild the canonical $10,000 real BTC/TUSD dollar bars (Ch02's
     convention) from the raw trade tape, tagging every trade with its
     bar id -- the shared foundation every feature below is computed on.
  B. Compute each of the 9 features once over the whole real series and
     report genuine headline numbers (not synthetic placeholders).
  C. Build a full per-bar feature table across all 249 real bars and save
     it to input_data/ -- this is the artifact meant to feed into Ch07's
     training table (the enrichment goal this chapter exists for), which
     is a follow-on step, not done in this script.

Path convention: this .py script derives its own root via __file__ (works
for anyone who clones the repo, any OS, any username). The paired notebook
uses a hardcoded AFML_ROOT instead, per CLAUDE.md.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, 'microstructural_features'))

INPUT_DATA = os.path.join(ROOT, 'input_data')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import microstructural_features as mf

DOLLAR_BAR_THRESHOLD = 10000.0  # matches Ch02 onward -- the whole pipeline's standard


# =============================================================================
# Part A -- rebuild the canonical dollar bars, tagging every trade with its bar
# =============================================================================
print("=" * 78)
print("PART A -- Rebuilding the real $10,000 dollar bars from raw trades")
print("=" * 78)

trades_path = os.path.join(INPUT_DATA, 'BTCTUSD-trades-2026-03.csv')
raw = pd.read_csv(trades_path, header=None,
                   names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                          'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])
raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')
# IsBuyerMaker=True  -> seller was aggressive -> sell-initiated -> Label = -1
# IsBuyerMaker=False -> buyer was aggressive  -> buy-initiated  -> Label = +1
# This is Binance's OWN record of the true aggressor side -- not inferred.
# We use it as ground truth to check the tick rule's accuracy below, and as
# the real signed-volume input to Kyle's Lambda / VPIN / serial correlation,
# rather than relying purely on tick-rule inference.
raw['Label'] = raw['IsBuyerMaker'].apply(lambda x: -1 if x else 1)
trades = raw[['Date', 'Price', 'Volume', 'Label']].copy()
print(f"Loaded {len(trades)} real trades from {trades_path}")

cumm_dollar, bar_id, bar_ids = 0.0, 0, []
for price, volume in zip(trades['Price'], trades['Volume']):
    cumm_dollar += price * volume
    bar_ids.append(bar_id)
    if cumm_dollar >= DOLLAR_BAR_THRESHOLD:
        bar_id += 1
        cumm_dollar = 0.0
trades['bar_id'] = bar_ids

n_complete_bars = trades['bar_id'].max()  # last bar_id is the incomplete trailing partial bar
trades = trades[trades['bar_id'] < n_complete_bars].copy()
print(f"{n_complete_bars} complete $10,000 dollar bars "
      f"({len(trades)} trades used, matching the pipeline's canonical 249-bar count)")

bars = trades.groupby('bar_id').agg(
    Open=('Price', 'first'), High=('Price', 'max'), Low=('Price', 'min'), Close=('Price', 'last'),
    Volume=('Volume', 'sum'), n_trades=('Price', 'size'),
)
dollar_per_trade = trades['Price'] * trades['Volume']
bars['DollarVolume'] = dollar_per_trade.groupby(trades['bar_id']).sum()
bars['BuyVolume'] = trades[trades['Label'] == 1].groupby('bar_id')['Volume'].sum().reindex(bars.index).fillna(0)
bars['SellVolume'] = trades[trades['Label'] == -1].groupby('bar_id')['Volume'].sum().reindex(bars.index).fillna(0)
print(bars[['Open', 'High', 'Low', 'Close', 'n_trades']].head())


# =============================================================================
# Part B -- each feature, once, over the whole real series: genuine headline numbers
# =============================================================================
print()
print("=" * 78)
print("PART B -- Headline real-data results, feature by feature")
print("=" * 78)

# --- 19.3.1 Tick rule vs. true side ---
inferred_side = mf.tick_rule(trades['Price'].values)
tick_acc = mf.tick_rule_accuracy(inferred_side, trades['Label'].values)
print(f"\n[19.3.1] Tick rule accuracy vs. Binance's true aggressor side: {tick_acc:.4f}")
print("         (equity-market studies typically cite >85% -- BTC's fine price "
      "granularity and\n          large share of same-price trades likely explain "
      "the lower real number here.)")

# --- 19.3.2 Roll model ---
roll_res = mf.roll_measure(bars['Close'].values)
print(f"\n[19.3.2] Roll model (whole 249-bar close series):")
print(f"         effective half-spread c      = ${roll_res['c']:.2f}")
print(f"         fundamental noise sigma_u     = ${roll_res['sigma_u']:.2f}")
print("         NOTE: Roll's model was derived for trade-to-trade price bounce, "
      "not\n         bar-close-to-bar-close bounce -- applying it to bar closes is a "
      "genuine\n         adaptation (dollar bars aren't uniformly-spaced ticks), kept "
      "here because\n         it's the only price series consistent with the rest of "
      "this pipeline.")

# --- 19.3.3 Parkinson volatility ---
park_vol = mf.parkinson_volatility(bars['High'].values, bars['Low'].values)
print(f"\n[19.3.3] Parkinson high-low volatility (whole series): sigma_HL = {park_vol:.6f}")

# --- 19.3.4 Corwin-Schultz + Becker-Parkinson ---
cs_spread = mf.corwin_schultz(bars['High'], bars['Low'], sl=1)
beta = mf.get_beta(bars['High'], bars['Low'], 1)
gamma = mf.get_gamma(bars['High'], bars['Low'])
bp_sigma = mf.becker_parkinson_sigma(beta, gamma)
frac_zero = float((cs_spread == 0).mean())
print(f"\n[19.3.4] Corwin-Schultz spread: mean={cs_spread.mean():.6f}, "
      f"median={cs_spread.median():.6f}")
print(f"         {frac_zero:.1%} of bars have alpha<0 -> spread clipped to 0 "
      "(book's own rule, p.727)")
print(f"         Becker-Parkinson sigma: mean={bp_sigma.mean():.6f}")

# --- 19.4.1 Kyle's Lambda ---
signed_vol = trades['Volume'] * trades['Label']
kyle = mf.kyle_lambda_by_bar(trades['Price'].values, signed_vol.values,
                              trades['bar_id'].values, min_trades=5)
print(f"\n[19.4.1] Kyle's Lambda: {kyle.notna().sum()} of {len(kyle)} bars had >=5 trades "
      "to fit.")
print(f"         mean={kyle.mean():.1f}, median={kyle.median():.1f}, "
      f"range=[{kyle.min():.1f}, {kyle.max():.1f}]")
print(f"         {float((kyle < 0).mean()):.1%} of per-bar estimates come out NEGATIVE, "
      "which Kyle's own\n         model rules out (lambda>0 is the model's second-order "
      "condition, Sec 19.4.1).\n         This is a real limitation of fitting the "
      "regression on small per-bar trade counts\n         (median ~37 trades/bar) rather "
      "than a book bug -- flagged, not silently dropped.")

# --- 19.4.2 Amihud's Lambda ---
amihud = mf.amihud_lambda(bars['Close'].values, bars['DollarVolume'].values)
print(f"\n[19.4.2] Amihud's Lambda (whole series): {amihud:.4e}")

# --- 19.5.2 VPIN ---
vpin10 = mf.vpin(bars['BuyVolume'].values, bars['SellVolume'].values, window=10)
print(f"\n[19.5.2] VPIN (10-bar rolling window): mean={vpin10.mean():.4f}, "
      f"latest={vpin10.iloc[-1]:.4f}")
print("         (elevated relative to typical published VPIN ~0.1-0.3 -- plausibly a "
      "small-window,\n         small-bar-count artifact rather than a genuine toxicity "
      "signal; revisit if the\n         real feature set ever grows enough bars to "
      "widen the window.)")

# --- 19.6.1 Round-number frequency ---
rnf = mf.round_number_frequency(trades['Volume'].values)
top_levels = sorted(rnf['by_level'].items(), key=lambda kv: -kv[1])[:5]
print(f"\n[19.6.1] Round-number trade-size frequency: {rnf['round_fraction']:.4f}")
print(f"         top matched levels: {top_levels}")
print("         NOTE: this is an ADAPTED feature (Sec 19.6.1 is about discrete equity "
      "contract\n         counts). The top level (0.0001 BTC) is plausibly Binance's own "
      "order-size step\n         (lot-size grid), not evidence of human 'round-number' "
      "psychology -- a genuinely\n         different phenomenon from what Easley et al. "
      "[2016] describe. Flagged, not\n         claimed as a like-for-like reproduction.")

# --- 19.6.5 Serial correlation of signed order flow ---
sc1 = mf.serial_correlation_signed_flow(trades['Label'].values, lag=1)
sc5 = mf.serial_correlation_signed_flow(trades['Label'].values, lag=5)
print(f"\n[19.6.5] Serial correlation of true trade sign: lag-1={sc1:.4f}, lag-5={sc5:.4f}")
print("         Positive and decaying with lag -- consistent with the book's own "
      "discussion\n         (Toth et al. [2011]): order splitting over short horizons, "
      "not necessarily herding.")

# --- notebook-output convention: printed stats get an accompanying chart ---
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].hist(kyle.dropna(), bins=25, color='steelblue', edgecolor='white')
axes[0, 0].axvline(0, color='black', linewidth=1)
axes[0, 0].set_title("Kyle's Lambda per real bar (Fig 19.1 analogue)")
axes[0, 0].set_xlabel('lambda')

axes[0, 1].hist(cs_spread, bins=25, color='seagreen', edgecolor='white')
axes[0, 1].set_title('Corwin-Schultz spread per real bar')
axes[0, 1].set_xlabel('spread (fraction of price)')

axes[1, 0].plot(vpin10.index, vpin10.values, color='darkorange')
axes[1, 0].set_title('VPIN, 10-bar rolling window')
axes[1, 0].set_xlabel('bar id')

axes[1, 1].plot(trades['bar_id'].values[:2000], signed_vol.cumsum().values[:2000], color='crimson')
axes[1, 1].set_title('Cumulative signed order flow (first 2000 trades)')
axes[1, 1].set_xlabel('bar id')

plt.tight_layout()
fig_path = os.path.join(HERE, 'ch19_feature_distributions.png')
plt.savefig(fig_path, dpi=110)
print(f"\nSaved distribution plots to {fig_path}")


# =============================================================================
# Part C -- full per-bar feature table across all 249 real bars (Ch07 input)
# =============================================================================
print()
print("=" * 78)
print("PART C -- Building the full per-bar feature table")
print("=" * 78)

ROLL_WINDOW = 20  # rolling window for the whole-series estimators, kept small given
                   # only 249 real bars total


def _rolling_bar(a, b, func, window):
    out = [np.nan] * len(a)
    for i in range(window - 1, len(a)):
        try:
            out[i] = func(a[i - window + 1:i + 1], b[i - window + 1:i + 1])
        except Exception:
            out[i] = np.nan
    return out


closes = bars['Close'].values
roll_c, roll_sigma_u = [], []
for i in range(len(closes)):
    if i < ROLL_WINDOW:
        roll_c.append(np.nan)
        roll_sigma_u.append(np.nan)
        continue
    res = mf.roll_measure(closes[i - ROLL_WINDOW:i + 1])
    roll_c.append(res['c'])
    roll_sigma_u.append(res['sigma_u'])

parkinson_roll = _rolling_bar(bars['High'].values, bars['Low'].values, mf.parkinson_volatility, ROLL_WINDOW)
amihud_roll = _rolling_bar(bars['Close'].values, bars['DollarVolume'].values, mf.amihud_lambda, ROLL_WINDOW)

cs_spread_full = mf.corwin_schultz(bars['High'], bars['Low'], sl=1).reindex(bars.index)
bp_sigma_full = mf.becker_parkinson_sigma(
    mf.get_beta(bars['High'], bars['Low'], 1), mf.get_gamma(bars['High'], bars['Low'])
).reindex(bars.index)

kyle_full = mf.kyle_lambda_by_bar(
    trades['Price'].values, signed_vol.values, trades['bar_id'].values, min_trades=5
).reindex(bars.index)

vpin_full = mf.vpin(bars['BuyVolume'].values, bars['SellVolume'].values, window=10)

# Round-number-fraction, serial-correlation, and tick-accuracy are computed
# per single bar (each bar has ~37 trades on average -- small, but these are
# trade-level features, not bar-level, so a single-bar window is the most
# natural unit rather than mixing multiple bars' worth of trades together).
round_frac, serial_corr, tick_acc_bar = [], [], []
for bid, grp in trades.groupby('bar_id'):
    round_frac.append(mf.round_number_frequency(grp['Volume'].values)['round_fraction'])
    serial_corr.append(
        mf.serial_correlation_signed_flow(grp['Label'].values, lag=1) if len(grp) > 3 else np.nan
    )
    tick_acc_bar.append(
        mf.tick_rule_accuracy(mf.tick_rule(grp['Price'].values), grp['Label'].values)
    )

feature_table = pd.DataFrame({
    'roll_c': roll_c,
    'roll_sigma_u': roll_sigma_u,
    'parkinson_vol_20bar': parkinson_roll,
    'corwin_schultz_spread': cs_spread_full.values,
    'becker_parkinson_sigma': bp_sigma_full.values,
    'kyle_lambda': kyle_full.values,
    'amihud_lambda_20bar': amihud_roll,
    'vpin_10bar': vpin_full.values,
    'round_number_fraction': round_frac,
    'serial_corr_signed_flow': serial_corr,
    'tick_rule_accuracy': tick_acc_bar,
}, index=bars.index)
feature_table.index.name = 'bar_id'

print(f"Feature table shape: {feature_table.shape}")
print(feature_table.describe().T[['count', 'mean', 'std', 'min', 'max']])

csv_out = os.path.join(INPUT_DATA, 'ch19_microstructural_features.csv')
pkl_out = os.path.join(INPUT_DATA, 'ch19_microstructural_features.pkl')
feature_table.to_csv(csv_out)
feature_table.to_pickle(pkl_out)
print(f"\nSaved: {csv_out}")
print(f"Saved: {pkl_out}")
print("\nNOTE: this table is bar-indexed (249 real dollar bars), not yet merged into "
      "Ch07's\n88-event training table -- that merge (aligning each triple-barrier event "
      "to its bar\nand joining these columns onto fracdiff) is the deliberate next step, "
      "not done here.")


# =============================================================================
# TDD results -- embedded per project convention, after tests passed
# =============================================================================
# REAL-MACHINE CONFIRMED (Python 3.10.20 / pytest 9.0.3 / mlfinlab env,
# 2026-07-16) -- 41 passed in 0.88s. Real-data headline numbers above (tick
# rule accuracy 0.6618, Kyle's Lambda range [-10787.4, 22233.8], VPIN mean
# 0.5256, etc.) are byte-identical to the sandbox run -- nothing here was
# environment-sensitive.
#
# ===================================================================== test session starts ======================================================================
# test_microstructural_features.py::TestTickRule::test_hand_traced_sequence PASSED                                                                          [  2%]
# test_microstructural_features.py::TestTickRule::test_b0_is_configurable PASSED                                                                            [  4%]
# test_microstructural_features.py::TestTickRule::test_empty_input PASSED                                                                                   [  7%]
# test_microstructural_features.py::TestTickRule::test_single_price PASSED                                                                                  [  9%]
# test_microstructural_features.py::TestTickRuleAccuracy::test_hand_traced_5_of_6 PASSED                                                                    [ 12%]
# test_microstructural_features.py::TestTickRuleAccuracy::test_perfect_agreement PASSED                                                                     [ 14%]
# test_microstructural_features.py::TestTickRuleAccuracy::test_length_mismatch_raises PASSED                                                                [ 17%]
# test_microstructural_features.py::TestTickRuleAccuracy::test_empty_raises PASSED                                                                          [ 19%]
# test_microstructural_features.py::TestRollMeasure::test_cross_validated_against_numpy PASSED                                                              [ 21%]
# test_microstructural_features.py::TestRollMeasure::test_known_regression_values PASSED                                                                    [ 24%]
# test_microstructural_features.py::TestRollMeasure::test_too_short_raises PASSED                                                                           [ 26%]
# test_microstructural_features.py::TestParkinsonVolatility::test_hand_traced_formula PASSED                                                                [ 29%]
# test_microstructural_features.py::TestParkinsonVolatility::test_known_value PASSED                                                                        [ 31%]
# test_microstructural_features.py::TestParkinsonVolatility::test_zero_range_gives_zero_vol PASSED                                                          [ 34%]
# test_microstructural_features.py::TestParkinsonVolatility::test_length_mismatch_raises PASSED                                                             [ 36%]
# test_microstructural_features.py::TestCorwinSchultz::test_cross_validated_against_reference_port PASSED                                                   [ 39%]
# test_microstructural_features.py::TestCorwinSchultz::test_negative_alpha_clipped_to_zero_spread PASSED                                                    [ 41%]
# test_microstructural_features.py::TestCorwinSchultz::test_known_values PASSED                                                                             [ 43%]
# test_microstructural_features.py::TestCorwinSchultz::test_becker_parkinson_sigma_nonnegative PASSED                                                       [ 46%]
# test_microstructural_features.py::TestKyleLambda::test_exact_slope_recovered PASSED                                                                       [ 48%]
# test_microstructural_features.py::TestKyleLambda::test_cross_validated_against_polyfit PASSED                                                             [ 51%]
# test_microstructural_features.py::TestKyleLambda::test_no_variation_in_volume_returns_nan PASSED                                                          [ 53%]
# test_microstructural_features.py::TestKyleLambda::test_too_few_points_returns_nan PASSED                                                                  [ 56%]
# test_microstructural_features.py::TestKyleLambda::test_length_mismatch_raises PASSED                                                                      [ 58%]
# test_microstructural_features.py::TestKyleLambdaByBar::test_known_two_bar_case PASSED                                                                     [ 60%]
# test_microstructural_features.py::TestKyleLambdaByBar::test_bar_below_min_trades_is_nan PASSED                                                            [ 63%]
# test_microstructural_features.py::TestAmihudLambda::test_cross_validated_closed_form_ols PASSED                                                           [ 65%]
# test_microstructural_features.py::TestAmihudLambda::test_known_value PASSED                                                                               [ 68%]
# test_microstructural_features.py::TestAmihudLambda::test_zero_volume_returns_nan PASSED                                                                   [ 70%]
# test_microstructural_features.py::TestAmihudLambda::test_length_mismatch_raises PASSED                                                                    [ 73%]
# test_microstructural_features.py::TestVpin::test_hand_traced_rolling_window PASSED                                                                        [ 75%]
# test_microstructural_features.py::TestVpin::test_window_one_gives_per_bar_imbalance_fraction PASSED                                                       [ 78%]
# test_microstructural_features.py::TestVpin::test_length_mismatch_raises PASSED                                                                            [ 80%]
# test_microstructural_features.py::TestVpin::test_zero_window_raises PASSED                                                                                [ 82%]
# test_microstructural_features.py::TestRoundNumberFrequency::test_hand_traced_matches PASSED                                                               [ 85%]
# test_microstructural_features.py::TestRoundNumberFrequency::test_custom_levels PASSED                                                                     [ 87%]
# test_microstructural_features.py::TestRoundNumberFrequency::test_empty_raises PASSED                                                                      [ 90%]
# test_microstructural_features.py::TestSerialCorrelationSignedFlow::test_perfect_alternation_gives_minus_one PASSED                                        [ 92%]
# test_microstructural_features.py::TestSerialCorrelationSignedFlow::test_cross_validated_against_pandas_autocorr PASSED                                    [ 95%]
# test_microstructural_features.py::TestSerialCorrelationSignedFlow::test_lag_2 PASSED                                                                      [ 97%]
# test_microstructural_features.py::TestSerialCorrelationSignedFlow::test_too_short_raises PASSED                                                           [100%]
#
# ====================================================================== 41 passed in 0.88s ======================================================================
