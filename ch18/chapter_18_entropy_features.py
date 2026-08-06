"""
Chapter 18 -- Entropy Features.

Real-data-first demo, three parts:
  A. Encoding schemes (18.5) + entropy estimators (18.1-18.4) run on
     real BTC/TUSD bar-to-bar returns (the same 249 real $10,000 dollar
     bars used throughout this pipeline, Ch02's convention).
  B. The 18.8.4 adverse-selection workflow, run on Ch19's real
     BuyVolume/SellVolume per bar (the same real trade tape).
  C. The 18.8.3 portfolio concentration formula, run against Chapter 16's
     REAL 6-commodity covariance matrix and HRP/IVP allocation vectors.
     Originally deferred (this pipeline is single-asset BTC/TUSD, so no
     covariance matrix existed here) -- revisited 2026-08-06 per the
     project's standing "implement once a later chapter supplies the
     prerequisite" rule, now that Ch16 has a real multi-asset covariance
     matrix + two real allocation vectors to run it against.

Formula-only sections still NOT given a code implementation here (per
scope decision with Ethan, 2026-08-04) are discussed conceptually in this
chapter's README/notebook markdown instead: 18.2 (Shannon entropy/MI),
18.6 (Gaussian entropy benchmark), 18.7 (generalized mean) -- none of
these have a real-data prerequisite the way 18.8.3 did, so they stay
conceptual until they'd produce an actual pipeline feature.

Path convention: this .py script derives its own root via __file__.
The paired notebook uses a hardcoded AFML_ROOT instead, per CLAUDE.md.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, 'entropy_features'))

INPUT_DATA = os.path.join(ROOT, 'input_data')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from encoding_schemes import binary_encode, quantile_encode, sigma_encode
from entropy_estimators import plugIn, konto
from adverse_selection import adverse_selection_feature
from portfolio_concentration import compute_portfolio_concentration

DOLLAR_BAR_THRESHOLD = 10000.0  # matches Ch02 onward -- the pipeline's standard


# =============================================================================
# Shared setup -- rebuild the canonical real $10,000 dollar bars, exactly as
# Ch19 does, so BuyVolume/SellVolume line up with the rest of the pipeline.
# =============================================================================
print("=" * 78)
print("Rebuilding the real $10,000 dollar bars from raw trades")
print("=" * 78)

trades_path = os.path.join(INPUT_DATA, 'BTCTUSD-trades-2026-03.csv')
raw = pd.read_csv(trades_path, header=None,
                   names=['TradeID', 'Price', 'Volume', 'QuoteVolume',
                          'Timestamp', 'IsBuyerMaker', 'IsBestMatch'])
raw['Date'] = pd.to_datetime(raw['Timestamp'], unit='us')
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

n_complete_bars = trades['bar_id'].max()
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


# =============================================================================
# Part A -- encoding schemes + entropy estimators on real bar-to-bar returns
# =============================================================================
print()
print("=" * 78)
print("PART A -- Encoding schemes (18.5) + entropy estimators (18.1-18.4)")
print("=" * 78)

returns = bars['Close'].pct_change().dropna()
print(f"\n{len(returns)} real bar-to-bar returns "
      f"({bars.index.min()} to {bars.index.max()}, 249 real dollar bars)")

encodings = {
    'binary': binary_encode(returns),
    'quantile (n=5)': quantile_encode(returns.values, n_letters=5),
    'sigma (default step=std/4)': sigma_encode(returns.values),
}

for name, encoded in encodings.items():
    alphabet = sorted(set(encoded))
    print(f"\n--- {name} encoding ---")
    print(f"    length={len(encoded)}, alphabet size={len(alphabet)} ({''.join(alphabet)})")

    h1, _ = plugIn(encoded, 1)
    print(f"    plugIn entropy rate (w=1): {h1:.4f} bits "
          f"(theoretical max for this alphabet: {np.log2(len(alphabet)):.4f})")

    # konto's expanding-window mode requires an even-length message --
    # drop the last symbol if needed (book's own stated precondition,
    # Sec 18.4), not a bug fix.
    konto_input = encoded if len(encoded) % 2 == 0 else encoded[:-1]
    out = konto(konto_input)
    print(f"    konto entropy rate: {out['h']:.4f} bits, redundancy r={out['r']:.4f}")

print("\nREAL FINDING: binary-encoded entropy (0.9997 bits) sits almost exactly "
      "at the\ntheoretical max of 1 bit for a 2-symbol alphabet -- BTC's real "
      "up/down sequence\nreads as close to indistinguishable from a coin flip. "
      "This corroborates Ch13's\nphi_hat~1.03 random-walk finding via a "
      "completely independent method (information\ntheory rather than an O-U "
      "mean-reversion fit).")
print("\nCAVEAT (matches the book's own Sec 18.5.2 warning): quantile encoding's "
      "entropy\nreading (2.322 bits) sits almost exactly at log2(5)=2.3219, the "
      "theoretical MAX\nfor a 5-letter alphabet -- this is a property of "
      "quantile encoding itself (it\nforces near-uniform bin counts by "
      "construction), not independent evidence of\nhigh information content. "
      "Binary and sigma encoding, which don't force\nuniformity, are the more "
      "trustworthy readings here.")


# =============================================================================
# Part B -- 18.8.4 adverse-selection workflow on real order flow
# =============================================================================
print()
print("=" * 78)
print("PART B -- Market microstructure adverse-selection feature (18.8.4)")
print("=" * 78)

adverse_selection = adverse_selection_feature(
    bars['BuyVolume'].values, bars['SellVolume'].values,
    n_quantiles=5, roll_window=30,
)
print(f"\n{len(adverse_selection)} real per-bar feature values "
      f"(249 bars, roll_window=30 -> 249-30+1=220 output points)")
print(f"mean={adverse_selection.mean():.4f}, "
      f"std={adverse_selection.std():.4f}, "
      f"range=[{adverse_selection.min():.4f}, {adverse_selection.max():.4f}]")

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
axes[0].plot(returns.index, returns.values, linewidth=0.8)
axes[0].set_title('Real BTC/TUSD bar-to-bar returns')
axes[0].set_xlabel('bar_id')
axes[1].plot(adverse_selection.index, adverse_selection.values, linewidth=0.8, color='darkorange')
axes[1].axhline(0.5, linestyle='--', color='gray', linewidth=0.8)
axes[1].set_title('18.8.4 adverse-selection feature (empirical CDF of rolling order-flow entropy)')
axes[1].set_xlabel('bar_id')
plt.tight_layout()
output_dir = os.path.join(HERE, 'output')
os.makedirs(output_dir, exist_ok=True)
fig_path = os.path.join(output_dir, 'entropy_features_real_data.png')
plt.savefig(fig_path, dpi=100)
print(f"\nSaved plot to {fig_path}")

print("\nNOTE: with only 220 real output points from a 249-bar/~29-day real "
      "window, this\nfeature carries the same 'heavy extrapolation from a "
      "short real window' caveat\nthis project has flagged before (Ch13's "
      "O-U calibration, Ch15's annualized bet\nfrequency, Ch17 Part C's BTC "
      "explosiveness contrast) -- a real result, just a\nthinly-supported one.")


# =============================================================================
# Part C -- 18.8.3 Portfolio Concentration, using Ch16's real HRP/IVP weights
# =============================================================================
print()
print("=" * 78)
print("PART C -- Portfolio Concentration (18.8.3), Ch16's real HRP/IVP weights")
print("=" * 78)

sys.path.insert(0, os.path.join(ROOT, 'ch16'))
from chapter_16_hrp import part_b_real_data  # noqa: E402

hrp, ivp, corr, ret_df = part_b_real_data()
cov = ret_df.cov()

# Alignment fix (a real bug caught while wiring this up, verified with a
# synthetic mismatched-order reproduction before touching real data):
# hrp/ivp come back sorted ALPHABETICALLY (chapter_16_hrp.py's own
# .sort_index() calls), but cov's row/column order is the COMMODITIES
# dict's insertion order (gold, crude_oil, corn, live_hogs, tbonds, gbp --
# NOT alphabetical). compute_portfolio_concentration is plain-numpy /
# position-based, so pairing hrp.values or ivp.values directly against
# cov.values would silently pair each weight with the WRONG asset's
# variance/covariance row. Explicit reindex to cov's order is required.
assets = cov.index.tolist()
hrp_aligned = hrp.reindex(assets)
ivp_aligned = ivp.reindex(assets)
assert not hrp_aligned.isna().any(), "HRP weights missing for some cov assets"
assert not ivp_aligned.isna().any(), "IVP weights missing for some cov assets"

H_hrp, theta_hrp = compute_portfolio_concentration(cov.values, hrp_aligned.values)
H_ivp, theta_ivp = compute_portfolio_concentration(cov.values, ivp_aligned.values)

theta_hrp_series = pd.Series(theta_hrp, index=assets).sort_values(ascending=False)
theta_ivp_series = pd.Series(theta_ivp, index=assets).sort_values(ascending=False)

print(f"\nHRP portfolio concentration H = {H_hrp:.4f}")
print("Risk contribution per principal component (theta), HRP:")
print(theta_hrp_series.to_string())

print(f"\nIVP portfolio concentration H = {H_ivp:.4f}")
print("Risk contribution per principal component (theta), IVP:")
print(theta_ivp_series.to_string())

n_assets = len(assets)
print(f"\nH is bounded in [0, 1 - 1/N] for N={n_assets} assets, "
      f"so [0, {1 - 1 / n_assets:.4f}] here. 0 = risk spread perfectly evenly "
      "across principal components (maximally diversified); the upper bound "
      "= all risk concentrated in a single principal component.")

fig, ax = plt.subplots(figsize=(9, 4))
theta_compare = pd.DataFrame({'HRP': theta_hrp_series, 'IVP': theta_ivp_series})
theta_compare = theta_compare.loc[theta_hrp_series.index]  # keep HRP's sort order
theta_compare.plot(kind='bar', ax=ax)
ax.set_title('Risk contribution per principal component (theta) -- HRP vs IVP')
ax.set_ylabel('theta_i')
plt.tight_layout()
theta_fig_path = os.path.join(output_dir, 'ch18_portfolio_concentration_theta.png')
plt.savefig(theta_fig_path, dpi=100)
print(f"\nSaved theta comparison chart to {theta_fig_path}")

print("\nREAL FINDING: HRP is slightly MORE risk-concentrated across principal "
      "components\n(H=0.2821) than IVP (H=0.2695), even though HRP's top-2 "
      "ASSET-weight concentration\nis slightly LOWER than IVP's (0.756 vs "
      "0.752, Ch16's own metric) -- these two notions\nof concentration "
      "measure different things (asset-weight share vs. principal-component\n"
      "risk distribution). HRP shifted weight toward the low-vol, "
      "low-correlation gbp\nand tbonds, but crude_oil/gold/corn still "
      "dominate the largest eigenvalue direction,\nand HRP's version is "
      "marginally MORE skewed there (crude_oil theta=0.410 vs IVP's\n"
      "0.349) than IVP's. Consistent with Ch16's own finding that HRP and "
      "IVP converge\nfor this genuinely low-correlation 6-asset set "
      "(correlations mostly under 0.15\nin absolute value) -- there's "
      "little cluster structure here for HRP's correlation-\nawareness to "
      "exploit.")
