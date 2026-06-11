# multi_product/__init__.py
# This file marks the `multi_product` directory as a Python PACKAGE
# and defines its public API — the three functions exposed to callers
# who write `import multi_product as mp` or `from multi_product import ...`.
#
# --- What lives in this package? ---
# The multi_product package handles everything related to combining or
# adjusting MULTIPLE instruments into a single tradeable series.
# All three tools below are from AFML Chapter 2, Section 2.4.

# --- ETF Trick (etf_trick.py) — Section 2.4.1 ---
# etf_trick : Takes a basket of instruments (e.g. three S&P 500 futures contracts)
#             and converts them into a single continuous $1 investment series K_t.
#             Solves three problems with naive price-stitching:
#               1. Different units / point values across instruments
#               2. Artificial roll gaps when switching contracts
#               3. Potential negative prices after backward adjustment
from .etf_trick import etf_trick

# --- PCA Weights (pca_weights.py) — Section 2.4.2 ---
# pca_weights : Given a covariance matrix of instrument returns, computes
#               portfolio allocation weights ω such that risk is spread across
#               principal components according to a target distribution.
#               Prevents apparent diversification that is actually concentrated
#               in one correlated "market direction".
from .pca_weights import pca_weights

# --- Roll gap correction (roll.py) — Section 2.4.3 ---
# roll_gaps               : Computes the cumulative price gaps introduced each time
#                           a futures contract expires and rolls to the next contract.
# get_rolled_series       : Subtracts the cumulative gaps from raw prices, producing
#                           a smooth continuous series with no artificial jumps.
# non_negative_rolled_prices : Further converts the rolled series into a $1 investment
#                           series via cumulative returns, guaranteeing positivity
#                           even when raw adjusted prices would go negative.
from .roll import roll_gaps, get_rolled_series, non_negative_rolled_prices
