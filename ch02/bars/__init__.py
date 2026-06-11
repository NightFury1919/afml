# bars/__init__.py
# This file marks the `bars` directory as a Python PACKAGE.
#
# --- What does __init__.py do? ---
# When Python sees a folder containing __init__.py, it treats that folder as
# a "package" — a namespace from which you can import things.
# Without this file, `import bars` or `from bars import tick_bars` would fail.
#
# This particular __init__.py also acts as a PUBLIC API definition:
# it decides WHICH names from the sub-modules are visible to the outside world
# when someone does `import bars` or `from bars import ...`.
#
# For example, after these imports a caller can write:
#   import bars
#   bars.tick_bars(df, thresh=100)     ← works because tick_bars is re-exported here
#   bars.cusum_filter(df, h=500)       ← works for the same reason
#
# The dot prefix (e.g. `.utils`) means "import from a module inside THIS package"
# (a relative import). This is required inside packages to avoid ambiguity.

# --- Utility functions (utils.py) ---
# ewma                 : Exponentially Weighted Moving Average — used everywhere
#                        to adaptively update bar parameters after each bar closes.
# delta                : Computes Δp_t (price change between consecutive ticks).
# tick_rule            : Labels each trade as buy (+1) or sell (-1) using price direction.
# estimate_buy_sell_probs : Estimates P[buy] and P[sell] from raw tick data.
from .utils import ewma, delta, tick_rule, estimate_buy_sell_probs

# --- Standard bars (standard_bars.py) ---
# Fixed-threshold bars — each bar closes when a simple counter crosses a threshold.
# time_bars   : closes after a fixed calendar period (e.g. daily, weekly)
# tick_bars   : closes after a fixed number of trades
# volume_bars : closes after a fixed total volume has traded
# dollar_bars : closes after a fixed total dollar value has traded
from .standard_bars import time_bars, tick_bars, volume_bars, dollar_bars

# --- Imbalance bars (imbalance_bars.py) ---
# Information-driven bars — close when the net signed imbalance (buys minus sells)
# exceeds an adaptive expected level. Sample more during informed trading.
# tick_imbalance_bars   : imbalance weighted by tick count only (every trade = ±1)
# volume_imbalance_bars : imbalance weighted by trade volume (large trades count more)
from .imbalance_bars import tick_imbalance_bars, volume_imbalance_bars

# --- Run bars (run_bars.py) ---
# Information-driven bars — close when the MAXIMUM of cumulative buy volume or
# cumulative sell volume exceeds an adaptive threshold. Detect one-sided sweeps.
# tick_run_bars   : run measured in tick counts
# volume_run_bars : run measured in trade volume
from .run_bars import tick_run_bars, volume_run_bars

# --- Filters (filters.py) ---
# Event-detection methods that identify when prices have moved significantly.
# cusum_filter : Symmetric CUSUM filter — fires an event when cumulative price
#                drift in either direction exceeds threshold h, then resets.
from .filters import cusum_filter
