"""
generate_bar_aligned_trades.py

generate_synthetic_trades.py injects the order-flow-imbalance edge over
FIXED TRADE-COUNT windows (e.g. 300 trades). The real pipeline aggregates
by DOLLAR-VOLUME bars (dynamic threshold, ~120 trades/bar on average but
genuinely variable), and Ch19's features (BuyVolume/SellVolume ratios,
signed-flow serial correlation, tick-rule accuracy, VPIN, etc.) are
computed PER BAR, not per fixed-trade-count window. The 2026-08-22 sweep
found the pipeline shows essentially zero DSR/PBO response to the
injected edge even at saturating strength (edge_strength=1.0-2.0) --
this module tests the leading hypothesis that the injection GRANULARITY
mismatch (trade-count windows vs. dollar bars) is why: if the edge is
injected at the SAME unit the features actually see, does detection
improve?

*** LOAD-BEARING (2026-08-22): two-pass generation, chicken-and-egg
problem ***
Dollar-bar boundaries are determined by cumulative Price*Volume, which
depends on the price path we want to inject drift into -- can't know bar
boundaries before generating prices, can't inject bar-aligned drift
before knowing bar boundaries. Resolved via two passes:
  Pass 1 (scaffold): generate a NULL-EDGE tape (pure random walk, real
    timestamps/volumes), then run it through the REAL pipeline's own
    rebuild.compute_dynamic_threshold() and features._retag_trades_
    with_bar_id() (imported and reused directly, NOT reimplemented) to
    discover real bar boundaries.
  Pass 2 (injection): reuse pass 1's Timestamp/Volume UNCHANGED (they
    don't depend on price/imbalance at all), regenerate Price/
    IsBuyerMaker with the edge injected using pass 1's bar boundaries as
    variable-length windows (bar i's realized imbalance -> bar i+1's
    price drift), via a NEW independent RNG draw (seed+1, not the same
    seed pass 1 used, to avoid any accidental correlation between the
    scaffold draw and the injection draw).

DOCUMENTED APPROXIMATION: pass 2's injected drift shifts prices slightly
from pass 1's scaffold, so pass 1's bar boundaries are not EXACTLY what
the real pipeline will recompute when it later re-tags pass 2's final
(edge-injected) trades. Accepted because drift magnitudes are tiny
relative to BTC's price level (tens of dollars against a $65k-77k price
level, confirmed via the live calibration pull) -- the resulting
boundary drift should be at most a trade or two per bar, not a
systematic distortion. Not verified to be negligible by direct
measurement in this module; if bar-count mismatches between pass 1's
scaffold and the real pipeline's eventual re-tagging of the final trades
turn out to be large, this approximation needs revisiting.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, HERE)

from generate_synthetic_trades import (                          # noqa: E402
    generate_synthetic_trades,
    CALIBRATED_PRICE_DIFF_STD, CALIBRATED_AVG_TRADE_SIZE,
    CALIBRATED_BASELINE_IMBALANCE, CALIBRATED_PRICE_START,
    CALIBRATED_AVG_TRADE_RATE_PER_SEC, CALIBRATED_START_TIMESTAMP,
)
from synthetic_trade_adapter import synthetic_to_raw_trades_schema  # noqa: E402
from rebuild import compute_dynamic_threshold                       # noqa: E402
from features import _retag_trades_with_bar_id                      # noqa: E402


def _find_bar_windows(n_trades, target_bars, calib_kwargs, seed):
    """Pass 1: generate a null-edge scaffold tape and discover real
    dollar-bar boundaries via the actual pipeline's own threshold and
    tagging logic (reused, not reimplemented).

    Returns
    -------
    windows : list of (start_idx, end_idx) exclusive index ranges into
        the scaffold's trade array, one per COMPLETE bar (trailing
        incomplete bar excluded, matching _retag_trades_with_bar_id's
        own convention).
    scaffold_raw : pd.DataFrame, the scaffold's raw_trades (schema-
        converted), for its Timestamp/Volume columns to be reused in
        pass 2.
    threshold : float, the dynamic dollar-bar threshold found (for
        diagnostics/logging).
    """
    scaffold_df = generate_synthetic_trades(
        n_windows=1, trades_per_window=n_trades, edge_strength=0.0,
        seed=seed, **calib_kwargs,
    )
    scaffold_raw = synthetic_to_raw_trades_schema(scaffold_df)
    threshold = compute_dynamic_threshold(scaffold_raw, target_bars=target_bars)
    tagged = _retag_trades_with_bar_id(scaffold_raw, threshold)

    windows = []
    for bid, grp in tagged.groupby('bar_id'):
        idx = grp.index.values
        # LOAD-BEARING (2026-08-22): assumes each bar's trade indices are
        # contiguous, which holds because _retag_trades_with_bar_id
        # assigns bar_id via a strictly sequential accumulate-and-reset
        # scan over already time-ordered trades -- no interleaving is
        # possible. Asserted, not just assumed, since a violation here
        # would silently corrupt every downstream window boundary.
        assert idx.max() - idx.min() + 1 == len(idx), (
            f'bar_id={bid} has non-contiguous trade indices -- '
            'this should be impossible given _retag_trades_with_bar_id\'s '
            'sequential scan; investigate before trusting window boundaries.'
        )
        windows.append((int(idx.min()), int(idx.max()) + 1))

    return windows, scaffold_raw, threshold


def generate_bar_aligned_synthetic_trades(
    n_trades,
    target_bars,
    edge_strength,
    drift_dollars_per_unit_imbalance=None,
    baseline_imbalance=CALIBRATED_BASELINE_IMBALANCE,
    price_diff_std=CALIBRATED_PRICE_DIFF_STD,
    avg_trade_rate_per_sec=CALIBRATED_AVG_TRADE_RATE_PER_SEC,
    avg_trade_size=CALIBRATED_AVG_TRADE_SIZE,
    start_price=CALIBRATED_PRICE_START,
    start_timestamp=CALIBRATED_START_TIMESTAMP,
    seed=None,
    return_diagnostics=False,
):
    """
    Bar-aligned counterpart to generate_synthetic_trades(): injects the
    order-flow-imbalance edge using REAL dollar-bar boundaries (discovered
    via a null-edge scaffold pass through the actual pipeline's own
    threshold/tagging logic) as windows, instead of fixed trade counts.

    Parameters mirror generate_synthetic_trades() except n_windows/
    trades_per_window are replaced by n_trades (total trade count to
    generate) and target_bars (passed through to
    rebuild.compute_dynamic_threshold()).

    Returns
    -------
    pd.DataFrame in the same raw_trades schema as
    synthetic_to_raw_trades_schema() produces, truncated to only the
    trades that fell within a COMPLETE bar in the pass-1 scaffold
    (matching the real pipeline's own trailing-incomplete-bar handling).
    Optionally also a diagnostics dict if return_diagnostics=True.
    """
    calib_kwargs = dict(
        baseline_imbalance=baseline_imbalance,
        price_diff_std=price_diff_std,
        avg_trade_rate_per_sec=avg_trade_rate_per_sec,
        avg_trade_size=avg_trade_size,
        start_price=start_price,
        start_timestamp=start_timestamp,
    )

    windows, scaffold_raw, threshold = _find_bar_windows(
        n_trades, target_bars, calib_kwargs, seed,
    )
    n_windows = len(windows)
    if n_windows < 2:
        raise ValueError(
            f'Only {n_windows} complete bar(s) found in the pass-1 '
            f'scaffold (n_trades={n_trades}, target_bars={target_bars}) '
            '-- need at least 2 to inject a lag-1 imbalance/drift edge. '
            'Increase n_trades or lower target_bars.'
        )
    n_used_trades = windows[-1][1]  # end index of the last complete bar

    if drift_dollars_per_unit_imbalance is None:
        # LOAD-BEARING (2026-08-22): auto-scale using the AVERAGE window
        # (bar) size, since bar sizes are now variable, not constant --
        # generate_synthetic_trades()'s own default used
        # sqrt(trades_per_window) directly; here we use the mean bar
        # size as the closest analogue.
        avg_window_size = n_used_trades / n_windows
        drift_dollars_per_unit_imbalance = price_diff_std * np.sqrt(avg_window_size)

    # Pass 2: independent RNG (seed+1, not the same seed pass 1's
    # scaffold used -- see module LOAD-BEARING note) for the actual
    # injected price/imbalance draws.
    pass2_seed = None if seed is None else seed + 1
    rng = np.random.default_rng(pass2_seed)

    timestamps = scaffold_raw['Timestamp'].values[:n_used_trades]
    volumes = scaffold_raw['Volume'].values[:n_used_trades]

    z = rng.normal(0.0, 1.0, size=n_windows)
    tilt = edge_strength * z
    p_buy = np.clip(baseline_imbalance + tilt, 0.02, 0.98)

    is_buyer_maker = np.empty(n_used_trades, dtype=bool)
    realized_imbalance = np.empty(n_windows)
    for i, (lo, hi) in enumerate(windows):
        w = hi - lo
        draws = rng.random(w) < p_buy[i]
        is_buyer_maker[lo:hi] = draws
        realized_imbalance[i] = draws.mean()

    per_trade_noise = rng.normal(0.0, price_diff_std, size=n_used_trades)
    per_trade_drift = np.zeros(n_used_trades)
    for i in range(1, n_windows):
        lo, hi = windows[i]
        w = hi - lo
        # Same fix as generate_synthetic_trades.py's 2026-08-22 bug fix:
        # tied to edge_strength * z[i-1] directly, NOT realized
        # imbalance, so edge_strength=0 gives an exact null case.
        window_drift_total = edge_strength * drift_dollars_per_unit_imbalance * z[i - 1]
        per_trade_drift[lo:hi] = window_drift_total / w

    increments = per_trade_noise + per_trade_drift
    prices = start_price + np.cumsum(increments)

    df = pd.DataFrame({
        'Price': prices,
        'Volume': volumes,
        'Timestamp': pd.to_datetime(timestamps, unit='us'),
        'IsBuyerMaker': is_buyer_maker,
    })
    raw_trades = synthetic_to_raw_trades_schema(df)

    if return_diagnostics:
        window_mean_price = np.array([
            prices[lo:hi].mean() for lo, hi in windows
        ])
        diagnostics = {
            'realized_imbalance': realized_imbalance,
            'window_mean_price': window_mean_price,
            'window_sizes': np.array([hi - lo for lo, hi in windows]),
            'n_windows': n_windows,
            'threshold': threshold,
            'n_used_trades': n_used_trades,
        }
        return raw_trades, diagnostics

    return raw_trades
