"""
pipeline/edge_harness/generate_bar_aligned_meanshift_trades.py

THIRD edge-injection mechanism, built 2026-08-27 in direct response to a
real gap identified while explaining the two existing null results to
Ethan: every full-real-pipeline detection test run so far (OFI's
bar_aligned_scaled_50seeds.csv, momentum's momentum_edge_sweep_50seeds.csv)
has used an edge mechanism that turned out to carry its own confound --

  - OFI (generate_bar_aligned_trades.py): the injected edge only reaches
    the classifier THROUGH an engineered buy/sell-imbalance feature.
    trace_ofi_signal_leakage.py's real fold-level trace found severe,
    inconsistent class imbalance across PurgedKFold folds (trivial-
    baseline accuracy 0.36-0.83, not near the clean 0.5 that would
    signal "folds are fine, non-detection is real") -- a genuinely
    different, still-not-fully-diagnosed artifact from momentum's.
  - Momentum (existing momentum generator): the injected edge uses a
    continuation_prob MARKOV CHAIN, which produces long, sustained
    single-direction regimes. trace_momentum_signal_leakage.py's real
    fold trace found this collides with PurgedKFold's chronological
    split -- trivial-baseline OOS accuracy averaging 0.33, well below
    chance, REGARDLESS of whether a real edge exists. Confirmed root
    cause, not a real non-detection finding.

Neither test is a clean positive control for "can the full pipeline
detect ANY edge at all." This module is that clean control:

  - The signal is PURE SINGLE-LAG PRICE-RETURN MOMENTUM (an AR(1)-style
    process, NOT a Markov continuation chain): window i's own realized
    return partially carries into window i+1's drift, with NO further
    memory beyond that one lag. This is visible via the most generic,
    universally-available price-based feature there is (a window's own
    realized return / fracdiff'd price series) -- it does NOT rely on
    any engineered feature like OFI's buy/sell-imbalance ratio.
  - Because it's a single, geometrically-non-persistent lag rather than
    a discrete regime-continuation chain, it should NOT reproduce
    momentum's long-sustained-regime / chronological-CV collision.
  - IsBuyerMaker is drawn at PLAIN baseline_imbalance with NO tilt at
    all (fully decoupled from the injected signal) -- this deliberately
    rules out any OFI-feature involvement, isolating this as a pure
    price-return signal.

Reuses generate_bar_aligned_trades.py's real, already-tested two-pass
windowing machinery (_find_bar_windows) UNMODIFIED -- same
chicken-and-egg resolution (null-edge scaffold -> real dollar-bar
boundaries via rebuild.compute_dynamic_threshold/
features._retag_trades_with_bar_id, reused not reimplemented), same
scaffold/injection RNG-independence discipline (seed for pass 1,
seed+1 for pass 2).

INJECTION MECHANISM (pass 2)
-----------------------------
For each bar window i, draw a fresh, independent z[i] ~ N(0,1) (no
persistence across windows -- this is the load-bearing difference from
momentum's Markov chain). Define this window's own drift contribution:

    d[i] = edge_strength * drift_dollars_per_unit_z * z[i]

Window i's TOTAL injected drift is d[i] (its own new contribution) PLUS
d[i-1] (a one-window carry-over of the PREVIOUS window's own
contribution):

    total_drift[i] = d[i] + d[i-1]      (d[-1] := 0 for the first window)

This means window i's own realized return reflects d[i] (making it a
real, observable "this window's return"), and window i+1's realized
return reflects that SAME d[i] via the carry term -- so "this window's
own realized return" is genuinely predictive of "next window's return,"
one lag only, with no discrete regime-persistence mechanism. This is
the same lag-1, per-window-independent-innovation STRUCTURE as the OFI
generator's own price-drift code (window i+1's drift tied to window i's
z), just applied directly to price/return instead of routed through a
synthetic imbalance feature.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python -c "from pipeline.edge_harness.generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades; print('import OK')"

Not run standalone -- called by run_meanshift_edge_sweep.py, same as
generate_bar_aligned_trades.py is called by run_bar_aligned_edge_sweep.py.
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
    CALIBRATED_PRICE_DIFF_STD, CALIBRATED_AVG_TRADE_SIZE,
    CALIBRATED_BASELINE_IMBALANCE, CALIBRATED_PRICE_START,
    CALIBRATED_AVG_TRADE_RATE_PER_SEC, CALIBRATED_START_TIMESTAMP,
)
from synthetic_trade_adapter import synthetic_to_raw_trades_schema  # noqa: E402
# REUSED UNMODIFIED from the OFI generator -- same scaffold/windowing
# logic, edge-mechanism-agnostic (it just needs a null-edge scaffold).
from generate_bar_aligned_trades import _find_bar_windows            # noqa: E402


def generate_bar_aligned_meanshift_trades(
    n_trades,
    target_bars,
    edge_strength,
    drift_dollars_per_unit_z=None,
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
    Clean single-lag price-return-momentum counterpart to
    generate_bar_aligned_synthetic_trades() (OFI) and the Markov-chain
    momentum generator. See module docstring for the injection
    mechanism and why it's a genuinely different, cleaner test than
    either existing one.

    Parameters mirror generate_bar_aligned_synthetic_trades() exactly
    except drift_dollars_per_unit_imbalance -> drift_dollars_per_unit_z,
    and there is no baseline_imbalance TILT anywhere (IsBuyerMaker is
    plain baseline draws, fully decoupled from the injected signal).

    Returns
    -------
    pd.DataFrame in the same raw_trades schema as
    synthetic_to_raw_trades_schema() produces. Optionally also a
    diagnostics dict if return_diagnostics=True.
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
    if n_windows < 3:
        raise ValueError(
            f'Only {n_windows} complete bar(s) found in the pass-1 '
            f'scaffold (n_trades={n_trades}, target_bars={target_bars}) '
            '-- need at least 3 to observe a lag-1 own-return -> '
            'next-window-drift relationship (window 0 has no carry-in, '
            'so at least 2 more windows are needed to see the effect). '
            'Increase n_trades or lower target_bars.'
        )
    n_used_trades = windows[-1][1]

    if drift_dollars_per_unit_z is None:
        avg_window_size = n_used_trades / n_windows
        drift_dollars_per_unit_z = price_diff_std * np.sqrt(avg_window_size)

    # Pass 2: independent RNG (seed+1), same discipline as the OFI
    # generator -- avoids any accidental correlation with the pass-1
    # scaffold draw.
    pass2_seed = None if seed is None else seed + 1
    rng = np.random.default_rng(pass2_seed)

    timestamps = scaffold_raw['Timestamp'].values[:n_used_trades]
    volumes = scaffold_raw['Volume'].values[:n_used_trades]

    # IsBuyerMaker: PLAIN baseline draws, NO tilt, fully decoupled from
    # z -- deliberately rules out any OFI-feature involvement (see
    # module docstring).
    is_buyer_maker = rng.random(n_used_trades) < baseline_imbalance

    # z[i]: fresh, independent innovation per window -- NO persistence
    # mechanism beyond the single explicit one-window carry below. This
    # is the load-bearing structural difference from momentum's Markov
    # continuation_prob chain.
    z = rng.normal(0.0, 1.0, size=n_windows)
    d = edge_strength * drift_dollars_per_unit_z * z  # each window's OWN drift contribution

    per_trade_noise = rng.normal(0.0, price_diff_std, size=n_used_trades)
    per_trade_drift = np.zeros(n_used_trades)
    window_total_drift = np.zeros(n_windows)
    for i, (lo, hi) in enumerate(windows):
        w = hi - lo
        carry_in = d[i - 1] if i > 0 else 0.0
        total = d[i] + carry_in
        window_total_drift[i] = total
        per_trade_drift[lo:hi] = total / w

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
        # Realized per-window return (from the actual price path,
        # INCLUDING noise) -- this is what a real "last bar's own
        # return" feature would compute. The lag-1 autocorrelation of
        # THIS series is the natural detectability diagnostic here,
        # parallel to OFI's corr(realized_imbalance, next_drift).
        window_returns = np.array([
            prices[hi - 1] - prices[lo] for lo, hi in windows
        ])
        lag1_autocorr = (
            float(np.corrcoef(window_returns[:-1], window_returns[1:])[0, 1])
            if n_windows > 2 else float('nan')
        )
        diagnostics = {
            'window_mean_price': window_mean_price,
            'window_returns': window_returns,
            'window_total_drift': window_total_drift,
            'lag1_autocorr': lag1_autocorr,
            'window_sizes': np.array([hi - lo for lo, hi in windows]),
            'n_windows': n_windows,
            'threshold': threshold,
            'n_used_trades': n_used_trades,
        }
        return raw_trades, diagnostics

    return raw_trades
