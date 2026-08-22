"""
generate_synthetic_trades.py

Purpose
-------
Synthetic BTC raw-trade generator for the order-flow-imbalance
edge-detection harness (design agreed 2026-08-21, build started
2026-08-22). Produces a fake raw-trade tape (Price/Volume/Timestamp/
IsBuyerMaker) with realistic baseline statistics -- calibrated against
the real March 2026 static CSV via calibrate_synthetic_trade_params.py --
plus an OPTIONAL, TUNABLE injected edge: buy/sell imbalance in window i
is correlated with price drift in window i+1, at a magnitude controlled
by a single `edge_strength` dial. At edge_strength=0, there is no
injected signal (null case).

Design intent: the edge lives at the ORDER-FLOW level, not the label
level. This is deliberate -- injecting the answer directly into
triple-barrier labels would make any downstream detection trivial and
meaningless. By injecting it into raw buy/sell flags and letting price
respond one window later, the real pipeline (dollar bars -> CUSUM ->
triple-barrier labeling -> Ch19 microstructural features -> Ch11 model
grid -> Ch14 DSR / Ch11 PBO) has to actually surface the signal through
every real stage, the same way it would have to surface a real edge.

Calibration source
-------------------
Baseline stats below are the REAL values measured by
calibrate_synthetic_trade_params.py against the March 2026 BTC/TUSD
static CSV (9,205 trades, run 2026-08-22):

    n_trades              = 9,205
    span_hours            = 743.4286968438888
    price_diff_std         = 111.05121984716983   (real trade-to-trade $ jitter)
    avg_trade_size         = 0.0042597099402498645 (BTC)
    baseline_imbalance     = 0.49788158609451383   (fraction IsBuyerMaker==True)
    price_start            = 67068.88

LOAD-BEARING (2026-08-22): tick arrival rate. The static CSV's MEDIAN
inter-trade gap (66.05 sec) implies a rate of 0.01514 trades/sec, but
the file's actual trade count over its actual span implies a much lower
average rate: 9205 trades / (743.4286968438888 * 3600 sec) =
0.0034391 trades/sec -- a ~4.4x discrepancy. This means real trade
arrivals are BURSTY (short gaps clustered, punctuated by long quiet
stretches), not a steady stream. Using the median-based rate as a
uniform arrival rate would generate ~4x too many trades for a given
span. DECISION: use the honest average rate (trade count / span) as a
homogeneous Poisson arrival rate. This reproduces the real trade COUNT
over a real SPAN but does NOT reproduce real burst clustering -- a known,
accepted fidelity gap in this harness, not an oversight. Revisit with a
Hawkes-process-style bursty arrival model only if burst structure turns
out to matter for detection results.

LOAD-BEARING (2026-08-22): trade-size distribution. Real trade sizes are
almost certainly heavy-tailed (many small trades, a few large ones) --
the calibration script only captured the MEAN (avg_trade_size), not the
full distribution shape. This generator uses an exponential distribution
matched to that mean (simplest right-skewed distribution with a single
free parameter). This preserves the mean and qualitative right-skew but
not real tail heaviness. Accepted simplification for a harness focused
on order-flow imbalance, not trade-size fidelity.

LOAD-BEARING (2026-08-22): edge mechanism parameterization. `edge_strength`
is a SINGLE dial, in probability units -- it is the standard deviation of
each window's imbalance tilt around baseline_imbalance (so edge_strength=0.1
means windows' buy-probability wobbles by roughly +/-0.1 around baseline,
before random per-trade Bernoulli noise on top of that). The price-drift
response to that imbalance is controlled by a SEPARATE parameter,
`drift_dollars_per_unit_imbalance`, which auto-scales by default to
price_diff_std * sqrt(trades_per_window) -- the natural noise scale of one
window's cumulative random walk -- so that edge_strength's sweep moves
through a "buried in noise" to "easily visible" range rather than being
arbitrary. Only edge_strength is meant to be the swept dial in the
harness; drift_dollars_per_unit_imbalance normally stays at its default.
"""

import numpy as np
import pandas as pd

# Real calibrated baseline stats -- see calibrate_synthetic_trade_params.py,
# run against the March 2026 static CSV, 2026-08-22.
CALIBRATED_PRICE_DIFF_STD = 111.05121984716983
CALIBRATED_AVG_TRADE_SIZE = 0.0042597099402498645
CALIBRATED_BASELINE_IMBALANCE = 0.49788158609451383
CALIBRATED_PRICE_START = 67068.88
CALIBRATED_AVG_TRADE_RATE_PER_SEC = 9205 / (743.4286968438888 * 3600)  # 0.0034391...
CALIBRATED_START_TIMESTAMP = pd.Timestamp('2026-03-01')


def generate_synthetic_trades(
    n_windows: int,
    trades_per_window: int,
    edge_strength: float,
    drift_dollars_per_unit_imbalance: float = None,
    baseline_imbalance: float = CALIBRATED_BASELINE_IMBALANCE,
    price_diff_std: float = CALIBRATED_PRICE_DIFF_STD,
    avg_trade_rate_per_sec: float = CALIBRATED_AVG_TRADE_RATE_PER_SEC,
    avg_trade_size: float = CALIBRATED_AVG_TRADE_SIZE,
    start_price: float = CALIBRATED_PRICE_START,
    start_timestamp: pd.Timestamp = CALIBRATED_START_TIMESTAMP,
    seed: int = None,
    return_diagnostics: bool = False,
):
    """
    Generate a synthetic raw-trade tape with an optional injected
    order-flow-imbalance -> next-window-drift edge.

    Parameters
    ----------
    n_windows : int
        Number of consecutive trade windows to generate.
    trades_per_window : int
        Trades per window (constant across windows for simplicity).
    edge_strength : float
        Standard deviation, in probability units, of each window's
        imbalance tilt around `baseline_imbalance`. 0.0 = null case,
        no injected signal.
    drift_dollars_per_unit_imbalance : float, optional
        Dollars of NEXT-window cumulative drift injected per unit of
        (window i's realized imbalance - baseline_imbalance). If None,
        auto-scaled to price_diff_std * sqrt(trades_per_window).
    seed : int, optional
        RNG seed. None = nondeterministic.
    return_diagnostics : bool
        If True, also return a dict with per-window realized imbalance
        and per-window mean price (for TDD verification / the sweep
        harness's own edge-strength confirmation step -- NOT meant to be
        fed to the real pipeline).

    Returns
    -------
    pd.DataFrame with columns ['Price', 'Volume', 'Timestamp', 'IsBuyerMaker']
    (optionally, dict of diagnostics if return_diagnostics=True)
    """
    rng = np.random.default_rng(seed)
    total_trades = n_windows * trades_per_window

    if drift_dollars_per_unit_imbalance is None:
        drift_dollars_per_unit_imbalance = price_diff_std * np.sqrt(trades_per_window)

    # --- Timestamps: homogeneous Poisson arrivals at the calibrated
    # average rate (see module-level LOAD-BEARING note on burstiness). ---
    inter_arrival_sec = rng.exponential(
        scale=1.0 / avg_trade_rate_per_sec, size=total_trades - 1
    )
    elapsed_sec = np.concatenate([[0.0], np.cumsum(inter_arrival_sec)])
    timestamps = start_timestamp + pd.to_timedelta(elapsed_sec, unit='s')

    # --- Per-window imbalance tilt (the injected "signal"). ---
    z = rng.normal(0.0, 1.0, size=n_windows)
    tilt = edge_strength * z
    p_buy = np.clip(baseline_imbalance + tilt, 0.02, 0.98)

    is_buyer_maker = np.empty(total_trades, dtype=bool)
    realized_imbalance = np.empty(n_windows)
    for i in range(n_windows):
        lo, hi = i * trades_per_window, (i + 1) * trades_per_window
        window_draws = rng.random(trades_per_window) < p_buy[i]
        is_buyer_maker[lo:hi] = window_draws
        realized_imbalance[i] = window_draws.mean()

    # --- Price path: baseline random-walk noise on every trade, PLUS
    # (for window i >= 1) a drift term derived from window i-1's
    # REALIZED imbalance -- not the target z -- since realized imbalance
    # is what a real feature extractor would actually observe. ---
    # LOAD-BEARING (2026-08-22), BUG FOUND AND FIXED same day: this drift
    # term MUST be tied to edge_strength * z[i-1] (the underlying injected
    # signal), NOT to (realized_imbalance[i-1] - baseline_imbalance).
    # realized_imbalance is a noisy finite-sample estimate -- it fluctuates
    # around baseline_imbalance from pure Bernoulli sampling noise even
    # when edge_strength=0 and NOTHING was intentionally tilted. An
    # earlier version of this function used realized_imbalance directly,
    # which meant drift_dollars_per_unit_imbalance (nonzero by default)
    # injected a real, nonzero, mechanical drift response to sampling
    # noise even at edge_strength=0 -- confirmed via hand-tracing:
    # null-case lag-1 correlation between window imbalance and next-window
    # drift averaged 0.0405 +/- SE 0.0032 across 200 seeds (12.6 SEs from
    # zero), i.e. the "null case" was not actually null. Tying drift to
    # edge_strength * z[i-1] instead makes drift EXACTLY zero for every
    # window whenever edge_strength=0, regardless of any realized-sample
    # noise -- eliminating the whole bug class. realized_imbalance[i] is
    # still genuinely correlated with drift_next[i] at edge_strength>0,
    # since both depend on the same underlying z[i]; that's the actually
    # intended relationship (a real feature detector only ever sees
    # realized_imbalance, never z directly).
    per_trade_noise = rng.normal(0.0, price_diff_std, size=total_trades)
    per_trade_drift = np.zeros(total_trades)
    for i in range(1, n_windows):
        lo, hi = i * trades_per_window, (i + 1) * trades_per_window
        window_drift_total = edge_strength * drift_dollars_per_unit_imbalance * z[i - 1]
        per_trade_drift[lo:hi] = window_drift_total / trades_per_window

    increments = per_trade_noise + per_trade_drift
    prices = start_price + np.cumsum(increments)
    # First trade's price should just be start_price + first increment,
    # which np.cumsum already gives correctly (no off-by-one: increments[0]
    # is trade 0's own noise, prices[0] = start_price + increments[0]).

    volumes = rng.exponential(scale=avg_trade_size, size=total_trades)

    df = pd.DataFrame({
        'Price': prices,
        'Volume': volumes,
        'Timestamp': timestamps,
        'IsBuyerMaker': is_buyer_maker,
    })

    if return_diagnostics:
        window_mean_price = np.array([
            prices[i * trades_per_window:(i + 1) * trades_per_window].mean()
            for i in range(n_windows)
        ])
        diagnostics = {
            'realized_imbalance': realized_imbalance,
            'window_mean_price': window_mean_price,
            'p_buy_target': p_buy,
        }
        return df, diagnostics

    return df


# =======================================================================
# REAL-MACHINE PYTEST CONFIRMATION (mlfinlab conda env)
# =======================================================================
# See test_generate_synthetic_trades.py for full embedded pytest output.
# Pass 1 (repo root), 2026-08-22: 7 passed in 3.54s, Python 3.10.20.
# Pass 2 (inside pipeline/edge_harness/): 7 passed in 3.54s, 2026-08-22,
# identical to pass 1 -- two-pass real-machine confirmation complete.
# =======================================================================
