"""
pipeline/orchestration/positive_control_data.py

Positive-control validation: generates SYNTHETIC raw trades carrying a
DELIBERATE, engineered momentum edge -- price continues in its current
"regime" direction with a fixed, elevated probability -- so the exact
same unmodified live pipeline (rebuild.py -> features.py ->
live_staging.py -> stages.run_live_trials) can be run against data with a
KNOWN, real edge.

Why this exists (see 2026-08-14 handoff, Part 6): three straight real
BTC/USDT live runs (plus the static baseline) all report "no reliable
edge" (PBO ~0.78-0.83, DSR well below survival). Negative-only testing on
real market data can't distinguish "the market really has no edge here"
from "the pipeline silently suppresses any signal, real or not" -- those
look identical from the outside. This generator produces data where an
edge is KNOWN to exist by construction, so a low-PBO/high-DSR result on
THIS data is real evidence the machinery can detect an edge when one
actually exists.

Sanctioned per this project's synthetic-data convention (real-data-first
policy: synthetic is acceptable for TDD/validation with known expected
behavior, not for teaching-chapter examples).

*** LOAD-BEARING (2026-08-15): Volume is held CONSTANT across all
synthetic trades, not randomized ***
This positive control is designed to test ONE thing: does the pipeline
detect a genuine price-momentum edge when it exists? Randomizing volume
would introduce a second, uncontrolled source of variation into Ch19's
volume-sensitive features (Amihud lambda, Kyle's lambda, VPIN, dollar-bar
boundaries themselves) that could either mask or fake a signal for
reasons having nothing to do with the momentum edge under test. Holding
volume constant removes that confound. This also keeps every generation
step exactly hand-traceable for the TDD suite
(test_positive_control_data.py).

*** LOAD-BEARING (2026-08-15): schema matches ingestion.py's
RAW_TRADE_COLUMNS exactly, so rebuild.py/features.py run UNMODIFIED ***
The whole point of this generator is that the downstream pipeline never
knows the trades are synthetic -- rebuild.build_bars_and_labels() and
features.build_enriched_events() are called exactly as
run_pipeline_live.py calls them on a real Binance pull. Only
ingestion.pull_recent_trades() itself is swapped out. See
run_pipeline_positive_control.py.

*** LOAD-BEARING (2026-08-15): calibration defaults chosen relative to
this project's REAL constants, not invented independently ***
start_price=65,000 and the tick_bp/noise_std/regime_length_range defaults
are chosen so that a ~24-trade bar (this project's usual ~250-bar
density) produces bar-to-bar dollar moves in the same order of magnitude
as rebuild.py's flat CUSUM_H=500 threshold, and so regimes typically
persist longer than rebuild.py's 3-day vertical barrier window (most
individual triple-barrier events should resolve within a single dominant
regime). These are a reasoned starting point, NOT re-derived from first
principles -- validate empirically on the first real run (see the driver
script's dry-run event-count / realized-correlation checks) and retune if
the resulting bar/event counts look degenerate (e.g. CUSUM firing on
almost every bar, or almost never).
"""
import time

import numpy as np
import pandas as pd

RAW_TRADE_COLUMNS = [
    'TradeID', 'Price', 'Volume', 'QuoteVolume', 'Timestamp',
    'IsBuyerMaker', 'IsBestMatch',
]


def _as_generator(random_state):
    """Project convention: a shared np.random.Generator created once and
    threaded through, not re-seeded piecemeal. Accepts an existing
    Generator (passed through unchanged), an int seed, or None (fresh,
    unseeded Generator)."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _build_regime_sequence(n_trades, regime_lengths, regime_directions):
    """Pure, deterministic: expands (regime_lengths, regime_directions)
    into a length-n_trades array of the TARGET direction (+1/-1) for each
    tick. Trims to n_trades if the regimes run longer; raises if they run
    short (caller's responsibility to supply enough).

    Example: regime_lengths=[3, 2], regime_directions=[1, -1] ->
    [1, 1, 1, -1, -1]
    """
    if len(regime_lengths) != len(regime_directions):
        raise ValueError('regime_lengths and regime_directions must be same length')
    pieces = [
        np.full(length, direction, dtype=int)
        for length, direction in zip(regime_lengths, regime_directions)
    ]
    seq = np.concatenate(pieces) if pieces else np.array([], dtype=int)
    if len(seq) < n_trades:
        raise ValueError(
            f'regimes only cover {len(seq)} ticks, need {n_trades} -- '
            'supply more/longer regimes'
        )
    return seq[:n_trades]


def _generate_tick_directions(target_directions, continuation_prob, rng):
    """For each tick, the REALIZED direction matches target_directions
    with probability continuation_prob, else is flipped. This IS the
    engineered edge: target_directions is the underlying 'regime' (never
    seen by the pipeline), continuation_prob controls how strong the
    momentum signal is (0.5 = no edge / pure coin flip, 1.0 =
    deterministic).

    draws = rng.random(n) is compared against continuation_prob:
    draw < continuation_prob -> keep target direction, else flip. (A draw
    exactly equal to continuation_prob is treated as a flip, matching the
    strict '<'.)
    """
    n = len(target_directions)
    draws = rng.random(n)
    realized = np.where(draws < continuation_prob, target_directions, -np.asarray(target_directions))
    return realized.astype(int)


def _directions_to_prices(tick_directions, start_price, tick_bp, noise):
    """price[i] = price[i-1] * (1 + direction[i]*tick_bp + noise[i]).
    noise is a pre-supplied array the same length as tick_directions (pass
    np.zeros(n) for a fully deterministic path, as the hand-traced tests
    do) -- the real caller draws it from rng.normal() once, matching this
    project's random_state convention of a single Generator threaded
    through, not re-seeded per call.
    """
    n = len(tick_directions)
    prices = np.empty(n, dtype=float)
    price = start_price
    for i in range(n):
        price = price * (1.0 + tick_directions[i] * tick_bp + noise[i])
        prices[i] = price
    return prices


def _directions_to_is_buyer_maker(tick_directions):
    """+1 (buy-initiated, price-up tick) -> IsBuyerMaker=False.
    -1 (sell-initiated, price-down tick) -> IsBuyerMaker=True.
    Matches ingestion.py's real convention (rebuild.py/features.py both
    derive Label = -1 if IsBuyerMaker else 1)."""
    return np.asarray(tick_directions) == -1


def _sample_regimes(n_trades, regime_length_range, rng):
    """Randomly samples (regime_lengths, regime_directions) covering at
    least n_trades ticks. Each regime's length is drawn uniformly from
    regime_length_range (inclusive of both ends); each regime's direction
    is an independent coin flip (+1/-1) -- NOT deterministically
    alternating, so consecutive regimes occasionally repeat direction (a
    real 'trend continues even longer' case is not excluded)."""
    lo, hi = regime_length_range
    lengths, directions = [], []
    total = 0
    while total < n_trades:
        length = int(rng.integers(lo, hi + 1))
        direction = int(rng.integers(0, 2)) * 2 - 1  # 0/1 -> -1/+1
        lengths.append(length)
        directions.append(direction)
        total += length
    return lengths, directions


def generate_momentum_trades(
    n_trades,
    regime_length_range=(300, 600),
    continuation_prob=0.85,
    start_price=65_000.0,
    tick_bp=0.0005,
    noise_std=0.0003,
    volume_per_trade=0.05,
    start_ts_us=None,
    total_span_hours=720.0,
    tick_interval_us=None,
    random_state=None,
):
    """Generates n_trades synthetic raw trades, schema-identical to
    ingestion.py's RAW_TRADE_COLUMNS, carrying a DELIBERATE momentum
    edge: price moves in a slowly-switching 'regime' direction, and each
    tick's realized direction matches that regime with probability
    continuation_prob (default 0.85 -- deliberately far stronger than any
    real market's momentum, since this is meant to be an OBVIOUS edge a
    working pipeline should detect easily).

    Parameters
    ----------
    n_trades : int
    regime_length_range : (int, int), inclusive tick-count range for how
        long each regime persists before a fresh coin-flip direction.
        Default (300, 600), at ~24 trades/bar (this project's usual
        target_bars=250 density), is roughly 12-25 bars per regime --
        long enough that most individual triple-barrier events (up to
        rebuild.py's 3-day vertical barrier) resolve within a single
        dominant regime.
    continuation_prob : float in (0.5, 1.0]. 0.5 would mean no edge.
    start_price : float, BTC-scale so rebuild.py's flat CUSUM_H=500
        dollar threshold fires at a similar frequency to real BTC/USDT
        data (see module LOAD-BEARING calibration note).
    tick_bp, noise_std : float, per-tick drift/noise scale (see module
        LOAD-BEARING calibration note -- a starting point, not
        re-derived from first principles).
    volume_per_trade : float, CONSTANT across all trades (see module
        LOAD-BEARING note on why volume is not randomized here).
    start_ts_us : int or None. None -> starts total_span_hours before
        "now".
    total_span_hours : float, the TOTAL calendar time the n_trades should
        cover, default 720h (30 days, matching this project's
        LOOKBACK_HOURS convention). Ignored if tick_interval_us is given
        explicitly. This -- not tick_interval_us directly -- is the
        parameter to tune: get_daily_vol()'s 1-calendar-day lookback (Ch03
        Snippet 3.1) and rebuild.py's 3-day vertical barrier both need
        real ELAPSED TIME to work at all, independent of trade count or
        bar count (dollar bars form on cumulative DOLLAR volume, not
        elapsed time, so a short total_span_hours with many trades still
        produces a normal bar count but an empty daily_vol -- caught in
        this project's own sandbox dry run 2026-08-15, see driver script
        docstring).
    tick_interval_us : int or None. None (default) -> derived as
        total_span_hours*3600e6 / n_trades, spacing trades evenly across
        the requested span. An explicit value overrides total_span_hours
        entirely (used by this module's own hand-traced tests, where the
        exact spacing needs to be a known, simple number). Deterministic,
        not jittered -- guarantees a strictly increasing, collision-free
        Timestamp column without needing ingestion.py's
        _disambiguate_timestamps tiebreaker.
    random_state : int, np.random.Generator, or None.

    Returns
    -------
    dict with keys:
      'raw_trades'         : pd.DataFrame, RAW_TRADE_COLUMNS schema,
        ready to pass directly to rebuild.build_bars_and_labels() /
        features.build_enriched_events() unmodified.
      'target_directions'  : np.ndarray, the underlying regime direction
        per tick (ground truth, NOT visible to the pipeline -- for the
        driver script's own sanity checks only).
      'tick_directions'    : np.ndarray, the REALIZED (post-continuation-
        prob) direction per tick.
    """
    if not (0.5 < continuation_prob <= 1.0):
        raise ValueError('continuation_prob must be in (0.5, 1.0] to encode a real edge')

    rng = _as_generator(random_state)

    regime_lengths, regime_directions = _sample_regimes(n_trades, regime_length_range, rng)
    target_directions = _build_regime_sequence(n_trades, regime_lengths, regime_directions)
    tick_directions = _generate_tick_directions(target_directions, continuation_prob, rng)

    noise = rng.normal(0.0, noise_std, n_trades)
    prices = _directions_to_prices(tick_directions, start_price, tick_bp, noise)
    is_buyer_maker = _directions_to_is_buyer_maker(tick_directions)
    volumes = np.full(n_trades, volume_per_trade, dtype=float)

    if tick_interval_us is None:
        tick_interval_us = int(total_span_hours * 3600 * 1_000_000 / n_trades)
        # LOAD-BEARING (2026-08-15): derived from total_span_hours, NOT a
        # fixed constant -- an earlier version defaulted to a fixed
        # tick_interval_us=500_000 (0.5s), which at n_trades=6000 spans
        # only ~50 minutes total, not 30 days. get_daily_vol()'s 1-day
        # lookback (Ch03 Snippet 3.1) then has NO valid prior-day
        # reference point for ANY bar and silently returns an empty
        # Series (count=0, all NaN) -- not an error, just an empty
        # result that only surfaces downstream as "Triple-barrier
        # labeling produced zero events". Caught in this project's own
        # sandbox dry run before being handed off, not by a real-machine
        # run -- worth re-confirming the actual bar/event counts on the
        # first real run in the mlfinlab env too.
    if start_ts_us is None:
        now_us = int(time.time() * 1_000_000)
        start_ts_us = now_us - int(total_span_hours * 3600 * 1_000_000)
    timestamps = start_ts_us + np.arange(n_trades, dtype=np.int64) * tick_interval_us

    raw_trades = pd.DataFrame({
        'TradeID': np.arange(n_trades),
        'Price': prices,
        'Volume': volumes,
        'QuoteVolume': prices * volumes,
        'Timestamp': timestamps,
        'IsBuyerMaker': is_buyer_maker,
        'IsBestMatch': True,
    })[RAW_TRADE_COLUMNS]

    return {
        'raw_trades': raw_trades,
        'target_directions': target_directions,
        'tick_directions': tick_directions,
    }
