"""
pipeline/orchestration/test_positive_control_data.py

TDD test suite for positive_control_data.py. Hand-traced expected values
for every genuinely new function, matching this project's TDD convention
(test_features.py's own style): pure/deterministic helpers get exact
hand-traced assertions; the one function that draws real randomness
internally (_sample_regimes) is tested via a scripted fake RNG object
that returns a fixed, known sequence of draws, so its control flow is
still exactly hand-traced rather than depending on numpy's Generator
internals matching across versions. The top-level generate_momentum_trades
gets integration-level checks (schema, reproducibility, realized-edge
sanity) rather than a full hand-trace, mirroring test_features.py's own
mix of hand-traced + monkeypatched-isolation style for its own top-level
orchestration function.

Run (two-pass, per project convention):
    From repo root:              pytest pipeline/orchestration/test_positive_control_data.py -v
    From pipeline/orchestration: pytest test_positive_control_data.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import positive_control_data as pcd  # real module under test


# ---------------------------------------------------------------------
# _build_regime_sequence
# ---------------------------------------------------------------------

def test_build_regime_sequence_hand_traced():
    """lengths=[3,2], directions=[1,-1] -> [1,1,1,-1,-1]"""
    seq = pcd._build_regime_sequence(5, regime_lengths=[3, 2], regime_directions=[1, -1])
    assert list(seq) == [1, 1, 1, -1, -1]


def test_build_regime_sequence_trims_to_n_trades():
    """Same regimes as above, but n_trades=4 -> trimmed to the first 4
    entries: [1,1,1,-1]"""
    seq = pcd._build_regime_sequence(4, regime_lengths=[3, 2], regime_directions=[1, -1])
    assert list(seq) == [1, 1, 1, -1]


def test_build_regime_sequence_raises_if_insufficient():
    """lengths=[2], directions=[1] only covers 2 ticks; n_trades=5 must
    raise rather than silently returning a short array."""
    with pytest.raises(ValueError, match='only cover'):
        pcd._build_regime_sequence(5, regime_lengths=[2], regime_directions=[1])


def test_build_regime_sequence_mismatched_lengths_raises():
    with pytest.raises(ValueError, match='same length'):
        pcd._build_regime_sequence(3, regime_lengths=[1, 2], regime_directions=[1])


# ---------------------------------------------------------------------
# _generate_tick_directions
# ---------------------------------------------------------------------

class _FakeDrawRNG:
    """Returns a fixed, pre-scripted array from .random(n), ignoring n."""
    def __init__(self, draws):
        self._draws = np.array(draws)

    def random(self, n):
        assert n == len(self._draws)
        return self._draws


def test_generate_tick_directions_hand_traced():
    """target=[1,1,-1,-1], continuation_prob=0.6, draws=[0.1,0.7,0.2,0.9]:
    i0: 0.1 < 0.6 -> keep target[0]=1        ->  1
    i1: 0.7 not< 0.6 -> flip target[1]=1     -> -1
    i2: 0.2 < 0.6 -> keep target[2]=-1       -> -1
    i3: 0.9 not< 0.6 -> flip target[3]=-1    ->  1
    expected: [1, -1, -1, 1]
    """
    target = [1, 1, -1, -1]
    rng = _FakeDrawRNG([0.1, 0.7, 0.2, 0.9])
    out = pcd._generate_tick_directions(target, continuation_prob=0.6, rng=rng)
    assert list(out) == [1, -1, -1, 1]


def test_generate_tick_directions_draw_equal_to_prob_is_a_flip():
    """draw exactly == continuation_prob: strict '<' means this is NOT a
    keep -- it's a flip. target=[1], draws=[0.6], continuation_prob=0.6
    -> flipped -> -1."""
    rng = _FakeDrawRNG([0.6])
    out = pcd._generate_tick_directions([1], continuation_prob=0.6, rng=rng)
    assert list(out) == [-1]


# ---------------------------------------------------------------------
# _directions_to_prices
# ---------------------------------------------------------------------

def test_directions_to_prices_hand_traced_zero_noise():
    """start_price=100, tick_bp=0.01, directions=[1,-1,1], noise=[0,0,0]:
    price1 = 100 * 1.01     = 101
    price2 = 101 * 0.99     = 99.99
    price3 = 99.99 * 1.01   = 100.9899
    """
    prices = pcd._directions_to_prices(
        tick_directions=[1, -1, 1], start_price=100.0, tick_bp=0.01,
        noise=[0.0, 0.0, 0.0],
    )
    assert np.allclose(prices, [101.0, 99.99, 100.9899])


def test_directions_to_prices_hand_traced_with_noise():
    """start_price=100, tick_bp=0.01, direction=[1], noise=[0.002]:
    price1 = 100 * (1 + 0.01 + 0.002) = 100 * 1.012 = 101.2
    """
    prices = pcd._directions_to_prices(
        tick_directions=[1], start_price=100.0, tick_bp=0.01, noise=[0.002],
    )
    assert np.allclose(prices, [101.2])


# ---------------------------------------------------------------------
# _directions_to_is_buyer_maker
# ---------------------------------------------------------------------

def test_directions_to_is_buyer_maker_hand_traced():
    """+1 (buy) -> False, -1 (sell) -> True."""
    out = pcd._directions_to_is_buyer_maker([1, -1, -1, 1])
    assert list(out) == [False, True, True, False]


# ---------------------------------------------------------------------
# _sample_regimes (scripted fake RNG, exact call-order hand-trace)
# ---------------------------------------------------------------------

class _ScriptedIntRNG:
    """Returns pre-scripted values from .integers(), in call order,
    ignoring the (low, high) arguments -- lets the test hand-trace
    _sample_regimes' exact control flow (length draw, then direction
    draw, per loop iteration) without depending on numpy's real Generator
    internals."""
    def __init__(self, values):
        self._values = list(values)

    def integers(self, low, high=None):
        return self._values.pop(0)


def test_sample_regimes_hand_traced():
    """Scripted calls, in the exact order _sample_regimes makes them
    (length draw, then direction raw-bit draw, per iteration):
    iter1: length=3, direction_bit=1 -> direction = 1*2-1 = +1
    iter2: length=2, direction_bit=0 -> direction = 0*2-1 = -1
    running total after iter1 = 3 (< n_trades=5, loop continues)
    running total after iter2 = 5 (>= n_trades=5, loop stops)
    expected: lengths=[3,2], directions=[1,-1]
    """
    rng = _ScriptedIntRNG([3, 1, 2, 0])
    lengths, directions = pcd._sample_regimes(5, regime_length_range=(1, 10), rng=rng)
    assert lengths == [3, 2]
    assert directions == [1, -1]


def test_sample_regimes_stops_as_soon_as_total_meets_n_trades():
    """Single regime whose length alone already covers n_trades -> loop
    must stop after ONE iteration, not draw again.
    length=10, direction_bit=1 -> +1. total=10 >= n_trades=7 -> stop.
    """
    rng = _ScriptedIntRNG([10, 1])
    lengths, directions = pcd._sample_regimes(7, regime_length_range=(1, 20), rng=rng)
    assert lengths == [10]
    assert directions == [1]


# ---------------------------------------------------------------------
# generate_momentum_trades (integration-level: schema, reproducibility,
# realized-edge sanity -- mirrors test_features.py's own mix of
# hand-traced + integration-style tests for top-level orchestration)
# ---------------------------------------------------------------------

def test_generate_momentum_trades_rejects_no_edge_probability():
    with pytest.raises(ValueError, match='continuation_prob'):
        pcd.generate_momentum_trades(n_trades=100, continuation_prob=0.5)


def test_generate_momentum_trades_schema_matches_ingestion():
    out = pcd.generate_momentum_trades(n_trades=500, random_state=42)
    raw = out['raw_trades']
    assert list(raw.columns) == pcd.RAW_TRADE_COLUMNS
    assert len(raw) == 500
    assert raw['Timestamp'].is_monotonic_increasing
    assert raw['Timestamp'].is_unique
    assert (raw['Volume'] == raw['Volume'].iloc[0]).all()  # constant by design
    assert np.allclose(raw['QuoteVolume'], raw['Price'] * raw['Volume'])
    assert raw['IsBuyerMaker'].dtype == bool


def test_generate_momentum_trades_reproducible_with_same_seed():
    """Reproducibility requires ALL inputs fixed, including start_ts_us --
    left at its default of None, it's derived from the real wall clock
    (time.time()), which genuinely differs between the two calls below by
    design (matches ingestion.py's live-pull spirit: 'now' means now).
    An earlier version of this test left start_ts_us=None and failed
    nondeterministically on the Timestamp column alone -- a test-
    authoring bug (asserting reproducibility of a value that's supposed
    to vary), not a bug in generate_momentum_trades. Fixed by pinning
    start_ts_us explicitly, which is what a caller wanting reproducible
    output would do anyway."""
    out1 = pcd.generate_momentum_trades(n_trades=300, random_state=7, start_ts_us=1_700_000_000_000_000)
    out2 = pcd.generate_momentum_trades(n_trades=300, random_state=7, start_ts_us=1_700_000_000_000_000)
    pd.testing.assert_frame_equal(out1['raw_trades'], out2['raw_trades'])
    assert np.array_equal(out1['target_directions'], out2['target_directions'])


def test_generate_momentum_trades_realized_edge_is_strong():
    """With continuation_prob=0.9 on a large sample, the realized tick
    directions should match the (hidden) target regime roughly 90% of
    the time -- a coarse statistical sanity check (not a hand-trace) that
    the engineered edge is actually as strong as requested, analogous to
    test_features.py's monkeypatched-isolation checks for its own
    top-level wiring."""
    out = pcd.generate_momentum_trades(
        n_trades=5000, continuation_prob=0.9, random_state=123,
    )
    match_rate = np.mean(out['tick_directions'] == out['target_directions'])
    assert 0.87 < match_rate < 0.93  # should land close to 0.9


def test_generate_momentum_trades_start_ts_us_respected():
    out = pcd.generate_momentum_trades(n_trades=10, start_ts_us=1_000_000, tick_interval_us=1000)
    ts = out['raw_trades']['Timestamp'].values
    assert ts[0] == 1_000_000
    assert list(np.diff(ts)) == [1000] * 9
