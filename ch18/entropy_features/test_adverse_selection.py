"""
TDD suite for Chapter 18, Section 18.8.4 -- the market-microstructure
adverse-selection workflow (order-flow imbalance -> quantile encoding ->
rolling Kontoyiannis entropy -> empirical CDF).

Formula-only, workflow-only section (the book gives six prose steps, no
code, and no algorithm for the rolling-entropy mechanics -- see the
module's own docstring for the documented judgment call). Known values
here are cross-checked via direct computation against
entropy_estimators.konto (already independently hand-traced/tested in
test_entropy_estimators.py), not re-derived by hand from scratch for
every case -- consistent with this project's treatment of other
workflow-level formula-only sections (e.g. Ch17's CUSUM).
"""
import numpy as np
import pandas as pd
import pytest

from adverse_selection import order_flow_imbalance, adverse_selection_feature
from entropy_estimators import konto


# -----------------------------------------------------------------------
# order_flow_imbalance
# -----------------------------------------------------------------------
class TestOrderFlowImbalance:
    def test_basic_ratio(self):
        vB = order_flow_imbalance([7, 3], [3, 7])
        assert vB.tolist() == pytest.approx([0.7, 0.3])

    def test_all_buy_is_one(self):
        vB = order_flow_imbalance([10], [0])
        assert vB.tolist() == pytest.approx([1.0])

    def test_all_sell_is_zero(self):
        vB = order_flow_imbalance([0], [10])
        assert vB.tolist() == pytest.approx([0.0])

    def test_zero_total_volume_guarded_to_neutral(self):
        """Degenerate edge case the book doesn't address: a bar with
        zero buy AND zero sell volume must not divide by zero, and
        gets the neutral 0.5 value per this module's documented
        design decision."""
        vB = order_flow_imbalance([0, 5], [0, 5])
        assert vB.tolist() == pytest.approx([0.5, 0.5])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            order_flow_imbalance([1, 2], [1])


# -----------------------------------------------------------------------
# adverse_selection_feature -- input validation
# -----------------------------------------------------------------------
class TestAdverseSelectionFeatureValidation:
    def test_odd_roll_window_rejected(self):
        """LOAD-BEARING: konto's own expanding-window mode requires
        len(msg)%2==0 (book's own stated precondition for Snippet
        18.4). An odd roll_window would silently truncate via floor
        division inside konto rather than raising -- reject it here
        instead, at the boundary where the mistake is easiest to spot."""
        with pytest.raises(ValueError):
            adverse_selection_feature([1] * 20, [1] * 20, roll_window=5)

    def test_too_small_roll_window_rejected(self):
        with pytest.raises(ValueError):
            adverse_selection_feature([1] * 20, [1] * 20, roll_window=2)

    def test_fewer_bars_than_window_rejected(self):
        with pytest.raises(ValueError):
            adverse_selection_feature([1, 2, 3], [3, 2, 1], roll_window=10)


# -----------------------------------------------------------------------
# adverse_selection_feature -- cross-checked known values
# -----------------------------------------------------------------------
class TestAdverseSelectionFeatureCrossChecked:
    def test_symmetric_alternating_flow_all_windows_tied(self):
        """
        buy/sell perfectly alternate every bar (vB = 1,0,1,0,...,1,0
        over 8 bars). With n_quantiles=2, this quantizes to '10101010'
        exactly (each half of the alternating series is its own
        quantile). Every roll_window=4 slice of this string is a
        cyclic rotation of '1010' -- konto('1010') and konto('0101')
        are CROSS-CHECKED (via direct call to entropy_estimators.konto,
        the already hand-traced/tested estimator) to give the
        IDENTICAL entropy value (0.7641604167868594), so all 5 rolling
        windows tie. With pandas' average-rank tie-breaking, 5 tied
        values all get percentile rank (1+2+3+4+5)/5/5 = 0.6.
        """
        buy = [10, 0, 10, 0, 10, 0, 10, 0]
        sell = [0, 10, 0, 10, 0, 10, 0, 10]

        # Cross-check the building block directly against konto.
        assert konto('1010')['h'] == pytest.approx(0.7641604167868594)
        assert konto('0101')['h'] == pytest.approx(0.7641604167868594)

        feat = adverse_selection_feature(buy, sell, n_quantiles=2, roll_window=4)
        assert feat.tolist() == pytest.approx([0.6] * 5)
        assert feat.index.tolist() == [3, 4, 5, 6, 7]

    def test_output_length_and_index_alignment(self):
        """
        n bars, roll_window w -> n-w+1 output points, indexed by the
        LAST bar in each window (positions w-1 through n-1, 0-indexed).
        """
        n, w = 30, 10
        buy = np.arange(n, dtype=float) % 5
        sell = 4 - buy
        feat = adverse_selection_feature(buy, sell, n_quantiles=3, roll_window=w)
        assert len(feat) == n - w + 1
        assert feat.index.tolist() == list(range(w - 1, n))

    def test_cdf_values_bounded_zero_to_one(self):
        rng = np.random.RandomState(3)
        buy = rng.uniform(0, 10, 60)
        sell = rng.uniform(0, 10, 60)
        feat = adverse_selection_feature(buy, sell, n_quantiles=4, roll_window=20)
        assert (feat >= 0).all() and (feat <= 1).all()

    def test_repetitive_flow_reads_lower_cdf_than_random_flow(self):
        """
        Regime-change sanity check (cross-checked by direct run, see
        module docstring for the full construction): the first half of
        the series has perfectly repetitive buy/sell flow, the second
        half pseudo-random flow. Windows sitting in the repetitive
        region should score reliably LOWER on the empirical CDF (less
        entropic relative to the whole sample) than windows sitting in
        the random region.
        """
        rng = np.random.RandomState(7)
        buy_repetitive = np.tile([10, 0], 20)
        sell_repetitive = np.tile([0, 10], 20)
        buy_random = rng.uniform(0, 10, 40)
        sell_random = rng.uniform(0, 10, 40)
        buy = np.concatenate([buy_repetitive, buy_random])
        sell = np.concatenate([sell_repetitive, sell_random])

        feat = adverse_selection_feature(buy, sell, n_quantiles=5, roll_window=20)
        mean_early = feat.iloc[:20].mean()
        mean_late = feat.iloc[-20:].mean()
        assert mean_early < mean_late
