"""
test_ch04.py — TDD tests for AFML Chapter 4 implementations
Run with: pytest ch04/tests/test_ch04.py -v

📁 C:\\ws\\AFML\\
└── ch04\\
    └── tests\\
        └── test_ch04.py   ← goes here

All expected values were computed by running the actual implementation
functions and recording their output, or by replicating the book's
hand-worked numerical example from Section 4.5.3. Tests verify specific
numeric values, not just types or shapes.

Note on randomness: seq_bootstrap, get_rnd_t1, and aux_mc all use np.random
internally. Tests that check exact output values seed the RNG first for
reproducibility. Tests of the underlying probability calculations (which
are deterministic) do not require seeding.
"""

import sys, os
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ch04.sample_weights.co_events             import mp_num_co_events
from ch04.sample_weights.uniqueness            import mp_sample_tw, get_average_uniqueness
from ch04.sample_weights.indicator_matrix      import get_ind_matrix
from ch04.sample_weights.avg_uniqueness_matrix import get_avg_uniqueness
from ch04.sample_weights.sequential_bootstrap  import seq_bootstrap
from ch04.sample_weights.monte_carlo           import get_rnd_t1, aux_mc
from ch04.sample_weights.return_attribution    import mp_sample_w, get_sample_weights
from ch04.sample_weights.time_decay            import get_time_decay
from ch04.sample_weights.real_data_bootstrap_comparison import compare_bootstrap_on_real_events


# ===========================================================================
# Snippet 4.1 — co_events.py
# ===========================================================================

class TestMpNumCoEvents:

    @pytest.fixture
    def setup(self):
        # Book's 3-observation example: obs0 spans bars 0-2, obs1 spans
        # bars 2-3, obs2 spans bars 4-5. Only bar 2 has concurrency 2.
        dates = pd.date_range('2020-01-01', periods=6, freq='D')
        t1 = pd.Series(
            [dates[2], dates[3], dates[5]],
            index=[dates[0], dates[2], dates[4]]
        )
        return dates, t1

    def test_known_concurrency_values(self, setup):
        dates, t1 = setup
        result = mp_num_co_events(dates, t1, t1.index)
        assert list(result.values) == [1, 1, 2, 1, 1, 1]

    def test_overlap_bar_has_highest_concurrency(self, setup):
        dates, t1 = setup
        result = mp_num_co_events(dates, t1, t1.index)
        assert result.iloc[2] == 2  # bar 2 is the only overlap point
        assert result.iloc[2] == result.max()

    def test_returns_series(self, setup):
        dates, t1 = setup
        result = mp_num_co_events(dates, t1, t1.index)
        assert isinstance(result, pd.Series)

    def test_open_event_uses_last_bar_as_end(self, setup):
        # Event with NaT end (still open) should be treated as ending at
        # the last available bar, not excluded.
        dates, t1 = setup
        t1_open = t1.copy()
        t1_open.iloc[-1] = pd.NaT
        result = mp_num_co_events(dates, t1_open, t1_open.index)
        # The "open" event (originally bars 4-5) should still count toward
        # concurrency at bar 5 (now extended to the last bar, which is bar 5)
        assert result.iloc[-1] >= 1


# ===========================================================================
# Snippet 4.2 — uniqueness.py
# ===========================================================================

class TestMpSampleTw:

    @pytest.fixture
    def setup(self):
        # Exact book example from Section 4.5.3
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        bar_ix = pd.Index(range(t1.max() + 1))
        num_co = mp_num_co_events(bar_ix, t1, t1.index)
        return t1, num_co

    def test_known_uniqueness_values(self, setup):
        t1, num_co = setup
        tw = mp_sample_tw(t1, num_co, t1.index)
        assert tw.iloc[0] == pytest.approx(5/6, abs=1e-9)   # obs0: 0.8333
        assert tw.iloc[1] == pytest.approx(0.75, abs=1e-9)  # obs1
        assert tw.iloc[2] == pytest.approx(1.0, abs=1e-9)   # obs2: no overlap

    def test_non_overlapping_event_has_uniqueness_one(self, setup):
        t1, num_co = setup
        tw = mp_sample_tw(t1, num_co, t1.index)
        assert tw.iloc[2] == pytest.approx(1.0)

    def test_uniqueness_bounded_between_zero_and_one(self, setup):
        t1, num_co = setup
        tw = mp_sample_tw(t1, num_co, t1.index)
        assert (tw > 0).all()
        assert (tw <= 1.0).all()


class TestGetAverageUniqueness:

    @pytest.fixture
    def events_setup(self):
        # Build a small events DataFrame compatible with get_events() output
        dates = pd.date_range('2020-01-01', periods=6, freq='D')
        close = pd.Series([100, 101, 102, 103, 104, 105], index=dates)
        t1 = pd.Series(
            [dates[2], dates[3], dates[5]],
            index=[dates[0], dates[2], dates[4]]
        )
        events = pd.DataFrame({'t1': t1})
        return close, events

    def test_returns_series(self, events_setup):
        close, events = events_setup
        tw = get_average_uniqueness(close, events, num_threads=1)
        assert isinstance(tw, pd.Series)

    def test_matches_known_values(self, events_setup):
        close, events = events_setup
        tw = get_average_uniqueness(close, events, num_threads=1)
        values = sorted(tw.values)
        expected = sorted([5/6, 0.75, 1.0])
        np.testing.assert_allclose(values, expected, rtol=1e-6)

    def test_output_length_matches_events(self, events_setup):
        close, events = events_setup
        tw = get_average_uniqueness(close, events, num_threads=1)
        assert len(tw) == len(events)


# ===========================================================================
# Snippet 4.3 — indicator_matrix.py
# ===========================================================================

class TestGetIndMatrix:

    def test_matches_book_example(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        ind_m = get_ind_matrix(range(t1.max() + 1), t1)
        expected = [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ]
        assert ind_m.values.tolist() == expected

    def test_shape_is_bars_by_events(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        ind_m = get_ind_matrix(range(t1.max() + 1), t1)
        assert ind_m.shape == (6, 3)

    def test_only_zeros_and_ones(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        ind_m = get_ind_matrix(range(t1.max() + 1), t1)
        assert set(ind_m.values.flatten()) <= {0.0, 1.0}

    def test_single_observation_no_overlap(self):
        t1 = pd.Series([3], index=[0])
        ind_m = get_ind_matrix(range(4), t1)
        assert ind_m[0].sum() == 4  # touches bars 0,1,2,3


# ===========================================================================
# Snippet 4.4 — avg_uniqueness_matrix.py
# ===========================================================================

class TestGetAvgUniqueness:

    @pytest.fixture
    def ind_m(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        return get_ind_matrix(range(t1.max() + 1), t1)

    def test_matches_book_example(self, ind_m):
        avg_u = get_avg_uniqueness(ind_m)
        assert avg_u.iloc[0] == pytest.approx(5/6, abs=1e-9)
        assert avg_u.iloc[1] == pytest.approx(0.75, abs=1e-9)
        assert avg_u.iloc[2] == pytest.approx(1.0, abs=1e-9)

    def test_matches_bar_by_bar_method(self, ind_m):
        # Matrix method and bar-by-bar method (mp_sample_tw) must agree
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        bar_ix = pd.Index(range(t1.max() + 1))
        num_co = mp_num_co_events(bar_ix, t1, t1.index)
        tw = mp_sample_tw(t1, num_co, t1.index)

        avg_u = get_avg_uniqueness(ind_m)
        np.testing.assert_allclose(avg_u.values, tw.values, rtol=1e-9)

    def test_returns_series(self, ind_m):
        avg_u = get_avg_uniqueness(ind_m)
        assert isinstance(avg_u, pd.Series)


# ===========================================================================
# Snippet 4.5 — sequential_bootstrap.py
# ===========================================================================

class TestSeqBootstrap:

    @pytest.fixture
    def ind_m(self):
        t1 = pd.Series([2, 3, 5], index=[0, 2, 4])
        return get_ind_matrix(range(t1.max() + 1), t1)

    def test_default_sample_length(self, ind_m):
        np.random.seed(1)
        phi = seq_bootstrap(ind_m)
        assert len(phi) == ind_m.shape[1]

    def test_custom_sample_length(self, ind_m):
        np.random.seed(1)
        phi = seq_bootstrap(ind_m, s_length=5)
        assert len(phi) == 5

    def test_all_drawn_values_are_valid_columns(self, ind_m):
        np.random.seed(1)
        phi = seq_bootstrap(ind_m)
        assert all(p in ind_m.columns for p in phi)

    def test_deterministic_with_seed(self, ind_m):
        np.random.seed(999)
        phi_a = seq_bootstrap(ind_m)
        np.random.seed(999)
        phi_b = seq_bootstrap(ind_m)
        assert phi_a == phi_b

    def test_probability_after_first_draw_matches_book(self, ind_m):
        # Replicates the book's hand-worked example exactly:
        # after drawing obs1 first, probabilities should be {5/14, 3/14, 6/14}
        phi = [1]
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()

        assert prob.iloc[0] == pytest.approx(5/14, abs=1e-9)
        assert prob.iloc[1] == pytest.approx(3/14, abs=1e-9)
        assert prob.iloc[2] == pytest.approx(6/14, abs=1e-9)

    def test_already_drawn_observation_gets_lowest_probability(self, ind_m):
        # Book's key claim: the observation already in phi should have the
        # LOWEST probability of being drawn again (most redundant with itself)
        phi = [1]
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()
        assert prob.iloc[1] == prob.min()

    def test_non_overlapping_observation_gets_highest_probability(self, ind_m):
        # Book's key claim: the observation with NO overlap to phi should
        # have the HIGHEST probability
        phi = [1]
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()
        assert prob.iloc[2] == prob.max()


# ===========================================================================
# Snippets 4.7-4.8 — monte_carlo.py (get_rnd_t1, aux_mc)
# ===========================================================================

class TestGetRndT1:

    def test_output_is_sorted(self):
        np.random.seed(0)
        t1 = get_rnd_t1(num_obs=10, num_bars=100, max_h=5)
        assert t1.index.is_monotonic_increasing

    def test_correct_number_of_observations(self):
        np.random.seed(0)
        t1 = get_rnd_t1(num_obs=10, num_bars=100, max_h=5)
        # Note: collisions (two obs starting on same bar) can reduce count slightly
        assert len(t1) <= 10
        assert len(t1) > 0

    def test_durations_within_bounds(self):
        np.random.seed(0)
        t1 = get_rnd_t1(num_obs=20, num_bars=200, max_h=5)
        durations = t1.values - t1.index.values
        assert (durations >= 1).all()
        assert (durations < 5).all()

    def test_start_bars_within_bounds(self):
        np.random.seed(0)
        t1 = get_rnd_t1(num_obs=20, num_bars=200, max_h=5)
        assert (t1.index >= 0).all()
        assert (t1.index < 200).all()


class TestAuxMc:

    def test_returns_dict_with_expected_keys(self):
        np.random.seed(42)
        result = aux_mc(num_obs=10, num_bars=100, max_h=5)
        assert 'stdU' in result
        assert 'seqU' in result

    def test_uniqueness_values_in_valid_range(self):
        np.random.seed(42)
        result = aux_mc(num_obs=10, num_bars=100, max_h=5)
        assert 0 < result['stdU'] <= 1.0
        assert 0 < result['seqU'] <= 1.0

    def test_sequential_tends_to_beat_standard_on_average(self):
        # Not guaranteed every single trial, but over enough trials the
        # mean sequential uniqueness should exceed mean standard uniqueness
        # (this is the core empirical claim of Section 4.5.4)
        np.random.seed(7)
        results = [aux_mc(num_obs=10, num_bars=100, max_h=5) for _ in range(30)]
        mean_std = np.mean([r['stdU'] for r in results])
        mean_seq = np.mean([r['seqU'] for r in results])
        assert mean_seq > mean_std


# ===========================================================================
# Snippet 4.10 — return_attribution.py
# ===========================================================================

class TestMpSampleW:

    @pytest.fixture
    def setup(self):
        dates = pd.date_range('2020-01-01', periods=6, freq='D')
        close = pd.Series([100, 105, 110, 108, 120, 125], index=dates)
        t1 = pd.Series(
            [dates[2], dates[3], dates[5]],
            index=[dates[0], dates[2], dates[4]]
        )
        num_co = mp_num_co_events(dates, t1, t1.index)
        return close, t1, num_co

    def test_known_weight_values(self, setup):
        close, t1, num_co = setup
        w = mp_sample_w(t1, num_co, close, t1.index)
        assert w.iloc[0] == pytest.approx(0.07205, abs=1e-4)
        assert w.iloc[1] == pytest.approx(0.004911, abs=1e-5)
        assert w.iloc[2] == pytest.approx(0.146183, abs=1e-5)

    def test_weights_are_non_negative(self, setup):
        close, t1, num_co = setup
        w = mp_sample_w(t1, num_co, close, t1.index)
        assert (w >= 0).all()

    def test_largest_return_gets_largest_weight(self, setup):
        # obs2 (Jan5-Jan9, the +10.5% bar) should have the largest weight
        close, t1, num_co = setup
        w = mp_sample_w(t1, num_co, close, t1.index)
        assert w.iloc[2] == w.max()


class TestGetSampleWeights:

    @pytest.fixture
    def events_setup(self):
        dates = pd.date_range('2020-01-01', periods=6, freq='D')
        close = pd.Series([100, 105, 110, 108, 120, 125], index=dates)
        t1 = pd.Series(
            [dates[2], dates[3], dates[5]],
            index=[dates[0], dates[2], dates[4]]
        )
        events = pd.DataFrame({'t1': t1})
        return close, events

    def test_weights_sum_to_number_of_observations(self, events_setup):
        close, events = events_setup
        w = get_sample_weights(close, events, num_threads=1)
        assert w.sum() == pytest.approx(len(events), abs=1e-6)

    def test_returns_series(self, events_setup):
        close, events = events_setup
        w = get_sample_weights(close, events, num_threads=1)
        assert isinstance(w, pd.Series)

    def test_weights_non_negative(self, events_setup):
        close, events = events_setup
        w = get_sample_weights(close, events, num_threads=1)
        assert (w >= 0).all()


# ===========================================================================
# Snippet 4.11 — time_decay.py
# ===========================================================================

class TestGetTimeDecay:

    @pytest.fixture
    def tw_uniform(self):
        # 5 non-overlapping observations, each with uniqueness 1.0
        return pd.Series(
            [1.0]*5,
            index=pd.date_range('2020-01-01', periods=5, freq='D')
        )

    def test_no_decay_when_clf_last_w_is_one(self, tw_uniform):
        result = get_time_decay(tw_uniform, clf_last_w=1.0)
        np.testing.assert_allclose(result.values, [1.0]*5)

    def test_known_values_clf_last_w_half(self, tw_uniform):
        result = get_time_decay(tw_uniform, clf_last_w=0.5)
        expected = [0.6, 0.7, 0.8, 0.9, 1.0]
        np.testing.assert_allclose(result.values, expected, rtol=1e-9)

    def test_known_values_clf_last_w_zero(self, tw_uniform):
        result = get_time_decay(tw_uniform, clf_last_w=0.0)
        expected = [0.2, 0.4, 0.6, 0.8, 1.0]
        np.testing.assert_allclose(result.values, expected, rtol=1e-9)

    def test_known_values_negative_clf_last_w_hard_excludes(self, tw_uniform):
        # With clf_last_w=-0.5, the two oldest observations should be
        # hard-clipped to exactly 0
        result = get_time_decay(tw_uniform, clf_last_w=-0.5)
        expected = [0.0, 0.0, 0.2, 0.6, 1.0]
        np.testing.assert_allclose(result.values, expected, rtol=1e-9)

    def test_newest_observation_always_gets_weight_one(self, tw_uniform):
        for clw in [1.0, 0.5, 0.0, -0.5, -0.9]:
            result = get_time_decay(tw_uniform, clf_last_w=clw)
            assert result.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_weights_never_negative(self, tw_uniform):
        for clw in [1.0, 0.5, 0.0, -0.5, -0.9]:
            result = get_time_decay(tw_uniform, clf_last_w=clw)
            assert (result >= 0).all()

    def test_monotonically_increasing_with_time(self, tw_uniform):
        # Weight should increase (or stay flat) from oldest to newest
        result = get_time_decay(tw_uniform, clf_last_w=0.3)
        diffs = result.values[1:] - result.values[:-1]
        assert (diffs >= -1e-9).all()


# ===========================================================================
# real_data_bootstrap_comparison.py
# ===========================================================================
# Companion to the Monte Carlo experiment (Snippets 4.7-4.9), built to run
# the same standard-vs-sequential bootstrap comparison directly on a
# student's own real, labeled events instead of synthetic data. Tests here
# focus on STRUCTURE and PERFORMANCE rather than exact values, since the
# function involves randomized bootstrap draws.

class TestCompareBootstrapOnRealEvents:

    @pytest.fixture
    def real_events_setup(self):
        # Build a realistically-sized synthetic stand-in for a student's
        # real triple-barrier events DataFrame: many bars, many events,
        # spanning a wide date range (similar in shape to real BTC dollar
        # bars + CUSUM events from Chapter 3).
        np.random.seed(7)
        n_bars = 2000
        dates = pd.date_range('2026-03-01', periods=n_bars, freq='h')
        close = pd.Series(
            100 + np.cumsum(np.random.randn(n_bars) * 0.3),
            index=dates
        )
        n_events = 150
        event_dates = sorted(np.random.choice(dates[:-50], size=n_events, replace=False))
        t1_dates = [
            dates[dates.get_loc(d) + np.random.randint(2, 40)]
            for d in event_dates
        ]
        events = pd.DataFrame({'t1': t1_dates}, index=event_dates)
        return close, events

    def test_returns_expected_keys(self, real_events_setup):
        close, events = real_events_setup
        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=5, seed=1
        )
        assert 'std_vals' in result
        assert 'seq_vals' in result
        assert 'n_events' in result
        assert 'n_bars' in result
        assert 'ind_m' in result

    def test_n_events_never_exceeds_max_events(self, real_events_setup):
        close, events = real_events_setup
        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=5, seed=1
        )
        assert result['n_events'] <= 12

    def test_no_subsampling_when_fewer_events_than_cap(self):
        # If the events DataFrame already has fewer rows than max_events,
        # all of them should be used (no unnecessary subsampling)
        np.random.seed(5)
        n_bars = 100
        dates = pd.date_range('2026-03-01', periods=n_bars, freq='h')
        close = pd.Series(100 + np.cumsum(np.random.randn(n_bars) * 0.3), index=dates)

        n_events = 8  # fewer than default max_events=12
        event_dates = sorted(np.random.choice(dates[:-15], size=n_events, replace=False))
        t1_dates = [dates[dates.get_loc(d) + np.random.randint(2, 10)] for d in event_dates]
        small_events = pd.DataFrame({'t1': t1_dates}, index=event_dates)

        result = compare_bootstrap_on_real_events(
            close, small_events, max_events=12, n_trials=5, seed=1
        )
        assert result['n_events'] == 8

    def test_correct_number_of_trials(self, real_events_setup):
        close, events = real_events_setup
        n_trials = 10
        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=n_trials, seed=1
        )
        assert len(result['std_vals']) == n_trials
        assert len(result['seq_vals']) == n_trials

    def test_uniqueness_values_in_valid_range(self, real_events_setup):
        close, events = real_events_setup
        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=10, seed=1
        )
        for v in result['std_vals'] + result['seq_vals']:
            assert 0 < v <= 1.0

    def test_drops_events_with_unresolved_t1(self):
        # Events with NaT (unresolved) t1 should be excluded before
        # building the indicator matrix
        np.random.seed(5)
        n_bars = 100
        dates = pd.date_range('2026-03-01', periods=n_bars, freq='h')
        close = pd.Series(100 + np.cumsum(np.random.randn(n_bars) * 0.3), index=dates)

        n_events = 8
        event_dates = sorted(np.random.choice(dates[:-15], size=n_events, replace=False))
        t1_dates = [dates[dates.get_loc(d) + np.random.randint(2, 10)] for d in event_dates]
        events = pd.DataFrame({'t1': t1_dates}, index=event_dates)
        events.iloc[0, 0] = pd.NaT  # mark one event as unresolved

        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=3, seed=1
        )
        assert result['n_events'] == 7  # one dropped

    def test_reproducible_with_seed(self, real_events_setup):
        close, events = real_events_setup
        r1 = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=5, seed=99
        )
        r2 = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=5, seed=99
        )
        assert r1['std_vals'] == r2['std_vals']
        assert r1['seq_vals'] == r2['seq_vals']

    def test_runs_within_reasonable_time(self, real_events_setup):
        # Performance regression guard: this function was specifically
        # rebuilt to avoid runaway runtime on large real datasets (see
        # real_data_bootstrap_comparison.py module docstring for the two
        # bugs that caused 40s+ runtimes before the fix). A correct
        # implementation on this dataset size should comfortably finish
        # in well under 10 seconds.
        import time
        close, events = real_events_setup
        start = time.time()
        compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=15, seed=1
        )
        elapsed = time.time() - start
        assert elapsed < 10.0

    def test_indicator_matrix_shape_matches_n_events(self, real_events_setup):
        close, events = real_events_setup
        result = compare_bootstrap_on_real_events(
            close, events, max_events=12, n_trials=3, seed=1
        )
        assert result['ind_m'].shape[1] == result['n_events']
        assert result['ind_m'].shape[0] == result['n_bars']
