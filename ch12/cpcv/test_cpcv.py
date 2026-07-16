"""
Chapter 12 -- TDD suite for the CPCV combinatorics + purge/embargo engine.

Since Ch12 has no printed code, the "known values" here come from two
places instead of a book snippet: (1) AFML's own explicit prose
description of path 1 and path 2 for its N=6, k=2 example (Section
12.4.1), and (2) exact formulas given in the chapter text (phi[N,k],
C(N,k)).
"""
import numpy as np
import pandas as pd
import pytest

from cpcv import (
    partition_groups,
    enumerate_splits,
    n_paths,
    build_path_assignment,
    generalized_train_test_positions,
    run_cpcv,
)


# ---------------------------------------------------------------------------
# partition_groups
# ---------------------------------------------------------------------------
class TestPartitionGroups:
    def test_book_example_T88_N6(self):
        # 88 // 6 = 14, remainder 4 -> groups 1-5 size 14, group 6 size 18
        bounds = partition_groups(88, 6)
        sizes = [e - s for s, e in bounds]
        assert sizes == [14, 14, 14, 14, 14, 18]
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 88

    def test_evenly_divisible(self):
        bounds = partition_groups(12, 6)
        assert [e - s for s, e in bounds] == [2, 2, 2, 2, 2, 2]

    def test_rejects_fewer_than_two_groups(self):
        with pytest.raises(ValueError):
            partition_groups(10, 1)


# ---------------------------------------------------------------------------
# enumerate_splits / n_paths -- AFML's own N=6,k=2 worked example
# ---------------------------------------------------------------------------
class TestSplitCounts:
    def test_book_example_15_splits(self):
        # book: "There are (6 choose 4) = 15 splits"
        assert len(enumerate_splits(6, 2)) == 15

    def test_book_example_5_paths(self):
        # book: "this train/test split scheme allows us to compute 5 backtest paths"
        assert n_paths(6, 2) == 5

    def test_k1_reduces_to_plain_cv(self):
        # book 12.4.3: "For k=1, we will obtain phi[N,1]=1 path... CPCV
        # reduces to CV"
        assert n_paths(6, 1) == 1
        assert len(enumerate_splits(6, 1)) == 6

    def test_k2_rule_of_thumb_N_minus_1_paths(self):
        # book 12.4.3: "For k=2, we will obtain phi[N,2] = N-1 paths"
        for n in (4, 6, 8, 10):
            assert n_paths(n, 2) == n - 1

    def test_splits_are_lexicographic_combinations(self):
        splits = enumerate_splits(6, 2)
        assert splits[0] == (0, 1)
        assert splits[1] == (0, 2)
        assert splits[5] == (1, 2)
        assert splits[-1] == (4, 5)

    def test_rejects_k_out_of_range(self):
        with pytest.raises(ValueError):
            enumerate_splits(6, 6)
        with pytest.raises(ValueError):
            enumerate_splits(6, 0)


# ---------------------------------------------------------------------------
# build_path_assignment -- the golden test: reproduce the book's own
# path 1 and path 2 compositions for N=6, k=2, verbatim from the prose.
# ---------------------------------------------------------------------------
class TestPathAssignment:
    def test_reproduces_book_path1_and_path2(self):
        # Using 0-indexed groups internally (G1->0, ..., G6->5) and
        # 0-indexed splits (S1->enumerate_splits index 0, etc, since
        # enumerate_splits is in lexicographic combination order and the
        # book's S-numbering follows the same order -- see cpcv.py
        # module docstring for the derivation).
        assignment, phi, splits = build_path_assignment(6, 2)
        assert phi == 5

        # Book prose: "path 1 is the result of combining the forecasts
        # from (G1,S1),(G2,S1),(G3,S2),(G4,S3),(G5,S4),(G6,S5)"
        # G1=0,G2=1,G3=2,G4=3,G5=4,G6=5 ; S1=idx0,S2=idx1,S3=idx2,S4=idx3,S5=idx4
        expected_path1 = {
            (0, 0): 1,  # (G1,S1)
            (0, 1): 1,  # (G2,S1)
            (1, 2): 1,  # (G3,S2)
            (2, 3): 1,  # (G4,S3)
            (3, 4): 1,  # (G5,S4)
            (4, 5): 1,  # (G6,S5)
        }
        for key, path_num in expected_path1.items():
            assert assignment[key] == path_num, f'{key} should be path {path_num}'

        # Book prose: "path 2 is the result of combining forecasts from
        # (G1,S2),(G2,S6),(G3,S6),(G4,S7),(G5,S8),(G6,S9)"
        # S6=idx5, S7=idx6, S8=idx7, S9=idx8
        expected_path2 = {
            (1, 0): 2,  # (G1,S2)
            (5, 1): 2,  # (G2,S6)
            (5, 2): 2,  # (G3,S6)
            (6, 3): 2,  # (G4,S7)
            (7, 4): 2,  # (G5,S8)
            (8, 5): 2,  # (G6,S9)
        }
        for key, path_num in expected_path2.items():
            assert assignment[key] == path_num, f'{key} should be path {path_num}'

    def test_every_group_contributes_exactly_once_per_path(self):
        for n_groups, k in [(6, 2), (8, 2), (5, 2), (6, 1)]:
            assignment, phi, splits = build_path_assignment(n_groups, k)
            for path_num in range(1, phi + 1):
                groups_in_path = [g for (s_idx, g), p in assignment.items() if p == path_num]
                assert sorted(groups_in_path) == list(range(n_groups)), (
                    f'path {path_num} (N={n_groups},k={k}) should contain every '
                    f'group exactly once'
                )

    def test_every_group_is_test_group_in_exactly_phi_splits(self):
        assignment, phi, splits = build_path_assignment(6, 2)
        for g in range(6):
            occurrences = [1 for (s_idx, gg) in assignment if gg == g]
            assert len(occurrences) == phi


# ---------------------------------------------------------------------------
# generalized_train_test_positions -- regression check against Ch07's
# original single-contiguous-block PurgedKFold formula (k=1 case), plus
# a genuinely-combinatorial k=2 case
# ---------------------------------------------------------------------------
def _make_synthetic_t1(n_obs, avg_span=3, seed=0):
    """Synthetic t0-indexed / t1-valued series with overlapping label
    windows, like real triple-barrier labels -- same spirit as Ch07's
    own synthetic PurgedKFold tests."""
    rng = np.random.RandomState(seed)
    t0 = pd.date_range('2026-01-01', periods=n_obs, freq='h')
    spans = rng.randint(1, avg_span * 2, size=n_obs)
    t1_vals = [t0[i] + pd.Timedelta(hours=int(spans[i])) for i in range(n_obs)]
    return pd.Series(t1_vals, index=t0)


class TestPurgeEmbargo:
    def test_k1_matches_original_purged_kfold(self):
        # Reimplements Ch07 PurgedKFold.split()'s single-fold formula
        # inline (not imported, to keep this test self-contained) and
        # checks our generalized k=1 case matches it exactly.
        t1 = _make_synthetic_t1(30, avg_span=4, seed=1)
        n_obs = len(t1)
        n_splits = 5
        pct_embargo = 0.1
        mbrg = int(n_obs * pct_embargo)
        indices = np.arange(n_obs)
        test_starts = [(part[0], part[-1] + 1) for part in np.array_split(indices, n_splits)]

        group_bounds = test_starts  # a plain n_splits-fold partition IS partition_groups' shape

        for fold_idx, (i, j) in enumerate(test_starts):
            # --- original Ch07 formula ---
            t0 = t1.index[i]
            test_indices_orig = indices[i:j]
            max_t1_idx = t1.index.searchsorted(t1.iloc[test_indices_orig].max())
            train_orig = t1.index.searchsorted(t1[t1 <= t0].index)
            if max_t1_idx < n_obs:
                train_orig = np.concatenate((train_orig, indices[max_t1_idx + mbrg:]))
            train_orig = np.sort(np.array(sorted(set(train_orig) - set(test_indices_orig))))

            # --- generalized formula, k=1 ---
            fold_as_group_bounds = test_starts  # reuse same boundaries as "groups"
            train_gen, test_gen = generalized_train_test_positions(
                t1, fold_as_group_bounds, [fold_idx], pct_embargo
            )

            assert np.array_equal(train_gen, train_orig), f'fold {fold_idx} train mismatch'
            assert np.array_equal(test_gen, test_indices_orig), f'fold {fold_idx} test mismatch'

    def test_k2_purges_around_both_test_groups(self):
        # Hand-constructed case: 3 groups of 4, test groups {0, 2}
        # (non-adjacent). An observation in group 1 (the middle,
        # untested group) whose label spans into group 2's time range
        # must be purged even though group 1 itself isn't a test group.
        t0 = pd.date_range('2026-01-01', periods=12, freq='D')
        t1_vals = list(t0)  # start with same-day resolution (no overlap)
        # group boundaries: [0,4)=G0, [4,8)=G1, [8,12)=G2
        # Make obs at position 5 (in G1) resolve deep into G2's span --
        # this must purge it when G2 is a test group.
        t1_vals[5] = t0[10]
        t1 = pd.Series(t1_vals, index=t0)
        group_bounds = [(0, 4), (4, 8), (8, 12)]

        train_pos, test_pos = generalized_train_test_positions(
            t1, group_bounds, test_group_idxs=[0, 2], pct_embargo=0.0
        )

        assert np.array_equal(test_pos, np.array([0, 1, 2, 3, 8, 9, 10, 11]))
        # position 5 must be purged (its label spans into the G2 test block)
        assert 5 not in train_pos
        # position 4, 6, 7 (G1, no overlap into test blocks) remain trainable
        for pos in (4, 6, 7):
            assert pos in train_pos

    def test_never_trains_on_test_rows(self):
        t1 = _make_synthetic_t1(24, avg_span=3, seed=2)
        group_bounds = partition_groups(24, 6)
        for test_groups in enumerate_splits(6, 2)[:5]:
            train_pos, test_pos = generalized_train_test_positions(
                t1, group_bounds, test_groups, pct_embargo=0.05
            )
            assert len(set(train_pos) & set(test_pos)) == 0


# ---------------------------------------------------------------------------
# run_cpcv -- end-to-end orchestration on synthetic data
# ---------------------------------------------------------------------------
class TestRunCPCV:
    def test_every_path_fully_populated_no_nans(self):
        n_obs = 60
        rng = np.random.RandomState(3)
        t1 = _make_synthetic_t1(n_obs, avg_span=3, seed=3)
        X = pd.DataFrame({'feat': rng.randn(n_obs)}, index=t1.index)
        y = pd.Series(np.sign(rng.randn(n_obs)), index=t1.index).replace(0, 1)
        w = pd.Series(rng.uniform(0.5, 1.5, n_obs), index=t1.index)

        path_prob, path_pred, group_bounds, phi = run_cpcv(
            X, y, w, t1, n_groups=6, k=2, pct_embargo=0.05, C=1.0, gamma='scale'
        )

        assert phi == 5
        assert set(path_prob.keys()) == {1, 2, 3, 4, 5}
        for p in range(1, phi + 1):
            assert not np.isnan(path_prob[p]).any(), f'path {p} has unfilled probabilities'
            assert not np.isnan(path_pred[p]).any(), f'path {p} has unfilled predictions'
            assert set(np.unique(path_pred[p])).issubset({-1.0, 1.0})

    def test_reproducible_with_fixed_random_state(self):
        n_obs = 42
        rng = np.random.RandomState(4)
        t1 = _make_synthetic_t1(n_obs, avg_span=3, seed=4)
        X = pd.DataFrame({'feat': rng.randn(n_obs)}, index=t1.index)
        y = pd.Series(np.sign(rng.randn(n_obs)), index=t1.index).replace(0, 1)
        w = pd.Series(rng.uniform(0.5, 1.5, n_obs), index=t1.index)

        r1 = run_cpcv(X, y, w, t1, n_groups=6, k=2, pct_embargo=0.0, C=1.0, gamma='scale', random_state=0)
        r2 = run_cpcv(X, y, w, t1, n_groups=6, k=2, pct_embargo=0.0, C=1.0, gamma='scale', random_state=0)
        for p in r1[0]:
            assert np.allclose(r1[0][p], r2[0][p])

    def test_predictions_invariant_to_per_feature_rescaling(self):
        # LOAD-BEARING regression test for the post-Ch19 StandardScaler fix
        # in fit_predict_split. Before that fix, run_cpcv's SVC fit raw
        # feature magnitude directly into its RBF kernel -- multiplying
        # one column by 1000x (simulating, e.g., Kyle's Lambda sitting
        # next to round_number_fraction) would have completely changed
        # which feature dominates kernel distance, and therefore the
        # predictions. With the fix, StandardScaler removes each column's
        # scale before the SVC ever sees it, so predictions on a
        # per-column-rescaled X must match predictions on the original X.
        n_obs = 48
        rng = np.random.RandomState(7)
        t1 = _make_synthetic_t1(n_obs, avg_span=3, seed=7)
        X = pd.DataFrame({
            'small_scale': rng.randn(n_obs) * 0.001,   # e.g. round_number_fraction's range
            'large_scale': rng.randn(n_obs) * 5000.0,  # e.g. Kyle's Lambda's range
        }, index=t1.index)
        y = pd.Series(np.sign(rng.randn(n_obs)), index=t1.index).replace(0, 1)
        w = pd.Series(rng.uniform(0.5, 1.5, n_obs), index=t1.index)

        # An arbitrary further per-column rescale -- if the classifier were
        # still scale-sensitive, this alone would change every prediction.
        X_rescaled = X.copy()
        X_rescaled['small_scale'] *= 37.0
        X_rescaled['large_scale'] *= 0.0002

        kwargs = dict(n_groups=6, k=2, pct_embargo=0.0, C=1.0, gamma='scale', random_state=0)
        path_prob_orig, path_pred_orig, _, _ = run_cpcv(X, y, w, t1, **kwargs)
        path_prob_rescaled, path_pred_rescaled, _, _ = run_cpcv(X_rescaled, y, w, t1, **kwargs)

        for p in path_prob_orig:
            assert np.allclose(path_prob_orig[p], path_prob_rescaled[p], atol=1e-8), (
                f'path {p} probabilities changed under per-feature rescaling -- '
                f'SVC is not properly scale-invariant'
            )
            assert np.array_equal(path_pred_orig[p], path_pred_rescaled[p])
