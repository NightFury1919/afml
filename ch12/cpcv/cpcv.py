"""
Chapter 12 -- Backtesting through Cross-Validation: Combinatorial Purged CV
============================================================================
Implements the Combinatorial Purged Cross-Validation (CPCV) algorithm from
AFML Section 12.4 (Lopez de Prado).

Book-fidelity note (important, unlike every other chapter so far)
-------------------------------------------------------------------
Chapter 12 has NO printed code snippets -- Sections 12.1-12.5 are pure
motivation, math, and algorithm-in-prose (the numbered steps in 12.4.2),
illustrated by Figures 12.1/12.2. There is nothing to diff against; this
module is built directly from that specification.

The one place the book is genuinely ambiguous is exactly which split
feeds which backtest path. Fig 12.1/12.2 illustrate it but aren't
reproducible from the garbled table text. The book's OWN PROSE, however,
gives the full group/split composition of path 1 and path 2 for its
N=6, k=2 example:
    path 1 = (G1,S1),(G2,S1),(G3,S2),(G4,S3),(G5,S4),(G6,S5)
    path 2 = (G1,S2),(G2,S6),(G3,S6),(G4,S7),(G5,S8),(G6,S9)
That's 12 (group, split) data points, used below as ground truth. See
build_path_assignment()'s docstring and test_cpcv.py::TestPathAssignment
for the derivation and verification.
"""

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Step 1 -- Partition T observations into N groups without shuffling
# ---------------------------------------------------------------------------
def partition_groups(n_obs, n_groups):
    """
    AFML 12.4.1: partition T observations into N groups without shuffling.
    Groups 1..N-1 have size floor(T/N); the Nth group absorbs whatever's
    left over (size T - floor(T/N)*(N-1)).

    Why not just split evenly? T rarely divides evenly by N, and the book
    is explicit that the remainder goes entirely to the last group rather
    than being spread out -- keeps every group's start/end boundary a
    clean multiple of floor(T/N) except the very last one.

    Returns
    -------
    list of (start, end) position tuples, one per group, `end` exclusive,
    in original chronological order.
    """
    if n_groups < 2:
        raise ValueError('n_groups must be >= 2 (need at least train + test)')
    base = n_obs // n_groups
    bounds = []
    start = 0
    for _ in range(n_groups - 1):
        end = start + base
        bounds.append((start, end))
        start = end
    bounds.append((start, n_obs))  # last group takes the remainder
    return bounds


# ---------------------------------------------------------------------------
# Step 2 -- Compute all possible training/testing splits
# ---------------------------------------------------------------------------
def enumerate_splits(n_groups, k):
    """
    All C(n_groups, k) combinations of TEST group indices, in standard
    lexicographic order (itertools.combinations). A split's train groups
    are implicitly every group not in the returned test tuple.
    """
    if not (1 <= k <= n_groups - 1):
        raise ValueError('k must satisfy 1 <= k <= n_groups - 1')
    return list(combinations(range(n_groups), k))


def n_paths(n_groups, k):
    """phi[N,k] = k * C(N,k) / N -- AFML eq. in Section 12.4.1."""
    return k * comb(n_groups, k) // n_groups


# ---------------------------------------------------------------------------
# The path-assignment algorithm (reverse-engineered from book prose, see
# module docstring)
# ---------------------------------------------------------------------------
def build_path_assignment(n_groups, k):
    """
    Assigns every (split, test-group) occurrence to one of the phi[N,k]
    backtest paths.

    Rule: for each group g, list every split (in enumerate_splits order,
    i.e. ascending split index) where g is a member of the test set.
    Group g's m-th such occurrence (1-indexed) becomes g's contribution
    to path m.

    Why this is correct: every group is a test group in exactly phi[N,k]
    splits (uniform by construction -- see 12.4.1), so this produces
    exactly phi paths, and every path ends up with exactly one
    contribution from every one of the N groups (a complete, T-length
    forecast). Verified against the book's own explicit path-1/path-2
    text for N=6,k=2 -- see test_cpcv.py.

    Returns
    -------
    assignment : dict {(split_idx, group_idx): path_number (1-indexed)}
    phi        : int, number of paths
    splits     : list of test-group-index tuples (same as enumerate_splits)
    """
    splits = enumerate_splits(n_groups, k)
    phi = n_paths(n_groups, k)
    assignment = {}
    for g in range(n_groups):
        occurrences = [s_idx for s_idx, test_groups in enumerate(splits) if g in test_groups]
        for path_num, s_idx in enumerate(occurrences, start=1):
            assignment[(s_idx, g)] = path_num
    return assignment, phi, splits


# ---------------------------------------------------------------------------
# Step 3 -- Generalized purge + embargo (AFML Ch07 Snippet 7.3's formula,
# extended from one contiguous test block to k simultaneous, possibly
# non-adjacent test groups)
# ---------------------------------------------------------------------------
def generalized_train_test_positions(t1, group_bounds, test_group_idxs, pct_embargo):
    """
    Generalizes Ch07 PurgedKFold's purge+embargo logic (which only
    supports one contiguous test block per split) to CPCV's k
    simultaneous test groups.

    Book Step 3: "For any pair of labels (yi, yj), where yi belongs to
    the training set and yj belongs to the testing set, apply the
    PurgedKFold class to purge yi if yi spans over a period used to
    determine label yj." A candidate training observation must therefore
    be safe with respect to EVERY held-out test group at once, not just
    one -- so we apply Ch07's exact per-block formula independently to
    each test group, then intersect the "safe" sets across all of them.

    Per-block formula (identical math to Ch07 PurgedKFold.split(), just
    applied once per test group instead of once per fold):
      leading-safe  = observations whose OWN label already resolved
                      before this block starts (t1[g] <= block's t0)
      trailing-safe = observations starting at/after this block's last
                      label resolution + an embargo gap of
                      int(n_obs * pct_embargo) further positions
      safe wrt this block = leading-safe OR trailing-safe

    Sanity: for k=1 (a single test group), this reduces to exactly the
    same train/test split Ch07's PurgedKFold produces for an equivalent
    single contiguous fold -- see test_cpcv.py::TestPurgeEmbargo.

    Parameters
    ----------
    t1 : pd.Series, index = observation start time (sorted), values =
        label end time. Same object PurgedKFold expects.
    group_bounds : list of (start, end) position tuples for ALL N groups
        (output of partition_groups).
    test_group_idxs : iterable of group indices held out as test in this
        split.
    pct_embargo : float

    Returns
    -------
    train_positions : np.ndarray of int positions safe to train on
    test_positions  : np.ndarray of int positions in the test set (union
        of the test groups, ascending position order)
    """
    n_obs = len(t1)
    mbrg = int(n_obs * pct_embargo)
    positions = np.arange(n_obs)
    test_group_idxs = set(test_group_idxs)

    test_positions = np.concatenate(
        [positions[s:e] for i, (s, e) in enumerate(group_bounds) if i in test_group_idxs]
    )
    test_positions.sort()

    safe_masks = []
    for i in test_group_idxs:
        s, e = group_bounds[i]
        t0_block = t1.index[s]
        max_t1_block = t1.iloc[s:e].max()
        max_t1_idx = t1.index.searchsorted(max_t1_block)

        leading_safe = (t1.values <= np.datetime64(t0_block))
        trailing_safe = np.zeros(n_obs, dtype=bool)
        if max_t1_idx < n_obs:
            trailing_safe[max_t1_idx + mbrg:] = True
        safe_masks.append(leading_safe | trailing_safe)

    safe_wrt_all_blocks = np.logical_and.reduce(safe_masks)
    train_mask = safe_wrt_all_blocks.copy()
    train_mask[test_positions] = False  # never train on the test rows themselves
    train_positions = positions[train_mask]
    return train_positions, test_positions


# ---------------------------------------------------------------------------
# Step 4 -- Fit classifiers on the C(N,N-k) training sets, predict on the
# respective testing sets
# ---------------------------------------------------------------------------
def fit_predict_split(X, y, w, train_pos, test_pos, C, gamma, random_state=0):
    """
    Fit SVC(C, gamma, probability=True) on one split's purged/embargoed
    training positions, predict_proba on its test positions. Mirrors
    Ch10's out_of_sample_probs loop body exactly (same model, same
    random_state-pinning rationale -- SVC(probability=True)'s internal
    Platt-scaling CV is otherwise non-deterministic).
    """
    clf = SVC(C=C, gamma=gamma, probability=True, random_state=random_state)
    clf.fit(X.iloc[train_pos, :], y.iloc[train_pos], sample_weight=w.iloc[train_pos].values)
    proba = clf.predict_proba(X.iloc[test_pos, :])
    idx_max = proba.argmax(axis=1)
    prob = proba[np.arange(len(test_pos)), idx_max]
    pred = clf.classes_[idx_max]
    return prob, pred


# ---------------------------------------------------------------------------
# Steps 3-5 orchestrated -- the full CPCV backtesting algorithm
# ---------------------------------------------------------------------------
def run_cpcv(X, y, w, t1, n_groups, k, pct_embargo, C, gamma, random_state=0):
    """
    Runs AFML 12.4.2 steps 1-4: partitions groups, enumerates every
    train/test split, purges+embargoes each one, fits a classifier per
    split, and reassembles every split's test predictions into complete,
    per-path forecast series (step 5's inputs).

    Returns
    -------
    path_prob, path_pred : dict {path_number: np.ndarray of length n_obs}
        Winning-class probability / predicted label at every one of the
        n_obs original positions, for each of the phi backtest paths.
    group_bounds : list of (start, end) position tuples
    phi : int, number of paths
    """
    n_obs = len(t1)
    group_bounds = partition_groups(n_obs, n_groups)
    assignment, phi, splits = build_path_assignment(n_groups, k)

    path_prob = {p: np.full(n_obs, np.nan) for p in range(1, phi + 1)}
    path_pred = {p: np.full(n_obs, np.nan) for p in range(1, phi + 1)}

    for s_idx, test_group_idxs in enumerate(splits):
        train_pos, test_pos = generalized_train_test_positions(
            t1, group_bounds, test_group_idxs, pct_embargo
        )
        prob, pred = fit_predict_split(X, y, w, train_pos, test_pos, C, gamma, random_state)
        pos_to_local = {pos: i for i, pos in enumerate(test_pos)}

        for g in test_group_idxs:
            s, e = group_bounds[g]
            path_num = assignment[(s_idx, g)]
            for pos in range(s, e):
                local_i = pos_to_local[pos]
                path_prob[path_num][pos] = prob[local_i]
                path_pred[path_num][pos] = pred[local_i]

    return path_prob, path_pred, group_bounds, phi


# ---------------------------------------------------------------------------
# Pytest results (sandbox validation -- Python 3.12.3, pandas 3.0.2,
# scipy 1.17.1, numpy 2.4.4, sklearn 1.8.0). Confirmed on real mlfinlab
# env (Python 3.10.20 / pandas 1.5.3 / sklearn 1.2.2) -- 17/17 pass,
# identical results -- see project chat, July 2026.
#
# Real-machine gotcha (not a code bug): bare `pytest` initially failed
# with ImportError: cannot import name 'partition_groups' from 'cpcv' --
# pytest's rootdir-insertion resolved `cpcv` to the *package*
# ch12\cpcv\__init__.py instead of the *module* ch12\cpcv\cpcv.py,
# because ch12\__init__.py was missing. Fixed by adding it. Standing
# convention going forward: invoke tests as `python -m pytest`, not bare
# `pytest`, wherever a module and its containing folder share a name.
#
# The golden test (TestPathAssignment::test_reproduces_book_path1_and_path2)
# reproduces AFML's own Fig 12.1/12.2 path 1 and path 2 compositions
# verbatim from the book's prose (no printed code exists for this chapter
# to diff against -- see module docstring). TestPurgeEmbargo::
# test_k1_matches_original_purged_kfold is a regression check against
# Ch07 PurgedKFold's exact formula for the k=1 special case.
#
# ============================= test session starts ==============================
# platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
# collected 17 items
#
# test_cpcv.py::TestPartitionGroups::test_book_example_T88_N6 PASSED
# test_cpcv.py::TestPartitionGroups::test_evenly_divisible PASSED
# test_cpcv.py::TestPartitionGroups::test_rejects_fewer_than_two_groups PASSED
# test_cpcv.py::TestSplitCounts::test_book_example_15_splits PASSED
# test_cpcv.py::TestSplitCounts::test_book_example_5_paths PASSED
# test_cpcv.py::TestSplitCounts::test_k1_reduces_to_plain_cv PASSED
# test_cpcv.py::TestSplitCounts::test_k2_rule_of_thumb_N_minus_1_paths PASSED
# test_cpcv.py::TestSplitCounts::test_splits_are_lexicographic_combinations PASSED
# test_cpcv.py::TestSplitCounts::test_rejects_k_out_of_range PASSED
# test_cpcv.py::TestPathAssignment::test_reproduces_book_path1_and_path2 PASSED
# test_cpcv.py::TestPathAssignment::test_every_group_contributes_exactly_once_per_path PASSED
# test_cpcv.py::TestPathAssignment::test_every_group_is_test_group_in_exactly_phi_splits PASSED
# test_cpcv.py::TestPurgeEmbargo::test_k1_matches_original_purged_kfold PASSED
# test_cpcv.py::TestPurgeEmbargo::test_k2_purges_around_both_test_groups PASSED
# test_cpcv.py::TestPurgeEmbargo::test_never_trains_on_test_rows PASSED
# test_cpcv.py::TestRunCPCV::test_every_path_fully_populated_no_nans PASSED
# test_cpcv.py::TestRunCPCV::test_reproducible_with_fixed_random_state PASSED
#
# ============================== 17 passed in 1.23s ===============================
#
# Real mlfinlab machine (Python 3.10.20 / pandas 1.5.3 / sklearn 1.2.2):
# ====================================================================== test session starts =======================================================================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# collected 17 items
# [... all 17 PASSED, identical to above ...]
# ======================================================================= 17 passed in 2.61s ========================================================================
# ---------------------------------------------------------------------------
