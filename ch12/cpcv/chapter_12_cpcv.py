"""
Chapter 12 -- Combinatorial Purged Cross-Validation: real-data demo
====================================================================
Runs CPCV end-to-end on the real 88-row BTC/TUSD triple-barrier table
(Ch03-05), using N=6 groups / k=2 test groups per split (mirrors AFML's
own Fig 12.1/12.2 worked example exactly: 15 splits, 5 backtest paths),
Ch09's real winning SVC hyperparameters (C=100, gamma=0.1), and Ch10's
real getSignal pipeline to turn each path's forecasts into position
sizes, then real per-event returns (ch03_events.csv's 'ret' column) to
get an actual Sharpe ratio per path.

Why this matters (the whole point of the chapter): a single walk-forward
or plain-CV backtest gives you ONE Sharpe ratio -- no way to tell luck
from skill. CPCV gives you phi=5 genuinely different backtest paths on
the SAME data, so you get a distribution. A tight distribution is
real evidence; a wide one (or one that straddles zero) means the
single-path Sharpe you'd have reported was mostly noise.
"""
import os
import sys

AFML_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from cpcv import run_cpcv, partition_groups, enumerate_splits, n_paths  # noqa: E402

sys.path.insert(0, os.path.join(AFML_ROOT, 'ch10', 'bet_sizing'))
from bet_sizing import getSignal  # noqa: E402

sys.path.insert(0, os.path.join(AFML_ROOT, 'ch07', 'cross_validation'))
try:
    from purged_kfold import PurgedKFold  # noqa: E402
except ImportError:
    PurgedKFold = None  # single-path baseline skipped if Ch07 isn't on this machine's path

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')

N_GROUPS = 6            # mirrors AFML Fig 12.1/12.2 exactly
K_TEST_GROUPS = 2
PCT_EMBARGO = 0.12       # matches Ch07/Ch10's real-data calibration for this dataset
STEP_SIZE = 0.01         # matches Ch10
# LOAD-BEARING (2026-07-22): SVC_C was 100.0 from Ch12's original commit through
# the Ch19-enrichment commit (97a5101) -- that commit fixed the StandardScaler
# bug but never migrated this constant, unlike Ch10/Ch11's SVC_C. 100.0 was
# Ch09's PRE-enrichment grid-search winner; the post-enrichment winner (what
# Ch10/Ch11 use) is 0.01. All Ch12/Ch14 numbers committed before this fix were
# produced under the stale, pre-enrichment value.
SVC_C = 0.01              # Ch09's real-data grid-search winner, POST-Ch19-enrichment
SVC_GAMMA = 0.1
RANDOM_STATE = 0         # SVC(probability=True) determinism -- see Ch09/Ch10 handoffs

# single-path baseline, for comparison -- same calibration Ch10 used
BASELINE_N_SPLITS = 4
BASELINE_PCT_EMBARGO = 0.12


def load_data():
    """Real Ch03-07(+19) BTC/TUSD pipeline: features/labels/weights/t1 from
    the enriched training table (fracdiff + Ch19's 11 microstructural
    features, falling back to the original fracdiff-only table if the
    enriched artifact isn't present), realized per-event returns from
    ch03_events.csv.

    Note: the enriched table has 87 rows (ch03_events has 88) -- one event
    was dropped by build_enriched_training_table.py for still being inside
    a Ch19 rolling-window warmup period. events is aligned to table's
    index (a strict equality check would incorrectly reject this)."""
    enriched_path = os.path.join(INPUT_DIR, 'ch07_training_table_enriched.csv')
    if os.path.exists(enriched_path):
        table = pd.read_csv(enriched_path, index_col=0, parse_dates=[0, 't1'])
    else:
        table = pd.read_csv(
            os.path.join(INPUT_DIR, 'ch07_training_table.csv'), index_col=0, parse_dates=[0, 't1']
        )
    events = pd.read_csv(
        os.path.join(INPUT_DIR, 'ch03_events.csv'), index_col=0, parse_dates=[0, 't1']
    )
    if not table.index.isin(events.index).all():
        raise ValueError('every row of the training table must be a real Ch03 event')
    events = events.loc[table.index]  # align (enriched table may be a subset of events)

    feature_cols = [c for c in table.columns if c not in ('bin', 'w', 't1')]
    X = table[feature_cols]
    y = table['bin']
    w = table['w']
    t1 = table['t1']
    ret = events['ret']
    return X, y, w, t1, ret


def path_to_signal_and_returns(t1, ret, prob_arr, pred_arr, step_size=STEP_SIZE):
    """One CPCV path's raw prob/pred arrays -> Ch10's real getSignal ->
    discretized bet size -> realized position return per event."""
    events = t1.to_frame('t1')
    prob_s = pd.Series(prob_arr, index=t1.index)
    pred_s = pd.Series(pred_arr, index=t1.index)
    signal = getSignal(events, stepSize=step_size, prob=prob_s, pred=pred_s,
                        numClasses=2, numThreads=1)
    signal = signal.reindex(t1.index).fillna(0.0)
    pos_returns = signal * ret
    sharpe = pos_returns.mean() / pos_returns.std(ddof=1) if pos_returns.std(ddof=1) > 0 else np.nan
    return signal, pos_returns, sharpe


def single_path_baseline(X, y, w, t1, ret):
    """The Ch10-style single out-of-sample path (plain PurgedKFold,
    n_splits=4) -- the one-Sharpe-ratio result CPCV is meant to improve on.

    Uses the same StandardScaler-wrapped SVC as cpcv.py's fit_predict_split
    (post-Ch19-enrichment fix -- see that function's docstring for why an
    unscaled SVC silently breaks once X has multiple, differently-scaled
    feature columns instead of just fracdiff)."""
    if PurgedKFold is None:
        return None
    gen = PurgedKFold(n_splits=BASELINE_N_SPLITS, t1=t1, pctEmbargo=BASELINE_PCT_EMBARGO)
    prob = pd.Series(index=X.index, dtype=float)
    pred = pd.Series(index=X.index, dtype=float)
    for train, test in gen.split(X=X):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=SVC_C, gamma=SVC_GAMMA, probability=True, random_state=RANDOM_STATE)),
        ])
        pipe.fit(X.iloc[train, :], y.iloc[train], svc__sample_weight=w.iloc[train].values)
        proba = pipe.predict_proba(X.iloc[test, :])
        idx_max = proba.argmax(axis=1)
        prob.iloc[test] = proba[np.arange(len(test)), idx_max]
        pred.iloc[test] = pipe.named_steps['svc'].classes_[idx_max]
    _, pos_returns, sharpe = path_to_signal_and_returns(t1, ret, prob.values, pred.values)
    return sharpe, pos_returns


def variance_reduction_check(path_returns_df, phi):
    """AFML Section 12.5: sigma^2[mu_i] = phi^-1 * sigma_i^2 * (1 + (phi-1)*rho_bar)
    where sigma_i^2 is the cross-path variance of the Sharpe estimate and
    rho_bar is the average off-diagonal correlation among path return
    series. Demonstrates the chapter's central claim -- CPCV's averaged
    estimate has lower variance than any single path's, as long as
    rho_bar < 1."""
    sharpes = path_returns_df.apply(lambda s: s.mean() / s.std(ddof=1))
    sigma_i2 = sharpes.var(ddof=1)
    corr = path_returns_df.corr()
    off_diag = corr.values[~np.eye(len(corr), dtype=bool)]
    rho_bar = off_diag.mean()
    var_of_mean = (sigma_i2 / phi) * (1 + (phi - 1) * rho_bar)
    return {
        'sigma_i2': sigma_i2,
        'rho_bar': rho_bar,
        'var_of_mean_sharpe': var_of_mean,
        'single_path_variance': sigma_i2,
    }


def main():
    X, y, w, t1, ret = load_data()
    n_obs = len(t1)
    print(f'Loaded {n_obs} real BTC/TUSD triple-barrier events.')
    print(f'Features ({X.shape[1]}): {list(X.columns)}')
    print(f'N={N_GROUPS} groups, k={K_TEST_GROUPS} test groups/split -> '
          f'{len(enumerate_splits(N_GROUPS, K_TEST_GROUPS))} splits, '
          f'{n_paths(N_GROUPS, K_TEST_GROUPS)} backtest paths.')
    print('Group sizes:', [e - s for s, e in partition_groups(n_obs, N_GROUPS)])

    path_prob, path_pred, group_bounds, phi = run_cpcv(
        X, y, w, t1, n_groups=N_GROUPS, k=K_TEST_GROUPS, pct_embargo=PCT_EMBARGO,
        C=SVC_C, gamma=SVC_GAMMA, random_state=RANDOM_STATE
    )

    path_returns = {}
    path_sharpes = {}
    path_mean_abs_bet = {}
    for p in range(1, phi + 1):
        signal, pos_returns, sharpe = path_to_signal_and_returns(
            t1, ret, path_prob[p], path_pred[p]
        )
        path_returns[p] = pos_returns
        path_sharpes[p] = sharpe
        path_mean_abs_bet[p] = signal.abs().mean()
        print(f'  path {p}: Sharpe = {sharpe:.4f}  '
              f'(mean bet size {signal.abs().mean():.3f}, '
              f'{(signal != 0).sum()}/{n_obs} nonzero bets)')

    returns_df = pd.DataFrame(path_returns)
    sharpe_series = pd.Series(path_sharpes)
    print('\nCPCV Sharpe-ratio distribution across paths:')
    print(sharpe_series.describe())

    baseline = single_path_baseline(X, y, w, t1, ret)
    if baseline is not None:
        baseline_sharpe, _ = baseline
        print(f'\nSingle-path (Ch10-style, n_splits={BASELINE_N_SPLITS}) baseline Sharpe: '
              f'{baseline_sharpe:.4f}')
        print('(This is the ONE number a walk-forward or plain-CV backtest would '
              'have reported -- compare it to the spread above.)')

    var_check = variance_reduction_check(returns_df, phi)
    print('\nSection 12.5 variance-reduction check:')
    print(f'  cross-path Sharpe variance (sigma_i^2): {var_check["sigma_i2"]:.6f}')
    print(f'  average pairwise path-return correlation (rho_bar): {var_check["rho_bar"]:.4f}')
    print(f'  implied variance of the CPCV mean Sharpe: {var_check["var_of_mean_sharpe"]:.6f}')

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(sharpe_series.index.astype(str), sharpe_series.values, color='steelblue')
    axes[0].axhline(0, color='gray', linewidth=0.8)
    if baseline is not None:
        axes[0].axhline(baseline_sharpe, color='firebrick', linestyle='--',
                         label=f'single-path baseline ({baseline_sharpe:.3f})')
        axes[0].legend()
    axes[0].set_title('CPCV: Sharpe ratio per backtest path')
    axes[0].set_xlabel('path')
    axes[0].set_ylabel('Sharpe ratio')

    cumret = returns_df.fillna(0.0).cumsum()
    for p in cumret.columns:
        axes[1].plot(cumret.index, cumret[p], label=f'path {p}', alpha=0.8)
    axes[1].set_title('CPCV: cumulative position return per path')
    axes[1].set_ylabel('cumulative return')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=30)

    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), 'ch12_cpcv_paths.png')
    fig.savefig(out_png, dpi=120)
    print(f'\nSaved plot: {out_png}')

    # --- save artifact, matching repo convention ---
    stats_df = pd.DataFrame({
        'path': list(range(1, phi + 1)),
        'sharpe': [path_sharpes[p] for p in range(1, phi + 1)],
        'mean_abs_bet_size': [path_mean_abs_bet[p] for p in range(1, phi + 1)],
    })
    stats_df['single_path_baseline_sharpe'] = baseline_sharpe if baseline is not None else np.nan
    stats_df['rho_bar'] = var_check['rho_bar']
    stats_df['var_of_mean_sharpe'] = var_check['var_of_mean_sharpe']

    artifact_csv = os.path.join(INPUT_DIR, 'ch12_cpcv_stats.csv')
    artifact_pkl = os.path.join(INPUT_DIR, 'ch12_cpcv_stats.pkl')
    stats_df.to_csv(artifact_csv, index=False)
    stats_df.to_pickle(artifact_pkl)
    print(f'Saved artifact: {artifact_csv} / {artifact_pkl}')

    return stats_df, returns_df


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# TDD results mirror -- same suite as cpcv.py's embedded block and this
# chapter's notebook, duplicated here per the .py/.ipynb mirror convention
# (this script is the ipynb's paired mirror, not cpcv.py).
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
