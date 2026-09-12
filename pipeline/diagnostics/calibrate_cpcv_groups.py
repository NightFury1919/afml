"""
pipeline/diagnostics/calibrate_cpcv_groups.py

Ch12's CPCV was validated standalone against the static 88-row BTC/TUSD
table using N_GROUPS=6/K_TEST_GROUPS=2 (mirrors AFML's own Fig 12.1/12.2
worked example) -- but that N/k was never derived for LIVE snapshot sizes.
Live T_effective ranges from ~102-355 (TAO) to ~250-758 (BTC/XRP, per
kraken_target_bars_calibration.csv) -- small enough that a naive N/k could
leave held-out test groups (or purged/embargoed train folds) too thin to
fit meaningfully, especially for TAO.

Two-phase design (cheap structural check first, expensive real fit second)
-- same "don't pay for what you can rule out first" principle as the
Kraken CUSUM_H two-pass validation:

  Phase A (no model fits): for every (n_groups, k) candidate, compute
    partition_groups()/enumerate_splits()/n_paths() and, for every split,
    generalized_train_test_positions() to get real train/test sizes and
    each split's train-side class balance. Flags candidates where any
    split's train fold is single-class (can't fit) or test group would
    be smaller than MIN_GROUP_SIZE.
  Phase B (real fits): only candidates that pass Phase A get a real
    run_cpcv() call (actual SVC fits per split) plus Ch10's real
    getSignal -> per-path Sharpe, exactly mirroring chapter_12_cpcv.py's
    own path_to_signal_and_returns() scoring.

Reuses rebuild.py/features.py/live_staging.py/stages.py exactly as
calibrate_kraken_target_bars.py does -- n_groups/k is the only thing
varied here. C is NOT a fixed constant (unlike the static Ch12 driver's
SVC_C=0.01): it is read from THIS live run's own Ch11 trial grid winner
(evaluate_overfitting()'s best_trial), matching the project's live-data
principle of never assuming a static-dataset constant transfers. GAMMA=0.1,
PCT_EMBARGO=0.12, STEP_SIZE=0.01 are reused unchanged -- these are Ch10/
Ch11's own established real calibration for this pipeline, not specific to
the static dataset's size.

Run (after capture_kraken_snapshot.py has produced a snapshot):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_cpcv_groups.py <snapshot_dir> <target_bars> [output_csv]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ROOT)
sys.path.insert(0, ORCH)

from rebuild import build_bars_and_labels                # noqa: E402
from features import build_enriched_events                # noqa: E402
from live_staging import stage_live_training_tables        # noqa: E402
from stages import (                                      # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

CH12_CPCV = os.path.join(ROOT, 'ch12', 'cpcv')
if CH12_CPCV not in sys.path:
    sys.path.insert(0, CH12_CPCV)
from cpcv import (                                         # noqa: E402
    partition_groups, enumerate_splits, n_paths,
    generalized_train_test_positions, run_cpcv,
)

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                            # noqa: E402

OUTPUT_CSV = os.path.join(HERE, 'cpcv_groups_calibration.csv')

# Reused unchanged from Ch11 (GAMMA, PCT_EMBARGO) and Ch10/Ch12 (STEP_SIZE)
# -- see module docstring. NOT re-derived here; only n_groups/k are swept.
GAMMA = 0.1
PCT_EMBARGO = 0.12
STEP_SIZE = 0.01
RANDOM_STATE = 0

# Candidate grid. N values chosen to stay usable down to TAO's ~100-bar
# floor (min group size at N=10 on 100 obs is already only 10); k values
# capped at N//2 (k >= N//2 leaves too few groups actually in-sample per
# split to be a meaningful "combinatorial" test, and phi(N,k) explodes).
N_GROUPS_GRID = [4, 5, 6, 8, 10]
K_GRID = [1, 2, 3]
MIN_GROUP_SIZE = 15   # floor below which a held-out group is treated as
                       # too thin to trust a single split's prediction on
                       # -- not book-derived, a first-pass heuristic to
                       # revisit once real numbers come back (same status
                       # as report.py's own T<150/n_trials<10 warnings)
MIN_TRAIN_SIZE = 30    # same status as MIN_GROUP_SIZE above

SWEEP_COLUMNS = [
    'n_groups', 'k', 'phi', 'genuine_cpcv', 'n_splits', 'n_obs',
    'min_group_size', 'min_train_size', 'phase_a_pass',
    'phase_a_fail_reason', 'phase_b_attempted', 'phase_b_error',
    'mean_path_sharpe', 'std_path_sharpe', 'min_path_sharpe',
    'max_path_sharpe', 'n_paths_negative', 'winning_C', 'notes',
]

# *** LOAD-BEARING (2026-09-10): k=1 is NOT real CPCV, and the first live
# run's output made that ambiguous ***
# phi(N, k=1) = 1 -- there is exactly one train/test split, so "Phase B"
# for k=1 is a single ordinary holdout wearing CPCV's machinery, not the
# combinatorial multi-path distribution CPCV exists to produce (the whole
# point of Ch12: one path can't distinguish luck from skill). It still
# structurally "passes" Phase A (nothing to combine, so nothing can be
# infeasible), which let it sit in the same table looking equivalent to
# a genuine phi>1 result in the first live run. genuine_cpcv=True marks
# the rows that are actually usable as CPCV evidence.


def _phase_a_check(n_obs, y, t1, n_groups, k, pct_embargo=PCT_EMBARGO):
    """Structural feasibility only -- no model fits. Returns a dict with
    pass/fail plus the sizes that drove the decision."""
    if n_groups > n_obs:
        return {'pass': False, 'reason': 'n_groups > n_obs',
                'phi': None, 'n_splits': None,
                'min_group_size': None, 'min_train_size': None}

    group_bounds = partition_groups(n_obs, n_groups)
    group_sizes = [e - s for s, e in group_bounds]
    min_group_size = min(group_sizes)

    splits = enumerate_splits(n_groups, k)
    phi = n_paths(n_groups, k)

    min_train_size = None
    for test_group_idxs in splits:
        train_pos, test_pos = generalized_train_test_positions(
            t1, group_bounds, test_group_idxs, pct_embargo,
        )
        if len(train_pos) == 0:
            return {'pass': False, 'reason': 'empty train fold',
                    'phi': phi, 'n_splits': len(splits),
                    'min_group_size': min_group_size,
                    'min_train_size': 0}
        y_train = y.iloc[train_pos] if hasattr(y, 'iloc') else y[train_pos]
        if len(np.unique(y_train)) < 2:
            return {'pass': False, 'reason': 'single-class train fold',
                    'phi': phi, 'n_splits': len(splits),
                    'min_group_size': min_group_size,
                    'min_train_size': len(train_pos)}
        if min_train_size is None or len(train_pos) < min_train_size:
            min_train_size = len(train_pos)

    if min_group_size < MIN_GROUP_SIZE:
        return {'pass': False,
                'reason': f'min_group_size {min_group_size} < {MIN_GROUP_SIZE}',
                'phi': phi, 'n_splits': len(splits),
                'min_group_size': min_group_size,
                'min_train_size': min_train_size}
    if min_train_size < MIN_TRAIN_SIZE:
        return {'pass': False,
                'reason': f'min_train_size {min_train_size} < {MIN_TRAIN_SIZE}',
                'phi': phi, 'n_splits': len(splits),
                'min_group_size': min_group_size,
                'min_train_size': min_train_size}

    return {'pass': True, 'reason': '', 'phi': phi, 'n_splits': len(splits),
            'min_group_size': min_group_size, 'min_train_size': min_train_size}


def _path_sharpe(t1, ret, prob_arr, pred_arr, step_size=STEP_SIZE):
    """Same real scoring as ch12/chapter_12_cpcv.py's path_to_signal_and_
    returns(): raw prob/pred arrays -> Ch10's real getSignal -> position
    returns -> Sharpe. Reused logic, not reimplemented math."""
    prob_s = pd.Series(prob_arr, index=t1.index)
    pred_s = pd.Series(pred_arr, index=t1.index)
    events = pd.DataFrame({'t1': t1})
    signal = getSignal(events, stepSize=step_size, prob=prob_s, pred=pred_s,
                        numClasses=2, numThreads=1)
    if signal.empty:
        return np.nan, 0
    pos = signal.reindex(ret.index, method='ffill').fillna(0.0)
    pos_returns = (pos.shift(1).fillna(0.0) * ret).dropna()
    pos_returns = pos_returns[pos_returns != 0]
    if len(pos_returns) < 2 or pos_returns.std(ddof=1) == 0:
        return np.nan, len(pos_returns)
    sharpe = pos_returns.mean() / pos_returns.std(ddof=1)
    return float(sharpe), len(pos_returns)


def _phase_b_run(X, y, w, t1, ret, n_groups, k, C):
    path_prob, path_pred, group_bounds, phi = run_cpcv(
        X, y, w, t1, n_groups, k, PCT_EMBARGO, C, GAMMA,
        random_state=RANDOM_STATE,
    )
    sharpes = []
    for p in range(1, phi + 1):
        sharpe, n_bets = _path_sharpe(t1, ret, path_prob[p], path_pred[p])
        if not np.isnan(sharpe):
            sharpes.append(sharpe)
    if not sharpes:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan,
                'max': np.nan, 'n_negative': 0}
    arr = np.array(sharpes)
    return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            'min': float(arr.min()), 'max': float(arr.max()),
            'n_negative': int((arr < 0).sum())}


def _parse_args(argv):
    """Positional: snapshot_dir, target_bars, [output_csv]. Optional flags
    anywhere: --drop=feat1,feat2 (see _ablate_features), --detrend (see
    below).
    --detrend subtracts ret.mean() from the event-level 'ret' series
    before it ever reaches _path_sharpe -- same trend-adjustment
    principle as calibrate_pbo_detrended.py, applied to CPCV's own
    event-level return series rather than Part C's bar-level one (see
    that script's SCOPE NOTE for why they're separate). Added 2026-09-10
    after diagnose_window_trend_bias.py found this window's low PBO was
    consistent with a trending-regime artifact rather than real edge."""
    positional = [a for a in argv if not a.startswith('--drop=') and a != '--detrend']
    drop_args = [a for a in argv if a.startswith('--drop=')]
    detrend = '--detrend' in argv
    if len(positional) not in (2, 3):
        raise SystemExit(
            'Usage: python calibrate_cpcv_groups.py <snapshot_dir> <target_bars> '
            '[output_csv] [--drop=feat1,feat2] [--detrend]\n'
            'Run capture_kraken_snapshot.py first to produce a snapshot.'
        )
    snapshot_dir = positional[0]
    target_bars = int(positional[1])
    output_csv = positional[2] if len(positional) == 3 else OUTPUT_CSV
    drop_features = drop_args[0].split('=', 1)[1].split(',') if drop_args else []
    drop_features = [f.strip() for f in drop_features if f.strip()]
    return snapshot_dir, target_bars, output_csv, drop_features, detrend


def _ablate_features(enriched_result, drop_features):
    """Returns a COPY of enriched_result with drop_features removed from
    both 'feature_table' and 'enriched_events' (no-op for any name not
    actually present -- fails loud instead if a requested name is
    missing, so a typo doesn't silently ablate nothing)."""
    if not drop_features:
        return enriched_result
    missing = [f for f in drop_features if f not in enriched_result['feature_table'].columns]
    if missing:
        raise SystemExit(f'--drop requested columns not in feature_table: {missing}')
    ablated = dict(enriched_result)
    ablated['feature_table'] = enriched_result['feature_table'].drop(columns=drop_features)
    ablated['enriched_events'] = enriched_result['enriched_events'].drop(columns=drop_features)
    return ablated


def main():
    snapshot_dir, target_bars, output_csv, drop_features, detrend = _parse_args(sys.argv[1:])
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')
    print(f'target_bars={target_bars}   writing results to: {output_csv}')

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    print(f"  {len(rebuild_result['bars'])} bars, {len(rebuild_result['events'])} events, "
          f"threshold=${rebuild_result['threshold']:,.2f}")

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f"  {enriched_result['n_events_after']}/{enriched_result['n_events_before']} "
          f"events survived enrichment")

    if drop_features:
        print(f'  ABLATION: dropping {drop_features} from the feature set '
              f'(both the Ch11 trial grid and the CPCV sweep below)')
        enriched_result = _ablate_features(enriched_result, drop_features)

    work_root = os.path.join(HERE, 'cpcv_groups_sweep_work')
    staging_dir = os.path.join(work_root, 'staging')
    here_dir = os.path.join(work_root, 'ch11_here')
    stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)

    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')
    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)
    winning_C = float(meta.loc[eval_result['best_trial'], 'C'])
    print(f"  live trial grid: winning C={winning_C}, "
          f"PBO={eval_result['prob_overfit']:.4f}, DSR={eval_result['dsr']:.4f}")

    # *** matches live_staging.py's own LOAD-BEARING note exactly: ***
    # enriched_events still carries rebuild.py's t1/trgt/ret/bin columns
    # alongside the real features -- 'trgt'/'ret' must be excluded from
    # feature_cols here too, or they'd leak in as bogus features exactly
    # the way live_staging.py's own docstring warns against. 'w' is NOT
    # a column of enriched_events at all (it's a separate pre-enrichment
    # Series on rebuild_result); reindex it the same way
    # evaluate_overfitting()'s tw_aligned already does above.
    enriched = enriched_result['enriched_events']
    feature_cols = list(enriched_result['feature_table'].columns)
    X = enriched[feature_cols]
    y = enriched['bin']
    t1 = enriched['t1']
    ret = enriched['ret']
    if detrend:
        ret_mean = ret.mean()
        print(f'  DETREND: subtracting ret.mean()={ret_mean:.6f} from the '
              f'event-level return series before CPCV path Sharpes')
        ret = ret - ret_mean
    w = rebuild_result['w'].reindex(enriched.index)
    if w.isna().any():
        raise ValueError('w has NaN after reindexing to the enriched event index.')
    n_obs = len(X)
    print(f'  CPCV feasibility sweep over n_obs={n_obs} events')

    rows = []
    for n_groups, k in itertools.product(N_GROUPS_GRID, K_GRID):
        if k >= n_groups:
            continue
        row = {
            'n_groups': n_groups, 'k': k, 'n_obs': n_obs,
            'winning_C': winning_C,
            'notes': ' '.join(filter(None, [
                f'dropped={drop_features}' if drop_features else '',
                'detrended' if detrend else '',
            ])),
        }
        phase_a = _phase_a_check(n_obs, y, t1, n_groups, k)
        genuine_cpcv = phase_a['phi'] is not None and phase_a['phi'] > 1
        row.update({
            'phi': phase_a['phi'], 'genuine_cpcv': genuine_cpcv,
            'n_splits': phase_a['n_splits'],
            'min_group_size': phase_a['min_group_size'],
            'min_train_size': phase_a['min_train_size'],
            'phase_a_pass': phase_a['pass'],
            'phase_a_fail_reason': phase_a['reason'],
            'phase_b_attempted': False, 'phase_b_error': '',
            'mean_path_sharpe': np.nan, 'std_path_sharpe': np.nan,
            'min_path_sharpe': np.nan, 'max_path_sharpe': np.nan,
            'n_paths_negative': np.nan,
        })

        if phase_a['pass']:
            tag = '' if genuine_cpcv else '  [NOT real CPCV -- phi=1, single holdout]'
            print(f'  [N={n_groups} k={k}] Phase A pass (phi={phase_a["phi"]}, '
                  f'min_group={phase_a["min_group_size"]}, '
                  f'min_train={phase_a["min_train_size"]}){tag} -- running Phase B...')
            row['phase_b_attempted'] = True
            try:
                b = _phase_b_run(X, y, w, t1, ret, n_groups, k, winning_C)
                row['mean_path_sharpe'] = b['mean']
                row['std_path_sharpe'] = b['std']
                row['min_path_sharpe'] = b['min']
                row['max_path_sharpe'] = b['max']
                row['n_paths_negative'] = b['n_negative']
                print(f'    -> mean path Sharpe={b["mean"]:.4f} '
                      f'({b["n_negative"]}/{phase_a["phi"]} paths negative)')
            except Exception as exc:  # noqa: BLE001 -- record, keep sweeping
                row['phase_b_error'] = repr(exc)
                print(f'    -> Phase B FAILED: {exc!r}')
        else:
            print(f'  [N={n_groups} k={k}] Phase A FAIL: {phase_a["reason"]}')

        rows.append(row)

    df = pd.DataFrame(rows, columns=SWEEP_COLUMNS)
    df.to_csv(output_csv, index=False)
    print(f'\nWrote {len(df)} rows to {output_csv}')
    print(df[['n_groups', 'k', 'phi', 'genuine_cpcv', 'phase_a_pass',
              'phase_a_fail_reason', 'mean_path_sharpe',
              'n_paths_negative']].to_string(index=False))

    n_genuine = int(df['genuine_cpcv'].sum())
    print(f'\n{n_genuine}/{len(df)} rows are genuine multi-path CPCV '
          f'(genuine_cpcv=True, phi>1). k=1 rows (phi=1) are a single '
          f'holdout, not CPCV -- do not average them in with the rest '
          f'when summarizing this sweep.')


if __name__ == '__main__':
    main()
