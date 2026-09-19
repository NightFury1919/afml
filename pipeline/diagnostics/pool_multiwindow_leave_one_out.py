"""
pipeline/diagnostics/pool_multiwindow_leave_one_out.py

Tests whether the REAL production pipeline (Ch09-style SVC grid, Ch10
getSignal, Ch11 trial construction) shows an edge when trained on POOLED
data from multiple independent real Kraken BTC windows and evaluated on
a genuinely held-out window -- as opposed to the single-window
methodology (one dataset in, one DSR/PBO number out) this project has
always used.

Motivated directly by pooled_ols_roll_c_test.py's synthetic finding
(2026-09-07/08, committed 41c95f5): a real-but-small relationship can be
invisible in any single window yet become statistically detectable when
many independent windows are pooled (pooled mean-PnL t-test significant
at edge_strength>=3.0, p<0.005). This script asks the same question of
the REAL production classifier on the REAL 8 independent Kraken BTC/USD
windows captured 2026-09-10/11 (see briefing_kraken_8window_resolution.md),
instead of synthetic injected data.

DESIGN NOTE -- why this can't just call ch11.part_c_build_trials()
directly on pooled data (2026-09-14):
part_c_build_trials()/out_of_sample_probs() rely on Ch07's PurgedKFold,
which purges/embargoes around fold boundaries IN TIME -- it assumes one
continuous series. The 8 windows are NOT continuous: they're 8 separate
30-day pulls scattered across ~7 months (window8 oldest, window1
newest, per the chain-capture logic in capture_kraken_windows_chain.py).
Naively concatenating windows and computing close.pct_change() across
the stitched series would produce a garbage "return" at every window
boundary, and PurgedKFold's time-based purging is meaningless across
month-long real gaps. Fix: use the windows THEMSELVES as natural folds
(they are already genuinely independent -- no synthetic CV machinery
needed to approximate that property), and compute bar returns
separately within each window, never across a concatenated close
series.

METHOD -- nested leave-one-window-out (rigorous option, chosen over the
cheaper single-validation-split option on 2026-09-14):
For each of the 8 windows, held out in turn as the TEST window:
  1. SELECTION (uses only the other 7 TRAINING windows, never touches
     the test window): for each of the real C_GRID's 4 values, rotate
     through all 7 training windows as an inner VALIDATION window (7
     rotations) -- fit Pipeline(StandardScaler, SVC(C, gamma=GAMMA,
     probability=True, random_state=0)) once on the pooled OTHER 6
     training windows' events, predict_proba on the held-back
     validation window, and score all 5 STEP_GRID values against it
     (getSignal doesn't require refitting the classifier -- it only
     resizes/discretizes bets downstream of prob/pred, so all 5 step
     values reuse the same single fit). This gives, per C, 7 rotations
     x 5 steps = 35 Sharpe readings; average each (C, step) pair's
     Sharpe across its 7 rotations, and pick the single (C, step) with
     the best average -- the same "pick the winner from real
     cross-validated performance" logic Ch11 always uses, just with
     real independent windows standing in for PurgedKFold's synthetic
     folds.
  2. REFIT: fit the winning C on ALL 7 training windows pooled (no
     held-back rotation now -- we already have a genuine external test
     set).
  3. EVALUATE: predict_proba on the TEST window's own events (this
     window has never been touched by fitting OR by config selection),
     run the real getSignal() with the winning step, mark-to-market
     against the TEST window's own close series (never a pooled/
     concatenated one), producing one realized PnL series for this
     fold.
Repeat for all 8 folds (window1..window8 each held out once). Total
real SVC fits: 8 folds x 4 C-values x 7 rotations (selection) + 8 folds
x 1 (winner refit) = 232 fits -- NOT 8 x 20 x 7, because stepSize
doesn't require a refit; it's scored via getSignal on the already-
fitted prob/pred, cheap to do all 5 per fit.

DECISIVE METRIC (same convention as pooled_ols_roll_c_test.py,
2026-09-07/08): pool all 8 folds' held-out realized PnL into ONE
series, then a one-sample t-test of the pooled per-bet PnL mean against
0. Not hit_rate/binomial test -- those can stay near chance even with a
real, small, persistent edge (see that script's own synthetic
validation note). hit_rate/Sharpe/binomial are still reported for
completeness.

VALIDATED (2026-09-14, sandbox, synthetic stand-in data): the
selection/refit/evaluate control flow was checked against a planted,
genuinely forward-looking synthetic signal (identical across 8 fake
windows) before being pointed at real data -- confirmed (a) the
held-out test window is never touched during selection, (b) the
correct, consistent winning config is recovered across folds, and (c)
the decisive pooled t-test detects the planted signal cleanly
(t=+12.6, p<0.0001). This validates the NEW control flow only, not the
real financial correctness of rebuild.py/features.py's labels/features,
which are reused unmodified and already real-machine confirmed.

No new AFML formula: reuses ch10's real getSignal, ch11's real
sharpe_ratio, sklearn's Pipeline/StandardScaler/SVC exactly as Ch11
configures them (same C_GRID/STEP_GRID/GAMMA). rebuild.py's
build_bars_and_labels() and features.py's build_enriched_events() build
each window's table exactly as run_pipeline_live.py does
(target_bars=1000, the real live-pipeline default, run_pipeline_live.py
line 105).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML

    # Step 1 (do this first): time ONE real fit+predict+score at real
    # scale to estimate total runtime before committing to the full
    # 232-fit sweep.
    python pipeline\\diagnostics\\pool_multiwindow_leave_one_out.py --time-one-fit

    # Step 2: full sweep once the timing looks reasonable
    python pipeline\\diagnostics\\pool_multiwindow_leave_one_out.py
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import t as t_dist
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
CH11_DIR = os.path.join(ROOT, 'ch11', 'backtest_dangers')
CH10_DIR = os.path.join(ROOT, 'ch10', 'bet_sizing')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, CH11_DIR)
sys.path.insert(0, CH10_DIR)

from rebuild import build_bars_and_labels           # noqa: E402 -- real, ch02/03/04 chain
from features import build_enriched_events           # noqa: E402 -- real, Ch19 + fracdiff
from live_staging import stage_live_training_tables   # noqa: E402 -- real, exact schema Ch11 expects
from pbo import sharpe_ratio                          # noqa: E402 -- real, ch11/backtest_dangers/pbo.py
from bet_sizing import getSignal                      # noqa: E402 -- real, ch10

DIAGNOSTICS_DIR = HERE
STAGING_ROOT = os.path.join(DIAGNOSTICS_DIR, 'multiwindow_staging')
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'pool_multiwindow_leave_one_out_results.csv')
FOLD_SUMMARY_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'pool_multiwindow_fold_summary.csv')

# Real 8 windows, per Ethan's 2026-09-14 `Get-ChildItem` confirmation.
# Excludes the superseded pre-chain window2 (2026-08-25) and the pooled/
# other-asset snapshots that share the "kraken_snapshot" prefix.
WINDOW_DIRS = {
    1: 'kraken_snapshot_720h_2026-08-25',
    2: 'kraken_snapshot_720h_window2_2026-09-08',
    3: 'kraken_snapshot_720h_window3_2026-09-10',
    4: 'kraken_snapshot_720h_window4_2026-09-11',
    5: 'kraken_snapshot_720h_window5_2026-09-10',
    6: 'kraken_snapshot_720h_window6_2026-09-10',
    7: 'kraken_snapshot_720h_window7_2026-09-11',
    8: 'kraken_snapshot_720h_window8_2026-09-11',
}

TARGET_BARS = 1000  # real live-pipeline default, run_pipeline_live.py line 105
GAMMA = 0.1  # same as ch11 (N_SPLITS/PCT_EMBARGO unused here -- no PurgedKFold)
C_GRID = [0.01, 0.1, 1.0, 10.0]
STEP_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]


def build_window_table(window_id):
    """Build (or load cached) window's real enriched event table + close
    series, via the exact real rebuild.py -> features.py -> live_staging.py
    chain run_pipeline_live.py uses. Cached under STAGING_ROOT/windowN/ so
    re-running this script doesn't rebuild from raw trades every time."""
    out_dir = os.path.join(STAGING_ROOT, f'window{window_id}')
    enriched_path = os.path.join(out_dir, 'ch07_training_table_enriched.csv')
    features_path = os.path.join(out_dir, 'ch05_features.csv')

    if os.path.exists(enriched_path) and os.path.exists(features_path):
        events = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
        events['t1'] = pd.to_datetime(events['t1'])
        close = pd.read_csv(features_path, index_col=0, parse_dates=True)['close']
        return events, close

    snapshot_dir = os.path.join(DIAGNOSTICS_DIR, WINDOW_DIRS[window_id])
    raw_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f'{raw_path} not found -- window {window_id}\'s snapshot dir '
            f'may have a different name than expected. Check '
            f'WINDOW_DIRS[{window_id}] against your real directory listing.'
        )
    raw_trades = pd.read_parquet(raw_path)

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    staged = stage_live_training_tables(rebuild_result, enriched_result, out_dir)
    print(f'  window {window_id}: staged {staged["n_events"]} events, '
          f'{len(staged["feature_cols"])} features -> {out_dir}')

    events = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    close = pd.read_csv(features_path, index_col=0, parse_dates=True)['close']
    return events, close


def fit_classifier(events_pool, feature_cols, C):
    """Fit ONE Pipeline(StandardScaler, SVC) on a pooled training table --
    same real Pipeline construction as ch11.out_of_sample_probs(), but a
    single fit-on-everything (no PurgedKFold split) since the caller
    already supplies a genuinely independent training pool with no
    overlap against whatever it will be evaluated on."""
    X = events_pool[feature_cols]
    y = events_pool['bin']
    w = events_pool['w'].values
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=C, gamma=GAMMA, probability=True, random_state=0)),
    ])
    clf.fit(X, y, svc__sample_weight=w)
    return clf


def predict_oos(clf, events_eval, feature_cols):
    """predict_proba on an entirely separate table the classifier was
    never fit on -- same class-argmax convention as
    ch11.out_of_sample_probs()'s per-fold prediction step."""
    X = events_eval[feature_cols]
    p = clf.predict_proba(X)
    prob = pd.Series(p.max(axis=1), index=X.index)
    pred = pd.Series(
        clf.named_steps['svc'].classes_[p.argmax(axis=1)], index=X.index,
    )
    return prob, pred


def realized_pnl(events_eval, close_eval, step, prob, pred):
    """Real Ch10 getSignal() -> position -> Ch11-style bar-level
    mark-to-market PnL, evaluated within ONE window's own close series
    only (never a pooled/concatenated one -- see module DESIGN NOTE)."""
    ev = events_eval.loc[prob.index]
    sig = getSignal(ev, step, prob, pred, numClasses=2, numThreads=1)
    pos = sig.reindex(close_eval.index, method='ffill').fillna(0.0)
    bar_ret = close_eval.pct_change().dropna()
    # .shift(1) LOAD-BEARING, same as ch11 -- yesterday's position earns
    # today's bar return, not the return that caused it (lookahead).
    pnl = pos.shift(1).reindex(bar_ret.index).fillna(0.0) * bar_ret
    return pnl.values


def run_one_test_fold(test_window_id, window_tables, feature_cols, timing_only=False):
    t_start = time.time()
    train_ids = [w for w in WINDOW_DIRS if w != test_window_id]

    # --- STEP 1: SELECTION (training windows only) ---
    scores = {(C, step): [] for C in C_GRID for step in STEP_GRID}
    for C in C_GRID:
        for val_id in train_ids:
            inner_train_ids = [w for w in train_ids if w != val_id]
            pool = pd.concat(
                [window_tables[w][0] for w in inner_train_ids], axis=0,
            )
            clf = fit_classifier(pool, feature_cols, C)
            events_val, close_val = window_tables[val_id]
            prob, pred = predict_oos(clf, events_val, feature_cols)
            for step in STEP_GRID:
                pnl = realized_pnl(events_val, close_val, step, prob, pred)
                scores[(C, step)].append(sharpe_ratio(pd.Series(pnl)))
            if timing_only:
                return {'single_fit_predict_score_sec': time.time() - t_start}

    avg_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    winner = max(avg_scores, key=avg_scores.get)
    winner_C, winner_step = winner

    # --- STEP 2: REFIT on all 7 training windows pooled ---
    full_train_pool = pd.concat(
        [window_tables[w][0] for w in train_ids], axis=0,
    )
    winner_clf = fit_classifier(full_train_pool, feature_cols, winner_C)

    # --- STEP 3: EVALUATE on the genuinely held-out test window ---
    events_test, close_test = window_tables[test_window_id]
    prob_test, pred_test = predict_oos(winner_clf, events_test, feature_cols)
    test_pnl = realized_pnl(events_test, close_test, winner_step, prob_test, pred_test)
    test_sharpe = sharpe_ratio(pd.Series(test_pnl))

    return {
        'test_window': test_window_id,
        'winner_C': winner_C,
        'winner_step': winner_step,
        'winner_avg_selection_sharpe': avg_scores[winner],
        'test_sharpe': test_sharpe,
        'n_test_bars': len(test_pnl),
        'n_test_active_bars': int((test_pnl != 0).sum()),
        'pnl': test_pnl,
        'wall_clock_sec': time.time() - t_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--time-one-fit', action='store_true',
                         help='Run exactly ONE fit+predict+score at real '
                              'scale (fold=window1, C=0.01, first inner '
                              'rotation) and report wall-clock time, '
                              'WITHOUT running the full 232-fit sweep. Run '
                              'this first.')
    args = parser.parse_args()

    print('Building/loading all 8 real window tables '
          f'(target_bars={TARGET_BARS}, live-pipeline default)...')
    window_tables = {}
    feature_cols = None
    for w in WINDOW_DIRS:
        events, close = build_window_table(w)
        window_tables[w] = (events, close)
        this_feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
        if feature_cols is None:
            feature_cols = this_feature_cols
        elif feature_cols != this_feature_cols:
            raise ValueError(
                f'window {w} has different feature columns than earlier '
                f'windows ({this_feature_cols} vs {feature_cols}) -- '
                'cannot pool windows with mismatched feature sets.'
            )
        print(f'  window {w}: {len(events)} events, '
              f'{len(this_feature_cols)} features, '
              f'{len(close)} bars')

    if args.time_one_fit:
        print('\nTIMING CHECK: one real fit+predict+score at full scale '
              '(fold=window1, C=0.01, first inner rotation)...')
        result = run_one_test_fold(1, window_tables, feature_cols, timing_only=True)
        sec = result['single_fit_predict_score_sec']
        print(f'\nOne (fit + predict + 5-step score) took {sec:.1f}s.')
        print(f'Full sweep = 8 folds x 4 C-values x 7 rotations = 224 of '
              f'these, plus 8 winner refits (no inner scoring, so ~1/7 '
              f'the cost each) -- rough estimate: '
              f'{(224 * sec + 8 * sec / 7) / 60:.1f} minutes total.')
        print('Re-run WITHOUT --time-one-fit once this looks reasonable.')
        return

    print('\nFull sweep: 8 folds x 4 C-values x 7 rotations = 224 '
          'selection fits, + 8 winner refits + 8 test evaluations.\n')

    fold_results = []
    all_pnl = []
    for test_window_id in WINDOW_DIRS:
        print(f'[fold {test_window_id}/8] test_window={test_window_id} ... ',
              end='', flush=True)
        try:
            result = run_one_test_fold(test_window_id, window_tables, feature_cols)
            print(f"done in {result['wall_clock_sec']:.1f}s "
                  f"(winner=C{result['winner_C']}_s{result['winner_step']}, "
                  f"selection_sharpe={result['winner_avg_selection_sharpe']:+.4f}, "
                  f"test_sharpe={result['test_sharpe']:+.4f}, "
                  f"{result['n_test_active_bars']}/{result['n_test_bars']} active)")
            all_pnl.append(result['pnl'])
            fold_results.append({k: v for k, v in result.items() if k != 'pnl'})
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            fold_results.append({
                'test_window': test_window_id, 'error': f'{type(e).__name__}: {e}',
            })

    pd.DataFrame(fold_results).to_csv(FOLD_SUMMARY_CSV_PATH, index=False)
    print(f'\nPer-fold summary written to {FOLD_SUMMARY_CSV_PATH}')

    if not all_pnl:
        print('No successful folds -- nothing to pool.')
        return

    pooled_pnl = np.concatenate(all_pnl)
    n_active = int((pooled_pnl != 0).sum())
    n_wins = int((pooled_pnl > 0).sum())
    pooled_hit_rate = n_wins / n_active if n_active >= 3 else np.nan
    pooled_sharpe = sharpe_ratio(pd.Series(pooled_pnl))
    binom_p = binomtest(n_wins, n_active, p=0.5).pvalue if n_active >= 3 else np.nan

    active_pnl = pooled_pnl[pooled_pnl != 0]
    if len(active_pnl) >= 3 and active_pnl.std(ddof=1) > 0:
        t_stat = float(active_pnl.mean() / (active_pnl.std(ddof=1) / np.sqrt(len(active_pnl))))
        t_p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(active_pnl) - 1)))
    else:
        t_stat, t_p_value = np.nan, np.nan

    summary = pd.DataFrame([{
        'n_folds': len(all_pnl),
        'n_pooled_bars': len(pooled_pnl),
        'n_pooled_active_bars': n_active,
        'pooled_hit_rate': pooled_hit_rate,
        'pooled_sharpe': pooled_sharpe,
        'binom_p_value': binom_p,
        'mean_pnl_t_stat': t_stat,
        'mean_pnl_t_p_value': t_p_value,
    }])
    summary.to_csv(RESULTS_CSV_PATH, index=False)

    print(f'\n{"=" * 74}')
    print('POOLED RESULT (8 real, independent, genuinely held-out windows)')
    print(f'{"=" * 74}')
    print(f'pooled_hit_rate       = {pooled_hit_rate:.4f} (0.5 = chance)')
    print(f'pooled_sharpe         = {pooled_sharpe:+.4f}')
    print(f'binom_p_value         = {binom_p:.4f}')
    print(f'mean_pnl_t_stat       = {t_stat:+.4f}')
    print(f'mean_pnl_t_p_value    = {t_p_value:.4f}  <-- THE DECISIVE NUMBER')
    print(f'\nSaved -> {RESULTS_CSV_PATH}')
    print('\nINTERPRETATION: mean_pnl_t_p_value < 0.05 with a POSITIVE '
          'mean_pnl_t_stat would mean the real production pipeline, '
          'trained and evaluated across genuinely independent real BTC '
          'windows, shows a statistically significant positive realized '
          'edge -- something no single-window run of this pipeline has '
          'ever shown. A non-significant or negative result means the '
          'real feature set does not show this effect even under the '
          'most favorable (pooled, multi-window) test this project has '
          'run -- consistent with, and would further reinforce, the '
          'existing 8-window PBO null finding.')


if __name__ == '__main__':
    main()
