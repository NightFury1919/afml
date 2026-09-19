"""
pipeline/diagnostics/diagnose_null_bias_multiwindow_v2.py

Improves on diagnose_null_bias_multiwindow.py (2026-09-14/15) in two
ways, per the n=1 diagnostic's own findings:

1. REPLACES the crude mean(pos) vs mean(bar_ret) sign-comparison with
   the actual bar-by-bar Pearson correlation between the (lagged)
   position and the real return -- that's what actually determines
   PnL, not whether two averages happen to point the same way. The v1
   diagnostic's own fold 5 was a clear example of the proxy being
   misleading: labeled "OPPOSITE" by the mean-comparison, but its real
   PnL was flat/slightly positive.

2. REPEATS across N_SEEDS=3 independent shuffles (Ethan's choice,
   2026-09-15, given ~13.5 min/seed) instead of n=1, because the first
   diagnostic's own conclusion was "mixed/inconclusive at n=1" -- 4/8
   folds each way on the crude proxy. A single shuffle can't
   distinguish "no real pattern" from "the proxy was too crude to see
   the real pattern."

STILL UNEXPLAINED, motivating this script: v1 found pred_frac_class1
swinging wildly per fold (6%, 87%, 60%, 67%, 26%, 69%, 18%, 15%)
despite every window's real 'bin' labels being exactly 50/50 balanced
(shuffling is a permutation -- it reorders, never changes the count).
An unbiased classifier facing a genuinely uninformative, balanced
target should predict close to 50/50 on a fresh window, not swing
this hard. This script checks whether that swing is just per-shuffle
noise (averages out near 0.5 across seeds) or a real, structural
oddity -- and, more importantly, whether the SIGN of those confident
calls actually anti-correlates with real returns often enough to
explain the calibration's 30/30-negative skew.

THE DECISIVE STAT: corr(pos.shift(1), bar_ret) per fold, computed
across the FULL bar series (including zero-position bars, exactly as
realized_pnl() does it) -- not mean-vs-mean. Pooled across all
N_SEEDS x 8 = 24 fold-seed combinations, a one-sample t-test on this
correlation tells us directly whether the null procedure is
mechanically anti-correlated with real BTC returns on average (which
would fully explain the calibration's negative skew) or whether it's
genuinely centered near zero (in which case the skew's source is still
open).

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\diagnose_null_bias_multiwindow_v2.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_SCRIPT_PATH = os.path.join(HERE, 'pool_multiwindow_leave_one_out.py')
RESULTS_CSV_PATH = os.path.join(HERE, 'diagnose_null_bias_multiwindow_v2_results.csv')

spec = importlib.util.spec_from_file_location('pool_multiwindow_leave_one_out', REAL_SCRIPT_PATH)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

# One of these (77) matches diagnose_null_bias_multiwindow.py's own
# single-shuffle run, so that earlier result is directly comparable to
# (not just alongside) this script's fold_1 output for seed 77.
SEEDS = [77, 555, 999]


def shuffle_bin_within_windows(window_tables, rng):
    shuffled = {}
    for w, (events, close) in window_tables.items():
        events2 = events.copy()
        events2['bin'] = rng.permutation(events2['bin'].values)
        shuffled[w] = (events2, close)
    return shuffled


def run_one_test_fold_with_diagnostics(test_window_id, window_tables, feature_cols):
    train_ids = [w for w in real.WINDOW_DIRS if w != test_window_id]

    scores = {(C, step): [] for C in real.C_GRID for step in real.STEP_GRID}
    for C in real.C_GRID:
        for val_id in train_ids:
            inner_train_ids = [w for w in train_ids if w != val_id]
            pool = pd.concat([window_tables[w][0] for w in inner_train_ids], axis=0)
            clf = real.fit_classifier(pool, feature_cols, C)
            events_val, close_val = window_tables[val_id]
            prob, pred = real.predict_oos(clf, events_val, feature_cols)
            for step in real.STEP_GRID:
                pnl = real.realized_pnl(events_val, close_val, step, prob, pred)
                scores[(C, step)].append(real.sharpe_ratio(pd.Series(pnl)))

    avg_scores = {k: float(np.mean(v)) for k, v in scores.items()}
    winner_C, winner_step = max(avg_scores, key=avg_scores.get)

    full_train_pool = pd.concat([window_tables[w][0] for w in train_ids], axis=0)
    winner_clf = real.fit_classifier(full_train_pool, feature_cols, winner_C)

    events_test, close_test = window_tables[test_window_id]
    prob_test, pred_test = real.predict_oos(winner_clf, events_test, feature_cols)

    ev = events_test.loc[prob_test.index]
    sig = real.getSignal(ev, winner_step, prob_test, pred_test, numClasses=2, numThreads=1)
    pos = sig.reindex(close_test.index, method='ffill').fillna(0.0)
    bar_ret = close_test.pct_change().dropna()
    pos_lagged = pos.shift(1).reindex(bar_ret.index).fillna(0.0)
    pnl = pos_lagged * bar_ret

    # THE decisive stat -- actual bar-by-bar correlation, not mean-vs-mean.
    if pos_lagged.std() > 0 and bar_ret.std() > 0:
        corr_pos_ret = float(pos_lagged.corr(bar_ret))
    else:
        corr_pos_ret = np.nan

    return {
        'test_window': test_window_id,
        'winner_C': winner_C,
        'winner_step': winner_step,
        'pred_frac_class1': float((pred_test == 1).mean()),
        'corr_pos_ret': corr_pos_ret,
        'fold_pnl_mean': float(pnl.mean()),
        'n_bars': len(pnl),
        # kept for continuity with v1's output, not the decisive stat here
        'mean_pos': float(pos.mean()),
        'mean_bar_ret': float(bar_ret.mean()),
    }


def main():
    print('Loading all 8 real window tables (reusing cached staging)...')
    window_tables = {}
    feature_cols = None
    for w in real.WINDOW_DIRS:
        events, close = real.build_window_table(w)
        window_tables[w] = (events, close)
        feature_cols = feature_cols or [c for c in events.columns if c not in ('bin', 'w', 't1')]

    all_rows = []
    for seed in SEEDS:
        print(f'\n{"=" * 78}')
        print(f'SEED {seed}')
        print(f'{"=" * 78}')
        rng = np.random.default_rng(seed)
        shuffled = shuffle_bin_within_windows(window_tables, rng)
        for test_window_id in real.WINDOW_DIRS:
            print(f'[seed {seed}, fold {test_window_id}/8] running...', end=' ', flush=True)
            row = run_one_test_fold_with_diagnostics(test_window_id, shuffled, feature_cols)
            row['seed'] = seed
            all_rows.append(row)
            print(f"pred_frac_class1={row['pred_frac_class1']:.3f}, "
                  f"corr_pos_ret={row['corr_pos_ret']:+.4f}, "
                  f"fold_pnl_mean={row['fold_pnl_mean']:+.6e}")
            pd.DataFrame(all_rows).to_csv(RESULTS_CSV_PATH, index=False)

    df = pd.DataFrame(all_rows)
    print(f'\nAll {len(df)} seed-fold rows saved -> {RESULTS_CSV_PATH}')

    print(f'\n{"=" * 78}')
    print(f'SUMMARY ACROSS {len(SEEDS)} SEEDS x 8 FOLDS = {len(df)} INDEPENDENT DRAWS')
    print(f'{"=" * 78}')
    print(df[['seed', 'test_window', 'pred_frac_class1', 'corr_pos_ret',
              'fold_pnl_mean']].to_string(index=False))

    pfc = df['pred_frac_class1']
    print(f'\npred_frac_class1: mean={pfc.mean():.4f}, std={pfc.std(ddof=1):.4f} '
          f'(0.5 = unbiased; real labels were exactly 50/50 balanced in every window)')

    corr = df['corr_pos_ret'].dropna()
    print(f'\ncorr_pos_ret (THE decisive stat): mean={corr.mean():+.4f}, '
          f'std={corr.std(ddof=1):.4f}, n={len(corr)}')
    n_negative = int((corr < 0).sum())
    print(f'Folds with negative correlation: {n_negative}/{len(corr)}')

    if len(corr) >= 3 and corr.std(ddof=1) > 0:
        t_stat = float(corr.mean() / (corr.std(ddof=1) / np.sqrt(len(corr))))
        p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(corr) - 1)))
        print(f'One-sample t-test (corr_pos_ret vs 0): t={t_stat:+.4f}, p={p_value:.4f}')
    else:
        t_stat, p_value = np.nan, np.nan

    print()
    if not np.isnan(p_value) and p_value < 0.05 and t_stat < 0:
        print('CONFIRMS a real, systematic mechanism: under a genuine null '
              '(scrambled labels), the model\'s position is significantly '
              'ANTI-correlated with real BTC returns on average. This fully '
              'explains the calibration\'s 30/30-negative skew -- it\'s not '
              'random noise averaging to zero, it\'s a structural artifact '
              'of fitting real, trending, non-i.i.d. features against an '
              'uninformative target across genuinely different real market '
              'regimes (the training windows\' feature distribution differs '
              'from the held-out window\'s, across ~7 real months). This is '
              'a property of the PROCEDURE, not evidence about whether BTC '
              'has a real edge -- but it means the multi-window pooled '
              'test\'s real p=0.0453 result cannot be trusted at face value '
              'until this is fixed (e.g. per-window feature standardization/'
              'detrending before pooling, or a selection criterion less '
              'sensitive to this effect).'
              )
    elif not np.isnan(p_value) and p_value < 0.05 and t_stat > 0:
        print('Correlation is significantly POSITIVE on average -- does NOT '
              "explain the calibration's negative skew. The skew's source "
              'is still unresolved; consider whether getSignal\'s averaging/'
              'discretization mechanics themselves introduce a bias '
              'independent of the correlation sign (e.g. an asymmetry in '
              'how position sizes get rounded).')
    else:
        print(f'Not significant (p={p_value:.4f} if computed) -- the '
              "correlation looks centered near zero across these "
              f'{len(corr)} draws. If so, the calibration\'s 30/30-negative '
              'skew is NOT simply explained by a mechanical anti-'
              'correlation, and the source remains genuinely open -- worth '
              'checking getSignal\'s discretization step directly, or '
              'whether volatility-drag-style variance effects (not mean '
              'correlation) are the real driver.')


if __name__ == '__main__':
    main()
