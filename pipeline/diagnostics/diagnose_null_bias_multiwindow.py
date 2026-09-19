"""
pipeline/diagnostics/diagnose_null_bias_multiwindow.py

Investigates WHY pool_multiwindow_null_calibration.py's 30 real
shuffles came back 30/30 negative (7/30 significantly so,
2026-09-14/15) -- a genuine null (scrambled labels) should average out
near zero, not skew systematically negative. That's a property of the
PROCEDURE, separate from whatever the real (unshuffled) BTC result
means, and needs to be understood before trusting either.

HYPOTHESIS: scrambling 'bin' preserves each window's real class COUNT
(a permutation reorders, it doesn't change how many 0s/1s exist), so
the classifier isn't trained on an imbalanced target. But the REAL
FEATURES are untouched, real, and genuinely autocorrelated/trending
(they're built from real BTC price action across ~7 months). A
classifier fit on real, trending, non-i.i.d. features against a
SCRAMBLED target can still end up systematically favoring one
predicted class on a given held-out window -- not because it learned
anything real, but because the training windows' feature distribution
differs from the held-out window's (real regime drift across months),
so the fitted decision boundary happens to land mostly on one side of
the held-out window's real feature values. If that predicted
direction happens to be mis-aligned with the held-out window's ACTUAL
net price drift, you get a systematic loss -- with no real signal
anywhere, just feature-distribution drift colliding with real price
history.

This script runs exactly ONE shuffled repeat (cheap, ~13.5 min -- NOT
another 6.75-hour calibration) and logs, per fold:
  - pred_frac_class1   : fraction of the held-out window's bars where
                         the (null-trained) classifier predicted class
                         1 -- systematically far from 0.5 would support
                         the hypothesis.
  - mean_bar_ret        : the held-out window's own REAL mean bar
                         return (its real net drift direction).
  - mean_pos             : the discretized position's own mean
                         (positive = net long bias, negative = net
                         short bias) -- compare its SIGN against
                         mean_bar_ret's sign.
  - fold_pnl_mean        : this fold's own realized PnL mean.
If mean_pos and mean_bar_ret consistently have OPPOSITE signs across
most folds, that directly confirms the hypothesis: the null-trained
model is systematically betting against each window's real drift, not
just guessing randomly.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\diagnose_null_bias_multiwindow.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_SCRIPT_PATH = os.path.join(HERE, 'pool_multiwindow_leave_one_out.py')

spec = importlib.util.spec_from_file_location('pool_multiwindow_leave_one_out', REAL_SCRIPT_PATH)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)

RNG_SEED = 77  # arbitrary, single diagnostic shuffle -- not the calibration's own seed sequence


def shuffle_bin_within_windows(window_tables, rng):
    shuffled = {}
    for w, (events, close) in window_tables.items():
        events2 = events.copy()
        events2['bin'] = rng.permutation(events2['bin'].values)
        shuffled[w] = (events2, close)
    return shuffled


def run_one_test_fold_with_diagnostics(test_window_id, window_tables, feature_cols):
    """Same selection/refit/evaluate logic as real.run_one_test_fold(),
    but keeps prob/pred/pos around afterward instead of discarding them,
    so we can inspect the actual betting behavior."""
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

    # Rebuild the position series directly (mirrors real.realized_pnl's
    # internals) so we can inspect it, not just the final PnL.
    from scipy.stats import norm
    ev = events_test.loc[prob_test.index]
    sig = real.getSignal(ev, winner_step, prob_test, pred_test, numClasses=2, numThreads=1)
    pos = sig.reindex(close_test.index, method='ffill').fillna(0.0)
    bar_ret = close_test.pct_change().dropna()
    pnl = pos.shift(1).reindex(bar_ret.index).fillna(0.0) * bar_ret

    return {
        'test_window': test_window_id,
        'winner_C': winner_C,
        'winner_step': winner_step,
        'pred_frac_class1': float((pred_test == 1).mean()),
        'mean_prob': float(prob_test.mean()),
        'mean_pos': float(pos.mean()),
        'mean_bar_ret': float(bar_ret.mean()),
        'sign_agree': (
            'SAME (betting WITH real drift)'
            if np.sign(pos.mean()) == np.sign(bar_ret.mean()) and pos.mean() != 0
            else 'OPPOSITE (betting AGAINST real drift)'
            if pos.mean() != 0
            else 'flat (no net position)'
        ),
        'fold_pnl_mean': float(pnl.mean()),
        'n_bars': len(pnl),
    }


def main():
    print('Loading all 8 real window tables (reusing cached staging)...')
    window_tables = {}
    feature_cols = None
    for w in real.WINDOW_DIRS:
        events, close = real.build_window_table(w)
        window_tables[w] = (events, close)
        feature_cols = feature_cols or [c for c in events.columns if c not in ('bin', 'w', 't1')]

    rng = np.random.default_rng(RNG_SEED)
    shuffled = shuffle_bin_within_windows(window_tables, rng)

    print('\nRunning ONE shuffled 8-fold sweep with full diagnostics '
          '(this is the same cost as one real run, ~13.5 min)...\n')
    rows = []
    for test_window_id in real.WINDOW_DIRS:
        print(f'[fold {test_window_id}/8] running...', end=' ', flush=True)
        row = run_one_test_fold_with_diagnostics(test_window_id, shuffled, feature_cols)
        rows.append(row)
        print(f"pred_frac_class1={row['pred_frac_class1']:.3f}, "
              f"mean_pos={row['mean_pos']:+.4f}, "
              f"mean_bar_ret={row['mean_bar_ret']:+.6f}, "
              f"{row['sign_agree']}")

    df = pd.DataFrame(rows)
    out_path = os.path.join(HERE, 'diagnose_null_bias_multiwindow_results.csv')
    df.to_csv(out_path, index=False)

    print(f'\n{"=" * 78}')
    print('DIAGNOSTIC SUMMARY')
    print(f'{"=" * 78}')
    print(df[['test_window', 'pred_frac_class1', 'mean_pos', 'mean_bar_ret',
              'sign_agree', 'fold_pnl_mean']].to_string(index=False))

    n_opposite = int((df['sign_agree'].str.startswith('OPPOSITE')).sum())
    n_same = int((df['sign_agree'].str.startswith('SAME')).sum())
    n_flat = int((df['sign_agree'].str.startswith('flat')).sum())
    print(f'\nFolds betting AGAINST their own window\'s real drift: {n_opposite}/8')
    print(f'Folds betting WITH their own window\'s real drift:     {n_same}/8')
    print(f'Folds with no net position:                            {n_flat}/8')

    if n_opposite >= 6:
        print('\nCONFIRMS the hypothesis: under a genuine null (scrambled '
              'labels), the classifier systematically bets AGAINST each '
              "held-out window's real net drift, most of the time -- not "
              'random noise around zero. This is a real feature-'
              'distribution-drift artifact (training windows\' features '
              'differ from the held-out window\'s, across ~7 real months), '
              'not evidence about whether BTC has an edge. It explains the '
              "calibration's systematic negative skew and means the whole "
              'procedure needs a fix (e.g. detrending features per window, '
              'or a different selection criterion) before ANY result from '
              'it -- positive or negative -- can be trusted at face value.')
    elif n_same >= 6:
        print('\nDoes NOT confirm the hypothesis -- folds mostly bet WITH '
              "the real drift, which would mean something ELSE explains "
              "the calibration's negative skew. Worth a different "
              'diagnostic angle.')
    else:
        print('\nMixed / inconclusive at n=1 shuffle -- no dominant pattern. '
              'Consider running 2-3 more single-shuffle diagnostics '
              '(different seeds) before concluding either way.')


if __name__ == '__main__':
    main()
