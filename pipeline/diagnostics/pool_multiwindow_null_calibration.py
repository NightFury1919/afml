"""
pipeline/diagnostics/pool_multiwindow_null_calibration.py

Calibrates pool_multiwindow_leave_one_out.py's decisive metric
(pooled mean-PnL t-test across 8 real leave-one-window-out folds)
against its OWN null distribution.

WHY THIS IS NEEDED (2026-09-14): the real run on 2026-09-14 gave
mean_pnl_t_p_value=0.0453 -- nominally significant, but barely, on an
untested procedure. Unlike DSR/PBO (calibrated exhaustively across this
whole project), this specific nested leave-one-window-out + pooled-
t-test procedure has never been checked against a KNOWN null. It's
possible the winner-selection step (picking the best of 20 configs via
only 7 inner rotations, a small sample itself) inflates the false-
positive rate above the nominal 5% -- in which case p=0.045 would mean
much less than it appears to.

METHOD: for each of N_SHUFFLES repeats, independently permute the
'bin' label WITHIN each of the 8 real windows (preserving each
window's own class balance, its real features, real close series, real
w, real t1 -- only the feature<->label link is broken). Then run the
EXACT SAME full nested leave-one-window-out procedure (all 232 real
SVC fits, real selection-then-refit-then-evaluate logic) from
pool_multiwindow_leave_one_out.py, completely unmodified, and record
the resulting pooled mean_pnl_t_stat / mean_pnl_t_p_value.

Shuffling only 'bin' -- not close, not features, not w, not t1 -- is
deliberate: realized_pnl() never reads 'bin' at all (PnL always comes
from the REAL close series' real future returns; 'bin' only supervises
what the classifier is TRAINED to predict). So a classifier trained on
shuffled labels should, in expectation, produce predictions
uncorrelated with real future returns -- exactly the null case this
calibration needs. Validated in sandbox first (2026-09-14, synthetic
data): shuffling collapsed a t=+12.6 planted-signal detection down to
t-stats of +0.46/+1.75/-0.71 across 3 shuffles -- confirms the
mechanism actually nulls out a real relationship rather than leaving
it partially intact.

After N_SHUFFLES repeats, the calibration number that matters:
  fraction of shuffles with mean_pnl_t_p_value < 0.05 AND
  mean_pnl_t_stat > 0 (matching the real run's OBSERVED direction)
If this fraction is close to 0.05, the real p=0.0453 result means
roughly what it claims. If it's meaningfully higher (e.g. 0.15-0.30),
the procedure itself produces false positives at an inflated rate and
the real result should not be trusted at face value.

COST (confirmed real, 2026-09-14): one full 8-fold sweep = ~13.5
minutes on Ethan's real machine (sum of the 8 real fold times from the
unshuffled run). N_SHUFFLES=30 -> ~6.75 hours. Results are written to
CSV after EVERY shuffle (not just at the end) so a long run surviving
an interruption doesn't lose completed shuffles -- re-running this
script resumes from wherever the results CSV left off rather than
starting over.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\pool_multiwindow_null_calibration.py

    # Resumable: if interrupted, just re-run the same command -- it
    # picks up from the last completed shuffle in the results CSV.
"""
import importlib.util
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_SCRIPT_PATH = os.path.join(HERE, 'pool_multiwindow_leave_one_out.py')
RESULTS_CSV_PATH = os.path.join(HERE, 'pool_multiwindow_null_calibration_results.csv')

N_SHUFFLES = 30
RNG_SEED = 20260914  # fixed for reproducibility -- date this calibration was designed

# --- Import the real, already-run script as a module so every function
# reused here (build_window_table, fit_classifier, predict_oos,
# realized_pnl, run_one_test_fold, WINDOW_DIRS) is IDENTICAL to the real
# run, not a re-typed copy that could silently drift out of sync. ---
spec = importlib.util.spec_from_file_location('pool_multiwindow_leave_one_out', REAL_SCRIPT_PATH)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)


def shuffle_bin_within_windows(window_tables, rng):
    """Return a NEW dict with 'bin' independently permuted within each
    window (real features/close/w/t1 untouched). Each window keeps its
    own class balance -- only the feature<->label pairing is broken."""
    shuffled = {}
    for w, (events, close) in window_tables.items():
        events2 = events.copy()
        events2['bin'] = rng.permutation(events2['bin'].values)
        shuffled[w] = (events2, close)
    return shuffled


def run_full_sweep_once(window_tables, feature_cols):
    """One complete 8-fold leave-one-window-out sweep (identical logic
    to real.main()'s loop, minus the printing/CSV-writing side effects),
    returning the pooled decisive metric."""
    all_pnl = []
    for test_window_id in real.WINDOW_DIRS:
        result = real.run_one_test_fold(test_window_id, window_tables, feature_cols)
        all_pnl.append(result['pnl'])

    pooled_pnl = np.concatenate(all_pnl)
    active_pnl = pooled_pnl[pooled_pnl != 0]
    if len(active_pnl) >= 3 and active_pnl.std(ddof=1) > 0:
        t_stat = float(active_pnl.mean() / (active_pnl.std(ddof=1) / np.sqrt(len(active_pnl))))
        t_p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(active_pnl) - 1)))
    else:
        t_stat, t_p_value = np.nan, np.nan
    return t_stat, t_p_value, len(active_pnl)


def main():
    print('Loading all 8 real window tables (reusing cached staging from '
          'the real run if present)...')
    window_tables = {}
    feature_cols = None
    for w in real.WINDOW_DIRS:
        events, close = real.build_window_table(w)
        window_tables[w] = (events, close)
        this_feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
        feature_cols = feature_cols or this_feature_cols
    print(f'  loaded {len(window_tables)} windows, {len(feature_cols)} features each.')

    # --- Resume support: pick up from the last completed shuffle ---
    completed = []
    if os.path.exists(RESULTS_CSV_PATH):
        existing = pd.read_csv(RESULTS_CSV_PATH)
        completed = existing.to_dict('records')
        print(f'\nFound {len(completed)} already-completed shuffles in '
              f'{RESULTS_CSV_PATH} -- resuming from shuffle {len(completed) + 1}.')

    rng = np.random.default_rng(RNG_SEED)
    # Advance the RNG state past already-completed shuffles so a resumed
    # run doesn't repeat the same shuffle draws.
    for _ in range(len(completed)):
        shuffle_bin_within_windows(window_tables, rng)

    for i in range(len(completed), N_SHUFFLES):
        t_start = time.time()
        shuffled = shuffle_bin_within_windows(window_tables, rng)
        t_stat, t_p, n_active = run_full_sweep_once(shuffled, feature_cols)
        elapsed = time.time() - t_start
        row = {
            'shuffle_index': i + 1,
            'mean_pnl_t_stat': t_stat,
            'mean_pnl_t_p_value': t_p,
            'n_pooled_active_bars': n_active,
            'wall_clock_sec': elapsed,
        }
        completed.append(row)
        # Write after EVERY shuffle -- a 6.75-hour run should never lose
        # more than one shuffle's work to an interruption.
        pd.DataFrame(completed).to_csv(RESULTS_CSV_PATH, index=False)
        print(f'[shuffle {i + 1}/{N_SHUFFLES}] t_stat={t_stat:+.4f}, '
              f'p={t_p:.4f} ({elapsed:.1f}s) -- saved to {RESULTS_CSV_PATH}')

    df = pd.DataFrame(completed)
    n_total = len(df)
    n_sig_two_sided = int((df['mean_pnl_t_p_value'] < 0.05).sum())
    n_sig_same_direction = int(
        ((df['mean_pnl_t_p_value'] < 0.05) & (df['mean_pnl_t_stat'] > 0)).sum()
    )

    print(f'\n{"=" * 74}')
    print(f'NULL CALIBRATION RESULT ({n_total} label-shuffle repeats)')
    print(f'{"=" * 74}')
    print(f'p < 0.05, either direction : {n_sig_two_sided}/{n_total} '
          f'({100 * n_sig_two_sided / n_total:.1f}%)  [nominal: 5%]')
    print(f'p < 0.05, POSITIVE t_stat  : {n_sig_same_direction}/{n_total} '
          f'({100 * n_sig_same_direction / n_total:.1f}%)  [nominal: 2.5%, '
          f'since the real result was one-directional]')
    print('\nCompare against the real (unshuffled) run: '
          'mean_pnl_t_stat=+2.0025, mean_pnl_t_p_value=0.0453')
    print('\nINTERPRETATION: if the positive-direction false-positive rate '
          'above is close to 2.5%, the real p=0.0453 result means roughly '
          'what it claims -- weak evidence, but not an artifact of the '
          'procedure. If it is meaningfully higher (e.g. 10%+), this '
          'exact procedure produces false positives at an inflated rate '
          'under a KNOWN null, and the real result should not be trusted '
          'at face value -- consistent with, and would reinforce, the '
          'existing null findings on real BTC.')


if __name__ == '__main__':
    main()
