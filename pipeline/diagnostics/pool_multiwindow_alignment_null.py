"""
pipeline/diagnostics/pool_multiwindow_alignment_null.py

RETURN-ALIGNMENT NULL for pool_multiwindow_leave_one_out.py's decisive
metric (pooled mean-PnL t-stat across 8 real leave-one-window-out folds).

WHY THIS REPLACES THE LABEL-SHUFFLE NULL (2026-09-18)
-----------------------------------------------------
pool_multiwindow_null_calibration.py shuffled 'bin' inside each window
and re-ran the whole procedure 30 times. All 30 t-stats came out negative
(mean -1.55, sd 0.67). That looked like a broken null, but it is really a
CONDITIONAL null: every shuffle reuses the same real features and the
same real returns, so the 30 draws share whatever fixed alignment those
real features happen to have with those real returns. v2's per-window
correlations repeated their sign across seeds, which is the same story.
The shuffle spread (sd 0.67) is spread AROUND a data-specific offset, not
around zero, so no p-value can be read off it.

THIS SCRIPT'S NULL
------------------
Keep everything real -- real labels, real features, real classifiers,
real selection grid -- and break only the alignment between the
POSITIONS and the RETURNS they are scored against:

    for each window, circularly shift its bar-return series by a random
    offset k (positions stay put; returns roll under them).

Circular shifting preserves the return series' own autocorrelation and
volatility clustering (which the bar-level t-test's iid assumption
ignores) while destroying any real position<->return timing. Because the
same shift is used for a window whenever that window's returns are used
(as a selection-validation window or as the held-out test window),
each draw is one coherent "returns arrived at the wrong time" world, and
the FULL selection-then-evaluate logic is re-run inside it.

The statistic is the SAME pooled mean-PnL t-stat, so the empirical null
distribution of that exact statistic absorbs the autocorrelation/pooling
problem automatically. The real result is compared to it directly.

WHY THIS IS CHEAP
-----------------
Classifier predictions never depend on returns. So the 8x4x7 selection
fits + 8x4 held-out refits are done ONCE (deduplicated: the inner fit for
(test=a, val=b) and (test=b, val=a) uses the identical training pool, so
it is fitted once and predicts on both), the resulting lagged positions
are cached, and every roll only recomputes  position * rolled_returns
plus Sharpe scoring. ~144 SVC fits once (~8 min), then ~0.1-0.3 s/roll.

SELF-CHECK
----------
Draw 0 uses zero shifts and must reproduce the real run exactly (pooled
t=+2.0025, n_active=5455, and each fold's winner C/step and active-bar
count from pool_multiwindow_fold_summary.csv). The script prints
MATCH/MISMATCH. If it says MISMATCH, do not trust the rolls.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\pool_multiwindow_alignment_null.py
    # options: --n-rolls 1000  --min-shift 50  --seed 20260918  --rebuild-cache

Cache: pipeline/diagnostics/alignment_null_cache.pkl (mlfinlab env only --
do not share across envs; do not commit).
"""
import argparse
import importlib.util
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_SCRIPT_PATH = os.path.join(HERE, 'pool_multiwindow_leave_one_out.py')
RESULTS_CSV_PATH = os.path.join(HERE, 'pool_multiwindow_alignment_null_results.csv')
CACHE_PATH = os.path.join(HERE, 'alignment_null_cache.pkl')

spec = importlib.util.spec_from_file_location('pool_multiwindow_leave_one_out', REAL_SCRIPT_PATH)
real = importlib.util.module_from_spec(spec)
spec.loader.exec_module(real)


# ---------------------------------------------------------------------------
# Position caching (returns-independent part of realized_pnl)
# ---------------------------------------------------------------------------
def lagged_position(events_eval, close_eval, step, prob, pred):
    """Exactly real.realized_pnl() minus the final multiplication by the
    bar returns. Returns pos.shift(1) aligned to close.pct_change().dropna()."""
    ev = events_eval.loc[prob.index]
    sig = real.getSignal(ev, step, prob, pred, numClasses=2, numThreads=1)
    pos = sig.reindex(close_eval.index, method='ffill').fillna(0.0)
    bar_ret = close_eval.pct_change().dropna()
    return pos.shift(1).reindex(bar_ret.index).fillna(0.0).values


def build_cache(window_tables, feature_cols, verbose=True):
    ids = list(real.WINDOW_DIRS)
    pos = {}
    ret = {w: window_tables[w][1].pct_change().dropna().values for w in ids}

    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    n_fits = len(pairs) * len(real.C_GRID) + len(ids) * len(real.C_GRID)
    done = 0
    t0 = time.time()

    # Inner (selection) fits: pool = all windows except {a, b}; the SAME
    # fit serves (test=a, val=b) and (test=b, val=a).
    for (a, b) in pairs:
        pool_ids = [w for w in ids if w not in (a, b)]
        pool = pd.concat([window_tables[w][0] for w in pool_ids], axis=0)
        for C in real.C_GRID:
            clf = real.fit_classifier(pool, feature_cols, C)
            for val, test in ((a, b), (b, a)):
                events_val, close_val = window_tables[val]
                prob, pred = real.predict_oos(clf, events_val, feature_cols)
                for step in real.STEP_GRID:
                    pos[('inner', C, val, test, step)] = lagged_position(
                        events_val, close_val, step, prob, pred)
            done += 1
            if verbose and done % 8 == 0:
                print(f'  cache: {done}/{n_fits} fits ({time.time() - t0:.0f}s)', flush=True)

    # Held-out refits, for EVERY C (the winner depends on the rolled returns)
    for test in ids:
        pool_ids = [w for w in ids if w != test]
        pool = pd.concat([window_tables[w][0] for w in pool_ids], axis=0)
        events_test, close_test = window_tables[test]
        for C in real.C_GRID:
            clf = real.fit_classifier(pool, feature_cols, C)
            prob, pred = real.predict_oos(clf, events_test, feature_cols)
            for step in real.STEP_GRID:
                pos[('refit', C, test, step)] = lagged_position(
                    events_test, close_test, step, prob, pred)
            done += 1
            if verbose and done % 8 == 0:
                print(f'  cache: {done}/{n_fits} fits ({time.time() - t0:.0f}s)', flush=True)

    if verbose:
        print(f'  cache built: {n_fits} fits in {time.time() - t0:.0f}s')
    return {'pos': pos, 'ret': ret}


# ---------------------------------------------------------------------------
# One draw = full selection-then-evaluate logic under a given set of shifts
# ---------------------------------------------------------------------------
def pooled_t(pooled_pnl):
    active = pooled_pnl[pooled_pnl != 0]
    if len(active) >= 3 and active.std(ddof=1) > 0:
        t_stat = float(active.mean() / (active.std(ddof=1) / np.sqrt(len(active))))
        p = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(active) - 1)))
    else:
        t_stat, p = np.nan, np.nan
    return t_stat, p, int(len(active))


def evaluate_draw(cache, shifts):
    """shifts: {window_id: int}. Zero shifts == the real, unrolled run."""
    ids = list(real.WINDOW_DIRS)
    pos, ret = cache['pos'], cache['ret']
    rolled = {w: np.roll(ret[w], shifts.get(w, 0)) for w in ids}

    all_pnl, folds = [], []
    for test in ids:
        train_ids = [w for w in ids if w != test]
        scores = {(C, step): [] for C in real.C_GRID for step in real.STEP_GRID}
        for C in real.C_GRID:
            for val in train_ids:
                for step in real.STEP_GRID:
                    pnl = pos[('inner', C, val, test, step)] * rolled[val]
                    scores[(C, step)].append(real.sharpe_ratio(pd.Series(pnl)))
        avg_scores = {k: float(np.mean(v)) for k, v in scores.items()}
        winner_C, winner_step = max(avg_scores, key=avg_scores.get)
        test_pnl = pos[('refit', winner_C, test, winner_step)] * rolled[test]
        all_pnl.append(test_pnl)
        folds.append({'test_window': test, 'winner_C': winner_C,
                      'winner_step': winner_step,
                      'n_test_active_bars': int((test_pnl != 0).sum())})

    pooled = np.concatenate(all_pnl)
    t_stat, p, n_active = pooled_t(pooled)
    return {'t_stat': t_stat, 't_p_value': p, 'n_active': n_active,
            'mean_pnl': float(pooled[pooled != 0].mean()) if n_active else np.nan,
            'folds': folds, 'pooled_pnl': pooled}


def draw_shifts(rng, ret_lengths, min_shift):
    shifts = {}
    for w, n in ret_lengths.items():
        lo, hi = min_shift, n - min_shift
        if hi < lo:
            raise ValueError(f'window {w}: {n} bars too short for min_shift={min_shift}')
        shifts[w] = int(rng.integers(lo, hi + 1))
    return shifts


def empirical_p(real_t, null_t):
    null_t = np.asarray(null_t, dtype=float)
    null_t = null_t[np.isfinite(null_t)]
    n = len(null_t)
    one_sided = (1 + int((null_t >= real_t).sum())) / (n + 1)
    two_sided = (1 + int((np.abs(null_t) >= abs(real_t)).sum())) / (n + 1)
    return one_sided, two_sided, n


# ---------------------------------------------------------------------------
def selfcheck_against_real_csvs(draw0):
    """Compare draw 0 to the committed real-run CSVs. Returns True/False/None."""
    res_path = os.path.join(HERE, 'pool_multiwindow_leave_one_out_results.csv')
    fold_path = os.path.join(HERE, 'pool_multiwindow_fold_summary.csv')
    if not (os.path.exists(res_path) and os.path.exists(fold_path)):
        print('SELF-CHECK skipped: real-run CSVs not found.')
        return None
    real_res = pd.read_csv(res_path).iloc[0]
    real_folds = pd.read_csv(fold_path)
    ok_t = abs(draw0['t_stat'] - real_res['mean_pnl_t_stat']) < 1e-6
    ok_n = draw0['n_active'] == int(real_res['n_pooled_active_bars'])
    fold_ok = True
    for f in draw0['folds']:
        r = real_folds[real_folds['test_window'] == f['test_window']].iloc[0]
        if not (np.isclose(f['winner_C'], r['winner_C'])
                and np.isclose(f['winner_step'], r['winner_step'])
                and f['n_test_active_bars'] == int(r['n_test_active_bars'])):
            fold_ok = False
    print(f"SELF-CHECK draw 0 vs real run: t {draw0['t_stat']:+.4f} vs "
          f"{real_res['mean_pnl_t_stat']:+.4f} ({'ok' if ok_t else 'DIFF'}); "
          f"n_active {draw0['n_active']} vs {int(real_res['n_pooled_active_bars'])} "
          f"({'ok' if ok_n else 'DIFF'}); fold winners/active counts "
          f"({'ok' if fold_ok else 'DIFF'})")
    good = ok_t and ok_n and fold_ok
    print('SELF-CHECK RESULT:', 'MATCH' if good else 'MISMATCH -- do not trust the rolls')
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-rolls', type=int, default=1000)
    ap.add_argument('--min-shift', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--rebuild-cache', action='store_true')
    args = ap.parse_args()

    print('Loading the 8 real window tables (cached staging)...')
    window_tables, feature_cols = {}, None
    for w in real.WINDOW_DIRS:
        events, close = real.build_window_table(w)
        window_tables[w] = (events, close)
        feature_cols = feature_cols or [c for c in events.columns if c not in ('bin', 'w', 't1')]

    if os.path.exists(CACHE_PATH) and not args.rebuild_cache:
        print(f'Loading position cache {CACHE_PATH} ...')
        with open(CACHE_PATH, 'rb') as fh:
            cache = pickle.load(fh)
    else:
        print('Building position cache (one-time, ~8 min)...')
        cache = build_cache(window_tables, feature_cols)
        with open(CACHE_PATH, 'wb') as fh:
            pickle.dump(cache, fh)

    print('\nDraw 0: zero shifts (must reproduce the real run)')
    draw0 = evaluate_draw(cache, {})
    selfcheck_against_real_csvs(draw0)
    real_t = draw0['t_stat']

    ret_lengths = {w: len(cache['ret'][w]) for w in cache['ret']}
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for i in range(args.n_rolls):
        shifts = draw_shifts(rng, ret_lengths, args.min_shift)
        out = evaluate_draw(cache, shifts)
        rows.append({'roll': i + 1, 't_stat': out['t_stat'],
                     't_p_value': out['t_p_value'], 'n_active': out['n_active'],
                     'mean_pnl': out['mean_pnl']})
        if (i + 1) % 100 == 0 or i + 1 == args.n_rolls:
            pd.DataFrame(rows).to_csv(RESULTS_CSV_PATH, index=False)
            print(f'  roll {i + 1}/{args.n_rolls} ({time.time() - t0:.0f}s)', flush=True)

    df = pd.DataFrame(rows)
    null_t = df['t_stat'].dropna().values
    p1, p2, n = empirical_p(real_t, null_t)

    print(f'\n{"=" * 74}')
    print(f'ALIGNMENT NULL ({n} random circular return shifts per window)')
    print(f'{"=" * 74}')
    print(f'real pooled t-stat      = {real_t:+.4f}')
    print(f'null t-stat mean / sd   = {null_t.mean():+.4f} / {null_t.std(ddof=1):.4f}   '
          f'[a healthy null is centred near 0]')
    print(f'null 2.5% / 97.5% quant = {np.quantile(null_t, 0.025):+.3f} / '
          f'{np.quantile(null_t, 0.975):+.3f}')
    print(f'naive iid-bar p (real)  = {draw0["t_p_value"]:.4f}')
    print(f'EMPIRICAL p, one-sided (null >= real)  = {p1:.4f}')
    print(f'EMPIRICAL p, two-sided (|null| >= |real|) = {p2:.4f}')
    print(f'\nSaved -> {RESULTS_CSV_PATH}')
    print('\nHOW TO READ IT: if the null is centred near 0 and the empirical '
          'p is well above 0.05, the real t=+2.0 is what alignment-free '
          'noise produces this often (the naive iid p=0.045 overstated it). '
          'If the empirical p is small, the real timing alignment is '
          'unusual under a null that respects the return series\' own '
          'autocorrelation. If the null mean is far from 0, the '
          'procedure has a bias that survives even without any alignment, '
          'which points at pooling/selection mechanics.')


if __name__ == '__main__':
    main()


# =============================================================================
# TDD RESULTS (synthetic data, sandbox run 2026-09-18: Python 3.12.3, pytest 9.1.1
# -- NOT the mlfinlab env; re-run there and overwrite this block if it differs)
# =============================================================================
# test_pool_multiwindow_alignment_null.py::test_zero_shift_reproduces_real_procedure_exactly PASSED
# test_pool_multiwindow_alignment_null.py::test_shifting_changes_pnl_but_not_its_length_or_position_cache PASSED
# test_pool_multiwindow_alignment_null.py::test_full_length_shift_is_identity PASSED
# test_pool_multiwindow_alignment_null.py::test_draw_shifts_respect_bounds_and_are_reproducible PASSED
# test_pool_multiwindow_alignment_null.py::test_empirical_p_known_values PASSED
# test_pool_multiwindow_alignment_null.py::test_noise_only_null_is_centred_near_zero PASSED
# test_pool_multiwindow_alignment_null.py::test_planted_edge_is_detected_against_the_alignment_null PASSED
# ============================== 7 passed in 33.13s ==============================
