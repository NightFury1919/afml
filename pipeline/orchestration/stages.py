"""
pipeline/orchestration/stages.py

Orchestration layer chaining real, already-tested AFML chapter modules into
one continuous flow: enriched feature table -> multi-trial purged CV ->
out-of-sample PnL reconstruction -> PBO / DSR overfitting diagnostics ->
bet-sizing signal.

This module implements NO new AFML formula. Every calculation below
delegates to existing, real-machine-confirmed chapter code:
  - ch07/cross_validation/purged_kfold.py           (PurgedKFold)
  - ch10/bet_sizing/bet_sizing.py                    (getSignal)
  - ch11/backtest_dangers/pbo.py                     (pbo, sharpe_ratio)
  - ch14/backtest_statistics/backtest_statistics.py  (deflated_sharpe_ratio)
This file is pure glue: load data, run several classifier "trials" through
the SAME purged folds (so their out-of-sample PnL series share an identical
set of timestamps -- PBO's cscv() requires synchronous rows across trial
columns, per its own docstring), then hand off to report.py.

Phase 1 scope: operates on the EXISTING static March 2026 BTC/TUSD real
artifacts (ch07_training_table_enriched.csv, ch03_events.csv). Phase 2 will
swap load_enriched_table() for a live-pull equivalent that runs raw trades
through the same Ch02-05/17-19 bar/feature pipeline; every function below
this loading step is asset-and-data-source agnostic already.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# --- import real chapter modules directly from the repo, no reimplementation ---
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for rel in (
    os.path.join('ch07', 'cross_validation'),
    os.path.join('ch10', 'bet_sizing'),
    os.path.join('ch11', 'backtest_dangers'),
    os.path.join('ch14', 'backtest_statistics'),
):
    p = os.path.join(ROOT, rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from purged_kfold import PurgedKFold           # ch07, real module
from bet_sizing import getSignal                # ch10, real module
from pbo import pbo, sharpe_ratio               # ch11, real module
from backtest_statistics import deflated_sharpe_ratio  # ch14, real module

FEATURE_COLS = [
    'fracdiff', 'roll_c', 'roll_sigma_u', 'parkinson_vol_20bar',
    'corwin_schultz_spread', 'becker_parkinson_sigma', 'kyle_lambda',
    'amihud_lambda_20bar', 'vpin_10bar', 'round_number_fraction',
    'serial_corr_signed_flow', 'tick_rule_accuracy',
]

AVG_U = 0.2288  # Ch04's mean sample uniqueness -> BaggingClassifier max_samples,
                 # same convention as ch07/chapter_7_cross_validation.py


def load_enriched_table(input_data_dir):
    """Load Ch19's enriched 12-feature training table (X, y, w, t1), plus
    the realized return `ret` merged in from Ch03's events artifact --
    the enriched table itself carries bin/w/t1 but not the raw `ret` needed
    to reconstruct out-of-sample PnL.

    Returns
    -------
    X : pd.DataFrame (12 feature columns, index = event start time t0)
    y : pd.Series (bin, -1/1)
    w : pd.Series (Ch04 sample weight)
    t1 : pd.Series (label end time, index = t0)
    ret : pd.Series (realized return over [t0, t1], index = t0)
    """
    table_path = os.path.join(input_data_dir, 'ch07_training_table_enriched.csv')
    events_path = os.path.join(input_data_dir, 'ch03_events.csv')

    table = pd.read_csv(table_path, index_col=0, parse_dates=True).sort_index()
    events = pd.read_csv(events_path, index_col=0, parse_dates=True).sort_index()

    X = table[FEATURE_COLS]
    y = table['bin']
    w = table['w']
    t1 = pd.to_datetime(table['t1'])
    ret = events.loc[table.index, 'ret']

    return X, y, w, t1, ret


def default_trials(random_state=1):
    """A small, honest set of trial configurations, not a hyperparameter
    fishing expedition. Mirrors Ch07's own RandomForest/BaggingClassifier
    choices (including Ch04's avgU as BaggingClassifier max_samples) plus
    one deeper RandomForest variant -- giving PBO/DSR a genuine, if small,
    set of real trials to evaluate selection risk across."""
    return {
        'rf_shallow': RandomForestClassifier(
            n_estimators=100, max_depth=3, class_weight='balanced_subsample',
            random_state=random_state,
        ),
        'rf_deep': RandomForestClassifier(
            n_estimators=100, max_depth=None, class_weight='balanced_subsample',
            random_state=random_state,
        ),
        'bagging_avgU': BaggingClassifier(
            n_estimators=100, max_samples=AVG_U, random_state=random_state,
        ),
    }


def run_trial_oos_pnl(clf, X, y, w, t1, ret, n_splits=4, pct_embargo=0.12):
    """Run one classifier through PurgedKFold, stitch each fold's held-out
    predictions into a single out-of-sample PnL series covering the full
    sample: pnl_t = ret_t * pred_t (a naive unit-size directional bet; real
    sizing happens later via Ch10's getSignal, not here). Also returns the
    stitched out-of-sample class-probability table for later bet sizing.
    """
    gen = PurgedKFold(n_splits=n_splits, t1=t1, pctEmbargo=pct_embargo)
    pnl = pd.Series(index=X.index, dtype=float)
    prob_frames = []

    for train, test in gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :], y=y.iloc[train],
            sample_weight=w.iloc[train].values,
        )
        pred = fit.predict(X.iloc[test, :])
        pnl.iloc[test] = ret.iloc[test].to_numpy() * pred

        prob = fit.predict_proba(X.iloc[test, :])
        prob_frames.append(
            pd.DataFrame(prob, index=X.index[test], columns=fit.classes_)
        )

    probs = pd.concat(prob_frames).sort_index()
    return pnl, probs


def assemble_pnl_matrix(X, y, w, t1, ret, trials, n_splits=4, pct_embargo=0.12):
    """Run every trial through the SAME PurgedKFold configuration (identical
    n_splits/t1/pctEmbargo -> identical, deterministic fold boundaries), so
    each trial's stitched OOS PnL series covers the identical set of
    timestamps. This gives PBO's cscv() a true (T, N) matrix with
    synchronous rows, per its documented requirement."""
    pnl_columns = {}
    trial_probs = {}
    for name, clf in trials.items():
        pnl, probs = run_trial_oos_pnl(clf, X, y, w, t1, ret, n_splits, pct_embargo)
        pnl_columns[name] = pnl
        trial_probs[name] = probs
    M = pd.DataFrame(pnl_columns)
    return M, trial_probs


def evaluate_overfitting(M, S=8):
    """Wraps Ch11's real pbo() and Ch14's real deflated_sharpe_ratio() on
    the trial PnL matrix M. Returns per-trial Sharpe, the best trial, the
    PBO probability, and the deflated Sharpe ratio of the best trial given
    how many trials were actually tried (multiple-testing correction)."""
    trial_sharpes = M.apply(sharpe_ratio, axis=0)
    best_trial = trial_sharpes.idxmax()
    sr_hat = trial_sharpes[best_trial]

    prob_overfit, cscv_df = pbo(M, S=S)

    n_trials = M.shape[1]
    var_sr_trials = float(trial_sharpes.var(ddof=1)) if n_trials > 1 else 0.0
    T = M.shape[0]
    dsr = deflated_sharpe_ratio(sr_hat, var_sr_trials, n_trials, T)

    return {
        'trial_sharpes': trial_sharpes,
        'best_trial': best_trial,
        'sr_hat': sr_hat,
        'prob_overfit': prob_overfit,
        'cscv_df': cscv_df,
        'n_trials': n_trials,
        'var_sr_trials': var_sr_trials,
        'T': T,
        'dsr': dsr,
    }


def latest_bet_signal(best_trial_name, trial_probs, t1, step_size=0.05,
                       num_threads=1):
    """Use the best trial's out-of-sample class probabilities at the most
    recent available event to size a bet via Ch10's real getSignal. Returns
    a discretized signal in [-1, 1], or None if unavailable."""
    probs = trial_probs[best_trial_name]
    if probs.empty:
        return None

    num_classes = probs.shape[1]
    latest_t0 = probs.index.max()
    row = probs.loc[latest_t0]
    pred_class = row.idxmax()
    prob_val = row.max()

    events = pd.DataFrame({'t1': [t1.loc[latest_t0]]}, index=[latest_t0])
    prob = pd.Series([prob_val], index=[latest_t0])
    pred = pd.Series([pred_class], index=[latest_t0])

    signal = getSignal(
        events=events, stepSize=step_size, prob=prob, pred=pred,
        numClasses=num_classes, numThreads=num_threads,
    )
    if latest_t0 not in signal.index:
        return None
    val = signal[latest_t0]
    return float(val) if pd.notna(val) else None
