"""
pipeline/diagnostics/trace_cv_fold_class_balance.py

Follow-up to trace_momentum_signal_leakage.py (2026-08-25 session). That
script found: at continuation_prob=0.7 (raw price autocorr=0.679), several
features correlate meaningfully with the event label at the event level
(parkinson_vol_20bar r=+0.22, amihud_lambda_20bar r=+0.22, fracdiff
r=-0.21, kyle_lambda r=-0.20 against `bin`) -- yet the winning trial's real
out-of-sample directional accuracy was 0.445, BELOW a coin flip, and every
one of the 20 trials had a negative Sharpe. StandardScaler is already in
the SVC pipeline (ruled out as the cause).

HYPOTHESIS: chronological PurgedKFold (Ch07, N_SPLITS=4, PCT_EMBARGO=0.12
-- established constants, calibrated on real BTC's near-random-walk return
series, Ch13 phi_hat~1.03) may not suit this generator's output. At
continuation_prob=0.7 the injected process produces long, persistent
directional runs (autocorr 0.68 is far larger than anything in real BTC).
A chronological fold split on a persistently-trending series can produce
severe train/test class-distribution shift -- a fold's training data might
be almost entirely one direction (e.g. a single long uptrend), and its
test fold the opposite -- which would make even a TRIVIAL "always predict
train's majority class" baseline perform below chance, with no need to
invoke any bug in the classifier itself.

METHOD: reproduce continuation_prob=0.7/seed=0 through the exact same
chain (generate -> rebuild -> features -> stage), then run PurgedKFold
directly (the same real Ch07 class, same N_SPLITS/PCT_EMBARGO/t1 the real
trial grid uses) and for each fold report:
  - train fold class balance (count and % of each bin value)
  - test fold class balance
  - accuracy of a TRIVIAL majority-class baseline (train's majority class,
    applied to every test-fold row) -- isolates whether below-chance
    performance is a property of the fold split itself, independent of
    the SVC

If the trivial baseline is ALSO below chance on average across folds, that
confirms a fold-level distribution-shift artifact of this generator's
persistence colliding with chronological CV -- not a broken classifier,
and not evidence against the real-BTC null findings (BTC doesn't have
this kind of persistence). If the trivial baseline is fine (~50%+) but the
real SVC still underperforms, that points back at the classifier itself.

Diagnostic-only: no new AFML formula, reuses ch07's real PurgedKFold
directly, no modification to any committed chapter module.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\trace_cv_fold_class_balance.py
    python pipeline\\diagnostics\\trace_cv_fold_class_balance.py --continuation-prob 0.7 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
EDGE_HARNESS_DIR = os.path.join(PIPELINE_DIR, 'edge_harness')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, EDGE_HARNESS_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from positive_control_data import generate_momentum_trades    # noqa: E402
from rebuild import build_bars_and_labels                     # noqa: E402
from features import build_enriched_events                     # noqa: E402
from live_staging import stage_live_training_tables             # noqa: E402
from ch07.cross_validation.purged_kfold import PurgedKFold      # noqa: E402

DIAGNOSTICS_DIR = HERE
MOMENTUM_BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_momentum_baseline_params.json'
)

N_TRADES = 360_000
TARGET_BARS = 3000
TOTAL_SPAN_HOURS = 720.0

# Same established constants Ch11's real driver uses -- reused, not
# reinvented, so this diagnostic tests the EXACT same fold behavior the
# real trial grid experiences.
N_SPLITS, PCT_EMBARGO = 4, 0.12

SWEEP_WORK_DIR = os.path.join(EDGE_HARNESS_DIR, 'trace_work_momentum')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')


def load_calibrated_params():
    if not os.path.exists(MOMENTUM_BASELINE_PARAMS_PATH):
        raise SystemExit(
            f'{MOMENTUM_BASELINE_PARAMS_PATH} not found -- run '
            'calibrate_synthetic_momentum_params.py first.'
        )
    with open(MOMENTUM_BASELINE_PARAMS_PATH) as f:
        params = json.load(f)
    if params['n_trades'] != N_TRADES or params['target_bars'] != TARGET_BARS:
        raise SystemExit(
            "synthetic_momentum_baseline_params.json scale doesn't match "
            f"this script's N_TRADES={N_TRADES}/TARGET_BARS={TARGET_BARS}."
        )
    return params['calibrated_tick_bp'], params['calibrated_noise_std']


def class_balance_str(y):
    vc = y.value_counts(normalize=False).sort_index()
    total = len(y)
    parts = [f'{int(k) if float(k).is_integer() else k}: {v} ({v/total:.1%})'
             for k, v in vc.items()]
    return f'n={total}  [' + ', '.join(parts) + ']'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation-prob', type=float, default=0.70)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tick_bp, noise_std = load_calibrated_params()
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('=' * 78)
    print(f'CV FOLD CLASS BALANCE: continuation_prob={args.continuation_prob}, '
          f'seed={args.seed}')
    print(f'PurgedKFold(n_splits={N_SPLITS}, pctEmbargo={PCT_EMBARGO})  '
          f'<- same real constants ch11 uses')
    print('=' * 78)

    synth = generate_momentum_trades(
        N_TRADES,
        continuation_prob=args.continuation_prob,
        tick_bp=tick_bp,
        noise_std=noise_std,
        total_span_hours=TOTAL_SPAN_HOURS,
        random_state=args.seed,
    )
    raw_trades = synth['raw_trades']
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    staged = stage_live_training_tables(rebuild_result, enriched_result, STAGING_DIR)

    enriched_path = os.path.join(STAGING_DIR, 'ch07_training_table_enriched.csv')
    events = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
    X, y, t1 = events[feature_cols], events['bin'], events['t1']

    print(f'\nOverall label distribution: {class_balance_str(y)}\n')

    pkf = PurgedKFold(n_splits=N_SPLITS, t1=t1, pctEmbargo=PCT_EMBARGO)

    fold_rows = []
    for i, (tr, te) in enumerate(pkf.split(X=X)):
        y_train = y.iloc[tr]
        y_test = y.iloc[te]

        majority_class = y_train.value_counts().idxmax()
        trivial_pred = pd.Series(majority_class, index=y_test.index)
        trivial_acc = (trivial_pred == y_test).mean()

        print(f'--- Fold {i+1}/{N_SPLITS} ---')
        print(f'  train: {class_balance_str(y_train)}')
        print(f'  test:  {class_balance_str(y_test)}')
        print(f'  train majority class = {majority_class}')
        print(f'  trivial "always predict train majority" OOS accuracy '
              f'on this fold = {trivial_acc:.4f}')
        print()

        fold_rows.append({
            'fold': i + 1,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'train_majority_class': majority_class,
            'train_majority_pct': (y_train == majority_class).mean(),
            'test_majority_class': y_test.value_counts().idxmax() if len(y_test) else np.nan,
            'test_majority_pct': y_test.value_counts(normalize=True).max() if len(y_test) else np.nan,
            'trivial_baseline_accuracy': trivial_acc,
        })

    df = pd.DataFrame(fold_rows)
    mean_trivial_acc = df['trivial_baseline_accuracy'].mean()

    print('=' * 78)
    print('SUMMARY')
    print('=' * 78)
    print(df.to_string(index=False))
    print(f'\nMean trivial-baseline OOS accuracy across all {N_SPLITS} folds: '
          f'{mean_trivial_acc:.4f}')
    print("""
INTERPRETATION:
  - If mean trivial-baseline accuracy is ALSO clearly below 0.5 (and/or
    individual folds show train_majority_class flipping direction between
    train and test), that confirms a fold-level train/test distribution
    shift caused by this generator's strong persistence colliding with
    chronological CV -- the real SVC's 0.445 OOS accuracy from
    trace_momentum_signal_leakage.py is then explained by the fold
    structure itself, not a classifier bug. This would NOT be evidence
    against the real-BTC null findings (BTC's actual returns don't show
    this kind of sustained persistence -- Ch13 phi_hat~1.03, consistent
    with a random walk).
  - If mean trivial-baseline accuracy is at or above 0.5 while the real
    SVC still scored 0.445, the fold split itself isn't the problem --
    the classifier is doing something actively wrong on top of a fold
    structure that's actually fine, and the next step would be looking at
    the SVC/getSignal stage directly (e.g. does `pred` derive from the
    right class index; is GAMMA=0.1 badly mismatched to this feature
    distribution).
""")

    out_path = os.path.join(DIAGNOSTICS_DIR, 'cv_fold_class_balance_trace.csv')
    df.to_csv(out_path, index=False)
    print(f'Per-fold results written to {out_path}')


if __name__ == '__main__':
    main()
