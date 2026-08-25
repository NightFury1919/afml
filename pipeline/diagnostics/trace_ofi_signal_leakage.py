"""
pipeline/diagnostics/trace_ofi_signal_leakage.py

Direct follow-on to today's momentum trace (trace_momentum_signal_leakage.py
/ trace_cv_fold_class_balance.py), which found the 2026-08-24 momentum
sweep's flat-DSR null was explained by a chronological-CV regime-shift
artifact: continuation_prob's Markov-chain persistence produces long,
sustained single-direction regimes, which collide with PurgedKFold's
chronological split to produce below-chance OOS accuracy regardless of
whether a real edge is present.

Ethan's question (2026-08-25 session, continued): does the SAME mechanism
explain the 2026-08-23 OFI null (bar_aligned_scaled_50seeds.csv, 400
combos, DSR 0.508-0.526, correlation with signal strength 0.016, 0/400
DSR>=0.95), or is OFI's detection failure a separate, still-unexplained
finding?

STRUCTURAL DIFFERENCE, confirmed by direct source inspection before
writing this script: generate_bar_aligned_trades.py draws a FRESH,
INDEPENDENT z ~ N(0,1) for EVERY bar window (`z = rng.normal(0.0, 1.0,
size=n_windows)`), with each bar's injected drift tied to that bar's own
independent draw (`edge_strength * drift_dollars_per_unit_imbalance *
z[i-1]`). There is no persistence mechanism analogous to momentum's
continuation_prob Markov chain -- one bar's imbalance/drift has no
influence on the next bar's. This means OFI's injected process should
NOT produce the kind of long single-direction regime that caused
momentum's chronological-CV confound, and there's no a priori structural
reason to expect the same fold-class-shift artifact here.

METHOD: identical two-stage trace to the momentum version, run on OFI's
own generator at a deliberately strong edge_strength=1.0 (already known
to reach raw_signal_corr~0.39, per edge_sweep_extreme_sanity_check.csv --
about as strong as this generator gets; unlike momentum, OFI's
correlation does not keep climbing toward 1.0 even at saturating
edge_strength, which is itself worth noting for later interpretation):

  1. Confirm the injected signal is really present (raw_signal_corr,
     matching the sweep's own logged diagnostic).
  2. Feature-vs-label / feature-vs-return correlation at the event level
     -- does the signal reach the classifier's inputs?
  3. Real Ch11 trial grid: winning trial's DSR/PBO, and its own real
     out-of-sample directional accuracy / prob-vs-label correlation.
  4. PurgedKFold fold-by-fold class balance + trivial "predict train's
     majority class" baseline -- the same isolation test that confirmed
     the mechanism for momentum, run here to see whether it reproduces.

Diagnostic-only: no new AFML formula, no change to any committed chapter
or pipeline module. Read-only against the real chain.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\trace_ofi_signal_leakage.py
    python pipeline\\diagnostics\\trace_ofi_signal_leakage.py --edge-strength 1.0 --seed 0
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

from rebuild import build_bars_and_labels                       # noqa: E402
from features import build_enriched_events                       # noqa: E402
from live_staging import stage_live_training_tables               # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402
from generate_bar_aligned_trades import generate_bar_aligned_synthetic_trades  # noqa: E402
from ch07.cross_validation.purged_kfold import PurgedKFold        # noqa: E402

DIAGNOSTICS_DIR = HERE
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)

N_TRADES = 120_000
TARGET_BARS = 1000
PBO_S = 12
N_SPLITS, PCT_EMBARGO = 4, 0.12  # same real constants ch11 uses

SWEEP_WORK_DIR = os.path.join(EDGE_HARNESS_DIR, 'trace_work_ofi')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')


def load_calibration():
    if not os.path.exists(BASELINE_PARAMS_PATH):
        raise SystemExit(
            f'{BASELINE_PARAMS_PATH} not found -- run '
            'calibrate_synthetic_trade_params.py --source live first.'
        )
    with open(BASELINE_PARAMS_PATH) as f:
        params = json.load(f)
    mean_rate_per_sec = params['n_trades'] / (params['span_hours'] * 3600.0)
    return {
        'baseline_imbalance': params['baseline_imbalance'],
        'price_diff_std': params['price_diff_std'],
        'avg_trade_size': params['avg_trade_size'],
        'avg_trade_rate_per_sec': mean_rate_per_sec,
        'start_price': params['price_start'],
    }


def corr_report(feature_table, target, target_name, label):
    print(f'\n  --- {label}: feature correlation vs. {target_name} ---')
    common = feature_table.index.intersection(target.index)
    if len(common) < 3:
        print(f'    (too few overlapping rows: {len(common)})')
        return
    ft = feature_table.loc[common]
    tgt = target.loc[common].astype(float)
    rows = []
    for col in ft.columns:
        x = ft[col].astype(float)
        mask = x.notna() & tgt.notna()
        if mask.sum() < 3 or x[mask].std() == 0 or tgt[mask].std() == 0:
            rows.append((col, np.nan, int(mask.sum())))
            continue
        r = np.corrcoef(x[mask], tgt[mask])[0, 1]
        rows.append((col, r, int(mask.sum())))
    rows.sort(key=lambda t: (t[1] is not None and not np.isnan(t[1]), abs(t[1]) if t[1] == t[1] else -1), reverse=True)
    for col, r, n in rows:
        r_str = f'{r:+.4f}' if r == r else '  NaN '
        print(f'    {col:<32s} r={r_str}   (n={n})')


def class_balance_str(y):
    vc = y.value_counts(normalize=False).sort_index()
    total = len(y)
    parts = [f'{int(k) if float(k).is_integer() else k}: {v} ({v/total:.1%})'
             for k, v in vc.items()]
    return f'n={total}  [' + ', '.join(parts) + ']'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edge-strength', type=float, default=1.0,
                         help='Deliberately strong, well past the main '
                              'sweep grid top (0.5) -- reaches '
                              'raw_signal_corr~0.39 per the extreme '
                              'sanity check, about as strong as this '
                              'generator gets.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    calib = load_calibration()
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('=' * 78)
    print(f'OFI TRACE: edge_strength={args.edge_strength}, seed={args.seed}')
    print('=' * 78)

    # --- Stage 1: generate + confirm injected signal is present ---
    raw_trades, diag = generate_bar_aligned_synthetic_trades(
        n_trades=N_TRADES,
        target_bars=TARGET_BARS,
        edge_strength=args.edge_strength,
        seed=args.seed,
        baseline_imbalance=calib['baseline_imbalance'],
        price_diff_std=calib['price_diff_std'],
        avg_trade_rate_per_sec=calib['avg_trade_rate_per_sec'],
        avg_trade_size=calib['avg_trade_size'],
        start_price=calib['start_price'],
        return_diagnostics=True,
    )
    imb = diag['realized_imbalance']
    wmp = diag['window_mean_price']
    drift_next = np.diff(wmp)
    raw_corr = float(np.corrcoef(imb[:-1], drift_next)[0, 1])
    print(f'\n[Stage 1] n_windows={diag["n_windows"]}, '
          f'n_used_trades={diag["n_used_trades"]}')
    print(f'  raw_signal_corr (imbalance[i] vs. next-window drift) = '
          f'{raw_corr:.4f}  <- injected signal, confirmed present')

    # --- Stage 2: build real bars/events/features ---
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    enriched_events = enriched_result['enriched_events']
    print(f'\n[Stage 2] Enriched events: {enriched_result["n_events_before"]} '
          f'-> {enriched_result["n_events_after"]} after dropna, '
          f'fracdiff_d={enriched_result["fracdiff_d"]}')
    if enriched_result['n_events_after'] >= 3:
        feat_cols = [c for c in enriched_events.columns
                     if c not in ('t1', 'trgt', 'ret', 'bin', 'w')]
        corr_report(enriched_events[feat_cols], enriched_events['bin'],
                    'event label (bin)', 'Stage 2 (event-level)')
        corr_report(enriched_events[feat_cols], enriched_events['ret'],
                    'event realized return (ret)', 'Stage 2 (event-level)')
    else:
        print('    (too few enriched events to correlate)')

    print(f'\nOverall label distribution: '
          f'{class_balance_str(enriched_events["bin"])}')

    # --- Stage 3: run the real trial grid ---
    print('\n[Stage 3] Running Ch11 real trial grid (SVC x getSignal, '
          'purged CV)...')
    ch11 = load_ch11_driver()
    staged = stage_live_training_tables(rebuild_result, enriched_result, STAGING_DIR)
    M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)

    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    eval_result = evaluate_overfitting(M, meta, ch11, S=PBO_S, tw=tw_aligned)

    print(f'\n  T_raw={eval_result["T_raw"]}, tw_mean={eval_result["tw_mean"]:.4f}, '
          f'T_effective={eval_result["T"]:.2f}')
    print(f'  sr_hat={eval_result["sr_hat"]:.4f}, dsr={eval_result["dsr"]:.4f}, '
          f'pbo={eval_result["prob_overfit"]:.4f}')
    print(f'  trial_sharpes range: [{eval_result["trial_sharpes"].min():.4f}, '
          f'{eval_result["trial_sharpes"].max():.4f}]')

    best_trial = eval_result['best_trial']
    C = meta.loc[best_trial, 'C']
    enriched_path = os.path.join(STAGING_DIR, 'ch07_training_table_enriched.csv')
    events_for_oos = pd.read_csv(enriched_path, index_col=0, parse_dates=True)
    events_for_oos['t1'] = pd.to_datetime(events_for_oos['t1'])
    feat_cols_oos = [c for c in events_for_oos.columns if c not in ('bin', 'w', 't1')]
    X = events_for_oos[feat_cols_oos]
    y = events_for_oos['bin']
    w = events_for_oos['w']
    t1 = events_for_oos['t1']

    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)
    print(f'\n[Stage 3b] Winning trial (C={C}) out-of-sample predictions: '
          f'{len(prob)} predictions')
    if len(prob) >= 3:
        y_aligned = y.loc[prob.index]
        pred_correct = (pred == y_aligned)
        print(f'  OOS directional accuracy (pred == true bin): '
              f'{pred_correct.mean():.4f}  (0.5 = coin flip on a binary label)')
        if prob.nunique() > 1:
            prob_corr = np.corrcoef(prob.values.astype(float),
                                     y_aligned.values.astype(float))[0, 1]
            print(f'  corr(prob, true_label) = {prob_corr:+.4f}  '
                  f'(0 = no relationship)')
    else:
        print('  (too few OOS predictions to evaluate)')

    # --- Stage 4: PurgedKFold class balance, same isolation test as momentum ---
    print('\n' + '=' * 78)
    print('[Stage 4] PurgedKFold fold-by-fold class balance '
          f'(n_splits={N_SPLITS}, pctEmbargo={PCT_EMBARGO})')
    print('=' * 78)

    pkf = PurgedKFold(n_splits=N_SPLITS, t1=t1, pctEmbargo=PCT_EMBARGO)
    fold_rows = []
    for i, (tr, te) in enumerate(pkf.split(X=X)):
        y_train = y.iloc[tr]
        y_test = y.iloc[te]
        majority_class = y_train.value_counts().idxmax()
        trivial_pred = pd.Series(majority_class, index=y_test.index)
        trivial_acc = (trivial_pred == y_test).mean()

        print(f'\n--- Fold {i+1}/{N_SPLITS} ---')
        print(f'  train: {class_balance_str(y_train)}')
        print(f'  test:  {class_balance_str(y_test)}')
        print(f'  train majority class = {majority_class}')
        print(f'  trivial "always predict train majority" OOS accuracy '
              f'on this fold = {trivial_acc:.4f}')

        fold_rows.append({
            'fold': i + 1, 'n_train': len(y_train), 'n_test': len(y_test),
            'train_majority_class': majority_class,
            'train_majority_pct': (y_train == majority_class).mean(),
            'test_majority_class': y_test.value_counts().idxmax() if len(y_test) else np.nan,
            'test_majority_pct': y_test.value_counts(normalize=True).max() if len(y_test) else np.nan,
            'trivial_baseline_accuracy': trivial_acc,
        })

    df = pd.DataFrame(fold_rows)
    mean_trivial_acc = df['trivial_baseline_accuracy'].mean()
    print(f'\nMean trivial-baseline OOS accuracy across all {N_SPLITS} '
          f'folds: {mean_trivial_acc:.4f}')

    out_path = os.path.join(DIAGNOSTICS_DIR, 'ofi_cv_fold_class_balance_trace.csv')
    df.to_csv(out_path, index=False)
    print(f'Per-fold results written to {out_path}')

    print('\n' + '=' * 78)
    print('INTERPRETATION GUIDE')
    print('=' * 78)
    print("""
  - Stage 1 confirms the injected OFI signal is really present.
  - Stage 2: if feature-vs-label correlations are near 0 (unlike
    momentum's 0.15-0.22), that suggests OFI's edge doesn't reach the
    feature table the way momentum's did -- a DIFFERENT failure point
    than momentum's, worth its own follow-up.
  - Stage 3b: OOS accuracy near 0.5 with near-zero prob/label
    correlation, IF Stage 4's trivial baseline is ALSO near 0.5 (unlike
    momentum's 0.33), means the fold structure is fine here and the
    non-detection is real and NOT explained by the regime-shift artifact
    found for momentum -- i.e. a genuinely different, still-open finding.
  - Stage 4: if mean trivial-baseline accuracy comes back well below 0.5
    (like momentum's 0.33), the SAME regime-shift artifact would apply to
    OFI too, despite the structurally different (non-persistent)
    injection mechanism -- this would be a surprise given the module's
    per-bar-independent z draws, and would be worth understanding before
    trusting either sweep's null.
""")


if __name__ == '__main__':
    main()
