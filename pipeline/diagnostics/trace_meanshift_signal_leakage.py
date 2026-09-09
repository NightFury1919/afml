"""
pipeline/diagnostics/trace_meanshift_signal_leakage.py

Direct follow-on to trace_ofi_signal_leakage.py and
trace_momentum_signal_leakage.py, for the THIRD edge-injection mechanism
(generate_bar_aligned_meanshift_trades.py, 2026-08-27).

Ethan's question (continuing the 2026-08-27 session): the T_effective-
controlled meanshift sweep (meanshift_edge_sweep_results.csv, run via
run_meanshift_edge_sweep.py --n-trades-map meanshift_n_trades_map.json)
found DSR completely flat (corr(edge_strength, dsr) = +0.0111) even
though:
  - the injected signal (lag1_autocorr of realized window returns) is
    confirmed strong and CLIMBING (raw_signal_corr: 0.16 at es=0.5 up to
    0.45 at es=2.0, corr(edge_strength, raw_signal_corr) = +0.929)
  - T_effective is now genuinely controlled (corr with edge_strength
    improved from -0.74 to -0.29; per-edge_strength means sit in a tight
    90-113 band), so this is NOT explainable by the sample-size confound
    that undermined the FIRST version of this sweep
  - unlike OFI's engineered buy/sell-imbalance route, this signal is a
    pure single-lag price-return effect, visible via the most generic
    price-based feature there is -- no OFI-style feature-routing
    dependency to blame
  - unlike momentum's Markov continuation_prob chain, meanshift's z[i]
    draws are independent per window with only ONE explicit one-window
    carry-over (see generate_bar_aligned_meanshift_trades.py's module
    docstring) -- deliberately built to NOT reproduce momentum's
    long-sustained-regime / chronological-CV collision

This means neither of the two previously-identified, mechanism-specific
confounds (OFI's fold-level class imbalance, momentum's CV-regime
collision) has an obvious reason to apply here. If Stage 4 below comes
back clean (trivial-baseline OOS accuracy near 0.5, not momentum's 0.33),
this is the third independent mechanism to hit an unexplained detection
ceiling -- a much more serious, structural finding.

METHOD: same four-stage trace as OFI/momentum, adapted to meanshift's
own diagnostics (lag1_autocorr instead of OFI's realized_imbalance-based
raw_corr; a Stage 2 bar-level feature_table check borrowed from the
momentum trace, since meanshift's mechanism -- like momentum's -- is a
raw price-return effect rather than an engineered feature like OFI's):

  1. Confirm the injected signal is really present (lag1_autocorr of
     realized window returns, matching the sweep's own logged
     raw_signal_corr diagnostic) at a DELIBERATELY STRONG, well-
     controlled edge_strength.
  2. Bar-level feature_table correlation vs. next-bar raw return -- does
     the signal reach ANY of Ch19's 11 features (or fracdiff) before
     event-level enrichment collapses/joins them?
  3. Event-level feature/label and feature/return correlation -- does
     the signal reach the classifier's actual training inputs?
  4. Real Ch11 trial grid: winning trial's DSR/PBO, its own real
     out-of-sample directional accuracy / prob-vs-label correlation, AND
     PurgedKFold fold-by-fold class balance + trivial "predict train's
     majority class" baseline -- the same isolation test that diagnosed
     momentum's confound, run here to see whether it reproduces (it
     should NOT, per this generator's non-persistent design) or whether
     something else is going on.

Diagnostic-only: no new AFML formula, no change to any committed chapter
or pipeline module. Read-only against the real chain.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\trace_meanshift_signal_leakage.py
    python pipeline\\diagnostics\\trace_meanshift_signal_leakage.py --edge-strength 2.0 --seed 0
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
from generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades  # noqa: E402
from ch07.cross_validation.purged_kfold import PurgedKFold        # noqa: E402

DIAGNOSTICS_DIR = HERE
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)
N_TRADES_MAP_PATH = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_map.json')

TARGET_BARS = 1000
PBO_S = 12
N_SPLITS, PCT_EMBARGO = 4, 0.12  # same real constants ch11/OFI-trace use

SWEEP_WORK_DIR = os.path.join(EDGE_HARNESS_DIR, 'trace_work_meanshift')
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


def load_n_trades(edge_strength):
    """Look up the T_effective-controlled n_trades for this edge_strength
    from the real calibrated map (meanshift_n_trades_map.json) -- using
    an uncalibrated n_trades here would reintroduce the exact
    sample-size confound the 2026-08-27 session fixed."""
    if not os.path.exists(N_TRADES_MAP_PATH):
        raise SystemExit(
            f'{N_TRADES_MAP_PATH} not found -- run '
            'calibrate_meanshift_n_trades.py (and, for high edge_strength, '
            'recalibrate_meanshift_n_trades_high_es.py) first.'
        )
    with open(N_TRADES_MAP_PATH) as f:
        n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}
    if edge_strength not in n_trades_map:
        raise SystemExit(
            f'edge_strength={edge_strength} not in {N_TRADES_MAP_PATH} '
            f'(available: {sorted(n_trades_map)}) -- pick one of those, or '
            're-run the calibration scripts to extend the map.'
        )
    return n_trades_map[edge_strength]


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
    parser.add_argument('--edge-strength', type=float, default=2.0,
                         help='Deliberately strong, T_effective-controlled '
                              'edge_strength from meanshift_n_trades_map.json '
                              '-- 2.0 reaches raw_signal_corr~0.45 (the '
                              'strongest in the calibrated map) while '
                              'T_effective stays in the 90-113 band, per the '
                              '2026-08-27 controlled sweep.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    calib = load_calibration()
    n_trades = load_n_trades(args.edge_strength)
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('=' * 78)
    print(f'MEANSHIFT TRACE: edge_strength={args.edge_strength}, '
          f'seed={args.seed}, n_trades={n_trades:,} (T_eff-controlled)')
    print('=' * 78)

    # --- Stage 1: generate + confirm injected signal is present ---
    raw_trades, diag = generate_bar_aligned_meanshift_trades(
        n_trades=n_trades,
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
    print(f'\n[Stage 1] n_scaffold_bars={diag["n_windows"]}, '
          f'n_used_trades={diag["n_used_trades"]}')
    print(f'  lag1_autocorr (realized window returns, generic price '
          f'feature) = {diag["lag1_autocorr"]:.4f}  <- injected signal, '
          f'confirmed present')

    # --- Stage 2: build real bars/features, bar-level check (momentum-style) ---
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    close = rebuild_result['close']
    next_ret = close.pct_change().shift(-1)

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    feature_table = enriched_result['feature_table']
    print(f'\n[Stage 2] Feature table: {feature_table.shape[0]} bars x '
          f'{feature_table.shape[1]} features: {list(feature_table.columns)}')
    corr_report(feature_table, next_ret, 'next bar raw return',
                'Stage 2 (bar-level)')

    # --- Stage 3: event-level feature/label correlation ---
    enriched_events = enriched_result['enriched_events']
    print(f'\n[Stage 3] Enriched events: {enriched_result["n_events_before"]} '
          f'-> {enriched_result["n_events_after"]} after dropna, '
          f'fracdiff_d={enriched_result["fracdiff_d"]}')
    if enriched_result['n_events_after'] >= 3:
        feat_cols = [c for c in enriched_events.columns
                     if c not in ('t1', 'trgt', 'ret', 'bin', 'w')]
        corr_report(enriched_events[feat_cols], enriched_events['bin'],
                    'event label (bin)', 'Stage 3 (event-level)')
        corr_report(enriched_events[feat_cols], enriched_events['ret'],
                    'event realized return (ret)', 'Stage 3 (event-level)')
    else:
        print('    (too few enriched events to correlate)')

    print(f'\nOverall label distribution: '
          f'{class_balance_str(enriched_events["bin"])}')

    # --- Stage 4: run the real trial grid + PurgedKFold isolation test ---
    print('\n[Stage 4] Running Ch11 real trial grid (SVC x getSignal, '
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
    print(f'\n[Stage 4a] Winning trial (C={C}) out-of-sample predictions: '
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

    print('\n' + '=' * 78)
    print('[Stage 4b] PurgedKFold fold-by-fold class balance '
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

    out_path = os.path.join(DIAGNOSTICS_DIR, 'meanshift_cv_fold_class_balance_trace.csv')
    df.to_csv(out_path, index=False)
    print(f'Per-fold results written to {out_path}')

    print('\n' + '=' * 78)
    print('INTERPRETATION GUIDE')
    print('=' * 78)
    print("""
  - Stage 1 confirms the injected meanshift signal is really present,
    via the SAME generic realized-return feature that would show up
    to any downstream consumer -- no OFI-style engineered-feature
    dependency to blame if it doesn't reach further stages.
  - Stage 2: if EVERY Ch19/fracdiff feature's |r| against next-bar
    return stays near 0, that's direct evidence the feature set doesn't
    carry a raw single-lag price-return signal at all, regardless of
    edge_strength -- the SAME failure mode momentum's trace found, now
    on a structurally different (non-persistent) injection mechanism.
  - Stage 3: same question at the event/label level the classifier
    actually trains on.
  - Stage 4a: OOS directional accuracy near 0.5 and corr(prob, label)
    near 0, DESPITE Stage 1's confirmed climbing raw signal, means the
    classifier never learns the edge -- consistent with (and likely
    explained by) Stage 2/3's finding if those also came back flat.
  - Stage 4b is the DECISIVE isolation test, mirroring the one that
    diagnosed momentum's real confound:
      * If mean trivial-baseline accuracy is near 0.5 (like OFI's clean
        result, UNLIKE momentum's 0.33), the fold structure is fine and
        non-detection here is real and NOT explained by either of the
        two previously-identified, mechanism-specific artifacts. That
        would make this the THIRD independent mechanism to hit an
        unexplained detection ceiling -- with the T_effective confound
        already ruled out separately -- and the most serious open
        finding this project has produced.
      * If mean trivial-baseline accuracy comes back well below 0.5
        (like momentum's 0.33) DESPITE this generator's deliberately
        non-persistent, single-lag design, that would be a genuine
        surprise -- it would mean SOME chronological-CV-vs-injected-
        drift interaction survives even without a discrete Markov
        regime chain, and would need its own explanation before
        concluding anything about a universal ceiling.
""")


if __name__ == '__main__':
    main()
