"""
pipeline/diagnostics/trace_momentum_signal_leakage.py

Ethan's question (2026-08-25 session): the 400-combo momentum sweep found
DSR flat ~0.50-0.54 across the ENTIRE signal-strength grid (correlation
with signal 0.031), even though the 2026-08-19 detection-power calibration
(calibrate_detection_power.py) shows DSR *should* respond meaningfully to
a real injected edge at this project's actual N=20/T~66-200 scale (mean
DSR climbing to 0.57-0.72 at true_sharpe=0.2). That mismatch is the
anomaly -- not DSR's formula (book-faithful, TDD-confirmed), and not the
DSR>=0.95 threshold (already known to be nearly unreachable at this scale
even with a real edge, per calibrate_min_reliable_T.py).

HYPOTHESIS: the injected price-level momentum (bar_lag1_autocorr, verified
present in rebuild_result['close']) never reaches the classifier, because
this pipeline's feature set (Ch19's 9 microstructural features + Ch05
frac-diff, see features.py) contains no literal lagged-return / momentum
feature -- it's liquidity/toxicity/volatility-oriented, per the book's own
Ch19 scope. Frac-diff is the only feature with any direct claim on price-
level memory, and whether pure autocorrelation shows up usably through
that channel is untested.

METHOD: run ONE strong-signal seed (continuation_prob near the top of the
sweep grid, well above the 0.566 null) through the EXACT same chain
run_momentum_edge_sweep.py uses, and at each stage report whether the
injected signal is present and detectable:

  1. Raw bars:        bar_lag1_autocorr on rebuild_result['close']
                       (this is what the sweep already logs -- confirms
                       the injected signal really is in the price series)
  2. Feature table:    correlation of EACH feature (9 Ch19 + fracdiff)
                        against the NEXT bar's raw return -- does any
                        feature actually carry directional information?
  3. Enriched events:  correlation of each feature against the realized
                        event-level label (bin: -1/0/1) -- same question,
                        at the event/label granularity the classifier
                        actually trains on
  4. Out-of-sample:    for the winning trial's own out-of-sample
                        probability (ch11.out_of_sample_probs), does
                        predicted probability actually track the true
                        label direction better than chance?

This is diagnostic-only: no new AFML formula, no change to any committed
chapter or pipeline module. Read-only against the real chain.

SYNTHETIC BY DESIGN (per CLAUDE.md): uses positive_control_data's
calibrated momentum generator at a KNOWN strong continuation_prob --
same precedent as the sweep itself.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\trace_momentum_signal_leakage.py
    python pipeline\\diagnostics\\trace_momentum_signal_leakage.py --continuation-prob 0.7 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
EDGE_HARNESS_DIR = os.path.join(PIPELINE_DIR, 'edge_harness')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, EDGE_HARNESS_DIR)

from positive_control_data import generate_momentum_trades    # noqa: E402
from rebuild import build_bars_and_labels                     # noqa: E402
from features import build_enriched_events                     # noqa: E402
from live_staging import stage_live_training_tables             # noqa: E402
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402
from momentum_correlation import bar_lag1_autocorr               # noqa: E402

DIAGNOSTICS_DIR = HERE
MOMENTUM_BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_momentum_baseline_params.json'
)

N_TRADES = 360_000
TARGET_BARS = 3000
TOTAL_SPAN_HOURS = 720.0
PBO_S = 12

SWEEP_WORK_DIR = os.path.join(EDGE_HARNESS_DIR, 'trace_work_momentum')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')


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
            f"this script's N_TRADES={N_TRADES}/TARGET_BARS={TARGET_BARS} "
            "-- re-calibrate at this scale first."
        )
    return params['calibrated_tick_bp'], params['calibrated_noise_std']


def corr_report(feature_table, target, target_name, label):
    """Pearson correlation of every feature column against `target`,
    printed sorted by |corr| descending. `target` must share (a subset
    of) feature_table's index."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continuation-prob', type=float, default=0.70,
                         help='Deliberately strong, well above the sweep '
                              'null grid top (0.566), to make any real '
                              'leakage as easy as possible to see.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tick_bp, noise_std = load_calibrated_params()
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('=' * 78)
    print(f'TRACING continuation_prob={args.continuation_prob}, seed={args.seed}')
    print('=' * 78)

    # --- Stage 1: generate + build bars/labels, confirm raw signal exists ---
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
    close = rebuild_result['close']
    raw_corr = bar_lag1_autocorr(close.values)
    print(f'\n[Stage 1] Raw bars: n_bars={len(rebuild_result["bars"])}, '
          f'n_events={len(rebuild_result["events"])}')
    print(f'  bar_lag1_autocorr (close-to-close) = {raw_corr:.4f}  '
          f'<- injected signal, confirmed present in price series')

    # next-bar raw return, for a direct "does raw price autocorr exist"
    # sanity check independent of any feature computation
    next_ret = close.pct_change().shift(-1)
    this_ret = close.pct_change()
    both = pd.concat([this_ret.rename('ret_t'), next_ret.rename('ret_t+1')], axis=1).dropna()
    manual_corr = both['ret_t'].corr(both['ret_t+1'])
    print(f'  manual check (ret_t vs ret_t+1 corr)  = {manual_corr:.4f}  '
          f'(should ~match bar_lag1_autocorr above)')

    # --- Stage 2: build features, check feature-vs-next-return correlation ---
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    feature_table = enriched_result['feature_table']
    print(f'\n[Stage 2] Feature table: {feature_table.shape[0]} bars x '
          f'{feature_table.shape[1]} features: {list(feature_table.columns)}')
    corr_report(feature_table, next_ret, 'next bar raw return', 'Stage 2 (bar-level)')

    # --- Stage 3: enriched events, check feature-vs-label correlation ---
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

    # --- Stage 4: run the real trial grid, check OOS probability quality ---
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
    print(f'  n_trials={eval_result["n_trials"]}, '
          f'var_sr_trials={eval_result["var_sr_trials"]:.6f}')
    print(f'  trial_sharpes range: [{eval_result["trial_sharpes"].min():.4f}, '
          f'{eval_result["trial_sharpes"].max():.4f}]')

    # Winning trial's own out-of-sample probability vs. true label --
    # the single most direct test of "did the classifier learn ANYTHING
    # about direction from these features."
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
    print(f'\n[Stage 4b] Winning trial (C={C}) out-of-sample predictions: '
          f'{len(prob)} predictions')
    if len(prob) >= 3:
        y_aligned = y.loc[prob.index]
        pred_correct = (pred == y_aligned)
        print(f'  OOS directional accuracy (pred == true bin): '
              f'{pred_correct.mean():.4f}  (0.5 = coin flip on a binary label)')
        # correlation between predicted probability of the positive class
        # and the true label, as a continuous-signal check independent of
        # the hard pred==y_aligned threshold
        if prob.nunique() > 1:
            prob_corr = np.corrcoef(prob.values.astype(float),
                                     y_aligned.values.astype(float))[0, 1]
            print(f'  corr(prob, true_label) = {prob_corr:+.4f}  '
                  f'(0 = no relationship)')
    else:
        print('  (too few OOS predictions to evaluate)')

    print('\n' + '=' * 78)
    print('INTERPRETATION GUIDE')
    print('=' * 78)
    print("""
  - Stage 1 confirms the injected signal is really in the price series
    (should be clearly nonzero, matching the sweep's own bar_lag1_autocorr
    logging at this continuation_prob).
  - Stage 2/3: if EVERY feature's |r| stays near 0 against next-bar return
    and against the event label, that's direct evidence the feature set
    (Ch19 microstructural + Ch05 fracdiff) doesn't carry the injected
    price-momentum signal at all -- supports the "no momentum-carrying
    feature exists" hypothesis, independent of the classifier/DSR stage.
  - Stage 4b: if OOS directional accuracy sits at ~0.5 and corr(prob,
    label) sits near 0 even at this strong, near-saturating
    continuation_prob, that confirms the classifier never learns the
    injected edge -- consistent with (and likely explained by) Stage 2/3's
    finding, not a separate DSR-stage problem.
  - If instead Stage 2/3 shows a real nonzero correlation on fracdiff (or
    any feature) but Stage 4b's OOS accuracy is still ~0.5, that would
    point at the classifier/CV/getSignal stage instead -- a different,
    separate diagnosis path.
""")


if __name__ == '__main__':
    main()
