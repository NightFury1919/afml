"""
pipeline/diagnostics/isolate_single_feature_meanshift.py

Direct follow-on to trace_meanshift_signal_leakage.py's real run
(2026-09-06 session, edge_strength=2.0, seed=0). That trace found a
result genuinely different from the OFI/momentum traces: the injected
signal does NOT die before reaching the feature table -- roll_c (Roll's
effective-spread estimator, built from the serial covariance of price
changes, mechanistically exactly the feature that should pick up an
injected lag-1 return relationship) showed r=+0.2894 against next-bar
return at the bar level, and the fold-balance isolation test (Stage 4b)
came back clean (mean trivial-baseline accuracy 0.5681, matching the
label's own 56.8% base rate almost exactly -- not momentum's below-
chance 0.33 artifact). Yet the winning trial's real OOS accuracy was
0.5052 and corr(prob, label) was -0.0166 -- the classifier extracted
essentially nothing, despite training on 14 features including one that
carries real signal.

OPEN QUESTION this script answers: does the classifier fail to learn
roll_c's real signal because it's a genuine model/CV-stage detection
ceiling (14 features and T_effective~106 is just below what an SVC can
resolve, consistent with this project's own "T>=200 needed for
meaningful discrimination" framing) -- or because 13 other, mostly-
uncorrelated features are DILUTING/crowding out roll_c's signal inside a
14-dimensional RBF kernel?

METHOD: run the EXACT same chain trace_meanshift_signal_leakage.py used
(same generator call, same rebuild/enrich, same real Ch11 out_of_sample_
probs/getSignal/PBO/DSR machinery), but STAGE ONLY t1/bin/w plus ONE
feature column (roll_c by default) before handing the training table to
Ch11's driver -- instead of all 14. Ch11's part_c_build_trials() derives
feature_cols generically from whatever's in the CSV (`[c for c in
events.columns if c not in ('bin', 'w', 't1')]`), so a single-feature
table is a fully legitimate input, not a hack around its interface.

If OOS accuracy/corr(prob,label) move meaningfully off ~0.50/~0.00 with
ONLY roll_c present, that points at dilution among the 13 other
features as the real culprit -- a fixable feature-selection problem, not
a fundamental ceiling. If it stays flat even with dilution removed, that
is much stronger evidence of a genuine model/CV-stage detection floor at
this pipeline's real scale (T_effective~106), independent of feature
count.

Diagnostic-only: no new AFML formula, no change to any committed chapter
or pipeline module (Ch11's chapter_11_backtest_dangers.py is reused via
the same monkeypatch-INPUT/HERE pattern stages.run_live_trials() already
uses -- this script only edits the STAGED CSV, never the chapter file).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\isolate_single_feature_meanshift.py
    python pipeline\\diagnostics\\isolate_single_feature_meanshift.py --feature roll_c --edge-strength 2.0 --seed 0
    python pipeline\\diagnostics\\isolate_single_feature_meanshift.py --feature roll_sigma_u
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

DIAGNOSTICS_DIR = HERE
BASELINE_PARAMS_PATH = os.path.join(
    DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json'
)
N_TRADES_MAP_PATH = os.path.join(DIAGNOSTICS_DIR, 'meanshift_n_trades_map.json')

TARGET_BARS = 1000
PBO_S = 12

SWEEP_WORK_DIR = os.path.join(EDGE_HARNESS_DIR, 'trace_work_meanshift_isolate')
STAGING_DIR = os.path.join(SWEEP_WORK_DIR, 'staging')
LIVE_HERE_DIR = os.path.join(SWEEP_WORK_DIR, 'live_here')

# From the real 2026-09-06 mlfinlab run of trace_meanshift_signal_leakage.py
# --edge-strength 2.0 --seed 0 -- the 14-feature baseline this script's
# single-feature result should be compared against. Not recomputed here
# (that would mean re-running the full 14-feature grid, which is exactly
# the run already on record) -- printed for direct side-by-side reading.
FULL_FEATURE_BASELINE = {
    'n_features': 14,
    'oos_accuracy': 0.5052,
    'prob_label_corr': -0.0166,
    'dsr': 0.4930,
    'pbo': 0.8485,
    'T_effective': 105.88,
}


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


def restrict_staged_table_to_one_feature(enriched_csv_path, feature):
    """Overwrite the just-staged ch07_training_table_enriched.csv in place,
    keeping only t1/bin/w plus ONE feature column. Ch11's part_c_build_
    trials() derives feature_cols generically from the CSV's own columns,
    so this is a legitimate single-feature input, not a workaround."""
    df = pd.read_csv(enriched_csv_path, index_col=0, parse_dates=True)
    if feature not in df.columns:
        available = [c for c in df.columns if c not in ('t1', 'bin', 'w')]
        raise SystemExit(
            f"feature '{feature}' not found in the staged table. "
            f'Available features: {available}'
        )
    restricted = df[['t1', 'bin', 'w', feature]].copy()
    restricted.to_csv(enriched_csv_path)
    return restricted.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, default='roll_c',
                         help="Single feature to isolate. Default 'roll_c' "
                              '-- the standout from the 2026-09-06 trace '
                              '(r=+0.2894 vs next-bar return, the only '
                              'feature with a clear mechanistic reason to '
                              "carry a lag-1 price-return signal).")
    parser.add_argument('--edge-strength', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    calib = load_calibration()
    n_trades = load_n_trades(args.edge_strength)
    os.makedirs(SWEEP_WORK_DIR, exist_ok=True)

    print('=' * 78)
    print(f'SINGLE-FEATURE ISOLATION: feature={args.feature!r}, '
          f'edge_strength={args.edge_strength}, seed={args.seed}, '
          f'n_trades={n_trades:,}')
    print('=' * 78)

    # --- Stage 1: generate + build bars/events/features (identical to trace) ---
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
    print(f'\n[Stage 1] lag1_autocorr = {diag["lag1_autocorr"]:.4f}  '
          f'(injected signal, confirmed present -- should match the '
          f"trace script's own logging at the same edge_strength/seed)")

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    print(f'\n[Stage 2] Enriched events: {enriched_result["n_events_after"]} '
          f'after dropna (full 14-feature table, before restriction)')

    # --- Stage 3: stage the full table, then restrict to ONE feature ---
    staged = stage_live_training_tables(rebuild_result, enriched_result, STAGING_DIR)
    shape = restrict_staged_table_to_one_feature(
        staged['enriched_csv_path'], args.feature,
    )
    print(f"\n[Stage 3] Staged table restricted to ['t1', 'bin', 'w', "
          f"'{args.feature}'] -- shape={shape}")

    # --- Stage 4: run Ch11's real trial grid on the single-feature table ---
    print('\n[Stage 4] Running Ch11 real trial grid (SVC x getSignal, '
          f"purged CV) on '{args.feature}' ALONE...")
    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, STAGING_DIR, LIVE_HERE_DIR)

    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    eval_result = evaluate_overfitting(M, meta, ch11, S=PBO_S, tw=tw_aligned)

    print(f'\n  T_raw={eval_result["T_raw"]}, tw_mean={eval_result["tw_mean"]:.4f}, '
          f'T_effective={eval_result["T"]:.2f}')
    print(f'  sr_hat={eval_result["sr_hat"]:.4f}, dsr={eval_result["dsr"]:.4f}, '
          f'pbo={eval_result["prob_overfit"]:.4f}')

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
    oos_accuracy = None
    prob_corr = None
    if len(prob) >= 3:
        y_aligned = y.loc[prob.index]
        pred_correct = (pred == y_aligned)
        oos_accuracy = float(pred_correct.mean())
        print(f'\n[Stage 4b] Winning trial (C={C}) out-of-sample predictions: '
              f'{len(prob)} predictions')
        print(f'  OOS directional accuracy (pred == true bin): '
              f'{oos_accuracy:.4f}  (0.5 = coin flip on a binary label)')
        if prob.nunique() > 1:
            prob_corr = float(np.corrcoef(
                prob.values.astype(float), y_aligned.values.astype(float)
            )[0, 1])
            print(f'  corr(prob, true_label) = {prob_corr:+.4f}  '
                  f'(0 = no relationship)')
    else:
        print('  (too few OOS predictions to evaluate)')

    print('\n' + '=' * 78)
    print(f"SIDE-BY-SIDE: '{args.feature}' ALONE  vs.  full 14-feature baseline "
          '(same 2026-09-06 edge_strength=2.0/seed=0 run)')
    print('=' * 78)
    print(f'{"metric":<22s}{"single-feature":>18s}{"14-feature baseline":>22s}')
    def _fmt(v):
        return f'{v:.4f}' if v is not None else '   n/a'
    print(f'{"OOS accuracy":<22s}{_fmt(oos_accuracy):>18s}'
          f'{FULL_FEATURE_BASELINE["oos_accuracy"]:>22.4f}')
    print(f'{"corr(prob,label)":<22s}{_fmt(prob_corr):>18s}'
          f'{FULL_FEATURE_BASELINE["prob_label_corr"]:>22.4f}')
    print(f'{"dsr":<22s}{eval_result["dsr"]:>18.4f}'
          f'{FULL_FEATURE_BASELINE["dsr"]:>22.4f}')
    print(f'{"pbo":<22s}{eval_result["prob_overfit"]:>18.4f}'
          f'{FULL_FEATURE_BASELINE["pbo"]:>22.4f}')
    print(f'{"T_effective":<22s}{eval_result["T"]:>18.2f}'
          f'{FULL_FEATURE_BASELINE["T_effective"]:>22.2f}')

    print('\n' + '=' * 78)
    print('INTERPRETATION GUIDE')
    print('=' * 78)
    print(f"""
  - If OOS accuracy/corr(prob,label) move MEANINGFULLY off ~0.50/~0.00
    here with ONLY '{args.feature}' present (e.g. accuracy climbing
    toward 0.55-0.60+, |corr| climbing toward 0.10-0.20+), that points at
    the other 13 features DILUTING '{args.feature}''s real signal inside
    a 14-dimensional RBF kernel -- a fixable feature-selection problem,
    not a fundamental ceiling. Worth then testing a feature-SELECTION
    variant of the full sweep (e.g. keep only the top-3 by |Stage 2 r|)
    rather than concluding the pipeline can't detect this edge at all.
  - If it stays flat (accuracy ~0.50, |corr|~0.00) even with dilution
    fully removed, that is much stronger evidence of a genuine model/
    CV-stage detection floor at this pipeline's real scale
    (T_effective~106) -- independent of feature count, and consistent
    with this project's own established "T>=200 needed for meaningful
    discrimination, even though false-positive control holds at any T"
    framing. That would mean even a single, real, r~0.29-correlated
    feature isn't enough signal-to-noise for an SVC to resolve at
    n~100 effective observations -- a much more fundamental (and much
    more defensible) explanation for ALL of this project's null results,
    including the already-closed BTC/XRP/TAO ones.
""")


if __name__ == '__main__':
    main()
