"""
pipeline/diagnostics/roll_c_overlap_inflation_test.py

IDEA 4 (2026-09-06 session), following ols_roll_c_split_half_test.py's
result: a properly out-of-sample-tested linear fit of next_bar_return on
roll_c showed real, positive, edge_strength-tracking in-sample fits
(ols_beta correctly positive and growing, r2_train up to 0.28 at
edge_strength=8.0) that did NOT survive testing on held-out data
(hit_rate_oos flat at ~0.49 overall, corr(edge_strength, hit_rate_oos) =
-0.35 -- if anything getting WORSE as the injected signal strengthens).
That result explicitly raised (but did not directly test) a specific
hypothesis for why: roll_c[i] is computed from closes[i-roll_window:i+1]
(ROLL_WINDOW=20, features.py) -- roll_c[i] and roll_c[i-1] therefore
share 19 of their 20 underlying closes. Consecutive roll_c values are
NOT independent draws; a whole-sample correlation (or OLS fit) computed
across every bar effectively has far fewer independent observations than
its raw row count suggests, which can produce a correlation/R^2 that
looks real in-sample but reflects shared-data artifact rather than
genuine bar-to-bar predictability -- exactly consistent with what
ols_roll_c_split_half_test.py found.

THIS SCRIPT tests that specific mechanism directly, rather than just
inferring it after the fact: compute the SAME correlation
(roll_c[t-1] vs bar_ret[t]) TWO ways on the identical series --

  1. FULL: using every bar (the same computation trace_meanshift_
     signal_leakage.py's Stage 2 and every other script this session
     have used -- includes heavily overlapping roll_c windows).
  2. NON-OVERLAPPING: using only every ROLL_WINDOW-th bar (so
     consecutive roll_c values used in this second correlation share
     NO underlying closes at all -- genuinely independent draws of the
     same estimator).

If the FULL correlation is systematically much larger than the NON-
OVERLAPPING correlation at the same edge_strength, that is direct,
mechanistic confirmation of the overlap-inflation hypothesis -- not just
an inference from a downstream OOS test failing. If the two stay
comparable, the hypothesis is WRONG and the OOS failure needs a
different explanation (e.g. genuine regime instability between train/
test halves, or something else entirely) -- worth knowing either way
before writing up tonight's conclusion.

No new AFML formula, no model fitting at all here -- this is pure
correlation arithmetic on two different samplings of the same series.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\roll_c_overlap_inflation_test.py --smoke-test
    python pipeline\\diagnostics\\roll_c_overlap_inflation_test.py --edge-strengths 0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0 --seeds 0,1,2,3,4 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
EDGE_HARNESS_DIR = os.path.join(PIPELINE_DIR, 'edge_harness')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, EDGE_HARNESS_DIR)

from rebuild import build_bars_and_labels                       # noqa: E402
from features import build_enriched_events, ROLL_WINDOW          # noqa: E402 -- real constant, 20
from generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades  # noqa: E402

DIAGNOSTICS_DIR = HERE
BASELINE_PARAMS_PATH = os.path.join(DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json')
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'roll_c_overlap_inflation_results.csv')

TARGET_BARS = 1000
EDGE_STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
SEEDS = [0, 1, 2, 3, 4]
DEFAULT_N_TRADES = 120_000  # fallback only; --n-trades-map should cover
                             # every edge_strength above


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


def safe_corr(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def run_one_combo(edge_strength, seed, calib, n_trades, target_bars, roll_window):
    t_start = time.time()

    raw_trades, diag = generate_bar_aligned_meanshift_trades(
        n_trades=n_trades,
        target_bars=target_bars,
        edge_strength=edge_strength,
        seed=seed,
        baseline_imbalance=calib['baseline_imbalance'],
        price_diff_std=calib['price_diff_std'],
        avg_trade_rate_per_sec=calib['avg_trade_rate_per_sec'],
        avg_trade_size=calib['avg_trade_size'],
        start_price=calib['start_price'],
        return_diagnostics=True,
    )
    raw_corr = diag['lag1_autocorr']

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    close = rebuild_result['close']
    feature_table = enriched_result['feature_table']

    roll_c_series = feature_table['roll_c'].reindex(close.index)
    bar_ret = close.pct_change().dropna()
    x_full = roll_c_series.shift(1).reindex(bar_ret.index)
    y_full = bar_ret

    valid = x_full.notna() & y_full.notna()
    x_full = x_full[valid].values
    y_full = y_full[valid].values
    n_full = len(x_full)

    if n_full < roll_window * 3:
        raise ValueError(
            f'Only {n_full} valid (x, y) pairs -- need at least '
            f'{roll_window * 3} to take a meaningful non-overlapping '
            f'subsample at stride={roll_window}.'
        )

    corr_full = safe_corr(x_full, y_full)

    # Non-overlapping subsample: every roll_window-th observation, so
    # consecutive roll_c values used here share NO underlying closes
    # (roll_c[i] uses closes[i-roll_window:i+1] -- stepping by
    # roll_window guarantees the next sampled window starts exactly
    # where the previous one's data ended).
    x_nonoverlap = x_full[::roll_window]
    y_nonoverlap = y_full[::roll_window]
    n_nonoverlap = len(x_nonoverlap)
    corr_nonoverlap = safe_corr(x_nonoverlap, y_nonoverlap)

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'roll_window': roll_window,
        'raw_signal_corr': raw_corr,
        'n_full': n_full,
        'corr_full': corr_full,
        'n_nonoverlap': n_nonoverlap,
        'corr_nonoverlap': corr_nonoverlap,
        'corr_ratio': (
            corr_nonoverlap / corr_full
            if corr_full not in (0, np.nan) and not np.isnan(corr_full) else np.nan
        ),
        'wall_clock_sec': wall_clock_sec,
        'error': '',
    }


def append_result_row(row, results_path):
    df_row = pd.DataFrame([row])
    file_exists = os.path.exists(results_path)
    mode = 'a' if file_exists else 'w'
    header = not file_exists
    df_row.to_csv(results_path, mode=mode, header=header, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--edge-strengths', type=str, default=None)
    parser.add_argument('--seeds', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-trades', type=int, default=None)
    parser.add_argument(
        '--n-trades-map', type=str, default=None,
        help='Path to meanshift_n_trades_map.json -- reused so every '
             'combo here is directly comparable to every other sweep '
             'this session at the same edge_strength.',
    )
    parser.add_argument('--target-bars', type=int, default=None)
    parser.add_argument(
        '--roll-window', type=int, default=None,
        help=f'Stride for the non-overlapping subsample. Default '
             f'{ROLL_WINDOW} (the REAL roll_c window width used '
             f'throughout this project, features.py\'s ROLL_WINDOW -- '
             'not an arbitrary choice).',
    )
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else DEFAULT_N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS
    roll_window = args.roll_window if args.roll_window else ROLL_WINDOW

    n_trades_map = {}
    if args.n_trades_map:
        with open(args.n_trades_map) as f:
            n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}
        print(f'Loaded n_trades map from {args.n_trades_map}:')
        for es, nt in sorted(n_trades_map.items()):
            print(f'  edge_strength={es}: n_trades={nt:,}')
        print()

    calib = load_calibration()
    print(f'Using roll_window={roll_window} for non-overlapping subsample '
          f'(features.py\'s real ROLL_WINDOW={ROLL_WINDOW})\n')

    if args.smoke_test:
        combos = [(0.5, 0)]
        print(f'SMOKE TEST: running 1 combo only (edge_strength=0.5, seed=0), '
              f'n_trades={n_trades_map.get(0.5, default_n_trades)}\n')
    else:
        edge_strengths = (
            [float(x) for x in args.edge_strengths.split(',')]
            if args.edge_strengths else EDGE_STRENGTHS
        )
        seeds = (
            [int(x) for x in args.seeds.split(',')]
            if args.seeds else SEEDS
        )
        combos = [(es, s) for es in edge_strengths for s in seeds]
        print(f'SWEEP: {len(edge_strengths)} edge_strengths x '
              f'{len(seeds)} seeds = {len(combos)} combos '
              f'(corr_full vs. corr_nonoverlap, pure arithmetic -- no '
              f'model fitting, no SVC/PurgedKFold/DSR)\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        n_trades = n_trades_map.get(edge_strength, default_n_trades)
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed}, n_trades={n_trades:,} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, n_trades,
                                 target_bars, roll_window)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(corr_full={row['corr_full']:+.4f} [n={row['n_full']}], "
                  f"corr_nonoverlap={row['corr_nonoverlap']:+.4f} "
                  f"[n={row['n_nonoverlap']}], "
                  f"ratio={row['corr_ratio']:.4f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'roll_window': roll_window, 'raw_signal_corr': np.nan,
                'n_full': np.nan, 'corr_full': np.nan,
                'n_nonoverlap': np.nan, 'corr_nonoverlap': np.nan,
                'corr_ratio': np.nan, 'wall_clock_sec': np.nan,
                'error': f'{type(e).__name__}: {e}',
            }
        results.append(row)
        append_result_row(row, results_path)

    print(f'\nAll combos done. Results written incrementally to {results_path}')
    n_failed = sum(1 for r in results if r['error'])
    if n_failed:
        print(f'WARNING: {n_failed}/{len(results)} combos failed -- see '
              f"the 'error' column in {results_path}")

    if not args.smoke_test:
        df = pd.DataFrame(results).dropna(subset=['corr_full', 'corr_nonoverlap'])
        if len(df) >= 2:
            mean_full = df['corr_full'].mean()
            mean_nonoverlap = df['corr_nonoverlap'].mean()
            mean_ratio = df['corr_ratio'].mean()
            print(f'\nmean corr_full         = {mean_full:+.4f}')
            print(f'mean corr_nonoverlap    = {mean_nonoverlap:+.4f}')
            print(f'mean corr_ratio (nonoverlap/full) = {mean_ratio:.4f}')

    print(f"""
INTERPRETATION GUIDE
---------------------
  - corr_full is the SAME computation trace_meanshift_signal_leakage.py's
    Stage 2 and ols_roll_c_split_half_test.py's in-sample fit both relied
    on -- includes heavily overlapping roll_c windows (each shares
    {roll_window - 1}/{roll_window} of its underlying closes with its
    neighbor).
  - corr_nonoverlap uses only every {roll_window}-th bar -- consecutive
    roll_c values sampled here share ZERO underlying closes, so this is
    the genuinely-independent-observations version of the same
    correlation.
  - If corr_nonoverlap is SUBSTANTIALLY SMALLER than corr_full (say,
    less than half), that is direct, mechanistic confirmation: roll_c's
    overlapping rolling-window construction inflates the whole-sample
    correlation well beyond what independent observations would show --
    consistent with, and now DIRECTLY explaining, ols_roll_c_split_half_
    test.py's finding that a real-looking in-sample fit didn't survive
    out-of-sample testing. This would be the closing piece of tonight's
    investigation: the detection ceiling was never about the classifier,
    the CV scheme, dilution, or signal strength -- it was that the
    feature's own construction produces a correlation statistic that
    doesn't represent genuine bar-to-bar predictability.
  - If corr_nonoverlap stays COMPARABLE to corr_full (not dramatically
    smaller), the overlap-inflation hypothesis is WRONG, and the OOS
    test's failure needs a different explanation -- worth investigating
    directly rather than closing the thread on this hypothesis.
""")


if __name__ == '__main__':
    main()
