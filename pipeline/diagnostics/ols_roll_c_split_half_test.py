"""
pipeline/diagnostics/ols_roll_c_split_half_test.py

IDEA 3 (2026-09-06 session), following bypass_svc_roll_c_strategy.py's
result: a genuinely two-sided, well-powered trivial rule (sign of
roll_c's deviation from its own trailing mean) showed hit_rate flat at
~0.48-0.51 across edge_strength 0.5-8.0 -- no directional edge, even with
the entire SVC/PurgedKFold/DSR chain removed.

OPEN GAP that result left: the deviation-from-trailing-mean rule tests a
DIFFERENT statistic than the one that started this whole investigation.
trace_meanshift_signal_leakage.py's real Stage 2 found roll_c's RAW LEVEL
correlates with next-bar return (r=+0.2894 at es=2.0) -- not roll_c's
deviation from a moving average. Those are related but genuinely
different quantities; a rule built on the wrong one could show no edge
even if the original correlation is real and exploitable in principle.

THIS SCRIPT tests the RAW-LEVEL relationship directly and as simply as
possible: fit an ordinary least-squares line

    next_bar_return ~= alpha + beta * roll_c

on the FIRST HALF of each series (chronological split, matching this
project's own chronological-CV ethos), then apply that exact fitted line
to the SECOND HALF only:

    position[t] = sign(alpha + beta * roll_c[t-1])   (t in the held-out half)

This is deliberately the simplest possible linear model -- no
regularization, no kernel, no cross-validation folds, just two numbers
(alpha, beta) estimated once and tested once on data they never touched.
It is the most direct way to ask "is there ANY linear relationship here
that a model could exploit, or does even a perfectly-fitted straight line
fail" -- without needing an SVC's nonlinear RBF kernel at all.

TWO POSSIBLE OUTCOMES:
  - If the held-out second half shows hit_rate/Sharpe meaningfully above
    chance, that is a real, actionable result: the correlation IS
    exploitable by a properly fitted model, and the specific failure is
    in Ch11's SVC (which sweeps its C parameter but NEVER tunes the RBF
    kernel's gamma, fixed at a single constant) or its kernel choice
    itself -- not the underlying data or the feature.
  - If the held-out half ALSO shows chance-level hit_rate/Sharpe even
    with a correctly-signed, properly-fitted linear relationship, that
    is a much more fundamental finding: it would mean the raw
    correlation the original trace measured does not translate into
    genuine bar-to-bar directional predictability at all -- plausibly
    because roll_c's OWN rolling-window construction (each roll_c[i]
    shares most of its underlying price data with roll_c[i-1], since
    both draw from largely-overlapping windows) inflates whole-sample
    correlation-style statistics without implying the kind of
    bar-to-bar predictability a trading rule needs. That would mean the
    detection ceiling isn't about model class at all -- it's about
    what the correlation itself actually measures.

No new AFML formula: sharpe_ratio is imported directly from Ch11's own
backtest_dangers/pbo.py, real and unmodified. No SVC/PurgedKFold/DSR
machinery is invoked -- OLS fit is plain numpy (np.polyfit), no sklearn.

*** CAVEAT, stated explicitly: this is a single chronological split
(first half fit / second half test), not k-fold cross-validation. That
is a deliberate simplicity choice for a same-day diagnostic, not a claim
that this is as rigorous as Ch11's real PurgedKFold machinery. A result
here should be read as "does a linear relationship survive ONE honest
train/test split," not as a publication-grade backtest. ***

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\ols_roll_c_split_half_test.py --smoke-test
    python pipeline\\diagnostics\\ols_roll_c_split_half_test.py --edge-strengths 0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0 --seeds 0,1,2,3,4 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
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
ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')
EDGE_HARNESS_DIR = os.path.join(PIPELINE_DIR, 'edge_harness')
PBO_MODULE_DIR = os.path.join(ROOT, 'ch11', 'backtest_dangers')

sys.path.insert(0, ORCH_DIR)
sys.path.insert(0, EDGE_HARNESS_DIR)
sys.path.insert(0, PBO_MODULE_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rebuild import build_bars_and_labels                       # noqa: E402
from features import build_enriched_events                       # noqa: E402
from generate_bar_aligned_meanshift_trades import generate_bar_aligned_meanshift_trades  # noqa: E402
from pbo import sharpe_ratio                                     # noqa: E402 -- real, unmodified, ch11/backtest_dangers/pbo.py

DIAGNOSTICS_DIR = HERE
BASELINE_PARAMS_PATH = os.path.join(DIAGNOSTICS_DIR, 'synthetic_trade_baseline_params.json')
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'ols_roll_c_split_half_results.csv')

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


def run_one_combo(edge_strength, seed, calib, n_trades, target_bars):
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

    # x[t] = roll_c[t-1] (only info available up to bar t-1's close),
    # y[t] = bar_ret[t] (realized return of bar t) -- same shift(1)
    # lookahead discipline as every other script this session.
    roll_c_series = feature_table['roll_c'].reindex(close.index)
    bar_ret = close.pct_change().dropna()
    x_full = roll_c_series.shift(1).reindex(bar_ret.index)
    y_full = bar_ret

    valid = x_full.notna() & y_full.notna()
    x_full = x_full[valid].values
    y_full = y_full[valid].values
    n = len(x_full)

    if n < 20:
        raise ValueError(
            f'Only {n} valid (x, y) pairs after alignment/dropna -- need '
            'at least 20 for a meaningful train/test split.'
        )

    # Chronological split -- first half fits, second half is genuinely
    # held out (matches this project's own chronological-CV ethos, not a
    # shuffled/random split).
    split = n // 2
    x_train, y_train = x_full[:split], y_full[:split]
    x_test, y_test = x_full[split:], y_full[split:]

    beta, alpha = np.polyfit(x_train, y_train, 1)

    fitted_train = alpha + beta * x_train
    ss_res = float(np.sum((y_train - fitted_train) ** 2))
    ss_tot = float(np.sum((y_train - y_train.mean()) ** 2))
    r2_train = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    pred_test = alpha + beta * x_test
    pos_test = np.sign(pred_test)
    pnl_test = pos_test * y_test

    sharpe_oos = sharpe_ratio(pnl_test)
    active_test = pos_test != 0
    pct_active_test = float(active_test.mean())
    hit_rate_oos = (
        float((pnl_test[active_test] > 0).mean()) if active_test.sum() >= 3 else np.nan
    )

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'n_raw_trades_used': diag['n_used_trades'],
        'n_scaffold_bars': diag['n_windows'],
        'n_bars': len(rebuild_result['bars']),
        'raw_signal_corr': raw_corr,
        'n_train': split,
        'n_test': n - split,
        'ols_alpha': float(alpha),
        'ols_beta': float(beta),
        'r2_train': r2_train,
        'pct_active_test': pct_active_test,
        'sharpe_oos': sharpe_oos,
        'hit_rate_oos': hit_rate_oos,
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
             'combo here is directly comparable to the SVC-based and '
             'bypass sweeps at the same edge_strength.',
    )
    parser.add_argument('--target-bars', type=int, default=None)
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else DEFAULT_N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS

    n_trades_map = {}
    if args.n_trades_map:
        with open(args.n_trades_map) as f:
            n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}
        print(f'Loaded n_trades map from {args.n_trades_map}:')
        for es, nt in sorted(n_trades_map.items()):
            print(f'  edge_strength={es}: n_trades={nt:,}')
        print()

    calib = load_calibration()

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
              f'(OLS fit on first half, tested on held-out second half)\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        n_trades = n_trades_map.get(edge_strength, default_n_trades)
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed}, n_trades={n_trades:,} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, n_trades, target_bars)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(beta={row['ols_beta']:+.4f}, r2_train={row['r2_train']:.4f}, "
                  f"sharpe_oos={row['sharpe_oos']:.4f}, "
                  f"hit_rate_oos={row['hit_rate_oos']:.4f}, "
                  f"pct_active_test={row['pct_active_test']:.4f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'n_raw_trades_used': np.nan, 'n_scaffold_bars': np.nan,
                'n_bars': np.nan, 'raw_signal_corr': np.nan,
                'n_train': np.nan, 'n_test': np.nan,
                'ols_alpha': np.nan, 'ols_beta': np.nan, 'r2_train': np.nan,
                'pct_active_test': np.nan, 'sharpe_oos': np.nan,
                'hit_rate_oos': np.nan, 'wall_clock_sec': np.nan,
                'error': f'{type(e).__name__}: {e}',
            }
        results.append(row)
        append_result_row(row, results_path)

    print(f'\nAll combos done. Results written incrementally to {results_path}')
    n_failed = sum(1 for r in results if r['error'])
    if n_failed:
        print(f'WARNING: {n_failed}/{len(results)} combos failed -- see '
              f"the 'error' column in {results_path}")

    if not args.smoke_test and len(results) >= 2:
        df = pd.DataFrame(results).dropna(subset=['sharpe_oos', 'edge_strength'])
        if len(df) >= 2 and df['edge_strength'].nunique() >= 2:
            corr_sharpe = df['edge_strength'].corr(df['sharpe_oos'])
            corr_hit = df['edge_strength'].corr(df['hit_rate_oos'])
            corr_beta = df['edge_strength'].corr(df['ols_beta'])
            print(f'\ncorr(edge_strength, sharpe_oos)   = {corr_sharpe:+.4f}')
            print(f'corr(edge_strength, hit_rate_oos)  = {corr_hit:+.4f}')
            print(f'corr(edge_strength, ols_beta)      = {corr_beta:+.4f}  '
                  f'(should be positive and roughly track raw_signal_corr '
                  f'if the fit is picking up the same real relationship)')
            print(f'mean hit_rate_oos across all combos = {df["hit_rate_oos"].mean():.4f}  '
                  f'(0.5 = chance)')

    print("""
INTERPRETATION GUIDE
---------------------
  - ols_beta should come out POSITIVE and grow with edge_strength if the
    fit is correctly picking up the same real, already-confirmed
    relationship trace_meanshift_signal_leakage.py found (roll_c level
    positively correlated with next-bar return). If beta is inconsistent
    in sign or doesn't track edge_strength, the fit itself may be
    unstable -- check r2_train and n_train before trusting hit_rate_oos.
  - THE DECISIVE NUMBER: hit_rate_oos on the HELD-OUT second half.
      * Meaningfully above 0.5 and/or climbing with edge_strength =>
        a linear relationship IS exploitable out-of-sample. This pins
        the original DSR null specifically on Ch11's SVC/kernel choice
        (gamma is fixed, never tuned in the real trial grid) -- a
        concrete, fixable next hypothesis.
      * Flat at ~0.5 regardless of edge_strength, DESPITE ols_beta being
        positive and r2_train being nonzero in-sample => the in-sample
        correlation does NOT survive being tested out-of-sample even in
        its simplest possible linear form. This would point at roll_c's
        own overlapping-rolling-window construction as a likely source
        of an inflated in-sample statistic that isn't real bar-to-bar
        predictability -- a finding about the FEATURE, not the model.
  - r2_train will likely be small in absolute terms (a single feature
    explaining bar-return variance) -- judge it relative to its own
    trend across edge_strength, not against some large absolute bar.
""")


if __name__ == '__main__':
    main()
