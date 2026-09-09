"""
pipeline/diagnostics/bypass_svc_roll_c_strategy.py

IDEA 2 (2026-09-06 session), following the saturation sweep's decisive
result: run_meanshift_edge_sweep.py's full-14-feature sweep, extended to
edge_strength up to 8.0 (raw_signal_corr climbing cleanly from 0.155 to
~0.50, r=+0.72 vs edge_strength, p<0.0001 -- the injected signal is
unambiguously real and strong), found corr(edge_strength, dsr) =
+0.0000018 (p=0.9999) -- literally zero. This ruled out "the signal just
isn't strong enough yet" as an explanation: DSR does not respond to
ANY signal strength tested, including one more than 3x stronger than the
original null.

That leaves the classifier/CV/getSignal/DSR chain itself as the prime
suspect. This script tests that directly by REMOVING it entirely: instead
of Ch11's real SVC(gamma=fixed) x PurgedKFold x 20-trial grid, build the
simplest possible genuinely-two-sided rule --

    position[t] = sign(roll_c[t-1] - rolling_mean(roll_c)[t-1])
                  (no fitting, no CV, no parameters beyond a fixed
                  trailing normalization window)

*** BUG FOUND AND FIXED (2026-09-06, same session): the FIRST version of
this script used position[t] = sign(roll_c[t-1]) directly. This is
broken by construction: roll_c = sqrt(max(0, -cov_dp)) (see Ch19's
roll_measure()) is NEVER negative -- it is a spread MAGNITUDE, not a
directional quantity. sign(roll_c) is therefore ~always +1 when the
sqrt's argument is positive and exactly 0 (inactive) whenever cov_dp>=0.
Worse, this project's own injected meanshift signal IS positive serial
correlation in price changes (window i's return carries into window
i+1's drift) -- which is EXACTLY what clamps roll_c's argument to zero.
The result: pct_bars_active collapsed from ~30% at edge_strength=0.5 to
under 1% at edge_strength=5.0/8.0 in the first real run of this script
(bypass_svc_roll_c_sweep_results.csv, first version) -- the rule was
being mechanically starved of activity by the very signal it was meant
to detect, producing a flat corr(edge_strength, sharpe)=+0.0227 that
reflected a broken test, not a real finding about the classifier chain.

FIX: trade the sign of roll_c's deviation from its OWN trailing rolling
mean (window=NORM_WINDOW bars, causal -- pandas .rolling() only looks
backward) instead of roll_c's raw value. This is genuinely two-sided
(deviation can be positive or negative regardless of roll_c's own
non-negativity) and does not degenerate as edge_strength grows, since
"is roll_c higher or lower than its own recent average right now" stays
a well-posed, roughly-50%-active question at any signal strength.

roll_c itself is computed at bar i from closes[i-roll_window:i+1] (see
features.py's compute_ch19_features) -- i.e. using ONLY information
available up to bar i's own close. The rolling mean subtracted from it
is likewise strictly trailing (pandas .rolling(NORM_WINDOW).mean(), no
centering). Shifting the resulting sign by one further bar before
multiplying by that bar's return (the SAME shift(1) discipline Ch11's
part_c_build_trials() uses for its own mark-to-market PnL) keeps the
whole chain genuinely lookahead-free, not an approximation.

This is not a strategy anyone would trade -- it has no risk management,
no position sizing, no thresholding beyond a bare sign() of a deviation.
That is the point: it removes every piece of machinery between "the
feature carries real correlation with next-bar return" (already
confirmed, trace_meanshift_signal_leakage.py, r=+0.2894 at es=2.0) and
"does a Sharpe ratio respond to edge_strength" -- no SVC kernel/gamma
choice, no PurgedKFold fold structure, no getSignal position-sizing/
averaging, no DSR trial-selection correction. If Sharpe STILL doesn't
respond to edge_strength here, the problem is not the classifier, the CV
scheme, or DSR's overfitting correction -- it would have to be something
even more basic: the PnL/mark-to-market computation itself, the bar-
return series, or how the injected drift actually manifests in realized
bar-to-bar returns. If Sharpe DOES respond here where DSR didn't, that
pins the problem specifically on the SVC/CV/DSR machinery (e.g. the
fixed GAMMA never being tuned in Ch11's real trial grid -- only C is
swept -- is one concrete, testable follow-on hypothesis that would fall
out of this result).

No new AFML formula: sharpe_ratio is imported directly from Ch11's own
backtest_dangers/pbo.py, real and unmodified. No SVC/PurgedKFold/DSR
machinery is invoked at all -- this deliberately bypasses stages.py's
run_live_trials()/evaluate_overfitting(), not because they're wrong, but
because isolating them out is the whole point of this test.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\bypass_svc_roll_c_strategy.py --smoke-test
    python pipeline\\diagnostics\\bypass_svc_roll_c_strategy.py --edge-strengths 0.5,0.75,1.0,1.5,2.0,3.0,5.0,8.0 --seeds 0,1,2,3,4 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
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
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'bypass_svc_roll_c_sweep_results.csv')

TARGET_BARS = 1000
EDGE_STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
SEEDS = [0, 1, 2, 3, 4]
DEFAULT_N_TRADES = 120_000  # fallback only; --n-trades-map should cover
                             # every edge_strength above
NORM_WINDOW = 20  # trailing bars used to compute roll_c's own rolling
                   # mean for the deviation signal -- see module BUG
                   # FOUND AND FIXED note. Matches Ch19's other 20-bar
                   # rolling estimators (parkinson_vol_20bar,
                   # amihud_lambda_20bar) for consistency, not tuned.


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


def run_one_combo(edge_strength, seed, calib, n_trades, target_bars, norm_window):
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

    # --- THE ENTIRE "MODEL": sign(roll_c - trailing_mean(roll_c)) ---
    # roll_c[i] uses only closes up to and including bar i (see module
    # docstring). roll_c is NON-NEGATIVE by construction (sqrt clamp in
    # Ch19's roll_measure()), so its RAW sign is degenerate -- trade the
    # sign of its deviation from a strictly trailing rolling mean instead
    # (see module BUG FOUND AND FIXED note), which is genuinely two-sided
    # and does not collapse in activity as edge_strength grows.
    roll_c_series = feature_table['roll_c'].reindex(close.index)
    roll_c_trailing_mean = roll_c_series.rolling(
        window=norm_window, min_periods=norm_window,
    ).mean()
    deviation = roll_c_series - roll_c_trailing_mean
    sig = np.sign(deviation).fillna(0.0)

    bar_ret = close.pct_change().dropna()
    # shift(1) is what makes trading on it lookahead-free, same
    # discipline as Ch11's own pos.shift(1) before multiplying by bar_ret.
    pos = sig.shift(1).reindex(bar_ret.index).fillna(0.0)
    pnl = (pos * bar_ret).values

    sharpe = sharpe_ratio(pnl)
    pct_bars_active = float((pos != 0).mean())
    active_mask = pos.values != 0
    hit_rate = (
        float((pnl[active_mask] > 0).mean()) if active_mask.sum() >= 3 else np.nan
    )

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'seed': seed,
        'norm_window': norm_window,
        'n_raw_trades_used': diag['n_used_trades'],
        'n_scaffold_bars': diag['n_windows'],
        'n_bars': len(rebuild_result['bars']),
        'raw_signal_corr': raw_corr,
        'n_bar_returns': len(bar_ret),
        'pct_bars_active': pct_bars_active,
        'sharpe_trivial_strategy': sharpe,
        'hit_rate': hit_rate,
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
        help='Path to meanshift_n_trades_map.json. This script does not '
             'depend on T_effective (no DSR/PBO computed here at all -- '
             'just a raw Sharpe on a fixed rule), but reusing the SAME '
             'calibrated n_trades keeps every combo directly comparable '
             'to the SVC-based saturation sweep at the same edge_strength.',
    )
    parser.add_argument('--target-bars', type=int, default=None)
    parser.add_argument(
        '--norm-window', type=int, default=None,
        help=f'Trailing window (bars) for roll_c\'s own rolling mean, '
             f'used to build the deviation signal. Default {NORM_WINDOW} '
             '(matches Ch19\'s other 20-bar rolling estimators).',
    )
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else DEFAULT_N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS
    norm_window = args.norm_window if args.norm_window else NORM_WINDOW

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
              f'(NO SVC/PurgedKFold/DSR -- sign(roll_c - trailing_mean), '
              f'norm_window={norm_window})\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, (edge_strength, seed) in enumerate(combos):
        n_trades = n_trades_map.get(edge_strength, default_n_trades)
        print(f'[{i+1}/{len(combos)}] edge_strength={edge_strength}, '
              f'seed={seed}, n_trades={n_trades:,} ... ', end='', flush=True)
        try:
            row = run_one_combo(edge_strength, seed, calib, n_trades,
                                 target_bars, norm_window)
            print(f"done in {row['wall_clock_sec']:.1f}s "
                  f"(sharpe={row['sharpe_trivial_strategy']:.4f}, "
                  f"hit_rate={row['hit_rate']:.4f}, "
                  f"pct_active={row['pct_bars_active']:.4f}, "
                  f"raw_corr={row['raw_signal_corr']:.4f})")
        except Exception as e:
            print(f'FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength, 'seed': seed,
                'norm_window': norm_window,
                'n_raw_trades_used': np.nan, 'n_scaffold_bars': np.nan,
                'n_bars': np.nan, 'raw_signal_corr': np.nan,
                'n_bar_returns': np.nan, 'pct_bars_active': np.nan,
                'sharpe_trivial_strategy': np.nan, 'hit_rate': np.nan,
                'wall_clock_sec': np.nan,
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
        df = pd.DataFrame(results).dropna(subset=['sharpe_trivial_strategy', 'edge_strength'])
        if len(df) >= 2 and df['edge_strength'].nunique() >= 2:
            corr_sharpe = df['edge_strength'].corr(df['sharpe_trivial_strategy'])
            corr_hit = df['edge_strength'].corr(df['hit_rate'])
            print(f'\ncorr(edge_strength, sharpe_trivial_strategy) = {corr_sharpe:+.4f}')
            print(f'corr(edge_strength, hit_rate)                 = {corr_hit:+.4f}')
            print('(compare against the SVC-based saturation sweep\'s '
                  'corr(edge_strength, dsr) = +0.0000 -- '
                  'meanshift_saturation_sweep_results.csv)')

    print("""
INTERPRETATION GUIDE
---------------------
  - If sharpe_trivial_strategy/hit_rate respond CLEARLY to edge_strength
    here (sharpe rising, hit_rate climbing above 0.5 as edge_strength
    grows) where the full SVC-based saturation sweep showed DSR flat at
    corr=+0.0000, that PINS THE PROBLEM ON the classifier/CV/DSR chain
    specifically -- not the data, not the feature, not the PnL mechanics.
    A concrete next hypothesis: Ch11's real trial grid sweeps C but NEVER
    tunes the SVC's RBF kernel gamma (fixed constant) -- worth testing
    whether a gamma sweep, or a simpler linear-kernel SVC, changes this.
  - If sharpe_trivial_strategy ALSO stays flat/near-zero across
    edge_strength even with EVERY piece of modeling machinery removed,
    that would point to something even more basic than the classifier:
    the mark-to-market PnL computation, the bar-return series itself, or
    how the injected drift actually manifests in realized returns after
    passing through bar construction -- worth checking those directly
    before returning to the classifier stage at all.
  - hit_rate is a more interpretable secondary check than Sharpe alone:
    Sharpe can be distorted by a few large outlier bars, while hit_rate
    (fraction of ACTIVE bars where the trivial rule's direction matched
    the bar's realized return) answers the more basic "is the SIGN
    useful at all" question directly. 0.5 = coin flip; systematically
    above 0.5 and climbing with edge_strength would be the clean,
    intuitive confirmation this bypass test is looking for.
""")


if __name__ == '__main__':
    main()
