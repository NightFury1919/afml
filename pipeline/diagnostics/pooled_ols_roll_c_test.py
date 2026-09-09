"""
pipeline/diagnostics/pooled_ols_roll_c_test.py

IDEA 5 (2026-09-06 session), following roll_c_overlap_inflation_test.py's
30-seed result: pooling corr_full across 200 combos revealed a real,
statistically significant relationship between roll_c's correlation with
next-bar return and edge_strength (r=+0.320, p<0.0001) -- but any SINGLE
combo's corr_full estimate is far too noisy (std~0.09) to be individually
reliable, since n~975 bars per combo isn't enough to precisely estimate
an effect this small. That directly explains why ols_roll_c_split_half_
test.py's single-window fit (train on one series' first half, test on
its second half) found nothing: one window's worth of data was never
enough to pin down a real-but-small relationship precisely.

THIS SCRIPT tests the direct, practical consequence of that finding: if
the effect is real but needs pooled data to estimate precisely, does a
model FIT ON POOLED DATA FROM MANY INDEPENDENT WINDOWS (different random
seeds -- genuinely independent draws, not just different halves of one
series) show a real, detectable edge when tested on ENTIRELY NEW,
never-seen windows?

METHOD, at a fixed edge_strength:
  1. TRAIN: generate N_TRAIN_SEEDS independent series (different seeds),
     compute (roll_c[t-1], bar_ret[t]) pairs for each, and POOL every
     pair from every training seed into one large combined dataset.
     Fit a single OLS line (alpha, beta) on this pooled dataset.
  2. TEST: generate N_TEST_SEEDS entirely NEW, never-seen-in-training
     seeds. Apply the pooled-fit line to each test seed's own series
     (position[t] = sign(alpha + beta*roll_c[t-1])), and pool the
     resulting (position, realized_return) pairs from EVERY test seed
     into one large combined evaluation set.
  3. Report hit_rate and Sharpe on the POOLED test set (not per-seed
     averages -- pooling the actual bets, matching the same "many
     independent draws" logic that revealed the real signal in
     roll_c_overlap_inflation_test.py), plus a binomial test of whether
     the pooled hit_rate is significantly different from 0.5.

Train and test seeds are drawn from disjoint ranges (TRAIN: 0-N_TRAIN-1,
TEST: 1000+) to guarantee no accidental overlap with any seed used
elsewhere this session.

TWO POSSIBLE OUTCOMES:
  - If the pooled-test hit_rate comes out significantly above 0.5 (and
    especially if it climbs with edge_strength), that is the concrete,
    actionable confirmation of tonight's finding: the effect IS real and
    IS exploitable, but only once estimated from enough pooled
    independent data -- a single backtest or a single train/test split
    will never see it, but a properly pooled estimation approach could.
    This has a direct, concrete implication for the live BTC/XRP/TAO
    pipeline: a real edge could exist yet be undetectable by any single
    backtest at this pipeline's typical scale.
  - If the pooled-test hit_rate STILL stays at ~0.5 even with a properly
    pooled fit and pooled evaluation, that would mean the corr_full-vs-
    edge_strength relationship, despite being statistically significant
    in aggregate, still doesn't translate into genuine directional
    predictability even with maximal pooling -- pointing to a deeper
    issue (e.g. the relationship might be significant but driven by
    higher moments / nonlinear structure an OLS line can't capture, or
    might not be causally sensible despite being statistically real).

No new AFML formula: sharpe_ratio is imported directly from Ch11's own
backtest_dangers/pbo.py, real and unmodified. Plain numpy OLS
(np.polyfit), no sklearn, no SVC/PurgedKFold/DSR.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\pooled_ols_roll_c_test.py --smoke-test
    python pipeline\\diagnostics\\pooled_ols_roll_c_test.py --edge-strengths 0.5,1.0,2.0,3.0,5.0,8.0 --n-train-seeds 20 --n-test-seeds 20 --n-trades-map pipeline\\diagnostics\\meanshift_n_trades_map.json
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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
RESULTS_CSV_PATH = os.path.join(DIAGNOSTICS_DIR, 'pooled_ols_roll_c_results.csv')

TARGET_BARS = 1000
EDGE_STRENGTHS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
N_TRAIN_SEEDS = 20
N_TEST_SEEDS = 20
TRAIN_SEED_START = 0
TEST_SEED_START = 1000  # disjoint from every seed used elsewhere this
                          # session (0-29), guarantees no accidental reuse
DEFAULT_N_TRADES = 120_000


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


def get_xy_pairs(edge_strength, seed, calib, n_trades, target_bars):
    """Generate one series and return its (roll_c[t-1], bar_ret[t]) pairs,
    same alignment as every other script this session. Returns (x, y) or
    raises on a real pipeline failure (caller decides whether to skip)."""
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
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    close = rebuild_result['close']
    feature_table = enriched_result['feature_table']
    roll_c_series = feature_table['roll_c'].reindex(close.index)
    bar_ret = close.pct_change().dropna()
    x = roll_c_series.shift(1).reindex(bar_ret.index)
    y = bar_ret
    valid = x.notna() & y.notna()
    return x[valid].values, y[valid].values


def run_one_edge_strength(edge_strength, calib, n_trades, target_bars,
                           n_train_seeds, n_test_seeds):
    t_start = time.time()

    # --- TRAIN: pool (x, y) pairs from many independent seeds ---
    x_train_pool, y_train_pool = [], []
    n_train_ok, n_train_failed = 0, 0
    for seed in range(TRAIN_SEED_START, TRAIN_SEED_START + n_train_seeds):
        try:
            x, y = get_xy_pairs(edge_strength, seed, calib, n_trades, target_bars)
            x_train_pool.append(x)
            y_train_pool.append(y)
            n_train_ok += 1
        except Exception:
            n_train_failed += 1
            continue
    if n_train_ok < 3:
        raise ValueError(
            f'Only {n_train_ok} training seeds succeeded -- need at '
            'least 3 to pool a meaningful fit.'
        )
    x_train_pool = np.concatenate(x_train_pool)
    y_train_pool = np.concatenate(y_train_pool)

    beta, alpha = np.polyfit(x_train_pool, y_train_pool, 1)
    fitted = alpha + beta * x_train_pool
    ss_res = float(np.sum((y_train_pool - fitted) ** 2))
    ss_tot = float(np.sum((y_train_pool - y_train_pool.mean()) ** 2))
    r2_train_pooled = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # --- TEST: apply the pooled fit to entirely new, never-seen seeds ---
    all_pnl = []
    n_test_ok, n_test_failed = 0, 0
    for seed in range(TEST_SEED_START, TEST_SEED_START + n_test_seeds):
        try:
            x, y = get_xy_pairs(edge_strength, seed, calib, n_trades, target_bars)
            pred = alpha + beta * x
            pos = np.sign(pred)
            pnl = pos * y
            all_pnl.append(pnl)
            n_test_ok += 1
        except Exception:
            n_test_failed += 1
            continue
    if n_test_ok < 3:
        raise ValueError(
            f'Only {n_test_ok} test seeds succeeded -- need at least 3 '
            'for a meaningful pooled evaluation.'
        )
    all_pnl = np.concatenate(all_pnl)

    n_active = int((all_pnl != 0).sum())
    n_wins = int((all_pnl > 0).sum())
    pooled_hit_rate = n_wins / n_active if n_active >= 3 else np.nan
    pooled_sharpe = sharpe_ratio(all_pnl)
    binom_p = (
        binomtest(n_wins, n_active, p=0.5).pvalue if n_active >= 3 else np.nan
    )
    # One-sample t-test on the pooled PnL mean vs. 0 -- THE metric that
    # actually matters for a small persistent edge. See module note:
    # synthetic validation showed hit_rate/Sharpe can stay barely above
    # chance even with a REAL, perfectly-estimated small effect --
    # pooling training data improves how PRECISELY beta is estimated, not
    # the per-bet signal-to-noise ratio. A small but statistically
    # significant positive mean PnL (t-test), even at hit_rate~0.50-0.52,
    # is the realistic signature of a genuine-but-small tradeable edge.
    active_pnl = all_pnl[all_pnl != 0]
    if len(active_pnl) >= 3 and active_pnl.std(ddof=1) > 0:
        t_stat, t_p_value = float(
            active_pnl.mean() / (active_pnl.std(ddof=1) / np.sqrt(len(active_pnl)))
        ), None
        from scipy.stats import t as t_dist
        t_p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(active_pnl) - 1)))
    else:
        t_stat, t_p_value = np.nan, np.nan

    wall_clock_sec = time.time() - t_start

    return {
        'edge_strength': edge_strength,
        'n_train_seeds_requested': n_train_seeds,
        'n_train_seeds_ok': n_train_ok,
        'n_train_seeds_failed': n_train_failed,
        'n_train_pairs_pooled': len(x_train_pool),
        'ols_alpha': float(alpha),
        'ols_beta': float(beta),
        'r2_train_pooled': r2_train_pooled,
        'n_test_seeds_requested': n_test_seeds,
        'n_test_seeds_ok': n_test_ok,
        'n_test_seeds_failed': n_test_failed,
        'n_test_bars_pooled': len(all_pnl),
        'n_test_active_bars': n_active,
        'pooled_hit_rate': pooled_hit_rate,
        'pooled_sharpe': pooled_sharpe,
        'binom_p_value': binom_p,
        'mean_pnl_t_stat': t_stat,
        'mean_pnl_t_p_value': t_p_value,
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
    parser.add_argument('--n-train-seeds', type=int, default=N_TRAIN_SEEDS)
    parser.add_argument('--n-test-seeds', type=int, default=N_TEST_SEEDS)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--n-trades', type=int, default=None)
    parser.add_argument('--n-trades-map', type=str, default=None)
    parser.add_argument('--target-bars', type=int, default=None)
    args = parser.parse_args()

    results_path = args.output if args.output else RESULTS_CSV_PATH
    default_n_trades = args.n_trades if args.n_trades else DEFAULT_N_TRADES
    target_bars = args.target_bars if args.target_bars else TARGET_BARS

    n_trades_map = {}
    if args.n_trades_map:
        with open(args.n_trades_map) as f:
            n_trades_map = {float(k): int(v) for k, v in json.load(f).items()}

    calib = load_calibration()

    if args.smoke_test:
        edge_strengths = [0.5]
        n_train_seeds, n_test_seeds = 5, 5
        print(f'SMOKE TEST: edge_strength=0.5, {n_train_seeds} train seeds, '
              f'{n_test_seeds} test seeds\n')
    else:
        edge_strengths = (
            [float(x) for x in args.edge_strengths.split(',')]
            if args.edge_strengths else EDGE_STRENGTHS
        )
        n_train_seeds = args.n_train_seeds
        n_test_seeds = args.n_test_seeds
        print(f'POOLED OLS TEST: {len(edge_strengths)} edge_strengths, '
              f'{n_train_seeds} pooled TRAIN seeds (seeds '
              f'{TRAIN_SEED_START}-{TRAIN_SEED_START+n_train_seeds-1}), '
              f'{n_test_seeds} pooled TEST seeds (seeds '
              f'{TEST_SEED_START}-{TEST_SEED_START+n_test_seeds-1}, '
              f'disjoint from training and from every seed used elsewhere '
              f'this session)\n')
        print(f'Results -> {results_path} '
              f'({"appending to existing file" if os.path.exists(results_path) else "new file"})\n')

    results = []
    for i, edge_strength in enumerate(edge_strengths):
        n_trades = n_trades_map.get(edge_strength, default_n_trades)
        print(f'[{i+1}/{len(edge_strengths)}] edge_strength={edge_strength}, '
              f'n_trades={n_trades:,} ... ', flush=True)
        try:
            row = run_one_edge_strength(
                edge_strength, calib, n_trades, target_bars,
                n_train_seeds, n_test_seeds,
            )
            print(f"  done in {row['wall_clock_sec']:.1f}s "
                  f"(train: {row['n_train_seeds_ok']}/{row['n_train_seeds_requested']} ok, "
                  f"{row['n_train_pairs_pooled']} pooled pairs, "
                  f"beta={row['ols_beta']:+.4f}, r2={row['r2_train_pooled']:.5f} | "
                  f"test: {row['n_test_seeds_ok']}/{row['n_test_seeds_requested']} ok, "
                  f"{row['n_test_active_bars']} active bars, "
                  f"hit_rate={row['pooled_hit_rate']:.4f}, "
                  f"sharpe={row['pooled_sharpe']:.4f}, "
                  f"MEAN_PNL_T_P={row['mean_pnl_t_p_value']:.4f})")
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            traceback.print_exc()
            row = {
                'edge_strength': edge_strength,
                'n_train_seeds_requested': n_train_seeds, 'n_train_seeds_ok': np.nan,
                'n_train_seeds_failed': np.nan, 'n_train_pairs_pooled': np.nan,
                'ols_alpha': np.nan, 'ols_beta': np.nan, 'r2_train_pooled': np.nan,
                'n_test_seeds_requested': n_test_seeds, 'n_test_seeds_ok': np.nan,
                'n_test_seeds_failed': np.nan, 'n_test_bars_pooled': np.nan,
                'n_test_active_bars': np.nan, 'pooled_hit_rate': np.nan,
                'pooled_sharpe': np.nan, 'binom_p_value': np.nan,
                'mean_pnl_t_stat': np.nan, 'mean_pnl_t_p_value': np.nan,
                'wall_clock_sec': np.nan, 'error': f'{type(e).__name__}: {e}',
            }
        results.append(row)
        append_result_row(row, results_path)

    print(f'\nAll edge_strengths done. Results written to {results_path}')

    if not args.smoke_test:
        df = pd.DataFrame(results).dropna(subset=['pooled_hit_rate'])
        if len(df) >= 2:
            corr = df['edge_strength'].corr(df['pooled_hit_rate'])
            print(f'\ncorr(edge_strength, pooled_hit_rate) = {corr:+.4f}  '
                  f'(secondary -- see IMPORTANT CORRECTION below)')
            print(f'mean pooled_hit_rate across all edge_strengths = '
                  f'{df["pooled_hit_rate"].mean():.4f} (0.5 = chance)')
            n_sig_t = int((df['mean_pnl_t_p_value'] < 0.05).sum())
            print(f'{n_sig_t}/{len(df)} edge_strengths show a pooled mean '
                  f'PnL significantly different from 0 (t-test p<0.05) '
                  f'-- THIS is the number that matters, see below')

    print(f"""
IMPORTANT CORRECTION FROM SYNTHETIC VALIDATION (see module docstring):
before trusting this test's output, note that a controlled synthetic
check (a REAL, known, small true beta, fit on 20 pooled training
windows, tested on 20 new pooled windows) showed pooled_hit_rate and
pooled_sharpe BOTH staying barely above chance/zero (0.506, 0.003) even
though the fit recovered the true beta almost exactly. Pooling training
data improves how PRECISELY beta is estimated -- it does NOT increase
the per-bet signal-to-noise ratio, which is what actually bounds
hit_rate and Sharpe. So do NOT expect a dramatic hit_rate jump as proof
of a real effect; the roll_c_overlap_inflation_test.py correlation
result (r=+0.32, p<0.0001) already proved statistical detectability.
THE REAL QUESTION HERE is narrower and more honest: mean_pnl_t_p_value.

INTERPRETATION GUIDE
---------------------
  - THE DECISIVE NUMBER: mean_pnl_t_p_value (one-sample t-test of the
    pooled per-bet PnL mean against 0), not hit_rate or the binomial
    test. A small, even TINY, positive mean PnL that is statistically
    significant (p<0.05) across thousands of pooled bets is the
    realistic signature of a genuine, tradeable-in-aggregate edge -- this
    is exactly how many real, small, persistent market edges look: hit
    rate barely above 50%, but a consistently positive expected value
    that compounds over enough bets. If this comes out significant AND
    positive, especially growing with edge_strength, that is real,
    actionable confirmation.
  - If mean_pnl_t_p_value stays non-significant (or the sign flips
    inconsistently across edge_strength) even with maximal pooling on
    both the fitting and evaluation sides, that means the aggregate
    statistical relationship found in roll_c_overlap_inflation_test.py,
    despite being real, does not translate into ANY exploitable
    directional edge -- pointing to something more fundamental (possibly
    nonlinear structure, or a relationship that's real in aggregate
    correlation terms but not causally exploitable bet-by-bet).
  - hit_rate and pooled_sharpe are still reported for completeness and
    cross-reference, but per the synthetic validation above, do not
    expect either to move dramatically even in a genuine-signal scenario.
""")


if __name__ == '__main__':
    main()
