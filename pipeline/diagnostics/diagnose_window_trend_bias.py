"""
pipeline/diagnostics/diagnose_window_trend_bias.py

Checks whether the tb1000-5000 CPCV sweep's anomalously low PBO
(0.0087-0.0498, vs the established ~0.78-0.83 baseline from three
Binance.US live runs + the static baseline) could be explained by a
simple trending-regime artifact rather than genuine feature-driven skill.

WHY THIS MATTERS (the mechanism, not just "check for a trend"): a strong
one-directional price move inflates rank-stability across ALL 20 trial
configs uniformly, because every config that leans even slightly long
(or short) rides the same drift. PBO measures how consistently the
in-sample winner stays the out-of-sample winner across CSCV splits --
a dominant trend can make that consistent WITHOUT any real feature-
driven skill, since "long" beats "short" in every split for the same
market-beta reason, not because any feature predicted anything. This is
mechanically distinct from overfitting-immunity; it would look identical
in the PBO/DSR numbers while meaning something completely different.

Two real checks, both against the SAME real bars/trial-grid this
snapshot/target_bars already produced (calibrate_kraken_target_bars.py-
style loading, stages.py's real run_live_trials/evaluate_overfitting --
nothing reimplemented):

  1. Window trend: total return and OLS slope/R^2 of log(close) vs bar
     index. A high R^2 one-directional trend is the signature this
     mechanism predicts.
  2. Winning trial's position bias + a naive always-long/always-short
     baseline Sharpe (Ch11's own real sharpe_ratio(), reused unmodified,
     for direct comparability with the trial grid's own numbers). If a
     naive always-long baseline scores comparably to the actual winning
     trial, that's real evidence the "edge" is market beta, not the
     model.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\diagnose_window_trend_bias.py <snapshot_dir> <target_bars>
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ROOT)
sys.path.insert(0, ORCH)

from rebuild import build_bars_and_labels                # noqa: E402
from features import build_enriched_events                 # noqa: E402
from live_staging import stage_live_training_tables         # noqa: E402
from stages import (                                       # noqa: E402
    load_ch11_driver, run_live_trials, evaluate_overfitting,
)

CH11_BACKTEST_DANGERS = os.path.join(ROOT, 'ch11', 'backtest_dangers')
if CH11_BACKTEST_DANGERS not in sys.path:
    sys.path.insert(0, CH11_BACKTEST_DANGERS)
from pbo import sharpe_ratio                                # noqa: E402  real, unmodified

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                             # noqa: E402


def _trend_stats(close):
    log_close = np.log(close.values)
    x = np.arange(len(log_close))
    slope, intercept = np.polyfit(x, log_close, 1)
    pred = slope * x + intercept
    ss_res = np.sum((log_close - pred) ** 2)
    ss_tot = np.sum((log_close - log_close.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    total_return = close.iloc[-1] / close.iloc[0] - 1
    return {
        'start_price': float(close.iloc[0]), 'end_price': float(close.iloc[-1]),
        'total_return': float(total_return), 'log_close_ols_slope': float(slope),
        'log_close_ols_r2': float(r2),
    }


def _directional_baselines(bar_ret):
    """Naive always-long / always-short Sharpe on THIS window's real bar
    returns, via Ch11's own real sharpe_ratio() -- directly comparable to
    the trial grid's own sharpe_full_sample numbers."""
    always_long = sharpe_ratio(bar_ret.values)
    always_short = sharpe_ratio((-bar_ret).values)
    return {'always_long_sharpe': float(always_long) if np.isfinite(always_long) else None,
            'always_short_sharpe': float(always_short) if np.isfinite(always_short) else None}


def _winning_trial_position_bias(ch11, X, y, w, t1, close, winning_C, winning_step):
    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, winning_C)  # real, unmodified
    events = pd.DataFrame({'t1': t1}).loc[prob.index]
    sig = getSignal(events, winning_step, prob, pred, numClasses=2, numThreads=1)
    pos = sig.reindex(close.index, method='ffill').fillna(0.0)
    in_market = pos[pos != 0]
    if len(in_market) == 0:
        return {'pct_long': None, 'pct_short': None, 'n_in_market': 0}
    pct_long = float((in_market > 0).mean())
    pct_short = float((in_market < 0).mean())
    return {'pct_long': pct_long, 'pct_short': pct_short, 'n_in_market': int(len(in_market))}


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            'Usage: python diagnose_window_trend_bias.py <snapshot_dir> <target_bars>'
        )
    snapshot_dir = sys.argv[1]
    target_bars = int(sys.argv[2])
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')
    print(f'target_bars={target_bars}')

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    close = rebuild_result['close']
    bar_ret = close.pct_change().dropna()
    print(f"  {len(rebuild_result['bars'])} bars, {len(rebuild_result['events'])} events")

    trend = _trend_stats(close)
    print('\n--- Window trend (log-close OLS vs bar index) ---')
    for k, v in trend.items():
        print(f'  {k}: {v:.6f}' if isinstance(v, float) else f'  {k}: {v}')

    baselines = _directional_baselines(bar_ret)
    print('\n--- Naive directional baselines (Ch11 real sharpe_ratio, per-bar, unannualized) ---')
    for k, v in baselines.items():
        print(f'  {k}: {v}')

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    work_root = os.path.join(HERE, 'trend_bias_work')
    staging_dir = os.path.join(work_root, 'staging')
    here_dir = os.path.join(work_root, 'ch11_here')
    stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)
    tw_aligned = rebuild_result['tw'].reindex(enriched_result['enriched_events'].index)
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')
    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)
    winning_C = float(meta.loc[eval_result['best_trial'], 'C'])
    winning_step = float(meta.loc[eval_result['best_trial'], 'stepSize'])
    print(f"\n--- Winning trial: C={winning_C}, step={winning_step}, "
          f"Sharpe={eval_result['sr_hat']:+.4f}, PBO={eval_result['prob_overfit']:.4f} ---")

    enriched = enriched_result['enriched_events']
    feature_cols = list(enriched_result['feature_table'].columns)
    X = enriched[feature_cols]
    y = enriched['bin']
    t1 = enriched['t1']
    w = rebuild_result['w'].reindex(enriched.index)

    bias = _winning_trial_position_bias(ch11, X, y, w, t1, close, winning_C, winning_step)
    print('\n--- Winning trial position direction bias ---')
    for k, v in bias.items():
        print(f'  {k}: {v}')

    print('\n--- Read this together ---')
    print(f"  Winning trial Sharpe:      {eval_result['sr_hat']:+.4f}")
    print(f"  Always-long baseline:      {baselines['always_long_sharpe']}")
    print(f"  Always-short baseline:     {baselines['always_short_sharpe']}")
    print(f"  log-close trend R^2:       {trend['log_close_ols_r2']:.4f}")
    print(f"  Winning trial % long:      {bias['pct_long']}")
    print('  If the winning trial\'s Sharpe is close to whichever naive '
          'baseline matches its own long/short bias, AND log-close R^2 is '
          'high, that is real evidence for the trending-regime explanation '
          'over a genuine feature-driven edge.')


if __name__ == '__main__':
    main()
