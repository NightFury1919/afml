"""
pipeline/diagnostics/calibrate_pbo_detrended.py

diagnose_window_trend_bias.py (2026-09-10) found that on the tb1000
Kraken BTC window, the winning trial's Sharpe (+0.0122) was less than a
quarter of a naive always-long baseline's Sharpe (+0.0578), despite the
winning trial being 95.9% long itself -- with a strongly trending window
underneath (log-close R^2=0.63). Mechanism: PBO measures whether the
in-sample winner stays the out-of-sample winner across CSCV splits: when
all 20 trial configs share the same directional (mostly-long) exposure
to one dominant real trend, their rank order stays artificially stable
for reasons having nothing to do with any feature predicting anything.
That is what produced this window's anomalously low PBO (0.0087-0.0498
across target_bars, vs. the established ~0.78-0.83 baseline from three
Binance.US live runs + the static baseline) -- not evidence of the
pipeline finding a real edge.

This script does NOT change ch11's real trial construction or its
book-faithful PnL formula (pos.shift(1) * bar_ret is exactly Ch11's own
construction) -- it runs it TWICE, side by side, once on the real raw
bar returns (reproducing exactly what run_live_trials already reports,
as a consistency check) and once on TREND-ADJUSTED excess returns
(bar_ret - bar_ret.mean()), which removes the shared market-beta
component before PBO/DSR ever see it. A naive always-long/always-short
strategy nets to ~0 Sharpe on excess returns by construction (its PnL
literally IS the trend, which is now zero-mean) -- so if the DETRENDED
winning trial's Sharpe is still comparably small vs. its own naive
baseline, this window still shows no real timing/selection skill even
with the trend artifact removed. If instead a real gap opens up between
the winning trial and its naive baseline only after detrending, that
would be new evidence the model IS doing something beyond riding beta,
just masked by the trend in the raw numbers.

*** SCOPE NOTE: bar-level Part C only, not the CPCV sweep ***
This detrends the Part C 20-trial PBO/DSR grid (bar_ret-based). The
separate CPCV feasibility sweep (calibrate_cpcv_groups.py) is scored on
event-level holding-period 'ret', a different granularity -- it now has
its own '--detrend' flag (same excess-return principle, applied to that
series instead) rather than being folded into this script.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_pbo_detrended.py <snapshot_dir> <target_bars>
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
from stages import evaluate_overfitting, load_ch11_driver   # noqa: E402

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                             # noqa: E402


def _build_trial_matrices(ch11, X, y, w, t1, close):
    """Mirrors ch11.part_c_build_trials()'s real construction exactly
    (out_of_sample_probs -> getSignal -> pos.shift(1)*bar_ret), computed
    TWICE per trial: once on raw bar_ret (should reproduce run_live_
    trials' own M/meta), once on trend-adjusted excess_ret. Nothing here
    reimplements out_of_sample_probs/getSignal/sharpe_ratio -- all reused
    from ch11 directly."""
    bar_ret = close.pct_change().dropna()
    excess_ret = bar_ret - bar_ret.mean()

    cols_raw, cols_detrended, meta_rows = {}, {}, []
    for C in ch11.C_GRID:
        prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)  # real, unmodified
        events_c = pd.DataFrame({'t1': t1}).loc[prob.index]
        for step in ch11.STEP_GRID:
            sig = getSignal(events_c, step, prob, pred, numClasses=2, numThreads=1)
            pos = sig.reindex(close.index, method='ffill').fillna(0.0)
            name = f'C{C:.5g}_s{step:g}'
            cols_raw[name] = pos.shift(1).reindex(bar_ret.index).fillna(0.0) * bar_ret
            cols_detrended[name] = pos.shift(1).reindex(excess_ret.index).fillna(0.0) * excess_ret
            meta_rows.append({'trial': name, 'C': C, 'stepSize': step})

    M_raw = pd.DataFrame(cols_raw)
    M_detrended = pd.DataFrame(cols_detrended)
    meta = pd.DataFrame(meta_rows).set_index('trial')
    meta_raw = meta.copy()
    meta_raw['sharpe_full_sample'] = M_raw.apply(ch11.sharpe_ratio)
    meta_detrended = meta.copy()
    meta_detrended['sharpe_full_sample'] = M_detrended.apply(ch11.sharpe_ratio)
    return M_raw, meta_raw, M_detrended, meta_detrended, bar_ret, excess_ret


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            'Usage: python calibrate_pbo_detrended.py <snapshot_dir> <target_bars>'
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
    print(f"  {len(rebuild_result['bars'])} bars, {len(rebuild_result['events'])} events")

    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    enriched = enriched_result['enriched_events']
    feature_cols = list(enriched_result['feature_table'].columns)
    X = enriched[feature_cols]
    y = enriched['bin']
    t1 = enriched['t1']
    w = rebuild_result['w'].reindex(enriched.index)
    if w.isna().any():
        raise ValueError('w has NaN after reindexing to the enriched event index.')

    ch11 = load_ch11_driver()
    M_raw, meta_raw, M_det, meta_det, bar_ret, excess_ret = _build_trial_matrices(
        ch11, X, y, w, t1, close,
    )

    tw_aligned = rebuild_result['tw'].reindex(enriched.index)
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')

    eval_raw = evaluate_overfitting(M_raw, meta_raw, ch11, S=12, tw=tw_aligned)
    eval_det = evaluate_overfitting(M_det, meta_det, ch11, S=12, tw=tw_aligned)

    naive_long_raw = ch11.sharpe_ratio(bar_ret.values)
    naive_long_det = ch11.sharpe_ratio(excess_ret.values)

    print('\n' + '=' * 78)
    print('RAW vs. TREND-ADJUSTED (excess-return) PBO/DSR, same trial grid')
    print('=' * 78)
    print(f"{'':28s} {'RAW':>15s} {'DETRENDED':>15s}")
    def _ratio_str(winner, naive):
        if abs(naive) < 1e-6:
            return '  N/A (naive baseline ~0)'
        return f'{winner / naive:15.4f}'

    print(f"{'PBO':28s} {eval_raw['prob_overfit']:15.4f} {eval_det['prob_overfit']:15.4f}")
    print(f"{'DSR':28s} {eval_raw['dsr']:15.4f} {eval_det['dsr']:15.4f}")
    print(f"{'winning trial Sharpe':28s} {eval_raw['sr_hat']:15.4f} {eval_det['sr_hat']:15.4f}")
    print(f"{'winning trial':28s} {eval_raw['best_trial']:>15s} {eval_det['best_trial']:>15s}")
    print(f"{'naive always-long Sharpe':28s} {naive_long_raw:15.4f} {naive_long_det:15.4f}")
    print(f"{'winner vs. naive-long ratio':28s} "
          f"{_ratio_str(eval_raw['sr_hat'], naive_long_raw)} "
          f"{_ratio_str(eval_det['sr_hat'], naive_long_det)}")

    print('\nRead this as: does the DETRENDED winning trial still beat its own '
          "naive-long baseline by a similar (small) margin to the raw numbers? "
          'If so, this window shows no real timing/selection skill either way -- '
          'the raw PBO was just a trend artifact, and removing the trend does not '
          'reveal a hidden edge underneath it. If the detrended winner opens a '
          'real gap over naive-long that was NOT visible in the raw comparison, '
          'that is new evidence worth taking seriously.')


if __name__ == '__main__':
    main()
