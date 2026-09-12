"""
pipeline/diagnostics/log_live_prediction.py

Prototype forward-testing, part 1 (2026-09-11): records what the model
would do RIGHT NOW -- both the OLD binary-gate sizing and the NEW
continuous confidence-exposure sizing (apply_confidence_exposure.py's
logic, reused here directly rather than replacing the old model) --
alongside a real-time reference price, so score_live_predictions.py can
check later how each would actually have performed. Prototype means
BOTH paths are logged side by side; nothing about the existing pipeline
changes as a result of running this.

Two real data sources, deliberately different in cost:
  1. The (already-captured, possibly hours/days old) frozen snapshot --
     used for the actual model computation (bars, features, trial grid,
     PBO/DSR/confidence). This is expensive (the same real pipeline
     apply_confidence_exposure.py runs), not something to re-pull fresh
     every time.
  2. A QUICK fresh Kraken pull (default 2 hours lookback -- a few
     thousand trades, seconds, not the 30+ minute cost of a full 720h
     snapshot) -- used ONLY to get the actual current real-time price
     as this prediction's reference point. This is intentionally cheap
     and re-run every time this script runs.

Honest limitation: the model's own signal is only as fresh as the frozen
snapshot's last bar/event -- if that snapshot is a day old, "right now"
means "using yesterday's most recent signal, priced at today's real
market." That gap is reported explicitly, not hidden.

Appends one row per run to live_predictions_log.csv (created if absent).

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\log_live_prediction.py <snapshot_dir> <target_bars>
"""
import os
import sys
import time
import uuid
from datetime import datetime, timezone

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
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402
from ingestion_kraken import pull_recent_trades_kraken       # noqa: E402

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                             # noqa: E402

OLD_BINARY_DSR_THRESHOLD = 0.95
THRESHOLD_FILE = os.path.join(HERE, 'confidence_threshold.txt')
FALLBACK_ZERO_EXPOSURE_THRESHOLD = 0.7235
LOG_PATH = os.path.join(HERE, 'live_predictions_log.csv')
QUICK_PRICE_LOOKBACK_HOURS = 2
PAIR = 'XBTUSD'


def _load_zero_exposure_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return float(f.read().strip())
    return FALLBACK_ZERO_EXPOSURE_THRESHOLD


def _exposure_scalar(confidence, zero_exposure_threshold):
    if confidence <= zero_exposure_threshold:
        return 0.0
    return min(1.0, (confidence - zero_exposure_threshold) / (1.0 - zero_exposure_threshold))


def _get_quick_current_price():
    """Fast, cheap pull -- a couple thousand trades, seconds -- purely
    to get an actual current real-time price. NOT a substitute for a
    full snapshot; do not reuse this for model computation."""
    trades = pull_recent_trades_kraken(PAIR, QUICK_PRICE_LOOKBACK_HOURS, max_calls=30)
    latest = trades.sort_values('Timestamp').iloc[-1]
    return float(latest['Price']), pd.to_datetime(latest['Timestamp'], unit='us')


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            'Usage: python log_live_prediction.py <snapshot_dir> <target_bars>'
        )
    snapshot_dir = sys.argv[1]
    target_bars = int(sys.argv[2])
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    zero_exposure_threshold = _load_zero_exposure_threshold()

    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded frozen snapshot: {len(raw_trades)} raw trades from {snapshot_dir}')

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=target_bars)
    close = rebuild_result['close']
    last_bar_time = close.index[-1]
    print(f'  {len(rebuild_result["bars"])} bars, last bar: {last_bar_time}')

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

    work_root = os.path.join(HERE, 'live_prediction_work')
    staging_dir = os.path.join(work_root, 'staging')
    here_dir = os.path.join(work_root, 'ch11_here')
    stage_live_training_tables(rebuild_result, enriched_result, staging_dir)

    ch11 = load_ch11_driver()
    M, meta = run_live_trials(ch11, staging_dir, here_dir)
    tw_aligned = rebuild_result['tw'].reindex(enriched.index)
    if tw_aligned.isna().any():
        raise ValueError('tw has NaN after reindexing to the enriched event index.')
    eval_result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw_aligned)

    dsr = eval_result['dsr']
    pbo = eval_result['prob_overfit']
    confidence = dsr * (1 - pbo)
    scalar = _exposure_scalar(confidence, zero_exposure_threshold)
    old_binary_decision = 1.0 if dsr >= OLD_BINARY_DSR_THRESHOLD else 0.0

    winning_C = float(meta.loc[eval_result['best_trial'], 'C'])
    winning_step = float(meta.loc[eval_result['best_trial'], 'stepSize'])

    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, winning_C)  # real, unmodified
    events_c = pd.DataFrame({'t1': t1}).loc[prob.index]
    sig = getSignal(events_c, winning_step, prob, pred, numClasses=2, numThreads=1)
    pos_unscaled = sig.reindex(close.index, method='ffill').fillna(0.0)

    current_signal = float(pos_unscaled.iloc[-1])
    direction = 'long' if current_signal > 0 else ('short' if current_signal < 0 else 'flat')
    new_size = current_signal * scalar
    old_size = current_signal * old_binary_decision

    print('\nFetching current real-time price (quick pull, not the model input)...')
    current_price, current_price_time = _get_quick_current_price()
    staleness_hours = (current_price_time - last_bar_time).total_seconds() / 3600.0

    logged_at = datetime.now(timezone.utc)
    row = {
        'prediction_id': str(uuid.uuid4())[:8],
        'logged_at_utc': logged_at.isoformat(),
        'snapshot_dir': snapshot_dir,
        'target_bars': target_bars,
        'last_bar_time': str(last_bar_time),
        'reference_price': current_price,
        'reference_price_time_utc': current_price_time.isoformat(),
        'model_signal_staleness_hours': staleness_hours,
        'dsr': dsr, 'pbo': pbo, 'confidence': confidence,
        'zero_exposure_threshold': zero_exposure_threshold,
        'exposure_scalar': scalar,
        'direction': direction,
        'winning_C': winning_C, 'winning_step': winning_step,
        'old_binary_position': old_size,
        'new_continuous_position': new_size,
        'scored': False,
    }

    file_exists = os.path.exists(LOG_PATH)
    pd.DataFrame([row]).to_csv(LOG_PATH, mode='a', header=not file_exists, index=False)

    print(f'\n{"=" * 74}')
    print(f'LOGGED prediction {row["prediction_id"]} at {logged_at.isoformat()}')
    print('=' * 74)
    print(f'  Model signal is based on data up to {last_bar_time} '
          f'({staleness_hours:.1f}h stale vs. the real-time reference price)')
    print(f'  Reference price: ${current_price:,.2f} at {current_price_time}')
    print(f'  Direction: {direction}   DSR={dsr:.4f}  PBO={pbo:.4f}  '
          f'confidence={confidence:.4f}')
    print(f'  OLD (binary) position:      {old_size:+.4f}')
    print(f'  NEW (continuous) position:  {new_size:+.4f}')
    print(f'\nRun score_live_predictions.py anytime later (an hour, a day, '
          'whenever) to see how this would have performed so far -- '
          're-runnable as many times as you like, shows RUNNING P&L at '
          'whatever elapsed time has passed, not a one-shot final verdict.')


if __name__ == '__main__':
    main()
