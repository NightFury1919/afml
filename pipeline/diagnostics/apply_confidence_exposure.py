"""
pipeline/diagnostics/apply_confidence_exposure.py

Option 4, part 2 (2026-09-11). Design, per Ethan's explicit choices:
  - confidence = DSR * (1 - PBO) -- penalizes both lack of statistical
    significance AND overfitting risk, not DSR alone.
  - exposure_scalar = linear ramp from 0 (at the EMPIRICALLY calibrated
    null threshold from calibrate_confidence_threshold.py -- not an
    assumed/guessed curve shape) to 1 (at confidence=1.0, the
    theoretical max).
  - Recomputed FRESH each run -- no smoothing across time.

This replaces the binary "deploy at full size if DSR>=0.95, else don't
trade at all" strategy-level gate with a continuous scalar. It does NOT
touch per-bet sizing, which was already continuous (out_of_sample_probs
-> getSignal, Ch10/Ch11's real code, reused completely unmodified here)
-- only the single OVERALL exposure multiplier changes from a step
function to a smooth ramp.

Reports the OLD binary decision alongside the NEW continuous one on the
same real run, so the contrast is concrete rather than theoretical.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\apply_confidence_exposure.py <snapshot_dir> <target_bars>
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
from stages import load_ch11_driver, run_live_trials, evaluate_overfitting  # noqa: E402

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                             # noqa: E402

OLD_BINARY_DSR_THRESHOLD = 0.95   # the gate this whole option replaces
THRESHOLD_FILE = os.path.join(HERE, 'confidence_threshold.txt')
FALLBACK_ZERO_EXPOSURE_THRESHOLD = 0.7235  # real 2026-09-11 calibration,
    # used only if confidence_threshold.txt is missing -- re-run
    # calibrate_confidence_threshold.py to refresh this properly rather
    # than relying on the fallback long-term.


def _load_zero_exposure_threshold():
    if os.path.exists(THRESHOLD_FILE):
        with open(THRESHOLD_FILE) as f:
            return float(f.read().strip())
    print(f'  NOTE: {THRESHOLD_FILE} not found -- using fallback '
          f'{FALLBACK_ZERO_EXPOSURE_THRESHOLD} (re-run '
          'calibrate_confidence_threshold.py to refresh from current data).')
    return FALLBACK_ZERO_EXPOSURE_THRESHOLD


def exposure_scalar(confidence, zero_exposure_threshold):
    """Linear ramp: 0 at/below the empirically-calibrated null threshold,
    1 at confidence=1.0 (theoretical max), linear between."""
    if confidence <= zero_exposure_threshold:
        return 0.0
    return min(1.0, (confidence - zero_exposure_threshold) / (1.0 - zero_exposure_threshold))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            'Usage: python apply_confidence_exposure.py <snapshot_dir> <target_bars>'
        )
    snapshot_dir = sys.argv[1]
    target_bars = int(sys.argv[2])
    raw_trades_path = os.path.join(snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_trades_path):
        raise SystemExit(f'{raw_trades_path} not found -- wrong snapshot dir?')

    zero_exposure_threshold = _load_zero_exposure_threshold()

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

    work_root = os.path.join(HERE, 'confidence_exposure_work')
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
    scalar = exposure_scalar(confidence, zero_exposure_threshold)
    old_binary_decision = 1.0 if dsr >= OLD_BINARY_DSR_THRESHOLD else 0.0

    winning_C = float(meta.loc[eval_result['best_trial'], 'C'])
    winning_step = float(meta.loc[eval_result['best_trial'], 'stepSize'])

    print(f'\n{"=" * 74}')
    print('OLD (binary) vs. NEW (continuous) strategy-level decision')
    print('=' * 74)
    print(f'  DSR={dsr:.4f}   PBO={pbo:.4f}   confidence=DSR*(1-PBO)={confidence:.4f}')
    print(f'  zero_exposure_threshold (empirical, 2026-09-11 calibration): '
          f'{zero_exposure_threshold:.4f}')
    print(f'  OLD: DSR>={OLD_BINARY_DSR_THRESHOLD} ? '
          f'{"YES -> trade at 100%" if old_binary_decision else "NO -> trade at 0%"}')
    print(f'  NEW: exposure_scalar = {scalar:.4f} -> trade at {scalar*100:.1f}% of '
          'per-bet sizing')

    prob, pred = ch11.out_of_sample_probs(X, y, w, t1, winning_C)  # real, unmodified
    events_c = pd.DataFrame({'t1': t1}).loc[prob.index]
    sig = getSignal(events_c, winning_step, prob, pred, numClasses=2, numThreads=1)
    pos_unscaled = sig.reindex(close.index, method='ffill').fillna(0.0)
    pos_new = pos_unscaled * scalar
    pos_old = pos_unscaled * old_binary_decision

    print(f'\n{"=" * 74}')
    print('Resulting position sizing (winning trial\'s own real per-bet signal, '
          'unmodified -- only the overall scalar differs)')
    print('=' * 74)
    print(f'  mean |position|, OLD (binary):      {pos_old.abs().mean():.4f}')
    print(f'  mean |position|, NEW (continuous):  {pos_new.abs().mean():.4f}')
    print(f'  % bars in market, OLD:              {(pos_old != 0).mean()*100:.1f}%')
    print(f'  % bars in market, NEW:              {(pos_new != 0).mean()*100:.1f}%'
          f'  (same bars as unscaled -- only SIZE changes, not which bars)' )

    print('\nRead this as: the OLD rule either fully deploys or fully abstains --')
    print('a strategy that clears DSR>=0.95 rarely, if ever, at realistic edge')
    print('sizes (see calibrate_n_trials_floor.py) means the old rule almost')
    print('always outputs exactly zero. The NEW rule still outputs a small,')
    print('honest, continuously-scaled exposure even when statistical confidence')
    print('is modest -- rather than an all-or-nothing cliff at an unreachable bar.')


if __name__ == '__main__':
    main()
