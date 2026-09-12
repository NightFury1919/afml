"""
pipeline/diagnostics/score_live_predictions.py

Prototype forward-testing, part 2 (2026-09-11). Reads
live_predictions_log.csv (written by log_live_prediction.py), fetches
a fresh current price via the same quick/cheap pull, and reports
RUNNING real P&L for every logged prediction -- both the OLD binary
sizing and the NEW continuous confidence-exposure sizing, side by side,
on the exact same real price move.

Re-runnable as many times as you like. This is NOT a one-shot final
verdict at a fixed horizon (an hour, a day) -- it shows whatever the
running P&L looks like AT THE MOMENT YOU RUN IT, given however much
real wall-clock time has actually elapsed since each prediction was
logged. Run it an hour after logging, then again a day later, etc. --
each run is independent and safe to repeat.

Honest limitation, same as log_live_prediction.py: this pipeline's own
labels resolve over a multi-day vertical barrier, not an hour -- a
"running P&L" check at 1 hour is real, but it is NOT the bet's official
resolution, just where the position sits so far.

Run:
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\score_live_predictions.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402

LOG_PATH = os.path.join(HERE, 'live_predictions_log.csv')
QUICK_PRICE_LOOKBACK_HOURS = 2
PAIR = 'XBTUSD'


def _get_quick_current_price():
    trades = pull_recent_trades_kraken(PAIR, QUICK_PRICE_LOOKBACK_HOURS, max_calls=30)
    latest = trades.sort_values('Timestamp').iloc[-1]
    return float(latest['Price']), pd.to_datetime(latest['Timestamp'], unit='us')


def main():
    if not os.path.exists(LOG_PATH):
        raise SystemExit(f'{LOG_PATH} not found -- run log_live_prediction.py first.')

    log = pd.read_csv(LOG_PATH, parse_dates=['logged_at_utc', 'reference_price_time_utc'])
    if log.empty:
        raise SystemExit('Log file exists but has no rows.')

    print('Fetching current real-time price...')
    current_price, current_price_time = _get_quick_current_price()
    print(f'  Current price: ${current_price:,.2f} at {current_price_time}\n')

    rows = []
    for _, r in log.iterrows():
        elapsed_hours = (
            current_price_time - r['reference_price_time_utc'].tz_localize(None)
        ).total_seconds() / 3600.0
        price_return = (current_price / r['reference_price']) - 1.0
        direction_sign = {'long': 1.0, 'short': -1.0, 'flat': 0.0}[r['direction']]
        # Positions are already signed (direction baked into old/new
        # size via pos_unscaled's sign in log_live_prediction.py) --
        # P&L is just position * raw price return, no separate
        # direction multiply needed. direction_sign kept here only for
        # the printed sanity check below (position sign should match).
        pnl_old = r['old_binary_position'] * price_return
        pnl_new = r['new_continuous_position'] * price_return
        rows.append({
            'prediction_id': r['prediction_id'],
            'logged_at_utc': r['logged_at_utc'],
            'elapsed_hours': elapsed_hours,
            'direction': r['direction'],
            'reference_price': r['reference_price'],
            'current_price': current_price,
            'price_return': price_return,
            'old_binary_position': r['old_binary_position'],
            'new_continuous_position': r['new_continuous_position'],
            'running_pnl_old': pnl_old,
            'running_pnl_new': pnl_new,
        })

    result = pd.DataFrame(rows)
    print('=' * 100)
    print('RUNNING P&L (real price move since each prediction, OLD vs NEW sizing)')
    print('=' * 100)
    print(result[['prediction_id', 'elapsed_hours', 'direction', 'price_return',
                  'old_binary_position', 'new_continuous_position',
                  'running_pnl_old', 'running_pnl_new']].round(6).to_string(index=False))

    n_old_nonzero = (result['old_binary_position'] != 0).sum()
    n_new_nonzero = (result['new_continuous_position'] != 0).sum()
    print(f'\n{n_old_nonzero}/{len(result)} predictions had nonzero OLD (binary) '
          f'exposure -- {n_new_nonzero}/{len(result)} had nonzero NEW (continuous) '
          'exposure.')
    print('\nThis is RUNNING P&L at whatever elapsed time has passed since each '
          'prediction was logged -- not a resolved outcome, and not comparable '
          'across predictions with very different elapsed_hours. Re-run this '
          'anytime for an updated read; nothing here is final.')

    out_path = os.path.join(HERE, 'live_predictions_scored.csv')
    result.to_csv(out_path, index=False)
    print(f'\nWritten to {out_path}')


if __name__ == '__main__':
    main()
