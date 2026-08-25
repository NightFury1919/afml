"""
pipeline/diagnostics/verify_kraken_pull.py

FIRST real-machine test of ingestion_kraken.py's pull_recent_trades_kraken()
-- deliberately small (2h default), before trusting it for a full 720h+
pull. Per this project's real-data-first, verify-before-trusting
convention: two things in that module are inferences, not yet confirmed
(see its own LOAD-BEARING notes), and this script exists specifically
to let you eyeball whether they hold up against real data:

  1. Does the returned data actually cover the requested window (pagination
     correctness, including the since-parameter precision workaround)?
  2. Does the buy/sell -> IsBuyerMaker mapping look plausible (roughly
     balanced split, not a suspicious ~100/0 skew that would suggest a
     sign error)?

Also runs the resulting trades through rebuild.py's REAL
compute_dynamic_threshold() and preprocess_raw_trades() (reused, not
reimplemented) as a further sanity check -- if Kraken's schema mapping
is subtly wrong, this is where a real, loud failure is most likely to
surface (same defense-in-depth this project already uses for the
Binance.US pull's own timestamp-collision issue).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\verify_kraken_pull.py
    python pipeline\\diagnostics\\verify_kraken_pull.py --hours 6
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402
from rebuild import preprocess_raw_trades, compute_dynamic_threshold  # noqa: E402

PAIR = 'XBTUSD'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=2.0,
                         help='Deliberately small for a first real check.')
    args = parser.parse_args()

    print('=' * 70)
    print(f'VERIFYING Kraken pull: pair={PAIR}, lookback_hours={args.hours}')
    print('=' * 70)

    print('\nPulling...')
    raw_trades = pull_recent_trades_kraken(PAIR, args.hours)
    print(f'  {len(raw_trades)} trades pulled')

    print('\n--- Schema check ---')
    print(raw_trades.dtypes)
    print('\nFirst 3 rows:')
    print(raw_trades.head(3).to_string())
    print('\nLast 3 rows:')
    print(raw_trades.tail(3).to_string())

    print('\n--- Coverage check ---')
    span_start = pd.to_datetime(raw_trades['Timestamp'].min(), unit='us')
    span_end = pd.to_datetime(raw_trades['Timestamp'].max(), unit='us')
    span_hours = (span_end - span_start).total_seconds() / 3600.0
    print(f'  Requested: last {args.hours}h')
    print(f'  Actual coverage: {span_start} to {span_end} '
          f'({span_hours:.2f}h)')
    print(f'  Rate: {len(raw_trades) / span_hours:.1f} trades/hour '
          f'(scouting check earlier today estimated ~3,676/hour)')
    now = pd.Timestamp.utcnow().tz_localize(None)
    gap_to_now_min = (now - span_end).total_seconds() / 60.0
    print(f'  Most recent trade is {gap_to_now_min:.1f} minutes before now '
          f'(should be small -- large gap may mean pagination stopped early)')

    print('\n--- Buy/sell -> IsBuyerMaker sanity check ---')
    vc = raw_trades['IsBuyerMaker'].value_counts(normalize=True)
    print(vc)
    print('  Expect roughly balanced (neither side under ~20-25% typically) '
          '-- a wildly skewed split could indicate the mapping is inverted '
          'or mis-parsed, not necessarily real market imbalance.')

    print('\n--- TradeID uniqueness check ---')
    n_total = len(raw_trades)
    n_unique = raw_trades['TradeID'].nunique()
    print(f'  {n_unique}/{n_total} TradeIDs unique '
          f'({"OK" if n_unique == n_total else "MISMATCH -- investigate"})')

    print('\n--- Downstream compatibility check (real rebuild.py functions) ---')
    try:
        preprocessed = preprocess_raw_trades(raw_trades)
        print(f'  preprocess_raw_trades() succeeded: {len(preprocessed)} rows, '
              f'columns: {list(preprocessed.columns)}')
        threshold = compute_dynamic_threshold(raw_trades, target_bars=50)
        print(f'  compute_dynamic_threshold(target_bars=50) succeeded: '
              f'${threshold:,.2f}')
    except Exception as e:
        print(f'  FAILED: {type(e).__name__}: {e}')
        print('  This means something in the schema/mapping needs fixing '
              'before Kraken data can flow through the real pipeline chain.')

    print('\n' + '=' * 70)
    print('If all the above looks right, re-run with a longer --hours '
          '(e.g. 24, matching the earlier BTCUSD/BTCUSDT comparison) '
          'before trusting this for a full-scale pull.')
    print('=' * 70)


if __name__ == '__main__':
    main()
