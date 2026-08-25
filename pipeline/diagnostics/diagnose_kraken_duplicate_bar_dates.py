"""
pipeline/diagnostics/diagnose_kraken_duplicate_bar_dates.py

Confirms (or rules out) the working hypothesis behind the frac_diff_ffd
"duplicate index labels" failures in today's calibrate_kraken_target_bars.py
sweep: at small enough dollar-bar thresholds, a single large Kraken trade
may exceed several bars' worth of threshold in one step, and
ch02/bars/standard_bars.py's real dollar_bars() may assign that SAME
trade's timestamp as the closing Date for more than one bar -- a
different mechanism from the trade-level timestamp collision already
fixed in ingestion_kraken.py today (that fix ensures no two TRADES share
a timestamp; it says nothing about whether two different BARS can still
end up sharing a closing Date).

Diagnostic-only -- does NOT modify ch02/bars/standard_bars.py or propose
a fix. Per this project's book-fidelity rule, any change to that module
needs the real book snippet checked first and an explicit design
decision, not a same-session patch under time pressure.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\diagnose_kraken_duplicate_bar_dates.py pipeline\\diagnostics\\kraken_snapshot_720h_2026-08-25 --target-bars 5000
"""
import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)

from rebuild import build_bars_and_labels   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('snapshot_dir')
    parser.add_argument('--target-bars', type=int, default=5000,
                         help='Worst-observed case in today\'s sweep '
                              '(212 duplicates) -- start here.')
    args = parser.parse_args()

    raw_trades_path = os.path.join(args.snapshot_dir, 'raw_trades.parquet')
    raw_trades = pd.read_parquet(raw_trades_path)
    print(f'Loaded {len(raw_trades)} raw trades from {args.snapshot_dir}')

    rebuild_result = build_bars_and_labels(raw_trades, target_bars=args.target_bars)
    bars = rebuild_result['bars']
    threshold = rebuild_result['threshold']
    print(f'\ntarget_bars={args.target_bars}, threshold=${threshold:,.2f}')
    print(f'{len(bars)} bars produced')

    dup_mask = bars.index.duplicated(keep=False)
    n_dup_rows = dup_mask.sum()
    n_dup_dates = bars.index[dup_mask].nunique()
    print(f'\n{n_dup_rows} bar rows share a Date with at least one other bar '
          f'({n_dup_dates} distinct duplicated Date values)')

    if n_dup_rows == 0:
        print('\nNo duplicates at this target_bars -- try a higher value '
              '(the sweep showed duplicate COUNT rising with target_bars).')
        return

    print('\n--- First 3 duplicated Date groups, with bar details ---')
    dup_dates = bars.index[dup_mask].unique()[:3]
    for dt in dup_dates:
        group = bars.loc[[dt]]
        print(f'\nDate = {dt}  ({len(group)} bars share this exact timestamp)')
        print(group[['Open', 'High', 'Low', 'Close', 'Vwap']].to_string())

    print('\n--- Checking: does a single raw trade\'s dollar value alone '
          'exceed the threshold? ---')
    raw_trades_sorted = raw_trades.sort_values('Timestamp')
    dollar_per_trade = raw_trades_sorted['Price'] * raw_trades_sorted['Volume']
    big_trades = dollar_per_trade[dollar_per_trade >= threshold]
    print(f'{len(big_trades)} individual trades have dollar value >= the '
          f'${threshold:,.2f} threshold on their own')
    if len(big_trades) > 0:
        multiples = (big_trades / threshold).round(1)
        print(f'  Largest such trade is {multiples.max():.1f}x the threshold '
              f'by itself')
        print(f'  Distribution of how many threshold-multiples these trades '
              f'represent:')
        print(f'    {multiples.describe().to_string()}')
        print('\n  If this count is nonzero and multiples go well above 1x, '
              'that CONFIRMS the hypothesis: single large trades are '
              'completing multiple bars at once, and dollar_bars() has no '
              'other trade to assign as each bar\'s distinct closing Date.')
    else:
        print('\n  Hypothesis NOT confirmed by this check -- no single trade '
              'reaches the threshold alone. Duplicate bar Dates have some '
              'OTHER cause -- needs further investigation before proposing '
              'any fix.')


if __name__ == '__main__':
    main()
