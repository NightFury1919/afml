"""
pipeline/diagnostics/audit_cusum_h_staleness.py

Measures whether CUSUM_H=500 (rebuild.py's flat dollar threshold, carried
over unchanged from the March 2026 BTC/TUSD static-data calibration -- see
rebuild.py's own KNOWN OPEN QUESTION) is still calibrated to the CURRENT
BTC/USDT price/volatility regime.

Unlike the 2026-08-16/2026-08-18/2026-08-20 CUSUM_H sensitivity work (which
asked "what happens to T_effective/DSR/PBO if h is a DIFFERENT value"), this
script asks a previously-unasked question: is h=500 STILL the right
relative threshold for TODAY's data, independent of any downstream metric
impact? It measures the actual bar-to-bar dollar-move distribution CUSUM_H
is applied against, on both the March static baseline and a fresh live
pull, run through the SAME real rebuild.py chain -- not a raw-trade-level
proxy.

Reuses rebuild.py's build_bars_and_labels() and ch02's real
cusum_filter() directly -- no reimplementation.

Usage
-----
    conda activate mlfinlab
    cd C:\ws\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\diagnostics\audit_cusum_h_staleness.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ORCH)
sys.path.insert(0, ROOT)

from ingestion import pull_recent_trades, RAW_TRADE_COLUMNS  # noqa: E402
import rebuild                                                # noqa: E402
from ch02.bars import filters as ch02_filters                 # noqa: E402

STATIC_CSV = os.path.join(ROOT, 'input_data', 'BTCTUSD-trades-2026-03.csv')
LOOKBACK_HOURS = 720  # matches run_pipeline_live.py / capture_t_effective_snapshot.py


def bar_close_diff_stats(raw_trades, label):
    """Runs raw_trades through rebuild.py's REAL build_bars_and_labels()
    (target_bars=250, matching this project's established regime), then
    reports the bar-to-bar CLOSE price diff distribution -- the exact
    series CUSUM_H is applied against inside rebuild.py, not a raw-trade
    proxy."""
    result = rebuild.build_bars_and_labels(raw_trades, target_bars=250)
    close = result['close']
    diffs = close.diff().dropna()

    cusum_df = pd.DataFrame({'Date': close.index, 'Price': close.values})
    events_at_h = ch02_filters.cusum_filter(cusum_df, h=rebuild.CUSUM_H)

    print(f'\n--- {label} ---')
    print(f'  n_bars: {len(close)}')
    print(f'  bar close price: mean=${close.mean():,.2f}  std=${close.std():,.2f}')
    print(f'  bar-to-bar close diff: mean|d|=${diffs.abs().mean():,.2f}  '
          f'std=${diffs.std():,.2f}')
    print(f'  CUSUM events at h={rebuild.CUSUM_H}: {len(events_at_h)} '
          f'({len(events_at_h) / len(close):.1%} of bars)')
    return {
        'close': close, 'diffs': diffs, 'n_events': len(events_at_h),
        'diff_std': diffs.std(),
    }


def main():
    print('Loading March 2026 static baseline...')
    static_trades = pd.read_csv(STATIC_CSV, header=None, names=RAW_TRADE_COLUMNS)
    static_stats = bar_close_diff_stats(static_trades, 'March 2026 static (BTC/TUSD)')

    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set -- see ingestion.py\'s docstring '
            'for how to get a free read-only key.'
        )
    print(f'\nPulling last {LOOKBACK_HOURS}h of live BTCUSDT trades from Binance.US...')
    live_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(live_trades)} raw trades pulled')
    live_stats = bar_close_diff_stats(live_trades, 'Live BTC/USDT (today)')

    # First-order "what h would reproduce March's relative CUSUM event rate
    # on today's bar series" -- a candidate MEASUREMENT, not a new committed
    # calibration. Scales CUSUM_H by the ratio of the two series' bar-to-bar
    # diff std, since CUSUM_H is a threshold on cumulative dollar moves of
    # that same kind.
    ratio = live_stats['diff_std'] / static_stats['diff_std']
    h_equivalent = rebuild.CUSUM_H * ratio

    print('\n=== Staleness summary ===')
    print(f'Live/March bar-to-bar diff std ratio: {ratio:.3f}x')
    print(f'CUSUM_H={rebuild.CUSUM_H} event rate -- '
          f'March: {static_stats["n_events"] / len(static_stats["close"]):.1%} of bars, '
          f'Live: {live_stats["n_events"] / len(live_stats["close"]):.1%} of bars')
    print(f'Rough h-equivalent to match March\'s relative event rate on '
          f'today\'s data: h~{h_equivalent:.0f} '
          f'(current CUSUM_H={rebuild.CUSUM_H})')
    print('\nThis is a MEASUREMENT, not a new calibration decision -- if the '
          'gap is large, the next step is a deliberate h-per-day-volatility '
          'redesign (rebuild.py\'s own KNOWN OPEN QUESTION), not silently '
          'swapping in h_equivalent.')


if __name__ == '__main__':
    main()