"""
pipeline/diagnostics/audit_kraken_cusum_h_staleness.py

Kraken counterpart to audit_cusum_h_staleness.py (2026-08-21's original
Binance.US methodology, reused directly here, not reimplemented). That
script asked: is CUSUM_H still the right relative threshold for TODAY's
BTCUSDT data, independent of any downstream metric impact? This script
asks the analogous question ACROSS venues: is CUSUM_H=313 (currently
reused unvalidated on Kraken data throughout the 2026-08-25 evaluation
-- see CALIBRATION_AUDIT.md's "Kraken Evaluated as a Higher-Density Data
Source" section) the right relative threshold for Kraken's OWN bar-to-
bar price-move distribution, or does Kraken's data character (same
underlying BTC/USD price, but a different venue's order flow/spread/
trader mix) call for a different value?

MIN_RET is deliberately NOT addressed here -- CALIBRATION_AUDIT.md's own
audit table already classifies it as a modeling choice (transaction-
cost dependent), never data-derived even for Binance.US. There is no
staleness-audit methodology to mirror for it; re-deriving it per-venue
would be inventing a new calibration question, not re-running an
established one. CUSUM_H is the one constant with a real, repeatable,
data-driven measurement methodology already established -- this script
extends that methodology, not invents a new one.

METHOD (identical to the original script's, applied cross-venue instead
of cross-time): runs a FRESH Binance.US 720h live pull and the ALREADY-
CAPTURED Kraken 720h snapshot (kraken_snapshot_720h_2026-08-25 -- no
need to re-pull ~1.68M trades again) through the SAME real
rebuild.build_bars_and_labels() chain, measures each venue's bar-to-bar
CLOSE price diff distribution (the exact series CUSUM_H is applied
against), and derives what h would reproduce Binance's relative CUSUM
firing rate on Kraken's own bar series.

target_bars=1000 used here (NOT the original script's target_bars=250)
-- 250 was that script's own comment "matching this project's
established regime" as of 2026-08-21's writing; production has since
moved to target_bars=1000 (see CALIBRATION_AUDIT.md's "Adopted as
Production Default" section), so this comparison uses the CURRENT
production default on both venues for a fair, currently-relevant
comparison.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\\diagnostics\\audit_kraken_cusum_h_staleness.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ORCH)
sys.path.insert(0, ROOT)

from ingestion import pull_recent_trades          # noqa: E402
import rebuild                                     # noqa: E402
from ch02.bars import filters as ch02_filters      # noqa: E402

LOOKBACK_HOURS = 720
TARGET_BARS = 1000  # current production default -- see module docstring
KRAKEN_SNAPSHOT_DIR = os.path.join(HERE, 'kraken_snapshot_720h_2026-08-25')


def bar_close_diff_stats(raw_trades, label):
    """Identical to audit_cusum_h_staleness.py's real function -- runs
    raw_trades through rebuild.py's REAL build_bars_and_labels(), then
    reports the bar-to-bar CLOSE price diff distribution CUSUM_H is
    actually applied against."""
    result = rebuild.build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
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
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set -- see ingestion.py\'s docstring '
            'for how to get a free read-only key.'
        )
    print(f'Pulling last {LOOKBACK_HOURS}h of live BTCUSDT trades from Binance.US...')
    binance_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(binance_trades)} raw trades pulled')
    binance_stats = bar_close_diff_stats(binance_trades, 'Binance.US BTCUSDT (fresh live pull)')

    kraken_path = os.path.join(KRAKEN_SNAPSHOT_DIR, 'raw_trades.parquet')
    if not os.path.exists(kraken_path):
        raise SystemExit(
            f'{kraken_path} not found -- run capture_kraken_snapshot.py '
            '--hours 720 first (see 2026-08-25 handoff).'
        )
    print(f'\nLoading frozen Kraken snapshot from {KRAKEN_SNAPSHOT_DIR}...')
    kraken_trades = pd.read_parquet(kraken_path)
    print(f'  {len(kraken_trades)} raw trades loaded')
    kraken_stats = bar_close_diff_stats(kraken_trades, 'Kraken XBTUSD (720h snapshot)')

    # Same ratio-scaling approach as the original Binance-vs-March script:
    # what h would reproduce Binance's relative CUSUM firing rate on
    # Kraken's own bar series, given the two venues' differing bar-to-bar
    # diff distributions.
    ratio = kraken_stats['diff_std'] / binance_stats['diff_std']
    h_equivalent = rebuild.CUSUM_H * ratio

    print('\n=== Cross-venue staleness summary ===')
    print(f'Kraken/Binance bar-to-bar diff std ratio: {ratio:.3f}x')
    print(f'CUSUM_H={rebuild.CUSUM_H} event rate -- '
          f'Binance.US: {binance_stats["n_events"] / len(binance_stats["close"]):.1%} of bars, '
          f'Kraken: {kraken_stats["n_events"] / len(kraken_stats["close"]):.1%} of bars')
    print(f'Rough h-equivalent to match Binance.US\'s relative event rate on '
          f'Kraken\'s data: h~{h_equivalent:.0f} '
          f'(current CUSUM_H={rebuild.CUSUM_H}, reused unvalidated on Kraken '
          f'throughout the 2026-08-25 evaluation)')
    print('\nThis is a MEASUREMENT, not a new calibration decision -- same '
          'caution as the original script. If the gap is large, the next '
          'step is deciding whether to adopt h_equivalent for Kraken '
          'specifically (a real, deliberate design decision, not a silent '
          'swap), and re-running today\'s target_bars sweep under it before '
          'trusting any DSR reading built on the current, unvalidated '
          'CUSUM_H=313.')


if __name__ == '__main__':
    main()
