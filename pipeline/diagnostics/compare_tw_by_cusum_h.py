"""
compare_tw_by_cusum_h.py -- ONE-OFF DIAGNOSTIC, not part of the pipeline.

Pulls raw trades ONCE, then runs the real build_bars_and_labels() chain
TWICE against the exact same trades -- once at CUSUM_H=500 (committed
default), once at CUSUM_H=100 (2026-08-16 experimental value) -- via the
same monkeypatch-and-restore pattern as run_pipeline_live_experimental_
cusum.py. Prints Ch04's real get_average_uniqueness() (tw) stats for
both, side by side, to test the barrier-overlap hypothesis: that h=100's
much higher DSR (0.8858 vs 0.5943 at h=500, even as PBO got WORSE, 80.00%
vs 71.43%) may be an artifact of triple-barrier windows overlapping more
heavily at a denser CUSUM sampling rate -- inflating the effective sample
size DSR's T assumes, since deflated_sharpe_ratio() currently uses raw
bet count, not uniqueness-weighted count.

Pulling ONCE and reusing the SAME raw_trades for both regimes isolates
the CUSUM_H effect cleanly -- a fresh second pull would confound this
with normal pull-to-pull price-path variation.

Requires BINANCE_API_KEY, same as run_pipeline_live.py.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python compare_tw_by_cusum_h.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(HERE, 'pipeline')
sys.path.insert(0, os.path.join(PIPELINE_DIR, 'orchestration'))

from ingestion import pull_recent_trades              # noqa: E402
import rebuild as rebuild_module                        # noqa: E402
from rebuild import build_bars_and_labels               # noqa: E402

LOOKBACK_HOURS = 720


def describe(label, rebuild_result):
    tw = rebuild_result['tw']
    w = rebuild_result['w']
    print(f"\n--- {label} ---")
    print(f"  n_events: {len(rebuild_result['events'])}")
    print(f"  tw (avg uniqueness): mean={tw.mean():.4f}  median={tw.median():.4f}  "
          f"min={tw.min():.4f}  max={tw.max():.4f}")
    print(f"  w (sample weight):   mean={w.mean():.4f}  median={w.median():.4f}")
    # Effective independent sample size implied by average uniqueness --
    # a rough intuition, not a formal DSR correction: if events are on
    # average X% unique, the "effective T" DSR should arguably be seeing
    # is closer to n_events * X than raw n_events.
    print(f"  n_events * tw.mean() (rough 'effective T' if uniqueness-weighted): "
          f"{len(rebuild_result['events']) * tw.mean():.1f}")


def main():
    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            'BINANCE_API_KEY is not set. See ingestion.py\'s module '
            'docstring for how to get a free read-only key.'
        )

    print(f'Pulling last {LOOKBACK_HOURS}h of BTCUSDT trades from Binance.US '
          f'(ONE pull, reused for both CUSUM_H regimes below)...')
    raw_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(raw_trades)} raw trades pulled')

    original_h = rebuild_module.CUSUM_H

    try:
        rebuild_module.CUSUM_H = 500
        result_h500 = build_bars_and_labels(raw_trades)
    finally:
        rebuild_module.CUSUM_H = original_h
    describe('CUSUM_H=500 (committed default)', result_h500)

    try:
        rebuild_module.CUSUM_H = 100
        result_h100 = build_bars_and_labels(raw_trades)
    finally:
        rebuild_module.CUSUM_H = original_h
    describe('CUSUM_H=100 (experimental)', result_h100)

    print('\n--- Summary ---')
    print(f"  tw.mean() dropped from {result_h500['tw'].mean():.4f} to "
          f"{result_h100['tw'].mean():.4f} "
          f"({'CONFIRMS' if result_h100['tw'].mean() < result_h500['tw'].mean() else 'DOES NOT confirm'} "
          f"the barrier-overlap hypothesis if it dropped substantially)")


if __name__ == '__main__':
    main()
