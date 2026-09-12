"""
pipeline/diagnostics/measure_real_btc_roll_c_signal.py

Stage 0 of the "is BTC's sample size high enough" investigation
(2026-09-08 session, following directly from the meanshift edge-detection
close-out).

Question: the closed BTC null (Ch11 PBO~0.83, Ch12 CPCV all 5 paths
negative, Ch13 O-U phi_hat~1.03, Ch14 DSR 0/5 survive) was built entirely
on T_effective~100-scale backtests -- the SAME scale the meanshift
investigation just proved is blind to real, injected signals below
edge_strength=3.0. Before trusting the BTC null as "no edge" rather than
"edge below detection floor", we need to know how big BTC's OWN real
correlation actually is, in the same units the meanshift investigation
already calibrated (edge_strength / raw_signal_corr).

This script answers the narrower, first half of that question: what is
roll_c's (and every other Ch19 feature's) REAL bar-level correlation
against next-bar return, on real Kraken BTC/USD data? This is EXACTLY
Stage 2 of trace_meanshift_signal_leakage.py's methodology, applied to
real trades instead of a synthetic injection -- no new measurement
concept, just pointed at real data.

Deliberately NOT yet doing: mapping this number onto the synthetic
edge_strength ladder (that requires sweeping the synthetic generator's
OWN roll_c bar-level correlation across edge_strength, which the existing
sweep scripts never measured directly -- they tracked lag1_autocorr, a
different diagnostic). That mapping is real, separate follow-up work,
deliberately deferred until this real number is in hand and worth
mapping.

CUSUM_H=313 note: audit_kraken_cusum_h_staleness.py (2026-09-08) measured
h_equivalent~303 for Kraken vs the current CUSUM_H=313 -- a ~3% gap, well
within single-window measurement noise. Reusing CUSUM_H=313 unchanged
here is a deliberate, confirmed-reasonable choice, not an oversight.

Diagnostic-only: no new AFML formula, no change to any committed chapter
or pipeline module. Read-only against the real chain.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\measure_real_btc_roll_c_signal.py
    python pipeline\\diagnostics\\measure_real_btc_roll_c_signal.py --snapshot-dir pipeline\\diagnostics\\kraken_snapshot_xbtusd_720h_2026-09-08
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(PIPELINE_DIR, '..'))
ORCH_DIR = os.path.join(PIPELINE_DIR, 'orchestration')

sys.path.insert(0, ORCH_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rebuild import build_bars_and_labels                       # noqa: E402
from features import build_enriched_events                       # noqa: E402

DEFAULT_SNAPSHOT_DIR = os.path.join(
    HERE, 'kraken_snapshot_xbtusd_720h_2026-09-08'
)
TARGET_BARS = 1000  # current production default -- matches the closed
                     # BTC diagnostics (Ch11-15) and the meanshift
                     # investigation's own scale, for a fair comparison


def corr_report(feature_table, target, target_name):
    """Identical logic to trace_meanshift_signal_leakage.py's corr_report
    -- reused, not reimplemented, so this measurement is directly
    comparable to that script's Stage 2 output."""
    print(f'\n  --- feature correlation vs. {target_name} ---')
    common = feature_table.index.intersection(target.index)
    if len(common) < 3:
        print(f'    (too few overlapping rows: {len(common)})')
        return []
    ft = feature_table.loc[common]
    tgt = target.loc[common].astype(float)
    rows = []
    for col in ft.columns:
        x = ft[col].astype(float)
        mask = x.notna() & tgt.notna()
        if mask.sum() < 3 or x[mask].std() == 0 or tgt[mask].std() == 0:
            rows.append((col, np.nan, int(mask.sum())))
            continue
        r = np.corrcoef(x[mask], tgt[mask])[0, 1]
        rows.append((col, r, int(mask.sum())))
    rows.sort(
        key=lambda t: (t[1] is not None and not np.isnan(t[1]),
                        abs(t[1]) if t[1] == t[1] else -1),
        reverse=True,
    )
    for col, r, n in rows:
        r_str = f'{r:+.4f}' if r == r else '  NaN '
        print(f'    {col:<32s} r={r_str}   (n={n})')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot-dir', type=str,
                         default=DEFAULT_SNAPSHOT_DIR,
                         help='Directory containing raw_trades.parquet '
                              '(default: today\'s fresh 720h Kraken '
                              'XBTUSD snapshot).')
    args = parser.parse_args()

    raw_path = os.path.join(args.snapshot_dir, 'raw_trades.parquet')
    if not os.path.exists(raw_path):
        raise SystemExit(
            f'{raw_path} not found -- run capture_kraken_snapshot.py '
            '--pair XBTUSD --hours 720 first.'
        )

    print('=' * 78)
    print(f'REAL BTC SIGNAL MEASUREMENT: {args.snapshot_dir}')
    print(f'target_bars={TARGET_BARS}, CUSUM_H=313 (confirmed OK for '
          f'Kraken, 2026-09-08 -- see audit_kraken_cusum_h_staleness.py)')
    print('=' * 78)

    raw_trades = pd.read_parquet(raw_path)
    print(f'\nLoaded {len(raw_trades)} raw trades')

    # --- Real bars + real next-bar return (identical to the meanshift
    # trace's Stage 2, but on real trades) ---
    rebuild_result = build_bars_and_labels(raw_trades, target_bars=TARGET_BARS)
    close = rebuild_result['close']
    next_ret = close.pct_change().shift(-1)
    print(f'\n[Bars] n_bars={len(close)}')

    # --- Real feature table (Ch19's 11 features + fracdiff + roll_c) ---
    enriched_result = build_enriched_events(
        raw_trades, rebuild_result['threshold'], rebuild_result['events'],
    )
    feature_table = enriched_result['feature_table']
    print(f'\n[Feature table] {feature_table.shape[0]} bars x '
          f'{feature_table.shape[1]} features: {list(feature_table.columns)}')

    rows = corr_report(feature_table, next_ret, 'next bar raw return (REAL BTC)')

    roll_c_row = next((r for r in rows if r[0] == 'roll_c'), None)
    print('\n' + '=' * 78)
    print('SUMMARY')
    print('=' * 78)
    if roll_c_row is not None and roll_c_row[1] == roll_c_row[1]:
        print(f'roll_c real bar-level correlation vs next-bar return: '
              f'r={roll_c_row[1]:+.4f}  (n={roll_c_row[2]})')
        print('\nFor comparison: the meanshift investigation\'s ORIGINAL '
              'leakage trace (2026-09-06, edge_strength=2.0/seed=0, '
              'synthetic-injected) measured roll_c at r=+0.2894 -- that '
              'was the single strongest bar-level feature correlation '
              'found anywhere in the eight-diagnostic chain, and it was '
              'STILL too weak for DSR to detect at single-window scale.')
    else:
        print('roll_c correlation could not be computed (see NaN above).')
    print('\nThis is a single real-data measurement, not yet mapped to '
          'the synthetic edge_strength ladder -- that mapping is separate, '
          'deliberately deferred follow-up work (see module docstring).')


if __name__ == '__main__':
    main()