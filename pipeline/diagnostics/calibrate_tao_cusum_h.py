"""
pipeline/diagnostics/calibrate_tao_cusum_h.py

TAO counterpart to calibrate_xrp_cusum_h.py -- same real methodology
(measure bar-to-bar close-diff std for BTC and the new asset via the
real dollar_bars() function, bypassing cusum_filter entirely, then
ratio-scale BTC's currently-adopted CUSUM_H by the two diff_stds),
applied to TAO instead of XRP.

WHY THIS IS NEEDED, NOT OPTIONAL: CUSUM_H=313 is a flat DOLLAR
threshold calibrated to BTC's ~$70,000 price level. TAO trades around
$190-350 (see tonight's real scout output) -- an entirely different
scale from both BTC and XRP ($0.50-3). Reusing CUSUM_H=313 unchanged
would either never fire or crash build_bars_and_labels() outright, the
same problem XRP had -- confirmed by the same code-inspection reasoning
calibrate_xrp_cusum_h.py's docstring lays out (build_bars_and_labels()
calls cusum_filter() internally and raises ValueError on zero events,
so the real pipeline can't even be used to measure TAO's own
price-move distribution without a candidate H already in hand -- hence
this script calling the lower-level bar-construction functions
directly, bypassing only the CUSUM step, exactly like the XRP version).

METHOD: pulls a short, fresh sample of BOTH XBTUSD and TAOUSD (same
window, same day, same target_bars) via Kraken, builds bars via the
real dollar_bars() function for each (bypassing CUSUM), measures each
asset's bar-to-bar close-diff distribution, then scales CUSUM_H's
currently-adopted BTC value by the ratio of the two diff_stds -- same
ratio-scaling logic as the XRP version, reused unmodified except for
the asset pair.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_tao_cusum_h.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ORCH)
sys.path.insert(0, ROOT)

from ingestion_kraken import pull_recent_trades_kraken   # noqa: E402
import rebuild                                            # noqa: E402
from ch02.bars import standard_bars                        # noqa: E402

CALIBRATION_HOURS = 24.0   # short, fresh sample -- a calibration
                            # measurement doesn't need the full 720h
                            # dataset, just a stable diff_std estimate
TARGET_BARS = 1000          # matches production default
CURRENT_BTC_CUSUM_H = 313   # rebuild.py's current, real, adopted value


def bar_close_diffs_no_cusum(raw_trades, target_bars, label):
    """Real bars via the real dollar_bars() function, WITHOUT ever
    calling cusum_filter -- see module docstring on why
    build_bars_and_labels() can't be used here for a new, uncalibrated
    asset. preprocess_raw_trades()/compute_dynamic_threshold() are
    rebuild.py's real functions, reused unmodified."""
    df = rebuild.preprocess_raw_trades(raw_trades)
    threshold = rebuild.compute_dynamic_threshold(raw_trades, target_bars=target_bars)
    bars = standard_bars.dollar_bars(df, thresh=threshold)
    if bars.empty:
        raise ValueError(f'{label}: dollar_bars() produced zero bars.')
    bars = bars.set_index('Date')
    close = bars['Close']
    diffs = close.diff().dropna()

    print(f'\n--- {label} ---')
    print(f'  n_bars: {len(close)}')
    print(f'  bar close price: mean=${close.mean():,.6f}  std=${close.std():,.6f}')
    print(f'  bar-to-bar close diff: mean|d|=${diffs.abs().mean():,.6f}  '
          f'std=${diffs.std():,.6f}')
    return {'close': close, 'diffs': diffs, 'diff_std': diffs.std()}


def main():
    print(f'Pulling {CALIBRATION_HOURS}h of XBTUSD (reference) and TAOUSD '
          f'(new asset) from Kraken...')

    btc_trades = pull_recent_trades_kraken('XBTUSD', CALIBRATION_HOURS)
    print(f'  BTC: {len(btc_trades)} raw trades pulled')
    tao_trades = pull_recent_trades_kraken('TAOUSD', CALIBRATION_HOURS)
    print(f'  TAO: {len(tao_trades)} raw trades pulled')

    btc_stats = bar_close_diffs_no_cusum(btc_trades, TARGET_BARS, 'XBTUSD (reference)')
    tao_stats = bar_close_diffs_no_cusum(tao_trades, TARGET_BARS, 'TAOUSD (new asset)')

    ratio = tao_stats['diff_std'] / btc_stats['diff_std']
    h_candidate = CURRENT_BTC_CUSUM_H * ratio

    print('\n=== TAO CUSUM_H derivation ===')
    print(f'TAO/BTC bar-to-bar diff std ratio: {ratio:.8f}x')
    print(f'Current BTC CUSUM_H: {CURRENT_BTC_CUSUM_H}')
    print(f'Candidate TAO CUSUM_H (ratio-scaled): {h_candidate:.6f}')
    print("""
This is a MEASUREMENT-DERIVED STARTING POINT, not a validated final
calibration -- same caution as every other h_equivalent measurement in
this project. Before trusting it:
  1. Sanity-check by actually calling ch02_filters.cusum_filter() at
     this candidate h on the TAO bar series above and confirming a
     reasonable event rate results (comparable to BTC's own ~28-30%
     of bars at h=313) -- not zero, not every bar.
  2. This was measured on a SHORT (24h) sample -- same single-pull
     caveat as every other calibration measurement in this project;
     worth a second check on a different day before treating it as
     final.
  3. MIN_RET (0.005, a FRACTIONAL/percentage threshold) does NOT need
     this same re-derivation -- it's already asset-price-level-
     invariant by construction, unlike CUSUM_H's flat dollar design.
  4. TAO's real observed trade density is meaningfully lower than
     BTC's or XRP's (confirmed 308.3 trades/hour on tonight's real
     720h capture, declining across all three of tonight's readings:
     474.9/hr at 6h scout, 401.4/hr at 24h scout, 308.3/hr on the full
     720h pull) -- a real, separate risk from CUSUM_H's own validity,
     flagged here so it isn't lost by the time the target_bars sweep
     runs.
""")

    out_path = os.path.join(HERE, 'tao_cusum_h_calibration.csv')
    pd.DataFrame([{
        'asset': 'TAOUSD', 'reference_asset': 'XBTUSD',
        'btc_diff_std': btc_stats['diff_std'], 'tao_diff_std': tao_stats['diff_std'],
        'ratio': ratio, 'current_btc_cusum_h': CURRENT_BTC_CUSUM_H,
        'candidate_tao_cusum_h': h_candidate, 'calibration_hours': CALIBRATION_HOURS,
        'target_bars': TARGET_BARS,
    }]).to_csv(out_path, index=False)
    print(f'Result written to {out_path}')


if __name__ == '__main__':
    main()
