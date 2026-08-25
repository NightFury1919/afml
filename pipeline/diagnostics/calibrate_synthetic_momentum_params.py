"""
pipeline/diagnostics/calibrate_synthetic_momentum_params.py

Purpose
-------
positive_control_data.generate_momentum_trades()'s price-generation
parameters (tick_bp=0.0005, noise_std=0.0003) were hand-picked on
2026-08-15 as "a reasoned starting point, NOT re-derived from first
principles" (see that module's own LOAD-BEARING docstring), calibrated
loosely against CUSUM_H=500. CUSUM_H moved to 313 on 2026-08-21
(rebuild.py) and the momentum generator was never revisited -- this
script closes that gap.

Method
------
Reuses audit_cusum_h_staleness.py's own precedent measurement exactly:
bar-to-bar CLOSE PRICE DIFF STD, the literal series CUSUM_H is applied
against inside rebuild.py -- not a raw-trade-level proxy, not CUSUM
firing rate directly (firing rate is reported too, as a confirmation
check, but the diff_std ratio is the actual calibration mechanism).

tick_bp and noise_std are scaled TOGETHER by a single dial `k`, not
independently retuned -- see 2026-08-24 handoff decision. A single
diff_std target under-determines two free parameters; scaling both by
one factor preserves the drift:noise ratio (signal-to-noise character
of a regime) while moving the overall scale to match real BTC
volatility at CUSUM_H=313.

k is computed via ONE closed-form ratio (real diff_std / synthetic
diff_std at k=1), not a bisection search -- valid because both
tick_bp and noise_std operate at basis-point-per-trade magnitudes,
where bar-to-bar cumulative diffs scale ~linearly with the per-trade
volatility inputs. Confirmed empirically below via a second pass at
the derived k, not just assumed.

generate_momentum_trades() requires continuation_prob in (0.5, 1.0] --
0.5001 is used as the closest achievable approximation to a true null
edge, since price DYNAMICS (not injected signal strength) is what this
script calibrates.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    $env:BINANCE_API_KEY = 'your-key-here'
    python pipeline\\diagnostics\\calibrate_synthetic_momentum_params.py --n-trades 360000 --target-bars 3000

Output
------
    pipeline/diagnostics/synthetic_momentum_baseline_params.json
"""
import argparse
import inspect
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
EDGE_HARNESS = os.path.join(HERE, '..', 'edge_harness')
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ORCH)
sys.path.insert(0, EDGE_HARNESS)
sys.path.insert(0, ROOT)

from ingestion import pull_recent_trades          # noqa: E402
import rebuild                                     # noqa: E402
from ch02.bars import filters as ch02_filters       # noqa: E402
from positive_control_data import generate_momentum_trades  # noqa: E402

LOOKBACK_HOURS = 720  # matches run_pipeline_live.py / audit_cusum_h_staleness.py

# LOAD-BEARING (2026-08-24): pulled LIVE from generate_momentum_trades()'s
# own function signature, not hardcoded as duplicate literals -- avoids the
# exact silent-drift failure mode this project has hit before (default-arg
# values diverging from the source of truth without anything erroring).
_momentum_defaults = inspect.signature(generate_momentum_trades).parameters
BASE_TICK_BP = _momentum_defaults['tick_bp'].default
BASE_NOISE_STD = _momentum_defaults['noise_std'].default


def bar_close_diff_stats(raw_trades, target_bars, label):
    """Mirrors audit_cusum_h_staleness.py's bar_close_diff_stats() exactly
    -- same real rebuild.py chain, same statistic -- so this calibration
    is directly comparable to the existing CUSUM_H staleness audit's
    methodology, not a new ad hoc measurement."""
    result = rebuild.build_bars_and_labels(raw_trades, target_bars=target_bars)
    close = result['close']
    diffs = close.diff().dropna()

    cusum_df = pd.DataFrame({'Date': close.index, 'Price': close.values})
    events_at_h = ch02_filters.cusum_filter(cusum_df, h=rebuild.CUSUM_H)

    print(f'\n--- {label} ---')
    print(f'  n_bars: {len(close)}')
    print(f'  bar-to-bar close diff std: ${diffs.std():,.2f}')
    print(f'  CUSUM events at h={rebuild.CUSUM_H}: {len(events_at_h)} '
          f'({len(events_at_h) / len(close):.1%} of bars)')
    return {
        'n_bars': len(close),
        'diff_std': float(diffs.std()),
        'n_events': len(events_at_h),
        'firing_rate': len(events_at_h) / len(close),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trades', type=int, default=360_000,
                         help='Matches the corrected momentum sweep scale '
                              '(2026-08-24 handoff) -- must match whatever '
                              'run_momentum_edge_sweep.py actually uses.')
    parser.add_argument('--target-bars', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get('BINANCE_API_KEY')
    if not api_key:
        raise SystemExit(
            "BINANCE_API_KEY is not set -- see ingestion.py's docstring "
            "for how to get a free read-only key."
        )

    print(f'Pulling last {LOOKBACK_HOURS}h of live BTCUSDT trades from Binance.US...')
    live_trades = pull_recent_trades('BTCUSDT', LOOKBACK_HOURS, api_key)
    print(f'  {len(live_trades)} raw trades pulled')
    real_stats = bar_close_diff_stats(
        live_trades, args.target_bars, 'Live BTC/USDT (today)'
    )

    print(f'\nGenerating null-edge momentum tape at k=1.0 (base tick_bp='
          f'{BASE_TICK_BP}, noise_std={BASE_NOISE_STD}) at the SAME '
          'n_trades/target_bars scale, to measure synthetic diff_std...')
    null_edge = generate_momentum_trades(
        n_trades=args.n_trades,
        continuation_prob=0.5001,
        tick_bp=BASE_TICK_BP,
        noise_std=BASE_NOISE_STD,
        total_span_hours=LOOKBACK_HOURS,
        random_state=args.seed,
    )['raw_trades']
    synth_stats = bar_close_diff_stats(
        null_edge, args.target_bars,
        f'Synthetic null-edge (k=1.0, n_trades={args.n_trades})'
    )

    # Single closed-form ratio -- see module docstring for why this
    # doesn't need a bisection search.
    k = real_stats['diff_std'] / synth_stats['diff_std']
    calibrated_tick_bp = BASE_TICK_BP * k
    calibrated_noise_std = BASE_NOISE_STD * k

    print('\n=== Calibration result ===')
    print(f'  k = {k:.4f}')
    print(f'  calibrated tick_bp   = {calibrated_tick_bp:.6f} (was {BASE_TICK_BP})')
    print(f'  calibrated noise_std = {calibrated_noise_std:.6f} (was {BASE_NOISE_STD})')

    print(f'\nConfirming: regenerating null-edge tape at k={k:.4f} and '
          're-measuring against the real target (empirical check, not '
          'just assuming linearity held)...')
    confirm = generate_momentum_trades(
        n_trades=args.n_trades,
        continuation_prob=0.5001,
        tick_bp=calibrated_tick_bp,
        noise_std=calibrated_noise_std,
        total_span_hours=LOOKBACK_HOURS,
        random_state=args.seed,
    )['raw_trades']
    confirm_stats = bar_close_diff_stats(
        confirm, args.target_bars,
        f'Synthetic null-edge (k={k:.4f}, confirmation pass)'
    )

    print(f'\nFiring rate -- real: {real_stats["firing_rate"]:.1%}, '
          f'synthetic (calibrated): {confirm_stats["firing_rate"]:.1%}')
    residual_gap = abs(confirm_stats['diff_std'] - real_stats['diff_std']) / real_stats['diff_std']
    print(f'Residual diff_std gap after calibration: {residual_gap:.1%} '
          '(should be small if the linear-scaling assumption held; if '
          'large, the k=1.0 -> confirmation pass moved further from the '
          'real target than expected and the linearity assumption needs '
          'revisiting, not just accepting this k).')

    out = {
        'n_trades': args.n_trades,
        'target_bars': args.target_bars,
        'seed': args.seed,
        'base_tick_bp': BASE_TICK_BP,
        'base_noise_std': BASE_NOISE_STD,
        'k': k,
        'calibrated_tick_bp': calibrated_tick_bp,
        'calibrated_noise_std': calibrated_noise_std,
        'real_diff_std': real_stats['diff_std'],
        'real_firing_rate': real_stats['firing_rate'],
        'synthetic_diff_std_at_k1': synth_stats['diff_std'],
        'synthetic_firing_rate_at_k1': synth_stats['firing_rate'],
        'confirm_diff_std': confirm_stats['diff_std'],
        'confirm_firing_rate': confirm_stats['firing_rate'],
        'residual_diff_std_gap': residual_gap,
    }
    out_path = os.path.join(HERE, 'synthetic_momentum_baseline_params.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote calibrated params to {out_path}')


if __name__ == '__main__':
    main()
