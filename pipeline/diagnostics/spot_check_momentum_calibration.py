"""
pipeline/diagnostics/spot_check_momentum_calibration.py

Purpose
-------
calibrate_synthetic_momentum_params.py derived k from a SINGLE seed
(seed=0). This spot-checks whether the resulting calibrated_tick_bp/
calibrated_noise_std reproduce a similar CUSUM firing rate across a
couple more seeds -- same pattern as the 2026-08-23 momentum-correlation
mapping's own "real-machine spot-check before trusting" precedent.

Deliberately does NOT re-pull live BTC data -- reuses the real firing
rate/diff_std already measured and written to
synthetic_momentum_baseline_params.json by the calibration script. A
fresh live pull would introduce day-to-day BTC drift as a confound; what
this script tests is the SYNTHETIC generator's seed-to-seed stability at
a fixed k, not whether real BTC has moved since the calibration run.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\spot_check_momentum_calibration.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ORCH)
sys.path.insert(0, HERE)

from positive_control_data import generate_momentum_trades   # noqa: E402
from calibrate_synthetic_momentum_params import (             # noqa: E402
    bar_close_diff_stats, LOOKBACK_HOURS,
)

SPOT_CHECK_SEEDS = [1, 2, 3]  # deliberately different from the
                               # calibration run's seed=0


def main():
    params_path = os.path.join(HERE, 'synthetic_momentum_baseline_params.json')
    if not os.path.exists(params_path):
        raise SystemExit(
            f'{params_path} not found -- run '
            'calibrate_synthetic_momentum_params.py first.'
        )
    with open(params_path) as f:
        params = json.load(f)

    real_firing_rate = params['real_firing_rate']
    real_diff_std = params['real_diff_std']
    k = params['k']
    tick_bp = params['calibrated_tick_bp']
    noise_std = params['calibrated_noise_std']
    n_trades = params['n_trades']
    target_bars = params['target_bars']

    print(f'Spot-checking k={k:.4f} (calibrated_tick_bp={tick_bp:.6f}, '
          f'calibrated_noise_std={noise_std:.6f}) across seeds '
          f'{SPOT_CHECK_SEEDS} -- against cached real target: '
          f'firing_rate={real_firing_rate:.1%}, diff_std=${real_diff_std:,.2f} '
          f'(from the calibration run, not re-pulled here)')

    results = []
    for seed in SPOT_CHECK_SEEDS:
        tape = generate_momentum_trades(
            n_trades=n_trades,
            continuation_prob=0.5001,
            tick_bp=tick_bp,
            noise_std=noise_std,
            total_span_hours=LOOKBACK_HOURS,
            random_state=seed,
        )['raw_trades']
        stats = bar_close_diff_stats(
            tape, target_bars, f'Synthetic null-edge (seed={seed})'
        )
        results.append({
            'seed': seed,
            'firing_rate': stats['firing_rate'],
            'diff_std': stats['diff_std'],
        })

    print('\n=== Spot-check summary ===')
    print(f'{"seed":>6} {"firing_rate":>14} {"diff_std":>12}')
    print(f'{"real":>6} {real_firing_rate:>13.1%} {real_diff_std:>11,.2f}')
    for r in results:
        print(f'{r["seed"]:>6} {r["firing_rate"]:>13.1%} {r["diff_std"]:>11,.2f}')

    firing_rates = [r['firing_rate'] for r in results]
    fr_spread = max(firing_rates) - min(firing_rates)
    print(f'\nFiring rate spread across seeds {SPOT_CHECK_SEEDS}: '
          f'{fr_spread:.1%} (min={min(firing_rates):.1%}, '
          f'max={max(firing_rates):.1%})')
    print('If this spread is small and all seeds stay reasonably close to '
          f'the real target ({real_firing_rate:.1%}), the k=0 seed used '
          'for calibration was not a lucky/unlucky outlier and k can be '
          'trusted. If any seed diverges sharply, the single-seed k needs '
          'a proper multi-seed re-derivation (e.g. averaging diff_std '
          'across several null-edge seeds before computing k), not just '
          'accepting the seed=0 value.')


if __name__ == '__main__':
    main()
