"""
pipeline/diagnostics/calibrate_kraken_detection_power.py

Kraken counterpart to calibrate_detection_power.py (2026-08-19's real
methodology, reused directly -- N=20 trials matching production C_GRID
x STEP_GRID, real ch14.deflated_sharpe_ratio(), Gaussian + fat-tailed
jump-mixture regimes). That script answered: at Binance.US's real
observed T_effective range, would DSR actually detect a real edge if
one existed, and what does its null-hypothesis false-positive rate look
like at that T? This script asks the SAME question at Kraken's real
observed T_effective range -- the piece needed to finally interpret
today's Kraken sweep's DSR readings (0.55-0.68, T_effective 123-758),
which have sat uninterpretable pending exactly this calibration (see
CALIBRATION_AUDIT.md's "Kraken Evaluated as a Higher-Density Data
Source" section, "why full production replacement is deferred").

SCOPE REDUCTION vs. the original script (deliberate, not a precision
cut): T_GRID here is ONLY the 5 real T_effective values Kraken's own
2026-08-25 target_bars sweep actually produced (123.38, 277.33, 402.45,
588.46, 758.19), not the original's broader 11-point canonical grid --
the question here is specifically "how do I read TODAY's five real
numbers," not a general power curve. n_reps stays at 20,000 (unchanged
precision) -- with 5 T-points instead of 11, this script's total
simulation cost is proportionally smaller.

TRUE_SHARPE_GRID and N_REPS are unchanged from the original script for
direct comparability.

FAT-TAILED REGIME: calibrated to KRAKEN's own real observed skew/
kurtosis (read directly from kraken_target_bars_calibration.csv's
skew/kurtosis columns -- the actual winning-trial values from today's
five real sweep runs), NOT Binance.US's 2026-08-19 values. Uses the
same calibrate_jump_mixture() approach as the original script --
reused, not reimplemented -- just re-tuned (p_jump/jump_size grid-
searched over a small range) to land in the neighborhood of Kraken's
own observed range, same "neighborhood match, not exact fit" caveat the
original script documents.

Estimated runtime: proportionally less than the original's ~1 hour
(5/11 of the T-grid, roughly ~25-30 minutes) -- still budget generously,
same caution as every other Monte Carlo calibration in this project.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_kraken_detection_power.py
    python pipeline\\diagnostics\\calibrate_kraken_detection_power.py <sweep_csv> [output_csv]

ADDED 2026-08-25: optional <sweep_csv>/[output_csv] args for running
against a SECOND window's sweep results (replication check) without
overwriting the first window's kraken_detection_power_calibration.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'ch14', 'backtest_statistics'))

from backtest_statistics import deflated_sharpe_ratio  # noqa: E402, real ch14 module

N_TRIALS = 20
N_REPS = 20_000

KRAKEN_SWEEP_CSV = os.path.join(HERE, 'kraken_target_bars_calibration.csv')

# Real T_effective values from today's Kraken target_bars sweep (2026-08-25)
# -- see module docstring. Read directly from the CSV at runtime (not
# hardcoded independently) so this script can never silently drift from
# the actual sweep results it's meant to interpret.
TRUE_SHARPE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_kraken_sweep_stats(csv_path=None):
    """Reads a Kraken sweep's T_effective and skew/kurtosis columns
    directly -- this script's T grid and fat-tailed regime calibration
    are BOTH derived from this file, not independently guessed, so they
    can never silently drift from the sweep they exist to interpret.

    csv_path : str or None, ADDED 2026-08-25 for the second-window
    replication check -- defaults to KRAKEN_SWEEP_CSV (the original
    window's results) if not given."""
    path = csv_path if csv_path is not None else KRAKEN_SWEEP_CSV
    if not os.path.exists(path):
        raise SystemExit(
            f'{path} not found -- run calibrate_kraken_target_bars.py first.'
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=['T_effective'])
    if df.empty:
        raise SystemExit(f'{path} has no successful rows to read.')
    return df


def simulate_power_gaussian(T, true_sharpe, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Identical to the original script's real function -- i.i.d.
    standard-normal trials, one given a real population Sharpe via a
    mean shift, skew=0/kurtosis=3 fed to deflated_sharpe_ratio()."""
    rng = np.random.default_rng(seed)
    dsrs = np.empty(n_reps)
    correct_selection = np.empty(n_reps, dtype=bool)
    T_int = int(round(T))
    for i in range(n_reps):
        pnl = rng.standard_normal(size=(T_int, N))
        pnl[:, 0] += true_sharpe
        sharpes = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
        best_idx = sharpes.argmax()
        sr_hat = sharpes[best_idx]
        var_sr_trials = sharpes.var(ddof=1)
        dsrs[i] = deflated_sharpe_ratio(sr_hat, var_sr_trials, N, T, skew=0., kurtosis=3.)
        correct_selection[i] = (best_idx == 0)
    return dsrs, correct_selection


def calibrate_jump_mixture(p, s, tilt=0.0, n=200_000, seed=0):
    """*** REPLACED (2026-08-25, second pass): single-direction jump
    mixture swapped for a symmetric DOUBLE-SIDED jump mixture ***
    The original single-direction version (`x[jumps] -= jump_size`,
    even after allowing jump_size to go negative) structurally cannot
    hit a near-zero-skew + high-kurtosis target simultaneously -- a
    single jump component always biases the distribution's mass toward
    one side, so kurtosis and skew move together in that family, not
    independently. Kraken's real observed target (skew=-0.0014,
    kurtosis=7.0104, from the sweep's actual mean) is almost exactly
    this hard case. Confirmed via direct testing before this rewrite:
    the single-direction version's best achievable match to this exact
    target was skew=0.7959 (should be ~0), a real, meaningful miss.

    This version uses INDEPENDENT positive and negative jump
    probabilities (p_pos = p*(1-tilt), p_neg = p*(1+tilt)) at the SAME
    jump size `s` on both sides. tilt=0 gives an EXACTLY symmetric
    mixture -- population skew=0 by construction, not by search luck --
    while kurtosis is tuned via p and s together. Non-zero tilt biases
    toward one side for future re-runs where the real target skew isn't
    close to zero (this project's own convention: general enough to
    keep working correctly if tomorrow's real Kraken data looks
    different, not hardcoded to today's specific near-zero-skew case).

    Confirmed via direct testing: for today's real target
    (skew=-0.0014, kurtosis=7.0104), this family's best match came in
    at skew=-0.0084, kurtosis=7.2754 -- both within a few percent of
    target, a real improvement over the prior version's skew=0.7959
    miss.

    n=200,000 for the GRID SEARCH phase (same speed reasoning as the
    prior version); the chosen (p, s, tilt) is re-calibrated once at
    n=2,000,000 before use in simulate_power_fat_tailed() (see
    find_jump_mixture_for_kraken).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    p_pos = p * (1 - tilt)
    p_neg = p * (1 + tilt)
    draws = rng.random(n)
    pos_jumps = draws < p_pos
    neg_jumps = (draws >= p_pos) & (draws < p_pos + p_neg)
    x = x.copy()
    x[pos_jumps] += s
    x[neg_jumps] -= s
    return {
        'p': p, 's': s, 'tilt': tilt,
        'mean0': x.mean(), 'std0': x.std(ddof=1),
        'skew0': stats.skew(x), 'kurtosis0': stats.kurtosis(x, fisher=False),
    }


def find_jump_mixture_for_kraken(target_skew, target_kurtosis):
    """Grid search over (p, s, tilt) for the double-sided jump mixture
    (see calibrate_jump_mixture's own note on why this replaced the
    original single-direction version). tilt=0 is included in the grid
    so an exactly-symmetric target (like today's real Kraken data) gets
    an exact-by-construction skew=0 match, not a searched approximation."""
    best = None
    best_dist = np.inf
    for p in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for s in [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]:
            for tilt in [-0.6, -0.3, 0.0, 0.3, 0.6]:
                mix = calibrate_jump_mixture(p, s, tilt)
                dist = (
                    (mix['skew0'] - target_skew) ** 2
                    + ((mix['kurtosis0'] - target_kurtosis) / 3.0) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best = mix
    # Re-calibrate the CHOSEN (p, s, tilt) at full precision.
    best = calibrate_jump_mixture(best['p'], best['s'], best['tilt'], n=2_000_000)
    return best


def simulate_power_fat_tailed(T, true_sharpe, mix, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Same design as the original script's real function -- fat-tailed
    jump-mixture returns, REALIZED skew/kurtosis from the selected best
    trial's own sample fed to deflated_sharpe_ratio(), mirroring
    stages.py's evaluate_overfitting() convention exactly.

    Updated (2026-08-25, second pass) for the double-sided jump mixture
    -- two independent jump draws per element (positive at rate p_pos,
    negative at rate p_neg) instead of the original's single-direction
    jump, matching calibrate_jump_mixture's own updated construction."""
    rng = np.random.default_rng(seed)
    dsrs = np.empty(n_reps)
    correct_selection = np.empty(n_reps, dtype=bool)
    T_int = int(round(T))
    p, s, tilt = mix['p'], mix['s'], mix['tilt']
    p_pos, p_neg = p * (1 - tilt), p * (1 + tilt)
    mean0, std0 = mix['mean0'], mix['std0']

    shift_noise = -mean0
    shift_edge = true_sharpe * std0 - mean0

    for i in range(n_reps):
        base = rng.standard_normal(size=(T_int, N))
        draws = rng.random(size=(T_int, N))
        pos_jumps = draws < p_pos
        neg_jumps = (draws >= p_pos) & (draws < p_pos + p_neg)
        base[pos_jumps] += s
        base[neg_jumps] -= s
        base[:, 1:] += shift_noise
        base[:, 0] += shift_edge

        sharpes = base.mean(axis=0) / base.std(axis=0, ddof=1)
        best_idx = sharpes.argmax()
        sr_hat = sharpes[best_idx]
        var_sr_trials = sharpes.var(ddof=1)

        best_col = base[:, best_idx]
        if T_int > 2:
            realized_skew = float(stats.skew(best_col))
            realized_kurtosis = float(stats.kurtosis(best_col, fisher=False))
        else:
            realized_skew, realized_kurtosis = 0., 3.

        dsrs[i] = deflated_sharpe_ratio(
            sr_hat, var_sr_trials, N, T, skew=realized_skew, kurtosis=realized_kurtosis,
        )
        correct_selection[i] = (best_idx == 0)
    return dsrs, correct_selection


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    sweep_df = load_kraken_sweep_stats(csv_path)
    t_grid = sorted(sweep_df['T_effective'].round(2).tolist())
    real_dsr = dict(zip(sweep_df['T_effective'].round(2), sweep_df['dsr']))
    target_skew = float(sweep_df['skew'].mean())
    target_kurtosis = float(sweep_df['kurtosis'].mean())

    actual_csv = csv_path if csv_path is not None else KRAKEN_SWEEP_CSV
    print(f'Read {len(sweep_df)} real Kraken sweep rows from {actual_csv}')
    print(f'T_effective grid (real, from the sweep): {t_grid}')
    print(f'Target fat-tailed regime (mean of real observed values): '
          f'skew={target_skew:.4f}, kurtosis={target_kurtosis:.4f}')

    print('\n' + '=' * 74)
    print('PART 1 -- detection power, Gaussian returns (skew=0, kurtosis=3)')
    print('=' * 74)
    rows1 = []
    for true_sharpe in TRUE_SHARPE_GRID:
        for T in t_grid:
            dsrs, correct = simulate_power_gaussian(T, true_sharpe)
            rows1.append({
                'true_sharpe': true_sharpe, 'T': T,
                'P[DSR>0.5]': (dsrs > 0.5).mean(),
                'P[correct_selection]': correct.mean(),
                'mean_dsr': dsrs.mean(),
            })
    df1 = pd.DataFrame(rows1)
    pivot1 = df1.pivot(index='T', columns='true_sharpe', values='mean_dsr')
    print('\n  mean_dsr (Gaussian regime) by T (rows) x true_sharpe (cols):\n')
    print(pivot1.round(4).to_string())

    print('\n' + '=' * 74)
    print('PART 2 -- detection power, fat-tailed returns (Kraken-calibrated jump-mixture)')
    print('=' * 74)
    mix = find_jump_mixture_for_kraken(target_skew, target_kurtosis)
    print(f"\n  jump-mixture calibration: p={mix['p']}, s={mix['s']}, "
          f"tilt={mix['tilt']}\n"
          f"  achieved (unshifted) skew={mix['skew0']:.4f}, "
          f"kurtosis={mix['kurtosis0']:.4f}\n"
          f"  (target, from Kraken's real sweep: skew={target_skew:.4f}, "
          f"kurtosis={target_kurtosis:.4f} -- neighborhood match, not "
          f"exact, see module docstring)\n")

    rows2 = []
    for true_sharpe in TRUE_SHARPE_GRID:
        for T in t_grid:
            dsrs, correct = simulate_power_fat_tailed(T, true_sharpe, mix)
            rows2.append({
                'true_sharpe': true_sharpe, 'T': T,
                'P[DSR>0.5]': (dsrs > 0.5).mean(),
                'P[correct_selection]': correct.mean(),
                'mean_dsr': dsrs.mean(),
            })
    df2 = pd.DataFrame(rows2)
    pivot2 = df2.pivot(index='T', columns='true_sharpe', values='mean_dsr')
    print('\n  mean_dsr (fat-tailed regime) by T (rows) x true_sharpe (cols):\n')
    print(pivot2.round(4).to_string())

    out_path = output_csv if output_csv is not None else os.path.join(
        HERE, 'kraken_detection_power_calibration.csv'
    )
    df1['regime'] = 'gaussian'
    df2['regime'] = 'fat_tailed'
    pd.concat([df1, df2], ignore_index=True).to_csv(out_path, index=False)
    print(f'\nFull result grid written to {out_path}')

    print('\n' + '=' * 74)
    print('READING TODAY\'S REAL KRAKEN DSR VALUES AGAINST THIS CALIBRATION')
    print('=' * 74)
    null_fat_tailed = pivot2[0.0]
    for T in t_grid:
        observed = real_dsr.get(T)
        null_here = null_fat_tailed.get(T)
        if observed is None or null_here is None:
            continue
        gap = observed - null_here
        print(f'  T={T:.2f}: real observed DSR={observed:.4f}, '
              f'fat-tailed null baseline={null_here:.4f}, '
              f'gap={gap:+.4f}')
    print("""
INTERPRETATION:
  - "gap" is the real observed DSR minus what a GENUINELY ZERO-EDGE
    strategy would be expected to read at that same T, in a return
    regime shaped like Kraken's own real data. A gap near 0 means
    today's DSR readings are indistinguishable from noise at this T --
    consistent with "no edge detected," not "edge ruled out." A gap
    clearly positive and growing with T would be a real signal worth
    taking seriously; compare its size against the true_sharpe columns
    in the fat-tailed pivot table above to get a rough sense of what
    edge size (if any) it corresponds to.
  - This calibration is a NEIGHBORHOOD match to Kraken's real skew/
    kurtosis, not an exact distributional fit -- same caveat the
    original Binance.US version carries.
  - This answers "how should I read the DSR numbers I already have,"
    not "is there an edge" -- same distinction the original detection-
    power calibration drew for Binance.US.
""")


if __name__ == '__main__':
    main()
