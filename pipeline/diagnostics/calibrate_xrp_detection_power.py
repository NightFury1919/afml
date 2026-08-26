"""
pipeline/diagnostics/calibrate_xrp_detection_power.py

XRP counterpart to calibrate_kraken_detection_power.py (which itself
reused calibrate_detection_power.py's original 2026-08-19 methodology
for Binance.US -- N=20 trials matching production C_GRID x STEP_GRID,
real ch14.deflated_sharpe_ratio(), Gaussian + fat-tailed jump-mixture
regimes). The Kraken version answered "at Kraken's real observed
T_effective range, would DSR actually detect a real edge if one
existed, and what does its null-hypothesis false-positive rate look
like at that T?" -- and that calibration is exactly what turned
Kraken's ambiguous DSR readings into an interpretable gap (see the
2026-08-25/26 handoff, Part 4). This script asks the SAME question at
XRP's real observed T_effective range -- the piece needed to interpret
tonight's XRP target_bars sweep's DSR readings (0.46-0.71, T_effective
83.81-625.37), which are explicitly flagged as NOT yet trustworthy
pending exactly this calibration (see the handoff's Part 6).

SCOPE: T_GRID here is the 5 real T_effective values XRP's own
2026-08-25/26 target_bars sweep actually produced (83.81, 188.72,
331.05, 504.82, 625.37), read directly from xrp_target_bars_
calibration.csv -- not independently guessed, so this script's T grid
can never silently drift from the sweep it exists to interpret. Same
"answer today's five real numbers" scope reduction the Kraken version
used, for the same reason.

TRUE_SHARPE_GRID and N_REPS are unchanged from both prior scripts for
direct comparability.

FAT-TAILED REGIME: calibrated to XRP's own real observed skew/kurtosis
(read directly from xrp_target_bars_calibration.csv's skew/kurtosis
columns -- the actual winning-trial values from tonight's five real
sweep runs), NOT Binance.US's or Kraken BTC's values. XRP's observed
regime is notably fatter-tailed and less stably-signed than Kraken
BTC's (mean kurtosis ~11.9 vs Kraken BTC's ~7.0; skew ranges -0.43 to
+0.38 across the five rows rather than sitting near zero throughout),
so the achieved calibration match is reported explicitly rather than
assumed to land as cleanly as Kraken BTC's near-zero-skew case did.
Reuses calibrate_jump_mixture()/find_jump_mixture_for_kraken()'s real
double-sided grid-search machinery unmodified (it was already written
generally, not hardcoded to Kraken's specific target) -- only the
target skew/kurtosis fed into it changes.

Estimated runtime: same order as the Kraken version, ~25-30 minutes
(5-point T grid, N_REPS=20,000 x 2 regimes x 6 true_sharpe values).

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_xrp_detection_power.py
    python pipeline\\diagnostics\\calibrate_xrp_detection_power.py <sweep_csv> [output_csv]

Optional <sweep_csv>/[output_csv] args mirror the Kraken version's
2026-08-25 addition, for running against a second window's sweep
results (replication check) without overwriting the first window's
xrp_detection_power_calibration.csv -- kept even though that second
XRP window doesn't exist yet, so this script is ready for it the same
day it's built, per this project's own "the two checks that caught
BTC's false lead haven't been built for XRP yet" open item.
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

XRP_SWEEP_CSV = os.path.join(HERE, 'xrp_target_bars_calibration.csv')

# Real T_effective values from tonight's XRP target_bars sweep
# (2026-08-25/26) -- see module docstring. Read directly from the CSV
# at runtime (not hardcoded independently) so this script can never
# silently drift from the actual sweep results it's meant to interpret.
TRUE_SHARPE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_xrp_sweep_stats(csv_path=None):
    """Reads an XRP sweep's T_effective and skew/kurtosis columns
    directly -- this script's T grid and fat-tailed regime calibration
    are BOTH derived from this file, not independently guessed, so they
    can never silently drift from the sweep they exist to interpret.

    csv_path : str or None -- defaults to XRP_SWEEP_CSV (this first
    window's results) if not given, mirroring the Kraken script's
    second-window replication hook."""
    path = csv_path if csv_path is not None else XRP_SWEEP_CSV
    if not os.path.exists(path):
        raise SystemExit(
            f'{path} not found -- run calibrate_xrp_target_bars.py first.'
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=['T_effective'])
    if df.empty:
        raise SystemExit(f'{path} has no successful rows to read.')
    return df


def simulate_power_gaussian(T, true_sharpe, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Identical to the Kraken/original script's real function -- i.i.d.
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
    """Reused UNMODIFIED from calibrate_kraken_detection_power.py's
    2026-08-25 double-sided jump mixture (see that script's own note on
    why the single-direction version was replaced -- it structurally
    cannot hit a near-zero-skew + high-kurtosis target simultaneously).
    Kept general on purpose: independent positive/negative jump
    probabilities (p_pos = p*(1-tilt), p_neg = p*(1+tilt)) at the same
    jump size `s`, so tilt != 0 can bias toward one side -- exactly what
    XRP's real target (skew ranging -0.43 to +0.38 across the sweep's
    five rows, not settled near zero the way Kraken BTC's was) may
    require, unlike Kraken BTC's exact-symmetric tilt=0 case."""
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


def find_jump_mixture_for_target(target_skew, target_kurtosis):
    """Reused UNMODIFIED grid search over (p, s, tilt) from the Kraken
    script (there named find_jump_mixture_for_kraken -- renamed here
    only for clarity since it was already asset-general, not
    Kraken-specific). Grid unchanged: XRP's kurtosis target (~11.9 mean,
    up to 19.2 on individual rows) is well outside this grid's easiest
    reach (max achievable kurtosis is bounded by the p/s grid's extremes),
    so the achieved match is reported explicitly below rather than
    assumed -- same 'neighborhood match, not exact fit' caveat both
    prior scripts carry, but worth flagging XRP may land further off
    than Kraken BTC did."""
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
    """Reused UNMODIFIED from the Kraken script -- fat-tailed jump-
    mixture returns, REALIZED skew/kurtosis from the selected best
    trial's own sample fed to deflated_sharpe_ratio(), mirroring
    stages.py's evaluate_overfitting() convention exactly."""
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

    sweep_df = load_xrp_sweep_stats(csv_path)
    t_grid = sorted(sweep_df['T_effective'].round(2).tolist())
    real_dsr = dict(zip(sweep_df['T_effective'].round(2), sweep_df['dsr']))
    target_skew = float(sweep_df['skew'].mean())
    target_kurtosis = float(sweep_df['kurtosis'].mean())

    actual_csv = csv_path if csv_path is not None else XRP_SWEEP_CSV
    print(f'Read {len(sweep_df)} real XRP sweep rows from {actual_csv}')
    print(f'T_effective grid (real, from the sweep): {t_grid}')
    print(f'Target fat-tailed regime (mean of real observed values): '
          f'skew={target_skew:.4f}, kurtosis={target_kurtosis:.4f}')
    print(f'  (individual-row skew range: '
          f'{sweep_df["skew"].min():.4f} to {sweep_df["skew"].max():.4f}; '
          f'kurtosis range: {sweep_df["kurtosis"].min():.4f} to '
          f'{sweep_df["kurtosis"].max():.4f} -- notably wider spread than '
          f'Kraken BTC\'s, flagged per module docstring)')

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
    print('PART 2 -- detection power, fat-tailed returns (XRP-calibrated jump-mixture)')
    print('=' * 74)
    mix = find_jump_mixture_for_target(target_skew, target_kurtosis)
    print(f"\n  jump-mixture calibration: p={mix['p']}, s={mix['s']}, "
          f"tilt={mix['tilt']}\n"
          f"  achieved (unshifted) skew={mix['skew0']:.4f}, "
          f"kurtosis={mix['kurtosis0']:.4f}\n"
          f"  (target, from XRP's real sweep: skew={target_skew:.4f}, "
          f"kurtosis={target_kurtosis:.4f} -- neighborhood match, not "
          f"exact, see module docstring; check the achieved-vs-target gap "
          f"above before trusting Part 2's readings, since XRP's wider "
          f"observed spread makes a loose match more likely than it was "
          f"for Kraken BTC)\n")

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
        HERE, 'xrp_detection_power_calibration.csv'
    )
    df1['regime'] = 'gaussian'
    df2['regime'] = 'fat_tailed'
    pd.concat([df1, df2], ignore_index=True).to_csv(out_path, index=False)
    print(f'\nFull result grid written to {out_path}')

    print('\n' + '=' * 74)
    print('READING TONIGHT\'S REAL XRP DSR VALUES AGAINST THIS CALIBRATION')
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
    regime shaped like XRP's own real data. A gap near 0 means today's
    DSR readings are indistinguishable from noise at this T --
    consistent with "no edge detected," not "edge ruled out." A gap
    clearly positive and growing with T would be a real signal worth
    taking seriously; compare its size against the true_sharpe columns
    in the fat-tailed pivot table above to get a rough sense of what
    edge size (if any) it corresponds to.
  - This calibration is a NEIGHBORHOOD match to XRP's real skew/
    kurtosis, not an exact distributional fit -- same caveat both prior
    scripts (Binance.US, Kraken BTC) carry. Check the achieved-vs-target
    gap printed above Part 2's table before leaning on this regime's
    numbers; XRP's wider per-row skew/kurtosis spread means this match
    may be looser than Kraken BTC's was.
  - This answers "how should I read the DSR numbers I already have,"
    not "is there an edge" -- same distinction both prior scripts drew.
  - Per the standing project rule: even a clearly positive gap here is
    NOT sufficient on its own -- Kraken BTC's window-1 result looked
    promising under this exact kind of calibration and did not survive
    an independent second-window replication check. That check has not
    yet been built for XRP (see calibrate_xrp_detection_power.py's
    sibling item on the handoff's next-session list). Do not treat a
    positive gap here as a finding until that replication check runs.
""")


if __name__ == '__main__':
    main()
