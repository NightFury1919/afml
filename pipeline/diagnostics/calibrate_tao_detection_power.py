"""
pipeline/diagnostics/calibrate_tao_detection_power.py

TAO counterpart to calibrate_xrp_detection_power.py (which itself mirrored
calibrate_kraken_detection_power.py's real 2026-08-19 methodology -- N=20
trials matching production C_GRID x STEP_GRID, real
ch14.deflated_sharpe_ratio(), Gaussian + fat-tailed jump-mixture regimes).
PARTS 1-2 below ask the same base question at TAO's own real observed
T_effective range: at TAO's real target_bars sweep results (2026-08-26,
see tao_target_bars_calibration.csv), would DSR actually detect a real
edge if one existed, and what does its null-hypothesis false-positive
rate look like at that T?

PARTS 3-5 are a REFINED DESIGN beyond what the Kraken BTC/XRP scripts did
(Ethan's proposal, end of 2026-08-26 session, handoff Part 7) -- added
because TAO's own T_effective values (102.99-355.01) are thin enough,
and TAO's target_bars sweep DSR trend already runs the wrong direction
for optimism (0.475 -> 0.364, monotonically DOWN as T rises), that a
plain 6-point calibration risks conflating two different questions:
"is there no edge" vs. "is the sample too small to tell." The three
additions:

  PART 3 (Stage A, positive control): inject an obvious edge
  (true_sharpe=0.5) at a large, TAO-unrealistic T (T_LARGE=5000) and
  confirm P[DSR>0.5] comes back near-certain in BOTH regimes. This is a
  pure apparatus sanity check -- isolates "is the detector broken" from
  "is there no edge in TAO specifically." Directly resolves the standing
  open item from the 2026-08-23 momentum-vs-OFI sanity-check session,
  generalized to this asset/methodology.

  PART 4 (Stage B, ideal-conditions minimum-detectable-edge): same
  T_LARGE, but a FINE true_sharpe grid (0.5 down to 0.0) to find the
  smallest edge this pipeline can EVER reliably detect (P[DSR>0.5] >=
  DETECTION_THRESHOLD) under the most forgiving sample-size conditions
  imaginable. This is the pipeline's best-case sensitivity floor.

  PART 5 (TAO-realistic minimum-detectable-edge): the SAME fine grid,
  but at TAO's actual, real, thin T_effective values (102.99-355.01)
  instead of T_LARGE. Finds the smallest edge TAO's pipeline could
  realistically notice given its real sample-size constraints.

Comparing Part 4 vs Part 5 separates two different diagnoses that are
easy to conflate: if Part 4's floor is small (detects tiny edges) but
Part 5's floor is much larger, TAO's SAMPLE SIZE -- not the methodology
-- is the bottleneck, a materially more actionable finding than "DSR
came out low." If both floors are similarly small, that's a stronger
case that TAO's low DSR readings reflect genuine absence of edge rather
than a power problem.

SCOPE / LOAD-BEARING DECISIONS (confirmed with Ethan before writing,
2026-08-27):
  - N_REPS=20,000 kept at FULL production precision for every stage,
    including the two new fine-grid stages (Parts 4-5) -- explicitly
    chosen over a faster/lower-precision option. Total combos: Part
    1+2 base = 5 T's x 6 true_sharpes x 2 regimes = 60; Part 3 = 1 T x
    1 true_sharpe x 2 regimes = 2; Part 4 = 1 T x 9 true_sharpes x 2
    regimes = 18; Part 5 = 5 T's x 9 true_sharpes x 2 regimes = 90.
    Total 170 combos at N_REPS=20,000 (vs. the base-only scripts'
    60) -- expect roughly 75-90 minutes total runtime, budget
    generously as with every other Monte Carlo calibration in this
    project.
  - T_LARGE = 5,000 for Stage A/B (Part 3-4) -- generous, well beyond
    anything TAO's real data could produce (TAO's own max T_effective
    is 355.01), deliberately unrealistic-for-TAO to isolate best-case
    sensitivity.
  - DETECTION_THRESHOLD = 0.80 -- an edge counts as "reliably
    detected" at a given true_sharpe/T combination if P[DSR>0.5] >=
    0.80 there. Used to locate the minimum-detectable-edge floor in
    both Part 4 and Part 5 (walk FINE_SHARPE_GRID from largest to
    smallest true_sharpe; floor = smallest true_sharpe where power is
    still >= threshold).

FAT-TAILED REGIME: calibrated to TAO's own real observed skew/kurtosis
(read directly from tao_target_bars_calibration.csv's skew/kurtosis
columns -- the actual winning-trial values from 2026-08-26's five real
sweep runs), NOT Binance.US's, Kraken BTC's, or XRP's values. TAO's
observed regime has the WIDEST per-row spread of any asset so far (skew
ranges -0.063 to +1.042 across the five rows -- individual rows, not
settled near zero the way Kraken BTC's was, and wider than XRP's -0.43
to +0.38; kurtosis ranges 7.83 to 27.80, mean ~14.46, vs XRP's mean
~11.9 and Kraken BTC's ~7.0), so the achieved-vs-target calibration
match is reported explicitly and should be checked before trusting any
fat-tailed-regime reading here -- same "neighborhood match, not exact
fit" caveat every prior script in this family carries, flagged as
possibly the loosest fit yet given TAO's spread.

Reuses calibrate_jump_mixture()/find_jump_mixture_for_target()'s real
double-sided grid-search machinery UNMODIFIED from the XRP script (it
was already written generally, not hardcoded to any one asset's
target) -- only the target skew/kurtosis fed into it changes.

Usage
-----
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\calibrate_tao_detection_power.py
    python pipeline\\diagnostics\\calibrate_tao_detection_power.py <sweep_csv> [output_csv]

Optional <sweep_csv>/[output_csv] args mirror the Kraken/XRP versions'
second-window replication hook -- kept ready even though TAO's second
window doesn't exist yet (queued as this handoff's next item after this
calibration), so this script needs no changes when that window is
captured.
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

TAO_SWEEP_CSV = os.path.join(HERE, 'tao_target_bars_calibration.csv')

# Real T_effective values from TAO's own 2026-08-26 target_bars sweep
# -- see module docstring. Read directly from the CSV at runtime (not
# hardcoded independently) so this script can never silently drift from
# the actual sweep results it exists to interpret.
TRUE_SHARPE_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

# Part 7 additions -- fine grid for the two minimum-detectable-edge
# stages, largest to smallest so find_minimum_detectable_edge() can walk
# it in order.
FINE_SHARPE_GRID = [0.5, 0.3, 0.2, 0.15, 0.10, 0.05, 0.02, 0.01, 0.0]
T_LARGE = 5_000
DETECTION_THRESHOLD = 0.80


def load_tao_sweep_stats(csv_path=None):
    """Reads a TAO sweep's T_effective and skew/kurtosis columns
    directly -- this script's T grid and fat-tailed regime calibration
    are BOTH derived from this file, not independently guessed, so they
    can never silently drift from the sweep they exist to interpret.

    csv_path : str or None -- defaults to TAO_SWEEP_CSV (this first
    window's results) if not given, mirroring the Kraken/XRP scripts'
    second-window replication hook."""
    path = csv_path if csv_path is not None else TAO_SWEEP_CSV
    if not os.path.exists(path):
        raise SystemExit(
            f'{path} not found -- run calibrate_tao_target_bars.py first.'
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=['T_effective'])
    if df.empty:
        raise SystemExit(f'{path} has no successful rows to read.')
    return df


def simulate_power_gaussian(T, true_sharpe, N=N_TRIALS, n_reps=N_REPS, seed=42):
    """Reused UNMODIFIED from the Kraken/XRP scripts -- i.i.d.
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
    """Reused UNMODIFIED from the Kraken/XRP scripts' double-sided jump
    mixture (see calibrate_kraken_detection_power.py's own note on why
    the single-direction version was replaced -- it structurally cannot
    hit a near-zero-skew + high-kurtosis target simultaneously). Kept
    general on purpose: independent positive/negative jump probabilities
    (p_pos = p*(1-tilt), p_neg = p*(1+tilt)) at the same jump size `s`,
    so tilt != 0 can bias toward one side -- relevant here since TAO's
    real target skew (mean ~0.368, individual rows -0.063 to +1.042) is
    the least settled-near-zero of any asset calibrated so far."""
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
    """Reused UNMODIFIED grid search over (p, s, tilt) from the
    Kraken/XRP scripts (already asset-general, not hardcoded). TAO's
    kurtosis target (~14.46 mean, up to 27.80 on the tb3000 row) is
    well outside this grid's easiest reach, and TAO's skew target
    (~0.368 mean, up to 1.042) is the furthest from zero of any asset
    calibrated so far -- the achieved match is reported explicitly
    below rather than assumed, same 'neighborhood match, not exact fit'
    caveat every prior script carries, flagged as possibly the loosest
    fit yet."""
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
    """Reused UNMODIFIED from the Kraken/XRP scripts -- fat-tailed
    jump-mixture returns, REALIZED skew/kurtosis from the selected best
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


def find_minimum_detectable_edge(sharpe_grid, power_by_sharpe, threshold=DETECTION_THRESHOLD):
    """Part 7 addition. sharpe_grid must be sorted LARGEST to smallest
    (matches FINE_SHARPE_GRID's own order). power_by_sharpe is a dict
    true_sharpe -> P[DSR>0.5]. Walks the grid from the largest edge
    down and returns the smallest true_sharpe at which power is still
    >= threshold -- i.e. the smallest edge this regime/T combination can
    still reliably detect. Returns None if even the largest edge in the
    grid fails to clear threshold (apparatus/power problem at this T),
    and returns 0.0 with a caveat printed by the caller if EVERY point
    clears threshold (floor may be below the grid's own bottom)."""
    floor = None
    for true_sharpe in sharpe_grid:
        if power_by_sharpe.get(true_sharpe, 0.0) >= threshold:
            floor = true_sharpe
        else:
            break
    return floor


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    sweep_df = load_tao_sweep_stats(csv_path)
    t_grid = sorted(sweep_df['T_effective'].round(2).tolist())
    real_dsr = dict(zip(sweep_df['T_effective'].round(2), sweep_df['dsr']))
    target_skew = float(sweep_df['skew'].mean())
    target_kurtosis = float(sweep_df['kurtosis'].mean())

    actual_csv = csv_path if csv_path is not None else TAO_SWEEP_CSV
    print(f'Read {len(sweep_df)} real TAO sweep rows from {actual_csv}')
    print(f'T_effective grid (real, from the sweep): {t_grid}')
    print(f'Target fat-tailed regime (mean of real observed values): '
          f'skew={target_skew:.4f}, kurtosis={target_kurtosis:.4f}')
    print(f'  (individual-row skew range: '
          f'{sweep_df["skew"].min():.4f} to {sweep_df["skew"].max():.4f}; '
          f'kurtosis range: {sweep_df["kurtosis"].min():.4f} to '
          f'{sweep_df["kurtosis"].max():.4f} -- widest spread of any asset '
          f'calibrated so far, flagged per module docstring)')

    all_rows = []

    # ------------------------------------------------------------------
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
    df1['stage'] = 'base'
    df1['regime'] = 'gaussian'
    all_rows.append(df1)

    # ------------------------------------------------------------------
    print('\n' + '=' * 74)
    print('PART 2 -- detection power, fat-tailed returns (TAO-calibrated jump-mixture)')
    print('=' * 74)
    mix = find_jump_mixture_for_target(target_skew, target_kurtosis)
    print(f"\n  jump-mixture calibration: p={mix['p']}, s={mix['s']}, "
          f"tilt={mix['tilt']}\n"
          f"  achieved (unshifted) skew={mix['skew0']:.4f}, "
          f"kurtosis={mix['kurtosis0']:.4f}\n"
          f"  (target, from TAO's real sweep: skew={target_skew:.4f}, "
          f"kurtosis={target_kurtosis:.4f} -- neighborhood match, not "
          f"exact, see module docstring; check the achieved-vs-target gap "
          f"above before trusting Part 2's readings, given TAO's spread is "
          f"the widest calibrated so far)\n")

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
    df2['stage'] = 'base'
    df2['regime'] = 'fat_tailed'
    all_rows.append(df2)

    # ------------------------------------------------------------------
    print('\n' + '=' * 74)
    print(f'PART 3 -- Stage A positive control (apparatus sanity check, T={T_LARGE}, true_sharpe=0.5)')
    print('=' * 74)
    stage_a_sharpe = 0.5
    dsrs_g, _ = simulate_power_gaussian(T_LARGE, stage_a_sharpe)
    power_g = (dsrs_g > 0.5).mean()
    dsrs_f, _ = simulate_power_fat_tailed(T_LARGE, stage_a_sharpe, mix)
    power_f = (dsrs_f > 0.5).mean()
    print(f'\n  Gaussian:   P[DSR>0.5] = {power_g:.4f}  (expect near 1.0)')
    print(f'  Fat-tailed: P[DSR>0.5] = {power_f:.4f}  (expect near 1.0)')
    if power_g >= 0.99 and power_f >= 0.99:
        print('\n  PASS -- apparatus correctly detects an obvious edge under '
              'generous conditions in both regimes. Low DSR readings '
              'elsewhere in this script are not attributable to a broken '
              'detector.')
    else:
        print('\n  *** WARNING: apparatus did NOT reliably detect an obvious '
              'edge under generous conditions. Investigate the simulation '
              'or deflated_sharpe_ratio() call before trusting any other '
              'result in this script. ***')
    all_rows.append(pd.DataFrame([
        {'true_sharpe': stage_a_sharpe, 'T': T_LARGE, 'P[DSR>0.5]': power_g,
         'P[correct_selection]': np.nan, 'mean_dsr': dsrs_g.mean(),
         'stage': 'stage_a_positive_control', 'regime': 'gaussian'},
        {'true_sharpe': stage_a_sharpe, 'T': T_LARGE, 'P[DSR>0.5]': power_f,
         'P[correct_selection]': np.nan, 'mean_dsr': dsrs_f.mean(),
         'stage': 'stage_a_positive_control', 'regime': 'fat_tailed'},
    ]))

    # ------------------------------------------------------------------
    print('\n' + '=' * 74)
    print(f'PART 4 -- Stage B ideal-conditions minimum-detectable-edge (T={T_LARGE})')
    print('=' * 74)
    power_g_by_sharpe = {}
    power_f_by_sharpe = {}
    rows4 = []
    for true_sharpe in FINE_SHARPE_GRID:
        dsrs_g, _ = simulate_power_gaussian(T_LARGE, true_sharpe)
        p_g = (dsrs_g > 0.5).mean()
        power_g_by_sharpe[true_sharpe] = p_g
        dsrs_f, _ = simulate_power_fat_tailed(T_LARGE, true_sharpe, mix)
        p_f = (dsrs_f > 0.5).mean()
        power_f_by_sharpe[true_sharpe] = p_f
        rows4.append({'true_sharpe': true_sharpe, 'T': T_LARGE,
                       'P[DSR>0.5]': p_g, 'P[correct_selection]': np.nan,
                       'mean_dsr': dsrs_g.mean(), 'stage': 'stage_b_ideal_floor',
                       'regime': 'gaussian'})
        rows4.append({'true_sharpe': true_sharpe, 'T': T_LARGE,
                       'P[DSR>0.5]': p_f, 'P[correct_selection]': np.nan,
                       'mean_dsr': dsrs_f.mean(), 'stage': 'stage_b_ideal_floor',
                       'regime': 'fat_tailed'})
        print(f'  true_sharpe={true_sharpe:.2f}: '
              f'P[DSR>0.5] gaussian={p_g:.4f}, fat_tailed={p_f:.4f}')
    df4 = pd.DataFrame(rows4)
    all_rows.append(df4)

    floor_g_ideal = find_minimum_detectable_edge(FINE_SHARPE_GRID, power_g_by_sharpe)
    floor_f_ideal = find_minimum_detectable_edge(FINE_SHARPE_GRID, power_f_by_sharpe)
    print(f'\n  IDEAL-CONDITIONS FLOOR (T={T_LARGE}, threshold P[DSR>0.5]>='
          f'{DETECTION_THRESHOLD}):')
    print(f'    Gaussian:   {floor_g_ideal if floor_g_ideal is not None else "none in grid reliably detected"}')
    print(f'    Fat-tailed: {floor_f_ideal if floor_f_ideal is not None else "none in grid reliably detected"}')

    # ------------------------------------------------------------------
    print('\n' + '=' * 74)
    print('PART 5 -- TAO-realistic minimum-detectable-edge (TAO\'s own real T_effective values)')
    print('=' * 74)
    rows5 = []
    floors_g_realistic = {}
    floors_f_realistic = {}
    for T in t_grid:
        power_g_by_sharpe_T = {}
        power_f_by_sharpe_T = {}
        for true_sharpe in FINE_SHARPE_GRID:
            dsrs_g, _ = simulate_power_gaussian(T, true_sharpe)
            p_g = (dsrs_g > 0.5).mean()
            power_g_by_sharpe_T[true_sharpe] = p_g
            dsrs_f, _ = simulate_power_fat_tailed(T, true_sharpe, mix)
            p_f = (dsrs_f > 0.5).mean()
            power_f_by_sharpe_T[true_sharpe] = p_f
            rows5.append({'true_sharpe': true_sharpe, 'T': T,
                           'P[DSR>0.5]': p_g, 'P[correct_selection]': np.nan,
                           'mean_dsr': dsrs_g.mean(), 'stage': 'tao_realistic_floor',
                           'regime': 'gaussian'})
            rows5.append({'true_sharpe': true_sharpe, 'T': T,
                           'P[DSR>0.5]': p_f, 'P[correct_selection]': np.nan,
                           'mean_dsr': dsrs_f.mean(), 'stage': 'tao_realistic_floor',
                           'regime': 'fat_tailed'})
        floor_g = find_minimum_detectable_edge(FINE_SHARPE_GRID, power_g_by_sharpe_T)
        floor_f = find_minimum_detectable_edge(FINE_SHARPE_GRID, power_f_by_sharpe_T)
        floors_g_realistic[T] = floor_g
        floors_f_realistic[T] = floor_f
        print(f'  T={T:.2f}: floor gaussian='
              f'{floor_g if floor_g is not None else "none detected"}, '
              f'floor fat_tailed={floor_f if floor_f is not None else "none detected"}')
    df5 = pd.DataFrame(rows5)
    all_rows.append(df5)

    # ------------------------------------------------------------------
    out_path = output_csv if output_csv is not None else os.path.join(
        HERE, 'tao_detection_power_calibration.csv'
    )
    pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
    print(f'\nFull result grid (base + Stage A + Stage B + TAO-realistic) '
          f'written to {out_path}')

    # ------------------------------------------------------------------
    print('\n' + '=' * 74)
    print('READING TAO\'S REAL DSR VALUES AGAINST THIS CALIBRATION')
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

    print('\n' + '=' * 74)
    print('PART 4 vs PART 5 -- SAMPLE-SIZE-BOTTLENECK vs GENUINE-NO-EDGE DIAGNOSTIC')
    print('=' * 74)
    print(f"""
  Ideal-conditions floor (T={T_LARGE}, best case): gaussian={floor_g_ideal}, fat_tailed={floor_f_ideal}
  TAO-realistic floors (TAO's own real T range, {t_grid[0]:.2f}-{t_grid[-1]:.2f}):
""")
    for T in t_grid:
        print(f'    T={T:.2f}: gaussian={floors_g_realistic[T]}, fat_tailed={floors_f_realistic[T]}')
    print("""
  INTERPRETATION:
  - If the TAO-realistic floors above are MUCH LARGER than the ideal-
    conditions floor, that means TAO's real sample size -- not the
    detection methodology -- is the bottleneck: a real edge could exist
    and still not be reliably flagged at TAO's actual T_effective range.
    That is a materially different, more actionable finding than "DSR
    came out low," and argues for more data (longer capture windows,
    denser bar construction) before concluding "no edge" for TAO.
  - If the TAO-realistic floors are SIMILAR to the ideal-conditions
    floor, sample size is not the limiting factor here -- TAO's low DSR
    readings are more likely to reflect a genuine absence of a
    detectable edge at this feature set, the same conclusion this
    project has already reached independently for BTC (Ch11 PBO, Ch12
    CPCV, Ch14 DSR, Ch18 entropy) and for XRP (window-1 pattern that did
    not replicate).
  - Per the standing project rule: even if Part 4/5 point toward "no
    sample-size bottleneck, no edge," the second-window replication
    check (queued as this handoff's next item) is still required before
    treating that as a finding -- BTC's and XRP's own false leads both
    looked plausible under a single-window calibration and did not
    survive replication.
""")


if __name__ == '__main__':
    main()
