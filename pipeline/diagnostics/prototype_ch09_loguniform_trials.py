"""
pipeline/diagnostics/prototype_ch09_loguniform_trials.py

PROTOTYPE, per Ethan's explicit framing (2026-09-10): "try a prototype
with Ch09, see how it goes" -- not a change to production orchestration.
Does NOT touch ch11/chapter_11_backtest_dangers.py.

Question: Ch11's trial grid uses a fixed, hand-picked C_GRID=[0.01, 0.1,
1.0, 10.0]. Ch09's own stated rationale for logUniform (Snippet 9.3) is
that a scale hyperparameter like SVC's C deserves a log-uniform prior,
not hand-picked points, so every order of magnitude gets equal
representation rather than however many points a person happened to
pick. Does swapping the fixed grid for real log-uniform draws change
the PBO/DSR/signal conclusion on the SAME static established dataset?

*** HONEST SCOPE NOTE -- a real design compromise, not a full Ch09
integration ***
Ch09's real clfHyperFit() is a black box: it returns only a single fitted
best_estimator_, discarding every other candidate's score. PBO needs the
FULL set of N trial PnL series, not the winner alone (see Phase 1a's own
mistake: too few trials was part of what broke it). Wiring clfHyperFit()
wholesale is therefore not what this script does. Instead, ONLY Ch09's
real logUniform_gen (Snippet 9.3's actual probability distribution
object) is reused, unmodified, to DRAW C values -- replacing Ch11's
C_GRID's 4 fixed points with 4 random log-uniform draws over the exact
same real range those 4 points already spanned (0.01 to 10.0). Every
other piece of Ch11's real trial-construction loop (out_of_sample_probs,
getSignal, the shift(1) mark-to-market PnL, N_SPLITS/PCT_EMBARGO/GAMMA)
is reused via ch11 itself, byte-for-byte unmodified -- this script only
supplies a different C_GRID. Trial count is kept at 20 (4 C draws x 5
STEP_GRID values), matching Ch11's own established N=20 -- the same
comparability reasoning Ch11's own C_GRID migration comment gives for
why trial count must not silently change alongside whatever else varies.

Run (static established dataset, NOT live data -- this is a controlled
apples-to-apples comparison against the Phase 1b baseline):
    conda activate mlfinlab
    cd C:\\ws\\AFML
    python pipeline\\diagnostics\\prototype_ch09_loguniform_trials.py [seed]
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ORCH = os.path.join(HERE, '..', 'orchestration')
sys.path.insert(0, ROOT)
sys.path.insert(0, ORCH)

from stages import load_ch11_driver, evaluate_overfitting  # noqa: E402

CH09_DIR = os.path.join(ROOT, 'ch09', 'hyper_parameter_tuning')
if CH09_DIR not in sys.path:
    sys.path.insert(0, CH09_DIR)
from hyper_parameter_tuning import logUniform               # noqa: E402

CH10_BET_SIZING = os.path.join(ROOT, 'ch10', 'bet_sizing')
if CH10_BET_SIZING not in sys.path:
    sys.path.insert(0, CH10_BET_SIZING)
from bet_sizing import getSignal                             # noqa: E402

# Reused unchanged from Ch11's own real, established grid -- see module
# docstring. Only C's SOURCE changes (draws vs. fixed points); the range
# it's drawn over is the exact same range the fixed points spanned.
STEP_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
N_C_DRAWS = 4          # matches len(C_GRID) -- keeps total trials at 20
C_LOW, C_HIGH = 0.01, 10.0   # matches C_GRID's real min/max exactly


def build_loguniform_trials(ch11, seed=0):
    """Mirrors ch11.part_c_build_trials() exactly, except C_GRID's 4 fixed
    points are replaced by 4 real logUniform(0.01, 10.0) draws. Every
    other line -- data loading, out_of_sample_probs, getSignal, the
    shift(1) mark-to-market PnL -- is ch11's own real code, called
    through the loaded module, not copied/reimplemented."""
    rng = np.random.default_rng(seed)
    c_dist = logUniform(a=C_LOW, b=C_HIGH)
    c_draws = c_dist.rvs(size=N_C_DRAWS, random_state=seed)
    print(f'  logUniform({C_LOW}, {C_HIGH}) draws (seed={seed}): '
          f'{sorted(round(float(c), 5) for c in c_draws)}')

    enriched = os.path.join(ch11.INPUT, 'ch07_training_table_enriched.csv')
    if os.path.exists(enriched):
        events = pd.read_csv(enriched, index_col=0, parse_dates=True)
    else:
        events = pd.read_csv(os.path.join(ch11.INPUT, 'ch07_training_table.csv'),
                             index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    feats = pd.read_csv(os.path.join(ch11.INPUT, 'ch05_features.csv'),
                        index_col=0, parse_dates=True)

    feature_cols = [c for c in events.columns if c not in ('bin', 'w', 't1')]
    X, y, w, t1 = events[feature_cols], events['bin'], events['w'], events['t1']
    close = feats['close']
    bar_ret = close.pct_change().dropna()
    print(f'  events: {events.shape[0]}   features: {len(feature_cols)}   '
          f'bars: {close.shape[0]}   classes: {sorted(y.unique())}')

    cols, meta = {}, []
    for C in c_draws:
        C = float(C)
        prob, pred = ch11.out_of_sample_probs(X, y, w, t1, C)  # real, unmodified
        ev_c = events.loc[prob.index]
        for step in STEP_GRID:
            sig = getSignal(ev_c, step, prob, pred, numClasses=2, numThreads=1)
            pos = sig.reindex(close.index, method='ffill').fillna(0.0)
            pnl = pos.shift(1).reindex(bar_ret.index).fillna(0.0) * bar_ret
            name = f'C{C:.5g}_s{step:g}'
            cols[name] = pnl
            meta.append({'trial': name, 'C': C, 'stepSize': step,
                         'pct_bars_in_market': (pos != 0).mean()})

    M = pd.DataFrame(cols)
    meta_df = pd.DataFrame(meta).set_index('trial')
    meta_df['sharpe_full_sample'] = M.apply(ch11.sharpe_ratio) if hasattr(ch11, 'sharpe_ratio') \
        else M.apply(lambda s: s.mean() / s.std(ddof=1) if s.std(ddof=1) > 0 else np.nan)
    print(f'\n  M = {M.shape[0]} bars x {M.shape[1]} trials (logUniform C draws)\n')
    print(meta_df.round(4).to_string())
    return M, meta_df


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print('=' * 74)
    print(f'Ch09 logUniform-C prototype vs. established Phase 1b baseline (seed={seed})')
    print('=' * 74)

    ch11 = load_ch11_driver()
    M, meta = build_loguniform_trials(ch11, seed=seed)

    # Same tw reindex pattern stages.py's evaluate_overfitting() docstring
    # specifies for the STATIC baseline.
    weights_path = os.path.join(ch11.INPUT, 'ch04_weights.csv')
    enriched_path = os.path.join(ch11.INPUT, 'ch07_training_table_enriched.csv')
    weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    enriched_idx = pd.read_csv(enriched_path, index_col=0, parse_dates=True).index
    tw = weights['tw'].reindex(enriched_idx)
    if tw.isna().any():
        raise SystemExit('tw has NaN after reindexing -- check ch04_weights.csv alignment.')

    result = evaluate_overfitting(M, meta, ch11, S=12, tw=tw)

    print('\n' + '=' * 74)
    print('RESULT vs. established Phase 1b baseline')
    print('=' * 74)
    print(f"  PBO  : {result['prob_overfit']:.4f}   (baseline: ~0.83)")
    print(f"  DSR  : {result['dsr']:.4f}   (baseline: ~0.54)")
    print(f"  best trial: {result['best_trial']}  "
          f"(Sharpe {result['sr_hat']:+.4f})")
    print(f"  T_effective: {result['T']:.4f}   T_raw: {result['T_raw']}   "
          f"tw_mean: {result['tw_mean']:.4f}")
    print("\n  Compare against the established static-dataset numbers "
          "(PBO~0.83, DSR~0.54, 0/5 CPCV paths) before drawing any "
          "conclusion -- one seed is one draw, not a replication.")


if __name__ == '__main__':
    main()
