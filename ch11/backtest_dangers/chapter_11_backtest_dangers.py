"""
Chapter 11 -- The Dangers of Backtesting
========================================
Example script. Real BTC/TUSD data end to end.

AFML Chapter 11 has NO numbered code snippets -- it is an argument, not an
algorithm, until Section 11.6. This script therefore has two halves:

    Parts A-B  the ARGUMENT (11.2-11.5), made concrete with a simulation
    Parts C-D  the ALGORITHM (11.6): CSCV -> PBO, on our own real strategy

Run:
    conda activate mlfinlab
    python chapter_11_backtest_dangers.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.svm import SVC

# --- repo root via __file__ (portable; see CLAUDE.md path convention) -------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from ch07.cross_validation.purged_kfold import PurgedKFold          # noqa: E402
from ch10.bet_sizing.bet_sizing import getSignal                    # noqa: E402
from ch11.backtest_dangers.pbo import pbo, sharpe_ratio             # noqa: E402

INPUT = os.path.join(ROOT, 'input_data')
HERE = os.path.dirname(os.path.abspath(__file__))

# Established calibration for this dataset (Ch07 CV, Ch09 tuning, Ch10 sizing)
N_SPLITS, PCT_EMBARGO, GAMMA = 4, 0.12, 0.1
C_GRID = [0.1, 1.0, 10.0, 100.0]
STEP_GRID = [0.01, 0.02, 0.05, 0.10, 0.20]
S_BLOCKS = 8


# ===========================================================================
# PART A -- Why a flawless backtest is still probably wrong (11.2-11.3)
# ===========================================================================
def part_a_seven_sins():
    print('=' * 74)
    print('PART A -- The seven sins (11.2), and why fixing them is not enough')
    print('=' * 74)
    sins = [
        ('1. Survivorship bias', 'Universe = survivors only; the bankrupt are invisible.'),
        ('2. Look-ahead bias', 'Using data not yet public at decision time.'),
        ('3. Storytelling', 'Inventing an ex-post rationale for a random pattern.'),
        ('4. Data mining/snooping', 'Training on the testing set.'),
        ('5. Transaction costs', 'True cost is only knowable by actually trading.'),
        ('6. Outliers', 'An edge resting on a few freak events.'),
        ('7. Shorting', 'Assumes a lender, at a knowable cost. Often neither.'),
    ]
    for name, why in sins:
        print(f'  {name:26s} {why}')
    print("""
  Section 11.3's twist: avoid ALL SEVEN and the backtest is still probably
  wrong. Becoming skilled enough to produce a flawless backtest means having
  run thousands of them -- and the more you run, the more certain it is that
  one looks good by luck alone. Part B makes that concrete.
""")


# ===========================================================================
# PART B -- Multiple testing: the best of N zero-edge strategies (11.3)
# ===========================================================================
def part_b_multiple_testing(n_trials_grid=(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000),
                            n_obs=250, n_reps=200, seed=0):
    """
    SYNTHETIC BY DESIGN (the one sanctioned use, per CLAUDE.md): we need
    strategies with a KNOWN true Sharpe of exactly zero. No real dataset can
    give us that guarantee -- and the whole point is to watch a zero-edge
    strategy masquerade as a winner.

    Every "strategy" here is pure noise. True Sharpe = 0 for all of them.
    We simply run N of them and keep the best, as a researcher would.
    """
    print('=' * 74)
    print('PART B -- Best-of-N on strategies with a TRUE Sharpe of exactly 0')
    print('=' * 74)
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_trials_grid:
        best = []
        for _ in range(n_reps):
            pnl = rng.normal(size=(n_obs, n))          # zero edge, by construction
            sr = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
            best.append(sr.max())                      # keep only the winner
        rows.append({'n_trials': n, 'expected_best_sharpe': float(np.mean(best))})
    df = pd.DataFrame(rows).set_index('n_trials')
    print(df.round(4).to_string())
    print("""
  Read that column again. Nothing here has any edge whatsoever -- and yet the
  more configurations we try, the better our "best" strategy looks. The
  apparent Sharpe is manufactured entirely by selection. This is why Section
  11.4 insists a backtest is not a research tool: it cannot tell you WHY the
  winner won, and the winner is usually just the luckiest lottery ticket.

  (Chapter 12, Section 12.5, gives the closed form for this curve.)
""")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df.index, df['expected_best_sharpe'], 'o-', lw=2, color='crimson')
    ax.axhline(0, color='k', ls='--', lw=1, label='TRUE Sharpe of every strategy = 0')
    ax.set_xscale('log')
    ax.set_xlabel('number of backtests run (N)')
    ax.set_ylabel('Sharpe ratio of the BEST one')
    ax.set_title('Backtest overfitting from nothing but multiple testing')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'ch11_multiple_testing.png'), dpi=120)
    plt.close(fig)
    return df


# ===========================================================================
# PART C -- Build N REAL trials (the strategies a researcher would compare)
# ===========================================================================
def out_of_sample_probs(X, y, w, t1, C):
    """
    Out-of-sample predict_proba via Ch07's PurgedKFold -- never a plain
    in-sample refit, which would be lookahead bias in a bet-sizing signal.
    random_state=0 is LOAD-BEARING: SVC(probability=True) runs an internal
    randomised Platt-scaling CV and flips class predictions run-to-run on this
    small dataset without it (bug found in Ch10).
    """
    clf = SVC(C=C, gamma=GAMMA, probability=True, random_state=0)
    pkf = PurgedKFold(n_splits=N_SPLITS, t1=t1, pctEmbargo=PCT_EMBARGO)
    prob = pd.Series(index=X.index, dtype=float)
    pred = pd.Series(index=X.index, dtype=float)
    for tr, te in pkf.split(X=X):
        fit = clf.fit(X.iloc[tr], y.iloc[tr], sample_weight=w.iloc[tr].values)
        p = fit.predict_proba(X.iloc[te])
        prob.iloc[te] = p.max(axis=1)
        pred.iloc[te] = fit.classes_[p.argmax(axis=1)]
    return prob.dropna(), pred.dropna()


def part_c_build_trials():
    print('=' * 74)
    print('PART C -- N real strategy configurations, backtested on real BTC data')
    print('=' * 74)
    events = pd.read_csv(os.path.join(INPUT, 'ch07_training_table.csv'),
                         index_col=0, parse_dates=True)
    events['t1'] = pd.to_datetime(events['t1'])
    feats = pd.read_csv(os.path.join(INPUT, 'ch05_features.csv'),
                        index_col=0, parse_dates=True)

    X, y, w, t1 = events[['fracdiff']], events['bin'], events['w'], events['t1']
    close = feats['close']
    bar_ret = close.pct_change().dropna()
    print(f'  events: {events.shape[0]}   bars: {close.shape[0]}   '
          f'classes: {sorted(y.unique())}')

    cols, meta = {}, []
    for C in C_GRID:
        prob, pred = out_of_sample_probs(X, y, w, t1, C)
        ev_c = events.loc[prob.index]
        for step in STEP_GRID:
            # Ch10's real pipeline: prob -> getSignal -> avgActiveSignals ->
            # discreteSignal. numThreads=1: deterministic, and SVC-with-
            # probability is not spawn-safe under joblib/loky on Windows.
            sig = getSignal(ev_c, step, prob, pred, numClasses=2, numThreads=1)
            # Position is held between signal change-points; project onto bars.
            pos = sig.reindex(close.index, method='ffill').fillna(0.0)
            # Mark-to-market: yesterday's position earns today's bar return.
            # .shift(1) is LOAD-BEARING -- without it the position would earn
            # the return that CAUSED it, which is lookahead bias.
            pnl = pos.shift(1).reindex(bar_ret.index).fillna(0.0) * bar_ret
            name = f'C{C:g}_s{step:g}'
            cols[name] = pnl
            meta.append({'trial': name, 'C': C, 'stepSize': step,
                         'pct_bars_in_market': (pos != 0).mean()})

    M = pd.DataFrame(cols)
    meta = pd.DataFrame(meta).set_index('trial')
    meta['sharpe_full_sample'] = M.apply(sharpe_ratio)
    print(f'\n  M = {M.shape[0]} bars x {M.shape[1]} trials\n')
    print(meta.round(4).to_string())
    best = meta['sharpe_full_sample'].idxmax()
    print(f"\n  A naive researcher ships: {best} "
          f"(Sharpe {meta.loc[best, 'sharpe_full_sample']:+.4f})")
    print('  Part D asks whether that choice means anything at all.\n')

    fig, ax = plt.subplots(figsize=(10, 4.5))
    c = ['crimson' if t == best else '#4477aa' for t in meta.index]
    ax.bar(meta.index, meta['sharpe_full_sample'], color=c)
    ax.axhline(0, color='k', lw=1)
    ax.set_ylabel('full-sample Sharpe (per bar)')
    ax.set_title('The cherry-picking temptation: 20 configurations, one "winner"')
    ax.tick_params(axis='x', rotation=90)
    ax.grid(alpha=.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'ch11_trial_sharpes.png'), dpi=120)
    plt.close(fig)
    return M, meta


# ===========================================================================
# PART D -- CSCV / PBO (11.6)
# ===========================================================================
def part_d_pbo(M):
    print('=' * 74)
    print(f'PART D -- CSCV / PBO (Section 11.6), S={S_BLOCKS}')
    print('=' * 74)
    value, res = pbo(M, S=S_BLOCKS)
    n = M.shape[1]

    print(f'  combinations C(S, S/2)         : {len(res)}')
    print(f'  PBO                            : {value:.4f}')
    print(f'  median logit (lambda)          : {res["logit"].median():+.4f}')
    print(f'  mean IS  Sharpe of the winner  : {res["r_is"].mean():+.4f}')
    print(f'  mean OOS Sharpe of that winner : {res["r_oos"].mean():+.4f}  <-- decay')
    print(f'  mean OOS rank of the winner    : {res["rank_oos"].mean():.2f} / {n}')
    print(f'\n  distinct trials ever selected as "best": '
          f'{res["n_star"].nunique()} of {n}')
    print(res['n_star'].value_counts().to_string())

    verdict = ('selection is RELIABLE' if value < 0.25 else
               'selection is a COIN FLIP' if value < 0.55 else
               'selection is ACTIVELY HARMFUL')
    print(f"""
  PBO = {value:.2f} -> {verdict}.
  The configuration that wins in-sample lands, on average, in the bottom
  half out-of-sample. There is no stable winner here -- which is the honest
  answer for a strategy family built on a single feature (fracdiff) with no
  real edge. Section 11.4: the purpose of a backtest is to DISCARD bad
  models, not to improve them. This one is telling us to discard.
""")

    # Figure 11.1 analogue -- IS vs OOS Sharpe of the selected strategy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.scatter(res['r_is'], res['r_oos'], alpha=.6, s=30, color='#4477aa')
    lim = max(abs(res[['r_is', 'r_oos']].to_numpy()).max() * 1.1, 1e-9)
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, label='no decay (y = x)')
    ax.axhline(0, color='grey', lw=.8)
    ax.axvline(0, color='grey', lw=.8)
    ax.set_xlabel('IS Sharpe of the selected strategy')
    ax.set_ylabel('OOS Sharpe of that same strategy')
    ax.set_title('Fig 11.1 analogue: performance decay')
    ax.legend()
    ax.grid(alpha=.3)

    # Figure 11.2 analogue -- distribution of logits; PBO is the mass below 0
    ax = axes[1]
    ax.hist(res['logit'], bins=20, color='#4477aa', edgecolor='white')
    ax.axvline(0, color='k', ls='--', lw=1.5)
    lo, hi = ax.get_xlim()
    ax.axvspan(lo, 0, color='crimson', alpha=.15)
    ax.text(0.03, 0.92, f'PBO = {value:.2f}\n(mass left of 0)',
            transform=ax.transAxes, color='crimson', fontweight='bold', va='top')
    ax.set_xlabel(r'logit  $\lambda_c = \log[\bar\omega_c / (1-\bar\omega_c)]$')
    ax.set_ylabel('frequency')
    ax.set_title('Fig 11.2 analogue: distribution of OOS ranks')
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'ch11_pbo.png'), dpi=120)
    plt.close(fig)
    return value, res


def main():
    part_a_seven_sins()
    part_b_multiple_testing()
    M, meta = part_c_build_trials()
    value, res = part_d_pbo(M)

    out = os.path.join(INPUT, 'ch11_pbo_stats')
    stats = pd.DataFrame({'pbo': [value], 'S': [S_BLOCKS],
                          'n_trials': [M.shape[1]], 'n_obs': [M.shape[0]],
                          'median_logit': [res['logit'].median()],
                          'mean_is_sharpe': [res['r_is'].mean()],
                          'mean_oos_sharpe': [res['r_oos'].mean()]})
    stats.to_csv(out + '.csv', index=False)
    stats.to_pickle(out + '.pkl')
    M.to_csv(os.path.join(INPUT, 'ch11_trial_pnl.csv'))
    print(f'  saved: {out}.{{csv,pkl}} and ch11_trial_pnl.csv')
    print('  saved: ch11_multiple_testing.png, ch11_trial_sharpes.png, ch11_pbo.png')


if __name__ == '__main__':
    main()
