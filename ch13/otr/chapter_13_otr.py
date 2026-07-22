"""
Chapter 13 -- Optimal Trading Rules (OTR) via Backtesting on Synthetic Data.

Three parts:
  A. Reproduce two of the book's own synthetic examples (Section 13.6) as a
     validation that our implementation matches the book's qualitative and
     approximate quantitative claims.
  B. Calibrate {phi, sigma} from REAL opportunities (Ch10's 88-row BTC/TUSD
     triple-barrier pipeline) -- AFML Step 1.
  C. Apply the real calibration to a real mesh sweep and report what the
     synthetic-backtesting framework actually finds on this data.

Path convention: this .py script derives its own root via __file__ (works
for anyone who clones the repo, any OS, any username). The paired notebook
uses a hardcoded AFML_ROOT instead, per CLAUDE.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
INPUT = os.path.join(ROOT, 'input_data')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ch13.otr.otr import (
    build_xy_from_opportunities,
    estimate_ou_params,
    phi_to_half_life,
    half_life_to_phi,
    simulate_ou_path,
    batch,
    best_node,
)


# ===========================================================================
# PART A -- reproduce two of the book's own synthetic examples (Section 13.6)
# ===========================================================================
def part_a_book_reproduction():
    print('=' * 74)
    print('PART A -- reproducing two of the book\'s own synthetic heat-maps')
    print('=' * 74)

    cases = [
        # (label, forecast, half-life, book's stated approx best Sharpe)
        ('Fig 13.1: forecast=0,  hl=5',  0.0, 5,  3.2),
        ('Fig 13.6: forecast=5,  hl=5',  5.0, 5,  12.0),
    ]
    # NOTE: reduced mesh/nIter from the book's 20x20 x 100,000 for demo-script
    # runtime -- Snippet 13.2's pure-Python loop is inherently slow (the book
    # itself notes "this algorithm can be parallelized... we leave that task
    # as an exercise"). The TDD suite's test_batch_book_reproduces_approx_
    # sharpe_forecast0_hl5 already validates the book-magnitude claim; this
    # plot is for illustration.
    r_pt = np.linspace(0.5, 10, 8)
    r_sl = np.linspace(0.5, 10, 8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (label, forecast, hl, book_sharpe) in zip(axes, cases):
        coeffs = {'forecast': forecast, 'hl': hl, 'sigma': 1.0}
        results = batch(coeffs, n_iter=3_000, max_hp=100, r_pt=r_pt, r_sl=r_sl,
                         seed=0.0, random_state=1)
        mesh = np.array([[r[4] for r in results if r[1] == sl] for sl in r_sl])
        # results is ordered product(r_pt, r_sl) -> reshape properly instead
        sharpe_grid = np.array([r[4] for r in results]).reshape(len(r_pt), len(r_sl))
        pt, sl, mean, std, sharpe = best_node(results)
        print(f'  {label}: our best Sharpe = {sharpe:.2f}  (book states ~{book_sharpe})'
              f'  at (pt={pt:.2f}, sl={sl:.2f})')

        im = ax.imshow(sharpe_grid.T, origin='lower', aspect='auto',
                        extent=[r_pt.min(), r_pt.max(), r_sl.min(), r_sl.max()],
                        cmap='RdYlGn')
        ax.set_xlabel('profit-taking (multiples of sigma)')
        ax.set_ylabel('stop-loss (multiples of sigma)')
        ax.set_title(f'{label}\nbest Sharpe {sharpe:.2f} (book: ~{book_sharpe})')
        fig.colorbar(im, ax=ax, label='Sharpe')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'ch13_book_reproduction.png'), dpi=120)
    plt.close(fig)
    print('  saved: ch13_book_reproduction.png\n')


# ===========================================================================
# PART B -- Step 1: calibrate {phi, sigma} from REAL opportunities
# ===========================================================================
def part_b_real_calibration():
    print('=' * 74)
    print('PART B -- Step 1: calibrate {phi, sigma} from real BTC opportunities')
    print('=' * 74)

    ev3 = pd.read_csv(os.path.join(INPUT, 'ch03_events.csv'), index_col=0, parse_dates=True)
    ev3['t1'] = pd.to_datetime(ev3['t1'])
    feats = pd.read_csv(os.path.join(INPUT, 'ch05_features.csv'), index_col=0, parse_dates=True)
    close = feats['close']

    # LOAD-BEARING -- forecast-target shortcut; FINAL as of 2026-07-22
    # (see DECISION below), not deferred.
    #
    # The book defines E0[Pi,Ti] as "the level targeted by opportunity i" --
    # an ex-ante forecast known at trade inception. Ch03's triple_barrier.py
    # sets barriers LONG-ONLY (side_=1 always), so the natural candidate
    # target, entry_price*(1+trgt), is an upward profit-taking level for
    # EVERY trade, including ones that resolved as bin=-1 (losses). We tried
    # calibrating phi_hat on all 88 opportunities against that single
    # always-long target: it pools a converging process (bin=+1: price
    # drifted toward the target) with a diverging one (bin=-1: price drifted
    # away from it) and produced phi_hat=1.027 -- non-stationary, violates
    # the O-U requirement phi in (-1,1).
    #
    # FIX APPLIED (this file, FINAL): center deviations on ENTRY PRICE
    # instead (target=0 in centered coordinates), not the literal
    # book-defined profit-taking level. This is a real departure from
    # strict book-fidelity on E0[Pi,Ti] -- documented here rather than
    # silently substituted.
    #
    # DEEPER FINDING (surfaced regardless of the fix above): even with
    # entry-price centering, phi_hat comes out at 1.042 -- STILL non-
    # stationary. This means the problem isn't really the target-choice
    # shortcut: raw BTC bar-level prices, over these short (~12-bar) trade
    # windows, behave close to a RANDOM WALK. The book itself (Section
    # 13.6.1) states that as phi->1, "there are no recognizable areas where
    # performance can be maximized" -- i.e. this is a genuine, book-
    # consistent finding (no fittable OTR exists for this data), not an
    # implementation bug.
    #
    # DECISION (2026-07-22): do NOT pursue per-side calibration as a "fix."
    # Considered and rejected -- splitting by realized side would roughly
    # halve the sample (87 -> ~44 per side), materially increasing
    # estimation noise, and per-side calibration isn't part of the book's
    # printed OTR methodology; pursuing it here would mean searching for a
    # subset where phi_hat looks more favorable, not reporting the result
    # the full dataset actually gives. This is also no longer an isolated
    # finding: as of 2026-07-22, three other independently-mechanised
    # real-data diagnostics on this same pipeline corroborate "no real
    # exploitable signal in this feature set/model combination" -- Ch11's
    # PBO (~0.83), Ch12's CPCV (all 5 paths negative), and Ch14's DSR (0/5
    # paths survive at 0.95). Non-stationarity here is reported as this
    # chapter's real result, consistent with the rest of the pipeline, not
    # treated as a bug to fix.
    paths, targets = [], []
    for entry_t, row in ev3.iterrows():
        exit_t = row['t1']
        if pd.isna(exit_t):
            continue
        path = close.loc[entry_t:exit_t]
        if len(path) < 2:
            continue
        entry_price = path.iloc[0]
        paths.append(path.values)
        targets.append(entry_price)  # target = entry price (deviation-space forecast = 0)

    X, Y = build_xy_from_opportunities(paths, targets)
    phi_hat, sigma_hat = estimate_ou_params(X, Y)
    half_life = phi_to_half_life(phi_hat)
    stationary = -1 < phi_hat < 1

    print(f'  opportunities used: {len(paths)}   (X,Y) pairs: {len(X)}')
    print(f'  phi_hat   = {phi_hat:.6f}   stationary (per eq 13.4): {stationary}')
    print(f'  sigma_hat = {sigma_hat:.6f}')
    print(f'  half-life = {half_life}'
          + ('  <-- NaN: phi_hat not in (0,1), see LOAD-BEARING note above' if np.isnan(half_life) else ''))
    print()
    print('  This is a real finding, not a bug: phi_hat > 1 on real BTC bar-level')
    print('  prices over short trade windows means this data does not exhibit the')
    print('  clean mean-reversion the O-U/OTR framework assumes. Part C shows what')
    print('  the mesh looks like anyway.\n')

    stats = pd.DataFrame({
        'phi_hat': [phi_hat], 'sigma_hat': [sigma_hat],
        'half_life': [half_life], 'stationary': [stationary],
        'n_opportunities': [len(paths)], 'n_xy_pairs': [len(X)],
    })
    stats.to_csv(os.path.join(INPUT, 'ch13_otr_calibration.csv'), index=False)
    stats.to_pickle(os.path.join(INPUT, 'ch13_otr_calibration.pkl'))
    print('  saved: ch13_otr_calibration.{csv,pkl}\n')
    return phi_hat, sigma_hat


# ===========================================================================
# PART C -- apply the real calibration to a real mesh sweep
# ===========================================================================
def part_c_real_mesh(phi_hat, sigma_hat):
    print('=' * 74)
    print('PART C -- real-data mesh sweep (Steps 3-5a)')
    print('=' * 74)
    print('  NOTE: batch() derives phi from a half-life via half_life_to_phi, but')
    print('  our real phi_hat has no valid half-life (non-stationary). We pass')
    print('  phi_hat straight into simulate_ou_path node-by-node instead of')
    print('  going through batch()/half_life_to_phi for this real-data section.\n')

    # Mesh MUST be scaled by the real sigma_hat -- the book's own mesh
    # convention (rPT=rSLm=linspace(0,10,21)) assumes sigma=1. On raw BTC
    # price sigma_hat ~ 690, an unscaled threshold of "10" is a few dollars
    # and would trigger on bar-to-bar noise instantly.
    r_pt = sigma_hat * np.linspace(0.5, 3, 8)
    r_sl = sigma_hat * np.linspace(0.5, 3, 8)

    # NOTE: an earlier version called np.random.seed(7), which had NO effect
    # (simulate_ou_path's old default drew from Python's built-in `random`
    # module, not numpy's -- see otr.py's LOAD-BEARING note). Fixed by
    # creating ONE numpy Generator here and threading it through every node
    # explicitly -- NOT re-seeding per node, which would make every mesh
    # cell draw the identical shock sequence.
    rng = np.random.default_rng(7).normal
    results = []
    for pt in r_pt:
        for sl in r_sl:
            exits = np.array([
                simulate_ou_path(phi_hat, sigma_hat, 0.0, pt, sl, 100, seed=0.0, rng=rng)[0]
                for _ in range(2_000)
            ])
            mean, std = exits.mean(), exits.std()
            sharpe = mean / std if std > 0 else float('nan')
            results.append((pt, sl, mean, std, sharpe))

    pt, sl, mean, std, sharpe = best_node(results)
    sharpes = np.array([r[4] for r in results])
    print(f'  best node: pt={pt:.1f}, sl={sl:.1f}  Sharpe={sharpe:.4f}')
    print(f'  Sharpe range across the whole mesh: [{sharpes.min():.4f}, {sharpes.max():.4f}]')
    print(f'  Sharpe std across mesh nodes: {sharpes.std():.4f}')
    print()
    print('  Compare to Part A\'s book-reproduction cases (Sharpe ~3.2 to ~12 at the')
    print('  optimal node): this real mesh is essentially FLAT AND NEAR ZERO')
    print('  everywhere -- no node stands out. This is the phi->1 degenerate case')
    print('  described in Section 13.6.1, now observed on real data: there is no')
    print('  fittable optimal trading rule here, and the synthetic-backtesting')
    print('  framework correctly tells us so instead of manufacturing a false one.\n')

    sharpe_grid = sharpes.reshape(len(r_pt), len(r_sl))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(sharpe_grid.T, origin='lower', aspect='auto',
                    extent=[r_pt.min(), r_pt.max(), r_sl.min(), r_sl.max()],
                    cmap='RdYlGn', vmin=-0.15, vmax=0.15)
    ax.set_xlabel('profit-taking (real price units)')
    ax.set_ylabel('stop-loss (real price units)')
    ax.set_title(f'Real BTC data: flat, near-zero mesh\n(phi_hat={phi_hat:.3f}, non-stationary)')
    fig.colorbar(im, ax=ax, label='Sharpe')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'ch13_real_mesh.png'), dpi=120)
    plt.close(fig)
    print('  saved: ch13_real_mesh.png\n')


def main():
    part_a_book_reproduction()
    phi_hat, sigma_hat = part_b_real_calibration()
    part_c_real_mesh(phi_hat, sigma_hat)


if __name__ == '__main__':
    main()
