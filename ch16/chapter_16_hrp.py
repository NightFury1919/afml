"""
Chapter 16: Machine Learning Asset Allocation -- Hierarchical Risk Parity
==========================================================================

Demo script tying together ch16/hrp/hrp.py (Snippets 16.1-16.4, the book's
algorithm) and ch16/data_loader/continuous_futures.py (this project's
real-data infrastructure) into two worked examples:

  Part A -- the book's own synthetic numerical example (Section 16.4):
            build correlated data, cluster it, quasi-diagonalize the
            correlation matrix, and allocate weight top-down.
  Part B -- the same three stages applied to six REAL, genuinely
            diversified commodity futures (gold, crude oil, corn, live
            hogs, T-bonds, GBP), comparing HRP to plain inverse-variance
            (IVP) weighting.
  Part C -- Section 16.5/16.6's Monte Carlo comparison (HRP vs CLA vs
            IVP out-of-sample), book-exact synthetic methodology
            (Snippet 16.5 / Appendix 16.A.4). CLA reference: Bailey &
            Lopez de Prado (2013). Deliberately synthetic-only, not
            adapted to real data -- see ch16/README.md for the reasoning
            (Ethan sign-off 2026-08-10). Parallelized across 4 worker
            processes via this project's existing AFML multiprocessing
            engine (utils/multiprocess.py) -- num_threads changes
            wall-clock time only, never the result (see
            monte_carlo/test_monte_carlo.py's regression test).

Why HRP instead of Markowitz's CLA?
------------------------------------
A covariance matrix is a complete graph -- every asset a potential
substitute for every other. Inverting it, which CLA and mean-variance
optimization require, means solving for every pairwise relationship at
once, so a small estimation error in any one correlation can swing the
whole solution ("Markowitz's curse", Section 16.3). HRP never inverts
the covariance matrix. It replaces the complete graph with a TREE:

  Stage 1 (tree clustering):     group similar assets together.
  Stage 2 (quasi-diagonalization): reorder the covariance matrix so
                                    similar assets sit next to each
                                    other -- no rotation, just reordering.
  Stage 3 (recursive bisection): walk the tree top-down, splitting
                                    weight between each pair of branches
                                    in inverse proportion to their
                                    variance.

No inversion is ever performed, so HRP works even on a singular or
near-singular covariance matrix -- exactly the situation CLA breaks
down in.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless: this script writes PNGs, doesn't display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Path setup (per CLAUDE.md convention: .py scripts derive their own
# root via __file__, never a hardcoded absolute path). This script lives
# at ch16/, so AFML_ROOT is one hop up. -------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
AFML_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)  # exposes hrp/ and data_loader/ as namespace packages

from hrp.hrp import (
    getHRP, getIVP, getRecBipart, getQuasiDiag, correlDist,
    generateData, plotCorrMatrix,
)
from data_loader.continuous_futures import build_continuous_price, COMMODITIES
from monte_carlo.monte_carlo import hrpMC
import scipy.cluster.hierarchy as sch
import warnings

OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Part A -- the book's own synthetic numerical example (Section 16.4)
# =============================================================================
def part_a_synthetic_example():
    print("=" * 78)
    print("PART A -- Synthetic numerical example (book's own generateData)")
    print("=" * 78)

    # ASSUMPTION, flagged for Ethan: nObs=10000, size0=5, size1=5, sigma1=.25
    # is a reasonable scale for a from-scratch demo of the book's own
    # generator (5 independent base series + 5 series each correlated with
    # a random base series). If the book's own printed figure used
    # different exact values, swap them in here -- generateData's seeded
    # numpy.random.Generator (default seed 12345, per project convention)
    # makes this fully reproducible either way.
    nObs, size0, size1, sigma1 = 10000, 5, 5, .25
    x, cols = generateData(nObs, size0, size1, sigma1)
    print(f"Simulated {nObs} observations, {size0} base series + "
          f"{size1} correlated-perturbation series = {x.shape[1]} total assets.")
    print(f"Perturbation series built from base columns: {cols.tolist()}")

    cov, corr = x.cov(), x.corr()

    # --- Stage 1+2: cluster and quasi-diagonalize, BEFORE any weighting ---
    dist = correlDist(corr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', category=sch.ClusterWarning,
            message='.*uncondensed distance matrix.*')
        link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()

    # Save before/after correlation heatmaps -- this is the whole visual
    # point of quasi-diagonalization: same information, reordered so
    # clusters become visible as blocks along the diagonal.
    plotCorrMatrix(os.path.join(OUTPUT_DIR, 'synthetic_corr_original.png'),
                    corr, labels=corr.columns)
    corr_reordered = corr.loc[sortIx, sortIx]
    plotCorrMatrix(os.path.join(OUTPUT_DIR, 'synthetic_corr_quasidiag.png'),
                    corr_reordered, labels=corr_reordered.columns)
    print("Saved before/after correlation heatmaps to "
          f"{OUTPUT_DIR}/synthetic_corr_{{original,quasidiag}}.png")

    # --- Stage 3: recursive bisection -> final weights ---
    hrp = getRecBipart(cov, sortIx).sort_index()
    ivp = pd.Series(getIVP(cov), index=cov.index).sort_index()

    print("\nHRP weights:")
    print(hrp.to_string())
    print("\nIVP weights (comparison -- ignores correlation structure "
          "entirely, weights purely by inverse variance):")
    print(ivp.to_string())
    print(f"\nBoth sum to 1: HRP={hrp.sum():.6f}, IVP={ivp.sum():.6f}")

    return hrp, ivp, corr


# =============================================================================
# Part B -- real 6-asset commodity data
# =============================================================================
def part_b_real_data():
    print()
    print("=" * 78)
    print("PART B -- Real data: 6 genuinely diversified commodity futures")
    print("=" * 78)
    print("gold, crude oil, corn, live hogs, T-bonds, British Pound -- chosen")
    print("for different macro drivers (metals/energy/grains/livestock/")
    print("rates/FX), rejecting the existing SP00-SP99 S&P futures (all one")
    print("underlying instrument, correlations ~1.0 by construction).\n")

    returns = {}
    n_contracts = {}
    for name, (prefix, folder, rescale) in COMMODITIES.items():
        non_neg, front_month = build_continuous_price(
            folder, prefix, rescale_new_format_by=rescale)
        returns[name] = non_neg['Returns']
        n_contracts[name] = front_month['Instrument'].nunique()
        print(f"  {name:10s}: {len(non_neg):5d} bars, "
              f"{n_contracts[name]:3d} contracts, "
              f"{non_neg.index.min().date()} to {non_neg.index.max().date()}")

    # Inner-join on date: only keep days where ALL SIX commodities traded.
    ret_df = pd.DataFrame(returns).dropna()
    print(f"\nAligned daily returns (all 6 commodities trading): "
          f"{len(ret_df)} days, {ret_df.index.min().date()} to "
          f"{ret_df.index.max().date()}")

    cov, corr = ret_df.cov(), ret_df.corr()
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    hrp = getHRP(cov, corr)
    ivp = pd.Series(getIVP(cov), index=cov.index).sort_index()

    print("\nHRP weights:")
    print(hrp.to_string())
    print("\nIVP weights (comparison):")
    print(ivp.to_string())

    plotCorrMatrix(os.path.join(OUTPUT_DIR, 'real_data_corr_matrix.png'),
                    corr, labels=corr.columns)
    print(f"\nSaved correlation heatmap to {OUTPUT_DIR}/real_data_corr_matrix.png")

    # Concentration check (top-2 weight share) -- HRP's edge over IVP shows
    # up more when there's meaningful correlation CLUSTERING to exploit.
    # With N=6 genuinely diverse (low-correlation) assets, that clustering
    # structure barely exists, so HRP and IVP end up close -- see README
    # for why this is a legitimate real-data finding, not a bug.
    top2_hrp = hrp.sort_values(ascending=False).iloc[:2].sum()
    top2_ivp = ivp.sort_values(ascending=False).iloc[:2].sum()
    print(f"\nTop-2 weight concentration -- HRP: {top2_hrp:.3f}, "
          f"IVP: {top2_ivp:.3f}")

    return hrp, ivp, corr, ret_df


# =============================================================================
# Part C -- Section 16.5/16.6 Monte Carlo (HRP vs CLA vs IVP), book-exact
# =============================================================================
def part_c_monte_carlo():
    print()
    print("=" * 78)
    print("PART C -- Section 16.5/16.6 Monte Carlo (HRP vs CLA vs IVP)")
    print("=" * 78)
    print("Book-exact synthetic Monte Carlo (Snippet 16.5 / Appendix 16.A.4).")
    print("Deliberately NOT adapted to real data -- this experiment's whole")
    print("point is to inject KNOWN, controlled shocks (one common, one")
    print("idiosyncratic) and observe each method's response; that requires")
    print("synthetic control by construction, the same category of exception")
    print("already sanctioned for Ch08. See ch16/README.md for the full")
    print("reasoning and Ethan's sign-off.\n")
    print("Book defaults kept exactly as printed: numIters=10000, nObs=520")
    print("(2yrs daily), sLength=260 (1yr lookback), rebal=22 (~monthly).")
    print("Parallelized across 4 worker processes (this project's documented")
    print("multiprocessing sweet spot -- see CLAUDE.md), via the existing")
    print("AFML multiprocessing engine (utils/multiprocess.py), mirroring")
    print("ch04/sample_weights/monte_carlo.py's own established pattern.")
    print("num_threads only changes wall-clock time, never the result --")
    print("see monte_carlo/test_monte_carlo.py's num_threads regression test.\n")

    csv_path = os.path.join(OUTPUT_DIR, 'monte_carlo_stats.csv')
    stats, summary = hrpMC(
        numIters=10000, nObs=520, size0=5, size1=5, mu0=0, sigma0=1e-2,
        sigma1F=.25, sLength=260, rebal=22, random_state=12345,
        num_threads=4, output_csv_path=csv_path, verbose=True)

    print("\nOut-of-sample variance comparison (book's headline figures --")
    print("book reports sigma^2_CLA=0.1157, sigma^2_IVP=0.0928, ")
    print("sigma^2_HRP=0.0671; CLA 72.47% greater variance than HRP,")
    print("IVP 38.24% greater variance than HRP):\n")
    print(summary.to_string())
    print(f"\nPer-iteration results saved to {csv_path}")

    return stats, summary


if __name__ == '__main__':
    hrp_synth, ivp_synth, corr_synth = part_a_synthetic_example()
    hrp_real, ivp_real, corr_real, returns_real = part_b_real_data()
    mc_stats, mc_summary = part_c_monte_carlo()

# =============================================================================
# Real-machine pytest results
# =============================================================================
# This driver script has no dedicated test file of its own (consistent with
# other chapter drivers, e.g. ch15/chapter_15_strategy_risk.py) -- the
# algorithms it calls are covered by ch16/hrp/test_hrp.py (24/24),
# ch16/data_loader/test_continuous_futures.py (17/17), ch16/cla/test_cla.py
# (52/52), and ch16/monte_carlo/test_monte_carlo.py (14/14). hrp/ and
# data_loader/ real-machine confirmed 2026-07-31; cla/ and monte_carlo/
# passed 52/52 and 14/14 respectively in Claude's sandbox (numpy 2.4.4 /
# Python 3.12.3) -- STILL NEEDS real-machine (mlfinlab: numpy 1.23.5 /
# Python 3.10.20) two-pass confirmation, both test suites, from repo root
# AND from inside their own folders.
# STILL NEEDS: an actual end-to-end run of this full driver script
# (Parts A, B, AND C) against the real six-commodity input_data/ AND the
# real 10,000-iteration Monte Carlo, on the real mlfinlab machine --
# Ethan's next action. Part C in particular has not been run at book
# scale anywhere yet (only sandbox-verified at small numIters, e.g. 20
# iterations, which already reproduces the book's DIRECTIONAL finding:
# var_CLA > var_IVP > var_HRP).
