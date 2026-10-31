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
# other chapter drivers) -- the algorithms it calls are covered by:
#   hrp/test_hrp.py                                   24 tests
#   data_loader/test_continuous_futures.py            17 tests
#   cla/test_cla.py                                   52 tests
#   monte_carlo/test_monte_carlo.py                   16 tests
#   total                                            109 tests
#
# Real machine (mlfinlab env, Python 3.10.20, pytest 9.0.3), two-pass check re-run 2026-10-31:
#   repo root:  109 passed in 18.15s
#   inside cla/                 52 passed in 0.30s
#   inside data_loader/         17 passed in 7.90s
#   inside hrp/                 24 passed in 1.13s
#   inside monte_carlo/         16 passed in 3.62s
# Earlier confirmations: hrp/ and data_loader/ on 2026-07-31; cla/ and monte_carlo/ on 2026-08-10.
# This script itself was confirmed end-to-end on the real machine on 2026-08-10: Parts A, B, AND C, including the full book-scale Monte Carlo (numIters=10000, num_threads=4). See README.md.
#
# Full output, run from the repo root:
#
# (mlfinlab) PS C:\ws\AFML> python -m pytest ch16 -v
# ============================= test session starts =============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML
# collecting ... collected 109 items
#
# ch16/cla/test_cla.py::TestTurningPoints::test_number_of_turning_points_matches_table PASSED [  0%]
# ch16/cla/test_cla.py::TestTurningPoints::test_initial_state_coincides_with_first_turning_point PASSED [  1%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2[0] PASSED [  2%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2[1] PASSED [  3%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2[2] PASSED [  4%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2[3] PASSED [  5%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2[9] PASSED [  6%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2_up_to_tied_lambda_reordering[4] PASSED [  7%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2_up_to_tied_lambda_reordering[5] PASSED [  8%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2_up_to_tied_lambda_reordering[6] PASSED [  9%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2_up_to_tied_lambda_reordering[7] PASSED [ 10%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_match_table_2_up_to_tied_lambda_reordering[8] PASSED [ 11%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[0] PASSED [ 11%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[1] PASSED [ 12%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[2] PASSED [ 13%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[3] PASSED [ 14%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[4] PASSED [ 15%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[5] PASSED [ 16%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[6] PASSED [ 17%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[7] PASSED [ 18%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[8] PASSED [ 19%]
# ch16/cla/test_cla.py::TestTurningPoints::test_return_matches_table_2[9] PASSED [ 20%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[0] PASSED [ 21%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[1] PASSED [ 22%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[2] PASSED [ 22%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[3] PASSED [ 23%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[4] PASSED [ 24%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[5] PASSED [ 25%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[6] PASSED [ 26%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[7] PASSED [ 27%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[8] PASSED [ 28%]
# ch16/cla/test_cla.py::TestTurningPoints::test_risk_matches_table_2[9] PASSED [ 29%]
# ch16/cla/test_cla.py::TestTurningPoints::test_first_turning_point_lambda_is_none PASSED [ 30%]
# ch16/cla/test_cla.py::TestTurningPoints::test_last_turning_point_lambda_is_zero PASSED [ 31%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[1] PASSED [ 32%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[2] PASSED [ 33%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[3] PASSED [ 33%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[4] PASSED [ 34%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[5] PASSED [ 35%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[6] PASSED [ 36%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[7] PASSED [ 37%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[8] PASSED [ 38%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_matches_table_2[9] PASSED [ 39%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_sum_to_one_at_every_turning_point PASSED [ 40%]
# ch16/cla/test_cla.py::TestTurningPoints::test_weights_respect_bounds_at_every_turning_point PASSED [ 41%]
# ch16/cla/test_cla.py::TestTurningPoints::test_lambda_is_monotonically_decreasing PASSED [ 42%]
# ch16/cla/test_cla.py::TestGetMinVar::test_matches_paper_prose PASSED     [ 43%]
# ch16/cla/test_cla.py::TestGetMinVar::test_coincides_with_last_turning_point PASSED [ 44%]
# ch16/cla/test_cla.py::TestGetMaxSR::test_matches_paper_prose PASSED      [ 44%]
# ch16/cla/test_cla.py::TestGetMaxSR::test_weights_sum_to_one PASSED       [ 45%]
# ch16/cla/test_cla.py::TestDiagonalCovarianceMatchesIVP::test_min_var_equals_ivp_on_diagonal_covariance PASSED [ 46%]
# ch16/cla/test_cla.py::TestInitAlgo::test_first_free_asset_is_highest_return PASSED [ 47%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_no_header_format_parses_correctly PASSED [ 48%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_header_format_parses_correctly PASSED [ 49%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_leading_zero_january_dates_preserved PASSED [ 50%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_rescale_applies_to_header_format PASSED [ 51%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_rescale_none_leaves_header_format_unchanged PASSED [ 52%]
# ch16/data_loader/test_continuous_futures.py::TestLoadContract::test_rescale_does_not_apply_to_no_header_format PASSED [ 53%]
# ch16/data_loader/test_continuous_futures.py::TestBuildFrontMonthSeries::test_selects_highest_open_interest_contract PASSED [ 54%]
# ch16/data_loader/test_continuous_futures.py::TestBuildFrontMonthSeries::test_forward_fills_isolated_gap_in_dominant_contract PASSED [ 55%]
# ch16/data_loader/test_continuous_futures.py::TestBuildFrontMonthSeries::test_long_gaps_only_partially_filled_then_dropped PASSED [ 55%]
# ch16/data_loader/test_continuous_futures.py::TestBuildFrontMonthSeries::test_returns_exactly_one_row_per_date PASSED [ 56%]
# ch16/data_loader/test_continuous_futures.py::TestBuildContinuousPrice::test_output_starts_at_one_and_stays_positive PASSED [ 57%]
# ch16/data_loader/test_continuous_futures.py::TestBuildContinuousPrice::test_single_contract_no_roll_reproduces_its_own_returns PASSED [ 58%]
# ch16/data_loader/test_continuous_futures.py::TestRealDataIntegration::test_all_six_commodities_present PASSED [ 59%]
# ch16/data_loader/test_continuous_futures.py::TestRealDataIntegration::test_no_nans_or_infs PASSED [ 60%]
# ch16/data_loader/test_continuous_futures.py::TestRealDataIntegration::test_no_extreme_outlier_returns PASSED [ 61%]
# ch16/data_loader/test_continuous_futures.py::TestRealDataIntegration::test_correlation_matrix_valid PASSED [ 62%]
# ch16/data_loader/test_continuous_futures.py::TestRealDataIntegration::test_reasonable_sample_size PASSED [ 63%]
# ch16/hrp/test_hrp.py::TestCorrelDist::test_matches_book_example_16_1 PASSED [ 64%]
# ch16/hrp/test_hrp.py::TestCorrelDist::test_diagonal_is_zero PASSED       [ 65%]
# ch16/hrp/test_hrp.py::TestCorrelDist::test_perfect_correlation_gives_zero_distance PASSED [ 66%]
# ch16/hrp/test_hrp.py::TestCorrelDist::test_perfect_anticorrelation_gives_max_distance PASSED [ 66%]
# ch16/hrp/test_hrp.py::TestBookLinkage::test_d_tilde_matches_book_example_16_2 PASSED [ 67%]
# ch16/hrp/test_hrp.py::TestBookLinkage::test_merge_sequence_matches_book_examples_16_3_to_16_6 PASSED [ 68%]
# ch16/hrp/test_hrp.py::TestGetIVP::test_hand_computed_three_asset PASSED  [ 69%]
# ch16/hrp/test_hrp.py::TestGetIVP::test_sums_to_one PASSED                [ 70%]
# ch16/hrp/test_hrp.py::TestGetIVP::test_equal_variance_gives_equal_weight PASSED [ 71%]
# ch16/hrp/test_hrp.py::TestGetIVP::test_ignores_off_diagonal_by_design PASSED [ 72%]
# ch16/hrp/test_hrp.py::TestGetClusterVar::test_hand_computed_uncorrelated_pair PASSED [ 73%]
# ch16/hrp/test_hrp.py::TestGetClusterVar::test_single_item_cluster_returns_its_own_variance PASSED [ 74%]
# ch16/hrp/test_hrp.py::TestGetClusterVar::test_matrix_slice_ignores_items_outside_cluster PASSED [ 75%]
# ch16/hrp/test_hrp.py::TestGetQuasiDiag::test_matches_book_example_hand_trace PASSED [ 76%]
# ch16/hrp/test_hrp.py::TestGetQuasiDiag::test_returns_a_permutation_of_all_original_items PASSED [ 77%]
# ch16/hrp/test_hrp.py::TestGetRecBipart::test_matches_book_example_hand_trace PASSED [ 77%]
# ch16/hrp/test_hrp.py::TestGetRecBipart::test_weights_sum_to_one_and_are_non_negative PASSED [ 78%]
# ch16/hrp/test_hrp.py::TestGetRecBipart::test_single_asset_gets_full_weight PASSED [ 79%]
# ch16/hrp/test_hrp.py::TestGetHRPEndToEnd::test_full_pipeline_weights_sum_to_one PASSED [ 80%]
# ch16/hrp/test_hrp.py::TestGetHRPEndToEnd::test_returns_series_indexed_by_original_labels PASSED [ 81%]
# ch16/hrp/test_hrp.py::TestGetHRPEndToEnd::test_perfectly_correlated_assets_split_evenly_within_cluster PASSED [ 82%]
# ch16/hrp/test_hrp.py::TestGenerateData::test_shape PASSED                [ 83%]
# ch16/hrp/test_hrp.py::TestGenerateData::test_seeded_reproducibility PASSED [ 84%]
# ch16/hrp/test_hrp.py::TestGenerateData::test_perturbed_columns_correlate_with_their_source PASSED [ 85%]
# ch16/monte_carlo/test_monte_carlo.py::TestGenerateData::test_shape PASSED [ 86%]
# ch16/monte_carlo/test_monte_carlo.py::TestGenerateData::test_deterministic_given_seed PASSED [ 87%]
# ch16/monte_carlo/test_monte_carlo.py::TestGenerateData::test_common_shock_values_injected_exactly PASSED [ 88%]
# ch16/monte_carlo/test_monte_carlo.py::TestGenerateData::test_idiosyncratic_shock_only_hits_one_column PASSED [ 88%]
# ch16/monte_carlo/test_monte_carlo.py::TestGenerateData::test_different_seeds_give_different_paths PASSED [ 89%]
# ch16/monte_carlo/test_monte_carlo.py::TestGetHRPWrapper::test_matches_hrp_module_getHRP PASSED [ 90%]
# ch16/monte_carlo/test_monte_carlo.py::TestGetCLA::test_returns_valid_weight_vector PASSED [ 91%]
# ch16/monte_carlo/test_monte_carlo.py::TestGetCLA::test_min_var_equals_ivp_on_diagonal_covariance PASSED [ 92%]
# ch16/monte_carlo/test_monte_carlo.py::TestGetCLA::test_kargs_absorbs_corr_uniformly_with_other_methods PASSED [ 93%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_output_shape_and_columns PASSED [ 94%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_var_vs_hrp_column_is_zero_for_hrp_itself PASSED [ 95%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_deterministic_given_seed PASSED [ 96%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_different_seeds_give_different_results PASSED [ 97%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_num_threads_does_not_change_results PASSED [ 98%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_pointers_are_genuinely_out_of_sample PASSED [ 99%]
# ch16/monte_carlo/test_monte_carlo.py::TestHrpMC::test_output_csv_written_when_path_given PASSED [100%]
#
# ============================ 109 passed in 18.15s =============================
# ---------------------------------------------------------------------------
