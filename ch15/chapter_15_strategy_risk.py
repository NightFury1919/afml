"""
Chapter 15 -- Understanding Strategy Risk.

Real-data-first demo, three parts:
  A. Symmetric payouts (Sec 15.2): reproduce the book's own worked
     examples in closed form, then cross-validate with a direct Monte
     Carlo simulation (Snippet 15.1, ported to a seeded Generator).
  B. Asymmetric payouts (Sec 15.3): reproduce the book's own worked
     example (theta=1.173, implied precision .72) and the p_theta*=0=2/3
     special case.
  C. The probability of strategy failure (Sec 15.4), on REAL data: apply
     probFailure to Ch3's actual 88 real BTC/TUSD triple-barrier bet
     outcomes -- the strategy's own realized {pi_t} series, not a
     synthetic placeholder.

Path convention: this .py script derives its own root via __file__ (works
for anyone who clones the repo, any OS, any username). The paired
notebook uses a hardcoded AFML_ROOT instead, per CLAUDE.md.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, 'strategy_risk'))

INPUT_DATA = os.path.join(ROOT, 'input_data')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import symmetric as sym
import asymmetric as asym
import algorithm as algo


# =============================================================================
# Part A -- Symmetric payouts (Sec 15.2)
# =============================================================================
print("=" * 78)
print("PART A -- Symmetric payouts")
print("=" * 78)

# Book's own worked example: p=.55 gives a per-sqrt(n) Sharpe coefficient
# of 0.1005, so hitting an annualized Sharpe of 2 needs ~396 bets/year.
p_example = .55
coefficient = sym.sharpe_ratio_symmetric(p_example, n=1)
n_for_theta_2 = (2 / coefficient) ** 2
print(f"\n[15.2] p={p_example}: per-sqrt(n) Sharpe coefficient = {coefficient:.4f} "
      f"(book: 0.1005)")
print(f"       bets/year needed for annualized Sharpe=2: {n_for_theta_2:.1f} (book: 396)")

# Book's other worked example: weekly bets (n=52) need p=0.6336 for Sharpe=2.
p_weekly = sym.implied_precision_symmetric(n=52, tSR=2)
print(f"\n[15.2] Weekly bets (n=52) need precision p={p_weekly:.4f} for Sharpe=2 "
      f"(book: 0.6336)")

# Monte Carlo cross-check of Snippet 15.1 (ported to a seeded Generator,
# 1,000,000 draws matching the book -- Python 2's xrange/print updated).
rng = np.random.default_rng(2026)
mc_mean, mc_std, mc_sharpe = sym.simulate_symmetric_sharpe(p_example, n_draws=1_000_000, rng=rng)
print(f"\n[Snippet 15.1] Monte Carlo (1e6 draws, seed=2026): "
      f"mean={mc_mean:.4f}, std={mc_std:.4f}, sharpe={mc_sharpe:.4f} "
      f"(closed form: {coefficient:.4f})")

# Figure 15.1: Sharpe ratio vs precision, for a few betting frequencies.
p_grid = np.linspace(0.40, 0.60, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for n in [0, 25, 50, 75, 100]:
    if n == 0:
        theta_grid = np.zeros_like(p_grid)
    else:
        theta_grid = [sym.sharpe_ratio_symmetric(p, n) for p in p_grid]
    ax.plot(p_grid, theta_grid, label=f'n={n}')
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_xlabel('precision (p)')
ax.set_ylabel('Sharpe ratio')
ax.set_title('Figure 15.1 -- Sharpe ratio vs. precision, by betting frequency')
ax.legend()
fig.tight_layout()
fig_path = os.path.join(HERE, 'ch15_figure_15_1_sharpe_vs_precision.png')
fig.savefig(fig_path, dpi=100)
plt.close(fig)
print(f"\nSaved: {fig_path}")


# =============================================================================
# Part B -- Asymmetric payouts (Sec 15.3)
# =============================================================================
print()
print("=" * 78)
print("PART B -- Asymmetric payouts")
print("=" * 78)

SL, PT, FREQ = -.01, .005, 260  # book's own running example

theta_asym = asym.binSR(SL, PT, FREQ, p=.7)
print(f"\n[15.3] sl={SL}, pt={PT}, freq={FREQ}, p=.7 -> theta={theta_asym:.4f} "
      f"(book: 1.173)")

p_for_theta_2 = asym.binHR(SL, PT, FREQ, tSR=2)
print(f"[15.3] Same params, implied precision for theta=2: p={p_for_theta_2:.4f} "
      f"(book: .72)")

p_breakeven = asym.binHR(SL, PT, FREQ, tSR=0)
print(f"\n[15.4] Break-even precision (theta<=0 below this): "
      f"p_theta*=0={p_breakeven:.4f} (book: 2/3={2/3:.4f})")

# Figure 15.2 -- implied precision heat-map (n vs. sl, pt=.1, tSR=1.5)
sl_grid = np.linspace(-0.01, -0.001, 40)
freq_grid = np.arange(10, 100, 4)
heat = np.zeros((len(sl_grid), len(freq_grid)))
for i, sl in enumerate(sl_grid):
    for j, freq in enumerate(freq_grid):
        heat[i, j] = asym.binHR(sl, pt=0.1, freq=freq, tSR=1.5)
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heat, aspect='auto', cmap='gray',
                extent=[freq_grid.min(), freq_grid.max(), sl_grid.min(), sl_grid.max()],
                origin='upper')
ax.set_xlabel('frequency (n)')
ax.set_ylabel('stop loss (pi_-)')
ax.set_title('Figure 15.2 -- implied precision, pi_+=0.1, theta*=1.5')
fig.colorbar(im, ax=ax, label='implied precision p')
fig.tight_layout()
fig_path2 = os.path.join(HERE, 'ch15_figure_15_2_implied_precision.png')
fig.savefig(fig_path2, dpi=100)
plt.close(fig)
print(f"\nSaved: {fig_path2}")


# =============================================================================
# Part C -- Probability of strategy failure, on REAL data (Sec 15.4)
# =============================================================================
print()
print("=" * 78)
print("PART C -- Real-data strategy risk (Ch3's actual 88 BTC/TUSD bets)")
print("=" * 78)

events_path = os.path.join(INPUT_DATA, 'ch03_events.csv')
events = pd.read_csv(events_path, index_col=0, parse_dates=[0, 1])
ret = events['ret'].values
print(f"\nLoaded {len(events)} real triple-barrier bet outcomes from {events_path}")

elapsed = events['t1'].max() - events.index.min()
elapsed_years = elapsed.total_seconds() / (365.25 * 24 * 3600)
freq_real = len(events) / elapsed_years
print(f"Elapsed window: {elapsed} = {elapsed_years:.4f} years")
print(f"Annualized frequency n = T/y = {freq_real:.1f} bets/year")
print("NOTE: only ~29 real days of trade data are available (Ch2's single-month "
      "\nBTC/TUSD tape), so annualizing multiplies the apparent bet rate by "
      f"~{365.25/29:.1f}x.\nTreat 'annualized' figures below as a genuine but "
      "heavily-extrapolated real-data result,\nnot as a full-year track record.")

n_pos = (ret > 0).sum()
p_bar = n_pos / len(ret)
r_pos, r_neg = ret[ret > 0].mean(), ret[ret <= 0].mean()
print(f"\nRealized precision p_bar = {p_bar:.4f} ({n_pos} of {len(events)} bets positive)")
print(f"Mean winning return (pi_+) = {r_pos:.4f}, mean losing return (pi_-) = {r_neg:.4f}")

theta_realized = np.sqrt(freq_real) * ret.mean() / ret.std()
print(f"Realized (empirical) annualized Sharpe on this series: {theta_realized:.4f}")
print("NOTE: this looks striking, but it's inflated by the same short-window "
      "extrapolation\nabove (sqrt(freq) scales it up ~3.5x vs. a true full-year "
      "sample) -- probFailure below\nasks the more meaningful question of how "
      "reliable that precision actually is.")

print(f"\n{'tSR':>6} {'p_theta*':>10} {'P[fail]':>10}")
for tSR in [0.5, 1.0, 2.0]:
    risk = algo.probFailure(ret, freq_real, tSR)
    thres_p = asym.binHR(r_neg, r_pos, freq_real, tSR)
    print(f"{tSR:>6.1f} {thres_p:>10.4f} {risk:>10.4f}")

print("\nBook's own rule of thumb: disregard strategies with P[fail] > .05 as too "
      "risky.\nAt EVERY target Sharpe tested here (0.5, 1.0, 2.0), P[fail] is "
      "~0.45-0.47 --\nfar above that threshold. This corroborates, on an "
      "independent method, the same\n'no reliable exploitable signal in this "
      "feature set/model' finding from Ch11's PBO\n(~0.83), Ch12's CPCV (all 5 "
      "paths negative), Ch13's non-stationary O-U calibration,\nand Ch14's DSR "
      "(0/5 paths survive) -- strategy risk here is not a modeling artifact,\nit's "
      "consistent with everything else this pipeline has found on this data.")


# =============================================================================
# TDD results -- embedded per project convention, after tests passed
# =============================================================================
# REAL-MACHINE CONFIRMED (Python 3.10.20, pytest 9.0.3, mlfinlab env,
# 2026-07-24) -- identical pass count and warnings to the sandbox run. See
# strategy_risk/symmetric.py, asymmetric.py, algorithm.py for the same
# full pytest -v listing (34 tests, all three modules run together since
# they're interdependent).
#
# ============================= test session starts ==============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML\ch15\strategy_risk
# collected 34 items
#
# test_algorithm.py::TestProbFailureHandTraced::test_hand_traced_every_step PASSED [  2%]
# test_algorithm.py::TestProbFailureHandTraced::test_all_positive_returns PASSED [  5%]
# test_algorithm.py::TestProbFailureSeededRegression::test_book_parameters_seeded_regression PASSED [  8%]
# test_algorithm.py::TestProbFailureSeededRegression::test_fix_actually_changes_result_vs_literal_book_code PASSED [ 11%]
# test_algorithm.py::TestMixGaussians::test_output_length_matches_nObs PASSED [ 14%]
# test_algorithm.py::TestMixGaussians::test_reproducible_with_seeded_generator PASSED [ 17%]
# test_algorithm.py::TestMixGaussians::test_prob1_one_gives_pure_first_component PASSED [ 20%]
# test_asymmetric.py::TestBinSR::test_book_worked_example PASSED           [ 23%]
# test_asymmetric.py::TestBinSR::test_reduces_to_symmetric_case PASSED     [ 26%]
# test_asymmetric.py::TestBinSR::test_precision_half_gives_expected_value_only_pull PASSED [ 29%]
# test_asymmetric.py::TestBinHR::test_book_worked_example_theta_2 PASSED   [ 32%]
# test_asymmetric.py::TestBinHR::test_p_theta_star_zero_special_case PASSED [ 35%]
# test_asymmetric.py::TestBinHR::test_roundtrip_with_binsr PASSED          [ 38%]
# test_asymmetric.py::TestBinHR::test_negative_discriminant_raises PASSED  [ 41%]
# test_asymmetric.py::TestBinFreq::test_roundtrip_recovers_book_frequency PASSED [ 44%]
# test_asymmetric.py::TestBinFreq::test_roundtrip_with_binsr_general PASSED [ 47%]
# test_asymmetric.py::TestBinFreq::test_higher_precision_needs_fewer_bets PASSED [ 50%]
# test_asymmetric.py::TestBinFreq::test_extraneous_below_breakeven_returns_none PASSED [ 52%]
# test_asymmetric.py::TestBinFreq::test_at_or_above_breakeven_precision_has_valid_solution PASSED [ 55%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_book_worked_example_p55 PASSED [ 58%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_book_worked_example_396_bets PASSED [ 61%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_p_half_gives_zero_sharpe PASSED [ 64%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_symmetric_around_half PASSED [ 67%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_n_zero_gives_zero_sharpe PASSED [ 70%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_p_out_of_range_raises PASSED [ 73%]
# test_symmetric.py::TestSharpeRatioSymmetric::test_negative_n_raises PASSED [ 76%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_book_worked_example_weekly_bets PASSED [ 79%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_roundtrip_against_sharpe_ratio_symmetric PASSED [ 82%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_higher_n_needs_lower_precision PASSED [ 85%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_zero_frequency_zero_target_raises PASSED [ 88%]
# test_symmetric.py::TestImpliedPrecisionSymmetric::test_negative_n_raises PASSED [ 91%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_cross_validated_against_closed_form PASSED [ 94%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_reproducible_with_seeded_generator PASSED [ 97%]
# test_symmetric.py::TestSimulateSymmetricSharpe::test_p_one_gives_std_zero_and_nan_sharpe PASSED [100%]
#
# ======================== 34 passed, 2 warnings in 1.99s ========================
# (2 warnings: intentional NaN-propagation on an all-positive-returns edge
#  case in test_algorithm.py, documented in that test's own comment.)
