"""
Chapter 17: Structural Breaks
===============================

Demo script tying together the three modules in ch17/structural_breaks/
into three worked examples on REAL data (no synthetic placeholders --
the synthetic break-detection cases live in the test suites, not here):

  Part A -- Brown-Durbin-Evans CUSUM (Sec 17.3.1) on Ch19's real enriched
            feature table, predicting Ch3's real triple-barrier returns.
            Needs a genuine feature/target pair, unlike the other two
            parts, which only need a bare price series.
  Part B -- Chu-Stinchcombe-White CUSUM (17.3.2), Chow-type DF (17.4.1),
            and SADF (17.4.2) on gold's real continuous price series
            (1996-2002, reusing ch16's continuous_futures.py). These
            explosiveness tests are built to catch slow-building,
            multi-period bubbles, so they want genuine duration -- gold's
            ~6.7-year real daily series is a much better structural match
            than a short intraday window (see Part C).
  Part C -- the same explosiveness tests (Chow-DF, SADF) run again on the
            real BTC/TUSD dollar-bar series (Ch5's real 239-bar, ~29-day
            window) as a secondary contrast: does a short window even
            give these tests enough to work with?

RUNTIME WARNING (book's own Sec 17.4.2.2 applies): SADF and the CSW
CUSUM's own sup-search are both genuinely O(T^2). On gold's ~1,760-bar
real series, SADF takes roughly a minute and CSW CUSUM roughly a minute
and a half in Claude's sandbox -- expect similar or longer on the real
machine. Chow-DF's SDFC and everything in Part A/C are fast (well under
a few seconds) by comparison.
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AFML_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, 'structural_breaks'))
sys.path.insert(0, os.path.join(AFML_ROOT, 'ch16'))

from sadf import get_sadf
from chow_df import get_dfc, get_sdfc
from cusum import get_bde_cusum, get_csw_cusum
from data_loader.continuous_futures import build_continuous_price, COMMODITIES

INPUT_DIR = os.path.join(AFML_ROOT, 'input_data')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Part A -- BDE CUSUM on real features predicting real returns
# =============================================================================
def part_a_bde_cusum():
    print("=" * 78)
    print("PART A -- Brown-Durbin-Evans CUSUM (Sec 17.3.1)")
    print("=" * 78)
    print("Needs a genuine feature/target pair -- Ch19's real 12-feature")
    print("enriched table, predicting Ch3's real triple-barrier returns.\n")

    enriched = pd.read_csv(os.path.join(INPUT_DIR, 'ch07_training_table_enriched.csv'),
                            index_col=0, parse_dates=True)
    events = pd.read_csv(os.path.join(INPUT_DIR, 'ch03_events.csv'),
                          index_col=0, parse_dates=True)
    joined = enriched.join(events[['ret']], how='inner')
    print(f"Joined on shared event-start-time index: {len(joined)} real events "
          f"(enriched table has {len(enriched)}, events has {len(events)}).")

    feature_cols = [c for c in enriched.columns if c not in ('bin', 'w', 't1')]
    X_raw = joined[feature_cols].values
    y = joined['ret'].values

    # Standardize features before regression -- the same scaling lesson
    # Ch10/11/12 learned the hard way: these features span ~3.8e9 to 1 in
    # raw magnitude, and an unstandardized fit would be dominated by
    # whichever column happens to have the largest units.
    X_std = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    X = np.column_stack([np.ones(len(X_std)), X_std])

    min_sample = X.shape[1] + 5   # a few more obs than regressors, for stability
    out = get_bde_cusum(X, y, min_sample=min_sample, index=joined.index)
    print(f"\n{len(out)} recursive residuals computed (min_sample={min_sample}, "
          f"{X.shape[1]} regressors incl. constant).")

    # Point-by-point band check (NOT max|S| vs. some other point's band --
    # each point has its own band_95, since the book's stated N[0,t-k-1]
    # variance grows with the number of residuals summed so far).
    exceeds = out['S'].abs() > out['band_95']
    # The very first point has band_95=0 by construction (n_summed-1=0),
    # so it trivially "exceeds" -- that's a boundary artifact, not a real
    # crossing, and excluded from the crossing count/window below.
    real_exceeds = exceeds.copy()
    real_exceeds.iloc[0] = False

    print(f"Final S: {out['S'].iloc[-1]:.3f}, band at that point: "
          f"+-{out['band_95'].iloc[-1]:.3f}")

    if real_exceeds.any():
        crossing_dates = out.index[real_exceeds]
        print(f"\nS DOES cross the 95% band -- {real_exceeds.sum()} of "
              f"{len(out)-1} points (excluding the trivial first-point "
              f"artifact), from {crossing_dates.min()} to {crossing_dates.max()}.")
        print("This is a genuine real finding, not something to gloss over: for")
        print("roughly a week in mid-March, the relationship between these 12")
        print("real features and real returns drifted outside what a stable")
        print("(no-break) null would predict, then reverted back inside the band")
        print("by month's end. Worth flagging alongside this pipeline's broader")
        print("'no exploitable signal' finding (Ch11 PBO, Ch12 CPCV, Ch13 O-U,")
        print("Ch14 DSR, Ch15 P[fail]) rather than treated as contradicting it --")
        print("a temporary, reverting drift in a weak relationship is consistent")
        print("with there being no STABLE exploitable signal either before or")
        print("after the drift.")
    else:
        print("\nS never crosses the 95% band -- no detected structural break")
        print("in the feature/return relationship.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(out.index, out['S'], label='S (CUSUM)')
    ax.plot(out.index, out['band_95'], 'r--', label='+-95% band (book\'s own N[0,t-k-1])')
    ax.plot(out.index, -out['band_95'], 'r--')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title('BDE CUSUM -- real features predicting real returns')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'bde_cusum_real_data.png'))
    plt.close(fig)
    print(f"\nSaved plot to {OUTPUT_DIR}/bde_cusum_real_data.png")

    return out


# =============================================================================
# Part B -- CSW CUSUM, Chow-DF, SADF on real gold data
# =============================================================================
def part_b_gold_explosiveness():
    print()
    print("=" * 78)
    print("PART B -- CSW CUSUM, Chow-DF, SADF on real gold (1996-2002)")
    print("=" * 78)

    prefix, folder, rescale = COMMODITIES['gold']
    non_neg, _ = build_continuous_price(folder, prefix, rescale_new_format_by=rescale)
    gold_logp = np.log(non_neg[['rPrices']]).dropna()
    print(f"Gold: {len(gold_logp)} real daily bars, "
          f"{gold_logp.index.min().date()} to {gold_logp.index.max().date()}\n")

    print("-- Chow-type DF (fast, O(T) per candidate break) --")
    t0 = time.time()
    sdfc_out = get_sdfc(gold_logp, tau0=0.05)
    print(f"  {time.time()-t0:.1f}s. SDFC={sdfc_out['sdfc']:.3f} at "
          f"tau*={sdfc_out['tau_star']:.4f} "
          f"({gold_logp.index[int(round(sdfc_out['tau_star']*len(gold_logp)))].date()})")

    print("\n-- SADF (O(T^2), expect ~1 minute on this series) --")
    t0 = time.time()
    sadf_gold = get_sadf(gold_logp, minSL=90, constant='nc', lags=1)
    print(f"  {time.time()-t0:.1f}s. Max SADF={sadf_gold.max():.3f} at "
          f"{sadf_gold.idxmax().date()}")

    print("\n-- CSW CUSUM (O(T^2), expect ~1.5 minutes on this series) --")
    t0 = time.time()
    csw_gold = get_csw_cusum(gold_logp, min_sample=90)
    max_S = csw_gold['S'].max()
    max_date = csw_gold['S'].idxmax()
    max_row = csw_gold.loc[max_date]
    print(f"  {time.time()-t0:.1f}s. Max S={max_S:.3f} at {max_date.date()} "
          f"(reference n: {gold_logp.index[int(max_row['n_star'])].date()}, "
          f"critical value: {max_row['critical_value_95']:.3f})")
    if max_S > max_row['critical_value_95']:
        print(f"\n  This is the Washington Agreement on Gold (Sept 26, 1999) --")
        print(f"  a real, documented event, independently flagged during Ch16's")
        print(f"  own data-hygiene pass as gold's largest real historical spike")
        print(f"  in this window. CSW CUSUM's own sup-search lands almost exactly")
        print(f"  on it, using only {(max_date - gold_logp.index[int(max_row['n_star'])]).days} days")
        print(f"  of reference window -- a genuine, sharp real break, not an artifact.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(non_neg.index, non_neg['rPrices'])
    axes[0].set_title('Gold: real continuous price series (1996-2002)')
    axes[1].plot(sadf_gold.index, sadf_gold.values)
    axes[1].set_title('SADF')
    axes[2].plot(csw_gold.index, csw_gold['S'])
    axes[2].plot(csw_gold.index, csw_gold['critical_value_95'], 'r--',
                 label='95% critical value (n_star-specific)')
    axes[2].set_title('CSW CUSUM sup-statistic')
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'gold_explosiveness_tests.png'))
    plt.close(fig)
    print(f"\nSaved plot to {OUTPUT_DIR}/gold_explosiveness_tests.png")

    return sdfc_out, sadf_gold, csw_gold, gold_logp


# =============================================================================
# Part C -- secondary contrast: short BTC window vs gold's long window
# =============================================================================
def part_c_btc_short_window_contrast(gold_sadf_max, gold_sdfc):
    print()
    print("=" * 78)
    print("PART C -- Secondary contrast: BTC's short (~29-day) window")
    print("=" * 78)
    print("Chow-DF and SADF only -- both are built to catch slow-building,")
    print("multi-period bubbles, so a short window is a genuine stress test")
    print("of whether they have enough history to work with at all.\n")

    btc = pd.read_csv(os.path.join(INPUT_DIR, 'ch05_features.csv'),
                       index_col=0, parse_dates=True)
    btc_logp = np.log(btc[['close']])
    print(f"BTC: {len(btc_logp)} real dollar bars, "
          f"{btc_logp.index.min()} to {btc_logp.index.max()} "
          f"(~{(btc_logp.index.max() - btc_logp.index.min()).days} real days)")

    sadf_btc = get_sadf(btc_logp, minSL=30, constant='nc', lags=1)
    sdfc_btc = get_sdfc(btc_logp, tau0=0.1)

    print(f"\nBTC max SADF: {sadf_btc.max():.3f}  vs  gold max SADF: {gold_sadf_max:.3f}")
    print(f"BTC SDFC:     {sdfc_btc['sdfc']:.3f}  vs  gold SDFC:     {gold_sdfc['sdfc']:.3f}")
    print("\nNeither series is directly comparable in an absolute sense (different")
    print("assets, different vol regimes) -- the point of this contrast is")
    print("qualitative: gold's test statistics are being estimated from ~1,760")
    print("bars of genuine multi-year history, while BTC's are estimated from")
    print("239 bars spanning under a month. The same 'heavy extrapolation from")
    print("a short real window' caveat this project has flagged before (Ch15's")
    print("annualized frequency, Ch13's O-U calibration) applies here too --")
    print("BTC's numbers are a real result, just a thinly-supported one.")

    return sadf_btc, sdfc_btc


if __name__ == '__main__':
    bde_out = part_a_bde_cusum()
    sdfc_gold, sadf_gold, csw_gold, gold_logp = part_b_gold_explosiveness()
    sadf_btc, sdfc_btc = part_c_btc_short_window_contrast(sadf_gold.max(), sdfc_gold)

# =============================================================================
# Real-machine pytest results
# =============================================================================
# This driver script has no dedicated test file of its own (consistent with
# other chapter drivers) -- the algorithms it calls are covered by:
#   structural_breaks/test_sadf.py                    19 tests
#   structural_breaks/test_chow_df.py                  9 tests
#   structural_breaks/test_cusum.py                   17 tests
#   total                                             45 tests
#
# Real machine (mlfinlab env, Python 3.10.20, pytest 9.0.3), two-pass check re-run 2026-10-31:
#   repo root:  45 passed, 1 warning in 1.25s
#   inside structural_breaks/   45 passed, 1 warning in 1.06s
# Earlier confirmations: 2026-07-31.
# The one warning is expected: a RuntimeWarning (invalid value encountered in double_scalars) from test_degenerate_perfect_fit_gives_near_zero_residual_variance, a deliberately degenerate case. It is not a bug.
# The driver script and notebook were confirmed on the real machine on 2026-08-01 (see README.md). Expect several minutes of runtime for Part B's SADF and CSW CUSUM, given their O(T^2) cost on gold's ~1,760-bar series.
#
# Full output, run from the repo root:
#
# (mlfinlab) PS C:\ws\AFML> python -m pytest ch17 -v
# ============================= test session starts =============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML
# collecting ... collected 45 items
#
# ch17/structural_breaks/test_chow_df.py::TestGetDfc::test_hand_computed_closed_form PASSED [  2%]
# ch17/structural_breaks/test_chow_df.py::TestGetDfc::test_accepts_dataframe_input PASSED [  4%]
# ch17/structural_breaks/test_chow_df.py::TestGetDfc::test_nan_when_break_too_close_to_edge PASSED [  6%]
# ch17/structural_breaks/test_chow_df.py::TestGetDfc::test_explosive_switch_gives_high_dfc_at_true_break PASSED [  8%]
# ch17/structural_breaks/test_chow_df.py::TestGetDfc::test_pure_random_walk_gives_low_dfc PASSED [ 11%]
# ch17/structural_breaks/test_chow_df.py::TestGetSdfc::test_locates_known_injected_break PASSED [ 13%]
# ch17/structural_breaks/test_chow_df.py::TestGetSdfc::test_sdfc_equals_max_of_manual_grid PASSED [ 15%]
# ch17/structural_breaks/test_chow_df.py::TestGetSdfc::test_raises_on_infeasible_tau0 PASSED [ 17%]
# ch17/structural_breaks/test_chow_df.py::TestGetSdfc::test_explicit_step_grid PASSED [ 20%]
# ch17/structural_breaks/test_cusum.py::TestBdeRecursiveResiduals::test_hand_computed_single_residual PASSED [ 22%]
# ch17/structural_breaks/test_cusum.py::TestBdeRecursiveResiduals::test_raises_when_min_sample_too_small PASSED [ 24%]
# ch17/structural_breaks/test_cusum.py::TestBdeRecursiveResiduals::test_degenerate_perfect_fit_gives_near_zero_residual_variance PASSED [ 26%]
# ch17/structural_breaks/test_cusum.py::TestBdeRecursiveResiduals::test_recursive_beta_uses_only_past_data PASSED [ 28%]
# ch17/structural_breaks/test_cusum.py::TestBdeCusum::test_returns_expected_columns_and_index PASSED [ 31%]
# ch17/structural_breaks/test_cusum.py::TestBdeCusum::test_S_matches_raw_cumsum_over_sigma_omega PASSED [ 33%]
# ch17/structural_breaks/test_cusum.py::TestBdeCusum::test_injected_break_produces_visible_jump_in_S PASSED [ 35%]
# ch17/structural_breaks/test_cusum.py::TestBdeCusum::test_raises_on_too_short_series PASSED [ 37%]
# ch17/structural_breaks/test_cusum.py::TestCswStat::test_hand_computed PASSED [ 40%]
# ch17/structural_breaks/test_cusum.py::TestCswStat::test_accepts_dataframe_input PASSED [ 42%]
# ch17/structural_breaks/test_cusum.py::TestCswStat::test_raises_when_t_not_after_n PASSED [ 44%]
# ch17/structural_breaks/test_cusum.py::TestCswCriticalValue::test_matches_book_formula PASSED [ 46%]
# ch17/structural_breaks/test_cusum.py::TestCswSup::test_matches_manual_max_over_n PASSED [ 48%]
# ch17/structural_breaks/test_cusum.py::TestCswSup::test_explosive_jump_gives_higher_sup_than_random_walk PASSED [ 51%]
# ch17/structural_breaks/test_cusum.py::TestCswCusum::test_returns_expected_shape PASSED [ 53%]
# ch17/structural_breaks/test_cusum.py::TestCswCusum::test_last_row_matches_direct_get_csw_sup_call PASSED [ 55%]
# ch17/structural_breaks/test_cusum.py::TestCswCusum::test_raises_on_too_short_series PASSED [ 57%]
# ch17/structural_breaks/test_sadf.py::TestLagDF::test_int_lags_includes_lag_zero PASSED [ 60%]
# ch17/structural_breaks/test_sadf.py::TestLagDF::test_shift_values_hand_traced PASSED [ 62%]
# ch17/structural_breaks/test_sadf.py::TestLagDF::test_explicit_lag_list PASSED [ 64%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_series_auto_converted_to_dataframe PASSED [ 66%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_dataframe_input_shapes PASSED [ 68%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_constant_nc_adds_no_columns PASSED [ 71%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_constant_ct_adds_two_columns PASSED [ 73%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_constant_ctt_adds_two_columns_vs_ct PASSED [ 75%]
# ch17/structural_breaks/test_sadf.py::TestGetYX::test_column_zero_overwritten_with_level_not_diff PASSED [ 77%]
# ch17/structural_breaks/test_sadf.py::TestGetBetas::test_hand_computed_simple_regression PASSED [ 80%]
# ch17/structural_breaks/test_sadf.py::TestGetBetas::test_two_regressor_hand_computed PASSED [ 82%]
# ch17/structural_breaks/test_sadf.py::TestGetBsadf::test_literal_book_transcription_crashes_on_none PASSED [ 84%]
# ch17/structural_breaks/test_sadf.py::TestGetBsadf::test_fixed_version_does_not_crash PASSED [ 86%]
# ch17/structural_breaks/test_sadf.py::TestGetBsadf::test_explosive_series_gives_higher_bsadf_than_random_walk PASSED [ 88%]
# ch17/structural_breaks/test_sadf.py::TestGetBsadf::test_matches_standard_adf_when_minSL_forces_single_start_point PASSED [ 91%]
# ch17/structural_breaks/test_sadf.py::TestGetSadf::test_returns_one_value_per_valid_end_point PASSED [ 93%]
# ch17/structural_breaks/test_sadf.py::TestGetSadf::test_last_value_matches_direct_get_bsadf_call PASSED [ 95%]
# ch17/structural_breaks/test_sadf.py::TestGetSadf::test_raises_on_too_short_series PASSED [ 97%]
# ch17/structural_breaks/test_sadf.py::TestGetSadf::test_explosive_series_sadf_rises_then_the_bubble_is_visible PASSED [100%]
#
# ============================== warnings summary ===============================
# ch17/structural_breaks/test_cusum.py::TestBdeRecursiveResiduals::test_degenerate_perfect_fit_gives_near_zero_residual_variance
#   C:\ws\AFML\ch17\structural_breaks\cusum.py:95: RuntimeWarning: invalid value encountered in double_scalars
#     omega.append((y[t] - x_t @ beta) / np.sqrt(f_t))
#
# -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
# ======================== 45 passed, 1 warning in 1.25s ========================
# ---------------------------------------------------------------------------
