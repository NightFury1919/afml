"""End-to-end positive control for the ETF pipeline (pre-registered design).

QUESTION
    If a real edge of known size exists in this universe, how often does our whole
    pipeline (features -> labels -> purged panel CV -> model -> detection statistic)
    find it?  And how often does it "find" one when there is none?

DESIGN (see preregistration_etf_positive_control.md; frozen before the full run)
    World      Moving-block bootstrap of the real 48-ETF daily log returns, whole
               cross-sections at a time, block length 63. Keeps fat tails, volatility
               clustering and the real cross-ETF correlation. Destroys any real
               long-horizon momentum, so IC = 0 is a true no-edge world.
    Plant      z = centered unit-variance rank of mom_12_1 on the UNPLANTED prices.
               Next-day return gets  + a * sigma_i * z,  a = IC / sqrt(h)  (h = 21).
               The tilt is relative (z sums to zero), so market returns are unchanged.
               Features and labels are then recomputed from the PLANTED prices, as they
               would be live. Nominal IC is the planted skill at the 21-day horizon;
               the realized IC of the planted score is measured and reported.
    Pipeline   build_dataset (5 ranked features, relative 21-day labels) ->
               PanelPurgedKFold(5 folds, 252-date embargo) -> logistic regression (C=1)
               -> out-of-fold probabilities for every row.
    Statistic  Cross-sectional Spearman IC between the out-of-fold probability and the
               realized 21-day excess return, on dates 21 apart (non-overlapping), then
               t = mean / (sd / sqrt(n)).
    Detection  t above the 95th percentile of the t-stats from the IC = 0 worlds
               (same replicates). Power = share of planted worlds above it.
    Also       Long-only top-quintile active IR vs the equal-weight universe (model and
               oracle), the capture ratio, and the false-positive rate of t >= 1.96.

Variant v1.1 (--demean): the same design, with each ETF's average return removed before
the bootstrap, so the IC = 0 world is a true no-edge world. See
preregistration_etf_positive_control_v1_1_demeaned.md.

Not in v1 (on purpose): sample weights / time decay, costs, DSR and PBO (one declared
model = one trial), shorting. The lever what-ifs come after the baseline.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from etf_features import (DEFAULT_MIN_ASSETS, DEFAULT_MIN_GROUP_ASSETS, DEFAULT_WINDOWS,
                          FEATURE_NAMES, compute_features)
from etf_labels import DEFAULT_HORIZON, build_dataset
from panel_data import drop_hedge, load_prices, wide_to_long
from panel_purged_cv import PanelPurgedKFold

DEFAULT_IC_LEVELS = [0.0, 0.02, 0.03, 0.05, 0.075, 0.10]
BLOCK_LENGTH = 63
BASE_SEED = 20261004
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- the world
def block_bootstrap_returns(returns, block_length, rng):
    """Moving-block bootstrap of whole rows; keeps the original date index and length."""
    n = len(returns)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    rows = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]
    out = returns.iloc[rows].copy()
    out.index = returns.index
    return out


def demean_returns(returns):
    """Subtract each ETF's own average return, so no ETF has a higher expected return than another.

    v1 kept the real ETF averages (they differ a lot) and the "no-edge" world still held static
    differences that a characteristic-based sort could exploit. Volatility is unchanged.
    """
    return returns - returns.mean()


def returns_to_prices(returns, start=100.0):
    return start * np.exp(returns.cumsum())


# ------------------------------------------------------------------ planting
def planted_score(prices, windows=DEFAULT_WINDOWS, min_assets=DEFAULT_MIN_ASSETS,
                  groups=None, min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """Centered, unit-variance score from the mom_12_1 rank; NaN where an ETF is not eligible.

    With groups ({ticker: class}) the rank is taken inside each ETF's own class and the score is
    centered and scaled to unit variance inside each class on each date, so the planted skill is
    a within-class skill (the same kind of skill the within-class labels reward). A class with
    fewer than min_group_assets eligible ETFs that day gets no score, hence no tilt.
    """
    feats = compute_features(prices, windows, min_assets, groups, min_group_assets)
    rank = feats["mom_12_1"].unstack("asset").reindex(index=prices.index, columns=prices.columns)
    if groups is None:
        return (rank - 0.5) * np.sqrt(12.0)
    z = pd.DataFrame(np.nan, index=rank.index, columns=rank.columns)
    for g in sorted({groups[c] for c in rank.columns}):
        cols = [c for c in rank.columns if groups[c] == g]
        block = rank[cols]
        sd = block.std(axis=1, ddof=0).replace(0.0, np.nan)
        z[cols] = block.sub(block.mean(axis=1), axis=0).div(sd, axis=0)
    return z


def plant_signal(returns, score, ic_horizon, horizon=DEFAULT_HORIZON):
    """Add a * sigma_i * z (known at the previous close) to each day's return. a = IC / sqrt(h)."""
    a = ic_horizon / np.sqrt(horizon)
    sigma = returns.std()
    tilt = score.reindex_like(returns).shift(1).fillna(0.0).mul(sigma, axis=1) * a
    return returns + tilt


# ---------------------------------------------------------------- statistics
def date_ics(pred, target):
    """Cross-sectional Spearman IC on each date. pred, target: Series indexed by (date, asset)."""
    rp = pred.groupby(level="date").rank()
    rt = target.groupby(level="date").rank()
    rp = rp - rp.groupby(level="date").transform("mean")
    rt = rt - rt.groupby(level="date").transform("mean")
    num = (rp * rt).groupby(level="date").sum()
    den = np.sqrt((rp ** 2).groupby(level="date").sum() * (rt ** 2).groupby(level="date").sum())
    return num / den


def ic_tstat(ics, spacing):
    """t-statistic of the mean IC using every `spacing`-th date (non-overlapping labels)."""
    s = ics.iloc[::spacing].dropna()
    n = len(s)
    mean = float(s.mean())
    t = mean / (float(s.std(ddof=1)) / np.sqrt(n))
    return t, mean, n


def top_quintile_active_ir(pred, fwd_ret, horizon, top_share=0.2):
    """Annualized IR of 'hold the top fifth by pred' minus the equal-weight universe, per period."""
    pct = pred.groupby(level="date").rank(pct=True)
    top = pct > (1.0 - top_share)
    top_mean = fwd_ret.where(top).groupby(level="date").mean()
    all_mean = fwd_ret.groupby(level="date").mean()
    active = (top_mean - all_mean).dropna().iloc[::horizon]
    mean_active = float(active.mean())
    ir = mean_active / float(active.std(ddof=1)) * np.sqrt(252.0 / horizon)
    return ir, mean_active, len(active)


# ------------------------------------------------------------------ pipeline
def out_of_fold_predictions(dataset, n_splits=5, embargo_dates=252, C=1.0):
    """Probability of label 1 for every row, each from a model that never saw that row's fold."""
    X, y = dataset[FEATURE_NAMES], dataset["label"]
    cv = PanelPurgedKFold(n_splits=n_splits, t1=dataset["t1"], embargo_dates=embargo_dates)
    pred = pd.Series(np.nan, index=dataset.index)
    for train, test in cv.split(X):
        model = LogisticRegression(C=C, max_iter=200)
        model.fit(X.iloc[train].to_numpy(), y.iloc[train].to_numpy())
        pred.iloc[test] = model.predict_proba(X.iloc[test].to_numpy())[:, 1]
    return pred


def run_pipeline(prices, horizon=DEFAULT_HORIZON, n_splits=5, embargo_dates=252,
                 min_assets=DEFAULT_MIN_ASSETS, windows=DEFAULT_WINDOWS, C=1.0, oracle_score=None,
                 groups=None, min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    ds = build_dataset(prices, horizon, windows, min_assets, groups, min_group_assets)
    pred = out_of_fold_predictions(ds, n_splits, embargo_dates, C)
    t, mean_ic, n_ic = ic_tstat(date_ics(pred, ds["excess_ret"]), horizon)
    ir, _, _ = top_quintile_active_ir(pred, ds["fwd_ret"], horizon)
    out = dict(t=float(t), mean_ic=float(mean_ic), n_ic=int(n_ic), active_ir=float(ir),
               oracle_ic=float("nan"), oracle_active_ir=float("nan"))
    if oracle_score is not None:
        z = wide_to_long(oracle_score, "z").reindex(ds.index)
        _, out["oracle_ic"], _ = ic_tstat(date_ics(z, ds["excess_ret"]), horizon)
        out["oracle_active_ir"], _, _ = top_quintile_active_ir(z, ds["fwd_ret"], horizon)
    return out


def one_replicate(returns, seed, ic_levels, block_length=BLOCK_LENGTH, horizon=DEFAULT_HORIZON,
                  **pipeline_kwargs):
    """One bootstrapped world, every IC level planted into it (common random numbers)."""
    rng = np.random.default_rng(seed)
    boot = block_bootstrap_returns(returns, block_length, rng)
    score = planted_score(returns_to_prices(boot),
                          min_assets=pipeline_kwargs.get("min_assets", DEFAULT_MIN_ASSETS),
                          groups=pipeline_kwargs.get("groups"),
                          min_group_assets=pipeline_kwargs.get("min_group_assets", DEFAULT_MIN_GROUP_ASSETS))
    rows = []
    for ic in ic_levels:
        prices = returns_to_prices(plant_signal(boot, score, ic, horizon))
        res = run_pipeline(prices, horizon=horizon, oracle_score=score, **pipeline_kwargs)
        rows.append({"seed": seed, "ic_nominal": ic, **res})
    return rows


# ------------------------------------------------------------------ analysis
def analyze(results):
    """Power curve. Threshold = 95th percentile of the IC = 0 t-stats in the same results."""
    null95 = float(np.percentile(results.loc[results["ic_nominal"] == 0.0, "t"], 95))
    rows = []
    for ic, g in results.groupby("ic_nominal"):
        oracle_ir = g["oracle_active_ir"].mean()
        rows.append(dict(
            ic_nominal=ic, n_reps=len(g), null95=null95,
            power=float((g["t"] > null95).mean()), fpr_t196=float((g["t"] >= 1.96).mean()),
            mean_t=g["t"].mean(), mean_model_ic=g["mean_ic"].mean(), mean_oracle_ic=g["oracle_ic"].mean(),
            mean_active_ir=g["active_ir"].mean(), mean_oracle_active_ir=oracle_ir,
            capture_ir=g["active_ir"].mean() / oracle_ir if oracle_ir else float("nan")))
    return pd.DataFrame(rows)


def _worker(args):
    from threadpoolctl import threadpool_limits
    returns, seed, ic_levels, block_length, horizon, groups = args
    with threadpool_limits(limits=1):
        return one_replicate(returns, seed, ic_levels, block_length, horizon, groups=groups)


def results_filename(demean=False, within_class=False):
    """Results file per variant. v1 and v1.1 names are kept so old results are never overwritten."""
    name = "positive_control_etf"
    if demean:
        name += "_demeaned"
    if within_class:
        name += "_within_class"
    return name + "_results.csv"


def full_universe_returns(prices_path):
    px = drop_hedge(load_prices(prices_path))
    return np.log(px).diff().dropna(how="any")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", default=os.path.join(HERE, "prices_daily_asof_2026-10-02.csv"))
    ap.add_argument("--out", default=None,
                    help="results CSV (default depends on the flags, see results_filename())")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ic-levels", type=float, nargs="+", default=DEFAULT_IC_LEVELS)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--demean", action="store_true",
                    help="v1.1: remove each ETF's average return before the bootstrap")
    ap.add_argument("--within-class", action="store_true",
                    help="rank, label and plant within each ETF's asset class (etf_classes.CLASS_OF)")
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(HERE, results_filename(args.demean, args.within_class))
    groups = None
    if args.within_class:
        from etf_classes import CLASS_OF
        groups = CLASS_OF

    if not args.analyze_only:
        returns = full_universe_returns(args.prices)
        if groups is not None:
            from etf_classes import validate_classes
            validate_classes(list(returns.columns))
        if args.demean:
            returns = demean_returns(returns)
        done = set(pd.read_csv(args.out)["seed"]) if os.path.exists(args.out) else set()
        seeds = [BASE_SEED + k for k in range(args.reps) if BASE_SEED + k not in done]
        print(f"{len(done)} replicates already done, {len(seeds)} to run, "
              f"{len(args.ic_levels)} IC levels each, {args.workers} workers")
        jobs = [(returns, s, args.ic_levels, BLOCK_LENGTH, DEFAULT_HORIZON, groups) for s in seeds]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for k, rows in enumerate(pool.map(_worker, jobs), 1):
                pd.DataFrame(rows).to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
                print(f"  replicate {k}/{len(seeds)} done (seed {rows[0]['seed']})", flush=True)

    res = pd.read_csv(args.out)
    print(f"\n{res['seed'].nunique()} replicates. Threshold = 95th percentile of the IC=0 t-stats.\n")
    print(analyze(res).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# REAL-DATA RUN v1 (mlfinlab env, 2026-10-04; prices_daily_asof_2026-10-02.csv)
# Pre-registered in preregistration_etf_positive_control.md. 500 bootstrapped
# worlds x 6 IC levels, seeds 20261004-20261503, 4 workers. Run before the
# --demean option was added; the default behaviour is unchanged.
# $ python positive_control_etf.py --workers 4
#
#  ic_nominal  n_reps  null95  power  fpr_t196  mean_t  mean_model_ic  mean_oracle_ic  mean_active_ir  mean_oracle_active_ir  capture_ir
#       0.000     500   4.307  0.050     0.694   2.453          0.057           0.024           0.122                  0.025       4.842
#       0.020     500   4.307  0.212     0.872   3.344          0.073           0.053           0.327                  0.264       1.240
#       0.030     500   4.307  0.402     0.958   3.975          0.084           0.067           0.464                  0.383       1.212
#       0.050     500   4.307  0.814     0.996   5.398          0.108           0.095           0.762                  0.621       1.227
#       0.075     500   4.307  0.996     1.000   7.259          0.140           0.130           1.120                  0.918       1.220
#       0.100     500   4.307  1.000     1.000   9.078          0.171           0.164           1.454                  1.214       1.197
#
# OUTCOME against the pre-registered primary bar: NOT MET.
#   - Power at nominal IC 0.05 = 81.4% (needed >= 50%): met.
#   - False-positive rate of t >= 1.96 at IC 0 = 69.4% (needed 2% to 8%): NOT met.
#
# WHY (diagnosed after the run, development diagnostics only, seeds 900000-900039):
#   The block bootstrap keeps each ETF's own average return, and those differ widely
#   (annualized mean log return from -2.2% to +15.8%, std 4.1% across ETFs). In 40
#   fresh worlds the mom_12_1 rank had mean IC +0.020 against 21-day excess returns
#   with ETF means kept and -0.003 with means removed. So the "no-edge" world still
#   holds static cross-sectional differences that k-fold CV (which trains on later
#   periods) can exploit. Not yet explained: why the model's null IC (0.057) exceeds
#   the oracle's (0.024). Next: v1.1, demeaned worlds (--demean).
#
# READ THE CAPTURE RATIO WITH CARE: capture_ir above 1 includes the model exploiting
# those static differences, so it is not extra skill. The realized planted increment
# (oracle IC minus 0.024) runs about 1.4x nominal in this world; cause not verified
# (fat tails inflating the standard deviation is a guess; relative-return dispersion
# was checked and does not explain it).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REAL-DATA RUN v1.1, demeaned worlds (mlfinlab env, 2026-10-04/05;
# prices_daily_asof_2026-10-02.csv). Pre-registered in
# preregistration_etf_positive_control_v1_1_demeaned.md. Same 500 seeds and IC levels as v1.
# $ python positive_control_etf.py --workers 4 --demean
#
#  ic_nominal  n_reps  null95  power  fpr_t196  mean_t  mean_model_ic  mean_oracle_ic  mean_active_ir  mean_oracle_active_ir  capture_ir
#       0.000     500   2.972  0.050     0.260   1.199          0.028           0.002           0.061                 -0.054      -1.128
#       0.020     500   2.972  0.140     0.430   1.679          0.037           0.031           0.146                  0.190       0.767
#       0.030     500   2.972  0.290     0.614   2.233          0.048           0.045           0.249                  0.311       0.800
#       0.050     500   2.972  0.722     0.934   3.635          0.073           0.074           0.521                  0.555       0.939
#       0.075     500   2.972  0.994     0.996   5.499          0.107           0.110           0.875                  0.858       1.019
#       0.100     500   2.972  1.000     1.000   7.323          0.139           0.145           1.200                  1.161       1.033
#
# OUTCOME against the pre-registered clean-null criteria at IC 0: NOT CLEAN (2 of 3 fail).
#   1. False-positive rate of t >= 1.96 in 2% to 8%:      26.0%   FAIL (v1: 69.4%)
#   2. |mean model IC| <= 0.01:                           0.028   FAIL (v1: 0.057)
#   3. |mean oracle IC (mom_12_1 rank)| <= 0.01:          0.002   PASS (v1: 0.024)
# Because the null is not clean, the primary detection bar is not evaluated. For the
# record: power at nominal IC 0.05 was 72.2%, but the false-positive condition fails.
#
# WHAT CHANGED vs v1: removing ETF means fixed the momentum part (oracle IC 0.024 -> 0.002)
# and cut the inflation roughly in half (null95 4.307 -> 2.972, mean null t 2.45 -> 1.20).
#
# DIAGNOSTICS AFTER THE RUN (development only, 40 demeaned worlds, seeds 900000-900039):
#   Mean IC of each ranked feature vs 21-day excess return: mom_12_1 -0.003, mom_3m -0.009,
#   ret_5d -0.018, vol_60 +0.034, trend_200 -0.012.
#   Mean out-of-fold model IC: 0.0253 (SE 0.0046) with all five features; 0.0102 (SE 0.0039)
#   without vol_60.
#   Static difference that survives mean removal: the per-ETF MEDIAN 21-day return relative to
#   the cross-sectional median (std 0.27%) correlates +0.51 with volatility and -0.58 with
#   skew across the 48 ETFs (225 blocks per ETF, so noisy).
# READING: rank IC credits differences in typical (median) outcomes, which track volatility
# and skew even when means are equal, and the model learns that through vol_60. A rank
# statistic needs a null that equalizes ranks, not means. The block bootstrap also keeps real
# short-horizon structure (ret_5d IC -0.018), so the no-edge world is not free of real
# dynamics. The skew explanation is supported by the diagnostics above but not proven.
#
# Planted IC: realized oracle IC runs about 1.45x to 1.55x nominal in this clean-oracle
# world. The model recovers about 60% to 77% of the planted IC (model IC minus null model IC,
# divided by oracle IC).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_positive_control_etf.py::test_block_bootstrap_keeps_shape_index_and_uses_only_original_rows PASSED [ 28%]
# test_positive_control_etf.py::test_block_bootstrap_blocks_are_consecutive_runs_of_the_original PASSED [ 29%]
# test_positive_control_etf.py::test_block_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds PASSED [ 30%]
# test_positive_control_etf.py::test_returns_to_prices_known_values PASSED [ 31%]
# test_positive_control_etf.py::test_planted_score_is_centered_and_ordered_by_momentum PASSED [ 32%]
# test_positive_control_etf.py::test_plant_signal_known_values PASSED      [ 34%]
# test_positive_control_etf.py::test_zero_ic_changes_nothing_and_missing_scores_mean_no_tilt PASSED [ 35%]
# test_positive_control_etf.py::test_date_ics_known_values PASSED          [ 36%]
# test_positive_control_etf.py::test_ic_tstat_takes_every_nth_date_and_matches_the_formula PASSED [ 37%]
# test_positive_control_etf.py::test_top_quintile_active_ir_known_values PASSED [ 38%]
# test_positive_control_etf.py::test_out_of_fold_predictions_cover_every_row_once_and_are_probabilities PASSED [ 39%]
# test_positive_control_etf.py::test_pipeline_finds_a_strong_planted_signal PASSED [ 40%]
# test_positive_control_etf.py::test_pipeline_finds_nothing_when_nothing_is_planted PASSED [ 42%]
# test_positive_control_etf.py::test_realized_oracle_ic_is_close_to_the_nominal_planted_ic PASSED [ 43%]
# test_positive_control_etf.py::test_one_replicate_returns_one_row_per_ic_level_and_is_reproducible PASSED [ 44%]
# test_positive_control_etf.py::test_analyze_power_uses_the_null_95th_percentile_threshold PASSED [ 45%]
# test_positive_control_etf.py::test_demean_returns_known_values_and_shape PASSED [ 46%]
# test_positive_control_etf.py::test_demeaned_world_has_no_static_edge_but_the_raw_world_does PASSED [ 47%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): same-day planting score, independent per-ETF bootstrap, a 50th-percentile threshold, and a no-op demean each made tests fail; original restored.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TDD RESULTS, within-class addition (pytest, 2026-10-06; sandbox run: Linux, Python 3.10,
# pandas 1.5.3, numpy 1.23.5, scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_positive_control_etf.py -v
#
# collected 26 items
#
# test_positive_control_etf.py::test_block_bootstrap_keeps_shape_index_and_uses_only_original_rows PASSED [  3%]
# test_positive_control_etf.py::test_block_bootstrap_blocks_are_consecutive_runs_of_the_original PASSED [  7%]
# test_positive_control_etf.py::test_block_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds PASSED [ 11%]
# test_positive_control_etf.py::test_returns_to_prices_known_values PASSED [ 15%]
# test_positive_control_etf.py::test_planted_score_is_centered_and_ordered_by_momentum PASSED [ 19%]
# test_positive_control_etf.py::test_plant_signal_known_values PASSED      [ 23%]
# test_positive_control_etf.py::test_zero_ic_changes_nothing_and_missing_scores_mean_no_tilt PASSED [ 26%]
# test_positive_control_etf.py::test_date_ics_known_values PASSED          [ 30%]
# test_positive_control_etf.py::test_ic_tstat_takes_every_nth_date_and_matches_the_formula PASSED [ 34%]
# test_positive_control_etf.py::test_top_quintile_active_ir_known_values PASSED [ 38%]
# test_positive_control_etf.py::test_out_of_fold_predictions_cover_every_row_once_and_are_probabilities PASSED [ 42%]
# test_positive_control_etf.py::test_pipeline_finds_a_strong_planted_signal PASSED [ 46%]
# test_positive_control_etf.py::test_pipeline_finds_nothing_when_nothing_is_planted PASSED [ 50%]
# test_positive_control_etf.py::test_realized_oracle_ic_is_close_to_the_nominal_planted_ic PASSED [ 53%]
# test_positive_control_etf.py::test_one_replicate_returns_one_row_per_ic_level_and_is_reproducible PASSED [ 57%]
# test_positive_control_etf.py::test_analyze_power_uses_the_null_95th_percentile_threshold PASSED [ 61%]
# test_positive_control_etf.py::test_demean_returns_known_values_and_shape PASSED [ 65%]
# test_positive_control_etf.py::test_demeaned_world_has_no_static_edge_but_the_raw_world_does PASSED [ 69%]
# test_positive_control_etf.py::test_planted_score_with_groups_is_standardized_inside_each_group PASSED [ 73%]
# test_positive_control_etf.py::test_planted_score_without_groups_is_unchanged_by_the_new_parameters PASSED [ 76%]
# test_positive_control_etf.py::test_groups_with_too_few_eligible_etfs_get_no_planted_score PASSED [ 80%]
# test_positive_control_etf.py::test_run_pipeline_hands_groups_to_build_dataset PASSED [ 84%]
# test_positive_control_etf.py::test_grouped_pipeline_finds_a_strong_within_class_planted_signal PASSED [ 88%]
# test_positive_control_etf.py::test_grouped_pipeline_finds_nothing_when_nothing_is_planted PASSED [ 92%]
# test_positive_control_etf.py::test_one_replicate_with_groups_is_reproducible_and_differs_from_pooled PASSED [ 96%]
# test_positive_control_etf.py::test_results_filename_is_distinct_for_every_variant PASSED [100%]
#
# 26 passed in 10.83s (the 18 earlier tests unchanged, 8 new)
# Wider regression: test_positive_control_etf, test_etf_labels, test_etf_features, test_etf_classes,
#   test_panel_data together: 72 passed.
# Mutation checks (sandbox): run_pipeline ignoring groups, and planted_score ignoring groups, each made a
#   test fail; original restored. CLI smoke test (--within-class, synthetic prices, 1 world, 2 IC levels) ran.
# Re-run on your Windows mlfinlab env and replace this block with that output if you prefer.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REAL-DATA RUN v1.2, within-class ranking, demeaned worlds (mlfinlab env, 2026-10-09;
# prices_daily_asof_2026-10-02.csv). Pre-registered in
# preregistration_etf_positive_control_v1_2_within_class.md (committed before the run). Same 500 seeds and
# IC levels as v1 and v1.1.
# $ python positive_control_etf.py --workers 4 --demean --within-class
#
#  ic_nominal  n_reps  null95  power  fpr_t196  mean_t  mean_model_ic  mean_oracle_ic  mean_active_ir  mean_oracle_active_ir  capture_ir
#       0.000     500   2.341  0.050     0.100   0.272          0.004          -0.000           0.030                 -0.038      -0.799
#       0.020     500   2.341  0.472     0.606   2.168          0.032           0.039           0.277                  0.316       0.876
#       0.030     500   2.341  0.870     0.926   3.582          0.053           0.058           0.478                  0.493       0.970
#       0.050     500   2.341  1.000     1.000   6.164          0.092           0.097           0.859                  0.847       1.015
#       0.075     500   2.341  1.000     1.000   9.157          0.136           0.144           1.306                  1.289       1.013
#       0.100     500   2.341  1.000     1.000  11.991          0.176           0.190           1.730                  1.730       1.000
#
# OUTCOME against the frozen pass rule: MET. Power at nominal IC 0.03 is 0.870 (needed at least 0.39) and power
# at nominal IC 0.05 is 1.000 (needed at least 0.67). Within-class ranking detects a planted edge far more often
# than the pooled design (v1.1: 0.140 / 0.290 / 0.722 at nominal 0.02 / 0.03 / 0.05).
# FAIRER COMPARISON by realized oracle IC (the planted skill actually delivered): pooled v1.1 had oracle IC
# 0.031 / 0.045 / 0.074 and power 0.140 / 0.290 / 0.722; within-class had oracle IC 0.039 / 0.058 / 0.097 and
# power 0.472 / 0.870 / 1.000. Interpolating, within-class power at oracle IC 0.045 is about 0.6 against 0.29
# pooled, so the gain is real after allowing for the within-class world delivering about 1.9x its nominal IC
# (pooled delivered about 1.5x; cause of the difference not verified).
# NULL QUALITY: much cleaner than v1.1. Mean model IC at IC 0 is 0.004 (pooled 0.028), mean oracle IC is
# -0.000, null95 is 2.341 (pooled 2.972), and the false-positive rate of t >= 1.96 is 10.0% (pooled 26.0%).
# That 10.0% is still above the 2% to 8% band set for v1.1; that band was not part of the v1.2 pass rule.
# Rough reading: about 50% power at a realized rank IC near 0.04 and about 85% near 0.06.
# CAVEAT: nominal IC means a within-class skill here and a pooled skill there, and the statistic pools
# within-class excess returns across classes. Not a test of live profitability or of class budgets.
# ---------------------------------------------------------------------------
