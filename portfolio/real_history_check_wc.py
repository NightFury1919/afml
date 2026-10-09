"""Within-class real-history check: does a within-class model add anything beyond a within-class volatility sort?

Pre-registered in preregistration_real_history_check_wc.md. Run ONCE:
    python real_history_check_wc.py                 # uses prices_daily_asof_2026-10-08.csv

WHY
    real_history_check.py found the pooled baseline indistinguishable from a volatility sort (its tilt
    was 0.885). Pooled ranking mostly sorts classes (bonds vs. emerging-market stocks); the positive
    control showed within-class ranking is the better-powered design (v1.2). This repeats the check
    with every feature, label and portfolio built INSIDE each of the 6 asset classes.

THREE ARMS (all scored out-of-fold with PanelPurgedKFold, 5 folds, 252-date embargo)
    wc_current   logistic regression, all five within-class-ranked features, label = forward 21-day
                 return above the CLASS median.
    wc_no_vol    the same without the vol_60 feature, so volatility cannot be used directly (the
                 remaining features can still carry some of it).
    wc_vol_rule  no model: score = the within-class vol_60 rank.

HOW THEY ARE JUDGED (raw forward returns, dates 21 apart)
    ir          annualized IR of the class-neutral active return: in each class hold the top fifth
                (ceil, at least 1) by score minus that class's equal-weight mean; classes weighted by
                their size. Class-wide moves cancel, so only within-class selection counts.
    mean_ic     mean over (date, class) of the out-of-fold Spearman IC against excess return vs. the class median.
    vol_tilt    mean over (date, class) of the Spearman correlation between score and the vol_60 rank.
    CIs         moving-block bootstrap (block 4 periods): paired differences between arms and a
                standalone interval for each arm.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from etf_classes import CLASS_OF
from etf_features import DEFAULT_MIN_ASSETS, DEFAULT_MIN_GROUP_ASSETS, DEFAULT_WINDOWS, FEATURE_NAMES, compute_features
from etf_labels import DEFAULT_HORIZON, make_labels
from panel_data import drop_hedge, load_prices
from panel_purged_cv import PanelPurgedKFold
from real_history_check import (BOOT_BLOCK, BOOT_SEED, CI_LEVEL, MAX_VOL_TILT, N_BOOT, annualized_ir,
                                paired_block_bootstrap_ci)

DEFAULT_PRICES = "prices_daily_asof_2026-10-08.csv"
RESULTS = "real_history_check_wc_results.csv"
ACTIVE = "real_history_check_wc_active_returns.csv"
ARMS = ["wc_current", "wc_no_vol", "wc_vol_rule"]
FITTED = ["wc_current", "wc_no_vol"]
PAIRS = [("wc_current", "wc_vol_rule"), ("wc_no_vol", "wc_vol_rule"), ("wc_no_vol", "wc_current")]
NO_VOL_FEATURES = [f for f in FEATURE_NAMES if f != "vol_60"]


def _classes(index, groups):
    cls = pd.Index(index.get_level_values("asset")).map(groups)
    if cls.isna().any():
        raise ValueError("an ETF has no class")
    return cls.to_numpy()


# --------------------------------------------------------------------- dataset
def build_wc_dataset(prices, groups, min_assets=DEFAULT_MIN_ASSETS, min_group_assets=DEFAULT_MIN_GROUP_ASSETS,
                     horizon=DEFAULT_HORIZON):
    """Within-class features and class-median labels on the same (date, asset) rows."""
    feats = compute_features(prices, DEFAULT_WINDOWS, min_assets, groups, min_group_assets)
    lab = make_labels(prices, feats.index, horizon, min_assets, groups, min_group_assets)
    return feats.reindex(lab.index).join(lab)


def oof_with_features(ds, features, n_splits=5, embargo_dates=252, C=1.0):
    """Out-of-fold probability of label 1 using only the listed features."""
    X, y = ds[list(features)], ds["label"]
    cv = PanelPurgedKFold(n_splits=n_splits, t1=ds["t1"], embargo_dates=embargo_dates)
    pred = pd.Series(np.nan, index=ds.index)
    for train, test in cv.split(X):
        model = LogisticRegression(C=C, max_iter=200)
        model.fit(X.iloc[train].to_numpy(), y.iloc[train].to_numpy())
        pred.iloc[test] = model.predict_proba(X.iloc[test].to_numpy())[:, 1]
    return pred


# ------------------------------------------------------------------ statistics
def _top_mask(pred, groups, top_share):
    """True for the top ceil(top_share * n) (at least 1) of each (date, class) by pred; ties by order."""
    date = pred.index.get_level_values("date")
    key = [date, _classes(pred.index, groups)]
    n = pred.groupby(key).transform("size")
    k = np.maximum(np.ceil(top_share * n.to_numpy() - 1e-9), 1)       # guard: 0.28 * 25 = 7.000000000000001
    rank = pred.groupby(key).rank(ascending=False, method="first")
    return pd.Series(rank.to_numpy() <= k, index=pred.index)


def active_series_within_class(pred, fwd_ret, groups, horizon=DEFAULT_HORIZON, top_share=0.2):
    """Per-period class-neutral active return, every `horizon`-th date.

    In each class: mean forward return of the top picks minus the class mean. Classes are weighted by
    their number of ETFs that date. Class-wide moves cancel.
    """
    date = pred.index.get_level_values("date")
    key = [date, _classes(pred.index, groups)]
    top = _top_mask(pred, groups, top_share)
    top_mean = fwd_ret.where(top).groupby(key).mean()
    cls_mean = fwd_ret.groupby(key).mean()
    size = fwd_ret.groupby(key).size()
    contrib = (top_mean - cls_mean) * size
    num = contrib.groupby(level=0).sum()
    den = size.groupby(level=0).sum()
    return (num / den).dropna().iloc[::horizon]


def _mean_class_spearman(a, b, groups):
    date = a.index.get_level_values("date")
    key = [date, _classes(a.index, groups)]
    ra, rb = a.groupby(key).rank(), b.groupby(key).rank()
    ra = ra - ra.groupby(key).transform("mean")
    rb = rb - rb.groupby(key).transform("mean")
    num = (ra * rb).groupby(key).sum()
    den = np.sqrt((ra ** 2).groupby(key).sum() * (rb ** 2).groupby(key).sum())
    return float((num / den).replace([np.inf, -np.inf], np.nan).mean())


def within_class_tilt(pred, vol_rank, groups):
    """Mean over (date, class) of the Spearman correlation between a score and the vol_60 rank."""
    return _mean_class_spearman(pred, vol_rank, groups)


def block_bootstrap_ir_ci(active, horizon=DEFAULT_HORIZON, block=BOOT_BLOCK, n_boot=N_BOOT,
                          seed=BOOT_SEED, level=CI_LEVEL):
    """(low, high) interval for the annualized IR of one active-return series."""
    x = np.asarray(active, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    irs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        rows = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        irs[k] = annualized_ir(x[rows], horizon)
    lo, hi = np.percentile(irs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# ----------------------------------------------------------------------- run
def evaluate_wc(prices, groups, n_splits=5, embargo_dates=252, min_assets=DEFAULT_MIN_ASSETS,
                min_group_assets=DEFAULT_MIN_GROUP_ASSETS, horizon=DEFAULT_HORIZON, C=1.0, n_boot=N_BOOT):
    px = drop_hedge(prices)
    ds = build_wc_dataset(px, groups, min_assets, min_group_assets, horizon)
    preds = {"wc_current": oof_with_features(ds, FEATURE_NAMES, n_splits, embargo_dates, C),
             "wc_no_vol": oof_with_features(ds, NO_VOL_FEATURES, n_splits, embargo_dates, C),
             "wc_vol_rule": ds["vol_60"]}
    fwd, excess, vrank = ds["fwd_ret"], ds["excess_ret"], ds["vol_60"]

    active, summary, standalone = {}, {}, {}
    for name in ARMS:
        pred = preds[name]
        act = active_series_within_class(pred, fwd, groups, horizon)
        active[name] = act
        picks = _top_mask(pred, groups, 0.2)
        half = len(act) // 2
        summary[name] = dict(
            ir=annualized_ir(act, horizon), mean_ic=_mean_class_spearman(pred, excess, groups),
            vol_tilt=within_class_tilt(pred, vrank, groups),
            mean_vol_rank_of_picks=float(vrank[picks].mean()),
            ir_first_half=annualized_ir(act.iloc[:half], horizon),
            ir_second_half=annualized_ir(act.iloc[half:], horizon), n_periods=int(len(act)))
        standalone[name] = block_bootstrap_ir_ci(act, horizon, n_boot=n_boot)
    paired = {f"{x}-{y}": paired_block_bootstrap_ci(active[x], active[y], horizon, n_boot=n_boot)
              for x, y in PAIRS}
    return dict(arms=summary, paired=paired, standalone=standalone, active=pd.DataFrame(active))


def decide_wc(res):
    """Apply the pre-registered rules to an evaluate_wc() result."""
    out = {}
    for arm in FITTED:
        adds = bool(res["paired"][f"{arm}-wc_vol_rule"][0] > 0 and res["arms"][arm]["vol_tilt"] <= MAX_VOL_TILT)
        alone = bool(res["standalone"][arm][0] > 0)
        out[f"{arm}_adds_beyond_volatility"] = adds
        out[f"{arm}_positive_alone"] = alone
        out[f"{arm}_edge_found"] = adds and alone
    return out


# ----------------------------------------------------------------------- files
def _paths(folder):
    folder = Path(folder)
    return folder / RESULTS, folder / ACTIVE


def check_not_run(folder):
    for p in _paths(folder):
        if p.exists():
            raise FileExistsError(f"{p.name} already exists. This check is run once; see the "
                                  "pre-registration. Not overwriting.")


def save_results_wc(res, folder):
    check_not_run(folder)
    rows = []
    for arm, m in res["arms"].items():
        rows += [dict(kind="arm", name=arm, metric=k, value=v) for k, v in m.items()]
    for arm, (lo, hi) in res["standalone"].items():
        rows += [dict(kind="standalone_ir", name=arm, metric="ci_low", value=lo),
                 dict(kind="standalone_ir", name=arm, metric="ci_high", value=hi)]
    for pair, (lo, hi) in res["paired"].items():
        rows += [dict(kind="paired_ir_diff", name=pair, metric="ci_low", value=lo),
                 dict(kind="paired_ir_diff", name=pair, metric="ci_high", value=hi)]
    rows += [dict(kind="decision", name=k, metric="rule", value=v) for k, v in decide_wc(res).items()]
    results_path, active_path = _paths(folder)
    pd.DataFrame(rows).to_csv(results_path, index=False)
    res["active"].to_csv(active_path, index_label="date")
    return results_path, active_path


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", default=str(here / DEFAULT_PRICES))
    ap.add_argument("--folder", default=str(here))
    args = ap.parse_args()
    check_not_run(args.folder)

    prices = load_prices(args.prices)
    print(f"Prices: {Path(args.prices).name}, {prices.shape[0]} dates, {prices.shape[1]} tickers")
    res = evaluate_wc(prices, CLASS_OF)

    print("\nARMS (class-neutral active return: top fifth of each class minus the class mean)")
    print(pd.DataFrame(res["arms"]).T.round(3).to_string())
    print(f"\nSTANDALONE IR, {CI_LEVEL:.0%} block-bootstrap interval")
    for arm, (lo, hi) in res["standalone"].items():
        print(f"  {arm:12s} [{lo:+.2f}, {hi:+.2f}]")
    print(f"\nPAIRED DIFFERENCE IN IR, {CI_LEVEL:.0%} interval")
    for pair, (lo, hi) in res["paired"].items():
        print(f"  {pair:24s} [{lo:+.2f}, {hi:+.2f}]")
    print("\nPRE-REGISTERED RULES")
    for k, v in decide_wc(res).items():
        print(f"  {k}: {v}")
    r, a = save_results_wc(res, args.folder)
    print(f"\nSaved {r}\n      {a}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_real_history_check_wc.py -v
#
# collected 17 items
#
# test_real_history_check_wc.py::test_active_return_known_value_equal_class_sizes PASSED [  5%]
# test_real_history_check_wc.py::test_active_return_weights_classes_by_their_size_and_picks_a_rounded_up_fifth PASSED [ 11%]
# test_real_history_check_wc.py::test_a_class_wide_return_difference_does_not_change_the_active_return PASSED [ 17%]
# test_real_history_check_wc.py::test_a_class_of_six_picks_two_not_one PASSED [ 23%]
# test_real_history_check_wc.py::test_float_noise_does_not_add_a_pick PASSED [ 29%]
# test_real_history_check_wc.py::test_only_every_horizon_th_date_is_used PASSED [ 35%]
# test_real_history_check_wc.py::test_within_class_tilt_is_plus_one_for_a_volatility_sort_and_minus_one_for_its_reverse PASSED [ 41%]
# test_real_history_check_wc.py::test_tilt_is_measured_inside_each_class_not_across_classes PASSED [ 47%]
# test_real_history_check_wc.py::test_dataset_uses_class_ranks_and_class_median_labels PASSED [ 52%]
# test_real_history_check_wc.py::test_the_pooled_dataset_would_differ_which_proves_groups_are_used PASSED [ 58%]
# test_real_history_check_wc.py::test_the_no_volatility_arm_ignores_the_volatility_feature PASSED [ 64%]
# test_real_history_check_wc.py::test_evaluate_gives_the_no_vol_arm_four_features_and_the_full_arm_five PASSED [ 70%]
# test_real_history_check_wc.py::test_standalone_ir_interval_excludes_zero_for_a_steady_edge_and_straddles_it_for_noise PASSED [ 76%]
# test_real_history_check_wc.py::test_evaluate_runs_with_three_arms_and_the_volatility_rule_is_a_pure_volatility_sort PASSED [ 82%]
# test_real_history_check_wc.py::test_an_edge_needs_all_three_conditions PASSED [ 88%]
# test_real_history_check_wc.py::test_the_decision_reports_each_part_separately PASSED [ 94%]
# test_real_history_check_wc.py::test_save_results_refuses_to_overwrite PASSED [100%]
# ============================== 17 passed in 5.35s ==============================
#
# Mutation checks (sandbox): ceil guard removed, floor instead of ceil, equal class weights, no horizon
#   spacing, pooled picks, pooled tilt, pooled dataset, vol feature kept in the no-vol arm (wiring), decision
#   using the upper bound, tilt cap removed, standalone upper bound, within-class IC dropped, and overwrite
#   allowed each made a test fail; original restored. One mutation is EQUIVALENT, not a gap: with classes
#   weighted by size, 'top minus class mean' summed over classes equals 'top minus universe mean'.
# Full-size synthetic run (48 ETFs x 4,700 dates, no edge): about 5 seconds.
# NOT yet run on the real snapshot. Pre-register first (preregistration_real_history_check_wc.md).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REAL-DATA RUN (mlfinlab env, 2026-10-09; prices_daily_asof_2026-10-08.csv, 8,481 dates, 48 ETFs + SH)
# Run once. PROCESS DEVIATION: the pre-registration was NOT committed before this run. The commit command
# failed (wrong git path) and the run command, chained after it, executed anyway. The script and the
# pre-registration text were unchanged between being written and being run, but "frozen at commit" did
# not hold. They are committed after the fact, together with the results.
# $ python real_history_check_wc.py
#
#                 ir  mean_ic  vol_tilt  mean_vol_rank_of_picks  ir_first_half  ir_second_half  n_periods
# wc_current   0.181    0.015     0.570                   0.712          0.136           0.242        320
# wc_no_vol   -0.022    0.010    -0.131                   0.450          0.010          -0.068        320
# wc_vol_rule  0.026    0.010     1.000                   0.869          0.000           0.062        320
#
# STANDALONE IR, 90% block-bootstrap interval
#   wc_current   [-0.13, +0.53]
#   wc_no_vol    [-0.32, +0.26]
#   wc_vol_rule  [-0.27, +0.31]
# PAIRED DIFFERENCE IN IR, 90% interval
#   wc_current-wc_vol_rule   [-0.19, +0.54]
#   wc_no_vol-wc_vol_rule    [-0.54, +0.44]
#   wc_no_vol-wc_current     [-0.55, +0.09]
#
# PRE-REGISTERED RULES: all six False (adds_beyond_volatility, positive_alone, edge_found for both arms).
#
# OUTCOME: row 1 of the pre-registered table, "neither fitted arm has edge_found". No demonstrated predictive
# edge in these features on this history. "Not shown", not "proved useless": ir standard error is near 0.2.
# NOTES: (1) Inside classes a pure volatility sort earns almost nothing (ir 0.026, mean_ic 0.010), versus
# ir 0.129 / mean_ic 0.072 pooled. Most of the pooled 'volatility signal' was a class effect (volatile
# classes vs calm ones), not selection inside classes. (2) wc_current's point estimate (ir 0.18) is the best
# of the three but its interval includes zero and its picks are still volatile (vol rank 0.71, tilt 0.57).
# (3) wc_no_vol minus wc_current [-0.55, +0.09] leans toward the volatility feature helping, descriptive only.
# (4) This is now 4 fitted variants on one history; do not pick the 0.18 and go looking for confirmation here.
# ---------------------------------------------------------------------------
