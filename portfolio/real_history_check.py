"""Real-history check: does the baseline model add anything beyond 'hold the most volatile ETFs'?

Pre-registered in preregistration_real_history_check.md. Run ONCE:
    python real_history_check.py                    # uses prices_daily_asof_2026-10-08.csv

WHY
    The first dry-run plan was 0.92 rank-correlated with 60-day volatility and 7 of its 10 picks were
    among the 10 most volatile ETFs. Before any paper money, test whether the model beats a plain
    volatility sort on the REAL history, and whether the book's remedy (scale by volatility, AFML
    Chapter 3.2 and 3.3) removes the tilt.

THREE ARMS (all scored out-of-fold with PanelPurgedKFold, 5 folds, 252-date embargo)
    current     logistic regression on the frozen label: forward 21-day return beats the cross-
                sectional median (raw returns).
    vol_scaled  the same model and features, but the label compares forward return divided by the
                ETF's own volatility at the label date (daily volatility x sqrt(21)).
    vol_rule    no model: score = the vol_60 rank. Hold the most volatile fifth.

HOW THEY ARE JUDGED (all on raw forward returns, on dates 21 apart)
    ir          annualized information ratio of 'hold the top fifth by score' minus the equal-weight
                universe (the same statistic as the positive control).
    mean_ic     mean out-of-fold Spearman IC against raw 21-day excess return.
    vol_tilt    mean over dates of the Spearman correlation between the score and the vol_60 rank.
    paired CIs  moving-block bootstrap (block 4 periods) of the difference in ir between arms.

Volatility (documented deviation from Snippet 3.1): the book's get_daily_vol looks "one day back" with
a calendar-day search that suits intraday bars. On daily bars it lands 1 to 3 bars back depending on the
weekday and drops the first two dates. Here the same estimator (exponentially weighted standard
deviation, span 100) is applied to plain one-bar returns, with min_periods = span.

Limits: k-fold purged CV trains on later periods too (standard AFML CV, not walk-forward); about 18
years of data gives an IR standard error near 0.25 per arm, so small differences cannot be detected;
the 48 ETFs are survivors of today's universe.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etf_features import DEFAULT_MIN_ASSETS, DEFAULT_WINDOWS, compute_features
from etf_labels import DEFAULT_HORIZON, make_labels
from panel_data import drop_hedge, load_prices, wide_to_long
from positive_control_etf import date_ics, out_of_fold_predictions

SPAN0 = 100
BOOT_BLOCK = 4            # periods (about 84 trading days)
BOOT_SEED = 20261009
N_BOOT = 5000
CI_LEVEL = 0.90
MAX_VOL_TILT = 0.7        # a score this correlated with volatility is still a volatility sort
MIN_TILT_REDUCTION = 0.4  # scaling "works" if it lowers the tilt by at least this much (about two
                          # sampling spreads of the null; see the pre-registration for the calibration)
DEFAULT_PRICES = "prices_daily_asof_2026-10-08.csv"
RESULTS = "real_history_check_results.csv"
ACTIVE = "real_history_check_active_returns.csv"
ARMS = ["current", "vol_scaled", "vol_rule"]
PAIRS = [("current", "vol_rule"), ("vol_scaled", "vol_rule"), ("vol_scaled", "current")]


# ------------------------------------------------------------------ volatility
def daily_vol_panel(prices, span0=SPAN0):
    """Exponentially weighted std of one-bar returns, per ETF (see the module note on Snippet 3.1)."""
    return prices.pct_change().ewm(span=span0, min_periods=span0).std()


def vol_scaled_labels(prices, index, vol, horizon=DEFAULT_HORIZON, min_assets=DEFAULT_MIN_ASSETS):
    """Labels where each ETF's forward return is divided by vol x sqrt(horizon) before ranking.

    Columns: fwd_ret, excess_ret (raw), raw_label (the frozen label), scaled_excess, label (scaled),
    t1. Rows without a volatility estimate are dropped, and a date still needs min_assets ETFs.
    """
    cols = ["fwd_ret", "excess_ret", "raw_label", "scaled_excess", "label", "t1"]
    base = make_labels(prices, index, horizon, min_assets)
    if base.empty:
        return pd.DataFrame(columns=cols)
    sig = wide_to_long(vol * np.sqrt(horizon), "sig").reindex(base.index)
    scaled = (base["fwd_ret"] / sig).replace([np.inf, -np.inf], np.nan)
    d = base[scaled.notna()].copy()
    d["scaled"] = scaled[scaled.notna()]
    d = d[d.groupby(level="date")["scaled"].transform("size") >= min_assets]
    if d.empty:
        return pd.DataFrame(columns=cols)
    excess = d["scaled"] - d.groupby(level="date")["scaled"].transform("median")
    out = pd.DataFrame({"fwd_ret": d["fwd_ret"], "excess_ret": d["excess_ret"], "raw_label": d["label"],
                        "scaled_excess": excess, "label": (excess > 0).astype(int), "t1": d["t1"]})
    return out[cols]


def build_arm_datasets(prices, min_assets=DEFAULT_MIN_ASSETS, horizon=DEFAULT_HORIZON, span0=SPAN0):
    """{'current': ..., 'vol_scaled': ...}: the same rows and features, different labels."""
    px = drop_hedge(prices)
    feats = compute_features(px, DEFAULT_WINDOWS, min_assets)
    sl = vol_scaled_labels(px, feats.index, daily_vol_panel(px, span0), horizon, min_assets)
    cur = feats.reindex(sl.index).join(sl[["fwd_ret", "excess_ret", "t1"]])
    cur["label"] = sl["raw_label"]
    scaled = cur.copy()
    scaled["label"] = sl["label"]
    return {"current": cur, "vol_scaled": scaled}


# ------------------------------------------------------------------ statistics
def active_series(pred, fwd_ret, horizon=DEFAULT_HORIZON, top_share=0.2):
    """Per-period return of 'hold the top fifth by pred' minus the equal-weight universe, every
    `horizon`-th date (non-overlapping)."""
    pct = pred.groupby(level="date").rank(pct=True)
    top = pct > (1.0 - top_share)
    top_mean = fwd_ret.where(top).groupby(level="date").mean()
    all_mean = fwd_ret.groupby(level="date").mean()
    return (top_mean - all_mean).dropna().iloc[::horizon]


def annualized_ir(active, horizon=DEFAULT_HORIZON):
    a = np.asarray(active, dtype=float)
    return float(a.mean() / a.std(ddof=1) * np.sqrt(252.0 / horizon))


def vol_tilt(pred, vol_rank):
    """Mean over dates of the Spearman correlation between a score and the vol_60 rank."""
    return float(date_ics(pred, vol_rank).mean())


def paired_block_bootstrap_ci(a, b, horizon=DEFAULT_HORIZON, block=BOOT_BLOCK, n_boot=N_BOOT,
                              seed=BOOT_SEED, level=CI_LEVEL):
    """(low, high) interval for ir(a) - ir(b), resampling the same blocks of periods for both."""
    common = a.index.intersection(b.index)
    x, y = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        rows = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        diffs[k] = annualized_ir(x[rows], horizon) - annualized_ir(y[rows], horizon)
    lo, hi = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


# -------------------------------------------------------------------- the run
def evaluate(prices, n_splits=5, embargo_dates=252, min_assets=DEFAULT_MIN_ASSETS,
             horizon=DEFAULT_HORIZON, C=1.0, n_boot=N_BOOT):
    arms = build_arm_datasets(prices, min_assets, horizon)
    cur = arms["current"]
    preds = {"current": out_of_fold_predictions(cur, n_splits, embargo_dates, C),
             "vol_scaled": out_of_fold_predictions(arms["vol_scaled"], n_splits, embargo_dates, C),
             "vol_rule": cur["vol_60"]}
    fwd, excess, vrank = cur["fwd_ret"], cur["excess_ret"], cur["vol_60"]

    active, summary = {}, {}
    for name in ARMS:
        pred = preds[name]
        act = active_series(pred, fwd, horizon)
        active[name] = act
        pct = pred.groupby(level="date").rank(pct=True)
        half = len(act) // 2
        summary[name] = dict(
            ir=annualized_ir(act, horizon), mean_ic=float(date_ics(pred, excess).mean()),
            vol_tilt=vol_tilt(pred, vrank),
            mean_vol_rank_of_picks=float(vrank[pct > 0.8].mean()),
            ir_first_half=annualized_ir(act.iloc[:half], horizon),
            ir_second_half=annualized_ir(act.iloc[half:], horizon), n_periods=int(len(act)))
    paired = {f"{x}-{y}": paired_block_bootstrap_ci(active[x], active[y], horizon, n_boot=n_boot)
              for x, y in PAIRS}
    return dict(arms=summary, paired=paired, active=pd.DataFrame(active))


def decide(res):
    """Apply the pre-registered rules to an evaluate() result."""
    out = {}
    for arm in ("current", "vol_scaled"):
        lo, _ = res["paired"][f"{arm}-vol_rule"]
        out[f"{arm}_adds_beyond_volatility"] = bool(lo > 0 and res["arms"][arm]["vol_tilt"] <= MAX_VOL_TILT)
    drop = res["arms"]["current"]["vol_tilt"] - res["arms"]["vol_scaled"]["vol_tilt"]
    out["scaling_reduces_tilt"] = bool(drop >= MIN_TILT_REDUCTION - 1e-12)
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


def save_results(res, folder):
    check_not_run(folder)
    rows = []
    for arm, m in res["arms"].items():
        rows += [dict(kind="arm", name=arm, metric=k, value=v) for k, v in m.items()]
    for pair, (lo, hi) in res["paired"].items():
        rows += [dict(kind="paired_ir_diff", name=pair, metric="ci_low", value=lo),
                 dict(kind="paired_ir_diff", name=pair, metric="ci_high", value=hi)]
    rows += [dict(kind="decision", name=k, metric="rule", value=v) for k, v in decide(res).items()]
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
    check_not_run(args.folder)                      # fail before the slow part

    prices = load_prices(args.prices)
    print(f"Prices: {Path(args.prices).name}, {prices.shape[0]} dates, {prices.shape[1]} tickers")
    res = evaluate(prices)

    print("\nARMS (hold the top fifth by score, minus the equal-weight universe)")
    print(pd.DataFrame(res["arms"]).T.round(3).to_string())
    print(f"\nPAIRED DIFFERENCE IN IR, {CI_LEVEL:.0%} block-bootstrap interval")
    for pair, (lo, hi) in res["paired"].items():
        print(f"  {pair:22s} [{lo:+.2f}, {hi:+.2f}]")
    print("\nPRE-REGISTERED RULES")
    for k, v in decide(res).items():
        print(f"  {k}: {v}")
    r, a = save_results(res, args.folder)
    print(f"\nSaved {r}\n      {a}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_real_history_check.py -v
#
# collected 14 items
#
# test_real_history_check.py::test_daily_vol_is_an_exponentially_weighted_std_of_one_day_returns PASSED [  7%]
# test_real_history_check.py::test_daily_vol_reacts_on_the_day_of_a_jump_not_two_days_later PASSED [ 14%]
# test_real_history_check.py::test_daily_vol_uses_only_data_up_to_each_date PASSED [ 21%]
# test_real_history_check.py::test_scaling_by_volatility_flips_the_label_of_the_volatile_etf PASSED [ 28%]
# test_real_history_check.py::test_rows_without_a_volatility_estimate_are_dropped_and_dates_stay_balanced PASSED [ 35%]
# test_real_history_check.py::test_scaled_labels_never_use_future_volatility PASSED [ 42%]
# test_real_history_check.py::test_active_series_matches_the_positive_control_top_quintile_ir PASSED [ 50%]
# test_real_history_check.py::test_vol_tilt_is_plus_one_for_a_vol_sort_and_minus_one_for_its_reverse PASSED [ 57%]
# test_real_history_check.py::test_bootstrap_ci_of_a_constant_advantage_excludes_zero_and_of_identical_series_straddles_it PASSED [ 64%]
# test_real_history_check.py::test_arms_share_features_and_rows_and_differ_only_in_the_label PASSED [ 71%]
# test_real_history_check.py::test_evaluate_runs_and_the_volatility_rule_wins_when_volatility_really_pays PASSED [ 78%]
# test_real_history_check.py::test_decision_requires_both_a_clear_gain_over_the_volatility_rule_and_a_small_volatility_tilt PASSED [ 85%]
# test_real_history_check.py::test_decision_on_whether_scaling_removes_the_volatility_tilt PASSED [ 92%]
# test_real_history_check.py::test_save_results_refuses_to_overwrite PASSED [100%]
#
# 14 passed in 2.54s
# Mutation checks (sandbox): two-bar returns, no volatility scaling in the label, decision using the
#   upper interval bound, tilt check removed, unpaired bootstrap, overwrite allowed, future volatility,
#   and overlapping periods each made tests fail; original restored.
# Sandbox calibration (synthetic no-edge worlds, not the real data): ir spread about 0.3 per arm; tilt
#   difference between label designs sd 0.20 over 16 worlds, which set MIN_TILT_REDUCTION = 0.4.
# Full-size synthetic run (48 ETFs x 4,700 dates) took about 5 seconds.
# NOT yet run on the real snapshot. Pre-register first (preregistration_real_history_check.md).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REAL-DATA RUN (mlfinlab env, 2026-10-09; prices_daily_asof_2026-10-08.csv, 8,481 dates, 48 ETFs + SH)
# Pre-registered in preregistration_real_history_check.md (committed before this run). Run once.
# $ python real_history_check.py
#
#                ir  mean_ic  vol_tilt  mean_vol_rank_of_picks  ir_first_half  ir_second_half  n_periods
# current     0.237    0.073     0.885                   0.847          0.279           0.185      320
# vol_scaled -0.031    0.010    -0.339                   0.330          0.039          -0.121      320
# vol_rule    0.129    0.072     1.000                   0.895          0.123           0.144      320
#
# PAIRED DIFFERENCE IN IR, 90% block-bootstrap interval
#   current-vol_rule       [-0.09, +0.28]
#   vol_scaled-vol_rule    [-0.67, +0.34]
#   vol_scaled-current     [-0.73, +0.21]
#
# PRE-REGISTERED RULES
#   current_adds_beyond_volatility:    False  (interval includes 0; tilt 0.885 is above 0.7)
#   vol_scaled_adds_beyond_volatility: False  (interval includes 0)
#   scaling_reduces_tilt:              True   (0.885 - (-0.339) = 1.22, needed 0.4)
#
# OUTCOME: row 1 of the pre-registered table, "neither arm adds beyond volatility". The baseline is
# indistinguishable from a volatility sort on the real history. It is not proof of no skill: 320 periods
# (about 27 years) give each arm an ir standard error near 0.2, so only a large gap could pass.
# NOTES: (1) The model's mean_ic (0.073) equals the volatility rule's (0.072): all of the baseline's rank IC
# is the volatility signal, and rank IC overstates its worth (the vol rule's top-fifth ir is only 0.13).
# (2) The vol-scaled label over-corrected: it flipped the tilt to low-volatility ETFs (picks' vol rank 0.33)
# and earned nothing (ir -0.03). (3) There were 320 periods, more than the roughly 210 assumed in the
# pre-registration, because early dates have 10+ ETFs; those early years use a smaller universe.
# ---------------------------------------------------------------------------
