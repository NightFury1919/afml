"""Forward shadow scoring for the ETF panel (stage 1 of preregistration_etf_shadow_model.md).

Run once per trading day, after update_prices.py:
    python shadow_scoring.py --prices prices_daily_asof_2026-10-16.csv --as-of 2026-10-16

WHAT IT DOES
    Scores the 48 ranked ETFs with the three frozen fitted arms plus the volatility reference, using only
    information available at the as-of close, and writes one never-overwritten log file per day:
        shadow/scores_asof_<date>.csv        the scores, with a header of audit facts
        shadow/models/model_asof_<R>.json    the model fitted on refit day R, reused until the next refit
    Nothing is ordered. These scores only feed the forward record.

THE ARMS (frozen, see the pre-registration)
    wc_current   logistic regression (C = 1), five within-class-ranked features, unweighted
    wc_no_vol    the same without vol_60
    wc_book      wc_current plus AFML Chapter 4: average-uniqueness sample weights times time decay (c = 0.5),
                 weights scaled to average 1 so C = 1 regularises the same way
    wc_vol_rule  no model: the within-class vol_60 rank (reference only)

NO LOOK-AHEAD
    Everything is cut at the as-of date before anything is computed. A training row exists only if its 21-day
    label had resolved by then (its end date t1 is on or before the as-of date), so no unresolved label is used.

REFIT SCHEDULE
    Refit on the first scoring day (2026-10-16) and on every 21st trading day counted from it. Between refits
    the stored model is reused. If a refit day was missed, the model is fitted late from the refit date's data
    in the current snapshot and the log says late_fit: True.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from etf_classes import CLASS_OF
from etf_features import (DEFAULT_MIN_ASSETS, DEFAULT_MIN_GROUP_ASSETS, DEFAULT_WINDOWS, FEATURE_NAMES,
                          compute_features)
from etf_labels import DEFAULT_HORIZON, make_labels
from panel_data import drop_hedge, load_prices
from panel_time_decay import get_time_decay_by_date
from panel_uniqueness import get_panel_average_uniqueness

FIRST_ASOF = "2026-10-16"          # frozen: first shadow scoring date
REFIT_EVERY = 21                   # frozen: trading days between refits
DECAY_C = 0.5                      # frozen: the book's c for the wc_book arm (declared, not tuned)
LOGIT_C = 1.0                      # frozen: logistic regression regularisation, as in the history checks
NO_VOL_FEATURES = [f for f in FEATURE_NAMES if f != "vol_60"]
FEATURES_OF = {"wc_current": FEATURE_NAMES, "wc_no_vol": NO_VOL_FEATURES, "wc_book": FEATURE_NAMES}
FITTED = ["wc_current", "wc_no_vol", "wc_book"]
SCORE_COLUMNS = ["asset", "class", "wc_current", "wc_no_vol", "wc_book", "wc_vol_rule", "vol_60_rank"]


# ------------------------------------------------------------------ refit schedule
def _position(dates, day):
    day = pd.Timestamp(day)
    pos = dates.searchsorted(day)
    if pos >= len(dates) or dates[pos] != day:
        raise ValueError(f"{day.date()} is not a trading date in the price history")
    return int(pos)


def is_refit_day(dates, as_of, first_asof=FIRST_ASOF, every=REFIT_EVERY):
    """True on the first scoring day and on every `every`-th trading day after it."""
    k = _position(dates, as_of) - _position(dates, first_asof)
    return k >= 0 and k % every == 0


def last_refit_date(dates, as_of, first_asof=FIRST_ASOF, every=REFIT_EVERY):
    """The most recent refit day on or before as_of."""
    start = _position(dates, first_asof)
    k = _position(dates, as_of) - start
    if k < 0:
        raise ValueError("as_of is before the first scoring date")
    return dates[start + (k // every) * every]


# ------------------------------------------------------------------ training data and weights
def build_training(px, groups, windows=DEFAULT_WINDOWS, min_assets=DEFAULT_MIN_ASSETS,
                   min_group_assets=DEFAULT_MIN_GROUP_ASSETS, horizon=DEFAULT_HORIZON):
    """Within-class features and class-median labels on the same (date, asset) rows.

    Same construction as real_history_check_wc.build_wc_dataset. Pass prices already cut at the as-of date:
    a label needs the price `horizon` days later, so only labels resolved by then exist.
    """
    feats = compute_features(px, windows, min_assets, groups, min_group_assets)
    lab = make_labels(px, feats.index, horizon, min_assets, groups, min_group_assets)
    return feats.reindex(lab.index).join(lab)


def book_weights(ds, close, c=DECAY_C):
    """AFML Chapter 4 weights for the training rows: average uniqueness x time decay, scaled to mean 1.

    Uniqueness (Sections 4.3 and 4.4) is computed inside each ETF from its own overlapping label windows.
    Decay (Section 4.7, Snippet 4.11) runs along cumulative uniqueness and gives the newest rows weight 1 and
    the oldest weight c (before the final scaling).
    """
    close_by_asset, events_by_asset = {}, {}
    for asset in ds.index.get_level_values("asset").unique():
        rows = ds.xs(asset, level="asset")
        events_by_asset[asset] = pd.DataFrame({"t1": rows["t1"]})
        close_by_asset[asset] = close[asset].dropna()
    tw = get_panel_average_uniqueness(close_by_asset, events_by_asset).reindex(ds.index)
    if tw.isna().any():
        raise ValueError("a training row has no uniqueness value")
    w = tw * get_time_decay_by_date(tw, c)
    return w * (len(w) / w.sum())


# ------------------------------------------------------------------ fit and score
def fit_models(prices, as_of, groups, windows=DEFAULT_WINDOWS, min_assets=DEFAULT_MIN_ASSETS,
               min_group_assets=DEFAULT_MIN_GROUP_ASSETS, horizon=DEFAULT_HORIZON, c=DECAY_C, C=LOGIT_C):
    """Fit the three arms on labels that had resolved by as_of. Returns {'meta': ..., 'arms': ...}."""
    as_of = pd.Timestamp(as_of)
    px = prices.loc[:as_of]
    ds = build_training(px, groups, windows, min_assets, min_group_assets, horizon)
    if ds.empty:
        raise ValueError("no labeled training rows by this date")
    if ds["t1"].max() > as_of:
        raise ValueError("a training label ends after the as-of date")
    y = ds["label"].to_numpy()
    weights = {"wc_book": book_weights(ds, px, c).to_numpy()}
    arms = {}
    for arm in FITTED:
        lr = LogisticRegression(C=C, max_iter=200)
        lr.fit(ds[FEATURES_OF[arm]].to_numpy(), y, sample_weight=weights.get(arm))
        arms[arm] = {"features": list(FEATURES_OF[arm]), "coef": lr.coef_[0].tolist(),
                     "intercept": float(lr.intercept_[0])}
    meta = {"fit_prices_through": str(px.index[-1].date()), "n_rows": int(len(ds)),
            "max_label_end": str(ds["t1"].max().date()), "decay_c": c, "C": C}
    return {"meta": meta, "arms": arms}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def score_day(prices, as_of, model, groups, windows=DEFAULT_WINDOWS, min_assets=DEFAULT_MIN_ASSETS,
              min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """Scores for every eligible ETF on as_of, from a fitted model. Uses prices up to as_of only."""
    as_of = pd.Timestamp(as_of)
    feats = compute_features(prices.loc[:as_of], windows, min_assets, groups, min_group_assets)
    if as_of not in feats.index.get_level_values("date"):
        raise ValueError("no ETF is eligible for scoring on the as-of date")
    live = feats.xs(as_of, level="date")
    out = pd.DataFrame({"asset": live.index, "class": [groups[a] for a in live.index]})
    for arm in FITTED:
        m = model["arms"][arm]
        z = live[m["features"]].to_numpy() @ np.asarray(m["coef"]) + m["intercept"]
        out[arm] = _sigmoid(z)
    out["wc_vol_rule"] = live["vol_60"].to_numpy()
    out["vol_60_rank"] = live["vol_60"].to_numpy()
    return out[SCORE_COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------------ files
def save_model(folder, refit_date, model):
    path = Path(folder) / "models" / f"model_asof_{pd.Timestamp(refit_date).date()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:                       # "x": refuses to overwrite an existing model
        json.dump(model, f, indent=2)
    return path


def load_model(path):
    with open(path) as f:
        return json.load(f)


def write_scores(folder, as_of, scores, header):
    """One file per as-of date, '# key: value' header lines then the CSV. Never overwrites."""
    path = Path(folder) / f"scores_asof_{pd.Timestamp(as_of).date()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="") as f:           # "x": a second write for the same day fails
        for key, value in header.items():
            f.write(f"# {key}: {value}\n")
        scores.to_csv(f, index=False, lineterminator="\n")
    return path


def read_scores(path):
    """(header dict of strings, scores DataFrame) from a log file."""
    header = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("# "):
                break
            key, _, value = line[2:].rstrip("\n").partition(": ")
            header[key] = value
    return header, pd.read_csv(path, comment="#")


def _code_version():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parent, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ the daily run
def run_shadow(prices_path, as_of, folder, groups=CLASS_OF, first_asof=FIRST_ASOF, every=REFIT_EVERY,
               allow_before_start=False, code_version=None, windows=DEFAULT_WINDOWS,
               min_assets=DEFAULT_MIN_ASSETS, min_group_assets=DEFAULT_MIN_GROUP_ASSETS,
               horizon=DEFAULT_HORIZON):
    """Score one as-of date and write its log. Returns the log file path."""
    as_of, first = pd.Timestamp(as_of), pd.Timestamp(first_asof)
    pre_start = as_of < first
    if pre_start:
        if not allow_before_start:
            raise ValueError(f"{as_of.date()} is before the first scoring date {first.date()}; nothing dated "
                             "before it may enter the forward record")
        first = as_of                                 # testing only: treat this day as the first day
    prices_path = Path(prices_path)
    px = drop_hedge(load_prices(prices_path))
    if as_of not in px.index:
        raise ValueError(f"{as_of.date()} is not in the price file {prices_path.name}; run update_prices first")
    px = px.loc[:as_of]
    refit = last_refit_date(px.index, as_of, first, every)
    kw = dict(windows=windows, min_assets=min_assets, min_group_assets=min_group_assets)

    model_path = Path(folder) / "models" / f"model_asof_{refit.date()}.json"
    if model_path.exists():
        model = load_model(model_path)
    else:
        model = fit_models(px, refit, groups, horizon=horizon, **kw)
        model["meta"]["late_fit"] = bool(refit != as_of)
        save_model(folder, refit, model)
    meta = model["meta"]

    scores = score_day(px, as_of, model, groups, **kw)
    header = {
        "as_of": str(as_of.date()),
        "first_scoring_date": str(first.date()),
        "last_refit_date": str(refit.date()),
        "refit_today": bool(refit == as_of),
        "late_fit": bool(meta.get("late_fit", False)),
        "fit_prices_through": meta["fit_prices_through"],
        "model_rows": meta["n_rows"],
        "max_label_end": meta["max_label_end"],
        "decay_c": meta["decay_c"],
        "logit_C": meta["C"],
        "n_etfs_scored": len(scores),
        "snapshot_file": prices_path.name,
        "snapshot_sha256": hashlib.sha256(prices_path.read_bytes()).hexdigest(),
        "code_version": code_version or _code_version(),
    }
    if pre_start:
        header["pre_start"] = True
    return write_scores(folder, as_of, scores, header)


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", required=True, help="price snapshot CSV that includes the as-of date")
    ap.add_argument("--as-of", dest="as_of", default=None, help="default: the last date in the price file")
    ap.add_argument("--folder", default=str(here / "shadow"))
    ap.add_argument("--code-version", default=None)
    ap.add_argument("--allow-before-start", action="store_true", help="testing only")
    args = ap.parse_args(argv)
    as_of = args.as_of or str(load_prices(args.prices).index[-1].date())
    try:
        path = run_shadow(args.prices, as_of, args.folder, code_version=args.code_version,
                          allow_before_start=args.allow_before_start)
    except (FileExistsError, ValueError) as err:
        print(f"REFUSED: {err}")
        return 1
    header, scores = read_scores(path)
    print(f"Scored {len(scores)} ETFs for {header['as_of']} (model from {header['last_refit_date']}, "
          f"refit today: {header['refit_today']}, late fit: {header['late_fit']})")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_shadow_scoring.py -v
#
# collected 34 items
#
# test_shadow_scoring.py::test_refit_day_is_the_first_day_and_every_21st_trading_day_after_it PASSED [  2%]
# test_shadow_scoring.py::test_a_day_before_the_first_scoring_date_is_not_a_refit_day PASSED [  5%]
# test_shadow_scoring.py::test_days_exactly_one_or_two_periods_before_the_first_date_are_not_refit_days PASSED [  8%]
# test_shadow_scoring.py::test_last_refit_date_is_the_most_recent_refit_on_or_before_the_as_of_date PASSED [ 11%]
# test_shadow_scoring.py::test_last_refit_date_rejects_a_date_before_the_first_scoring_date PASSED [ 14%]
# test_shadow_scoring.py::test_a_non_trading_date_is_rejected PASSED       [ 17%]
# test_shadow_scoring.py::test_training_rows_are_only_labels_that_had_resolved_by_the_as_of_date PASSED [ 20%]
# test_shadow_scoring.py::test_fit_ignores_everything_after_the_as_of_date PASSED [ 23%]
# test_shadow_scoring.py::test_scores_ignore_everything_after_the_as_of_date PASSED [ 26%]
# test_shadow_scoring.py::test_scoring_hands_the_feature_code_prices_only_up_to_the_as_of_date PASSED [ 29%]
# test_shadow_scoring.py::test_fit_refuses_training_data_whose_labels_end_after_the_as_of_date PASSED [ 32%]
# test_shadow_scoring.py::test_training_matches_the_history_check_dataset_at_the_frozen_defaults PASSED [ 35%]
# test_shadow_scoring.py::test_the_current_arm_matches_a_plain_logistic_regression_on_all_five_features PASSED [ 38%]
# test_shadow_scoring.py::test_the_no_volatility_arm_has_four_features_and_never_sees_vol_60 PASSED [ 41%]
# test_shadow_scoring.py::test_the_book_arm_uses_the_weights_and_differs_from_the_unweighted_arm PASSED [ 44%]
# test_shadow_scoring.py::test_the_book_arm_decay_value_is_the_declared_half_and_is_recorded PASSED [ 47%]
# test_shadow_scoring.py::test_book_weights_known_values_uniqueness_times_half_decay_scaled_to_mean_one PASSED [ 50%]
# test_shadow_scoring.py::test_book_weights_without_decay_are_just_scaled_uniqueness PASSED [ 52%]
# test_shadow_scoring.py::test_book_weights_keep_the_row_order_of_the_dataset PASSED [ 55%]
# test_shadow_scoring.py::test_the_logistic_formula_matches_sklearn_predict_proba PASSED [ 58%]
# test_shadow_scoring.py::test_the_volatility_rule_score_is_the_within_class_vol_60_rank PASSED [ 61%]
# test_shadow_scoring.py::test_scores_have_the_registered_columns_one_row_per_etf_and_probabilities_between_zero_and_one PASSED [ 64%]
# test_shadow_scoring.py::test_model_file_round_trips_and_is_never_overwritten PASSED [ 67%]
# test_shadow_scoring.py::test_score_log_has_a_header_the_scores_and_refuses_to_overwrite PASSED [ 70%]
# test_shadow_scoring.py::test_first_day_fits_a_model_and_the_next_day_reuses_it PASSED [ 73%]
# test_shadow_scoring.py::test_day_21_refits_and_a_new_model_file_appears PASSED [ 76%]
# test_shadow_scoring.py::test_a_missed_refit_day_is_fitted_late_on_the_refit_dates_data_and_flagged PASSED [ 79%]
# test_shadow_scoring.py::test_header_records_the_snapshot_hash_the_code_version_and_training_facts PASSED [ 82%]
# test_shadow_scoring.py::test_running_the_same_day_twice_is_refused_and_leaves_the_first_log_alone PASSED [ 85%]
# test_shadow_scoring.py::test_a_date_before_the_first_scoring_date_is_refused_unless_testing PASSED [ 88%]
# test_shadow_scoring.py::test_an_as_of_date_missing_from_the_price_file_is_refused PASSED [ 91%]
# test_shadow_scoring.py::test_the_hedge_ticker_is_never_scored PASSED     [ 94%]
# test_shadow_scoring.py::test_two_independent_runs_give_identical_scores PASSED [ 97%]
# test_shadow_scoring.py::test_command_line_run_at_the_frozen_defaults PASSED [100%]
#
# ============================= 34 passed in 26.35s ==============================
#
# Mutation checks (sandbox): 42 deliberate bugs, all caught. Refit every day / never on day 0 / negative offsets
#   counted as refit days / last refit off by one or rounded up; fit or score not cut at the as-of date; label-end
#   guard removed; book arm unweighted / weights on every arm / decay ignored / uniqueness ignored / weights not
#   scaled to mean 1; declared c, C, first date or refit interval changed; no-vol arm keeping vol_60 or dropping
#   the wrong feature; sign or intercept of the logistic formula wrong; wrong vol-rule or rank column; wrong class
#   column; model or log overwrite allowed; header not written; stored model not reused; late fit never flagged;
#   refit fitted on the as-of day instead of the refit day; refit_today wrong; hash of the wrong thing; before-start
#   dates allowed or unmarked; missing as-of date accepted; hedge ETF kept; exit code 0 on refusal; pooled instead of
#   within-class training; label horizon ignored. Three survived at first (negative offsets, the no-look-ahead
#   guards in fit and score) and each now has a test.
#
# REAL-DATA CHECK (sandbox, prices_daily_asof_2026-10-08.csv, a code test only: the real forward record starts
# 2026-10-16 and nothing from this run is part of it)
# $ python shadow_scoring.py --prices prices_daily_asof_2026-10-08.csv --as-of 2026-10-08 --allow-before-start
#   Scored 48 ETFs; training rows 283,762 (identical to build_wc_dataset on the same file); the wc_current
#   coefficients equal a direct LogisticRegression fit on those rows to 4 decimals. About 2.5 minutes on a refit
#   day (the uniqueness step), seconds on other days.
#   Rank correlation of the scores across the 48 ETFs: wc_current vs wc_book 0.99, vs wc_no_vol 0.61, vs the
#   volatility rule 0.68. wc_book picks almost the same ETFs as wc_current. Scores sit between 0.45 and 0.51:
#   the model is only weakly confident, as expected with no demonstrated edge.
# NOT yet run on the 2026-10-16 snapshot in the mlfinlab env.
# ---------------------------------------------------------------------------
