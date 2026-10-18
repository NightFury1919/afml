"""Tier B noise test: run the REAL daily shadow pipeline on a few worlds with no edge.

Registered in preregistration_noise_test_etf_shadow_tier_b.md. Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python noise_test_tier_b.py --prices prices_daily_asof_2026-10-16.csv --workers 4

WHAT IT DOES
    World     the real 48-ETF daily log returns over the period when all 48 exist, each ETF's average removed, resampled
              in moving blocks of 63 whole days (as in positive_control_etf.py), turned into prices. No ETF has a
              higher expected return than another and nothing forecasts anything: a true no-edge world, with real fat
              tails, volatility clustering and cross-ETF correlation. HIST days of history, then 36 scoring dates 21
              days apart, then 21 more days so the last period resolves.
    Pipeline  shadow_scoring.run_shadow is called on each scoring date exactly as the nightly job calls it: the first
              date is the first scoring date, every scoring date is a refit day (they are 21 apart), the model is fitted
              from the data up to that day, all three fitted arms and the volatility rule are scored, a log file is
              written. Nothing is changed from the live settings.
    Judging   the 36 log files give the same statistics the formal evaluation uses: class-neutral active returns, ir,
              mean IC, vol_tilt, and the registered rule shadow_evaluate.decide_formal.

SIZE
    Each world refits 36 times. HIST = 1260 days (5 years) keeps one world to roughly 15 to 30 minutes; the live model
    will have 30+ years. Eight worlds can only expose gross faults (a leak or a bug that makes noise look skilled),
    not measure a false-positive rate; Tier A measured that.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from etf_classes import CLASS_OF
from etf_features import DEFAULT_MIN_ASSETS, DEFAULT_MIN_GROUP_ASSETS
from etf_labels import DEFAULT_HORIZON, make_labels
from panel_data import drop_hedge, load_prices
from real_history_check_wc import _mean_class_spearman, active_series_within_class, within_class_tilt
import shadow_evaluate as ev
import shadow_scoring as ss

HIST = 1260
PERIODS = 36
STEP = ss.REFIT_EVERY               # 21 trading days between scoring dates
BLOCK = 63
WORLDS = 8
BASE_SEED = 20261018
HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- the world
def common_returns(prices):
    """Daily log returns of the 48 ranked ETFs over the days when all of them exist, each ETF's mean removed."""
    px = drop_hedge(prices)
    r = np.log(px).diff().dropna(how="any")
    return r - r.mean()


def world_length(hist=HIST, periods=PERIODS, step=STEP, horizon=DEFAULT_HORIZON):
    """Days needed so the last scoring date's forward return is known."""
    return hist + step * (periods - 1) + horizon + 1


def make_world_prices(returns, n_days, seed, block=BLOCK):
    """Prices from a moving-block bootstrap of whole rows of `returns`; synthetic business-day dates."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    rows = np.concatenate([np.arange(s, s + block) for s in starts])[:n_days]
    r = returns.iloc[rows].copy()
    r.index = pd.bdate_range("2000-01-03", periods=n_days)
    return 100.0 * np.exp(r.cumsum())


def scoring_dates(dates, hist=HIST, periods=PERIODS, step=STEP):
    return [dates[hist + step * k] for k in range(periods)]


# ------------------------------------------------------------------ the pipeline
def run_world(prices, prices_path, folder, hist=HIST, periods=PERIODS, step=STEP, groups=CLASS_OF, shadow_kw=None):
    """Write the prices file, then call run_shadow on every scoring date. Returns the log paths in date order."""
    prices_path = Path(prices_path)
    prices.to_csv(prices_path, index_label="date")
    days = scoring_dates(prices.index, hist, periods, step)
    first = days[0]
    return [ss.run_shadow(prices_path, d, folder, groups=groups, first_asof=str(first.date()), every=step,
                          code_version="tier_b", **(shadow_kw or {})) for d in days]


# ------------------------------------------------------------------ the statistics
def evaluate_world(prices, log_paths, groups=CLASS_OF, step=STEP, horizon=DEFAULT_HORIZON,
                   min_assets=DEFAULT_MIN_ASSETS, min_group_assets=DEFAULT_MIN_GROUP_ASSETS, n_boot=None):
    """One result row from the log files of one world."""
    frames, headers = [], []
    for p in log_paths:
        h, s = ss.read_scores(p)
        s["date"] = pd.Timestamp(h["as_of"])
        frames.append(s)
        headers.append(h)
    scores = pd.concat(frames).set_index(["date", "asset"]).sort_index()
    labels = make_labels(drop_hedge(prices), scores.index, horizon, min_assets, groups, min_group_assets)
    scores = scores.reindex(labels.index)
    fwd, excess = labels["fwd_ret"], labels["excess_ret"]
    arms = {a: scores[a] for a in ev.ARMS}
    active = pd.DataFrame({a: active_series_within_class(p, fwd, groups, horizon=1) for a, p in arms.items()})
    tilt = {a: within_class_tilt(arms[a], scores["vol_60_rank"], groups) for a in ev.FITTED}
    kw = {} if n_boot is None else dict(n_boot=n_boot)
    res = ev.decide_formal(active, tilt, periods=len(active), **kw)
    row = dict(n_periods=len(active),
               all_dates_logged=len(log_paths) == len(set(h["as_of"] for h in headers)),
               min_etfs_scored=min(int(h["n_etfs_scored"]) for h in headers),
               all_refit_today=all(h["refit_today"] == "True" for h in headers),
               any_late_fit=any(h["late_fit"] == "True" for h in headers),
               ir_vol_rule=ev.ir(active["wc_vol_rule"]))
    for arm in ev.FITTED:
        r = res[arm]
        row.update({f"{arm}_ir": r["ir"], f"{arm}_mean_ic": _mean_class_spearman(arms[arm], excess, groups),
                    f"{arm}_vol_tilt": r["vol_tilt"], f"{arm}_adds_beyond_volatility": r["adds_beyond_volatility"],
                    f"{arm}_positive_alone": r["positive_alone"], f"{arm}_edge_found": r["edge_found"]})
    row["any_edge_found"] = any(res[a]["edge_found"] for a in ev.FITTED)
    return row


def judge(results):
    """Pass rule of the registration: integrity everywhere, and no arm's mean ir or mean IC more than 2 SE from zero."""
    n = len(results)
    out = dict(n_worlds=n,
               integrity=bool(results["all_dates_logged"].all() and results["all_refit_today"].all()
                              and not results["any_late_fit"].any() and (results["min_etfs_scored"] == 48).all()
                              and (results["n_periods"] == PERIODS).all()))
    for arm in ev.FITTED:
        for stat in ("ir", "mean_ic"):
            x = results[f"{arm}_{stat}"].to_numpy(float)
            se = x.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
            out[f"{arm}_{stat}_mean"] = float(x.mean())
            out[f"{arm}_{stat}_se"] = float(se)
            out[f"{arm}_{stat}_ok"] = bool(abs(x.mean()) <= 2 * se)
    out["no_edge_claimed_by_noise"] = bool(
        all(out[f"{a}_{s}_ok"] for a in ev.FITTED for s in ("ir", "mean_ic")))
    out["passed"] = bool(out["integrity"] and out["no_edge_claimed_by_noise"])
    return out


# ----------------------------------------------------------------------- run
def one_world(args):
    returns, seed, folder, hist, periods, step, groups, shadow_kw = args
    folder = Path(folder) / f"world_{seed}"
    result = folder / "result.csv"
    if result.exists():
        raise FileExistsError(f"{result} already exists; a registered run is done once")
    folder.mkdir(parents=True, exist_ok=True)
    prices = make_world_prices(returns, world_length(hist, periods, step), seed)
    logs = run_world(prices, folder / "prices.csv", folder / "shadow", hist, periods, step, groups, shadow_kw)
    row = dict(seed=seed, **evaluate_world(prices, logs, groups, step, **_label_kw(shadow_kw)))
    pd.DataFrame([row]).to_csv(result, index=False)
    return row


def _label_kw(shadow_kw):
    kw = dict(shadow_kw or {})
    out = {}
    for src, dst in (("horizon", "horizon"), ("min_assets", "min_assets"), ("min_group_assets", "min_group_assets")):
        if src in kw:
            out[dst] = kw[src]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", required=True)
    ap.add_argument("--worlds", type=int, default=WORLDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--folder", default=str(HERE / "noise_test_tier_b"))
    args = ap.parse_args(argv)
    summary_path = Path(args.folder) / "summary.csv"
    if summary_path.exists():
        print(f"REFUSED: {summary_path} already exists; a registered run is done once")
        return 1
    returns = common_returns(load_prices(args.prices))
    if len(returns) < 2 * BLOCK:
        print("REFUSED: too few days when all ETFs exist")
        return 1
    jobs = [(returns, BASE_SEED + i, args.folder, HIST, PERIODS, STEP, CLASS_OF, None) for i in range(args.worlds)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one_world, jobs))
    else:
        rows = [one_world(j) for j in jobs]
    res = pd.DataFrame(rows)
    res.to_csv(summary_path, index=False)
    verdict = judge(res)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res.round(3).T.to_string())
    for k, v in verdict.items():
        print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# TDD RESULTS (sandbox py3.10 venv, 2026-10-17): test_noise_test_tier_b.py, 25 passed. Mutation checks: 26 mutants of
# this file (settings, no demeaning, hedge kept, world too short, scoring-date offset, wrong first date or interval,
# wrong columns, integrity checks dropped, 2 SE changed to 3 SE, SE formula, ir check dropped, overwrite allowed,
# passed ignoring integrity): all caught except one that is equivalent (excess return and raw return rank the same
# within a class). One timing run on a 5-year world: first fit and score 26 s, last 45 s.
# Not yet run in the mlfinlab env. The registered run (Tier B) has not been run.
# ---------------------------------------------------------------------------
