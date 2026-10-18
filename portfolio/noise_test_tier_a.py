"""Tier A noise test: how often does the registered formal evaluation rule fire when there is NO edge?

Registered in preregistration_noise_test_etf_shadow.md. Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python noise_test_tier_a.py --prices prices_daily_asof_2026-10-16.csv --workers 4

WHAT IS SIMULATED
    One world = 36 resolved forward periods for the 48 ranked ETFs in their 6 classes (the exact point at which the
    formal evaluation runs). Each period is an independent cross-section:
      returns   ret_i = sigma_i * (0.6 * class_shock + 0.8 * idiosyncratic) + beta * sigma_i * s_i
                shocks are Student-t(5) scaled to unit variance (fat tails); sigma_i is each ETF's real 21-day
                volatility (from --prices); s_i is a within-class N(0,1) "true signal", planted only when beta > 0.
      scores    the reference wc_vol_rule score is the within-class volatility rank (standardized). Each fitted arm is
                tilt * vol_rank + sqrt(1 - tilt^2) * (q * s + sqrt(1 - q^2) * arm_noise), q = 0.5, so the arms see the
                signal only imperfectly. Tilts: wc_current 0.5, wc_no_vol 0.2, wc_book 0.5. Arm noise is shared
                (correlation 0.9 between arms, 0.98 between wc_current and wc_book), as the real arms are close cousins.
    beta = 0 is a true no-edge world: scores are independent of returns. beta > 0 plants an edge and gives power.
    Each world's per-period class-neutral active returns and vol_tilts go through shadow_evaluate.decide_formal,
    the same code the real evaluation will use.

WHAT IS REPORTED, per beta: realized mean ir of wc_current, and the share of worlds where each fitted arm passes each
rule, plus the share where at least one of the three arms has edge_found (the family-wise rate). At beta = 0 these are
false-positive rates, each with a Monte Carlo standard error.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from etf_classes import CLASS_OF
from panel_data import drop_hedge, load_prices
from real_history_check_wc import active_series_within_class, within_class_tilt
import shadow_evaluate as ev

PERIODS = 36
DF = 5                                   # Student-t degrees of freedom for the shocks
TILTS = {"wc_current": 0.5, "wc_no_vol": 0.2, "wc_book": 0.5}
Q = 0.5                                  # share of each arm's non-vol score that is the true signal
RHO_ARMS = 0.9                           # shared noise between the arms
RHO_BOOK = 0.98                          # extra closeness of wc_book to wc_current
BETAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
N_WORLDS = 2000
BASE_SEED = 20261017
HERE = Path(__file__).resolve().parent


def _t(rng, size):
    return rng.standard_t(DF, size=size) / np.sqrt(DF / (DF - 2.0))


def universe_params(prices):
    """(assets, class of each asset, 21-day sigma of each asset) from a price file's last five years."""
    px = drop_hedge(prices)
    rets = np.log(px).diff().iloc[-1260:]
    sigma = (rets.std() * np.sqrt(21)).dropna()
    assets = [a for a in px.columns if a in sigma.index]
    return assets, [CLASS_OF[a] for a in assets], sigma.loc[assets].to_numpy()


def _within_class_z(values, classes):
    """Within-class rank of `values`, centred and scaled to unit variance, per class (ranks are unique)."""
    out = np.empty(len(values))
    for c in set(classes):
        idx = [i for i, k in enumerate(classes) if k == c]
        r = pd.Series(values[idx]).rank(method="first").to_numpy()
        r = r - r.mean()
        out[idx] = r / (r.std() if len(idx) > 1 and r.std() > 0 else 1.0)
    return out


def simulate_world(assets, classes, sigma, beta, seed, periods=PERIODS):
    """(active DataFrame [period x arm], vol_tilt dict) for one simulated world."""
    rng = np.random.default_rng(seed)
    n = len(assets)
    cls_list = sorted(set(classes))
    cls_idx = np.array([cls_list.index(c) for c in classes])
    vol_z = _within_class_z(sigma, classes)
    dates = pd.date_range("2030-01-01", periods=periods, freq="21B")
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])
    groups = dict(zip(assets, classes))

    shock = _t(rng, (periods, len(cls_list)))[:, cls_idx]
    idio = _t(rng, (periods, n))
    signal = np.vstack([_within_class_z(rng.normal(size=n), classes) for _ in range(periods)])
    ret = sigma * (0.6 * shock + 0.8 * idio) + beta * sigma * signal

    g = rng.normal(size=(periods, n))
    own = {a: rng.normal(size=(periods, n)) for a in TILTS}
    noise = {"wc_current": np.sqrt(RHO_ARMS) * g + np.sqrt(1 - RHO_ARMS) * own["wc_current"],
             "wc_no_vol": np.sqrt(RHO_ARMS) * g + np.sqrt(1 - RHO_ARMS) * own["wc_no_vol"]}
    noise["wc_book"] = np.sqrt(RHO_BOOK) * noise["wc_current"] + np.sqrt(1 - RHO_BOOK) * own["wc_book"]
    scores = {"wc_vol_rule": np.tile(vol_z, (periods, 1))}
    for arm, tilt in TILTS.items():
        scores[arm] = tilt * vol_z + np.sqrt(1 - tilt ** 2) * (Q * signal + np.sqrt(1 - Q ** 2) * noise[arm])

    fwd = pd.Series(ret.ravel(), index=index)
    vol_rank = pd.Series(np.tile(vol_z, periods), index=index)
    active, tilt = {}, {}
    for arm in ev.ARMS:
        pred = pd.Series(np.asarray(scores[arm]).ravel(), index=index)
        active[arm] = active_series_within_class(pred, fwd, groups, horizon=1)
        if arm in TILTS:
            tilt[arm] = within_class_tilt(pred, vol_rank, groups)
    return pd.DataFrame(active), tilt


def one_world(args):
    assets, classes, sigma, beta, seed = args
    active, tilt = simulate_world(assets, classes, sigma, beta, seed)
    res = ev.decide_formal(active, tilt)
    row = dict(beta=beta, seed=seed, ir_vol_rule=ev.ir(active["wc_vol_rule"]))
    for arm, r in res.items():
        for k in ("ir", "paired_low", "standalone_low", "vol_tilt", "adds_beyond_volatility", "positive_alone",
                  "edge_found"):
            row[f"{arm}_{k}"] = r[k]
    row["any_edge_found"] = any(r["edge_found"] for r in res.values())
    return row


def analyze(results):
    rows = []
    for beta, g in results.groupby("beta"):
        n = len(g)
        row = dict(beta=beta, n_worlds=n, mean_ir_current=g["wc_current_ir"].mean(),
                   mean_ir_vol_rule=g["ir_vol_rule"].mean(), mean_vol_tilt_current=g["wc_current_vol_tilt"].mean())
        for arm in ev.FITTED:
            for k in ("adds_beyond_volatility", "positive_alone", "edge_found"):
                row[f"{arm}_{k}"] = g[f"{arm}_{k}"].mean()
        p = g["any_edge_found"].mean()
        row["any_edge_found"] = p
        row["any_edge_found_se"] = float(np.sqrt(p * (1 - p) / n))
        rows.append(row)
    return pd.DataFrame(rows)


def results_path(folder):
    return Path(folder) / "noise_test_tier_a_results.csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", required=True)
    ap.add_argument("--n-worlds", type=int, default=N_WORLDS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--folder", default=str(HERE))
    args = ap.parse_args(argv)
    out = results_path(args.folder)
    if out.exists():
        print(f"REFUSED: {out.name} already exists; a registered run is done once")
        return 1
    assets, classes, sigma = universe_params(load_prices(args.prices))
    jobs = [(assets, classes, sigma, beta, BASE_SEED + i) for beta in BETAS for i in range(args.n_worlds)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(one_world, jobs, chunksize=20))
    else:
        rows = [one_world(j) for j in jobs]
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    summary = analyze(res)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.round(3).to_string(index=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# TDD RESULTS (sandbox py3.10 venv, 2026-10-17): test_shadow_evaluate.py 17 tests and test_noise_test_tier_a.py 12 tests,
# 29 passed together. Mutation checks: 23 mutants of the two modules (alpha, level, period count, resample count, tilt
# limit strict/ignored, or-for-and, wrong reference arm, last-instead-of-first periods, high end for low end, no
# short-sample refusal, ddof, hedge kept, signal leaked to the vol rule, beta ignored, tilt dropped, book not a cousin,
# any-edge as all, standard error, wrong tilt): all caught after two tests were added for the first survivors.
# Not yet run in the mlfinlab env. The registered run (Tier A) has not been run.
# ---------------------------------------------------------------------------
