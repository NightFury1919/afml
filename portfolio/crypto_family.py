"""Crypto cross-sectional edge test: three horizons, one pre-registered family.

Pre-registered in preregistration_crypto_family.md. The held-out run happens ONCE:
    python crypto_family.py --prices binance_prices_daily_asof_2026-10-10.csv \
                            --ohlcv  binance_ohlcv_daily_asof_2026-10-10.csv \
                            --kraken-prices   crypto_prices_daily_asof_2026-10-10.csv \
                            --kraken-ohlcv    crypto_ohlcv_daily_asof_2026-10-10.csv \
                            --kraken-manifest crypto_pairs_manifest_asof_2026-10-10.csv

WHAT IT TESTS
    Arm A  1-day forecast,  short features        judged on the held-out period (2024-10-01 onward)
    Arm B  5-day forecast,  short features        judged on the held-out period
    Arm C  21-day forecast, the five ETF features judged by purged k-fold over the FULL history
    Each arm is compared with a control: no model, score = the within-tier volatility rank.
    Arms A and B that find an edge get one more check, kraken_confirmed (pre-registration section 8, rule 3): the frozen
    Binance-fitted model is scored on the Kraken USD panel with no refit.

HOW A COIN GETS INTO THE TEST (point in time, using only data up to each date)
    * at least 260 observed days of history; trailing 60-day median quote volume at least $1,000,000;
      at most 3 missing days in the trailing 60; not on the exclusion list; at least 30 coins that day.
    * the eligible coins are split into 3 tiers by that same volume (terciles, recomputed daily).
      Every feature rank, label, score comparison and portfolio is built INSIDE a tier.
    Coins that later delisted are included while they traded. A label that runs past a delisting
    uses the coin's LAST price (you cannot trade after it), so a delisting is a loss, not a missing row.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from etf_features import (DEFAULT_WINDOWS, FEATURE_NAMES, lagged_log_return, raw_features, trailing_vol,
                          trend_distance)
from panel_data import wide_to_long
from panel_purged_cv import PanelPurgedKFold

# ---------------------------------------------------------------- frozen choices
MIN_OBS = 260                  # observed days needed (longest ETF window is 252)
MIN_DV = 1_000_000.0           # trailing median quote volume, USDT
KRAKEN_MIN_DV = 100_000.0      # same rule on the Kraken panel, trailing median dollar volume (USD); $1M leaves 17 dates
DV_WINDOW = 60
MAX_MISSING = 3                # missing days allowed in the trailing 60
MIN_ASSETS = 30                # coins needed on a date
N_TIERS = 3
MIN_TIER = 10                  # coins needed in a tier on a date
TOP_SHARE = 0.2
DEV_END = pd.Timestamp("2024-09-30")
HELD_START = pd.Timestamp("2024-10-01")
FAMILY_SIZE = 3
ALPHA = 0.05 / FAMILY_SIZE     # one-sided level for every rule
MIN_IC = 0.01
BOOT_BLOCK = 5
N_BOOT = 5000
BOOT_SEED = 20261010
# Cost per side, by volume tier (0 = lowest). Kraken's lowest tier (under $2.5K of 30-day volume) charges 0.40% maker
# and 0.80% taker (Kraken Pro spot fee table, Tier 1, $0+ of 30-day volume; confirmed by Ethan 2026-10-11). The half-spread by tier is a
# PLACEHOLDER (0.40%, 0.15%, 0.05%) until it is measured. PRIMARY assumes limit orders (maker fee); SENSITIVITY
# assumes every trade pays the taker fee. Only the primary schedule decides "tradable".
MAKER_FEE, TAKER_FEE = 0.0040, 0.0080
HALF_SPREAD = {0: 0.0040, 1: 0.0015, 2: 0.0005}
COST_PER_SIDE = {t: MAKER_FEE + sp for t, sp in HALF_SPREAD.items()}
COST_TAKER = {t: TAKER_FEE + sp for t, sp in HALF_SPREAD.items()}
EXCLUDE = [   # stablecoin / fiat bases, near-stable pairs, ticker reuse, wrapped / staked / gold copies
    "AEURUSDT", "AUDUSDT", "BUSDUSDT", "DAIUSDT", "EURIUSDT", "EURUSDT", "FDUSDUSDT", "GBPUSDT", "PAXUSDT",
    "RLUSDUSDT", "SUSDUSDT", "TUSDUSDT", "USD1USDT", "USDCUSDT", "USDEUSDT", "USDPUSDT", "USDSBUSDT",
    "USDSUSDT", "USTCUSDT", "USTUSDT", "XUSDUSDT",
    "BFUSDUSDT", "BKRWUSDT", "KGSTUSDT", "UUSDT",
    "LUNAUSDT",
    "WBTCUSDT", "WBETHUSDT", "BETHUSDT", "BNSOLUSDT", "PAXGUSDT", "XAUTUSDT"]

SHORT_FEATURES = ["ret_1d", "ret_5d", "mom_20_5", "vol_20", "trend_50"]


def short_raw_features(prices):
    """Arms A and B: the ETF design's five roles with short windows (frozen in the pre-registration)."""
    sigma = trailing_vol(prices, 20)

    def scale(x):
        return (x / sigma).replace([np.inf, -np.inf], np.nan)

    return {"ret_1d": scale(lagged_log_return(prices, 0, 1)),
            "ret_5d": scale(lagged_log_return(prices, 0, 5)),
            "mom_20_5": scale(lagged_log_return(prices, 5, 20)),
            "vol_20": sigma.replace([np.inf, -np.inf], np.nan),
            "trend_50": scale(trend_distance(prices, 50))}


ARMS = {
    "A": dict(horizon=1, features=SHORT_FEATURES, vol_feature="vol_20", raw=short_raw_features, scope="heldout"),
    "B": dict(horizon=5, features=SHORT_FEATURES, vol_feature="vol_20", raw=short_raw_features, scope="heldout"),
    "C": dict(horizon=21, features=FEATURE_NAMES, vol_feature="vol_60",
              raw=lambda p: raw_features(p, DEFAULT_WINDOWS), scope="full_cv"),
}


# ------------------------------------------------------------------- universe
def build_universe(prices, qvol, exclude=EXCLUDE, min_obs=MIN_OBS, min_dv=MIN_DV, dv_window=DV_WINDOW,
                   max_missing=MAX_MISSING, min_assets=MIN_ASSETS):
    """(eligible mask, trailing median volume), both wide. A coin is eligible on t using data up to t only."""
    keep = [c for c in prices.columns if c not in set(exclude)]
    px, qv = prices[keep], qvol.reindex(index=prices.index, columns=keep)
    has_price = px.notna()
    enough_history = has_price.cumsum() >= min_obs
    dv = qv.rolling(dv_window, min_periods=dv_window // 2).median()
    liquid = dv >= min_dv
    few_gaps = px.isna().rolling(dv_window, min_periods=dv_window).sum() <= max_missing
    elig = has_price & enough_history & liquid & few_gaps
    elig.loc[elig.sum(axis=1) < min_assets] = False
    return elig, dv.where(elig)


def assign_tiers(dv, n_tiers=N_TIERS):
    """Wide frame of tier numbers 0..n-1 (0 = lowest volume), NaN where not eligible. Terciles per date."""
    rank = dv.rank(axis=1, method="first")
    n = dv.notna().sum(axis=1)
    tier = np.floor((rank.sub(1, axis=0)).div(n, axis=0) * n_tiers)
    return tier.where(dv.notna())


# -------------------------------------------------------------------- dataset
def forward_log_returns(prices, horizon, delist_days=3):
    """log(price at t+h / price at t). A delisted coin's label that runs past its last price uses that last
    price. A coin that is merely missing the end date (a halt, or the end of the data) gets NaN."""
    logp = np.log(prices)
    fwd = (logp.shift(-horizon) - logp).copy()
    last_date = prices.index[-1]
    arr, idx = logp.to_numpy(), prices.index
    for j, c in enumerate(prices.columns):
        valid = np.flatnonzero(~np.isnan(arr[:, j]))
        if len(valid) == 0:
            continue
        last = valid[-1]
        if idx[last] >= last_date - pd.Timedelta(days=delist_days):
            continue                                   # still trading: leave NaN where t+h is not available
        t = np.arange(len(idx))
        past_end = (t + horizon > last) & (t <= last) & ~np.isnan(arr[:, j])
        fwd.iloc[past_end, j] = arr[last, j] - arr[past_end, j]
    return fwd


def within_tier_rank(values, tier):
    """(rank - 0.5) / n among the rows of the same (date, tier). Both are Series on a (date, asset) index."""
    key = [values.index.get_level_values("date"), tier.to_numpy()]
    g = values.groupby(key)
    return (g.rank(method="average") - 0.5) / g.transform("size")


def build_dataset(prices, eligible, tiers, cfg, min_tier=MIN_TIER):
    """Long frame on (date, asset): ranked features, tier, fwd_ret, excess_ret, label, t1."""
    raw = cfg["raw"](prices)
    mask = eligible.copy()
    for n in cfg["features"]:
        mask &= raw[n].notna()
    cols = {n: wide_to_long(raw[n].where(mask), n) for n in cfg["features"]}
    cols["tier"] = wide_to_long(tiers.where(mask), "tier")
    df = pd.concat(cols, axis=1).dropna(how="any")
    size = df.groupby([df.index.get_level_values("date"), "tier"])["tier"].transform("size")
    df = df[size >= min_tier]
    ranked = {n: within_tier_rank(df[n], df["tier"]) for n in cfg["features"]}
    out = pd.DataFrame(ranked)
    out["tier"] = df["tier"].astype(int)

    h = cfg["horizon"]
    fwd = wide_to_long(forward_log_returns(prices, h), "fwd_ret").reindex(out.index)
    out = out.join(fwd).dropna(subset=["fwd_ret"])
    key = [out.index.get_level_values("date"), out["tier"].to_numpy()]
    g = out.groupby(key)["fwd_ret"]
    keep = g.transform("size") >= min_tier
    out = out[keep]
    key = [out.index.get_level_values("date"), out["tier"].to_numpy()]
    out["excess_ret"] = out["fwd_ret"] - out.groupby(key)["fwd_ret"].transform("median")
    out["label"] = (out["excess_ret"] > 0).astype(int)
    all_dates = prices.index
    pos = all_dates.get_indexer(out.index.get_level_values("date"))
    out["t1"] = all_dates[np.minimum(pos + h, len(all_dates) - 1)]
    return out.sort_index()


# ----------------------------------------------------------------- the model
def fit_holdout_model(ds, features, dev_end=DEV_END, C=1.0):
    """Fit once on development rows whose labels resolved by dev_end. The frozen model."""
    date = ds.index.get_level_values("date")
    train = ds[(date <= dev_end) & (ds["t1"] <= dev_end)]
    return LogisticRegression(C=C, max_iter=200).fit(train[features].to_numpy(), train["label"].to_numpy())


def holdout_scores(ds, features, dev_end=DEV_END, held_start=HELD_START, C=1.0):
    """Fit once on development rows whose labels resolved by dev_end; score every row from held_start."""
    model = fit_holdout_model(ds, features, dev_end, C)
    date = ds.index.get_level_values("date")
    test = ds[date >= held_start]
    return pd.Series(model.predict_proba(test[features].to_numpy())[:, 1], index=test.index)


def oof_scores(ds, features, n_splits=5, embargo_dates=21, C=1.0):
    """Out-of-fold probability for every row (purged k-fold by date)."""
    X, y = ds[list(features)], ds["label"]
    cv = PanelPurgedKFold(n_splits=n_splits, t1=ds["t1"], embargo_dates=embargo_dates)
    pred = pd.Series(np.nan, index=ds.index)
    for train, test in cv.split(X):
        m = LogisticRegression(C=C, max_iter=200).fit(X.iloc[train].to_numpy(), y.iloc[train].to_numpy())
        pred.iloc[test] = m.predict_proba(X.iloc[test].to_numpy())[:, 1]
    return pred


# ------------------------------------------------------------------ statistics
def date_tier_ics(score, target, tier):
    """Spearman correlation of score and target inside each (date, tier). Series indexed by (date, tier)."""
    key = [score.index.get_level_values("date"), tier.to_numpy()]
    ra, rb = score.groupby(key).rank(), target.groupby(key).rank()
    ra, rb = ra - ra.groupby(key).transform("mean"), rb - rb.groupby(key).transform("mean")
    num = (ra * rb).groupby(key).sum()
    den = np.sqrt((ra ** 2).groupby(key).sum() * (rb ** 2).groupby(key).sum())
    out = (num / den).replace([np.inf, -np.inf], np.nan)
    out.index.names = ["date", "tier"]
    return out


def period_ic_series(score, target, tier, horizon):
    """Mean IC over tiers per date, then every horizon-th date (non-overlapping periods)."""
    return date_tier_ics(score, target, tier).groupby(level="date").mean().dropna().iloc[::horizon]


def long_weights(score, tier, top_share=TOP_SHARE):
    """Long-only weights summing to 1 per date: in each tier hold the top ceil(share*n) (at least 1) coins
    equally, tiers weighted by their number of coins. Series on (date, asset)."""
    date = score.index.get_level_values("date")
    key = [date, tier.to_numpy()]
    n = score.groupby(key).transform("size")
    k = np.maximum(np.ceil(top_share * n.to_numpy() - 1e-9), 1)
    rank = score.groupby(key).rank(ascending=False, method="first")
    pick = (rank.to_numpy() <= k).astype(float)
    total = score.groupby(date).transform("size")
    return pd.Series(pick * (n.to_numpy() / total.to_numpy()) / k, index=score.index)


def active_returns(score, fwd, tier, horizon, cost=COST_PER_SIDE, top_share=TOP_SHARE):
    """Per sampled date: gross active return (picks minus the universe mean), turnover, and net active return."""
    w = long_weights(score, tier, top_share)
    date = score.index.get_level_values("date")
    gross = (w * fwd).groupby(date).sum() - fwd.groupby(date).mean()
    dates = gross.index[::horizon]
    W = w.unstack("asset").reindex(dates).fillna(0.0)
    rate = tier.map(cost).unstack("asset").reindex(dates).ffill().fillna(cost[1])
    delta = W.diff().abs()
    delta.iloc[0] = W.iloc[0].abs()
    rate = rate.reindex(columns=W.columns).fillna(cost[1])
    costs = (delta * rate).sum(axis=1)
    out = pd.DataFrame({"gross": gross.reindex(dates), "turnover": delta.sum(axis=1), "cost": costs})
    out["net"] = out["gross"] - out["cost"]
    return out


def block_bootstrap_means(x, block=BOOT_BLOCK, n_boot=N_BOOT, seed=BOOT_SEED):
    """Means of moving-block bootstrap resamples of a series."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    nb = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    rows = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    return x[rows].mean(axis=1)


def lower_bound(x, alpha=ALPHA, **kw):
    """One-sided lower end: the alpha quantile of the block-bootstrap means."""
    return float(np.quantile(block_bootstrap_means(x, **kw), alpha))


def paired_lower_bound(a, b, alpha=ALPHA, **kw):
    """Same blocks for both series: lower end for mean(a - b)."""
    d = pd.concat([pd.Series(a), pd.Series(b)], axis=1, join="inner").dropna()
    return lower_bound((d.iloc[:, 0] - d.iloc[:, 1]).to_numpy(), alpha, **kw)


# --------------------------------------------------------------------- the run
def evaluate_arm(name, ds, n_boot=N_BOOT):
    """All numbers for one arm. Primary scope: held-out for A and B, full purged CV for C."""
    cfg = ARMS[name]
    h, feats = cfg["horizon"], cfg["features"]
    if cfg["scope"] == "heldout":
        model = holdout_scores(ds, feats)
    else:
        model = oof_scores(ds, feats)
    rows = ds.loc[model.index]
    control = rows[cfg["vol_feature"]]
    tier, fwd, exc = rows["tier"], rows["fwd_ret"], rows["excess_ret"]
    res = {"arm": name, "scope": cfg["scope"], "horizon": h}
    ic_m = period_ic_series(model, exc, tier, h)
    ic_c = period_ic_series(control, exc, tier, h)
    act_m = active_returns(model, fwd, tier, h)
    act_c = active_returns(control, fwd, tier, h)
    act_t = active_returns(model, fwd, tier, h, cost=COST_TAKER)
    kw = dict(n_boot=n_boot)
    res.update(n_periods=int(len(ic_m)), mean_ic=float(ic_m.mean()), ic_lower=lower_bound(ic_m, **kw),
               control_mean_ic=float(ic_c.mean()), ic_vs_control_lower=paired_lower_bound(ic_m, ic_c, **kw),
               gross_mean=float(act_m["gross"].mean()), net_mean=float(act_m["net"].mean()),
               net_lower=lower_bound(act_m["net"], **kw), mean_turnover=float(act_m["turnover"].iloc[1:].mean()),
               control_net_mean=float(act_c["net"].mean()),
               net_taker_mean=float(act_t["net"].mean()), net_taker_lower=lower_bound(act_t["net"], **kw))
    if cfg["scope"] == "full_cv":                       # descriptive: the same scores on the held-out window only
        d = ic_m.index >= HELD_START
        res["heldout_only_n_periods"] = int(d.sum())
        res["heldout_only_mean_ic"] = float(ic_m[d].mean()) if d.any() else float("nan")
    res["_series"] = dict(ic=ic_m, ic_control=ic_c, active=act_m, active_control=act_c)
    return res


def decide(res):
    """Pre-registered rules for one arm's evaluate_arm() result."""
    edge = bool(res["ic_lower"] > 0 and res["mean_ic"] >= MIN_IC and res["ic_vs_control_lower"] > 0)
    return {"edge_found": edge, "tradable": bool(edge and res["net_lower"] > 0)}


# ------------------------------------------------------- Kraken confirmation (rule 3)
def kraken_exclusions(manifest, exclude=EXCLUDE):
    """Kraken pairs to leave out: the manifest's stablecoin and fiat bases, plus the base assets of the Binance
    exclusion list (wrapped, staked, gold, near-stable, ticker reuse). `manifest` is a DataFrame or a CSV path."""
    m = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    bases = {s[:-len("USDT")] for s in exclude if s.endswith("USDT")}
    flagged = m["stable_or_fiat_base"].astype(str).str.lower() == "true"
    return sorted(m.loc[flagged | m["base"].isin(bases), "pair"])


def kraken_confirmed(res):
    """Rule 3: mean IC above 0 AND mean net active return above 0. Point estimates; intervals do not decide."""
    return bool(res["mean_ic"] > 0 and res["net_mean"] > 0)


def kraken_check(name, model, prices, qvol, exclude, min_dv=KRAKEN_MIN_DV, n_boot=N_BOOT, cost=COST_PER_SIDE):
    """Score the frozen Binance-fitted `model` on the Kraken panel (never refit). Same features, tiers, top-fifth
    portfolio and primary (maker) costs; the universe rules of section 3 with a lower volume floor."""
    cfg = ARMS[name]
    h = cfg["horizon"]
    elig, dv = build_universe(prices, qvol, exclude=exclude, min_dv=min_dv)
    ds = build_dataset(prices, elig, assign_tiers(dv), cfg)
    if ds.empty:
        nan = float("nan")
        return dict(n_periods=0, n_rows=0, first_date="", last_date="", n_coins_median=0, mean_ic=nan, ic_lower=nan,
                    gross_mean=nan, net_mean=nan, net_lower=nan, mean_turnover=nan, kraken_confirmed=False)
    score = pd.Series(model.predict_proba(ds[cfg["features"]].to_numpy())[:, 1], index=ds.index)
    ic = period_ic_series(score, ds["excess_ret"], ds["tier"], h)
    act = active_returns(score, ds["fwd_ret"], ds["tier"], h, cost=cost)
    kw = dict(n_boot=n_boot)
    date = ds.index.get_level_values("date")
    res = dict(n_periods=int(len(ic)), n_rows=int(len(ds)), first_date=str(date.min().date()),
               last_date=str(date.max().date()), n_coins_median=int(elig.sum(axis=1)[elig.any(axis=1)].median()),
               mean_ic=float(ic.mean()), ic_lower=lower_bound(ic, **kw), gross_mean=float(act["gross"].mean()),
               net_mean=float(act["net"].mean()), net_lower=lower_bound(act["net"], **kw),
               mean_turnover=float(act["turnover"].iloc[1:].mean()))
    res["kraken_confirmed"] = kraken_confirmed(res)
    return res


def check_not_run(folder, name="crypto_family_results.csv"):
    p = Path(folder) / name
    if p.exists():
        raise FileExistsError(f"{p.name} already exists. This check is run once; see the pre-registration. "
                              "Not overwriting.")
    return p


def load_panels(prices_path, ohlcv_path, symbol_col="symbol", volume_col="quote_volume"):
    """Binance files by default; the Kraken files use symbol_col="pair", volume_col="dollar_volume"."""
    p = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index().astype(float)
    if p.index.has_duplicates or (p.index.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("prices must have one row per calendar day, no duplicates or holes")
    if (p.fillna(1.0) <= 0).any().any():
        raise ValueError("non-positive prices")
    p.columns.name = "asset"
    long = pd.read_csv(ohlcv_path, usecols=["date", symbol_col, volume_col], parse_dates=["date"])
    q = long.pivot(index="date", columns=symbol_col, values=volume_col).reindex(index=p.index, columns=p.columns)
    return p, q


def run_family(prices, qvol, n_boot=N_BOOT, arms=("A", "B", "C"), kraken=None):
    """kraken = dict(prices=, qvol=, exclude=): arms A and B that find an edge also get the Kraken check (rule 3)."""
    elig, dv = build_universe(prices, qvol)
    tiers = assign_tiers(dv)
    out = {}
    for name in arms:
        ds = build_dataset(prices, elig, tiers, ARMS[name])
        res = evaluate_arm(name, ds, n_boot)
        res["rules"] = decide(res)
        res["rules"]["kraken_confirmed"] = None               # None = not run (only arms A and B with an edge are checked)
        if kraken is not None and name in ("A", "B") and res["rules"]["edge_found"]:
            model = fit_holdout_model(ds, ARMS[name]["features"])
            res["kraken"] = kraken_check(name, model, kraken["prices"], kraken["qvol"], kraken["exclude"], n_boot=n_boot)
            res["rules"]["kraken_confirmed"] = res["kraken"]["kraken_confirmed"]
        res["n_rows"] = int(len(ds))
        out[name] = res
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", required=True)
    ap.add_argument("--ohlcv", required=True)
    ap.add_argument("--kraken-prices", required=True)
    ap.add_argument("--kraken-ohlcv", required=True)
    ap.add_argument("--kraken-manifest", required=True)
    ap.add_argument("--folder", default=str(here))
    a = ap.parse_args()
    results_path = check_not_run(a.folder)
    prices, qvol = load_panels(a.prices, a.ohlcv)
    kp, kq = load_panels(a.kraken_prices, a.kraken_ohlcv, symbol_col="pair", volume_col="dollar_volume")
    kraken = dict(prices=kp, qvol=kq, exclude=kraken_exclusions(a.kraken_manifest))     # loaded first: a bad path fails now
    print(f"Prices: {prices.shape[0]} days x {prices.shape[1]} symbols; Kraken: {kp.shape[0]} days x {kp.shape[1]} pairs")
    res = run_family(prices, qvol, kraken=kraken)
    rows = []
    for name, r in res.items():
        print(f"\nARM {name} ({r['scope']}, horizon {r['horizon']}d, {r['n_periods']} periods, {r['n_rows']:,} rows)")
        for k, v in r.items():
            if k not in ("_series", "rules", "arm", "scope", "horizon", "kraken"):
                print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
        for k, v in r.get("kraken", {}).items():
            print(f"  kraken_{k:17s} {v:.4f}" if isinstance(v, float) else f"  kraken_{k:17s} {v}")
        print("  RULES:", r["rules"])
        rows += [dict(arm=name, metric=k, value=v) for k, v in r.items() if k not in ("_series", "rules", "kraken")]
        rows += [dict(arm=name, metric=f"kraken_{k}", value=v) for k, v in r.get("kraken", {}).items()]
        rows += [dict(arm=name, metric=f"rule_{k}", value=v) for k, v in r["rules"].items()]
        r["_series"]["ic"].to_csv(Path(a.folder) / f"crypto_family_{name}_ic.csv", header=["ic"], index_label="date")
        r["_series"]["active"].to_csv(Path(a.folder) / f"crypto_family_{name}_active.csv", index_label="date")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-11, mlfinlab env on Ethan's machine: Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2). 41 tests: the original 29 plus 12 for the Kraken confirmation (pre-registration section 8, rule 3).
# The same 41 passed in the sandbox on the same versions before this run.
# $ conda activate mlfinlab ; cd C:\ws\AFML\portfolio ; pytest test_crypto_family.py -v --color=no
# ============================= test session starts =============================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# cachedir: .pytest_cache
# rootdir: C:\ws\AFML\portfolio
# collecting ... collected 41 items
#
# test_crypto_family.py::test_universe_applies_history_liquidity_gap_and_exclusion_rules PASSED [  2%]
# test_crypto_family.py::test_a_date_with_too_few_coins_is_removed_entirely PASSED [  4%]
# test_crypto_family.py::test_universe_uses_only_past_data PASSED          [  7%]
# test_crypto_family.py::test_tiers_are_terciles_by_volume_with_zero_the_lowest PASSED [  9%]
# test_crypto_family.py::test_tiers_leave_ineligible_coins_blank PASSED    [ 12%]
# test_crypto_family.py::test_forward_returns_known_values_and_delisting_rule PASSED [ 14%]
# test_crypto_family.py::test_within_tier_rank_known_values PASSED         [ 17%]
# test_crypto_family.py::test_dataset_labels_are_tier_median_excess_and_t1_is_h_days_later PASSED [ 19%]
# test_crypto_family.py::test_dataset_drops_a_tier_that_is_too_small PASSED [ 21%]
# test_crypto_family.py::test_ic_is_plus_one_for_a_perfect_score_and_minus_one_for_its_reverse PASSED [ 24%]
# test_crypto_family.py::test_ic_is_computed_inside_tiers_not_across_them PASSED [ 26%]
# test_crypto_family.py::test_ic_is_a_rank_correlation_inside_each_tier_even_when_tiers_interleave PASSED [ 29%]
# test_crypto_family.py::test_long_weights_pick_the_top_fifth_of_each_tier_and_sum_to_one PASSED [ 31%]
# test_crypto_family.py::test_a_tier_of_six_picks_two_and_float_noise_does_not_add_a_pick PASSED [ 34%]
# test_crypto_family.py::test_active_return_and_costs_known_values PASSED  [ 36%]
# test_crypto_family.py::test_cost_schedules_follow_kraken_fees_and_the_taker_sensitivity_is_dearer PASSED [ 39%]
# test_crypto_family.py::test_taker_costs_lower_net_returns_by_the_known_amount PASSED [ 41%]
# test_crypto_family.py::test_periods_are_spaced_one_horizon_apart PASSED  [ 43%]
# test_crypto_family.py::test_bootstrap_lower_bound_is_positive_for_a_steady_edge_and_negative_for_noise PASSED [ 46%]
# test_crypto_family.py::test_paired_lower_bound_is_zero_for_identical_series_and_positive_for_a_clear_gap PASSED [ 48%]
# test_crypto_family.py::test_bootstrap_is_reproducible_and_handles_a_block_longer_than_the_series PASSED [ 51%]
# test_crypto_family.py::test_an_edge_needs_all_three_conditions_and_tradable_needs_net_returns PASSED [ 53%]
# test_crypto_family.py::test_alpha_is_the_bonferroni_level_for_three_arms PASSED [ 56%]
# test_crypto_family.py::test_holdout_fit_excludes_labels_that_resolve_after_the_cutoff PASSED [ 58%]
# test_crypto_family.py::test_oof_scores_cover_every_row_and_find_a_planted_relation PASSED [ 60%]
# test_crypto_family.py::test_no_signal_gives_no_edge_in_any_arm PASSED    [ 63%]
# test_crypto_family.py::test_a_planted_one_day_signal_is_found_by_arm_a_but_does_not_pass_the_volatility_control_by_luck PASSED [ 65%]
# test_crypto_family.py::test_check_not_run_refuses_to_overwrite PASSED    [ 68%]
# test_crypto_family.py::test_load_panels_rejects_a_date_hole PASSED       [ 70%]
# test_crypto_family.py::test_the_kraken_floor_is_100k_and_the_binance_floor_stays_1m PASSED [ 73%]
# test_crypto_family.py::test_kraken_exclusions_are_the_flagged_stablecoins_plus_the_binance_exclusion_bases PASSED [ 75%]
# test_crypto_family.py::test_load_panels_reads_the_kraken_column_names PASSED [ 78%]
# test_crypto_family.py::test_kraken_confirmed_needs_a_positive_ic_and_a_positive_net_return PASSED [ 80%]
# test_crypto_family.py::test_the_frozen_model_gives_the_same_scores_as_holdout_scores PASSED [ 82%]
# test_crypto_family.py::test_a_signal_that_carries_over_is_confirmed_and_costs_can_still_sink_it PASSED [ 85%]
# test_crypto_family.py::test_a_signal_that_reverses_on_kraken_is_not_confirmed PASSED [ 87%]
# test_crypto_family.py::test_kraken_check_leaves_out_excluded_pairs_and_coins_below_the_floor PASSED [ 90%]
# test_crypto_family.py::test_kraken_check_with_too_few_coins_is_not_confirmed_and_keeps_the_same_keys PASSED [ 92%]
# test_crypto_family.py::test_run_family_checks_kraken_only_for_arms_a_and_b_that_found_an_edge PASSED [ 95%]
# test_crypto_family.py::test_the_command_line_run_writes_results_with_the_kraken_rows_and_refuses_a_second_run PASSED [ 97%]
# test_crypto_family.py::test_the_command_line_run_needs_the_kraken_files PASSED [100%]
#
# ============================= 41 passed in 10.56s =============================
#
# Mutation checks (sandbox), original 9: delisting rule removed, ceil guard removed, universe window peeking ahead, purge at
#   the cutoff removed, free first buy, IC pooled across tiers, Bonferroni removed, upper bound used, and the taker
#   sensitivity wired to the maker schedule each made a test fail; original restored. One mutation is EQUIVALENT, not a
#   gap: dropping the 'feature must exist' mask changes nothing because rows with a missing feature are dropped again
#   when the frames are joined.
# Mutation checks (sandbox), Kraken confirmation, 17 of 17 caught: IC rule '>=' ; OR instead of AND ; net return ignored ;
#   Kraken floor 1M ; floor ignored ; exclusions ignored ; stablecoin flag ignored ; Binance exclusion bases ignored ;
#   base matched by substring ; model refit on Kraken data ; costs ignored ; Kraken step run for arm C ; run with no
#   edge ; 'not run' recorded as False ; Binance column names used for Kraken ; purge removed from the frozen fit ;
#   empty Kraken result marked confirmed.
# Real-data check on the Kraken files (counts only, no returns): 633 pairs, 19 excluded, 461 dates with a universe
#   (2025-07-06 to 2026-10-09), 65 to 124 coins a date (median 84), at least 21 per tier, arm A 460 periods, arm B 92.
# Full-size synthetic run of the Binance part (700 coins x 3,332 days, no edge): about 52 seconds, 1.2 GB; no arm found an edge.
# NOT yet run on the real Binance panel. Pre-registered in preregistration_crypto_family.md (commits a170d2e, b42b39a);
#   the run is once only.
# ---------------------------------------------------------------------------
