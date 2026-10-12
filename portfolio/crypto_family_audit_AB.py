"""EXPLORATORY post-outcome audit (not pre-registered). Run from portfolio\ in the mlfinlab env."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_family as cf
U = str(Path(__file__).resolve().parent) + "/"
prices, qvol = cf.load_panels(U+"binance_prices_daily_asof_2026-10-10.csv", U+"binance_ohlcv_daily_asof_2026-10-10.csv")
elig, dv = cf.build_universe(prices, qvol); tiers = cf.assign_tiers(dv)
def lb(x): return cf.lower_bound(x, n_boot=2000)
out = []
for name in ("A", "B"):
    cfg = cf.ARMS[name]; h = cfg["horizon"]
    ds = cf.build_dataset(prices, elig, tiers, cfg)
    model = cf.holdout_scores(ds, cfg["features"])
    rows = ds.loc[model.index]
    tier, fwd, exc = rows["tier"], rows["fwd_ret"], rows["excess_ret"]
    vol = rows[cfg["vol_feature"]]
    lowvol = -vol
    print(f"\n===== ARM {name} (exploratory) rows={len(rows)}")
    # 1 sign-matched control
    ic_m = cf.period_ic_series(model, exc, tier, h); ic_l = cf.period_ic_series(lowvol, exc, tier, h)
    am = cf.active_returns(model, fwd, tier, h); al = cf.active_returns(lowvol, fwd, tier, h)
    print(f"1. IC model {ic_m.mean():.4f} | low-vol sort IC {ic_l.mean():.4f} | diff {(ic_m-ic_l).mean():.4f} lower95(one-sided a=.0167) {cf.paired_lower_bound(ic_m, ic_l, n_boot=2000):.4f}")
    print(f"   net/period model {am['net'].mean():.5f} | low-vol {al['net'].mean():.5f} ; gross model {am['gross'].mean():.5f} | low-vol {al['gross'].mean():.5f}")
    # spearman of model score vs lowvol within date-tier
    rm = model.groupby([model.index.get_level_values('date'), tier.to_numpy()]).rank()
    rl = lowvol.groupby([lowvol.index.get_level_values('date'), tier.to_numpy()]).rank()
    cc = pd.concat([rm.rename('m'), rl.rename('l')], axis=1)
    print(f"   rank corr model score vs low-vol (pooled within date-tier ranks): {cc.corr().iloc[0,1]:.3f}")
    # 2 simple vs log
    simple = np.expm1(fwd)
    gs = cf.active_returns(model, simple, tier, h)
    print(f"2. gross active, log fwd {am['gross'].mean():.5f} vs simple fwd {gs['gross'].mean():.5f}; net simple {gs['net'].mean():.5f}")
    # 3 absolute return of long-only book vs universe
    w = cf.long_weights(model, tier); date = rows.index.get_level_values('date')
    book = (w*simple).groupby(date).sum(); univ = simple.groupby(date).mean()
    d = book.index[::h]
    print(f"3. mean per-period simple return: book {book.loc[d].mean():.5f} (median {book.loc[d].median():.5f}) universe {univ.loc[d].mean():.5f} (median {univ.loc[d].median():.5f})")
    # 4 long vs short side
    k = lambda s: cf.long_weights(s, tier)
    wl = k(-model)  # bottom-ranked book
    bot = (wl*simple).groupby(date).sum()
    print(f"4. top-book minus universe {(book-univ).loc[d].mean():.5f} | universe minus bottom-book {(univ-bot).loc[d].mean():.5f}  (long-only trade captures only the first)")
    # 5 IC by year and tier
    t_ic = cf.date_tier_ics(model, exc, tier).reset_index(name='ic')
    t_ic['year'] = t_ic['date'].dt.year
    print("5. mean daily IC by year:"); print(t_ic.groupby('year')['ic'].mean().round(4).to_string())
    print("   by tier:"); print(t_ic.groupby('tier')['ic'].mean().round(4).to_string())
    cl = cf.date_tier_ics(lowvol, exc, tier).reset_index(name='ic'); cl['year']=cl['date'].dt.year
    print("   low-vol sort IC by year:"); print(cl.groupby('year')['ic'].mean().round(4).to_string())
