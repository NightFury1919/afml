"""EXPLORATORY post-outcome audit (not pre-registered). Run from portfolio\ in the mlfinlab env."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crypto_family as cf
U = str(Path(__file__).resolve().parent) + "/"
prices, qvol = cf.load_panels(U+"binance_prices_daily_asof_2026-10-10.csv", U+"binance_ohlcv_daily_asof_2026-10-10.csv")
elig, dv = cf.build_universe(prices, qvol); tiers = cf.assign_tiers(dv)
cfg = cf.ARMS["C"]; h = 21
ds = cf.build_dataset(prices, elig, tiers, cfg)
model = cf.oof_scores(ds, cfg["features"]); rows = ds.loc[model.dropna().index]; model = model.loc[rows.index]
tier, fwd, exc = rows["tier"], rows["fwd_ret"], rows["excess_ret"]; lowvol = -rows[cfg["vol_feature"]]
ic_m = cf.period_ic_series(model, exc, tier, h); ic_l = cf.period_ic_series(lowvol, exc, tier, h)
am = cf.active_returns(model, fwd, tier, h); al = cf.active_returns(lowvol, fwd, tier, h)
print(f"1. IC model {ic_m.mean():.4f} | low-vol {ic_l.mean():.4f} | diff {(ic_m-ic_l).mean():.4f} lower {cf.paired_lower_bound(ic_m, ic_l, n_boot=2000):.4f}")
print(f"   net model {am['net'].mean():.5f} | low-vol {al['net'].mean():.5f}; gross {am['gross'].mean():.5f} | {al['gross'].mean():.5f}")
simple = np.expm1(fwd); gs = cf.active_returns(model, simple, tier, h)
print(f"2. gross log {am['gross'].mean():.5f} simple {gs['gross'].mean():.5f}; net simple {gs['net'].mean():.5f}; lower(net simple) {cf.lower_bound(gs['net'], n_boot=2000):.5f}")
w = cf.long_weights(model, tier); date = rows.index.get_level_values('date')
book = (w*simple).groupby(date).sum(); univ = simple.groupby(date).mean(); d = book.index[::h]
bot = (cf.long_weights(-model, tier)*simple).groupby(date).sum()
print(f"3. book {book.loc[d].mean():.4f} (med {book.loc[d].median():.4f}) univ {univ.loc[d].mean():.4f} (med {univ.loc[d].median():.4f})")
print(f"4. top-univ {(book-univ).loc[d].mean():.4f} univ-bottom {(univ-bot).loc[d].mean():.4f}")
t = ic_m.to_frame('ic'); t['year']=t.index.year; print(t.groupby('year')['ic'].agg(['mean','count']).round(3).to_string())
tt = cf.date_tier_ics(model, exc, tier).reset_index(name='ic'); print(tt.groupby('tier')['ic'].mean().round(4).to_string())
