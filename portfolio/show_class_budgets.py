"""Print the three class-budget rules on the latest price snapshot, and the resulting weights.

Read-only: writes nothing. Descriptive, no performance claims.
    python show_class_budgets.py [--prices prices_daily_asof_2026-10-08.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from class_budgets import equal_budgets, inverse_vol_budgets, non_predictive_weights, size_budgets
from etf_classes import CLASS_NAMES, CLASS_OF
from panel_data import drop_hedge, load_prices


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(here / "prices_daily_asof_2026-10-08.csv"))
    args = ap.parse_args()
    px = drop_hedge(load_prices(args.prices))
    tickers = [t for t in px.columns if t in CLASS_OF and px[t].iloc[-252:].notna().all()]
    rules = {"size": size_budgets(CLASS_OF, tickers), "equal": equal_budgets(CLASS_OF, tickers),
             "inverse_vol": inverse_vol_budgets(px[tickers], CLASS_OF)}
    print(f"Prices: {Path(args.prices).name}, last date {px.index[-1].date()}, {len(tickers)} ETFs")
    print("\nCLASS BUDGETS (share of equity)")
    print(pd.DataFrame(rules).reindex(CLASS_NAMES).round(3).to_string())
    w = pd.DataFrame({k: pd.Series(non_predictive_weights(CLASS_OF, v, tickers)) for k, v in rules.items()})
    print("\nLARGEST SINGLE-ETF WEIGHTS")
    for k in w:
        top = w[k].sort_values(ascending=False).head(3)
        print(f"  {k:12s} " + ", ".join(f"{s} {v:.3f}" for s, v in top.items()))


if __name__ == "__main__":
    main()
