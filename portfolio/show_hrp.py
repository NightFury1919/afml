"""Compare HRP class shares with the other class-budget rules on the latest price snapshot.

Read-only: writes nothing. Descriptive, no performance claims.
    python show_hrp.py [--prices prices_daily_asof_2026-10-08.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from class_budgets import equal_budgets, inverse_vol_budgets, size_budgets
from etf_classes import CLASS_NAMES, CLASS_OF
from hrp_allocation import DEFAULT_WINDOW, hrp_class_shares, hrp_weights, trailing_returns
from panel_data import drop_hedge, load_prices


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", default=str(here / "prices_daily_asof_2026-10-08.csv"))
    args = ap.parse_args()
    px = drop_hedge(load_prices(args.prices))
    tickers = [t for t in px.columns if t in CLASS_OF and px[t].iloc[-(DEFAULT_WINDOW + 1):].notna().all()]
    w = hrp_weights(trailing_returns(px, tickers))
    shares = {"size": size_budgets(CLASS_OF, tickers), "equal": equal_budgets(CLASS_OF, tickers),
              "inverse_vol": inverse_vol_budgets(px[tickers], CLASS_OF), "hrp": hrp_class_shares(w, CLASS_OF)}
    print(f"Prices: {Path(args.prices).name}, last date {px.index[-1].date()}, {len(tickers)} ETFs, window {DEFAULT_WINDOW} days")
    print("\nCLASS SHARES OF EQUITY")
    print(pd.DataFrame(shares).reindex(CLASS_NAMES).round(3).to_string())
    print("\nHRP: 5 LARGEST AND 5 SMALLEST ETF WEIGHTS")
    s = w.sort_values(ascending=False)
    print("  largest  " + ", ".join(f"{k} {v:.3f}" for k, v in s.head(5).items()))
    print("  smallest " + ", ".join(f"{k} {v:.4f}" for k, v in s.tail(5).items()))
    print(f"  sum {w.sum():.6f}, min {w.min():.4f}, max {w.max():.4f}")


if __name__ == "__main__":
    main()
