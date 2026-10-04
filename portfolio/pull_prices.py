"""Pull daily adjusted close prices for the ETF universe.

Research use only: yfinance reads Yahoo Finance's unofficial endpoints, which can
change or rate-limit without notice. Do not depend on it for live trading.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python pull_prices.py
    python pull_prices.py --universe universe.txt --out prices_daily.csv

Outputs:
    prices_daily.csv    one row per date, one column per ticker (adjusted close)
    prices_summary.csv  per ticker: first_date, last_date, n_obs, max_gap_days
"""
import argparse
from pathlib import Path

import pandas as pd


def load_tickers(path):
    """Read tickers, one per line.

    Blank lines and '#' comments are ignored, duplicates are dropped (first
    occurrence kept), and everything is upper-cased.
    """
    tickers, seen = [], set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        t = line.split("#", 1)[0].strip().upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)
    return tickers


def summarize_prices(prices):
    """One summary row per ticker column.

    max_gap_days is the longest calendar-day gap between consecutive
    observations. A weekend gives 3 and a long holiday weekend gives 4, so a
    value above about 5 deserves a look.
    """
    rows = []
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            rows.append({"ticker": t, "first_date": "", "last_date": "",
                         "n_obs": 0, "max_gap_days": 0})
            continue
        gaps = s.index.to_series().diff().dt.days
        rows.append({
            "ticker": t,
            "first_date": s.index[0].date().isoformat(),
            "last_date": s.index[-1].date().isoformat(),
            "n_obs": int(len(s)),
            "max_gap_days": int(gaps.max()) if len(s) > 1 else 0,
        })
    return pd.DataFrame(rows)


def pull_prices(tickers):
    """Download full-history adjusted closes, one ticker at a time."""
    import yfinance as yf  # imported here so the tests do not need yfinance

    series = {}
    for t in tickers:
        hist = yf.Ticker(t).history(period="max", auto_adjust=True)
        if hist.empty:
            print(f"WARNING: no data returned for {t}")
            continue
        s = hist["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        series[t] = s
    return pd.DataFrame(series).sort_index()


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--universe", default=str(here / "universe.txt"))
    ap.add_argument("--out", default=str(here / "prices_daily.csv"))
    args = ap.parse_args()

    tickers = load_tickers(args.universe)
    print(f"Pulling {len(tickers)} tickers: {', '.join(tickers)}")
    prices = pull_prices(tickers)

    out = Path(args.out)
    prices.to_csv(out, index_label="date")
    summary = summarize_prices(prices)
    summary_path = out.with_name("prices_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False))
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"No data for: {', '.join(missing)}")
    have = summary[summary["n_obs"] > 0]
    if not have.empty:
        youngest = have.sort_values("first_date").iloc[-1]
        print(f"Shortest history: {youngest['ticker']} starts {youngest['first_date']}"
              f" (limits any pooled history)")
    print(f"Saved {out} and {summary_path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-03, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_pull_prices.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 3 items
#
# test_pull_prices.py::test_load_tickers_ignores_blanks_comments_and_duplicates PASSED [ 33%]
# test_pull_prices.py::test_summarize_known_values PASSED                              [ 66%]
# test_pull_prices.py::test_summarize_column_with_no_data PASSED                       [100%]
#
# 3 passed in 1.33s
# ---------------------------------------------------------------------------
