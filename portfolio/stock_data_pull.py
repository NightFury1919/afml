"""Pull free daily prices for every stock that was ever an S&P 500 member, and report how complete they are.

WHAT THIS DOES (in order)
    1. Reads the S&P 500 membership history (hanshof/sp500_constituents, one row per change date, tickers comma-separated).
    2. Makes a clean list of every ticker that ever appears (class shares like BRK.B become BRK-B for Yahoo).
    3. Downloads adjusted closes from Yahoo Finance in batches, saving each batch to disk so a stop or a rate limit
       can be resumed by simply running the same command again.
    4. Writes ONE price file, a per-ticker report, a coverage-by-year report and a list of suspicious one-day moves.

WHAT THIS DOES NOT DO
    It does not run any strategy and looks at no returns of a strategy. Everything here is about the DATA only, so it is
    safe to run before the pre-registration is frozen.

KNOWN FLAW (read this): free sources often lack prices for stocks that left the index (bankrupt, acquired). A missing
price means the stock cannot be held by a test, which biases any result in favor of survivors. The coverage report
measures exactly how big that gap is. Also, a ticker can be reused by a different company later; Yahoo then returns the
NEW company's prices under the old symbol. This script cannot detect that.

RUN (Windows PowerShell, conda env mlfinlab, from C:\\ws\\AFML\\portfolio):
    python stock_data_pull.py --limit 20          # small trial first: 20 tickers, checks Yahoo works
    python stock_data_pull.py                     # the full pull (about 1,100 tickers; expect 10 to 30 minutes)
    # if it stops or is rate-limited, run the same command again: finished batches are skipped.
"""
import argparse
import datetime as dt
import os
import re
import sys
import time

import numpy as np
import pandas as pd

MEMBERSHIP_URL = ("https://raw.githubusercontent.com/hanshof/sp500_constituents/main/"
                  "sp_500_historical_components.csv")
DEFAULT_START = "1996-01-01"
FLAG_MOVE = 0.60          # one-day move beyond +/-60% is listed for review (a data-error check, not a strategy rule)
COVERAGE_TARGET = 0.90    # share of members that must have prices for a date to count as well covered
BENCHMARK = "SPY"


# ---------------------------------------------------------------- pure functions (these are unit-tested)
def normalize_ticker(t):
    """Yahoo spelling: drop a note like 'RVTY (Previously PKI)' to 'RVTY', and class-share dots to dashes (BRK.B -> BRK-B)."""
    t = str(t).strip().upper()
    t = t.split(" ")[0]
    return t.replace(".", "-")


def parse_membership(df):
    """Frame with columns date and tickers (comma-separated) -> dict-like Series: date -> frozenset of normalized tickers.

    The index is sorted ascending; each row is the full membership from that date until the next row's date.
    If a date appears twice (the real file has two such days), the LAST row for that date is kept: it is the end-of-day state.
    """
    out = {}
    for d, tick in zip(pd.to_datetime(df["date"]), df["tickers"]):
        out[d] = frozenset(normalize_ticker(x) for x in str(tick).split(",") if str(x).strip())
    s = pd.Series(out)
    return s.sort_index()


def all_tickers(membership):
    """Sorted list of every ticker that ever appears."""
    s = set()
    for m in membership.values:
        s |= set(m)
    return sorted(s)


def members_on(membership, date):
    """Membership in force on `date`: the last row dated on or before it. Empty set before the first row."""
    date = pd.Timestamp(date)
    pos = membership.index.searchsorted(date, side="right") - 1
    if pos < 0:
        return frozenset()
    return membership.iloc[pos]


def month_end_dates(index):
    """The last trading date of each calendar month in a DatetimeIndex."""
    idx = pd.DatetimeIndex(index).sort_values()
    s = pd.Series(idx, index=idx.to_period("M"))
    return pd.DatetimeIndex(s.groupby(level=0).max().values)


def coverage_table(prices, membership, dates=None):
    """For each date: members that day, members with a price that day, and the share.

    prices: wide frame (dates x tickers) of adjusted closes, NaN where missing.
    """
    if dates is None:
        dates = month_end_dates(prices.index)
    last_known = membership.index.max()
    rows = []
    for d in dates:
        if d > last_known:      # the membership file says nothing past its last row
            continue
        mem = members_on(membership, d)
        if not mem:
            continue
        have = prices.loc[d]
        have = set(have.index[have.notna().to_numpy()])
        n_with = len(mem & have)
        rows.append({"date": d, "members": len(mem), "with_price": n_with, "share": n_with / len(mem)})
    return pd.DataFrame(rows).set_index("date")


def coverage_by_year(cov, target=COVERAGE_TARGET):
    """Per calendar year: average share and the fraction of month-ends at or above the target."""
    g = cov.groupby(cov.index.year)
    out = pd.DataFrame({
        "month_ends": g["share"].size(),
        "avg_share": g["share"].mean(),
        "share_of_months_at_target": g["share"].apply(lambda x: float((x >= target).mean())),
    })
    out.index.name = "year"
    return out


def flag_big_moves(prices, threshold=FLAG_MOVE):
    """List every stock-day whose one-day move is beyond +/-threshold (a check for data errors such as bad splits)."""
    rets = prices.pct_change(fill_method=None)
    rows = []
    for tick in rets.columns:
        r = rets[tick]
        bad = r[(r.abs() > threshold)]
        for d, v in bad.items():
            rows.append({"ticker": tick, "date": d, "move": float(v)})
    return pd.DataFrame(rows, columns=["ticker", "date", "move"])


def ticker_report(prices, membership):
    """Per ticker: first/last price date, number of price rows, first/last date it was a member."""
    first_mem, last_mem = {}, {}
    for d, m in membership.items():
        for t in m:
            first_mem.setdefault(t, d)
            last_mem[t] = d
    rows = []
    for t in all_tickers(membership):
        if t in prices.columns:
            col = prices[t].dropna()
        else:
            col = pd.Series(dtype=float)
        rows.append({
            "ticker": t,
            "price_rows": int(len(col)),
            "first_price": col.index.min() if len(col) else pd.NaT,
            "last_price": col.index.max() if len(col) else pd.NaT,
            "first_member_row": first_mem[t],
            "last_member_row": last_mem[t],
            "status": "ok" if len(col) else "NO PRICES",
        })
    return pd.DataFrame(rows).set_index("ticker")


def batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------- network part (needs Yahoo; not unit-tested)
def _adj_close_frame(raw, tickers):
    """Pull the 'Adj Close' block out of a yfinance result in either of its layouts."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        lvl0 = list(cols.get_level_values(0).unique())
        if "Adj Close" in lvl0:
            out = raw["Adj Close"]
        elif "Close" in lvl0:
            out = raw["Close"]
        else:
            return pd.DataFrame()
    else:  # single ticker, flat columns
        col = "Adj Close" if "Adj Close" in cols else "Close"
        out = raw[[col]].rename(columns={col: tickers[0]})
    if isinstance(out, pd.Series):
        out = out.to_frame(tickers[0])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def fetch_batch(tickers, start, end, tries=3, sleep=5.0):
    import yfinance as yf
    last = None
    for k in range(tries):
        try:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False,
                              threads=True, group_by="column")
            return _adj_close_frame(raw, tickers)
        except Exception as e:  # rate limit, network blip
            last = e
            time.sleep(sleep * (k + 1))
    print("  batch failed after retries: %r" % (last,))
    return None


def load_membership(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print("downloading membership file ...")
        # requests (not urllib) because it uses its own certificate list; Python's urllib reads the Windows
        # certificate store, and a damaged entry there raises "ssl.SSLError: [ASN1: NOT_ENOUGH_DATA]".
        import requests
        resp = requests.get(MEMBERSHIP_URL, timeout=60)
        resp.raise_for_status()
        raw = resp.content
        if path:
            with open(path, "wb") as f:
                f.write(raw)
        import io
        df = pd.read_csv(io.BytesIO(raw))
    return parse_membership(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--membership", default="sp_500_historical_components.csv", help="membership CSV (downloaded if missing)")
    ap.add_argument("--out-dir", default="stock_data", help="folder for the cache and the outputs")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None, help="last date, default today")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=3.0, help="pause between batches (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tickers (a trial run)")
    a = ap.parse_args(argv)

    end = a.end or dt.date.today().isoformat()
    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, "cache_batch%d%s" % (a.batch, "_limit%d" % a.limit if a.limit else ""))
    os.makedirs(cache, exist_ok=True)

    mem = load_membership(a.membership)
    tickers = all_tickers(mem)
    print("membership rows %d, %s to %s, %d distinct tickers"
          % (len(mem), mem.index.min().date(), mem.index.max().date(), len(tickers)))
    tickers = sorted(set(tickers) | {BENCHMARK})
    if a.limit:
        tickers = tickers[: a.limit]
        print("TRIAL RUN: first %d tickers only" % a.limit)

    frames = []
    for i, b in enumerate(batches(tickers, a.batch)):
        f = os.path.join(cache, "batch_%04d.csv" % i)
        if os.path.exists(f):
            frames.append(pd.read_csv(f, index_col=0, parse_dates=True))
            print("batch %d: cached" % i)
            continue
        print("batch %d: %d tickers (%s ... %s)" % (i, len(b), b[0], b[-1]))
        df = fetch_batch(b, a.start, end)
        if df is None:
            print("  skipped; run again later to retry this batch")
            continue
        df.to_csv(f)
        frames.append(df)
        time.sleep(a.sleep)

    if not frames:
        print("no data came back. Nothing written.")
        return 1
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    prices = prices.dropna(axis=1, how="all")

    stamp = prices.index.max().date().isoformat()
    f_prices = os.path.join(a.out_dir, "prices_stocks_daily_asof_%s.csv" % stamp)
    prices.to_csv(f_prices)
    rep = ticker_report(prices, mem)
    rep.to_csv(os.path.join(a.out_dir, "stock_ticker_report.csv"))
    cov = coverage_table(prices, mem)
    cby = coverage_by_year(cov)
    cby.to_csv(os.path.join(a.out_dir, "stock_coverage_by_year.csv"))
    flags = flag_big_moves(prices)
    flags.to_csv(os.path.join(a.out_dir, "stock_flagged_moves.csv"), index=False)

    pd.set_option("display.width", 140)
    print("\nprices: %d dates x %d tickers, %s to %s" % (prices.shape[0], prices.shape[1],
                                                         prices.index.min().date(), prices.index.max().date()))
    print("tickers with NO prices: %d of %d" % ((rep["status"] == "NO PRICES").sum(), len(rep)))
    print("\ncoverage by year (share of that month-end's S&P 500 members that have a price):")
    print(cby.round(3).to_string())
    print("\nflagged one-day moves beyond +/-%.0f%%: %d (see stock_flagged_moves.csv)" % (FLAG_MOVE * 100, len(flags)))
    print("\nmembership file ends %s; dates after that have no membership data." % mem.index.max().date())
    print("written to %s: %s" % (a.out_dir, os.path.basename(f_prices)))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------------------------------------
# TEST RESULTS (embedded 2026-11-07)
#
# Windows, conda env mlfinlab (Python 3.10.20, pandas 1.5.3, pytest 9.0.3), from C:\ws\AFML\portfolio:
#   $ python -m pytest test_stock_data_pull.py -v
#   16 passed in 5.25s
#     test_normalize_ticker_dots_notes_and_case, test_parse_membership_sets_and_sorted_index,
#     test_parse_membership_keeps_last_row_when_a_date_repeats, test_all_tickers_is_sorted_union,
#     test_members_on_uses_last_row_on_or_before_and_empty_before_first,
#     test_month_end_dates_picks_last_trading_day_of_each_month, test_coverage_table_known_shares,
#     test_coverage_ignores_dates_after_the_last_membership_row, test_coverage_table_ignores_prices_for_non_members,
#     test_coverage_by_year_average_and_target_fraction, test_flag_big_moves_both_directions_and_threshold_is_strict,
#     test_flag_big_moves_does_not_bridge_missing_days, test_ticker_report_marks_missing_and_dates,
#     test_batches_split_sizes, test_adj_close_frame_handles_multiindex_and_flat_layouts,
#     test_real_membership_file_parses_if_present   (all PASSED)
#
# Mutation check (sandbox, 5 deliberate bugs, each made at least one test fail; original restored):
#   membership as-of lookup using side="left"; coverage not stopping at the last membership row;
#   flag threshold >= instead of >; taking 'Close' instead of 'Adj Close'; not stripping notes like "(Previously PKI)".
#
# Real-data runs (2026-11-07, Windows): the 20-ticker trial worked (8 of 20 had no Yahoo prices); the full pull
# produced prices for 712 of 1,125 membership tickers (413 with no prices). Coverage of each year's S&P 500 members
# with prices: 40% in 1996, 68% in 2008, 80% in 2016, 87% in 2020, 92% to 97% from 2022. Only 2022-01 to 2025-07 meets
# the 90% rule. Fix made on the way: the first version downloaded the membership file with urllib, which failed on this
# machine (ssl.SSLError ASN1 NOT_ENOUGH_DATA from the Windows certificate store); it now uses requests.
# Not unit-tested: the Yahoo download (fetch_batch) and main(), checked by the real runs above.
# ---------------------------------------------------------------------------------------------------------
