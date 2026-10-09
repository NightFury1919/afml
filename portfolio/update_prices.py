"""Append new trading days to the ETF price snapshot, safely.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python update_prices.py
    python update_prices.py --folder . --universe universe.txt --overlap-days 14

What it does
    1. Finds the newest prices_daily_asof_YYYY-MM-DD.csv in the folder.
    2. Pulls the last `overlap-days` calendar days (plus anything newer) for every ticker.
    3. Drops today's bar if the US market might still be settling it (before 17:00 New York).
    4. Checks the overlap against the snapshot (see below), then appends only the new days.
    5. Writes a NEW snapshot named for the last data date. It never overwrites a file.

Why the overlap matters
    Yahoo's adjusted closes are restated whenever an ETF pays a dividend or splits: every
    earlier price is multiplied by one constant factor. Gluing new prices onto an old,
    unrestated history would show a fake one-day drop of about the dividend yield. So the
    overlap is compared first. If a ticker's old prices differ from the fresh ones by a constant
    factor, the old history is multiplied by it (exactly what the restatement does), which
    keeps every return unchanged. If the factor is not constant, or is implausibly large
    (more than 50%), the script stops and tells you which ticker, instead of guessing.

Research use only, like pull_prices.py: yfinance reads Yahoo's unofficial endpoints.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from panel_data import load_prices

SNAPSHOT_RE = re.compile(r"^prices_daily_asof_(\d{4}-\d{2}-\d{2})\.csv$")
NEW_YORK = "America/New_York"
SETTLED_HOUR = 17          # a US daily bar is treated as final from 17:00 New York time
DEFAULT_TOL = 1e-4         # ratios within 0.01% of 1, and a ratio spread below it, count as equal
DEFAULT_MAX_RESCALE = 0.5  # a restatement above 50% is treated as bad data


def drop_incomplete_today(fresh, now=None):
    """Remove bars dated today (New York) unless it is already 17:00 or later there."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    ny = now.tz_convert(NEW_YORK)
    if ny.hour >= SETTLED_HOUR:
        return fresh
    return fresh[fresh.index < pd.Timestamp(ny.date())]


def reconcile_and_append(existing, fresh, tol=DEFAULT_TOL, max_rescale=DEFAULT_MAX_RESCALE):
    """Return (merged, report). See the module docstring for the rules.

    report: rescaled {ticker: factor}, new_dates [Timestamp], stale [ticker], ignored [ticker],
            unchecked [ticker] (no overlapping prices, so the adjustment could not be compared).
    """
    ignored = sorted(set(fresh.columns) - set(existing.columns))
    fresh = fresh[[c for c in existing.columns if c in fresh.columns]]
    if (fresh.fillna(1.0) <= 0).any().any():
        raise ValueError("fresh prices contain non-positive values")

    overlap = existing.index.intersection(fresh.index)
    if len(overlap) == 0:
        raise ValueError("the fresh pull has no overlap with the history, so the adjustment "
                         "cannot be checked; pull a longer window")

    merged_old = existing.copy()
    rescaled, unchecked = {}, []
    for t in fresh.columns:
        both = pd.concat([existing.loc[overlap, t], fresh.loc[overlap, t]], axis=1).dropna()
        if both.empty:
            unchecked.append(t)
            continue
        ratio = both.iloc[:, 1] / both.iloc[:, 0]
        spread = (ratio.max() - ratio.min()) / ratio.median()
        if spread > tol:
            raise ValueError(f"{t}: old and fresh prices differ by a changing factor over the "
                             f"overlap (spread {spread:.2e}); the data disagree, not just re-adjusted")
        factor = float(ratio.iloc[-1])
        if abs(factor - 1.0) > max_rescale:
            raise ValueError(f"{t}: restatement factor {factor:.4f} is implausibly large")
        if abs(factor - 1.0) > tol:
            merged_old[t] = existing[t] * factor
            rescaled[t] = factor

    new = fresh[fresh.index > existing.index.max()].dropna(how="all")
    out = pd.concat([merged_old, new.reindex(columns=existing.columns)]).sort_index()
    out.index.name, out.columns.name = existing.index.name, existing.columns.name

    last_valid = out.apply(lambda s: s.last_valid_index())
    stale = sorted(t for t in out.columns if last_valid[t] is not None and last_valid[t] < out.index.max()
                   and (len(new) > 0))
    report = dict(rescaled=rescaled, new_dates=list(new.index), stale=stale,
                  ignored=ignored, unchecked=sorted(unchecked))
    return out, report


def snapshot_name(prices):
    return f"prices_daily_asof_{prices.index.max().date().isoformat()}.csv"


def write_snapshot(prices, folder):
    """Write a new dated snapshot. Raises FileExistsError rather than overwrite one."""
    path = Path(folder) / snapshot_name(prices)
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; snapshots are never overwritten")
    prices.to_csv(path, index_label="date")
    return path


def latest_snapshot(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    found = [(m.group(1), p) for p in folder.iterdir() if (m := SNAPSHOT_RE.match(p.name))
             and _valid_date(m.group(1))]
    if not found:
        raise FileNotFoundError(f"no prices_daily_asof_YYYY-MM-DD.csv in {folder}")
    return max(found)[1]


def _valid_date(text):
    try:
        pd.Timestamp(text)
        return True
    except ValueError:
        return False


def pull_since(tickers, start):
    """Adjusted closes from `start` onward, one ticker at a time (needs yfinance)."""
    import yfinance as yf

    series = {}
    for t in tickers:
        hist = yf.Ticker(t).history(start=str(start), auto_adjust=True)
        if hist.empty:
            print(f"WARNING: no data returned for {t}")
            continue
        s = hist["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        series[t] = s
    out = pd.DataFrame(series).sort_index()
    out.index.name = "date"
    return out


def main():
    from pull_prices import load_tickers

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here))
    ap.add_argument("--universe", default=str(here / "universe.txt"))
    ap.add_argument("--overlap-days", type=int, default=14)
    args = ap.parse_args()

    src = latest_snapshot(args.folder)
    existing = load_prices(src)
    print(f"Latest snapshot: {src.name} (last date {existing.index.max().date()})")
    start = (existing.index.max() - pd.Timedelta(days=args.overlap_days)).date()
    fresh = drop_incomplete_today(pull_since(load_tickers(args.universe), start))

    merged, rep = reconcile_and_append(existing, fresh)
    print(f"New trading days: {[d.date().isoformat() for d in rep['new_dates']] or 'none'}")
    if rep["rescaled"]:
        print("Re-adjusted history (dividend or split) for: " +
              ", ".join(f"{t} x{f:.5f}" for t, f in sorted(rep["rescaled"].items())))
    for key, label in [("stale", "No new price for"), ("ignored", "Ignored (not in history)"),
                       ("unchecked", "Adjustment not checked for")]:
        if rep[key]:
            print(f"{label}: {', '.join(rep[key])}")
    try:
        path = write_snapshot(merged, args.folder)
    except FileExistsError:
        print("Nothing new to save: the snapshot for this last date already exists.")
        return
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_update_prices.py -v
#
# collected 19 items
#
# test_update_prices.py::test_new_days_are_appended_and_old_values_are_kept PASSED [  5%]
# test_update_prices.py::test_no_new_days_returns_the_same_history PASSED  [ 10%]
# test_update_prices.py::test_rows_dated_on_or_before_the_last_known_date_never_add_rows PASSED [ 15%]
# test_update_prices.py::test_a_readjusted_ticker_is_rescaled_so_returns_stay_continuous PASSED [ 21%]
# test_update_prices.py::test_a_tiny_difference_inside_the_tolerance_is_not_rescaled PASSED [ 26%]
# test_update_prices.py::test_a_ratio_that_changes_across_the_overlap_is_rejected PASSED [ 31%]
# test_update_prices.py::test_an_implausibly_large_restatement_is_rejected PASSED [ 36%]
# test_update_prices.py::test_no_overlap_means_no_way_to_check_so_it_is_rejected PASSED [ 42%]
# test_update_prices.py::test_a_ticker_missing_from_the_fresh_pull_is_reported_stale_and_not_filled PASSED [ 47%]
# test_update_prices.py::test_tickers_not_in_the_history_are_ignored_and_reported PASSED [ 52%]
# test_update_prices.py::test_non_positive_new_prices_are_rejected PASSED  [ 57%]
# test_update_prices.py::test_new_rows_with_no_prices_at_all_are_dropped PASSED [ 63%]
# test_update_prices.py::test_snapshot_name_uses_the_last_data_date PASSED [ 68%]
# test_update_prices.py::test_write_snapshot_round_trips_and_never_overwrites PASSED [ 73%]
# test_update_prices.py::test_latest_snapshot_picks_the_newest_date_and_ignores_other_files PASSED [ 78%]
# test_update_prices.py::test_todays_bar_is_dropped_while_the_us_market_is_open_or_just_closed PASSED [ 84%]
# test_update_prices.py::test_todays_bar_is_kept_once_the_data_has_settled PASSED [ 89%]
# test_update_prices.py::test_the_cutoff_follows_new_york_time_not_the_computers_time_zone PASSED [ 94%]
# test_update_prices.py::test_earlier_days_are_never_dropped PASSED        [100%]
#
# 19 passed in 0.36s
# Mutation checks (sandbox): never rescaling, no constant-factor check, no size check, local time
#   instead of New York time, allowing overwrite, appending the last known date again, and allowing
#   no overlap each made tests fail; original restored.
# CLI smoke test (sandbox, fake data in place of Yahoo): appended 3 days, rescaled a dividend-paying
#   ticker by 0.99, saved a new dated file, and a second run reported nothing new.
# NOT yet run against real Yahoo data. Do that on your machine: python update_prices.py
# ---------------------------------------------------------------------------
