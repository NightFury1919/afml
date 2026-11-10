"""Nightly price panel for the stock momentum account: about 500 S&P 500 members plus SPY. Sends NO orders.

Plain English:
  * The ETF job appends new days to one snapshot. For stocks we do the simple thing instead: every night we download
    about 16 months of adjusted closes for every current member plus SPY and write a NEW dated snapshot
    (prices_daily_asof_YYYY-MM-DD.csv) into portfolio/stock_live/. Rebuilding the whole panel each time means a
    dividend or split restatement can never leave a fake jump at the seam.
  * 16 months is enough for the momentum signal (it looks back 273 trading days, about 13 months).
  * The stock list is the newest row of the S&P 500 membership file (hanshof/sp500_constituents). That file may lag the
    index by weeks or months, so stocks added recently are missing and a few removed ones may still be listed. For a
    labelled forward experiment of a few weeks this is acceptable. It is NOT usable for a historical test.
  * A day's bar is dropped if it might not be final yet (before 17:00 New York), like update_prices.py.
  * If too few tickers come back (Yahoo outage, rate limit) nothing is written and the run says so.
  * Snapshots are never overwritten. Older snapshots beyond `keep` are deleted so the folder does not grow forever.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python stock_update.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

import stock_data_pull as sdp
from update_prices import SNAPSHOT_RE, drop_incomplete_today, write_snapshot

BENCHMARK = "SPY"
HISTORY_DAYS = 480          # calendar days of history to pull (about 330 trading days)
MIN_TICKERS = 350           # fewer tickers than this means the pull failed, not that the index shrank
BATCH_SIZE = 100
KEEP = 7                    # newest snapshots to keep


def universe_from_membership(membership):
    """Sorted tickers of the newest membership row, plus SPY once."""
    newest = membership.iloc[-1]
    return sorted(set(newest) | {BENCHMARK})


def pull_panel(tickers, start, end, fetch_fn=sdp.fetch_batch, batch_size=BATCH_SIZE):
    """Adjusted closes for all tickers: batches joined side by side; all-empty columns dropped; dates sorted."""
    frames = []
    for chunk in sdp.batches(list(tickers), batch_size):
        got = fetch_fn(chunk, start, end)
        if got is not None and len(got):
            frames.append(got)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1)
    panel = panel.loc[:, ~panel.columns.duplicated()].sort_index()
    panel = panel.dropna(axis=1, how="all")
    panel.index.name = "date"
    return panel


def prune_snapshots(folder, keep=KEEP):
    """Delete all but the newest `keep` snapshots in `folder`; returns the names deleted."""
    snaps = sorted((m.group(1), p) for p in Path(folder).iterdir() if (m := SNAPSHOT_RE.match(p.name)))
    gone = []
    for _, p in snaps[:-keep] if keep > 0 else snaps:
        p.unlink()
        gone.append(p.name)
    return gone


def update(folder, membership, now=None, fetch_fn=sdp.fetch_batch, history_days=HISTORY_DAYS,
           min_tickers=MIN_TICKERS, keep=KEEP):
    """Build and write today's snapshot. Returns (path or None, message). None means nothing new was written."""
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    today = now.tz_convert("America/New_York").normalize().tz_localize(None)
    tickers = universe_from_membership(membership)
    start = (today - pd.Timedelta(days=history_days)).date().isoformat()
    end = (today + pd.Timedelta(days=1)).date().isoformat()
    panel = pull_panel(tickers, start, end, fetch_fn)
    if len(panel) == 0:
        return None, "no prices came back from the data source; nothing written"
    panel = drop_incomplete_today(panel, now)
    if len(panel) == 0:
        return None, "only an unsettled bar came back; nothing written"
    if BENCHMARK not in panel.columns or pd.isna(panel[BENCHMARK].iloc[-1]):
        return None, f"{BENCHMARK} has no price on {panel.index[-1].date()}; nothing written"
    have = int(panel.iloc[-1].notna().sum())
    if have < min_tickers:
        return None, f"only {have} tickers have a price on {panel.index[-1].date()} (need {min_tickers}); nothing written"
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        path = write_snapshot(panel, folder)
    except FileExistsError:
        return None, f"a snapshot for {panel.index[-1].date()} already exists; nothing new to write"
    gone = prune_snapshots(folder, keep)
    return path, f"wrote {path.name}: {panel.shape[1]} tickers, {len(panel)} days" + (f"; removed {len(gone)} old" if gone else "")


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here / "stock_live"))
    ap.add_argument("--membership", default=str(here / "stock_data" / "sp_500_historical_components.csv"),
                    help="S&P 500 membership csv; downloaded if missing")
    args = ap.parse_args(argv)
    membership = sdp.load_membership(args.membership)
    path, msg = update(args.folder, membership)
    print(("OK: " if path else "NOTHING WRITTEN: ") + msg)
    return 0 if path or "already exists" in msg else 1


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------------
# TDD results (2026-11-09, sandbox, Python 3.11):
#   pytest portfolio/tests/test_stock_momentum.py -v
#   test_universe_is_the_newest_row_plus_spy_once_and_sorted PASSED [  4%]
#   test_pull_panel_joins_batches_and_drops_empty_and_duplicate_columns PASSED [  9%]
#   test_pull_panel_with_every_batch_failing_returns_empty PASSED [ 14%]
#   test_update_writes_a_dated_snapshot_with_spy_and_the_members PASSED [ 19%]
#   test_update_start_date_is_480_days_back_and_end_is_tomorrow PASSED [ 23%]
#   test_update_drops_an_unsettled_bar_before_5pm_new_york PASSED [ 28%]
#   test_update_refuses_a_thin_panel_and_writes_nothing PASSED [ 33%]
#   test_update_refuses_when_spy_is_missing_on_the_last_day PASSED [ 38%]
#   test_update_with_no_data_writes_nothing PASSED [ 42%]
#   test_update_never_overwrites_and_says_so PASSED [ 47%]
#   test_prune_keeps_only_the_newest_snapshots PASSED [ 52%]
#   test_health_folds_per_ticker_warnings_and_keeps_errors PASSED [ 57%]
#   test_health_flags_a_thin_universe_as_an_error PASSED [ 61%]
#   test_health_still_catches_stale_data_and_missing_spy PASSED [ 66%]
#   test_signal_day_plan_holds_top_3_stocks_with_half_in_spy PASSED [ 71%]
#   test_run_report_uses_the_stock_title PASSED [ 76%]
#   test_a_thin_panel_blocks_the_run_and_makes_no_plan PASSED [ 80%]
#   test_unconfigured_stock_account_stays_inert_and_reads_no_prices PASSED [ 85%]
#   test_defaults_keep_the_etf_momentum_account_unchanged PASSED [ 90%]
#   test_update_accepts_exactly_the_minimum_and_refuses_one_less PASSED [ 95%]
#   test_health_accepts_exactly_the_minimum_and_flags_one_less PASSED [100%]
#   ============================== 21 passed in 3.10s ==============================
#   Regression: test_momentum_plan + test_momentum_daily + test_stock_momentum = 69 passed
# Rerun on Windows (mlfinlab env) before relying on it.
# ---------------------------------------------------------------------------
