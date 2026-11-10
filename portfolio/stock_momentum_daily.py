"""Nightly runner for the STOCK momentum account: same rule as the ETF momentum account, on current S&P 500 stocks.

Plain English: it is momentum_daily.py pointed at the stock price panel that stock_update.py writes. The rule is the
live momentum rule: every 21 trading days rank the stocks by return from 273 to 21 trading days back, hold the top 3
equal weight in a sleeve that is 50% of the account, the rest in SPY. It places NO orders: execute_plan.py does that,
by hand, with the floor and the order ledger. Until sleeve_n, sleeve_share and start are filled in run_config.json the
account stays inert.

This is a labelled forward experiment with no demonstrated edge. The ETF version of the rule did not beat SPY in the
25-year pre-registered test, and no stock-history test has been run.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env, with the stock account's paper keys loaded:
    python stock_momentum_daily.py --init-config
    python stock_momentum_daily.py --from-alpaca
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

import momentum_daily as md
from daily_run import fetch_account_inputs, health_check

MIN_TICKERS = 350
TITLE = "Stock momentum account"
STRATEGY = "stock_momentum"


def stock_health(prices, today, min_tickers=MIN_TICKERS, **kw):
    """The shared health check, adapted to a 500-stock panel.

    Errors from the shared check are kept. The per-ticker warnings (no price today, one-day moves over 25%) are folded
    into one line each, since with 500 stocks a few of them are normal. If fewer than `min_tickers` stocks have a price
    on the last date the panel is treated as broken (error): ranking a thin universe would be misleading.
    """
    issues = health_check(prices, today, **kw)
    out = [i for i in issues if i["code"] not in ("ticker_stale", "big_move")]
    for code, text in (("ticker_stale", "tickers have no price on the last date"),
                       ("big_move", "tickers had a one-day move over 25% in the last 10 days (earnings, splits or bad data)")):
        names = sorted(i["ticker"] for i in issues if i["code"] == code)
        if names:
            shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
            out.append(dict(level="warn", code=code, ticker=None, message=f"{len(names)} {text}: {shown}"))
    if prices is not None and len(prices):
        have = int(prices.iloc[-1].notna().sum())
        if have < min_tickers:
            out.append(dict(level="error", code="thin_universe", ticker=None,
                            message=f"only {have} tickers have a price on {prices.index[-1].date()} (need {min_tickers})"))
    return out


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here / "stock_live"), help="folder holding the stock price snapshots")
    ap.add_argument("--plans-dir", default=str(here / "accounts" / "stock" / "plans"))
    ap.add_argument("--runs-dir", default=str(here / "accounts" / "stock" / "runs"))
    ap.add_argument("--config", default=str(here / "accounts" / "stock" / "run_config.json"))
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    ap.add_argument("--no-update", action="store_true", help="accepted for symmetry; prices are never pulled here")
    ap.add_argument("--from-alpaca", action="store_true", help="read equity and positions from the paper account")
    ap.add_argument("--init-config", action="store_true", help="write run_config.json once and stop")
    args = ap.parse_args(argv)

    if args.init_config:
        try:
            print(f"Wrote {md.init_config(args.config)}")
            return 0
        except FileExistsError as e:
            print(e)
            return 2
    equity, holdings = None, {}
    if args.from_alpaca:
        if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
            print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys of the stock account).")
            return 2
        equity, holdings = fetch_account_inputs()
    cfg, found = md.load_config(args.config)
    today = args.today or pd.Timestamp.now(tz="America/New_York").date().isoformat()
    rep = md.run_momentum_daily(args.folder, args.plans_dir, args.runs_dir, cfg, today, holdings=holdings,
                                equity=equity, config_found=found, health_fn=stock_health, title=TITLE, strategy=STRATEGY)
    print(f"STOCK MOMENTUM DAILY RUN {rep['run_date']}: {rep['status'].upper()}")
    print("No orders were sent. This run only plans and records.\n")
    print(md.render_report(rep))
    print(f"Saved {rep['report_path']}")
    return 0 if rep["status"] == "ok" else 1


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
