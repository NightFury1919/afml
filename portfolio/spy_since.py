r"""SPY benchmark since a start date: what would $1,000 be worth now?

Run from C:\ws\AFML\portfolio in the mlfinlab env:
    python spy_since.py                      # since 2026-09-09, newest prices_daily_asof_*.csv
    python spy_since.py --start 2026-10-26   # since the live accounts started
    python spy_since.py --prices prices_daily_asof_2026-10-19.csv --amount 1000

Uses the close on or before --start as the entry price. Prices are adjusted
closes (dividends included); no costs or taxes.
"""
import argparse
import csv
import datetime as dt
import glob
import os

STALE_DAYS = 4          # a Friday close checked on the Tuesday after a Monday holiday is 4 days old; more than that means a missed pull


def load_spy(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i = header.index("SPY")
        return [(row[0], float(row[i])) for row in r if row[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", help="prices CSV (default: newest prices_daily_asof_*.csv here)")
    ap.add_argument("--start", default="2026-09-09", help="entry date, YYYY-MM-DD")
    ap.add_argument("--amount", type=float, default=1000.0)
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    a = ap.parse_args()

    files = sorted(glob.glob("prices_daily_asof_*.csv"))
    if not a.prices and not files:
        raise SystemExit("No prices_daily_asof_*.csv in this folder. Run from C:\\ws\\AFML\\portfolio or pass --prices FILE.")
    path = a.prices or files[-1]
    spy = load_spy(path)
    before = [(d, p) for d, p in spy if d <= a.start]
    if not before:
        raise SystemExit(f"No SPY close on or before {a.start} in {path}")
    d0, p0 = before[-1]
    d1, p1 = spy[-1]
    print(f"File:  {os.path.basename(path)}")
    print(f"Entry: {d0}  close {p0:.2f}")
    print(f"Last:  {d1}  close {p1:.2f}")
    print(f"Return: {p1 / p0 - 1:+.2%}   ${a.amount:,.0f} -> ${a.amount * p1 / p0:,.2f}")
    today = dt.date.fromisoformat(a.today) if a.today else dt.date.today()
    age = (today - dt.date.fromisoformat(d1)).days
    if age > STALE_DAYS:
        print(f"Note: the file's last close is {age} days before today. Re-run pull_prices.py for a newer file.")


if __name__ == "__main__":
    main()

# =============================================================================
# TDD RESULTS (2026-11-01)  sandbox: linux -- Python 3.13.18, pytest-9.1.1, pluggy-1.6.0 -- /usr/local/bin/python3
# Command: cd portfolio ; python -m pytest test_spy_since.py -v
# =============================================================================
# test_spy_since.py::test_load_spy_reads_the_spy_column_wherever_it_is_and_skips_blank_cells PASSED [  5%]
# test_spy_since.py::test_return_and_dollars_for_a_ten_percent_gain PASSED [ 11%]
# test_spy_since.py::test_a_loss_shows_a_minus_sign_and_less_money PASSED  [ 16%]
# test_spy_since.py::test_amount_scales_the_dollars_but_not_the_percentage PASSED [ 22%]
# test_spy_since.py::test_start_on_a_trading_day_uses_that_days_close PASSED [ 27%]
# test_spy_since.py::test_start_on_a_weekend_uses_the_previous_fridays_close PASSED [ 33%]
# test_spy_since.py::test_start_before_the_first_close_stops_with_a_message PASSED [ 38%]
# test_spy_since.py::test_start_after_the_last_close_gives_zero_return_from_the_last_close PASSED [ 44%]
# test_spy_since.py::test_blank_spy_days_are_ignored_when_picking_the_entry PASSED [ 50%]
# test_spy_since.py::test_default_file_is_the_newest_by_name PASSED        [ 55%]
# test_spy_since.py::test_no_price_file_stops_with_a_clear_message PASSED  [ 61%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-30-2026-10-30-False] PASSED [ 66%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-30-2026-11-02-False] PASSED [ 72%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-30-2026-11-03-False] PASSED [ 77%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-30-2026-11-04-True] PASSED [ 83%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-20-2026-10-21-False] PASSED [ 88%]
# test_spy_since.py::test_note_appears_only_when_the_last_close_is_more_than_four_days_old[2026-10-23-2026-11-23-True] PASSED [ 94%]
# test_spy_since.py::test_note_says_how_many_days_old_and_names_the_fix PASSED [100%]
# ============================== 18 passed in 0.08s ==============================
# Real-data check (sandbox, the pre-registered file prices_daily_asof_2026-10-30.csv, SHA-256 dff1a011... after CRLF conversion):
#   cut at 2026-10-27: entry 2026-09-09 close 760.51, last 769.66, +1.20%, $1,000 -> $1,012.03 (matches the 2026-10-31 handoff).
#   full file:         last 2026-10-30 close 755.72, -0.63%, $1,000 -> $993.70 since 2026-09-09; -1.26%, $987.39 since 2026-10-26.
# Mutation check (sandbox): 30 deliberate breakages across both scripts (entry day, return and dollar arithmetic, window length, percentile rule, thresholds, share denominators, blank cells, newest-file pick, the new guards); 29 caught, 1 survivor = `x < -0.10` vs `x <= -0.10`, an EQUIVALENT mutant (no float price ratio equals exactly -10%, checked by search).
# Defects found by the tests and fixed (all three failed before the fix): (1) spy_since.py compared the last close with a hard-coded 2026-10-28, so the 'file is old' note was wrong on any other day; now it counts days against today (new --today option for tests; limit 4 days = Friday close, Tuesday run after a Monday holiday). (2) both scripts crashed with IndexError when no prices_daily_asof_*.csv was in the folder; now a clear message. (3) spy_window_stats.py crashed with IndexError when there were too few closes for one window; now a clear message. Test-first: 11 of 33 tests failed on the original code, 22 passed (the arithmetic, windows and percentiles were already right).
# Real-machine confirmation (Windows, mlfinlab): PENDING. Paste the output of: python -m pytest test_spy_since.py test_spy_window_stats.py -v
# =============================================================================
