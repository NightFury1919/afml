r"""SPY outcome in every N-week window: the 'what is normal' benchmark.

Run from C:\ws\AFML\portfolio in the mlfinlab env:
    python spy_window_stats.py                       # 7 weeks, since 2016-01-04
    python spy_window_stats.py --weeks 7 --since 2016-01-04
    python spy_window_stats.py --prices prices_daily_asof_2026-10-19.csv

A window is weeks*5 trading days, rolled one day at a time (windows overlap, so
they are not independent samples). Adjusted closes; no costs or taxes.
"""
import argparse
import csv
import glob
import os


def load_spy(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        i = header.index("SPY")
        return [(row[0], float(row[i])) for row in r if row[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", help="prices CSV (default: newest prices_daily_asof_*.csv here)")
    ap.add_argument("--weeks", type=int, default=7)
    ap.add_argument("--since", default="2016-01-04")
    ap.add_argument("--amount", type=float, default=1000.0)
    a = ap.parse_args()

    files = sorted(glob.glob("prices_daily_asof_*.csv"))
    if not a.prices and not files:
        raise SystemExit("No prices_daily_asof_*.csv in this folder. Run from C:\\ws\\AFML\\portfolio or pass --prices FILE.")
    path = a.prices or files[-1]
    px = [p for d, p in load_spy(path) if d >= a.since]
    n = a.weeks * 5
    w = sorted(px[k + n] / px[k] - 1 for k in range(len(px) - n))
    if not w:
        raise SystemExit(f"Need more than {n} SPY closes since {a.since} for a {a.weeks}-week window; {os.path.basename(path)} has {len(px)}.")

    def pct(q):
        return w[int(q * (len(w) - 1))]

    print(f"File: {os.path.basename(path)}   windows: {len(w)} of {a.weeks} weeks ({n} trading days)")
    print(f"{'Worst':<16}{w[0]:+.1%}   ${a.amount * (1 + w[0]):,.0f}")
    print(f"{'5th percentile':<16}{pct(.05):+.1%}   ${a.amount * (1 + pct(.05)):,.0f}")
    print(f"{'Median':<16}{pct(.5):+.1%}   ${a.amount * (1 + pct(.5)):,.0f}")
    print(f"{'95th percentile':<16}{pct(.95):+.1%}   ${a.amount * (1 + pct(.95)):,.0f}")
    print(f"{'Best':<16}{w[-1]:+.1%}   ${a.amount * (1 + w[-1]):,.0f}")
    print()
    for t, label in ((0, "above 0%"), (0.10, "above +10%"), (0.20, "above +20%"),
                     (0.50, "above +50% (1.5x)"), (1.00, "above +100% (2x)")):
        print(f"Share ending {label:<20}{sum(x > t for x in w) / len(w):.1%}")
    print(f"Share ending {'below -10%':<20}{sum(x < -0.10 for x in w) / len(w):.1%}")


if __name__ == "__main__":
    main()

# =============================================================================
# TDD RESULTS (2026-11-01)  sandbox: linux -- Python 3.13.18, pytest-9.1.1, pluggy-1.6.0 -- /usr/local/bin/python3
# Command: cd portfolio ; python -m pytest test_spy_window_stats.py -v
# =============================================================================
# test_spy_window_stats.py::test_load_spy_reads_the_spy_column_wherever_it_is_and_skips_blank_cells PASSED [  6%]
# test_spy_window_stats.py::test_header_counts_the_windows PASSED          [ 13%]
# test_spy_window_stats.py::test_worst_and_best_are_the_smallest_and_largest_window PASSED [ 20%]
# test_spy_window_stats.py::test_percentiles_use_the_floor_index_not_an_average PASSED [ 26%]
# test_spy_window_stats.py::test_shares_of_windows_above_and_below_the_thresholds PASSED [ 33%]
# test_spy_window_stats.py::test_amount_scales_the_dollars_but_not_the_percentages PASSED [ 40%]
# test_spy_window_stats.py::test_weeks_sets_the_window_length_to_weeks_times_five_days PASSED [ 46%]
# test_spy_window_stats.py::test_exactly_one_more_close_than_the_window_gives_one_window PASSED [ 53%]
# test_spy_window_stats.py::test_not_enough_closes_for_one_window_stops_with_a_message PASSED [ 60%]
# test_spy_window_stats.py::test_closes_before_since_are_not_used PASSED   [ 66%]
# test_spy_window_stats.py::test_since_is_inclusive PASSED                 [ 73%]
# test_spy_window_stats.py::test_blank_spy_days_are_skipped_not_counted_as_closes PASSED [ 80%]
# test_spy_window_stats.py::test_default_file_is_the_newest_by_name PASSED [ 86%]
# test_spy_window_stats.py::test_no_price_file_stops_with_a_clear_message PASSED [ 93%]
# test_spy_window_stats.py::test_the_output_has_no_stray_nan_or_inf PASSED [100%]
# ============================== 15 passed in 0.05s ==============================
# Real-data check (sandbox, same file): cut at 2026-10-27, 7-week windows since 2016-01-04: 2,685 windows; worst -30.3%, 5th pct -7.1%,
#   median +2.8%, 95th pct +9.7%, best +28.6%; above +10% 4.3%, above +20% 0.2%, above 1.5x 0.0%, above 2x 0.0%, below -10% 2.9%
#   (matches the 2026-10-31 handoff). Full file through 2026-10-30: 2,688 windows, same percentiles to one decimal.
# Mutation check (sandbox): 30 deliberate breakages across both scripts (entry day, return and dollar arithmetic, window length, percentile rule, thresholds, share denominators, blank cells, newest-file pick, the new guards); 29 caught, 1 survivor = `x < -0.10` vs `x <= -0.10`, an EQUIVALENT mutant (no float price ratio equals exactly -10%, checked by search).
# Defects found by the tests and fixed (all three failed before the fix): (1) spy_since.py compared the last close with a hard-coded 2026-10-28, so the 'file is old' note was wrong on any other day; now it counts days against today (new --today option for tests; limit 4 days = Friday close, Tuesday run after a Monday holiday). (2) both scripts crashed with IndexError when no prices_daily_asof_*.csv was in the folder; now a clear message. (3) spy_window_stats.py crashed with IndexError when there were too few closes for one window; now a clear message. Test-first: 11 of 33 tests failed on the original code, 22 passed (the arithmetic, windows and percentiles were already right).
# Real-machine confirmation (Windows, mlfinlab): DONE 2026-11-01, Ethan's PC, from C:\ws\AFML\portfolio
# Command: python -m pytest test_spy_since.py test_spy_window_stats.py -v   (both files together: 33 passed in 0.65s)
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# collected 33 items
# test_spy_window_stats.py::test_load_spy_reads_the_spy_column_wherever_it_is_and_skips_blank_cells PASSED [ 57%]
# test_spy_window_stats.py::test_header_counts_the_windows PASSED [ 60%]
# test_spy_window_stats.py::test_worst_and_best_are_the_smallest_and_largest_window PASSED [ 63%]
# test_spy_window_stats.py::test_percentiles_use_the_floor_index_not_an_average PASSED [ 66%]
# test_spy_window_stats.py::test_shares_of_windows_above_and_below_the_thresholds PASSED [ 69%]
# test_spy_window_stats.py::test_amount_scales_the_dollars_but_not_the_percentages PASSED [ 72%]
# test_spy_window_stats.py::test_weeks_sets_the_window_length_to_weeks_times_five_days PASSED [ 75%]
# test_spy_window_stats.py::test_exactly_one_more_close_than_the_window_gives_one_window PASSED [ 78%]
# test_spy_window_stats.py::test_not_enough_closes_for_one_window_stops_with_a_message PASSED [ 81%]
# test_spy_window_stats.py::test_closes_before_since_are_not_used PASSED [ 84%]
# test_spy_window_stats.py::test_since_is_inclusive PASSED [ 87%]
# test_spy_window_stats.py::test_blank_spy_days_are_skipped_not_counted_as_closes PASSED [ 90%]
# test_spy_window_stats.py::test_default_file_is_the_newest_by_name PASSED [ 93%]
# test_spy_window_stats.py::test_no_price_file_stops_with_a_clear_message PASSED [ 96%]
# test_spy_window_stats.py::test_the_output_has_no_stray_nan_or_inf PASSED [100%]
# (15 of this file's tests PASSED; the other file's 18 also PASSED)
# Real-data run on Windows (prices_daily_asof_2026-10-30.csv, the committed pre-registered file):
#   File: prices_daily_asof_2026-10-30.csv windows: 2688 of 7 weeks (35 trading days)
#   Worst -30.3% $697
#   5th percentile -7.1% $929
#   Median +2.8% $1,028
#   95th percentile +9.7% $1,097
#   Best +28.6% $1,286
#   Share ending above 0% 73.1%
#   Share ending above +10% 4.3%
#   Share ending above +20% 0.2%
#   Share ending above +50% (1.5x) 0.0%
#   Share ending above +100% (2x) 0.0%
#   Share ending below -10% 2.9%
# =============================================================================
