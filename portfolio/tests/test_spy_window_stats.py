"""Tests for spy_window_stats.py: SPY's outcome in every N-week window ("what is normal?").

Plain English: the script slides a window of weeks*5 trading days along the SPY closes, one day at a time, and reports the
spread of the window returns. These tests build price series whose window returns are known on purpose.

The known series (used by most tests), for 1-week windows (5 trading days):
    first 5 closes = 100
    next 5 closes  = 100 * (1 + r)  with r = -20%, -5%, 0%, +5%, +30%
    last 5 closes  = previous * (1 + s) with s = +110%, +60%, +15%, -12%, +8%
That is 15 closes and 10 windows. Window k compares close k+5 with close k, so the ten returns are exactly r then s.
Sorted: -20, -12, -5, 0, +5, +8, +15, +30, +60, +110 (percent).

The script picks a percentile with w[int(q * (len(w) - 1))], a floor (no averaging of neighbours). With 10 windows that makes
the "median" the 5th smallest (+5%), not the midpoint of the 5th and 6th (+6.5%). A test below pins that behaviour down so
nobody changes it by accident; if the rule is changed on purpose, change the test with it.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python -m pytest test_spy_window_stats.py -v
"""
import datetime as dt
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spy_window_stats  # noqa: E402

R = [-0.20, -0.05, 0.00, 0.05, 0.30]
S = [1.10, 0.60, 0.15, -0.12, 0.08]


def known_prices():
    base = [100.0] * 5
    mid = [100.0 * (1 + r) for r in R]
    last = [m * (1 + s) for m, s in zip(mid, S)]
    return base + mid + last


def dates(n, start="2020-01-01"):
    d0 = dt.date.fromisoformat(start)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]


def write_prices(path, prices, start="2020-01-01", extra_rows=()):
    lines = ["date,SPY,QQQ"]
    lines += [f"{d},{p},5" for d, p in extra_rows]
    for d, p in zip(dates(len(prices), start), prices):
        lines.append(f"{d},{'' if p is None else p},5")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["spy_window_stats.py", *args])
    spy_window_stats.main()
    return capsys.readouterr().out


def row(out, label):
    """The line that starts with label."""
    for line in out.splitlines():
        if line.startswith(label):
            return line
    raise AssertionError(f"no line starting with {label!r} in:\n{out}")


@pytest.fixture
def known_file(tmp_path):
    f = tmp_path / "known.csv"
    write_prices(f, known_prices())
    return f


# ---------- load_spy ----------
def test_load_spy_reads_the_spy_column_wherever_it_is_and_skips_blank_cells(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("date,QQQ,SPY\n2026-01-02,50,100\n2026-01-05,51,\n2026-01-06,52,102.5\n", encoding="utf-8")
    assert spy_window_stats.load_spy(f) == [("2026-01-02", 100.0), ("2026-01-06", 102.5)]


# ---------- the distribution of window returns ----------
def test_header_counts_the_windows(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1")
    assert "windows: 10 of 1 weeks (5 trading days)" in out


def test_worst_and_best_are_the_smallest_and_largest_window(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1")
    assert row(out, "Worst").split() == ["Worst", "-20.0%", "$800"]
    assert row(out, "Best").split() == ["Best", "+110.0%", "$2,100"]


def test_percentiles_use_the_floor_index_not_an_average(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1")
    assert row(out, "5th percentile").split()[2:] == ["-20.0%", "$800"]      # index int(.05*9)=0
    assert row(out, "Median").split()[1:] == ["+5.0%", "$1,050"]             # index int(.5*9)=4, the 5th smallest; NOT +6.5%
    assert row(out, "95th percentile").split()[2:] == ["+60.0%", "$1,600"]   # index int(.95*9)=8, the 9th smallest; NOT the best


def test_shares_of_windows_above_and_below_the_thresholds(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1")
    assert row(out, "Share ending above 0%").endswith("60.0%")        # the 0% window is not "above" 0
    assert row(out, "Share ending above +10%").endswith("40.0%")
    assert row(out, "Share ending above +20%").endswith("30.0%")
    assert row(out, "Share ending above +50%").endswith("20.0%")
    assert row(out, "Share ending above +100%").endswith("10.0%")
    assert row(out, "Share ending below -10%").endswith("20.0%")


def test_amount_scales_the_dollars_but_not_the_percentages(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1", "--amount", "500")
    assert row(out, "Worst").split() == ["Worst", "-20.0%", "$400"]
    assert row(out, "Best").split() == ["Best", "+110.0%", "$1,050"]


# ---------- window length ----------
def test_weeks_sets_the_window_length_to_weeks_times_five_days(known_file, monkeypatch, capsys):
    # 2-week windows (10 days) over the same 15 closes: 5 windows, each (1+r)(1+s)-1 = +68%, +52%, +15%, -7.6%, +40.4%
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "2")
    assert "windows: 5 of 2 weeks (10 trading days)" in out
    assert row(out, "Worst").split()[1] == "-7.6%"
    assert row(out, "Median").split()[1] == "+40.4%"            # index int(.5*4)=2 of [-7.6, 15, 40.4, 52, 68]
    assert row(out, "95th percentile").split()[2] == "+52.0%"   # index int(.95*4)=3
    assert row(out, "Best").split()[1] == "+68.0%"


def test_exactly_one_more_close_than_the_window_gives_one_window(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [100, 100, 100, 100, 100, 120])
    out = run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1")
    assert "windows: 1 of 1 weeks" in out
    assert row(out, "Worst").split()[1] == "+20.0%"
    assert row(out, "Best").split()[1] == "+20.0%"
    assert row(out, "Share ending above 0%").endswith("100.0%")
    assert row(out, "Share ending below -10%").endswith("0.0%")


def test_not_enough_closes_for_one_window_stops_with_a_message(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [100, 100, 100, 100, 100])                 # 5 closes, a 5-day window needs 6
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1")
    assert "more than 5" in str(e.value)


# ---------- which closes are used ----------
def test_closes_before_since_are_not_used(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    early = [(d, 1.0) for d in dates(6, start="2015-12-20")]    # would add 6 windows (and a +huge one) if they leaked in
    write_prices(f, known_prices(), extra_rows=early)
    out = run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1")
    assert "windows: 10 of 1 weeks" in out
    assert row(out, "Worst").split()[1] == "-20.0%"


def test_since_is_inclusive(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, known_prices(), start="2020-01-01")
    out = run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1", "--since", "2020-01-01")
    assert "windows: 10 of 1 weeks" in out
    out = run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1", "--since", "2020-01-02")
    assert "windows: 9 of 1 weeks" in out                       # the first close is dropped, so one window less


def test_blank_spy_days_are_skipped_not_counted_as_closes(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    prices = known_prices()
    prices.insert(7, None)                                      # a blank cell in the middle
    write_prices(f, prices)
    out = run(monkeypatch, capsys, "--prices", str(f), "--weeks", "1")
    assert "windows: 10 of 1 weeks" in out
    assert row(out, "Best").split()[1] == "+110.0%"


# ---------- default file ----------
def test_default_file_is_the_newest_by_name(tmp_path, monkeypatch, capsys):
    write_prices(tmp_path / "prices_daily_asof_2026-10-19.csv", [100, 100, 100, 100, 100, 200])
    write_prices(tmp_path / "prices_daily_asof_2026-10-30.csv", [100, 100, 100, 100, 100, 110])
    monkeypatch.chdir(tmp_path)
    out = run(monkeypatch, capsys, "--weeks", "1")
    assert "File: prices_daily_asof_2026-10-30.csv" in out
    assert row(out, "Best").split()[1] == "+10.0%"


def test_no_price_file_stops_with_a_clear_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys)
    assert "prices_daily_asof_" in str(e.value)


def test_the_output_has_no_stray_nan_or_inf(known_file, monkeypatch, capsys):
    out = run(monkeypatch, capsys, "--prices", str(known_file), "--weeks", "1")
    assert not re.search(r"nan|inf", out, re.IGNORECASE)
