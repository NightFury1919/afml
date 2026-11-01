"""Tests for spy_since.py: the SPY benchmark ("what would $1,000 be worth now?").

Plain English: the script reads a prices file, takes the SPY close on or before a start date as the entry price, and compares it
with the last close. These tests build tiny price files whose answers can be checked by hand (100 -> 110 is +10%).

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python -m pytest test_spy_since.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spy_since  # noqa: E402


def write_prices(path, rows, header="date,SPY,QQQ"):
    """rows: (date, spy) pairs; spy None gives a blank cell. QQQ is filler so the SPY column is not the only one."""
    lines = [header]
    for d, p in rows:
        lines.append(f"{d},{'' if p is None else p},5")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(monkeypatch, capsys, *args):
    monkeypatch.setattr(sys, "argv", ["spy_since.py", *args])
    spy_since.main()
    return capsys.readouterr().out


# ---------- load_spy ----------
def test_load_spy_reads_the_spy_column_wherever_it_is_and_skips_blank_cells(tmp_path):
    f = tmp_path / "p.csv"
    f.write_text("date,QQQ,SPY\n2026-01-02,50,100\n2026-01-05,51,\n2026-01-06,52,102.5\n", encoding="utf-8")
    assert spy_since.load_spy(f) == [("2026-01-02", 100.0), ("2026-01-06", 102.5)]


# ---------- the arithmetic ----------
def test_return_and_dollars_for_a_ten_percent_gain(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", 110)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-05")
    assert "Entry: 2026-10-05  close 100.00" in out
    assert "Last:  2026-10-06  close 110.00" in out
    assert "Return: +10.00%" in out
    assert "$1,000 -> $1,100.00" in out


def test_a_loss_shows_a_minus_sign_and_less_money(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 200), ("2026-10-06", 150)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-05")
    assert "Return: -25.00%" in out
    assert "$1,000 -> $750.00" in out


def test_amount_scales_the_dollars_but_not_the_percentage(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", 80)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-05", "--amount", "2500")
    assert "Return: -20.00%" in out
    assert "$2,500 -> $2,000.00" in out


# ---------- which day is the entry ----------
def test_start_on_a_trading_day_uses_that_days_close(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", 105), ("2026-10-07", 120)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-06")
    assert "Entry: 2026-10-06  close 105.00" in out
    assert "Return: +14.29%" in out


def test_start_on_a_weekend_uses_the_previous_fridays_close(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-02", 100), ("2026-10-05", 90)])       # Fri, Mon
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-04")   # Sunday
    assert "Entry: 2026-10-02  close 100.00" in out
    assert "Return: -10.00%" in out
    assert "$1,000 -> $900.00" in out


def test_start_before_the_first_close_stops_with_a_message(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", 110)])
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-01")
    assert "No SPY close on or before 2026-10-01" in str(e.value)


def test_start_after_the_last_close_gives_zero_return_from_the_last_close(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", 110)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2027-01-01")
    assert "Entry: 2026-10-06  close 110.00" in out
    assert "Return: +0.00%" in out


def test_blank_spy_days_are_ignored_when_picking_the_entry(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-06", None), ("2026-10-07", 125)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-06")
    assert "Entry: 2026-10-05  close 100.00" in out          # the blank 10-06 is not a close
    assert "Return: +25.00%" in out


# ---------- default file ----------
def test_default_file_is_the_newest_by_name(tmp_path, monkeypatch, capsys):
    write_prices(tmp_path / "prices_daily_asof_2026-10-19.csv", [("2026-10-05", 100), ("2026-10-19", 200)])
    write_prices(tmp_path / "prices_daily_asof_2026-10-30.csv", [("2026-10-05", 100), ("2026-10-30", 110)])
    monkeypatch.chdir(tmp_path)
    out = run(monkeypatch, capsys, "--start", "2026-10-05", "--today", "2026-10-30")
    assert "File:  prices_daily_asof_2026-10-30.csv" in out
    assert "Return: +10.00%" in out


def test_no_price_file_stops_with_a_clear_message(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(monkeypatch, capsys, "--start", "2026-10-05")
    assert "prices_daily_asof_" in str(e.value)


# ---------- the "file is old" note ----------
@pytest.mark.parametrize("last, today, expect_note", [
    ("2026-10-30", "2026-10-30", False),    # same day
    ("2026-10-30", "2026-11-02", False),    # Friday close, Monday run: 3 days
    ("2026-10-30", "2026-11-03", False),    # Friday close, Tuesday run after a Monday holiday: 4 days
    ("2026-10-30", "2026-11-04", True),     # 5 days: something was missed
    ("2026-10-20", "2026-10-21", False),    # an old file checked on its own date must NOT warn (old code warned: date was hard-coded)
    ("2026-10-23", "2026-11-23", True),
])
def test_note_appears_only_when_the_last_close_is_more_than_four_days_old(tmp_path, monkeypatch, capsys, last, today, expect_note):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), (last, 110)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-05", "--today", today)
    assert ("Note:" in out) is expect_note


def test_note_says_how_many_days_old_and_names_the_fix(tmp_path, monkeypatch, capsys):
    f = tmp_path / "p.csv"
    write_prices(f, [("2026-10-05", 100), ("2026-10-23", 110)])
    out = run(monkeypatch, capsys, "--prices", str(f), "--start", "2026-10-05", "--today", "2026-11-23")
    assert "31 days" in out
    assert "pull_prices.py" in out
