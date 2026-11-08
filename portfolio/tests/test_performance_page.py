"""Tests for performance_page.py, written before the code. Every expected number is worked out by hand."""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

import performance_page as pp


def series(values, start="2026-10-26"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# ------------------------------------------------------------------ metrics, hand-worked
def test_total_return_and_days():
    m = pp.series_metrics(series([1000, 1100]))
    assert m["total_return"] == pytest.approx(0.10)
    assert m["days"] == 1                      # one daily return between two values


def test_worst_fall_is_peak_to_trough():
    # peak 120, trough 90: (120-90)/120 = 0.25, and the later recovery to 110 does not erase it
    m = pp.series_metrics(series([100, 120, 90, 110]))
    assert m["max_drawdown"] == pytest.approx(0.25)
    assert m["current_drawdown"] == pytest.approx(1 - 110 / 120)


def test_no_fall_means_zero():
    m = pp.series_metrics(series([100, 101, 102]))
    assert m["max_drawdown"] == 0.0 and m["current_drawdown"] == 0.0


def test_volatility_of_constant_growth_is_zero():
    m = pp.series_metrics(series([100 * 1.01 ** k for k in range(40)]))
    assert m["volatility"] == pytest.approx(0.0, abs=1e-12)


def test_volatility_matches_definition():
    v = [100, 102, 101, 104, 103, 106, 105, 108, 107, 110, 109, 112, 111, 114, 113, 116, 115, 118, 117, 120, 119]
    r = np.diff(v) / np.array(v[:-1])
    expect = r.std(ddof=1) * math.sqrt(252)
    assert pp.series_metrics(series(v))["volatility"] == pytest.approx(expect)


def test_too_short_for_volatility_is_none_not_a_number():
    # 5 returns is far too few to call a volatility; showing one would mislead
    m = pp.series_metrics(series([100, 101, 100, 102, 101, 103]))
    assert m["volatility"] is None and m["annualized_return"] is None


def test_annualized_return_needs_enough_days_and_matches_formula():
    n = pp.MIN_DAYS_ANNUALIZED
    vals = [100 * 1.001 ** k for k in range(n + 1)]
    m = pp.series_metrics(series(vals))
    assert m["days"] == n
    assert m["annualized_return"] == pytest.approx(1.001 ** 252 - 1)
    short = pp.series_metrics(series(vals[:n]))        # one day fewer: not shown
    assert short["annualized_return"] is None


def test_distance_to_floor():
    m = pp.series_metrics(series([1000, 950]), floor=900.0)
    assert m["last_value"] == 950.0
    assert m["floor_distance"] == pytest.approx(50.0)
    assert m["floor_distance_pct"] == pytest.approx(50 / 950)
    assert m["lowest_value"] == 950.0


def test_floor_breached_gives_negative_distance():
    assert pp.series_metrics(series([1000, 880]), floor=900.0)["floor_distance"] == pytest.approx(-20.0)


def test_no_floor_means_no_floor_fields():
    m = pp.series_metrics(series([1000, 1010]))
    assert m["floor_distance"] is None


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        pp.series_metrics(series([1000]))                    # needs two points
    with pytest.raises(ValueError):
        pp.series_metrics(series([1000, 0, 1000]))           # a zero value is not a price
    with pytest.raises(ValueError):
        pp.series_metrics(series([1000, float("nan"), 1010]))
    with pytest.raises(ValueError):
        pp.series_metrics(pd.Series([1.0, 2.0], index=[pd.Timestamp("2026-10-27"), pd.Timestamp("2026-10-26")]))


# ------------------------------------------------------------------ aligning series
def test_align_cuts_to_common_dates_and_rebases_to_start_value():
    a = series([10, 11, 12, 13], "2026-10-26")
    b = series([200, 210, 220], "2026-10-27")               # starts a day later
    out = pp.align({"a": a, "b": b}, 1000.0)
    assert list(out.index) == list(b.index)                 # only dates both have
    assert out["a"].iloc[0] == pytest.approx(1000.0) and out["b"].iloc[0] == pytest.approx(1000.0)
    assert out["a"].iloc[1] == pytest.approx(1000 * 12 / 11)
    assert out["b"].iloc[2] == pytest.approx(1000 * 220 / 200)


def test_align_with_no_overlap_is_an_error():
    with pytest.raises(ValueError):
        pp.align({"a": series([1, 2], "2026-10-26"), "b": series([1, 2], "2027-01-04")}, 1000.0)


# ------------------------------------------------------------------ rendering
def metrics_for_render():
    return {"Account A model": pp.series_metrics(series([1000, 1010, 990, 1020]), floor=900.0),
            "SPY": pp.series_metrics(series([1000, 1020, 1000, 1030]))}


def test_render_says_paper_and_gives_dates_and_sample_length():
    text = pp.render_markdown(metrics_for_render(), title="Test", as_of="2026-10-30", start="2026-10-26",
                              notes=["a note"])
    assert "paper" in text.lower()
    assert "2026-10-26" in text and "2026-10-30" in text
    assert "3 trading days" in text


def test_render_short_sample_shows_na_and_a_warning_not_numbers():
    text = pp.render_markdown(metrics_for_render(), title="Test", as_of="2026-10-30", start="2026-10-26")
    assert "n/a" in text
    assert "too short" in text.lower()


def test_render_is_free_of_nan_and_none():
    text = pp.render_markdown(metrics_for_render(), title="Test", as_of="2026-10-30", start="2026-10-26")
    assert not re.search(r"\bnan\b", text.lower()) and "None" not in text


def test_render_shows_floor_row_only_when_a_floor_exists():
    text = pp.render_markdown(metrics_for_render(), title="T", as_of="2026-10-30", start="2026-10-26")
    assert "$900" in text or "900" in text
    row = [ln for ln in text.splitlines() if ln.startswith("| Distance")][0]
    assert row.count("n/a") == 1                            # SPY has no floor; the model does


def test_long_sample_shows_annualized_and_no_short_warning():
    n = pp.MIN_DAYS_ANNUALIZED
    vals = [100 * 1.001 ** k + (k % 3) for k in range(n + 1)]
    m = {"X": pp.series_metrics(series(vals))}
    text = pp.render_markdown(m, title="T", as_of="x", start="y")
    assert "too short" not in text.lower()
    assert "Annualized" in text


# ------------------------------------------------------------------ end to end on a small price table
def small_prices():
    idx = pd.bdate_range("2026-10-19", periods=8)
    rng = np.random.default_rng(1)
    cols = {t: 100 * np.cumprod(1 + rng.normal(0, 0.005, len(idx))) for t in ("SPY", "SHY")}
    return pd.DataFrame(cols, index=idx)


def test_benchmark_series_comes_from_prices_from_start_date():
    px = small_prices()
    out = pp.benchmark_series(px, ["SPY", "SHY"], "2026-10-21")
    assert out["SPY"].index[0] == pd.Timestamp("2026-10-21")
    assert out["SHY"].iloc[0] == px["SHY"].loc["2026-10-21"]


def test_benchmark_missing_ticker_is_an_error():
    with pytest.raises(ValueError):
        pp.benchmark_series(small_prices(), ["SPY", "QQQ"], "2026-10-21")


def test_start_after_last_price_is_an_error():
    with pytest.raises(ValueError):
        pp.benchmark_series(small_prices(), ["SPY"], "2027-01-01")


def test_read_actual_equity_csv(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text("date,equity\n2026-10-26,1000.0\n2026-10-27,1012.5\n", encoding="utf-8")
    s = pp.read_equity_csv(p)
    assert list(s.values) == [1000.0, 1012.5] and s.index[0] == pd.Timestamp("2026-10-26")


def test_read_equity_csv_bad_files_are_errors(tmp_path):
    (tmp_path / "a.csv").write_text("day,value\n2026-10-26,1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("date,equity\n2026-10-26,abc\n", encoding="utf-8")
    for n in ("a.csv", "b.csv", "missing.csv"):
        with pytest.raises((ValueError, FileNotFoundError)):
            pp.read_equity_csv(tmp_path / n)


def test_build_report_writes_markdown_chart_and_json(tmp_path):
    px = small_prices()
    model = pp.benchmark_series(px, ["SPY"], "2026-10-19")["SPY"] * 3      # any value series
    res = pp.build_report({"Model": model, **pp.benchmark_series(px, ["SPY", "SHY"], "2026-10-19")},
                          out_dir=tmp_path, title="T", as_of="2026-10-28", start="2026-10-19", floor=None,
                          start_value=1000.0, notes=["n"], chart=True)
    assert (tmp_path / "performance.md").exists()
    assert (tmp_path / "performance.png").stat().st_size > 1000
    data = json.loads((tmp_path / "performance.json").read_text(encoding="utf-8"))
    assert set(data["series"]) == {"Model", "SPY", "SHY"}
    assert not re.search(r"\bnan\b", (tmp_path / "performance.json").read_text(encoding="utf-8").lower())
    assert res["markdown"].endswith("performance.md") or "performance.md" in str(res["markdown"])


def test_build_report_never_overwrites_silently_it_replaces_only_its_own_files(tmp_path):
    (tmp_path / "keep.txt").write_text("mine", encoding="utf-8")
    px = small_prices()
    pp.build_report(pp.benchmark_series(px, ["SPY", "SHY"], "2026-10-19"), out_dir=tmp_path, title="T",
                    as_of="a", start="b", floor=None, start_value=1000.0, notes=[], chart=False)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not (tmp_path / "performance.png").exists()


def test_duplicate_dates_are_rejected():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-10-26", "2026-10-27", "2026-10-27"]))
    with pytest.raises(ValueError):
        pp.series_metrics(s)


def test_has_live_days_needs_two_prices_from_the_start():
    px = small_prices()                                   # 2026-10-19 .. 2026-10-28
    assert pp.has_live_days(px, "2026-10-19")
    assert pp.has_live_days(px, "2026-10-27")             # start and one later day
    assert not pp.has_live_days(px, "2026-10-28")         # only the start day itself: nothing to draw
    assert not pp.has_live_days(px, "2026-11-02")         # start after the last price

# ---------------------------------------------------------------------------------------------
# pytest -v test_performance_page.py
# ============================= test session starts ==============================
# platform linux -- Python 3.10.20, pytest-9.1.1, pluggy-1.7.0 -- /tmp/claude-0/-home-claude/186a30d5-aecd-5688-830a-09fb4cd892bd/scratchpad/perf/../venv310/bin/python
# cachedir: .pytest_cache
# rootdir: /tmp/claude-0/-home-claude/186a30d5-aecd-5688-830a-09fb4cd892bd/scratchpad/perf
# collecting ... collected 27 items
# 
# test_performance_page.py::test_total_return_and_days PASSED              [  3%]
# test_performance_page.py::test_worst_fall_is_peak_to_trough PASSED       [  7%]
# test_performance_page.py::test_no_fall_means_zero PASSED                 [ 11%]
# test_performance_page.py::test_volatility_of_constant_growth_is_zero PASSED [ 14%]
# test_performance_page.py::test_volatility_matches_definition PASSED      [ 18%]
# test_performance_page.py::test_too_short_for_volatility_is_none_not_a_number PASSED [ 22%]
# test_performance_page.py::test_annualized_return_needs_enough_days_and_matches_formula PASSED [ 25%]
# test_performance_page.py::test_distance_to_floor PASSED                  [ 29%]
# test_performance_page.py::test_floor_breached_gives_negative_distance PASSED [ 33%]
# test_performance_page.py::test_no_floor_means_no_floor_fields PASSED     [ 37%]
# test_performance_page.py::test_rejects_bad_input PASSED                  [ 40%]
# test_performance_page.py::test_align_cuts_to_common_dates_and_rebases_to_start_value PASSED [ 44%]
# test_performance_page.py::test_align_with_no_overlap_is_an_error PASSED  [ 48%]
# test_performance_page.py::test_render_says_paper_and_gives_dates_and_sample_length PASSED [ 51%]
# test_performance_page.py::test_render_short_sample_shows_na_and_a_warning_not_numbers PASSED [ 55%]
# test_performance_page.py::test_render_is_free_of_nan_and_none PASSED     [ 59%]
# test_performance_page.py::test_render_shows_floor_row_only_when_a_floor_exists PASSED [ 62%]
# test_performance_page.py::test_long_sample_shows_annualized_and_no_short_warning PASSED [ 66%]
# test_performance_page.py::test_benchmark_series_comes_from_prices_from_start_date PASSED [ 70%]
# test_performance_page.py::test_benchmark_missing_ticker_is_an_error PASSED [ 74%]
# test_performance_page.py::test_start_after_last_price_is_an_error PASSED [ 77%]
# test_performance_page.py::test_read_actual_equity_csv PASSED             [ 81%]
# test_performance_page.py::test_read_equity_csv_bad_files_are_errors PASSED [ 85%]
# test_performance_page.py::test_build_report_writes_markdown_chart_and_json PASSED [ 88%]
# test_performance_page.py::test_build_report_never_overwrites_silently_it_replaces_only_its_own_files PASSED [ 92%]
# test_performance_page.py::test_duplicate_dates_are_rejected PASSED       [ 96%]
# test_performance_page.py::test_has_live_days_needs_two_prices_from_the_start PASSED [100%]
# 
# ============================== 27 passed in 0.84s ==============================
