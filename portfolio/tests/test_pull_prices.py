import pandas as pd

from pull_prices import load_tickers, summarize_prices


def test_load_tickers_ignores_blanks_comments_and_duplicates(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("spy\n\n# a comment\nQQQ  # inline note\nSPY\n gld \n", encoding="utf-8")
    assert load_tickers(p) == ["SPY", "QQQ", "GLD"]


def test_summarize_known_values():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"])
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [None, 5.0, 6.0]}, index=idx)
    out = summarize_prices(prices).set_index("ticker")
    assert out.loc["A", "first_date"] == "2020-01-01"
    assert out.loc["A", "last_date"] == "2020-01-10"
    assert out.loc["A", "n_obs"] == 3
    assert out.loc["A", "max_gap_days"] == 8
    assert out.loc["B", "first_date"] == "2020-01-02"
    assert out.loc["B", "n_obs"] == 2
    assert out.loc["B", "max_gap_days"] == 8


def test_summarize_column_with_no_data():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    prices = pd.DataFrame({"A": [1.0, 2.0], "Z": [None, None]}, index=idx)
    out = summarize_prices(prices).set_index("ticker")
    assert out.loc["Z", "n_obs"] == 0
    assert out.loc["Z", "first_date"] == ""

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
