import numpy as np
import pandas as pd
import pytest

from panel_data import drop_hedge, load_prices


def write(tmp_path, text):
    p = tmp_path / "prices.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_loads_dates_as_sorted_index_and_tickers_as_columns(tmp_path):
    p = write(tmp_path, "date,SPY,QQQ\n2020-01-03,102,202\n2020-01-02,101,201\n")
    out = load_prices(p)
    assert list(out.columns) == ["SPY", "QQQ"]
    assert out.index.is_monotonic_increasing
    assert out.index.name == "date" and out.columns.name == "asset"
    assert out.loc[pd.Timestamp("2020-01-02"), "QQQ"] == 201


def test_missing_prices_stay_nan_for_late_starting_etfs(tmp_path):
    p = write(tmp_path, "date,SPY,EMB\n2020-01-02,101,\n2020-01-03,102,50\n")
    out = load_prices(p)
    assert np.isnan(out.loc[pd.Timestamp("2020-01-02"), "EMB"])


def test_duplicate_dates_are_rejected(tmp_path):
    p = write(tmp_path, "date,SPY\n2020-01-02,101\n2020-01-02,102\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_non_positive_prices_are_rejected(tmp_path):
    p = write(tmp_path, "date,SPY\n2020-01-02,101\n2020-01-03,0\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_column_with_no_data_is_dropped(tmp_path):
    p = write(tmp_path, "date,SPY,ZZZ\n2020-01-02,101,\n2020-01-03,102,\n")
    out = load_prices(p)
    assert list(out.columns) == ["SPY"]


def test_drop_hedge_removes_sh_and_leaves_the_rest_untouched():
    prices = pd.DataFrame({"SPY": [1.0, 2.0], "SH": [3.0, 4.0], "QQQ": [5.0, 6.0]})
    out = drop_hedge(prices)
    assert list(out.columns) == ["SPY", "QQQ"]
    assert list(prices.columns) == ["SPY", "SH", "QQQ"]  # input not modified


def test_drop_hedge_ignores_a_hedge_that_is_not_present():
    prices = pd.DataFrame({"SPY": [1.0, 2.0]})
    assert list(drop_hedge(prices).columns) == ["SPY"]

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_panel_data.py::test_loads_dates_as_sorted_index_and_tickers_as_columns PASSED [ 93%]
# test_panel_data.py::test_missing_prices_stay_nan_for_late_starting_etfs PASSED [ 94%]
# test_panel_data.py::test_duplicate_dates_are_rejected PASSED             [ 95%]
# test_panel_data.py::test_non_positive_prices_are_rejected PASSED         [ 96%]
# test_panel_data.py::test_column_with_no_data_is_dropped PASSED           [ 97%]
# test_panel_data.py::test_drop_hedge_removes_sh_and_leaves_the_rest_untouched PASSED [ 98%]
# test_panel_data.py::test_drop_hedge_ignores_a_hedge_that_is_not_present PASSED [100%]
#
# 88 passed in 17.89s (all seven files)
# ---------------------------------------------------------------------------
