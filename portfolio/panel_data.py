"""Load the daily price panel and reshape wide <-> long.

Wide  : one row per date, one column per ETF (what pull_prices.py writes).
Long  : one row per (date, asset), the shape the models and labels use.

ETFs launched at different times, so early cells in a wide frame are NaN.
Nothing here fills them: an ETF only joins the panel once it has real prices.
"""
import numpy as np
import pandas as pd


def load_prices(path):
    """Read prices_daily.csv into a wide frame (date index, ticker columns).

    Raises ValueError on duplicate dates or non-positive prices. A column with
    no data at all is dropped. Missing prices stay NaN.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        raise ValueError("prices file has duplicate dates")
    df = df.set_index("date").sort_index().astype(float)
    df = df.dropna(axis=1, how="all")
    if (df.fillna(1.0) <= 0).any().any():
        raise ValueError("prices file has non-positive prices")
    df.index.name = "date"
    df.columns.name = "asset"
    return df


def wide_to_long(wide, name):
    """Wide frame -> Series indexed by (date, asset), named `name`. Keeps NaN rows."""
    w = wide.rename_axis(index="date", columns="asset")
    long = w.reset_index().melt(id_vars="date", var_name="asset", value_name=name)
    return long.set_index(["date", "asset"])[name].sort_index()


# SH is the inverse-S&P hedge candidate. It is pulled with the universe but is
# not one of the 48 ranked ETFs, so it must not be ranked or labeled with them.
HEDGE_TICKERS = ("SH",)


def drop_hedge(prices, hedge=HEDGE_TICKERS):
    """Prices without the hedge ETFs, so only the ranked universe is left."""
    return prices.drop(columns=[c for c in hedge if c in prices.columns])

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
