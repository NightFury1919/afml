"""Cross-sectional labels for the ETF panel (pre-registered design).

For ETF i on date t, with horizon h trading days:
    fwd_ret    = log(price at t+h / price at t)
    median     = median of fwd_ret across the ETFs in the universe on date t
    excess_ret = fwd_ret - median
    label      = 1 if excess_ret > 0 else 0   (ties with the median get 0)
    t1         = the date h trading days after t (when the label is resolved;
                purging and uniqueness need it)

The universe on date t is exactly the (date, asset) rows you pass in, the ETFs
whose features exist that day. An ETF with no price at t+h gets no label.
Dates with fewer than min_assets labeled ETFs are dropped. The last h dates
have no forward return, so they never appear here; they are only used live.

The default horizon is 21 trading days (about one month), a judgment call made
for cost reasons, not a tested result.
"""
import numpy as np
import pandas as pd

from etf_features import (DEFAULT_MIN_ASSETS, DEFAULT_MIN_GROUP_ASSETS, DEFAULT_WINDOWS,
                          compute_features)
from panel_data import wide_to_long

DEFAULT_HORIZON = 21


def forward_returns(prices, horizon):
    """Wide frame of log(price at t+horizon / price at t); NaN where unavailable."""
    logp = np.log(prices)
    return logp.shift(-horizon) - logp


def make_labels(prices, index, horizon=DEFAULT_HORIZON, min_assets=DEFAULT_MIN_ASSETS,
                groups=None, min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """Labels for the (date, asset) rows in `index`. Columns: fwd_ret, excess_ret, label, t1.

    With groups ({ticker: group name}) the median is taken within each ETF's own group on each
    date. A group with fewer than min_group_assets labeled ETFs on a date is dropped that day,
    and a date still needs at least min_assets labeled ETFs in total.
    """
    fwd = wide_to_long(forward_returns(prices, horizon), "fwd_ret")
    fwd = fwd.reindex(index).dropna()

    cols = ["fwd_ret", "excess_ret", "label", "t1"]
    if fwd.empty:
        return pd.DataFrame(columns=cols, index=fwd.index)

    dates = fwd.index.get_level_values("date")
    if groups is None:
        by = fwd.groupby(level="date")
        keep = by.transform("size") >= min_assets
    else:
        g = fwd.index.get_level_values("asset").map(groups)
        if g.isna().any():
            raise ValueError("an ETF in the index has no group")
        by = fwd.groupby([dates, g.to_numpy()])
        keep = by.transform("size") >= min_group_assets
    median = by.transform("median")
    fwd, median = fwd[keep], median[keep]
    if groups is not None and not fwd.empty:
        total = fwd.groupby(level="date").transform("size")
        fwd, median = fwd[total >= min_assets], median[total >= min_assets]
    if fwd.empty:
        return pd.DataFrame(columns=cols, index=fwd.index)

    all_dates = prices.index
    t1_map = pd.Series(all_dates[horizon:], index=all_dates[:len(all_dates) - horizon])
    out = pd.DataFrame({"fwd_ret": fwd, "excess_ret": fwd - median})
    out["label"] = (out["excess_ret"] > 0).astype(int)
    out["t1"] = pd.Series(out.index.get_level_values("date")).map(t1_map).values
    return out[cols]


def build_dataset(prices, horizon=DEFAULT_HORIZON, windows=DEFAULT_WINDOWS,
                  min_assets=DEFAULT_MIN_ASSETS, groups=None, min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """Features and labels on the same (date, asset) rows, ready for modeling.

    Pass groups ({ticker: class}) to rank and label every ETF within its own class.
    """
    feats = compute_features(prices, windows, min_assets, groups, min_group_assets)
    labels = make_labels(prices, feats.index, horizon, min_assets, groups, min_group_assets)
    return feats.reindex(labels.index).join(labels)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_etf_labels.py::test_default_horizon_is_21_trading_days PASSED       [ 75%]
# test_etf_labels.py::test_forward_return_known_values PASSED              [ 76%]
# test_etf_labels.py::test_labels_known_values_balanced_case PASSED        [ 77%]
# test_etf_labels.py::test_ties_with_the_median_get_label_zero PASSED      [ 78%]
# test_etf_labels.py::test_last_dates_without_a_forward_return_are_excluded PASSED [ 79%]
# test_etf_labels.py::test_t1_is_the_date_h_trading_days_ahead PASSED      [ 80%]
# test_etf_labels.py::test_median_uses_only_the_assets_in_the_index PASSED [ 81%]
# test_etf_labels.py::test_dates_with_too_few_assets_are_dropped PASSED    [ 82%]
# test_etf_labels.py::test_asset_with_missing_future_price_gets_no_label PASSED [ 84%]
# test_etf_labels.py::test_label_uses_future_prices_but_features_index_is_all_that_decides_the_universe PASSED [ 85%]
# test_etf_labels.py::test_build_dataset_joins_features_and_labels_on_the_same_rows PASSED [ 86%]
# test_etf_labels.py::test_labels_use_the_median_of_the_etfs_own_group PASSED [ 87%]
# test_etf_labels.py::test_pooled_and_grouped_labels_differ_when_groups_move_differently PASSED [ 88%]
# test_etf_labels.py::test_a_group_with_too_few_labeled_etfs_is_dropped_on_that_date PASSED [ 89%]
# test_etf_labels.py::test_labels_without_groups_are_unchanged_by_the_new_parameters PASSED [ 90%]
# test_etf_labels.py::test_build_dataset_with_groups_runs_end_to_end_and_keeps_the_same_columns PASSED [ 92%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): using the pooled median when groups are given made tests fail; original restored.
# ---------------------------------------------------------------------------
