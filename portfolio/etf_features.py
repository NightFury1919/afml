"""Daily cross-sectional features for the ETF panel (pre-registered, no tuning).

Five fixed features, each with a reason someone would pay for it:
    mom_12_1   return from t-252 to t-21 (skips the last month)   winners keep winning for months
    mom_3m     return over the last 63 days                       medium-horizon continuation
    ret_5d     return over the last 5 days                        short-term overreaction fades
    vol_60     realized volatility over 60 days                   low-risk assets are often under-priced
    trend_200  price / 200-day average - 1                        trend filter

Steps for every ETF and date t, using only prices up to and including t:
    1. compute the raw feature; momentum, short return and trend are divided by
       the ETF's own trailing 60-day volatility so bonds and gold miners are comparable
    2. keep an ETF on date t only if ALL five features exist (it has enough history)
    3. rank each feature across the eligible ETFs on that date, scaled to (0, 1)
       as (rank - 0.5) / n, so 0.5 is the middle and ties share their average rank
    4. drop dates with fewer than min_assets eligible ETFs; a ranking among 3 ETFs
       means little

The windows below are the frozen defaults. Passing other windows is only for
small hand-checkable tests.
"""
from functools import reduce

import numpy as np
import pandas as pd

from panel_data import wide_to_long

DEFAULT_WINDOWS = {"mom_long": 252, "mom_skip": 21, "mom_med": 63,
                   "ret_short": 5, "vol": 60, "trend": 200}
FEATURE_NAMES = ["mom_12_1", "mom_3m", "ret_5d", "vol_60", "trend_200"]
DEFAULT_MIN_ASSETS = 10
DEFAULT_MIN_GROUP_ASSETS = 3   # a group needs this many eligible ETFs on a date to be ranked


def trailing_vol(prices, window):
    """Sample standard deviation of daily log returns over the last `window` returns."""
    return np.log(prices).diff().rolling(window).std()


def trend_distance(prices, window):
    """Price divided by its `window`-day simple average, minus 1."""
    return prices / prices.rolling(window).mean() - 1


def lagged_log_return(prices, near, far):
    """log(price `near` days ago / price `far` days ago). near=0 means today."""
    logp = np.log(prices)
    return logp.shift(near) - logp.shift(far)


def raw_features(prices, windows=DEFAULT_WINDOWS):
    """Dict of wide frames, one per feature, before eligibility and ranking."""
    w = windows
    sigma = trailing_vol(prices, w["vol"])

    def scale(x):
        return (x / sigma).replace([np.inf, -np.inf], np.nan)

    return {
        "mom_12_1": scale(lagged_log_return(prices, near=w["mom_skip"], far=w["mom_long"])),
        "mom_3m": scale(lagged_log_return(prices, near=0, far=w["mom_med"])),
        "ret_5d": scale(lagged_log_return(prices, near=0, far=w["ret_short"])),
        "vol_60": sigma.replace([np.inf, -np.inf], np.nan),
        "trend_200": scale(trend_distance(prices, w["trend"])),
    }


def cross_sectional_rank(wide, groups=None):
    """Rank each row across its non-NaN entries, scaled to (0, 1) as (rank - 0.5) / n.

    With groups ({ticker: group name}), each ETF is ranked only against the others in its own
    group, so every group spreads evenly over (0, 1) by itself.
    """
    if groups is None:
        ranks = wide.rank(axis=1, method="average")
        n = wide.notna().sum(axis=1)
        return (ranks - 0.5).div(n, axis=0)

    _check_groups(wide.columns, groups)
    out = pd.DataFrame(np.nan, index=wide.index, columns=wide.columns)
    for g in sorted({groups[c] for c in wide.columns}):
        cols = [c for c in wide.columns if groups[c] == g]
        out[cols] = cross_sectional_rank(wide[cols])
    return out


def _check_groups(columns, groups):
    missing = [c for c in columns if c not in groups]
    if missing:
        raise ValueError(f"no group for: {missing}")


def compute_features(prices, windows=DEFAULT_WINDOWS, min_assets=DEFAULT_MIN_ASSETS,
                     groups=None, min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """Long frame indexed by (date, asset) with the five ranked features.

    Only eligible rows are returned: the ETF has all five features that day and
    the date has at least min_assets eligible ETFs.

    With groups ({ticker: group name}) each feature is ranked within the ETF's own group, and a
    group with fewer than min_group_assets eligible ETFs on a date is left out that day.
    """
    raw = raw_features(prices, windows)
    mask = reduce(lambda a, b: a & b, [raw[n].notna() for n in FEATURE_NAMES])

    if groups is not None:
        _check_groups(prices.columns, groups)
        for g in sorted({groups[c] for c in prices.columns}):
            cols = [c for c in prices.columns if groups[c] == g]
            active = (mask[cols].sum(axis=1) >= min_group_assets).to_numpy()
            mask[cols] = mask[cols] & active[:, None]

    enough = mask.sum(axis=1) >= min_assets
    mask.loc[~enough] = False

    ranked = [wide_to_long(cross_sectional_rank(raw[n].where(mask), groups), n) for n in FEATURE_NAMES]
    out = pd.concat(ranked, axis=1).dropna(how="any")
    return out.sort_index()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_etf_features.py::test_default_windows_are_the_preregistered_values PASSED [ 53%]
# test_etf_features.py::test_trailing_vol_known_value PASSED               [ 54%]
# test_etf_features.py::test_trend_distance_known_value PASSED             [ 55%]
# test_etf_features.py::test_lagged_log_return_known_values PASSED         [ 56%]
# test_etf_features.py::test_cross_sectional_rank_known_values PASSED      [ 57%]
# test_etf_features.py::test_raw_momentum_is_divided_by_trailing_vol PASSED [ 59%]
# test_etf_features.py::test_vol_feature_is_not_divided_by_itself PASSED   [ 60%]
# test_etf_features.py::test_features_are_ranks_strictly_between_zero_and_one PASSED [ 61%]
# test_etf_features.py::test_each_date_has_ranks_spread_evenly_over_the_eligible_assets PASSED [ 62%]
# test_etf_features.py::test_no_lookahead_changing_later_prices_leaves_earlier_features_alone PASSED [ 63%]
# test_etf_features.py::test_late_starting_etf_only_appears_once_it_has_enough_history PASSED [ 64%]
# test_etf_features.py::test_dates_with_too_few_eligible_etfs_are_dropped PASSED [ 65%]
# test_etf_features.py::test_flat_price_etf_is_left_out_not_given_an_infinite_feature PASSED [ 67%]
# test_etf_features.py::test_cross_sectional_rank_within_groups_known_values PASSED [ 68%]
# test_etf_features.py::test_grouped_rank_with_groups_none_equals_the_pooled_rank PASSED [ 69%]
# test_etf_features.py::test_grouped_features_rank_inside_each_group_and_each_group_is_evenly_spread PASSED [ 70%]
# test_etf_features.py::test_a_group_with_too_few_eligible_etfs_is_dropped_that_day PASSED [ 71%]
# test_etf_features.py::test_every_priced_etf_must_have_a_group PASSED     [ 72%]
# test_etf_features.py::test_default_behaviour_without_groups_is_unchanged PASSED [ 73%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): ignoring groups in the grouped ranking, and not dropping small groups, each made tests fail; original restored.
# ---------------------------------------------------------------------------
