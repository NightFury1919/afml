"""Class budgets and the non-predictive portfolio (no model, no scores).

The pooled and within-class real-history checks found no demonstrated predictive edge in the five
features (preregistration_real_history_check*.md). The honest fallback for the challenge is a
transparent portfolio: hold every ETF, decide only HOW MUCH goes to each asset class, equal-weight
inside a class. This module gives three ways to set those class budgets and builds the weights.

    size_budgets          budget = share of ETFs in the class (this is plain equal weight over all ETFs;
                          benchmark-neutral for this universe).
    equal_budgets         every class gets the same share.
    inverse_vol_budgets   budget proportional to 1 / trailing volatility of the class's equal-weight
                          daily return (naive risk parity across classes: ignores correlations).

The choice of rule, the exposure and any cap are boss decisions (see the strategy document). None of
the defaults here is a finding. The window (252 daily returns) is a convention, not tuned.

This is NOT Hierarchical Risk Parity (AFML Chapter 16). That needs the book's snippets (16.1 to 16.4)
and is a separate step.
"""
import pandas as pd

from portfolio_construction import target_weights

DEFAULT_VOL_WINDOW = 252


def _classes(groups, tickers=None):
    tickers = list(groups) if tickers is None else list(tickers)
    missing = [t for t in tickers if t not in groups]
    if missing:
        raise ValueError(f"no class for: {missing}")
    return tickers, sorted({groups[t] for t in tickers})


def size_budgets(groups, tickers=None):
    tickers, classes = _classes(groups, tickers)
    return {c: sum(groups[t] == c for t in tickers) / len(tickers) for c in classes}


def equal_budgets(groups, tickers=None):
    _, classes = _classes(groups, tickers)
    return {c: 1.0 / len(classes) for c in classes}


def class_returns(prices, groups):
    """Equal-weight daily return of each class, using the members that have a return that day."""
    cols = [c for c in prices.columns if c in groups]
    r = prices[cols].pct_change(fill_method=None)
    by_class = pd.Series({c: groups[c] for c in cols})
    return r.T.groupby(by_class).mean().T


def inverse_vol_budgets(prices, groups, window=DEFAULT_VOL_WINDOW):
    """Budget proportional to 1 / std of the class return over the last `window` daily returns."""
    cr = class_returns(prices, groups).iloc[-window:]
    if len(cr) < window or cr.isna().any().any():
        raise ValueError(f"need {window} complete daily class returns")
    inv = 1.0 / cr.std()
    return (inv / inv.sum()).to_dict()


def non_predictive_weights(groups, class_budgets, tickers=None, exposure=1.0, max_weight=None):
    """{symbol: weight of equity}: every ETF held, equal inside its class, class budgets across."""
    tickers, _ = _classes(groups, tickers)
    scores = pd.Series(1.0, index=tickers)
    return target_weights(scores, groups=groups, class_weights=class_budgets, top_share=1.0,
                          max_weight=max_weight, exposure=exposure, min_group_assets=1)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_class_budgets.py -v
#
# collected 13 items
#
# test_class_budgets.py::test_size_budgets_follow_the_number_of_etfs_and_sum_to_one PASSED [  7%]
# test_class_budgets.py::test_equal_budgets_give_every_class_the_same_share PASSED [ 15%]
# test_class_budgets.py::test_only_listed_tickers_count PASSED             [ 23%]
# test_class_budgets.py::test_an_unclassified_ticker_is_rejected PASSED    [ 30%]
# test_class_budgets.py::test_class_return_is_the_equal_weight_mean_of_the_members_available_that_day PASSED [ 38%]
# test_class_budgets.py::test_the_calmer_class_gets_the_larger_budget_in_proportion_to_one_over_vol PASSED [ 46%]
# test_class_budgets.py::test_inverse_vol_uses_only_the_trailing_window PASSED [ 53%]
# test_class_budgets.py::test_inverse_vol_needs_a_full_window PASSED       [ 61%]
# test_class_budgets.py::test_non_predictive_weights_hold_every_etf_equal_inside_its_class_with_class_budgets PASSED [ 69%]
# test_class_budgets.py::test_size_budgets_reproduce_the_plain_equal_weight_portfolio PASSED [ 76%]
# test_class_budgets.py::test_exposure_and_cap_pass_through PASSED         [ 84%]
# test_class_budgets.py::test_weights_feed_the_order_generator PASSED      [ 92%]
# test_class_budgets.py::test_a_zero_budget_class_is_left_out PASSED       [100%]
# ============================== 13 passed in 0.30s ==============================
#
# Mutation checks (sandbox): vol not inverted, full-sample vol, forward-filled returns, no window check,
#   top_share 0.2 instead of 1, size budgets replaced by equal, exposure ignored, cap ignored each made a test
#   fail; original restored. NOT yet run on the real snapshot (run_class_budgets step).
# ---------------------------------------------------------------------------
