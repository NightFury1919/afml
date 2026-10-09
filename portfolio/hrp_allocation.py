"""HRP weights for the portfolio, using the repo's Chapter 16 implementation.

The algorithm (AFML Snippets 16.1 to 16.4: tree clustering, quasi-diagonalization, recursive
bisection) already lives in ch16/hrp/hrp.py with its own tests. This file adds only what the
portfolio needs around it:

    trailing_returns   the last `window` daily returns of the listed ETFs, complete or an error
    hrp_weights        validate a returns table, build cov and corr, call ch16's getHRP
    hrp_class_shares   add the weights up by asset class, to compare with class budgets

HRP needs no covariance inversion, so it copes with 48 ETFs on one year of data. It is a RISK
weighting only: it says nothing about which ETFs will earn more. The 252-day window is a
convention, not tuned. Weights can stand alone as the non-predictive portfolio.
"""
import importlib.util
from pathlib import Path

import pandas as pd

DEFAULT_WINDOW = 252
MIN_ROWS = 30
_HRP_FILE = Path(__file__).resolve().parents[1] / "ch16" / "hrp" / "hrp.py"
_module = None


def _ch16():
    """Load ch16/hrp/hrp.py by path (its folder is not a package)."""
    global _module
    if _module is None:
        if not _HRP_FILE.exists():
            raise FileNotFoundError(f"{_HRP_FILE} not found; run from inside the afml repo")
        spec = importlib.util.spec_from_file_location("ch16_hrp", _HRP_FILE)
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
    return _module


def trailing_returns(prices, tickers, window=DEFAULT_WINDOW):
    """Last `window` simple daily returns for `tickers`; raises if any is missing."""
    px = prices[list(tickers)]
    r = px.pct_change(fill_method=None).iloc[1:].iloc[-window:]
    if len(r) < window or r.isna().any().any():
        raise ValueError(f"need {window} complete daily returns for every listed ETF")
    return r


def hrp_weights(returns):
    """HRP weights (Series, sums to 1) from a table of daily returns (rows = days, columns = ETFs)."""
    r = pd.DataFrame(returns)
    if r.shape[1] < 2:
        raise ValueError("need at least two assets")
    if r.shape[0] < MIN_ROWS:
        raise ValueError(f"need at least {MIN_ROWS} rows of returns")
    if r.isna().any().any():
        raise ValueError("returns contain missing values; pass a complete window")
    return _ch16().getHRP(r.cov(), r.corr())


def hrp_class_shares(weights, groups):
    missing = [s for s in weights.index if s not in groups]
    if missing:
        raise ValueError(f"no class for: {missing}")
    return weights.groupby(pd.Series({s: groups[s] for s in weights.index})).sum().to_dict()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, same versions as the mlfinlab env; ch16/hrp/hrp.py loaded from the repo)
# $ cd portfolio ; pytest test_hrp_allocation.py -v
#
# collected 10 items
#
# test_hrp_allocation.py::test_weights_are_positive_sum_to_one_and_keep_labels PASSED [ 10%]
# test_hrp_allocation.py::test_a_near_copy_is_treated_as_one_bet_not_two PASSED [ 20%]
# test_hrp_allocation.py::test_independent_assets_with_different_variances_get_inverse_variance_weights PASSED [ 30%]
# test_hrp_allocation.py::test_missing_returns_are_rejected_not_filled PASSED [ 40%]
# test_hrp_allocation.py::test_too_few_assets_or_rows_are_rejected PASSED  [ 50%]
# test_hrp_allocation.py::test_weights_do_not_depend_on_the_column_order PASSED [ 60%]
# test_hrp_allocation.py::test_trailing_returns_are_the_last_window_of_simple_returns_for_the_listed_tickers PASSED [ 70%]
# test_hrp_allocation.py::test_trailing_returns_reject_an_incomplete_window PASSED [ 80%]
# test_hrp_allocation.py::test_class_shares_add_the_weights_of_each_class PASSED [ 90%]
# test_hrp_allocation.py::test_the_book_function_receives_the_covariance_and_the_correlation_matrix PASSED [100%]
# ============================== 10 passed in 0.35s ==============================
#
# Mutation checks (sandbox): covariance passed as correlation, missing-value check removed, window ignored,
#   forward-filled returns, first (NaN) row kept, window check removed, class shares averaged instead of summed,
#   all columns instead of the listed tickers, row minimum removed each made a test fail; original restored.
# The HRP algorithm itself (Snippets 16.1-16.4) is tested in ch16/hrp/test_hrp.py, not here.
# NOT yet run on the real snapshot: python show_hrp.py
# ---------------------------------------------------------------------------
