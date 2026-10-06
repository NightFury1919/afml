from math import floor


def whole_shares(target_value, price):
    """Whole shares that fit in target_value at price.

    The tiny 1e-9 guards against floating-point error: 0.29 * 100 is stored as
    28.999999999999996, and floor(28.999999999999996 / 1.0) would lose a share.
    A real shortfall (399.99 at $400) is far bigger than 1e-9, so it still rounds down.
    """
    return int(floor(target_value / price + 1e-9))


def size_orders(weights, equity, prices=None, mode="fractional", min_notional=1.0):
    """weights: {symbol: weight}, long-only. Returns (orders, leftover_cash)."""
    orders, spent = [], 0.0
    for sym, w in weights.items():
        target = w * equity
        if mode == "fractional":
            notional = round(target, 2)
            if notional >= min_notional:
                orders.append({"symbol": sym, "notional": notional})
                spent += notional
        else:
            qty = whole_shares(target, prices[sym])
            if qty > 0:
                orders.append({"symbol": sym, "qty": qty})
                spent += qty * prices[sym]
    return orders, round(equity - spent, 2)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_sizing.py::test_fractional_equal_weight PASSED                      [  1%]
# test_sizing.py::test_whole_share_skips_expensive_etf PASSED              [  2%]
# test_sizing.py::test_whole_share_buys_one_when_affordable PASSED         [  3%]
# test_sizing.py::test_whole_share_floating_point_does_not_lose_a_share PASSED [  4%]
# test_sizing.py::test_whole_shares_helper_known_values PASSED             [  5%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): dropping the floating-point guard made the whole-share tests fail; original restored.
# ---------------------------------------------------------------------------
