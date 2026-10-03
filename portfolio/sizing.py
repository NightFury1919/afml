from math import floor

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
            qty = floor(target / prices[sym])
            if qty > 0:
                orders.append({"symbol": sym, "qty": qty})
                spent += qty * prices[sym]
    return orders, round(equity - spent, 2)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-02, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py -v
#
# test_sizing.py::test_fractional_equal_weight                 PASSED [ 33%]
# test_sizing.py::test_whole_share_skips_expensive_etf         PASSED [ 66%]
# test_sizing.py::test_whole_share_buys_one_when_affordable    PASSED [100%]
# ============================== 3 passed in 0.06s ==============================
#
# Notes:
# - First run had 1 failure: the hand-computed expected value was wrong, not
#   the code (B's $500 target buys floor(500/400) = 1 share at $400). The test
#   was changed to B = $600 (skipped, 0 shares) and a second test added for
#   the 1-share case. Expected values are hand-computed (synthetic data is
#   acceptable for TDD only; real-data validation still to do).
# ---------------------------------------------------------------------------
