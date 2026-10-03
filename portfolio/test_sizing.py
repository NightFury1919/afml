from sizing import size_orders

def test_fractional_equal_weight():
    w = {f"E{i}": 1/48 for i in range(48)}
    orders, left = size_orders(w, 1000)
    assert orders[0]["notional"] == 20.83

def test_whole_share_skips_expensive_etf():
    orders, left = size_orders({"A": 0.5, "B": 0.5}, 1000,
                               prices={"A": 100.0, "B": 600.0}, mode="whole")
    assert orders == [{"symbol": "A", "qty": 5}]
    assert left == 500.0

def test_whole_share_buys_one_when_affordable():
    orders, left = size_orders({"A": 0.5, "B": 0.5}, 1000,
                               prices={"A": 100.0, "B": 400.0}, mode="whole")
    assert orders == [{"symbol": "A", "qty": 5}, {"symbol": "B", "qty": 1}]
    assert left == 100.0

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
