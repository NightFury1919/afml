from sizing import size_orders, whole_shares


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


def test_whole_share_floating_point_does_not_lose_a_share():
    # 0.29 * 100 is 28.999999999999996 in floating point; the exact answer is 29 shares at $1.
    orders, left = size_orders({"A": 0.29}, 100.0, prices={"A": 1.0}, mode="whole")
    assert orders == [{"symbol": "A", "qty": 29}]
    assert left == 71.0


def test_whole_shares_helper_known_values():
    assert whole_shares(500.0, 100.0) == 5
    assert whole_shares(28.999999999999996, 1.0) == 29
    assert whole_shares(399.99, 400.0) == 0          # a real shortfall still rounds down
    assert whole_shares(0.0, 10.0) == 0

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
