import copy

import pytest

from rebalance import rebalance_orders


def buy(sym, notional):
    return {"symbol": sym, "side": "buy", "notional": notional}


def sell(sym, notional):
    return {"symbol": sym, "side": "sell", "notional": notional}


def close(sym):
    return {"symbol": sym, "side": "sell", "close_all": True}


# ------------------------------------------------------------- fractional mode
def test_buys_the_gap_to_target():
    orders, cash = rebalance_orders({"SPY": 0.7}, {"SPY": 5.0}, equity=10.0)
    assert orders == [buy("SPY", 2.0)]
    assert cash == 3.0


def test_sells_the_excess_over_target():
    orders, cash = rebalance_orders({"SPY": 0.5}, {"SPY": 9.0}, equity=10.0)
    assert orders == [sell("SPY", 4.0)]
    assert cash == 5.0


def test_closes_a_position_that_is_not_in_the_target():
    orders, cash = rebalance_orders({"SPY": 1.0}, {"SPY": 4.0, "GLD": 6.0}, equity=10.0)
    assert orders == [close("GLD"), buy("SPY", 6.0)]
    assert cash == 0.0


def test_zero_weight_is_the_same_as_absent():
    a, _ = rebalance_orders({"SPY": 1.0, "GLD": 0.0}, {"SPY": 4.0, "GLD": 6.0}, equity=10.0)
    b, _ = rebalance_orders({"SPY": 1.0}, {"SPY": 4.0, "GLD": 6.0}, equity=10.0)
    assert a == b


def test_opens_a_new_position():
    orders, cash = rebalance_orders({"QQQ": 0.3}, {}, equity=10.0)
    assert orders == [buy("QQQ", 3.0)]
    assert cash == 7.0


def test_gap_smaller_than_the_minimum_order_is_left_alone():
    orders, cash = rebalance_orders({"SPY": 0.55}, {"SPY": 5.0}, equity=10.0)   # gap 0.50 < 1.00
    assert orders == []
    assert cash == 5.0                                                          # still holding the $5


def test_the_minimum_applies_to_sells_too():
    orders, cash = rebalance_orders({"SPY": 0.5}, {"SPY": 5.5}, equity=10.0)
    assert orders == []
    assert cash == 4.5


def test_dust_below_the_minimum_is_not_sold():
    orders, cash = rebalance_orders({"SPY": 0.9}, {"SPY": 9.0, "C": 0.5}, equity=10.0)
    assert orders == []
    assert cash == 0.5


def test_mixed_example_known_values_and_sells_come_first():
    weights = {"A": 0.5, "B": 0.3, "C": 0.0}
    holdings = {"A": 40.0, "B": 35.0, "C": 10.0}
    orders, cash = rebalance_orders(weights, holdings, equity=100.0)
    assert orders == [sell("B", 5.0), close("C"), buy("A", 10.0)]
    assert cash == 20.0


def test_running_again_after_the_trades_does_nothing():
    weights = {"A": 0.5, "B": 0.3}
    after = {"A": 50.0, "B": 30.0}
    orders, cash = rebalance_orders(weights, after, equity=100.0)
    assert orders == []
    assert cash == 20.0


def test_inputs_are_not_modified():
    weights, holdings = {"A": 0.5}, {"A": 40.0, "B": 10.0}
    w0, h0 = copy.deepcopy(weights), copy.deepcopy(holdings)
    rebalance_orders(weights, holdings, equity=100.0)
    assert weights == w0 and holdings == h0


# ------------------------------------------------------------- whole-share mode
PRICES = {"A": 400.0, "B": 25.0}


def test_whole_share_buys_to_the_target_quantity():
    orders, cash = rebalance_orders({"A": 0.5, "B": 0.5}, {"A": 0, "B": 12}, equity=1000.0,
                                    prices=PRICES, mode="whole")
    # targets: A floor(500/400) = 1 share, B floor(500/25) = 20 shares
    assert orders == [{"symbol": "A", "side": "buy", "qty": 1}, {"symbol": "B", "side": "buy", "qty": 8}]
    assert cash == 100.0                                                    # 1000 - (400 + 500)


def test_whole_share_sells_extra_shares_before_buying():
    orders, cash = rebalance_orders({"A": 0.5, "B": 0.5}, {"A": 2, "B": 12}, equity=1000.0,
                                    prices=PRICES, mode="whole")
    assert orders == [{"symbol": "A", "side": "sell", "qty": 1}, {"symbol": "B", "side": "buy", "qty": 8}]
    assert cash == 100.0


def test_whole_share_floating_point_does_not_lose_a_share():
    orders, cash = rebalance_orders({"A": 0.29}, {}, equity=100.0, prices={"A": 1.0}, mode="whole")
    assert orders == [{"symbol": "A", "side": "buy", "qty": 29}]
    assert cash == 71.0


def test_whole_share_exit_sells_a_fractional_leftover_position():
    orders, cash = rebalance_orders({}, {"GLD": 0.013106745}, equity=15.0,
                                    prices={"GLD": 380.72}, mode="whole")
    assert orders == [{"symbol": "GLD", "side": "sell", "qty": 0.013106745}]
    assert cash == 15.0


def test_whole_share_running_again_after_the_trades_does_nothing():
    orders, cash = rebalance_orders({"A": 0.5, "B": 0.5}, {"A": 1, "B": 20}, equity=1000.0,
                                    prices=PRICES, mode="whole")
    assert orders == []
    assert cash == 100.0


# ---------------------------------------------------------------- bad inputs
def test_whole_share_mode_needs_prices():
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.5}, {}, equity=100.0, mode="whole")
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.5, "B": 0.5}, {}, equity=100.0, prices={"A": 10.0}, mode="whole")
    with pytest.raises(ValueError):
        rebalance_orders({}, {"B": 3}, equity=100.0, prices={"A": 10.0}, mode="whole")


def test_weights_must_be_non_negative_and_sum_to_at_most_one():
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.7, "B": 0.4}, {}, equity=100.0)
    with pytest.raises(ValueError):
        rebalance_orders({"A": -0.1}, {}, equity=100.0)


def test_equity_must_be_positive_and_holdings_non_negative_and_mode_known():
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.5}, {}, equity=0.0)
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.5}, {"A": -1.0}, equity=100.0)
    with pytest.raises(ValueError):
        rebalance_orders({"A": 0.5}, {}, equity=100.0, mode="shares")

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_sizing.py test_rebalance.py test_positive_control_etf.py test_etf_classes.py test_etf_features.py test_etf_labels.py test_panel_data.py -v
# Seven test files were run together (88 items); the lines for this file's tests are shown.
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 88 items
#
# test_rebalance.py::test_buys_the_gap_to_target PASSED                    [  6%]
# test_rebalance.py::test_sells_the_excess_over_target PASSED              [  7%]
# test_rebalance.py::test_closes_a_position_that_is_not_in_the_target PASSED [  9%]
# test_rebalance.py::test_zero_weight_is_the_same_as_absent PASSED         [ 10%]
# test_rebalance.py::test_opens_a_new_position PASSED                      [ 11%]
# test_rebalance.py::test_gap_smaller_than_the_minimum_order_is_left_alone PASSED [ 12%]
# test_rebalance.py::test_the_minimum_applies_to_sells_too PASSED          [ 13%]
# test_rebalance.py::test_dust_below_the_minimum_is_not_sold PASSED        [ 14%]
# test_rebalance.py::test_mixed_example_known_values_and_sells_come_first PASSED [ 15%]
# test_rebalance.py::test_running_again_after_the_trades_does_nothing PASSED [ 17%]
# test_rebalance.py::test_inputs_are_not_modified PASSED                   [ 18%]
# test_rebalance.py::test_whole_share_buys_to_the_target_quantity PASSED   [ 19%]
# test_rebalance.py::test_whole_share_sells_extra_shares_before_buying PASSED [ 20%]
# test_rebalance.py::test_whole_share_floating_point_does_not_lose_a_share PASSED [ 21%]
# test_rebalance.py::test_whole_share_exit_sells_a_fractional_leftover_position PASSED [ 22%]
# test_rebalance.py::test_whole_share_running_again_after_the_trades_does_nothing PASSED [ 23%]
# test_rebalance.py::test_whole_share_mode_needs_prices PASSED             [ 25%]
# test_rebalance.py::test_weights_must_be_non_negative_and_sum_to_at_most_one PASSED [ 26%]
# test_rebalance.py::test_equity_must_be_positive_and_holdings_non_negative_and_mode_known PASSED [ 27%]
#
# 88 passed in 17.89s (all seven files)
#
# Mutation check (sandbox): ignoring current holdings, buying before selling, and ignoring the minimum order size on sells each made tests fail; original restored.
# ---------------------------------------------------------------------------
