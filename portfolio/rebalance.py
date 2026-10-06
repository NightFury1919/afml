"""Turn target weights and current holdings into the orders that close the gap.

size_orders() (sizing.py) only knows the targets, so running it again would buy on top
of what you already hold. This function compares targets with holdings and sends only
the difference, including sells, which is what daily rebalancing needs.

Inputs
    weights   {symbol: target weight}, long-only, each >= 0, sum <= 1. A symbol that
              is held but missing from weights (or weighted 0) is exited.
    holdings  fractional mode: {symbol: current market value in dollars}
              whole mode:      {symbol: current quantity in shares}
    equity    total account value, cash included
    prices    {symbol: price}; required in whole mode for every symbol that is
              targeted or held
    mode      "fractional" or "whole"
    min_notional  smallest dollar order worth sending (fractional mode). A gap
              smaller than this is left alone, and so is a leftover position below it.

Output: (orders, cash_after)
    Sells come first, then buys, each sorted by symbol, so a sell can free cash for a buy.
    fractional: {"symbol", "side", "notional"}, or {"symbol", "side": "sell", "close_all": True}
                for a full exit (close the whole position instead of guessing a dollar amount)
    whole:      {"symbol", "side", "qty"}
    cash_after  equity minus the value of everything held once the orders are done
                (positions that were left alone count at their current value).

In whole mode, a position with a fractional quantity (left over from fractional
trading) is sold down to the whole-share target, so that sell quantity can be fractional.
"""
from sizing import whole_shares


def _validate(weights, holdings, equity, prices, mode):
    if mode not in ("fractional", "whole"):
        raise ValueError(f"mode must be 'fractional' or 'whole', got {mode!r}")
    if not equity > 0:
        raise ValueError("equity must be positive")
    if any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative (long-only)")
    if sum(weights.values()) > 1 + 1e-9:
        raise ValueError("weights sum to more than 1")
    if any(v < 0 for v in holdings.values()):
        raise ValueError("holdings must be non-negative")
    if mode == "whole":
        needed = {s for s, w in weights.items() if w > 0} | {s for s, v in holdings.items() if v > 0}
        missing = sorted(s for s in needed if not prices or s not in prices)
        if missing:
            raise ValueError(f"whole-share mode needs a price for: {missing}")


def rebalance_orders(weights, holdings, equity, prices=None, mode="fractional", min_notional=1.0):
    _validate(weights, holdings, equity, prices, mode)
    sells, buys, held_value = [], [], 0.0

    for sym in sorted(set(weights) | set(holdings)):
        w = weights.get(sym, 0.0)
        current = holdings.get(sym, 0.0)

        if mode == "fractional":
            target = round(w * equity, 2)
            if target <= 0:
                if current >= min_notional:
                    sells.append({"symbol": sym, "side": "sell", "close_all": True})
                else:
                    held_value += current                      # dust stays where it is
                continue
            diff = round(target - current, 2)
            if diff >= min_notional:
                buys.append({"symbol": sym, "side": "buy", "notional": diff})
                held_value += target
            elif diff <= -min_notional:
                sells.append({"symbol": sym, "side": "sell", "notional": round(-diff, 2)})
                held_value += target
            else:
                held_value += current
        else:
            price = prices[sym] if sym in prices else 0.0
            target_qty = whole_shares(w * equity, price) if w > 0 else 0
            diff = target_qty - current
            if diff > 1e-12:
                buys.append({"symbol": sym, "side": "buy", "qty": diff if diff != int(diff) else int(diff)})
                held_value += target_qty * price
            elif diff < -1e-12:
                qty = -diff
                sells.append({"symbol": sym, "side": "sell", "qty": qty if qty != int(qty) else int(qty)})
                held_value += target_qty * price
            else:
                held_value += current * price

    return sells + buys, round(equity - held_value, 2)

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
