"""Live plan builder for the momentum account: today's prices in, one plan file out. Sends NO orders.

Plain English:
  * Every 21 trading days (a "signal day", counted from the account's start date) it ranks the ETFs by the
    pre-registered momentum score, keeps the top n, and splits the account between that sleeve and SPY.
  * On every other day it holds what it has: the plan lists no orders.
  * The plan uses the same file layout as the other accounts (version 2), so execute_plan.py can place it with
    all of its guardrails (equity floor, order ledger, sells before buys). That script does the trading, not this one.
  * The signal code is the tested code in momentum_ladder.py, so the live account follows the same rule as the
    historical test.

Trading note: the plan is made after the close on the signal day and executed the next morning, close to the test's
"trade at the next close" timing, but at the next open: a small difference the test does not capture.
"""
import pandas as pd

import momentum_ladder as ml
from paper_runner_v2 import _trim_overspend
from rebalance import rebalance_orders

NEVER_HELD = ("SPY", "SH")           # SPY is the core holding, SH is an inverse fund


def momentum_weights(prices, n, sleeve_share):
    """Target account weights from the signal at the LAST row of `prices` (data up to today only)."""
    if not 0.0 <= sleeve_share <= 1.0:
        raise ValueError("sleeve_share must be between 0 and 1")
    if n < 1:
        raise ValueError("n must be at least 1")
    etfs = prices.drop(columns=[c for c in NEVER_HELD if c in prices.columns])
    score = ml.momentum_score(etfs).iloc[-1]
    tradable = etfs.iloc[-1].notna()
    row = score[tradable.reindex(score.index).fillna(False)]
    picks = ml.select_top_n(row, n) if sleeve_share > 0 else []
    if not picks:
        return {"SPY": 1.0}
    weights = {k: sleeve_share / len(picks) for k in picks}
    if sleeve_share < 1.0:
        weights["SPY"] = 1.0 - sleeve_share
    return weights


def _start_position(index, start):
    return int(index.searchsorted(pd.Timestamp(start)))


def is_signal_day(index, start, every=ml.REBALANCE_EVERY):
    """True if the last row of `index` is a rebalance signal day: the start day, then every `every` rows."""
    pos, start_pos = len(index) - 1, _start_position(index, start)
    return bool(pos >= start_pos and (pos - start_pos) % every == 0)


def next_signal_date(index, start, every=ml.REBALANCE_EVERY):
    """Date of the next signal day after the last row (business days only: holidays are not known in advance)."""
    pos, start_pos = len(index) - 1, _start_position(index, start)
    if pos < start_pos:
        ahead = start_pos - pos
    else:
        ahead = start_pos + ((pos - start_pos) // every + 1) * every - pos
    return pd.bdate_range(index[-1], periods=ahead + 1)[-1]


def make_momentum_plan(prices, holdings, equity, n, sleeve_share, start, every=ml.REBALANCE_EVERY,
                       min_notional=1.0, today=None, max_stale_days=5):
    """The plan as a plain dict in the layout execute_plan.py reads. Sends nothing."""
    if not equity > 0:
        raise ValueError("equity must be positive")
    last = prices.index[-1]
    if "SPY" not in prices.columns or pd.isna(prices["SPY"].iloc[-1]):
        raise ValueError(f"SPY has no price on {last.date()}")
    holdings = {k: float(v) for k, v in holdings.items()}
    signal = is_signal_day(prices.index, start, every)
    if signal:
        weights = momentum_weights(prices, n, sleeve_share)
        orders, cash_after = rebalance_orders(weights, holdings, equity, mode="fractional", min_notional=min_notional)
        orders, cash_after = _trim_overspend(orders, cash_after, min_notional)
    else:
        weights = {k: v / equity for k, v in holdings.items() if v > 0}
        orders, cash_after = [], round(equity - sum(holdings.values()), 2)

    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    warnings = []
    age = (today - last).days
    if age > max_stale_days:
        warnings.append(f"price data is {age} days old (last date {last.date()})")
    return dict(version=2, as_of=last.date().isoformat(), rebalance_day=signal,
                next_signal=str(next_signal_date(prices.index, start, every).date()),
                protocol=dict(strategy="momentum_ladder", sleeve_n=n, sleeve_share=sleeve_share, every=every,
                              lookback=ml.LOOKBACK, skip=ml.SKIP, start=str(pd.Timestamp(start).date()),
                              mode="fractional", min_notional=min_notional),
                spy_dollars=float(equity * (1.0 - sleeve_share)), qf_dollars=float(equity * sleeve_share),
                weights=weights, equity=float(equity), holdings=holdings, orders=orders,
                cash_after=float(cash_after), warnings=warnings)

# =============================================================================
# TDD RESULTS (2026-10-31)  sandbox: Linux, Python 3.11.18, numpy 1.23.5, pandas 1.5.3
# Command: cd portfolio ; python -m pytest test_momentum_plan.py -v
# Mutation check: 15 deliberate breakages (SPY/SH exclusion, schedule, weights, share maths, eligibility, floors of input
# checks, overspend trim, next-date maths) all caught after adding 4 tests for gaps the first round found.
# Synthetic worlds only; never touched real prices, Alpaca or any account. Real-machine confirmation: DONE 2026-10-31 (Windows block at the bottom).
# =============================================================================
# collecting ... collected 21 items
# test_momentum_plan.py::test_weights_hold_top_n_equal_and_rest_in_spy PASSED [ 4%]
# test_momentum_plan.py::test_spy_gets_the_rest_when_the_share_is_not_one_half PASSED [ 9%]
# test_momentum_plan.py::test_full_sleeve_has_no_spy_and_zero_sleeve_is_all_spy PASSED [ 14%]
# test_momentum_plan.py::test_spy_and_sh_are_never_picked_even_when_they_lead PASSED [ 19%]
# test_momentum_plan.py::test_no_eligible_etf_puts_everything_in_spy PASSED [ 23%]
# test_momentum_plan.py::test_etf_without_a_price_today_is_not_bought PASSED [ 28%]
# test_momentum_plan.py::test_weights_match_the_research_simulation_at_the_same_signal_row PASSED [ 33%]
# test_momentum_plan.py::test_future_rows_do_not_change_todays_weights PASSED [ 38%]
# test_momentum_plan.py::test_sleeve_share_out_of_range_refused PASSED [ 42%]
# test_momentum_plan.py::test_signal_day_every_21_rows_from_the_start_date PASSED [ 47%]
# test_momentum_plan.py::test_start_on_a_non_trading_day_uses_the_next_trading_day PASSED [ 52%]
# test_momentum_plan.py::test_next_signal_date_inside_and_beyond_the_data PASSED [ 57%]
# test_momentum_plan.py::test_signal_day_plan_from_cash_buys_the_target_weights PASSED [ 61%]
# test_momentum_plan.py::test_plan_keys_are_the_ones_execute_plan_reads PASSED [ 66%]
# test_momentum_plan.py::test_non_signal_day_plan_sends_no_orders_and_keeps_holdings PASSED [ 71%]
# test_momentum_plan.py::test_signal_day_with_holdings_sells_the_dropped_name_first_then_buys PASSED [ 76%]
# test_momentum_plan.py::test_nothing_to_trade_gives_an_empty_order_list_on_a_signal_day PASSED [ 80%]
# test_momentum_plan.py::test_equity_must_be_positive PASSED [ 85%]
# test_momentum_plan.py::test_equity_must_be_positive_on_a_quiet_day_too PASSED [ 90%]
# test_momentum_plan.py::test_spy_without_a_price_today_refuses PASSED [ 95%]
# test_momentum_plan.py::test_rounding_never_overspends_the_account PASSED [100%]
# ============================== 21 passed in 0.36s ==============================

# =============================================================================
# REAL-MACHINE RESULT (2026-10-31, Ethan's Windows PC, mlfinlab env), from C:\ws\AFML\portfolio
# Command: python -m pytest test_momentum_plan.py -v
# =============================================================================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# rootdir: C:\ws\AFML\portfolio
# collected 21 items
# test_momentum_plan.py::test_weights_hold_top_n_equal_and_rest_in_spy PASSED [ 4%]
# test_momentum_plan.py::test_spy_gets_the_rest_when_the_share_is_not_one_half PASSED [ 9%]
# test_momentum_plan.py::test_full_sleeve_has_no_spy_and_zero_sleeve_is_all_spy PASSED [ 14%]
# test_momentum_plan.py::test_spy_and_sh_are_never_picked_even_when_they_lead PASSED [ 19%]
# test_momentum_plan.py::test_no_eligible_etf_puts_everything_in_spy PASSED [ 23%]
# test_momentum_plan.py::test_etf_without_a_price_today_is_not_bought PASSED [ 28%]
# test_momentum_plan.py::test_weights_match_the_research_simulation_at_the_same_signal_row PASSED [ 33%]
# test_momentum_plan.py::test_future_rows_do_not_change_todays_weights PASSED [ 38%]
# test_momentum_plan.py::test_sleeve_share_out_of_range_refused PASSED [ 42%]
# test_momentum_plan.py::test_signal_day_every_21_rows_from_the_start_date PASSED [ 47%]
# test_momentum_plan.py::test_start_on_a_non_trading_day_uses_the_next_trading_day PASSED [ 52%]
# test_momentum_plan.py::test_next_signal_date_inside_and_beyond_the_data PASSED [ 57%]
# test_momentum_plan.py::test_signal_day_plan_from_cash_buys_the_target_weights PASSED [ 61%]
# test_momentum_plan.py::test_plan_keys_are_the_ones_execute_plan_reads PASSED [ 66%]
# test_momentum_plan.py::test_non_signal_day_plan_sends_no_orders_and_keeps_holdings PASSED [ 71%]
# test_momentum_plan.py::test_signal_day_with_holdings_sells_the_dropped_name_first_then_buys PASSED [ 76%]
# test_momentum_plan.py::test_nothing_to_trade_gives_an_empty_order_list_on_a_signal_day PASSED [ 80%]
# test_momentum_plan.py::test_equity_must_be_positive PASSED [ 85%]
# test_momentum_plan.py::test_equity_must_be_positive_on_a_quiet_day_too PASSED [ 90%]
# test_momentum_plan.py::test_spy_without_a_price_today_refuses PASSED [ 95%]
# test_momentum_plan.py::test_rounding_never_overspends_the_account PASSED [100%]
# === 21 passed in 1.07s ===
