"""Tests for momentum_plan.py, written BEFORE the code (TDD).

Synthetic worlds with hand-checked answers. Nothing here touches the real price file, Alpaca, or any account.
Run from the portfolio folder in the mlfinlab env:
    python -m pytest test_momentum_plan.py -v
"""
import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml
import momentum_plan as mp


def world(n_days=400, rates=None, with_sh=True):
    """ETFs growing at steady daily rates; SPY at 0.03% a day; SH flat."""
    rates = rates or {"AAA": 0.0010, "BBB": 0.0008, "CCC": 0.0006, "DDD": 0.0004, "EEE": 0.0001}
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    data = {k: (1 + g) ** np.arange(n_days) for k, g in rates.items()}
    data["SPY"] = 1.0003 ** np.arange(n_days)
    if with_sh:
        data["SH"] = np.ones(n_days)
    return pd.DataFrame(data, index=idx)


# ---------- the weights ----------
def test_weights_hold_top_n_equal_and_rest_in_spy():
    w = mp.momentum_weights(world(), n=2, sleeve_share=0.5)
    assert w == pytest.approx({"AAA": 0.25, "BBB": 0.25, "SPY": 0.5})


def test_spy_gets_the_rest_when_the_share_is_not_one_half():
    w = mp.momentum_weights(world(), n=2, sleeve_share=0.25)
    assert w == pytest.approx({"AAA": 0.125, "BBB": 0.125, "SPY": 0.75})


def test_full_sleeve_has_no_spy_and_zero_sleeve_is_all_spy():
    assert mp.momentum_weights(world(), 3, 1.0) == pytest.approx({"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3})
    assert mp.momentum_weights(world(), 3, 0.0) == {"SPY": 1.0}


def test_spy_and_sh_are_never_picked_even_when_they_lead():
    px = world(rates={"AAA": 0.0001, "BBB": 0.0001})
    px["SPY"] = 1.01 ** np.arange(len(px))                     # SPY is the strongest by far
    px["SH"] = 1.02 ** np.arange(len(px))
    w = mp.momentum_weights(px, 2, 1.0)
    assert set(w) == {"AAA", "BBB"}


def test_no_eligible_etf_puts_everything_in_spy():
    px = world(n_days=100)                                     # fewer rows than the 273-day lookback
    assert mp.momentum_weights(px, 3, 0.5) == {"SPY": 1.0}


def test_etf_without_a_price_today_is_not_bought():
    px = world()
    px.loc[px.index[-1], "AAA"] = np.nan                       # the leader has no price today
    w = mp.momentum_weights(px, 1, 1.0)
    assert "AAA" not in w and w == pytest.approx({"BBB": 1.0})


def test_weights_match_the_research_simulation_at_the_same_signal_row():
    px = world()
    etfs = px.drop(columns=["SPY", "SH"])
    sim = ml.simulate(etfs, px["SPY"], 2, 0.5, min_eligible=3)
    trade_date = sim.trade_dates[1]
    pos = list(px.index).index(trade_date) - ml.LAG            # the signal row
    live = mp.momentum_weights(px.iloc[: pos + 1], 2, 0.5)
    assert live == pytest.approx(sim.holdings[trade_date])


def test_future_rows_do_not_change_todays_weights():
    px = world()
    a = mp.momentum_weights(px.iloc[:350], 2, 0.5)
    px2 = px.copy()
    px2.iloc[351:, px2.columns.get_loc("EEE")] *= 1e6          # absurd future move
    assert mp.momentum_weights(px2.iloc[:350], 2, 0.5) == a


def test_sleeve_share_out_of_range_refused():
    with pytest.raises(ValueError):
        mp.momentum_weights(world(), 2, 1.5)
    with pytest.raises(ValueError):
        mp.momentum_weights(world(), 0, 0.5)


# ---------- the schedule ----------
def test_signal_day_every_21_rows_from_the_start_date():
    idx = pd.bdate_range("2024-01-01", periods=100)
    start = idx[10]
    assert mp.is_signal_day(idx[:11], start, 21) is True        # the start day itself
    assert mp.is_signal_day(idx[:12], start, 21) is False
    assert mp.is_signal_day(idx[:32], start, 21) is True        # row 31 = 10 + 21
    assert mp.is_signal_day(idx[:31], start, 21) is False
    assert mp.is_signal_day(idx[:10], start, 21) is False       # before the start


def test_start_on_a_non_trading_day_uses_the_next_trading_day():
    idx = pd.bdate_range("2024-01-01", periods=60)
    saturday = pd.Timestamp("2024-01-13")
    first = idx[idx >= saturday][0]
    pos = list(idx).index(first)
    assert mp.is_signal_day(idx[: pos + 1], saturday, 21) is True


def test_next_signal_date_inside_and_beyond_the_data():
    idx = pd.bdate_range("2024-01-01", periods=40)
    start = idx[10]
    # last row 39: next signal at row 52, 13 business days after the last row
    nxt = mp.next_signal_date(idx, start, 21)
    assert nxt == pd.bdate_range(idx[-1], periods=14)[-1]
    # on a signal day itself, the NEXT one is 21 rows later
    nxt2 = mp.next_signal_date(idx[:32], start, 21)
    assert nxt2 == pd.bdate_range(idx[31], periods=22)[-1]


# ---------- the plan ----------
START_ROW = 300


def plan_on(px, row, holdings=None, equity=1000.0, n=2, share=0.5, every=21, start_row=START_ROW):
    return mp.make_momentum_plan(px.iloc[: row + 1], holdings or {}, equity, n=n, sleeve_share=share,
                                 start=px.index[start_row], every=every, min_notional=1.0)


def test_signal_day_plan_from_cash_buys_the_target_weights():
    px = world()
    plan = plan_on(px, START_ROW)
    assert plan["version"] == 2 and plan["as_of"] == px.index[START_ROW].date().isoformat()
    assert plan["rebalance_day"] is True
    assert plan["weights"] == pytest.approx({"AAA": 0.25, "BBB": 0.25, "SPY": 0.5})
    spent = sum(o["notional"] for o in plan["orders"])
    assert spent == pytest.approx(1000.0, abs=0.05) and all(o["side"] == "buy" for o in plan["orders"])
    assert plan["cash_after"] >= 0.0 and plan["equity"] == 1000.0 and plan["holdings"] == {}
    assert plan["protocol"]["strategy"] == "momentum_ladder"
    assert plan["protocol"]["sleeve_n"] == 2 and plan["protocol"]["sleeve_share"] == 0.5
    assert plan["protocol"]["lookback"] == ml.LOOKBACK and plan["protocol"]["every"] == 21


def test_plan_keys_are_the_ones_execute_plan_reads():
    plan = plan_on(world(), START_ROW)
    for key in ("version", "as_of", "equity", "weights", "holdings", "orders", "protocol", "cash_after", "warnings"):
        assert key in plan


def test_non_signal_day_plan_sends_no_orders_and_keeps_holdings():
    px = world()
    held = {"EEE": 500.0, "SPY": 500.0}                       # NOT the target mix: a signal day would trade
    plan = plan_on(px, START_ROW + 5, holdings=held)
    assert plan["rebalance_day"] is False and plan["orders"] == []
    assert plan["holdings"] == held
    assert plan["weights"] == pytest.approx({"EEE": 0.5, "SPY": 0.5})                  # the weights it holds today


def test_signal_day_with_holdings_sells_the_dropped_name_first_then_buys():
    px = world()
    held = {"EEE": 500.0, "SPY": 500.0}                       # EEE is no longer a top-2 name
    plan = plan_on(px, START_ROW, holdings=held)
    sides = [o["side"] for o in plan["orders"]]
    assert sides == sorted(sides, key=lambda s: s != "sell")  # all sells before all buys
    sell = next(o for o in plan["orders"] if o["symbol"] == "EEE")
    assert sell["side"] == "sell" and sell.get("close_all") is True


def test_nothing_to_trade_gives_an_empty_order_list_on_a_signal_day():
    px = world()
    held = {"AAA": 250.0, "BBB": 250.0, "SPY": 500.0}
    assert plan_on(px, START_ROW, holdings=held)["orders"] == []


def test_equity_must_be_positive():
    with pytest.raises(ValueError):
        plan_on(world(), START_ROW, equity=0.0)


def test_equity_must_be_positive_on_a_quiet_day_too():
    with pytest.raises(ValueError):
        plan_on(world(), START_ROW + 5, holdings={"SPY": 500.0}, equity=0.0)


def test_spy_without_a_price_today_refuses():
    px = world()
    px.loc[px.index[START_ROW], "SPY"] = np.nan
    with pytest.raises(ValueError, match="SPY"):
        plan_on(px, START_ROW)


def test_rounding_never_overspends_the_account():
    # 7 equal positions of $142.857... round to $142.86 each = $1,000.02: one cent-level overshoot to trim
    px = world(rates={f"E{i:02d}": 0.0001 * (i + 1) for i in range(12)})
    plan = plan_on(px, START_ROW, n=7, share=1.0, equity=1000.0)
    assert sum(o["notional"] for o in plan["orders"]) <= 1000.0 + 1e-9
    assert plan["cash_after"] == 0.0
