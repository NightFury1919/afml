import numpy as np
import pandas as pd
import pytest

from class_budgets import (class_returns, equal_budgets, inverse_vol_budgets, non_predictive_weights,
                           size_budgets)
from rebalance import rebalance_orders

GROUPS = {"A1": "A", "A2": "A", "A3": "A", "A4": "A", "B1": "B", "B2": "B", "B3": "B", "C1": "C", "C2": "C", "C3": "C"}


def prices(vols, n=400, seed=0):
    """Columns A*, B*, C* with the daily vol given per class; no drift."""
    rng = np.random.default_rng(seed)
    cols = list(GROUPS)
    sig = np.array([vols[GROUPS[c]] for c in cols])
    rets = rng.normal(0, 1, (n, len(cols))) * sig
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), columns=cols, index=pd.bdate_range("2020-01-01", periods=n))
    return px


# ------------------------------------------------------------ size and equal
def test_size_budgets_follow_the_number_of_etfs_and_sum_to_one():
    b = size_budgets(GROUPS)
    assert b == pytest.approx({"A": 0.4, "B": 0.3, "C": 0.3}) and sum(b.values()) == pytest.approx(1)


def test_equal_budgets_give_every_class_the_same_share():
    assert equal_budgets(GROUPS) == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_only_listed_tickers_count():
    b = size_budgets(GROUPS, tickers=["A1", "A2", "B1", "B2"])
    assert b == pytest.approx({"A": 0.5, "B": 0.5})


def test_an_unclassified_ticker_is_rejected():
    with pytest.raises(ValueError):
        size_budgets(GROUPS, tickers=["A1", "ZZZ"])


# ------------------------------------------------------------ class returns
def test_class_return_is_the_equal_weight_mean_of_the_members_available_that_day():
    idx = pd.bdate_range("2020-01-01", periods=4)
    px = pd.DataFrame({"A1": [100, 110, 121, 133.1], "A2": [100, 100, 100, 100], "B1": [50, 50, 50, 50]}, index=idx)
    r = class_returns(px, {"A1": "A", "A2": "A", "B1": "B"})
    assert r["A"].iloc[1] == pytest.approx((0.10 + 0.0) / 2)
    px.loc[idx[2], "A2"] = np.nan                                   # A2 missing: only A1 counts, for both bars around it
    r = class_returns(px, {"A1": "A", "A2": "A", "B1": "B"})
    assert r["A"].iloc[2] == pytest.approx(0.10)


# ------------------------------------------------------------ inverse vol
def test_the_calmer_class_gets_the_larger_budget_in_proportion_to_one_over_vol():
    px = prices({"A": 0.02, "B": 0.01, "C": 0.005})
    b = inverse_vol_budgets(px, GROUPS, window=252)
    assert sum(b.values()) == pytest.approx(1)
    assert b["C"] > b["B"] > b["A"]
    r = class_returns(px, GROUPS).iloc[-252:].std()
    expect = (1 / r) / (1 / r).sum()
    assert b == pytest.approx(expect.to_dict())


def test_inverse_vol_uses_only_the_trailing_window():
    px = prices({"A": 0.02, "B": 0.01, "C": 0.005})
    changed = px.copy()
    changed.iloc[:100] *= np.linspace(1, 3, 100)[:, None]           # alter data far before the window
    a = inverse_vol_budgets(px, GROUPS, window=200)
    b = inverse_vol_budgets(changed.iloc[:], GROUPS, window=200)
    # the 100 altered rows lie outside the last 200 returns (rows 200..399)
    assert a == pytest.approx(b, rel=1e-6)


def test_inverse_vol_needs_a_full_window():
    px = prices({"A": 0.01, "B": 0.01, "C": 0.01}, n=100)
    with pytest.raises(ValueError):
        inverse_vol_budgets(px, GROUPS, window=252)


# ------------------------------------------------------------ the portfolio
def test_non_predictive_weights_hold_every_etf_equal_inside_its_class_with_class_budgets():
    w = non_predictive_weights(GROUPS, {"A": 0.5, "B": 0.25, "C": 0.25})
    assert w["A1"] == pytest.approx(0.125) and w["B1"] == pytest.approx(0.25 / 3) and w["C2"] == pytest.approx(0.25 / 3)
    assert sum(w.values()) == pytest.approx(1) and len(w) == 10


def test_size_budgets_reproduce_the_plain_equal_weight_portfolio():
    w = non_predictive_weights(GROUPS, size_budgets(GROUPS))
    assert all(v == pytest.approx(0.1) for v in w.values())


def test_exposure_and_cap_pass_through():
    w = non_predictive_weights(GROUPS, equal_budgets(GROUPS), exposure=0.5, max_weight=0.05)
    assert sum(w.values()) <= 0.5 + 1e-9 and max(w.values()) <= 0.05 + 1e-12
    assert w["B1"] == pytest.approx(0.05)                        # 0.5 / 3 / 3 = 0.0556 was capped


def test_weights_feed_the_order_generator():
    w = non_predictive_weights(GROUPS, equal_budgets(GROUPS))
    orders, _ = rebalance_orders(w, {}, 1000.0, prices={s: 50.0 for s in w}, mode="fractional")
    assert len(orders) == 10 and all(o["side"] == "buy" for o in orders)


def test_a_zero_budget_class_is_left_out():
    w = non_predictive_weights(GROUPS, {"A": 0.5, "B": 0.5, "C": 0.0})
    assert not any(s.startswith("C") for s in w) and sum(w.values()) == pytest.approx(1)
