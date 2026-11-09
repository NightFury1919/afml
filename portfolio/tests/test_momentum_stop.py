"""Tests for momentum_stop.py (stop-loss on the momentum sleeve).

Every expected number below is worked by hand from a tiny made-up price world (synthetic data is allowed in
unit tests with known answers). Real prices are never used here.

Small-world setup used in most tests: lookback=3, skip=1, min_eligible=2, lag=1. The first signal is row 3
(the first row where 2 funds have a price now and 3 rows ago). Signal at row 3 -> trade at the close of row 4.
The volatility table is set so that k x sigma x sqrt(21) = 0.10 exactly, i.e. the barrier is 10% below the buy price.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml
import momentum_stop as ms

SMALL = dict(lookback=3, skip=1, min_eligible=2, lag=1)
SIGMA_10 = 0.10 / (1.5 * np.sqrt(21))        # k=1.5, horizon 21 -> a 10% stop distance


def world(a, b=None, shy=None, spy=None):
    n = len(a)
    idx = pd.bdate_range("2020-01-01", periods=n)
    b = [100.0] * n if b is None else b
    shy = [100.0] * n if shy is None else shy
    spy = [100.0] * n if spy is None else spy
    prices = pd.DataFrame({"A": a, "B": b, "SHY": shy}, index=idx).astype(float)
    return prices, pd.Series(spy, index=idx).astype(float)


def vol_table(prices, sigma=SIGMA_10):
    return pd.DataFrame(sigma, index=prices.index, columns=prices.columns)


# A is bought at 108 on row 4 (barrier 97.2), closes 97 on row 7 (stop), is sold at 95 on row 8, then rebounds.
A_STOP = [100, 102, 104, 106, 108, 110, 104, 97, 95, 108, 110, 110, 110]


# ---------- the distance and the constants ----------
def test_constants_match_preregistration():
    assert ms.K == 1.5 and ms.HORIZON == 21 and ms.SPAN0 == 100
    assert ms.PARK == "SHY" and ms.SLEEVE_N == 3 and ms.SLEEVE_SHARE == 0.50


def test_stop_distance_hand_checked():
    # 1.5 x 0.02 x sqrt(21) = 0.03 x 4.58257569... = 0.13747727...
    assert ms.stop_distance(0.02) == pytest.approx(0.03 * 21 ** 0.5)
    assert ms.stop_distance(0.02) == pytest.approx(0.137477, abs=1e-6)
    assert ms.stop_distance(SIGMA_10) == pytest.approx(0.10)
    assert ms.stop_distance(0.02, k=3.0) == pytest.approx(2 * ms.stop_distance(0.02))


# ---------- the stop fires, is executed one close later, at the actual price ----------
def test_stop_triggers_and_sells_at_next_close_actual_price():
    prices, spy = world(A_STOP)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    assert len(res.stops) == 1
    s = res.stops[0]
    assert s["fund"] == "A"
    assert s["entry_row"] == 4 and s["entry_price"] == 108.0
    assert s["barrier"] == pytest.approx(97.2)                 # 108 x 0.90
    assert s["trigger_row"] == 7                                # close 97 <= 97.2
    assert s["sale_row"] == 8 and s["sale_price"] == 95.0       # sold at the actual next close, below the barrier
    assert s["to"] == "SHY"
    # account: half SPY (flat) plus half in A bought at 108. Hand-worked:
    #   row 4: 1.0   row 5: 0.5 + 0.5 x 110/108   row 6: 0.5 + 0.5 x 104/108   row 7: 0.5 + 0.5 x 97/108
    #   row 8: A sold at 95 -> 0.5 + 0.5 x 95/108 = 0.93981..., parked in flat SHY, so it stays there
    v = res.value.values
    assert v[0] == pytest.approx(1.0)
    assert v[1] == pytest.approx(0.5 + 0.5 * 110 / 108)
    assert v[2] == pytest.approx(0.5 + 0.5 * 104 / 108)
    assert v[3] == pytest.approx(0.5 + 0.5 * 97 / 108)
    assert v[4] == pytest.approx(0.5 + 0.5 * 95 / 108)
    assert all(x == pytest.approx(0.5 + 0.5 * 95 / 108) for x in v[4:])


def test_no_rebuy_before_next_rebalance_even_if_fund_rebounds():
    prices, spy = world(A_STOP)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    # A rebounds to 108-110 on rows 9-12 but the money stays parked: the value never moves after row 8
    assert res.value.iloc[-1] == pytest.approx(res.value.iloc[4])
    assert len(res.stops) == 1
    assert list(res.holdings.values())[0] == {"A": 0.5, "SPY": 0.5}


def test_costs_on_a_stop_sale_two_legs_into_shy():
    prices, spy = world(A_STOP)
    c = 0.01                                                    # 100 bps so the numbers are easy to see
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=100.0, every=100, **SMALL)
    # first trade: weights 0 -> A 0.5, SPY 0.5, traded 1.0, cost 1% -> account 0.99; A 0.495, SPY 0.495
    a_at_sale = 0.495 * 95 / 108
    stop_cost = c * 2 * a_at_sale                               # sell leg + buy-SHY leg
    assert res.stops[0]["value_sold"] == pytest.approx(a_at_sale)
    assert res.stop_cost_paid == pytest.approx(stop_cost)
    assert res.cost_paid == pytest.approx(0.01 + stop_cost)
    assert res.value.iloc[-1] == pytest.approx(0.495 + a_at_sale - stop_cost)


def test_proceeds_stay_in_cash_when_shy_has_no_price_and_only_one_leg_is_charged():
    shy = [np.nan] * 10 + [100.0] * 3                           # SHY does not exist yet on row 8
    prices, spy = world(A_STOP, shy=shy)
    c = 0.01
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=100.0, every=100, **SMALL)
    a_at_sale = 0.495 * 95 / 108
    assert res.stops[0]["to"] == "cash"
    assert res.stop_cost_paid == pytest.approx(c * a_at_sale)   # one leg only
    assert res.value.iloc[-1] == pytest.approx(0.495 + a_at_sale * (1 - c))


# ---------- the rebalance takes over ----------
def test_trigger_on_the_day_before_a_rebalance_is_dropped_and_equals_no_stop():
    prices, spy = world(A_STOP + [110, 110, 110])               # 16 rows; signals at rows 3, 7, 11, 15
    kw = dict(n=1, sleeve_share=0.5, cost_bps=0.0, every=4, **SMALL)
    with_stop = ms.simulate_stop(prices, spy, vol_table(prices), **kw)
    baseline = ml.simulate(prices, spy, 1, 0.5, cost_bps=0.0, every=4, **SMALL)
    # trigger at row 7 close; row 8 is a rebalance day (signal row 7), so no stop sale is made
    assert with_stop.stops == []
    assert with_stop.superseded == 1
    pd.testing.assert_series_equal(with_stop.value, baseline.value)


def test_barrier_is_reset_at_every_rebalance_for_a_fund_that_is_bought_again():
    # A keeps ranking first, so it is "bought" again at the row-8 rebalance at 116: barrier 116 x 0.9 = 104.4.
    # The close of 104 on row 10 breaks the NEW barrier (it would not have broken the old one, 108 x 0.9 = 97.2).
    a = [100, 102, 104, 106, 108, 110, 112, 114, 116, 110, 104, 103, 103, 103, 103, 103]
    prices, spy = world(a)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=4, **SMALL)
    assert len(res.stops) == 1
    s = res.stops[0]
    assert s["entry_row"] == 8 and s["entry_price"] == 116.0
    assert s["barrier"] == pytest.approx(104.4)
    assert s["trigger_row"] == 10 and s["sale_row"] == 11 and s["sale_price"] == 103.0


# ---------- more than one fund ----------
def test_two_funds_stopped_on_the_same_day_are_both_sold_next_close():
    a = [100, 102, 104, 106, 108, 110, 104, 97, 95, 95, 95, 95]
    b = [100, 101, 102, 103, 104, 105, 100, 93, 92, 92, 92, 92]   # B barrier 104 x 0.9 = 93.6; row 7 close 93
    prices, spy = world(a, b=b)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=2, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    assert sorted(s["fund"] for s in res.stops) == ["A", "B"]
    assert all(s["trigger_row"] == 7 and s["sale_row"] == 8 for s in res.stops)
    # sleeve funds hold 0.25 each (0.5 / 2): 0.25 x 95/108 and 0.25 x 92/104, plus SPY 0.5
    assert res.value.iloc[-1] == pytest.approx(0.5 + 0.25 * 95 / 108 + 0.25 * 92 / 104)


def test_a_fall_in_spy_never_triggers_a_stop():
    a = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124]
    spy = [100] * 6 + [60] * 7
    prices, spy_s = world(a, spy=spy)
    res = ms.simulate_stop(prices, spy_s, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    assert res.stops == []


def test_trigger_on_the_last_row_has_no_sale():
    prices, spy = world(A_STOP[:8])                             # data ends on the trigger day (row 7)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    assert res.stops == []
    assert res.value.iloc[-1] == pytest.approx(0.5 + 0.5 * 97 / 108)


# ---------- when no stop can be set ----------
def test_no_stop_when_volatility_is_missing_or_zero():
    prices, spy = world(A_STOP)
    baseline = ml.simulate(prices, spy, 1, 0.5, cost_bps=0.0, every=100, **SMALL)
    for bad in (np.nan, 0.0):
        v = vol_table(prices)
        v.loc[:, "A"] = bad
        res = ms.simulate_stop(prices, spy, v, n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
        assert res.stops == []
        pd.testing.assert_series_equal(res.value, baseline.value)


def test_stop_needs_a_volatility_table_unless_it_is_off():
    prices, spy = world(A_STOP)
    with pytest.raises(ValueError):
        ms.simulate_stop(prices, spy, None, n=1, sleeve_share=0.5, **SMALL)
    ms.simulate_stop(prices, spy, None, n=1, sleeve_share=0.5, k=None, **SMALL)   # fine: no stop


# ---------- the stop-off baseline is exactly the ladder ----------
def random_world(seed=1, rows=300, cols=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=rows)
    steps = rng.normal(0.0003, 0.015, size=(rows, cols))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                          columns=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "SHY"])
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=rows))), index=idx)
    return prices, spy


RW = dict(n=3, sleeve_share=0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)


def rw_vol(prices):
    return prices.pct_change().ewm(span=20).std()


def test_stop_off_equals_ladder_simulate_exactly():
    prices, spy = random_world()
    base = ml.simulate(prices, spy, 3, 0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    off = ms.simulate_stop(prices, spy, None, k=None, **RW)
    pd.testing.assert_series_equal(off.value, base.value)
    pd.testing.assert_series_equal(off.turnover, base.turnover)
    assert off.holdings == base.holdings and off.trade_dates == base.trade_dates
    assert off.stops == []


def test_a_stop_that_can_never_trigger_equals_the_baseline():
    prices, spy = random_world()
    base = ml.simulate(prices, spy, 3, 0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    never = ms.simulate_stop(prices, spy, rw_vol(prices), k=1e6, **RW)     # barrier far below zero
    assert never.stops == []
    pd.testing.assert_series_equal(never.value, base.value)


def test_a_real_k_on_a_volatile_world_triggers_some_stops_and_keeps_the_books_consistent():
    prices, spy = random_world(seed=3)
    res = ms.simulate_stop(prices, spy, rw_vol(prices), k=0.5, **RW)         # tight stop so some fire
    assert len(res.stops) > 0
    assert res.value.notna().all() and (res.value > 0).all()
    for s in res.stops:
        assert s["sale_row"] > s["trigger_row"] > s["entry_row"]
        assert s["barrier"] < s["entry_price"]
        assert s["sale_row"] == s["trigger_row"] + 1
    assert res.stop_cost_paid <= res.cost_paid


def test_no_look_ahead_changing_future_prices_does_not_change_the_past():
    prices, spy = random_world(seed=5)
    r = 200
    changed = prices.copy()
    changed.iloc[r + 1:] = changed.iloc[r + 1:] * 0.5
    spy2 = spy.copy()
    spy2.iloc[r + 1:] = spy2.iloc[r + 1:] * 0.5
    a = ms.simulate_stop(prices, spy, rw_vol(prices), k=0.5, **RW)
    b = ms.simulate_stop(changed, spy2, rw_vol(changed), k=0.5, **RW)
    upto = prices.index[r]
    pd.testing.assert_series_equal(a.value.loc[:upto], b.value.loc[:upto])
    assert [s for s in a.stops if s["sale_row"] <= r] == [s for s in b.stops if s["sale_row"] <= r]


# ---------- volatility input ----------
def test_daily_vol_frame_drops_missing_prices_and_realigns():
    idx = pd.bdate_range("2020-01-01", periods=8)
    prices = pd.DataFrame({"X": np.arange(1, 9, dtype=float), "Y": [np.nan, np.nan, np.nan] + [5.0] * 5,
                           "Z": [np.nan] * 7 + [1.0]}, index=idx)
    seen = {}

    def fake(close, span0):
        seen[close.name] = (len(close), span0)
        return pd.Series(span0 / 10000.0, index=close.index)

    v = ms.daily_vol_frame(prices, fake)
    assert seen["X"] == (8, 100) and seen["Y"] == (5, 100)       # span 100 passed; NaN rows dropped first
    assert v["X"].eq(0.01).all()
    assert v["Y"].iloc[:3].isna().all() and v["Y"].iloc[3:].eq(0.01).all()
    assert v["Z"].isna().all() and "Z" not in seen               # fewer than 3 prices: no volatility
    assert list(v.index) == list(idx) and list(v.columns) == ["X", "Y", "Z"]


def test_load_get_daily_vol_finds_the_function_in_a_fake_repo(tmp_path, monkeypatch):
    pkg = tmp_path / "ch03" / "labeling"
    pkg.mkdir(parents=True)
    (tmp_path / "ch03" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "triple_barrier.py").write_text("def get_daily_vol(close, span0=100):\n    return 'stub'\n")
    is_ch03 = lambda m: m == "ch03" or m.startswith("ch03.")
    saved = {m: sys.modules.pop(m) for m in [m for m in sys.modules if is_ch03(m)]}
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        fn = ms.load_get_daily_vol(tmp_path)
        assert fn(None) == "stub"
    finally:
        for m in [m for m in sys.modules if is_ch03(m)]:
            del sys.modules[m]
        sys.modules.update(saved)


def test_load_get_daily_vol_says_clearly_when_it_is_missing(tmp_path):
    with pytest.raises(ImportError):
        ms.load_get_daily_vol(tmp_path, modules=("no_such_module_for_this_test",))


def _find_repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "ch03").is_dir():
            return p
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _find_repo_root()


def real_get_daily_vol():
    try:
        return ms.load_get_daily_vol(REPO_ROOT)
    except ImportError:
        return None


@pytest.mark.skipif(real_get_daily_vol() is None, reason="repo's ch03 get_daily_vol not importable here")
def test_real_chapter3_get_daily_vol_is_causal_and_finite():
    f = real_get_daily_vol()
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    v = f(close, span0=100)
    assert v.iloc[10:].notna().all() and (v.iloc[10:] > 0).all()
    later = close.copy()
    later.iloc[300:] = later.iloc[300:] * 2.0
    v2 = f(later, span0=100)
    # the series starts at row 2, so compare by date: everything before the first changed price (row 300) is identical
    before = idx[299]
    pd.testing.assert_series_equal(v.loc[:before], v2.loc[:before])      # past volatility ignores future prices
    assert (v.loc[idx[300]:] != v2.loc[idx[300]:]).all()                  # and it does react from the changed day on


def test_real_chapter3_get_daily_vol_on_daily_bars_is_about_a_third_above_one_day_volatility():
    """Characterisation, not a bug fix: the book's snippet looks back to the bar BEFORE 'one day earlier', so on daily bars
    most returns span two days. The volatility it reports is therefore about sqrt(2) x too big on Tue-Fri (Mondays are
    one-day). The pre-registration states this and does not change the function (book fidelity)."""
    f = real_get_daily_vol()
    if f is None:
        pytest.skip("repo's ch03 get_daily_vol not importable here")
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2000-01-01", periods=5000)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 5000))), index=idx)
    true_daily = close.pct_change().std()
    ratio = f(close, span0=100).iloc[-2000:].mean() / true_daily
    assert 1.2 < ratio < 1.5


# ---------- descriptive statistics ----------
def test_rebound_stats_hand_checked():
    idx = pd.bdate_range("2020-01-01", periods=12)
    prices = pd.DataFrame({"A": [100, 100, 100, 100, 110, 100, 100, 100, 100, 100, 100, 100],
                           "B": [100, 100, 100, 100, 100, 100, 100, 95, 100, 100, 100, 100]}, index=idx, dtype=float)
    stops = [{"fund": "A", "sale_row": 2, "sale_price": 100.0},     # row 4 is 110: +10% after 2 rows
             {"fund": "B", "sale_row": 5, "sale_price": 100.0},     # row 7 is 95: -5% after 2 rows
             {"fund": "A", "sale_row": 11, "sale_price": 100.0}]    # row 13 does not exist: not measured
    out = ms.stop_rebound_stats(prices, stops, horizon=2)
    assert out["n_stops"] == 3 and out["n_measured"] == 2
    assert out["share_higher"] == pytest.approx(0.5)
    assert out["avg_change"] == pytest.approx(0.025)
    empty = ms.stop_rebound_stats(prices, [], horizon=2)
    assert empty["n_stops"] == 0 and np.isnan(empty["share_higher"])


def test_periods_with_stop_hand_checked():
    idx = pd.bdate_range("2020-01-01", periods=30)
    trade_dates = [idx[4], idx[14], idx[24]]
    stops = [{"sale_date": str(idx[6].date())}, {"sale_date": str(idx[8].date())},    # both in period 1
             {"sale_date": str(idx[26].date())}]                                      # period 3 (the last one)
    n, hit = ms.periods_with_stop(stops, trade_dates, idx[-1])
    assert n == 3 and hit == 2
    assert ms.periods_with_stop([], trade_dates, idx[-1]) == (3, 0)


def test_close_exactly_on_the_barrier_counts_as_a_stop(monkeypatch):
    # force a 25% distance so the barrier is exactly 108 x 0.75 = 81.0 (no rounding), then close at exactly 81.0
    monkeypatch.setattr(ms, "stop_distance", lambda sigma, k=ms.K, horizon=ms.HORIZON: 0.25)
    a = [100, 102, 104, 106, 108, 100, 90, 81, 80, 80, 80]
    prices, spy = world(a)
    res = ms.simulate_stop(prices, spy, vol_table(prices), n=1, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    assert len(res.stops) == 1
    assert res.stops[0]["barrier"] == 81.0
    assert res.stops[0]["trigger_row"] == 7 and res.stops[0]["sale_row"] == 8
