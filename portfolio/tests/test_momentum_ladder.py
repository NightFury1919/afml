"""Tests for momentum_ladder.py, written BEFORE the code (TDD).

Every test uses a tiny made-up world with a known answer worked out by hand.
Synthetic data is fine here: these are unit tests, not results.
Nothing in this file touches the real price file.

Run from the portfolio folder in the mlfinlab env:
    python -m pytest test_momentum_ladder.py -v
"""
import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml


# ---------- helpers ----------
def make_prices(rates, n_days, start_rows=None):
    """Prices P_i[t] = (1 + g_i)^t on business days; optional first row where an ETF starts trading."""
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    data = {}
    for name, g in rates.items():
        s = pd.Series((1.0 + g) ** np.arange(n_days), index=idx)
        if start_rows and name in start_rows:
            s.iloc[: start_rows[name]] = np.nan
        data[name] = s
    return pd.DataFrame(data)


# ---------- frozen constants (guard against drifting away from the pre-registration) ----------
def test_defaults_match_preregistration():
    assert ml.LOOKBACK == 273
    assert ml.SKIP == 21
    assert ml.REBALANCE_EVERY == 21
    assert ml.WINDOW == 35
    assert ml.MIN_ELIGIBLE == 20
    assert ml.LAG == 1
    assert ml.COST_BPS == 5.0
    assert ml.FLOOR == -0.10
    assert ml.SLEEVE_N == {"D": 10, "C": 3}
    assert ml.RUNGS == (0.10, 0.25, 0.50, 1.00)


# ---------- momentum score ----------
def test_momentum_score_value_and_warmup():
    # one ETF growing 1% a day: score at t = P[t-skip] / P[t-lookback] - 1 = 1.01 ** (lookback - skip) - 1
    p = make_prices({"A": 0.01}, 120)
    s = ml.momentum_score(p, lookback=30, skip=5)
    assert s["A"].iloc[:30].isna().all()                      # not enough history before row 30
    assert s["A"].iloc[30] == pytest.approx(1.01 ** 25 - 1)
    assert s["A"].iloc[100] == pytest.approx(1.01 ** 25 - 1)


def test_momentum_score_needs_price_today_and_lookback_ago():
    p = make_prices({"A": 0.01, "B": 0.02}, 80)
    p.loc[p.index[60], "A"] = np.nan                          # A has no price on row 60
    s = ml.momentum_score(p, lookback=30, skip=5)
    assert np.isnan(s["A"].iloc[60])                          # not eligible that day
    assert not np.isnan(s["B"].iloc[60])
    p2 = make_prices({"A": 0.01}, 80, start_rows={"A": 10})   # starts trading at row 10
    s2 = ml.momentum_score(p2, lookback=30, skip=5)
    assert s2["A"].iloc[:40].isna().all()                     # needs a price 30 rows earlier: first eligible row 40
    assert not np.isnan(s2["A"].iloc[40])


def test_momentum_score_uses_no_future_prices():
    p = make_prices({"A": 0.01, "B": 0.02}, 100)
    base = ml.momentum_score(p, lookback=30, skip=5)
    p_changed = p.copy()
    p_changed.iloc[70:] = p_changed.iloc[70:] * 5.0           # wild moves after row 69
    changed = ml.momentum_score(p_changed, lookback=30, skip=5)
    pd.testing.assert_frame_equal(base.iloc[:70], changed.iloc[:70])


# ---------- selection ----------
def test_select_top_n_orders_by_score():
    row = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20, "D": -0.05})
    assert ml.select_top_n(row, 2) == ["B", "C"]


def test_select_top_n_ties_broken_by_name():
    row = pd.Series({"C": 0.2, "A": 0.2, "B": 0.2, "D": 0.1})
    assert ml.select_top_n(row, 2) == ["A", "B"]


def test_select_top_n_skips_nan_and_handles_short_universe():
    row = pd.Series({"A": np.nan, "B": 0.1, "C": np.nan})
    assert ml.select_top_n(row, 3) == ["B"]                   # fewer eligible than n: hold what exists
    assert ml.select_top_n(pd.Series({"A": np.nan}), 3) == []


# ---------- start rule ----------
def test_first_signal_position_needs_min_eligible():
    # 19 ETFs trade from row 0, the 20th from row 50. With lookback 273 the 19 are eligible from row 273,
    # the 20th from row 323, so the first row with 20 eligible is 323.
    rates = {f"E{i:02d}": 0.001 for i in range(20)}
    p = make_prices(rates, 400, start_rows={"E19": 50})
    assert ml.first_signal_position(p, lookback=273, min_eligible=20) == 323


def test_first_signal_position_none_when_never_enough():
    p = make_prices({"A": 0.001, "B": 0.001}, 100)
    assert ml.first_signal_position(p, lookback=30, min_eligible=3) is None


# ---------- simulation ----------
# World: ETFs grow at constant daily rates; SPY separately. lookback=30, skip=5, rebalance every 10, lag 1.
RATES = {"HI": 0.010, "MID": 0.005, "LO": 0.001}
G_SPY = 0.002


def world(n_days=140):
    p = make_prices(RATES, n_days)
    spy = make_prices({"SPY": G_SPY}, n_days)["SPY"]
    return p, spy


KW = dict(cost_bps=0.0, every=10, lag=1, lookback=30, skip=5, min_eligible=3)


def test_sleeve_only_earns_best_etf_growth_after_trade_date():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=1.0, **KW)
    t0 = r.trade_dates[0]
    pos0 = list(p.index).index(t0)
    assert pos0 == 30 + 1                                     # first signal row 30, trade one day later
    assert r.holdings[t0] == {"HI": 1.0}
    assert r.value.loc[t0] == pytest.approx(1.0)              # value 1.0 at the first trade date (no cost)
    t_later = p.index[pos0 + 25]
    assert r.value.loc[t_later] == pytest.approx((1 + RATES["HI"]) ** 25)
    assert r.value.index[0] == t0                             # nothing before the first trade


def test_zero_sleeve_equals_spy():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=0.0, **KW)
    t0 = r.trade_dates[0]
    pos0 = list(p.index).index(t0)
    t_later = p.index[pos0 + 33]
    assert r.value.loc[t_later] == pytest.approx((1 + G_SPY) ** 33)


def test_half_and_half_resets_at_each_rebalance():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=0.5, **KW)
    t0, t1, t2 = r.trade_dates[:3]
    pos = [list(p.index).index(t) for t in (t0, t1, t2)]
    assert pos[1] - pos[0] == 10 and pos[2] - pos[1] == 10    # rebalance every 10 rows
    a = (1 + RATES["HI"]) ** 10
    b = (1 + G_SPY) ** 10
    v1 = 0.5 * a + 0.5 * b
    assert r.value.loc[t1] == pytest.approx(v1)
    assert r.value.loc[t2] == pytest.approx(v1 * v1)          # reset to 50/50, so the second block compounds the same


def test_holdings_use_only_data_up_to_signal_date():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=1.0, **KW)
    p_changed = p.copy()
    p_changed.iloc[31:, p_changed.columns.get_loc("LO")] *= 1e6   # LO explodes AFTER the first signal (row 30)
    r2 = ml.simulate(p_changed, spy, n=1, sleeve_share=1.0, **KW)
    assert r2.holdings[r2.trade_dates[0]] == r.holdings[r.trade_dates[0]] == {"HI": 1.0}


def test_equal_weight_across_n_holdings():
    p, spy = world()
    r = ml.simulate(p, spy, n=2, sleeve_share=0.5, **KW)
    h = r.holdings[r.trade_dates[0]]
    assert h == {"HI": 0.25, "MID": 0.25, "SPY": 0.5}


def test_first_purchase_cost_charged_on_traded_value():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=1.0,
                    cost_bps=100.0, every=10, lag=1, lookback=30, skip=5, min_eligible=3)
    t0 = r.trade_dates[0]
    assert r.value.loc[t0] == pytest.approx(0.99)             # 100 bps of 1.0 bought from cash


def test_no_cost_when_rebalance_changes_nothing():
    p, spy = world()
    r = ml.simulate(p, spy, n=1, sleeve_share=1.0,
                    cost_bps=100.0, every=10, lag=1, lookback=30, skip=5, min_eligible=3)
    t0 = r.trade_dates[0]
    pos0 = list(p.index).index(t0)
    t_later = p.index[pos0 + 35]                              # several rebalances later, still holding HI
    assert r.value.loc[t_later] == pytest.approx(0.99 * (1 + RATES["HI"]) ** 35)
    assert r.turnover.loc[r.trade_dates[1]] == pytest.approx(0.0)


def test_switching_holdings_charges_the_turnover():
    # HI is best until row 60, then collapses; MID becomes the best.
    n_days = 140
    p = make_prices({"HI": 0.010, "MID": 0.005, "LO": 0.001}, n_days)
    p.loc[p.index[60:], "HI"] = p["HI"].iloc[59] * (0.97 ** np.arange(1, n_days - 59))
    spy = make_prices({"SPY": G_SPY}, n_days)["SPY"]
    r = ml.simulate(p, spy, n=1, sleeve_share=1.0,
                    cost_bps=50.0, every=10, lag=1, lookback=30, skip=5, min_eligible=3)
    switched = [t for t in r.trade_dates[1:] if r.holdings[t] == {"MID": 1.0}]
    assert switched, "expected a switch from HI to MID"
    t = switched[0]
    # selling all of HI and buying all of MID: |0 - 1| + |1 - 0| = 2.0 of weight traded; cost = 50 bps of that
    assert r.turnover.loc[t] == pytest.approx(2.0)
    assert ml.COST_BPS == 5.0  # default unchanged


# ---------- window statistics ----------
V = pd.Series([1.0, 0.8, 1.0, 1.0, 0.85, 1.3])                 # hand-checked: window = 2


def test_window_outcomes_hand_checked():
    end, touch = ml.window_outcomes(V, window=2, floor=-0.10)
    # starts 0..3. end = V[s+2]/V[s]-1 ; touch = min(V[s..s+2])/V[s]-1 <= floor
    assert list(np.round(end, 10)) == [0.0, 0.25, -0.15, 0.3]
    assert list(touch) == [True, False, True, True]


def test_window_summary_hand_checked():
    s = ml.window_summary(V, window=2, floor=-0.10)
    assert s["n_windows"] == 4
    assert s["worst"] == pytest.approx(-0.15)
    assert s["best"] == pytest.approx(0.30)
    assert s["median"] == pytest.approx(0.0)                   # 'lower' percentile of [-0.15, 0, 0.25, 0.30]
    assert s["share_gt_0"] == pytest.approx(0.5)
    assert s["share_gt_10"] == pytest.approx(0.5)
    assert s["share_gt_20"] == pytest.approx(0.5)
    assert s["share_gt_50"] == pytest.approx(0.0)
    assert s["share_gt_100"] == pytest.approx(0.0)
    assert s["end_below_floor"] == pytest.approx(0.25)
    assert s["touch_floor"] == pytest.approx(0.75)


def test_touch_counts_exactly_at_the_floor():
    # 0.5 / 1.0 - 1 = -0.5 exactly in floating point. "At or below the floor" counts, "ending below" does not.
    v = pd.Series([1.0, 0.5, 1.0, 1.0])
    end, touch = ml.window_outcomes(v, window=2, floor=-0.5)
    assert touch[0]                                            # fell to exactly -50% inside the window
    v2 = pd.Series([1.0, 0.9, 0.5])
    s = ml.window_summary(v2, window=2, floor=-0.5)
    assert s["touch_floor"] == pytest.approx(1.0)             # touched -50% at the end
    assert s["end_below_floor"] == pytest.approx(0.0)         # ending exactly AT the floor is not below it


def test_simulate_defaults_match_preregistration():
    import inspect
    p = inspect.signature(ml.simulate).parameters
    assert p["cost_bps"].default == ml.COST_BPS == 5.0
    assert p["every"].default == ml.REBALANCE_EVERY == 21
    assert p["lag"].default == ml.LAG == 1
    assert p["lookback"].default == ml.LOOKBACK == 273
    assert p["skip"].default == ml.SKIP == 21
    assert p["min_eligible"].default == ml.MIN_ELIGIBLE == 20
    q = inspect.signature(ml.momentum_score).parameters
    assert q["lookback"].default == 273 and q["skip"].default == 21
    w = inspect.signature(ml.window_summary).parameters
    assert w["window"].default == 35 and w["floor"].default == -0.10


def test_window_summary_matches_spy_script_method():
    # Same percentile rule as spy_window_stats.py: index int(q * (n - 1)) on the sorted outcomes.
    rng = np.random.default_rng(7)
    v = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, 400)))
    s = ml.window_summary(v, window=35, floor=-0.10)
    w = sorted(v.iloc[k + 35] / v.iloc[k] - 1 for k in range(len(v) - 35))
    assert s["p05"] == pytest.approx(w[int(0.05 * (len(w) - 1))])
    assert s["p95"] == pytest.approx(w[int(0.95 * (len(w) - 1))])
    assert s["median"] == pytest.approx(w[int(0.5 * (len(w) - 1))])


def test_window_summary_too_short_series():
    with pytest.raises(ValueError):
        ml.window_summary(pd.Series([1.0, 1.1]), window=35, floor=-0.10)


# ---------- non-overlapping period returns and bootstrap ----------
def test_period_returns_every_21_rows():
    v = pd.Series(np.arange(1.0, 1.0 + 64))                    # 64 values: rows 0, 21, 42, 63
    r = ml.period_returns(v, every=21)
    assert list(np.round(r.values, 10)) == [round(22 / 1 - 1, 10), round(43 / 22 - 1, 10), round(64 / 43 - 1, 10)]


def test_block_bootstrap_constant_difference_gives_that_constant():
    d = pd.Series([0.01] * 60)
    lo, hi = ml.block_bootstrap_interval(d, block=4, n_resamples=500, seed=1, level=0.90)
    assert lo == pytest.approx(0.01 * 12)                      # mean period difference x 12 (annualized)
    assert hi == pytest.approx(0.01 * 12)


def test_block_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    d = pd.Series(rng.normal(0.002, 0.03, 120))
    a = ml.block_bootstrap_interval(d, block=4, n_resamples=500, seed=20261028, level=0.90)
    b = ml.block_bootstrap_interval(d, block=4, n_resamples=500, seed=20261028, level=0.90)
    assert a == b
    assert a[0] < a[1]


def test_block_bootstrap_interval_covers_the_true_mean_for_a_clear_edge():
    rng = np.random.default_rng(5)
    d = pd.Series(rng.normal(0.01, 0.01, 200))                 # clear positive edge
    lo, hi = ml.block_bootstrap_interval(d, block=4, n_resamples=1000, seed=11, level=0.90)
    assert lo > 0
    assert lo < d.mean() * 12 < hi
