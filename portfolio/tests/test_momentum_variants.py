"""Tests for momentum_variants.py. Hand-worked numbers on tiny made-up worlds; real prices are never used here.

Small-world setup: lookback=3, skip=1, min_eligible=2, lag=1, ma_len=3. First signal is row 3 (first row with a price now and
3 rows earlier); the trade is at the close of row 4. Score at row t is P[t-1]/P[t-3]-1.
"""
import numpy as np
import pandas as pd
import pytest

import momentum_ladder as ml
import momentum_variants as mv

SMALL = dict(lookback=3, skip=1, min_eligible=2, lag=1)


def world(cols, n=None):
    n = len(next(iter(cols.values())))
    idx = pd.bdate_range("2020-01-01", periods=n)
    spy = pd.Series(cols.pop("SPY", [100.0] * n), index=idx).astype(float)
    return pd.DataFrame(cols, index=idx).astype(float), spy


# ---------- the trend table ----------
def test_trend_ok_table_hand_checked():
    idx = pd.bdate_range("2020-01-01", periods=5)
    p = pd.DataFrame({"A": [100, 110, 120, 90, 95], "B": [100, np.nan, 100, 100, 100]}, index=idx, dtype=float)
    t = mv.trend_ok_table(p, ma_len=3)
    # A row 2: avg (100+110+120)/3 = 110 -> 120 > 110 True; row 3: (110+120+90)/3 = 106.67 -> 90 False; row 4: (120+90+95)/3 = 101.67 -> 95 False
    assert list(t["A"]) == [False, False, True, False, False]       # first two rows have no full window
    # B: the NaN at row 1 poisons every window that contains it (rows 1-3); row 4 window = 100,100,100 -> 100 > 100 False
    assert not t["B"].any()


def test_trend_ok_is_strict_greater_than():
    idx = pd.bdate_range("2020-01-01", periods=4)
    p = pd.DataFrame({"A": [100, 100, 100, 100]}, index=idx, dtype=float)
    assert not mv.trend_ok_table(p, ma_len=3)["A"].any()


# ---------- the trend variant ----------
# A is top ranked at row 3 (+20%) but closes at 90, below its 3-day average of 106.67, so it is not bought: its slice goes to SHY.
FALLING_TOP = dict(A=[100, 110, 120, 90, 92, 95, 95, 95], B=[100] * 8, SHY=[100] * 8)


def test_failed_pick_is_parked_in_shy_hand_checked():
    p, spy = world(dict(FALLING_TOP))
    res = mv.simulate_variant(p, spy, "trend", n=1, sleeve_share=0.5, cost_bps=0.0, every=100, ma_len=3, **SMALL)
    d = p.index[4]
    assert res.filtered == {d: ["A"]}
    assert res.holdings[d] == {"SHY": 0.5, "SPY": 0.5}
    assert all(v == pytest.approx(1.0) for v in res.value)         # SHY and SPY are flat, A is never owned
    base = ml.simulate(p, spy, 1, 0.5, cost_bps=0.0, every=100, **SMALL)
    assert base.value.iloc[-1] == pytest.approx(0.5 + 0.5 * 95 / 92)  # the unfiltered rule did hold A


def test_trend_test_uses_the_signal_day_not_the_trade_day():
    # A fails on the signal day (close 90 < average 106.67) but would pass on the trade day (close 130 > average 113.3).
    cols = dict(A=[100, 110, 120, 90, 130, 130, 130, 130], B=[100] * 8, SHY=[100] * 8)
    p, spy = world(cols)
    res = mv.simulate_variant(p, spy, "trend", n=1, sleeve_share=0.5, cost_bps=0.0, every=100, ma_len=3, **SMALL)
    assert res.holdings[p.index[4]] == {"SHY": 0.5, "SPY": 0.5}
    assert res.filtered == {p.index[4]: ["A"]}


def test_parking_cost_two_legs_worth_of_turnover():
    p, spy = world(dict(FALLING_TOP))
    res = mv.simulate_variant(p, spy, "trend", n=1, sleeve_share=0.5, cost_bps=100.0, every=100, ma_len=3, **SMALL)
    assert res.turnover.iloc[0] == pytest.approx(1.0)                # 0.5 SHY + 0.5 SPY bought from nothing
    assert res.value.iloc[-1] == pytest.approx(0.99)


def test_failed_pick_stays_cash_when_shy_has_no_price():
    d = dict(FALLING_TOP)
    d["SHY"] = [np.nan] * 8
    p, spy = world(d)
    res = mv.simulate_variant(p, spy, "trend", n=1, sleeve_share=0.5, cost_bps=100.0, every=100, ma_len=3, **SMALL)
    # only SPY is bought: traded 0.5, cost 0.005, account 0.995 (SPY 0.4975 + cash 0.4975)
    assert res.holdings[p.index[4]] == {"SPY": 0.5}
    assert res.turnover.iloc[0] == pytest.approx(0.5)
    assert all(v == pytest.approx(0.995) for v in res.value)


def test_two_of_three_picks_pass_only_the_failed_slice_is_parked():
    a = [100, 110, 120, 90, 92, 95, 95, 95]                      # top score, fails the trend test
    b = [100, 101, 102, 103, 104, 105, 106, 107]                  # +2% score, close 103 > avg 102 -> passes
    c = [100, 100.5, 101, 101.5, 102, 103, 104, 105]              # +1% score, close 101.5 > avg 101 -> passes
    p, spy = world(dict(A=a, B=b, C=c, SHY=[100] * 8))
    res = mv.simulate_variant(p, spy, "trend", n=3, sleeve_share=0.6, cost_bps=0.0, every=100, ma_len=3, **SMALL)
    d4 = p.index[4]
    assert res.filtered == {d4: ["A"]}
    want = {"SHY": 0.2, "B": 0.2, "C": 0.2, "SPY": 0.4}
    assert set(res.holdings[d4]) == set(want)
    for k, v in want.items():
        assert res.holdings[d4][k] == pytest.approx(v)


def test_trend_variant_equals_ladder_when_every_pick_passes():
    a = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122]
    b = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    p, spy = world(dict(A=a, B=b, SHY=[100] * 12, SPY=list(np.linspace(100, 108, 12))))
    for n in (1, 2):
        res = mv.simulate_variant(p, spy, "trend", n=n, sleeve_share=0.5, cost_bps=5.0, every=4, ma_len=3, **SMALL)
        base = ml.simulate(p, spy, n, 0.5, cost_bps=5.0, every=4, **SMALL)
        assert res.filtered == {}
        pd.testing.assert_series_equal(res.value, base.value)


def test_mode_none_equals_ladder_exactly_on_a_random_world():
    p, spy = random_world()
    base = ml.simulate(p, spy, 3, 0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    res = mv.simulate_variant(p, spy, None, n=3, sleeve_share=0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    pd.testing.assert_series_equal(res.value, base.value)
    pd.testing.assert_series_equal(res.turnover, base.turnover)
    assert res.holdings == base.holdings


# ---------- the cap variant ----------
CAP_COLS = dict(A=[100, 110, 120, 121, 130, 143, 150], B=[100, 105, 110, 111, 120, 100, 100],
                C=[100, 102, 104, 105, 110, 121, 121], SHY=[100] * 7)
CAP_MAP = {"A": "x", "B": "x", "C": "y", "SHY": "z"}


def test_select_capped_hand_checked():
    s = pd.Series({"A": 0.2, "B": 0.1, "C": 0.04, "SHY": 0.0, "D": np.nan})
    assert mv.select_capped(s, {"A": "x", "B": "x", "C": "y", "SHY": "z", "D": "w"}, 2) == ["A", "C"]
    assert mv.select_capped(s, {"A": "x", "B": "y", "C": "x", "SHY": "z", "D": "w"}, 2) == ["A", "B"]
    assert mv.select_capped(s, {k: "same" for k in s.index}, 3) == ["A"]            # one group: only one fund
    assert mv.select_capped(s, {k: k for k in s.index}, 10) == ["A", "B", "C", "SHY"]  # fewer than n exist


def test_select_capped_ties_are_broken_by_name():
    s = pd.Series({"Z": 0.1, "M": 0.1, "A": 0.1})
    assert mv.select_capped(s, {k: k for k in s.index}, 2) == ["A", "M"]


def test_cap_swaps_in_the_next_group_hand_checked():
    p, spy = world(dict(CAP_COLS))
    res = mv.simulate_variant(p, spy, "cap", CAP_MAP, n=2, sleeve_share=0.5, cost_bps=0.0, every=100, **SMALL)
    d = p.index[4]
    assert res.holdings[d] == {"A": 0.25, "C": 0.25, "SPY": 0.5}           # B (same group as A) is passed over for C
    assert res.skipped == {d: ["B"]}
    # trade at row 4 (A 130, C 110), next close row 5 (A 143, C 121): 0.5 SPY + 0.25*143/130 + 0.25*121/110
    assert res.value.iloc[1] == pytest.approx(0.5 + 0.25 * 143 / 130 + 0.25 * 121 / 110)
    base = ml.simulate(p, spy, 2, 0.5, cost_bps=0.0, every=100, **SMALL)
    assert base.value.iloc[1] == pytest.approx(0.5 + 0.25 * 143 / 130 + 0.25 * 100 / 120)


def test_cap_with_one_group_per_fund_equals_the_ladder():
    p, spy = random_world()
    own = {c: c for c in p.columns}
    base = ml.simulate(p, spy, 3, 0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    res = mv.simulate_variant(p, spy, "cap", own, n=3, sleeve_share=0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    pd.testing.assert_series_equal(res.value, base.value)
    assert res.skipped == {}


def test_cap_never_holds_two_funds_from_one_group_on_a_random_world():
    p, spy = random_world(seed=4)
    groups = {c: ("g%d" % (i % 3)) for i, c in enumerate(p.columns)}
    res = mv.simulate_variant(p, spy, "cap", groups, n=3, sleeve_share=0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5)
    for d, h in res.holdings.items():
        gs = [groups[k] for k in h if k != "SPY"]
        assert len(gs) == len(set(gs))
    assert len(res.skipped) > 0


def test_cap_needs_a_complete_group_map():
    p, spy = world(dict(CAP_COLS))
    with pytest.raises(ValueError, match="missing"):
        mv.simulate_variant(p, spy, "cap", {"A": "x"}, n=2, **SMALL)
    with pytest.raises(ValueError):
        mv.simulate_variant(p, spy, "cap", None, n=2, **SMALL)


def test_bad_mode_is_refused():
    p, spy = world(dict(CAP_COLS))
    with pytest.raises(ValueError):
        mv.simulate_variant(p, spy, "bogus", **SMALL)


# ---------- no look-ahead ----------
def random_world(seed=1, rows=320, cols=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=rows)
    steps = rng.normal(0.0003, 0.015, size=(rows, cols))
    p = pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                     columns=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "SHY"])
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=rows))), index=idx)
    return p, spy


RW = dict(n=3, sleeve_share=0.5, cost_bps=5.0, every=10, lookback=20, skip=5, min_eligible=5, ma_len=30)


@pytest.mark.parametrize("mode", ["trend", "cap"])
def test_no_look_ahead(mode):
    p, spy = random_world(seed=5)
    groups = {c: ("g%d" % (i % 3)) for i, c in enumerate(p.columns)}
    r = 200
    changed = p.copy()
    changed.iloc[r + 1:] = changed.iloc[r + 1:] * 0.5
    spy2 = spy.copy()
    spy2.iloc[r + 1:] = spy2.iloc[r + 1:] * 0.5
    a = mv.simulate_variant(p, spy, mode, groups, **RW)
    b = mv.simulate_variant(changed, spy2, mode, groups, **RW)
    upto = p.index[r]
    pd.testing.assert_series_equal(a.value.loc[:upto], b.value.loc[:upto])


def test_trend_variant_filters_something_and_books_stay_consistent():
    p, spy = random_world(seed=3)
    res = mv.simulate_variant(p, spy, "trend", **RW)
    assert len(res.filtered) > 0
    assert res.value.notna().all() and (res.value > 0).all()
    for d, h in res.holdings.items():
        assert sum(h.values()) <= 1.0 + 1e-12
