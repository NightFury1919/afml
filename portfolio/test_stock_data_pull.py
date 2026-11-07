"""Unit tests for stock_data_pull.py. Synthetic data with known answers (allowed for unit tests); the real pull is checked by running it."""
import numpy as np
import pandas as pd
import pytest

import stock_data_pull as sd


def mem():
    df = pd.DataFrame({
        "date": ["2020-01-02", "2020-03-02", "2020-06-01"],
        "tickers": ["AAA,BBB,CCC", "AAA,BBB,DDD", "AAA,BRK.B,RVTY (Previously PKI)"],
    })
    return sd.parse_membership(df)


def test_normalize_ticker_dots_notes_and_case():
    assert sd.normalize_ticker("BRK.B") == "BRK-B"
    assert sd.normalize_ticker("RVTY (Previously PKI)") == "RVTY"
    assert sd.normalize_ticker(" aapl ") == "AAPL"
    assert sd.normalize_ticker("BF-B") == "BF-B"


def test_parse_membership_sets_and_sorted_index():
    m = mem()
    assert list(m.index) == sorted(m.index)
    assert m.iloc[0] == frozenset({"AAA", "BBB", "CCC"})
    assert m.iloc[2] == frozenset({"AAA", "BRK-B", "RVTY"})


def test_parse_membership_keeps_last_row_when_a_date_repeats():
    df = pd.DataFrame({"date": ["2020-01-02", "2020-01-02", "2020-02-03"], "tickers": ["AAA,BBB", "AAA,CCC", "AAA"]})
    m = sd.parse_membership(df)
    assert len(m) == 2
    assert m.iloc[0] == frozenset({"AAA", "CCC"})


def test_all_tickers_is_sorted_union():
    assert sd.all_tickers(mem()) == ["AAA", "BBB", "BRK-B", "CCC", "DDD", "RVTY"]


def test_members_on_uses_last_row_on_or_before_and_empty_before_first():
    m = mem()
    assert sd.members_on(m, "2019-12-31") == frozenset()
    assert sd.members_on(m, "2020-01-02") == frozenset({"AAA", "BBB", "CCC"})
    assert sd.members_on(m, "2020-03-01") == frozenset({"AAA", "BBB", "CCC"})   # the day before the change
    assert sd.members_on(m, "2020-03-02") == frozenset({"AAA", "BBB", "DDD"})   # the change day itself
    assert sd.members_on(m, "2020-04-15") == frozenset({"AAA", "BBB", "DDD"})


def test_month_end_dates_picks_last_trading_day_of_each_month():
    idx = pd.to_datetime(["2020-01-02", "2020-01-30", "2020-01-31", "2020-02-03", "2020-02-27"])
    out = sd.month_end_dates(idx)
    assert list(out) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-27")]


def test_coverage_table_known_shares():
    m = mem()
    idx = pd.to_datetime(["2020-02-28", "2020-04-30"])
    prices = pd.DataFrame({"AAA": [1.0, 1.0], "BBB": [1.0, np.nan], "CCC": [1.0, 1.0],
                           "DDD": [np.nan, np.nan]}, index=idx)
    cov = sd.coverage_table(prices, m)
    # Feb-28: members AAA,BBB,CCC all priced -> 3/3.  Apr-30: members AAA,BBB,DDD; only AAA priced -> 1/3.
    assert cov.loc["2020-02-28", "members"] == 3 and cov.loc["2020-02-28", "with_price"] == 3
    assert cov.loc["2020-02-28", "share"] == pytest.approx(1.0)
    assert cov.loc["2020-04-30", "with_price"] == 1
    assert cov.loc["2020-04-30", "share"] == pytest.approx(1 / 3)


def test_coverage_ignores_dates_after_the_last_membership_row():
    m = mem()
    idx = pd.to_datetime(["2020-05-29", "2021-01-29"])      # last membership row is 2020-06-01
    prices = pd.DataFrame({"AAA": [1.0, 1.0]}, index=idx)
    cov = sd.coverage_table(prices, m)
    assert list(cov.index) == [pd.Timestamp("2020-05-29")]


def test_coverage_table_ignores_prices_for_non_members():
    m = mem()
    idx = pd.to_datetime(["2020-02-28"])
    prices = pd.DataFrame({"AAA": [1.0], "ZZZ": [1.0]}, index=idx)     # ZZZ never a member
    cov = sd.coverage_table(prices, m)
    assert cov.loc["2020-02-28", "with_price"] == 1


def test_coverage_by_year_average_and_target_fraction():
    cov = pd.DataFrame({"share": [1.0, 0.8, 0.95, 0.5]},
                       index=pd.to_datetime(["2020-01-31", "2020-02-28", "2021-01-29", "2021-02-26"]))
    out = sd.coverage_by_year(cov, target=0.9)
    assert out.loc[2020, "avg_share"] == pytest.approx(0.9)
    assert out.loc[2020, "share_of_months_at_target"] == pytest.approx(0.5)
    assert out.loc[2021, "share_of_months_at_target"] == pytest.approx(0.5)
    assert out.loc[2021, "month_ends"] == 2


def test_flag_big_moves_both_directions_and_threshold_is_strict():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    prices = pd.DataFrame({"UP": [10.0, 17.0, 17.0, 17.0],           # +70%
                           "DOWN": [10.0, 10.0, 3.0, 3.0],            # -70%
                           "EDGE": [10.0, 15.0, 15.0, 15.0],          # exactly +50% -> not flagged at threshold 0.5
                           "OK": [10.0, 10.5, 10.4, 10.6]}, index=idx)
    f = sd.flag_big_moves(prices, threshold=0.50)
    assert sorted(f["ticker"]) == ["DOWN", "UP"]
    assert f.loc[f["ticker"] == "UP", "move"].iloc[0] == pytest.approx(0.7)
    assert f.loc[f["ticker"] == "DOWN", "move"].iloc[0] == pytest.approx(-0.7)


def test_flag_big_moves_does_not_bridge_missing_days():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"GAP": [10.0, np.nan, 30.0]}, index=idx)    # price absent on the 3rd: no return across the gap
    assert len(sd.flag_big_moves(prices)) == 0


def test_ticker_report_marks_missing_and_dates():
    m = mem()
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    prices = pd.DataFrame({"AAA": [1.0, 1.0, 1.0], "BBB": [np.nan, 1.0, 1.0]}, index=idx)
    r = sd.ticker_report(prices, m)
    assert r.loc["AAA", "price_rows"] == 3 and r.loc["AAA", "status"] == "ok"
    assert r.loc["BBB", "first_price"] == pd.Timestamp("2020-01-03")
    assert r.loc["CCC", "status"] == "NO PRICES"
    assert r.loc["DDD", "first_member_row"] == pd.Timestamp("2020-03-02")
    assert r.loc["CCC", "last_member_row"] == pd.Timestamp("2020-01-02")


def test_batches_split_sizes():
    assert sd.batches(list("abcde"), 2) == [["a", "b"], ["c", "d"], ["e"]]
    assert sd.batches([], 3) == []


def test_adj_close_frame_handles_multiindex_and_flat_layouts():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    mi = pd.DataFrame(np.arange(8, dtype=float).reshape(2, 4),
                      index=idx, columns=pd.MultiIndex.from_product([["Adj Close", "Close"], ["A", "B"]]))
    out = sd._adj_close_frame(mi, ["A", "B"])
    assert list(out.columns) == ["A", "B"] and out.iloc[0, 0] == 0.0           # Adj Close block, not Close
    flat = pd.DataFrame({"Adj Close": [5.0, 6.0], "Close": [9.0, 9.0]}, index=idx)
    out2 = sd._adj_close_frame(flat, ["ZZ"])
    assert list(out2.columns) == ["ZZ"] and out2["ZZ"].iloc[1] == 6.0
    assert sd._adj_close_frame(pd.DataFrame(), ["A"]).empty


def test_real_membership_file_parses_if_present():
    import os
    p = "sp_500_historical_components.csv"
    if not os.path.exists(p):
        pytest.skip("membership file not downloaded")
    m = sd.parse_membership(pd.read_csv(p))
    assert 1000 < len(sd.all_tickers(m)) < 1300
    assert "BRK-B" in sd.all_tickers(m)
    assert all(" " not in t and "." not in t for t in sd.all_tickers(m))
