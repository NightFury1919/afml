import numpy as np
import pandas as pd
import pytest

from signed_flow_predictability import (
    MIN_SAME_SIGN_WINDOWS,
    WINDOW_DIRS,
    block_table,
    window_pairs,
    pooled_test,
    check_windows_disjoint,
)

H = pd.Timedelta(hours=1)
T0 = pd.Timestamp("2026-01-01 00:00:00")  # a 4-hour block boundary


def trades(rows):
    """rows: (hours_after_T0, price, volume, is_buyer_maker). Timestamps in microseconds."""
    df = pd.DataFrame(rows, columns=["h", "Price", "Volume", "IsBuyerMaker"])
    df["Timestamp"] = ((T0 + df["h"] * H) - pd.Timestamp("1970-01-01")) // pd.Timedelta(microseconds=1)
    df["TradeID"] = np.arange(len(df))
    df["QuoteVolume"] = df["Price"] * df["Volume"]
    return df[["TradeID", "Price", "Volume", "QuoteVolume", "Timestamp", "IsBuyerMaker"]]


def five_blocks(prices=(100, 100, 110, 99, 99), volumes_by_block=None, per_block=60):
    """Five 4-hour blocks of trades. Blocks 0 and 4 are partial edges and get dropped."""
    rows = []
    for b in range(5):
        for i in range(per_block):
            h = b * 4 + 0.5 + (i / per_block) * 3
            rows.append((h, prices[b], 1.0, bool(i % 2)))
    return trades(rows)


def test_imbalance_known_values():
    # Block 1 (hours 4-8): buy-initiated volume 3, sell-initiated volume 1 -> (3-1)/(3+1) = 0.5
    rows = [(4.1, 100, 1.0, False), (4.2, 100, 1.0, False), (4.3, 100, 1.0, False), (4.4, 100, 1.0, True)]
    rows = [(0.5, 100, 1.0, False)] + rows + [(8.5, 100, 1.0, False), (12.5, 100, 1.0, False)]
    out = block_table(trades(rows), block_hours=4, min_trades=1)
    first = out.iloc[0]
    assert first["buy_vol"] == pytest.approx(3.0)
    assert first["sell_vol"] == pytest.approx(1.0)
    assert first["imbalance"] == pytest.approx(0.5)


def test_buyer_maker_false_means_buy_initiated():
    rows = [(0.5, 100, 1.0, False), (4.1, 100, 2.0, False), (4.2, 100, 1.0, True),
            (8.5, 100, 1.0, False), (12.5, 100, 1.0, False)]
    out = block_table(trades(rows), block_hours=4, min_trades=1)
    assert out.iloc[0]["buy_vol"] == pytest.approx(2.0)
    assert out.iloc[0]["sell_vol"] == pytest.approx(1.0)


def test_partial_first_and_last_blocks_are_dropped():
    out = block_table(five_blocks(), block_hours=4, min_trades=1)
    assert list(out.index) == [T0 + 4 * H, T0 + 8 * H, T0 + 12 * H]


def test_forward_return_known_values():
    out = block_table(five_blocks(), block_hours=4, min_trades=1)
    # closes of blocks 1, 2, 3 are 100, 110, 99
    assert out["fwd_ret"].iloc[0] == pytest.approx(np.log(110 / 100))
    assert out["fwd_ret"].iloc[1] == pytest.approx(np.log(99 / 110))
    assert np.isnan(out["fwd_ret"].iloc[2])  # last kept block has no next block


def test_blocks_with_too_few_trades_are_dropped_and_break_the_pairing():
    df = five_blocks(per_block=60)
    # thin out block 2 (hours 8-12) down to 5 trades
    block2 = (df["Timestamp"] >= (T0 + 8 * H - pd.Timestamp("1970-01-01")) // pd.Timedelta(microseconds=1)) & \
             (df["Timestamp"] < (T0 + 12 * H - pd.Timestamp("1970-01-01")) // pd.Timedelta(microseconds=1))
    keep = pd.concat([df[~block2], df[block2].iloc[:5]])
    out = block_table(keep, block_hours=4, min_trades=50)
    assert list(out.index) == [T0 + 4 * H, T0 + 12 * H]
    assert out["fwd_ret"].isna().all()  # neither kept block has a consecutive next block


def test_imbalance_does_not_use_later_trades():
    base = five_blocks()
    changed = base.copy()
    later = changed["Timestamp"] >= (T0 + 8 * H - pd.Timestamp("1970-01-01")) // pd.Timedelta(microseconds=1)
    changed.loc[later, "IsBuyerMaker"] = False
    changed.loc[later, "Price"] = changed.loc[later, "Price"] * 2
    a = block_table(base, block_hours=4, min_trades=1)
    b = block_table(changed, block_hours=4, min_trades=1)
    block1 = T0 + 4 * H
    assert a.loc[block1, "imbalance"] == pytest.approx(b.loc[block1, "imbalance"])
    assert a.loc[block1, "close"] == pytest.approx(b.loc[block1, "close"])


def test_pairs_drop_rows_without_a_forward_return():
    out = block_table(five_blocks(), block_hours=4, min_trades=1)
    pairs = window_pairs(out)
    assert len(pairs) == 2
    assert list(pairs.columns) == ["imbalance", "fwd_ret"]


def _pairs(x, y):
    return pd.DataFrame({"imbalance": x, "fwd_ret": y})


def test_pooled_test_detects_a_strong_relationship():
    rng = np.random.default_rng(1)
    wins = []
    for _ in range(7):
        x = rng.normal(size=200)
        wins.append(_pairs(x, x + rng.normal(scale=0.5, size=200)))
    res = pooled_test(wins)
    assert res["rho"] > 0.8
    assert res["n"] == 1400
    assert res["n_same_sign"] == 7
    assert res["passes"] is True


def test_rule_rarely_fires_on_pure_noise():
    # A rule with a 5% significance bar and a sign-consistency requirement should pass
    # on noise only a few percent of the time (about 4% in a 2,000-run check).
    rng = np.random.default_rng(12345)
    hits = 0
    runs = 300
    for _ in range(runs):
        wins = [_pairs(rng.normal(size=177), rng.normal(size=177)) for _ in range(7)]
        hits += pooled_test(wins)["passes"]
    assert hits / runs < 0.10


def test_sign_consistency_counts_windows_matching_the_pooled_sign():
    rng = np.random.default_rng(3)
    pos = [_pairs(x, x + rng.normal(scale=0.2, size=100)) for x in (rng.normal(size=100) for _ in range(5))]
    neg = [_pairs(x, -x + rng.normal(scale=3.0, size=100)) for x in (rng.normal(size=100) for _ in range(2))]
    res = pooled_test(pos + neg)
    assert res["rho"] > 0
    assert res["n_same_sign"] == 5


def test_passing_needs_both_significance_and_sign_consistency():
    rng = np.random.default_rng(4)
    # Pooled signal is strong, but it comes from only 4 of 7 windows; the other 3 are opposite.
    pos = [_pairs(x, x + rng.normal(scale=0.2, size=300)) for x in (rng.normal(size=300) for _ in range(4))]
    neg = [_pairs(x, -x + rng.normal(scale=0.3, size=40)) for x in (rng.normal(size=40) for _ in range(3))]
    res = pooled_test(pos + neg)
    assert abs(res["t"]) >= 1.96
    assert res["n_same_sign"] < 5
    assert res["passes"] is False


def test_windows_must_not_overlap():
    a = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"))
    b = (pd.Timestamp("2026-01-30"), pd.Timestamp("2026-03-01"))
    with pytest.raises(ValueError):
        check_windows_disjoint([a, b])


def test_disjoint_windows_pass_in_any_order():
    a = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31"))
    b = (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-03-01"))
    check_windows_disjoint([b, a])


def test_window_list_is_the_seven_disjoint_windows_and_the_rule_needs_five_of_seven():
    # Window 1 (2026-08-25 snapshot, Jul 26 to Aug 25) overlaps window 2 (Jul 11 to Aug 10) by about 15 days,
    # so it is excluded. Windows 9 to 20 are reserved for replication.
    assert len(WINDOW_DIRS) == 7
    assert "kraken_snapshot_720h_2026-08-25" not in WINDOW_DIRS
    assert WINDOW_DIRS[0].startswith("kraken_snapshot_720h_window2_")
    assert MIN_SAME_SIGN_WINDOWS == 5

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-06, mlfinlab env: Python 3.10.20)
# $ cd pipeline\diagnostics ; pytest test_signed_flow_predictability.py -v
#
# collected 14 items
# 14 passed
#
# (Run after Amendment 1: window 1 dropped, rule = same sign in >= 5 of 7 windows.
#  Committed as 9d28bf7 before the real-data run.)
# ---------------------------------------------------------------------------
