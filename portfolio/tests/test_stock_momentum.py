"""Tests for stock_update.py and stock_momentum_daily.py (and the optional parameters added to momentum_daily/momentum_plan).

Made-up stocks only: no Yahoo, no Alpaca, no real account. Run: python -m pytest portfolio -q -k stock_momentum
"""
import json

import numpy as np
import pandas as pd
import pytest

import momentum_daily as md
import momentum_plan as mp
import stock_momentum_daily as smd
import stock_update as su

NOW = pd.Timestamp("2026-11-10 23:00", tz="UTC")           # 18:00 in New York: today's bar counts as settled


def membership(tickers_by_date):
    return pd.Series({pd.Timestamp(d): frozenset(t) for d, t in tickers_by_date.items()}).sort_index()


def fake_panel(tickers, end="2026-11-10", n=330, seed=0, spy_nan_last=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=n)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n))) for t in tickers}
    df = pd.DataFrame(data, index=idx)
    if spy_nan_last:
        df.loc[idx[-1], "SPY"] = np.nan
    df.index.name = "date"
    return df


def fake_fetch_factory(panel):
    calls = []

    def fetch(chunk, start, end):
        calls.append((list(chunk), start, end))
        cols = [c for c in chunk if c in panel.columns]
        return panel[cols].copy() if cols else pd.DataFrame()
    fetch.calls = calls
    return fetch


# ---------- universe and pull ----------
def test_universe_is_the_newest_row_plus_spy_once_and_sorted():
    m = membership({"2025-01-01": ["AAA", "OLD"], "2025-08-23": ["ZZZ", "AAA", "SPY"]})
    assert su.universe_from_membership(m) == ["AAA", "SPY", "ZZZ"]


def test_pull_panel_joins_batches_and_drops_empty_and_duplicate_columns():
    base = fake_panel(["A", "B", "C", "D", "E"], n=20)
    base["E"] = np.nan                                         # a ticker Yahoo returned empty
    fetch = fake_fetch_factory(base)
    out = su.pull_panel(["A", "B", "C", "D", "E"], "2026-01-01", "2026-11-11", fetch, batch_size=2)
    assert len(fetch.calls) == 3                               # batches of 2, 2, 1
    assert list(out.columns) == ["A", "B", "C", "D"]
    assert out.index.is_monotonic_increasing and out.index.name == "date"
    dup = lambda chunk, s, e: pd.concat([base[["A"]], base[["A"]]], axis=1)
    assert list(su.pull_panel(["A", "B"], "s", "e", dup, batch_size=5).columns) == ["A"]


def test_pull_panel_with_every_batch_failing_returns_empty():
    assert len(su.pull_panel(["A", "B"], "s", "e", lambda c, s, e: None)) == 0


# ---------- update ----------
def members(n=400):
    return membership({"2025-08-23": [f"S{i}" for i in range(n)]})


def test_update_writes_a_dated_snapshot_with_spy_and_the_members(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    fetch = fake_fetch_factory(fake_panel(names))
    path, msg = su.update(tmp_path, members(400), now=NOW, fetch_fn=fetch, min_tickers=350)
    assert path is not None and path.name == "prices_daily_asof_2026-11-10.csv"
    got = pd.read_csv(path, index_col=0)
    assert got.shape[1] == 401 and "SPY" in got.columns
    assert "401 tickers" in msg


def test_update_start_date_is_480_days_back_and_end_is_tomorrow(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    fetch = fake_fetch_factory(fake_panel(names))
    su.update(tmp_path, members(400), now=NOW, fetch_fn=fetch, min_tickers=350)
    _, start, end = fetch.calls[0]
    assert start == "2025-07-18" and end == "2026-11-11"       # 2026-11-10 minus 480 days; plus one day


def test_update_drops_an_unsettled_bar_before_5pm_new_york(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    early = pd.Timestamp("2026-11-10 15:00", tz="UTC")         # 10:00 in New York
    path, _ = su.update(tmp_path, members(400), now=early, fetch_fn=fake_fetch_factory(fake_panel(names)), min_tickers=350)
    assert path.name == "prices_daily_asof_2026-11-09.csv"


def test_update_refuses_a_thin_panel_and_writes_nothing(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    panel = fake_panel(names)
    panel.iloc[-1, 100:400] = np.nan                          # most stocks missing on the last day (SPY kept)
    path, msg = su.update(tmp_path, members(400), now=NOW, fetch_fn=fake_fetch_factory(panel), min_tickers=350)
    assert path is None and "need 350" in msg and not list(tmp_path.iterdir())


def test_update_refuses_when_spy_is_missing_on_the_last_day(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    path, msg = su.update(tmp_path, members(400), now=NOW,
                          fetch_fn=fake_fetch_factory(fake_panel(names, spy_nan_last=True)), min_tickers=350)
    assert path is None and "SPY" in msg


def test_update_with_no_data_writes_nothing(tmp_path):
    path, msg = su.update(tmp_path, members(400), now=NOW, fetch_fn=lambda c, s, e: None, min_tickers=350)
    assert path is None and "no prices" in msg


def test_update_never_overwrites_and_says_so(tmp_path):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    fetch = fake_fetch_factory(fake_panel(names))
    first, _ = su.update(tmp_path, members(400), now=NOW, fetch_fn=fetch, min_tickers=350)
    before = first.read_text()
    second, msg = su.update(tmp_path, members(400), now=NOW, fetch_fn=fetch, min_tickers=350)
    assert second is None and "already exists" in msg and first.read_text() == before


def test_prune_keeps_only_the_newest_snapshots(tmp_path):
    for d in ("2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"):
        (tmp_path / f"prices_daily_asof_{d}.csv").write_text("x")
    (tmp_path / "notes.txt").write_text("keep me")
    gone = su.prune_snapshots(tmp_path, keep=2)
    assert gone == ["prices_daily_asof_2026-11-02.csv", "prices_daily_asof_2026-11-03.csv"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt", "prices_daily_asof_2026-11-04.csv",
                                                           "prices_daily_asof_2026-11-05.csv"]


# ---------- health ----------
def test_health_folds_per_ticker_warnings_and_keeps_errors():
    p = fake_panel([f"S{i}" for i in range(400)] + ["SPY"], n=40)
    p.iloc[-1, 5:20] = np.nan                                  # 15 stocks without a price today
    p.iloc[-1, 30] = p.iloc[-2, 30] * 1.5                      # one 50% move
    out = smd.stock_health(p, "2026-11-10")
    codes = [i["code"] for i in out]
    assert codes.count("ticker_stale") == 1 and codes.count("big_move") == 1
    stale = [i for i in out if i["code"] == "ticker_stale"][0]
    assert stale["message"].startswith("15 tickers") and stale["level"] == "warn"
    assert not [i for i in out if i["level"] == "error"]


def test_health_flags_a_thin_universe_as_an_error():
    p = fake_panel([f"S{i}" for i in range(400)] + ["SPY"], n=40)
    p.iloc[-1, 10:] = np.nan
    p.iloc[-1, p.columns.get_loc("SPY")] = 100.0
    out = smd.stock_health(p, "2026-11-10")
    assert [i for i in out if i["code"] == "thin_universe" and i["level"] == "error"]


def test_health_still_catches_stale_data_and_missing_spy():
    p = fake_panel([f"S{i}" for i in range(400)] + ["SPY"], n=40)
    out = smd.stock_health(p, "2026-11-24")                    # two weeks after the last bar
    assert [i for i in out if i["code"] == "stale_data" and i["level"] == "error"]
    p2 = fake_panel([f"S{i}" for i in range(400)] + ["SPY"], n=40, spy_nan_last=True)
    assert [i for i in smd.stock_health(p2, "2026-11-10") if i["code"] == "spy_missing"]


# ---------- the plan and the runner on a stock panel ----------
def stock_world(folder, n_stocks=380, seed=1, end="2026-11-10"):
    names = [f"S{i}" for i in range(n_stocks)] + ["SPY"]
    p = fake_panel(names, end=end, n=330, seed=seed)
    p.to_csv(folder / f"prices_daily_asof_{p.index[-1].date().isoformat()}.csv")
    return p


def cfg(**kw):
    c = dict(md.DEFAULTS)
    c.update(sleeve_n=3, sleeve_share=0.5, start="2026-11-10", equity=1000.0)
    c.update(kw)
    return c


def test_signal_day_plan_holds_top_3_stocks_with_half_in_spy(tmp_path):
    p = stock_world(tmp_path)
    rep = md.run_momentum_daily(tmp_path, tmp_path / "plans", tmp_path / "runs", cfg(), "2026-11-10",
                                health_fn=smd.stock_health, title=smd.TITLE, strategy=smd.STRATEGY)
    assert rep["status"] == "ok"
    plan = json.loads((tmp_path / "plans" / "plan_v2_2026-11-10.json").read_text())
    assert plan["protocol"]["strategy"] == "stock_momentum" and plan["rebalance_day"] is True
    w = plan["weights"]
    assert w["SPY"] == pytest.approx(0.5) and len(w) == 4
    score = p.shift(21).iloc[-1] / p.shift(273).iloc[-1] - 1.0     # the live rule, recomputed here by hand
    top3 = list(score.drop("SPY").sort_values(ascending=False).index[:3])
    assert sorted(k for k in w if k != "SPY") == sorted(top3)
    assert all(w[k] == pytest.approx(1 / 6) for k in top3)
    assert len(plan["orders"]) == 4 and plan["cash_after"] < 1.0


def test_run_report_uses_the_stock_title(tmp_path):
    stock_world(tmp_path)
    rep = md.run_momentum_daily(tmp_path, tmp_path / "plans", tmp_path / "runs", cfg(), "2026-11-10",
                                health_fn=smd.stock_health, title=smd.TITLE, strategy=smd.STRATEGY)
    assert (tmp_path / "runs" / "run_2026-11-10.md").read_text().startswith("# Stock momentum account")
    assert rep["title"] == "Stock momentum account"


def test_a_thin_panel_blocks_the_run_and_makes_no_plan(tmp_path):
    stock_world(tmp_path, n_stocks=100)
    rep = md.run_momentum_daily(tmp_path, tmp_path / "plans", tmp_path / "runs", cfg(), "2026-11-10",
                                health_fn=smd.stock_health, title=smd.TITLE, strategy=smd.STRATEGY)
    assert rep["status"] == "blocked"
    assert not (tmp_path / "plans").exists() or not list((tmp_path / "plans").glob("*.json"))


def test_unconfigured_stock_account_stays_inert_and_reads_no_prices(tmp_path):
    rep = md.run_momentum_daily(tmp_path, tmp_path / "plans", tmp_path / "runs", dict(md.DEFAULTS), "2026-11-10",
                                health_fn=smd.stock_health, title=smd.TITLE, strategy=smd.STRATEGY)
    assert rep["status"] == "not_configured"


# ---------- the defaults of the shared modules are unchanged ----------
def test_defaults_keep_the_etf_momentum_account_unchanged(tmp_path):
    names = ["AAA", "BBB", "CCC", "DDD", "EEE", "SPY", "SH"]
    idx = pd.bdate_range(end="2026-10-30", periods=400)
    rates = {"AAA": 0.0010, "BBB": 0.0008, "CCC": 0.0006, "DDD": 0.0004, "EEE": 0.0001, "SPY": 0.0003, "SH": 0.0}
    df = pd.DataFrame({k: (1 + g) ** np.arange(400) for k, g in rates.items()}, index=idx)
    df.index.name = "date"
    df.to_csv(tmp_path / "prices_daily_asof_2026-10-30.csv")
    c = dict(md.DEFAULTS)
    c.update(sleeve_n=2, sleeve_share=0.5, start="2026-10-30")
    rep = md.run_momentum_daily(tmp_path, tmp_path / "plans", tmp_path / "runs", c, "2026-10-31")
    plan = json.loads((tmp_path / "plans" / "plan_v2_2026-10-30.json").read_text())
    assert plan["protocol"]["strategy"] == "momentum_ladder"
    assert rep["title"] == "Momentum account"
    assert (tmp_path / "runs" / "run_2026-10-30.md").read_text().startswith("# Momentum account")


# ---------- the ticker minimum is exact ----------
def _panel_with_exactly(k_present):
    names = [f"S{i}" for i in range(400)] + ["SPY"]
    p = fake_panel(names)
    p.iloc[-1, k_present - 1:400] = np.nan                    # k_present-1 stocks + SPY have a price on the last day
    assert int(p.iloc[-1].notna().sum()) == k_present
    return p


def test_update_accepts_exactly_the_minimum_and_refuses_one_less(tmp_path):
    p = _panel_with_exactly(350)
    ok, _ = su.update(tmp_path / "a", members(400), now=NOW, fetch_fn=fake_fetch_factory(p), min_tickers=350)
    assert ok is not None
    bad, msg = su.update(tmp_path / "b", members(400), now=NOW, fetch_fn=fake_fetch_factory(p), min_tickers=351)
    assert bad is None and "only 350" in msg


def test_health_accepts_exactly_the_minimum_and_flags_one_less():
    p = _panel_with_exactly(350)
    assert not [i for i in smd.stock_health(p, "2026-11-10", min_tickers=350) if i["code"] == "thin_universe"]
    assert [i for i in smd.stock_health(p, "2026-11-10", min_tickers=351) if i["code"] == "thin_universe"]
