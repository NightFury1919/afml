"""Tests for momentum_daily.py, written BEFORE the code (TDD).

Fake prices in a temporary folder; Alpaca is never touched (the tests pass equity and holdings in directly).
Run from the portfolio folder in the mlfinlab env:
    python -m pytest test_momentum_daily.py -v
"""
import json

import numpy as np
import pandas as pd
import pytest

import momentum_daily as md

END = "2026-10-30"          # a Friday
TODAY = "2026-10-31"        # the Saturday after: lag 0 business days


def write_snapshot(folder, n_days=400, end=END, spy_nan_last=False):
    rates = {"AAA": 0.0010, "BBB": 0.0008, "CCC": 0.0006, "DDD": 0.0004, "EEE": 0.0001}
    idx = pd.bdate_range(end=end, periods=n_days)
    data = {k: (1 + g) ** np.arange(n_days) for k, g in rates.items()}
    data["SPY"] = 1.0003 ** np.arange(n_days)
    data["SH"] = np.ones(n_days)
    df = pd.DataFrame(data, index=idx)
    if spy_nan_last:
        df.loc[idx[-1], "SPY"] = np.nan
    df.index.name = "date"
    df.to_csv(folder / f"prices_daily_asof_{idx[-1].date().isoformat()}.csv")
    return idx


def cfg(**kw):
    c = dict(md.DEFAULTS)
    c.update(sleeve_n=2, sleeve_share=0.5, start=END, equity=1000.0)
    c.update(kw)
    return c


def dirs(tmp_path):
    return tmp_path, tmp_path / "plans", tmp_path / "runs"


# ---------- config ----------
def test_defaults_leave_the_account_inert_and_set_the_500_floor():
    assert md.DEFAULTS["sleeve_n"] is None and md.DEFAULTS["sleeve_share"] is None and md.DEFAULTS["start"] is None
    assert md.DEFAULTS["equity_floor"] == 500.0 and md.DEFAULTS["every"] == 21


def test_missing_config_gives_defaults_and_found_false(tmp_path):
    c, found = md.load_config(tmp_path / "nope.json")
    assert found is False and c == md.DEFAULTS


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"sleeve_nn": 3}))
    with pytest.raises(ValueError, match="sleeve_nn"):
        md.load_config(p)


def test_init_config_writes_once_and_never_overwrites(tmp_path):
    p = tmp_path / "run_config.json"
    md.init_config(p)
    assert json.loads(p.read_text()) == md.DEFAULTS
    with pytest.raises(FileExistsError):
        md.init_config(p)


@pytest.mark.parametrize("bad", [dict(sleeve_n=0), dict(sleeve_n=2.5), dict(sleeve_share=1.2),
                                 dict(sleeve_share=-0.1), dict(sleeve_n=None), dict(sleeve_share=None),
                                 dict(start=None), dict(every=0)])
def test_config_problems_flags_each_bad_setting(bad):
    assert md.config_problems(cfg(**bad))
    assert md.config_problems(cfg()) == []


# ---------- the daily run ----------
def test_unconfigured_account_makes_no_plan_but_still_reports(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, dict(md.DEFAULTS), TODAY, config_found=False)
    assert rep["status"] == "not_configured" and rep["orders_sent"] == 0 and rep["plan"] is None
    assert not list(plans.glob("*.json")) if plans.exists() else True
    assert list(runs.glob("run_*.json")) and list(runs.glob("run_*.md"))


def test_configured_signal_day_saves_a_plan_and_reports_it(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    idx = write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    assert rep["status"] == "ok" and rep["as_of"] == END and rep["orders_sent"] == 0
    saved = json.loads((plans / f"plan_v2_{END}.json").read_text())
    assert saved["version"] == 2 and saved["rebalance_day"] is True and saved["orders"]
    assert rep["plan"]["n_orders"] == len(saved["orders"]) and rep["plan"]["rebalance_day"] is True
    assert rep["plan"]["saved_plan_is_this_run"] is True


def test_equity_comes_from_the_config_unless_given(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(equity=1000.0), TODAY, equity=1040.0)
    assert json.loads((plans / f"plan_v2_{END}.json").read_text())["equity"] == 1040.0
    assert rep["plan"]["equity"] == 1040.0


def test_second_run_never_overwrites_the_saved_plan(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    path = plans / f"plan_v2_{END}.json"
    before = path.read_text()
    rep2 = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    assert path.read_text() == before
    assert rep2["plan"]["saved_plan_is_this_run"] is False
    assert len(list(runs.glob("run_*.json"))) == 2            # a second report with an _r2 suffix


def test_saved_plan_made_with_other_settings_is_flagged(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    md.run_momentum_daily(folder, plans, runs, cfg(sleeve_share=0.5), TODAY)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(sleeve_share=0.25), TODAY)
    assert rep["saved_plan_mismatch"] and "sleeve_share" in rep["saved_plan_mismatch"]
    assert "does not match" in rep["warnings"][0]


def test_stale_prices_block_the_run_and_make_no_plan(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), "2026-11-20")
    assert rep["status"] == "blocked" and rep["plan"] is None
    assert not (plans.exists() and list(plans.glob("*.json")))


def test_missing_spy_price_blocks(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder, spy_nan_last=True)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    assert rep["status"] == "blocked"


def test_no_price_file_blocks(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    assert rep["status"] == "blocked"


def test_below_the_500_floor_is_warned_first_and_flagged(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY, equity=480.0)
    assert rep["below_equity_floor"] is True and rep["equity_floor"] == 500.0
    assert "equity floor" in rep["warnings"][0]
    ok = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY, equity=500.0)       # exactly at the floor
    assert ok["below_equity_floor"] is False


def test_bad_floor_value_is_refused():
    with pytest.raises(ValueError):
        md.run_momentum_daily(".", "p", "r", cfg(equity_floor="x"), TODAY)


def test_report_text_says_no_orders_and_shows_next_signal(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY)
    text = md.render_report(rep)
    assert "No orders were sent" in text and rep["plan"]["next_signal"] in text
    assert "momentum" in text.lower()
    assert (runs / f"run_{END}.md").read_text() == text


def test_a_share_of_exactly_zero_or_one_is_allowed():
    assert md.config_problems(cfg(sleeve_share=0.0)) == []
    assert md.config_problems(cfg(sleeve_share=1.0)) == []


def test_one_dollar_under_the_floor_still_counts_as_below(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY, equity=499.5)
    assert rep["below_equity_floor"] is True


def test_unconfigured_run_does_not_even_read_prices(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    rep = md.run_momentum_daily(folder, plans, runs, dict(md.DEFAULTS), TODAY)
    assert rep["snapshot"] is None and rep["health"] == [] and rep["as_of"] is None
    assert [s["name"] for s in rep["steps"]] == ["plan"]


def test_equity_more_than_five_percent_off_the_config_is_warned(tmp_path):
    folder, plans, runs = dirs(tmp_path)
    write_snapshot(folder)
    far = md.run_momentum_daily(folder, plans, runs, cfg(), TODAY, equity=1100.0)
    assert any("differs from the config equity" in w for w in far["warnings"])
    near = md.run_momentum_daily(folder, tmp_path / "p2", runs, cfg(), TODAY, equity=1040.0)
    assert not any("differs from the config equity" in w for w in near["warnings"])
