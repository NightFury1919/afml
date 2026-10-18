import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from daily_run import (account_to_inputs, config_hash, health_check, init_config, load_config,
                       protocol_from_config, render_report, run_daily)

DAY = pd.Timestamp


def frame(n=10, end="2026-10-16", tickers=("SPY", "AAA", "BBB")):
    """Flat prices on business days ending `end` (Friday 2026-10-16 by default)."""
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({t: 100.0 for t in tickers}, index=idx)


def codes(issues, level=None):
    return sorted(i["code"] for i in issues if level is None or i["level"] == level)


# ------------------------------------------------------------------------------ health_check
def test_clean_data_has_no_issues():
    assert health_check(frame(), today="2026-10-16") == []


def test_saturday_with_a_friday_last_date_is_not_stale():
    assert health_check(frame(), today="2026-10-17") == []


def test_monday_with_a_friday_last_date_is_not_stale_because_todays_bar_is_not_settled_yet():
    assert health_check(frame(), today="2026-10-19") == []


def test_lag_is_measured_to_the_last_business_day_so_a_sunday_with_a_thursday_bar_is_one_day_behind_not_two():
    px = frame(end="2026-10-15")                                       # Thursday
    assert health_check(px, today="2026-10-18") == []                  # Sunday: last business day is Friday, lag 1
    assert codes(health_check(px, today="2026-10-19")) == ["stale_data"]   # Monday: lag 2


def test_two_business_days_behind_is_a_warning_and_a_week_behind_is_an_error():
    assert codes(health_check(frame(), today="2026-10-20")) == ["stale_data"]            # Tue, last Fri: lag 2
    assert codes(health_check(frame(), today="2026-10-20"), "warn") == ["stale_data"]
    assert codes(health_check(frame(), today="2026-10-23"), "error") == ["stale_data"]   # Fri, last Fri: lag 5


def test_the_error_threshold_is_adjustable():
    assert codes(health_check(frame(), today="2026-10-20", max_lag=2), "error") == ["stale_data"]


def test_duplicate_dates_and_unsorted_dates_are_errors():
    px = frame()
    dup = pd.concat([px, px.iloc[[-1]]])
    assert codes(health_check(dup, today="2026-10-16"), "error") == ["bad_index"]
    assert codes(health_check(px.iloc[::-1], today="2026-10-16"), "error") == ["bad_index"]


def test_a_non_positive_price_is_an_error_naming_the_ticker():
    px = frame()
    px.iloc[4, 1] = 0.0
    issues = health_check(px, today="2026-10-16")
    bad = [i for i in issues if i["code"] == "bad_price"]
    assert len(bad) == 1 and bad[0]["level"] == "error" and bad[0]["ticker"] == "AAA"


def test_spy_without_a_last_day_price_is_an_error_but_another_ticker_is_only_a_warning():
    px = frame()
    px.iloc[-1, px.columns.get_loc("SPY")] = np.nan
    assert "spy_missing" in codes(health_check(px, today="2026-10-16"), "error")
    px = frame()
    px.iloc[-1, px.columns.get_loc("BBB")] = np.nan
    issues = health_check(px, today="2026-10-16")
    assert codes(issues, "error") == [] and [i["ticker"] for i in issues if i["code"] == "ticker_stale"] == ["BBB"]


def test_a_thirty_percent_one_day_move_is_flagged_but_ten_percent_is_not():
    px = frame()
    px.iloc[6:, 1] = 130.0                                   # AAA +30% on one day
    px.iloc[6:, 2] = 110.0                                   # BBB +10%
    issues = health_check(px, today="2026-10-16")
    moves = [i for i in issues if i["code"] == "big_move"]
    assert len(moves) == 1 and moves[0]["ticker"] == "AAA" and moves[0]["level"] == "warn"
    assert str(px.index[6].date()) in moves[0]["message"]


def test_the_jump_threshold_is_adjustable():
    px = frame()
    px.iloc[6:, 2] = 110.0
    assert [i["ticker"] for i in health_check(px, today="2026-10-16", jump=0.05) if i["code"] == "big_move"] == ["BBB"]


def test_only_the_recent_lookback_is_scanned_for_big_moves():
    px = frame(n=12)
    px.iloc[3:, 1] = 130.0                                   # old jump, row 3
    assert [i for i in health_check(px, today="2026-10-16", lookback=5) if i["code"] == "big_move"] == []
    assert len([i for i in health_check(px, today="2026-10-16", lookback=12) if i["code"] == "big_move"]) == 1


def test_an_empty_frame_is_an_error_not_a_crash():
    assert codes(health_check(frame().iloc[0:0], today="2026-10-16"), "error") == ["no_data"]


# ------------------------------------------------------------------------------------ config
def test_config_hash_is_stable_and_changes_with_every_protocol_field_and_the_start_date():
    base = dict(spy_share=0.5, qf_rule="size", max_weight=None, exposure=1.0, rebalance_every=1, start=None)
    h = config_hash(base)
    assert h == config_hash(dict(reversed(list(base.items()))))
    for key, new in [("spy_share", 0.6), ("qf_rule", "hrp"), ("max_weight", 0.1), ("exposure", 0.9),
                     ("rebalance_every", 5), ("start", "2026-10-19")]:
        assert config_hash({**base, key: new}) != h, key


def test_equity_is_not_part_of_the_hash_because_it_changes_every_day():
    base = dict(spy_share=0.5, qf_rule="size", max_weight=None, exposure=1.0, rebalance_every=1, start=None)
    assert config_hash({**base, "equity": 1000.0}) == config_hash({**base, "equity": 1234.5})


def test_a_missing_config_file_gives_the_defaults_and_says_so(tmp_path):
    cfg, found = load_config(tmp_path / "run_config.json")
    assert not found and cfg["spy_share"] == 0.5 and cfg["qf_rule"] == "size" and cfg["start"] is None


def test_init_config_writes_a_loadable_file_and_never_overwrites(tmp_path):
    path = tmp_path / "run_config.json"
    init_config(path)
    cfg, found = load_config(path)
    assert found and cfg["spy_share"] == 0.5 and cfg["equity"] == 1000.0
    with pytest.raises(FileExistsError):
        init_config(path)


def test_an_unknown_key_in_the_config_is_rejected_so_typos_cannot_silently_change_the_protocol(tmp_path):
    path = tmp_path / "run_config.json"
    path.write_text(json.dumps({"spy_shar": 0.7}), encoding="utf-8")
    with pytest.raises(ValueError, match="spy_shar"):
        load_config(path)


def test_protocol_from_config_applies_the_fields_and_validates_them():
    cfg, _ = load_config(Path("does_not_exist.json"))
    p = protocol_from_config({**cfg, "spy_share": 0.7, "qf_rule": "hrp", "max_weight": 0.2, "exposure": 0.8,
                              "rebalance_every": 5})
    assert (p.spy_share, p.qf_rule, p.max_weight, p.exposure, p.rebalance_every) == (0.7, "hrp", 0.2, 0.8, 5)
    with pytest.raises(ValueError):
        protocol_from_config({**cfg, "spy_share": 1.5})


# ------------------------------------------------------------------------------------ account
def test_account_to_inputs_takes_equity_and_market_values_as_floats():
    account = SimpleNamespace(equity="1012.34", cash="12.34")
    positions = [SimpleNamespace(symbol="SPY", market_value="500.5"), SimpleNamespace(symbol="QQQ", market_value="499.5")]
    equity, holdings = account_to_inputs(account, positions)
    assert equity == 1012.34 and holdings == {"SPY": 500.5, "QQQ": 499.5}


def test_account_with_no_positions_has_empty_holdings():
    assert account_to_inputs(SimpleNamespace(equity="1000", cash="1000"), []) == (1000.0, {})


# ------------------------------------------------------------------------------------ run_daily
def real_shaped(folder, n=300, seed=11):
    """Synthetic prices for the real 48 ETFs plus SH in the pull_prices.py layout (orchestration tests only)."""
    from etf_classes import CLASS_OF
    rng = np.random.default_rng(seed)
    cols = list(CLASS_OF) + ["SH"]
    idx = pd.bdate_range(end="2026-10-16", periods=n)
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, len(cols))), axis=0)), index=idx, columns=cols)
    px.index.name = "date"
    px.to_csv(folder / f"prices_daily_asof_{idx[-1].date().isoformat()}.csv")
    return px


def dirs(tmp_path):
    return dict(folder=tmp_path, plans_dir=tmp_path / "plans", runs_dir=tmp_path / "runs")


CFG = dict(spy_share=0.5, qf_rule="size", max_weight=None, exposure=1.0, rebalance_every=1, start=None, equity=1000.0)


def go(tmp_path, **kw):
    args = dict(dirs(tmp_path), config=CFG, today="2026-10-17", holdings={}, update_fn=None)
    args.update(kw)
    return run_daily(**args)


def test_a_normal_run_writes_a_plan_and_a_run_report_and_sends_nothing(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path)
    assert rep["status"] == "ok" and rep["as_of"] == "2026-10-16"
    assert (tmp_path / "plans" / "plan_v2_2026-10-16.json").exists()
    assert rep["plan"]["n_orders"] > 0 and abs(rep["plan"]["cash_after"]) < 0.005
    assert rep["orders_sent"] == 0
    saved = json.loads(Path(rep["report_path"]).read_text(encoding="utf-8"))
    assert saved["as_of"] == "2026-10-16" and Path(rep["report_path"]).with_suffix(".md").exists()


def test_the_steps_are_recorded_in_order(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path)
    assert [s["name"] for s in rep["steps"]] == ["update_prices", "health_check", "plan", "scoreboard"]
    assert {s["name"]: s["status"] for s in rep["steps"]}["update_prices"] == "skipped"


def test_the_update_step_runs_once_with_the_folder_and_its_result_is_recorded(tmp_path):
    real_shaped(tmp_path)
    calls = []
    rep = go(tmp_path, update_fn=lambda folder: calls.append(Path(folder)) or "added 1 day")
    assert calls == [tmp_path]
    step = rep["steps"][0]
    assert step["name"] == "update_prices" and step["status"] == "ok" and step["detail"] == "added 1 day"


def test_a_failed_update_is_recorded_and_the_run_carries_on_with_the_existing_snapshot(tmp_path):
    real_shaped(tmp_path)

    def boom(folder):
        raise RuntimeError("Yahoo is down")

    rep = go(tmp_path, update_fn=boom)
    step = rep["steps"][0]
    assert step["status"] == "failed" and "Yahoo is down" in step["detail"]
    assert rep["status"] == "ok" and (tmp_path / "plans" / "plan_v2_2026-10-16.json").exists()


def test_a_failed_update_on_stale_data_blocks_the_plan_because_the_health_check_catches_it(tmp_path):
    real_shaped(tmp_path)

    def boom(folder):
        raise RuntimeError("Yahoo is down")

    rep = go(tmp_path, update_fn=boom, today="2026-10-30")
    assert rep["status"] == "blocked" and "stale_data" in [i["code"] for i in rep["health"]]
    assert not (tmp_path / "plans").exists() or not list((tmp_path / "plans").glob("plan_v2_*.json"))


def test_an_error_level_health_issue_blocks_the_plan_and_says_why(tmp_path):
    px = real_shaped(tmp_path)
    bad = px.copy()
    bad.iloc[-1, bad.columns.get_loc("SPY")] = np.nan
    for f in tmp_path.glob("prices_daily_asof_*.csv"):
        f.unlink()
    bad.to_csv(tmp_path / "prices_daily_asof_2026-10-16.csv")
    rep = go(tmp_path)
    assert rep["status"] == "blocked" and rep["plan"] is None
    assert next(s for s in rep["steps"] if s["name"] == "plan")["status"] == "skipped"
    assert "spy_missing" in [i["code"] for i in rep["health"]]
    assert Path(rep["report_path"]).exists()                 # a blocked run still leaves its report


def test_a_prices_file_that_cannot_be_loaded_blocks_the_run_with_the_reason(tmp_path):
    px = real_shaped(tmp_path)
    path = next(tmp_path.glob("prices_daily_asof_*.csv"))
    text = path.read_text().splitlines()
    path.write_text("\n".join(text + [text[-1]]) + "\n")             # duplicate last date
    rep = go(tmp_path)
    assert rep["status"] == "blocked" and rep["plan"] is None
    issue = next(i for i in rep["health"] if i["code"] == "load_failed")
    assert issue["level"] == "error" and "duplicate" in issue["message"]


def test_no_snapshot_at_all_blocks_the_run_instead_of_crashing(tmp_path):
    rep = go(tmp_path)
    assert rep["status"] == "blocked" and "no_data" in [i["code"] for i in rep["health"]]
    assert Path(rep["report_path"]).exists()


def test_running_twice_on_the_same_day_keeps_the_first_plan_and_does_not_overwrite_the_first_report(tmp_path):
    real_shaped(tmp_path)
    first = go(tmp_path)
    plan_path = tmp_path / "plans" / "plan_v2_2026-10-16.json"
    before = plan_path.read_text(encoding="utf-8")
    second = go(tmp_path)
    assert plan_path.read_text(encoding="utf-8") == before
    assert next(s for s in second["steps"] if s["name"] == "plan")["detail"].startswith("plan already exists")
    assert first["report_path"] != second["report_path"] and Path(first["report_path"]).exists()
    assert len(list((tmp_path / "runs").glob("run_2026-10-16*.json"))) == 2


def test_a_changed_protocol_between_runs_is_reported_loudly(tmp_path):
    real_shaped(tmp_path)
    first = go(tmp_path)
    assert first["config_changed"] is False and first["previous_config_hash"] is None
    second = go(tmp_path, config={**CFG, "spy_share": 0.6})
    assert second["config_changed"] is True and second["previous_config_hash"] == first["config_hash"]
    assert any("protocol changed" in w for w in second["warnings"])
    third = go(tmp_path, config={**CFG, "spy_share": 0.6})
    assert third["config_changed"] is False


def test_the_scoreboard_start_in_the_future_is_a_note_not_a_crash(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path, config={**CFG, "start": "2026-10-19"})
    assert rep["scoreboard"] is None
    assert "after the last price date" in next(s for s in rep["steps"] if s["name"] == "scoreboard")["detail"]


def test_a_scoreboard_start_with_prices_reports_the_latest_row(tmp_path):
    px = real_shaped(tmp_path)
    start = px.index[-6].date().isoformat()
    rep = go(tmp_path, config={**CFG, "start": start})
    sb = rep["scoreboard"]
    assert sb["days"] == 6 and sb["start"] == start and sb["end"] == "2026-10-16"
    assert {"spy_value", "qf_value", "qf_minus_spy", "spy_drawdown", "qf_drawdown"} <= set(sb["last_row"])


def test_no_start_date_skips_the_scoreboard(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path)
    assert rep["scoreboard"] is None
    assert next(s for s in rep["steps"] if s["name"] == "scoreboard")["status"] == "skipped"


def test_holdings_and_equity_come_from_the_caller_and_a_second_account_changes_the_orders(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path, holdings={"SPY": 500.0}, config={**CFG, "equity": 2000.0})
    assert rep["plan"]["equity"] == 2000.0
    spy = [o for o in json.loads((tmp_path / "plans" / "plan_v2_2026-10-16.json").read_text())["orders"] if o["symbol"] == "SPY"]
    assert spy and spy[0]["side"] == "buy"


def test_placeholders_still_waiting_on_the_boss_are_listed_in_every_report(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path)
    assert rep["open_placeholders"] == ["max_weight", "spy_share"]
    rep2 = go(tmp_path, config={**CFG, "confirmed": ["spy_share"]})
    assert rep2["open_placeholders"] == ["max_weight"]
    rep3 = go(tmp_path, config={**CFG, "confirmed": ["spy_share", "max_weight"]})
    assert rep3["open_placeholders"] == []


def test_a_missing_config_file_is_flagged_in_the_report(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path, config_found=False)
    assert any("no run_config.json" in w for w in rep["warnings"])


def test_render_report_is_plain_text_that_says_nothing_was_sent_and_lists_issues(tmp_path):
    real_shaped(tmp_path)
    rep = go(tmp_path)
    text = render_report({**rep, "health": [dict(level="warn", code="big_move", ticker="AAA", message="AAA moved 30%")]})
    assert "2026-10-16" in text and "No orders were sent" in text and "AAA moved 30%" in text
    assert "spy_share" in text


# ------------------------------------------------------------------------------------ the CLI
def run_cli(monkeypatch, tmp_path, *extra):
    import daily_run
    monkeypatch.setattr(sys, "argv", ["daily_run.py", "--folder", str(tmp_path), "--plans-dir", str(tmp_path / "plans"),
                                      "--runs-dir", str(tmp_path / "runs"), "--config", str(tmp_path / "run_config.json"),
                                      "--today", "2026-10-17", *extra])
    with pytest.raises(SystemExit) as e:
        daily_run.main()
    return e.value.code


def test_cli_with_no_update_runs_end_to_end_exits_0_and_prints_a_summary(monkeypatch, tmp_path, capsys):
    real_shaped(tmp_path)
    assert run_cli(monkeypatch, tmp_path, "--no-update") == 0
    out = capsys.readouterr().out
    assert "DAILY RUN" in out and "No orders were sent" in out and "no run_config.json" in out


def test_cli_exits_1_when_the_run_is_blocked(monkeypatch, tmp_path, capsys):
    real_shaped(tmp_path)
    assert run_cli(monkeypatch, tmp_path, "--no-update", "--today", "2026-10-30") == 1
    assert "BLOCKED" in capsys.readouterr().out


def test_cli_init_config_writes_the_file_once(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, "--init-config") == 0
    assert (tmp_path / "run_config.json").exists()
    assert run_cli(monkeypatch, tmp_path, "--init-config") == 2           # refuses to overwrite


def test_cli_from_alpaca_uses_the_account_equity_and_positions_read_only(monkeypatch, tmp_path, capsys):
    real_shaped(tmp_path)
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    import types
    log = []

    class FakeClient:
        def __init__(self, key, secret, paper=False):
            log.append(("paper", paper))

        def get_account(self):
            log.append("get_account")
            return SimpleNamespace(equity="2000", cash="2000")

        def get_all_positions(self):
            log.append("get_all_positions")
            return []

        def __getattr__(self, name):
            raise AssertionError(f"orchestrator called client.{name}()")

    for name, attrs in {"alpaca": {}, "alpaca.trading": {}, "alpaca.trading.client": {"TradingClient": FakeClient}}.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
    assert run_cli(monkeypatch, tmp_path, "--no-update", "--from-alpaca") == 0
    assert ("paper", True) in log and "get_account" in log and "get_all_positions" in log
    plan = json.loads((tmp_path / "plans" / "plan_v2_2026-10-16.json").read_text())
    assert plan["equity"] == 2000.0


def test_cli_from_alpaca_without_keys_exits_2_and_does_nothing(monkeypatch, tmp_path, capsys):
    real_shaped(tmp_path)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    assert run_cli(monkeypatch, tmp_path, "--no-update", "--from-alpaca") == 2
    assert not (tmp_path / "plans").exists()


def test_the_module_never_places_orders_and_imports_no_trading_client_at_load():
    src = (Path(__file__).resolve().parent / "daily_run.py").read_text(encoding="utf-8")
    for forbidden in ("MarketOrderRequest", "LimitOrderRequest", "submit_order", "cancel_order", "close_position"):
        assert forbidden not in src
    import importlib
    for m in [m for m in sys.modules if m.startswith("alpaca")]:
        del sys.modules[m]
    importlib.reload(sys.modules["daily_run"])
    assert not [m for m in sys.modules if m.startswith("alpaca")]


# ------------------------------------------------------------------------------ default_update
def test_default_update_appends_the_new_day_saves_a_new_snapshot_and_a_second_call_says_nothing_new(monkeypatch, tmp_path):
    import update_prices
    from daily_run import default_update
    px = real_shaped(tmp_path)
    (tmp_path / "universe.txt").write_text("\n".join(px.columns) + "\n", encoding="utf-8")
    new_day = pd.Timestamp("2026-10-19")
    fresh = pd.concat([px.tail(15), (px.iloc[[-1]] * 1.01).set_axis([new_day])])
    fresh.index.name = "date"
    monkeypatch.setattr(update_prices, "pull_since", lambda tickers, start: fresh)
    monkeypatch.setattr(update_prices, "drop_incomplete_today", lambda f, now=None: f)
    first = default_update(tmp_path)
    assert "added 1 day(s)" in first and "prices_daily_asof_2026-10-19.csv" in first
    assert (tmp_path / "prices_daily_asof_2026-10-19.csv").exists()
    assert (tmp_path / "prices_daily_asof_2026-10-16.csv").exists()          # the old snapshot is kept
    assert default_update(tmp_path).startswith("nothing new")


def test_default_update_rejects_a_fresh_pull_that_disagrees_with_the_history(monkeypatch, tmp_path):
    import update_prices
    from daily_run import default_update
    px = real_shaped(tmp_path)
    (tmp_path / "universe.txt").write_text("\n".join(px.columns) + "\n", encoding="utf-8")
    fresh = px.tail(15).copy()
    fresh["SPY"] = fresh["SPY"] * np.linspace(1.0, 1.3, len(fresh))           # changing factor, not a re-adjustment
    monkeypatch.setattr(update_prices, "pull_since", lambda tickers, start: fresh)
    monkeypatch.setattr(update_prices, "drop_incomplete_today", lambda f, now=None: f)
    with pytest.raises(ValueError, match="SPY"):
        default_update(tmp_path)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# same versions as the mlfinlab env; Yahoo and Alpaca are faked in the tests, nothing here touched either)
# $ cd portfolio ; pytest test_daily_run.py -v
#
# collected 46 items
# test_daily_run.py::test_clean_data_has_no_issues PASSED                  [  2%]
# test_daily_run.py::test_saturday_with_a_friday_last_date_is_not_stale PASSED [  4%]
# test_daily_run.py::test_monday_with_a_friday_last_date_is_not_stale_because_todays_bar_is_not_settled_yet PASSED [  6%]
# test_daily_run.py::test_lag_is_measured_to_the_last_business_day_so_a_sunday_with_a_thursday_bar_is_one_day_behind_not_two PASSED [  8%]
# test_daily_run.py::test_two_business_days_behind_is_a_warning_and_a_week_behind_is_an_error PASSED [ 10%]
# test_daily_run.py::test_the_error_threshold_is_adjustable PASSED         [ 13%]
# test_daily_run.py::test_duplicate_dates_and_unsorted_dates_are_errors PASSED [ 15%]
# test_daily_run.py::test_a_non_positive_price_is_an_error_naming_the_ticker PASSED [ 17%]
# test_daily_run.py::test_spy_without_a_last_day_price_is_an_error_but_another_ticker_is_only_a_warning PASSED [ 19%]
# test_daily_run.py::test_a_thirty_percent_one_day_move_is_flagged_but_ten_percent_is_not PASSED [ 21%]
# test_daily_run.py::test_the_jump_threshold_is_adjustable PASSED          [ 23%]
# test_daily_run.py::test_only_the_recent_lookback_is_scanned_for_big_moves PASSED [ 26%]
# test_daily_run.py::test_an_empty_frame_is_an_error_not_a_crash PASSED    [ 28%]
# test_daily_run.py::test_config_hash_is_stable_and_changes_with_every_protocol_field_and_the_start_date PASSED [ 30%]
# test_daily_run.py::test_equity_is_not_part_of_the_hash_because_it_changes_every_day PASSED [ 32%]
# test_daily_run.py::test_a_missing_config_file_gives_the_defaults_and_says_so PASSED [ 34%]
# test_daily_run.py::test_init_config_writes_a_loadable_file_and_never_overwrites PASSED [ 36%]
# test_daily_run.py::test_an_unknown_key_in_the_config_is_rejected_so_typos_cannot_silently_change_the_protocol PASSED [ 39%]
# test_daily_run.py::test_protocol_from_config_applies_the_fields_and_validates_them PASSED [ 41%]
# test_daily_run.py::test_account_to_inputs_takes_equity_and_market_values_as_floats PASSED [ 43%]
# test_daily_run.py::test_account_with_no_positions_has_empty_holdings PASSED [ 45%]
# test_daily_run.py::test_a_normal_run_writes_a_plan_and_a_run_report_and_sends_nothing PASSED [ 47%]
# test_daily_run.py::test_the_steps_are_recorded_in_order PASSED           [ 50%]
# test_daily_run.py::test_the_update_step_runs_once_with_the_folder_and_its_result_is_recorded PASSED [ 52%]
# test_daily_run.py::test_a_failed_update_is_recorded_and_the_run_carries_on_with_the_existing_snapshot PASSED [ 54%]
# test_daily_run.py::test_a_failed_update_on_stale_data_blocks_the_plan_because_the_health_check_catches_it PASSED [ 56%]
# test_daily_run.py::test_an_error_level_health_issue_blocks_the_plan_and_says_why PASSED [ 58%]
# test_daily_run.py::test_a_prices_file_that_cannot_be_loaded_blocks_the_run_with_the_reason PASSED [ 60%]
# test_daily_run.py::test_no_snapshot_at_all_blocks_the_run_instead_of_crashing PASSED [ 63%]
# test_daily_run.py::test_running_twice_on_the_same_day_keeps_the_first_plan_and_does_not_overwrite_the_first_report PASSED [ 65%]
# test_daily_run.py::test_a_changed_protocol_between_runs_is_reported_loudly PASSED [ 67%]
# test_daily_run.py::test_the_scoreboard_start_in_the_future_is_a_note_not_a_crash PASSED [ 69%]
# test_daily_run.py::test_a_scoreboard_start_with_prices_reports_the_latest_row PASSED [ 71%]
# test_daily_run.py::test_no_start_date_skips_the_scoreboard PASSED        [ 73%]
# test_daily_run.py::test_holdings_and_equity_come_from_the_caller_and_a_second_account_changes_the_orders PASSED [ 76%]
# test_daily_run.py::test_placeholders_still_waiting_on_the_boss_are_listed_in_every_report PASSED [ 78%]
# test_daily_run.py::test_a_missing_config_file_is_flagged_in_the_report PASSED [ 80%]
# test_daily_run.py::test_render_report_is_plain_text_that_says_nothing_was_sent_and_lists_issues PASSED [ 82%]
# test_daily_run.py::test_cli_with_no_update_runs_end_to_end_exits_0_and_prints_a_summary PASSED [ 84%]
# test_daily_run.py::test_cli_exits_1_when_the_run_is_blocked PASSED       [ 86%]
# test_daily_run.py::test_cli_init_config_writes_the_file_once PASSED      [ 89%]
# test_daily_run.py::test_cli_from_alpaca_uses_the_account_equity_and_positions_read_only PASSED [ 91%]
# test_daily_run.py::test_cli_from_alpaca_without_keys_exits_2_and_does_nothing PASSED [ 93%]
# test_daily_run.py::test_the_module_never_places_orders_and_imports_no_trading_client_at_load PASSED [ 95%]
# test_daily_run.py::test_default_update_appends_the_new_day_saves_a_new_snapshot_and_a_second_call_says_nothing_new PASSED [ 97%]
# test_daily_run.py::test_default_update_rejects_a_fresh_pull_that_disagrees_with_the_history PASSED [100%]
# ============================== 46 passed in 2.63s ==============================
#
# Mutation checks (sandbox, 30): stale thresholds, lag measured from today instead of the last business day, bad-index
#   / non-positive-price / SPY-missing / stale-ticker / big-move checks, jump and lookback ignored, config hash missing a
#   field or including equity, unknown config keys accepted, config overwrite, config not validated, failed update fatal,
#   errors not blocking the plan, plan or report overwritten, protocol change not reported, wrong previous run picked,
#   future scoreboard start unguarded, confirmed placeholders still listed, equity override ignored, live instead of paper
#   API, exit codes, missing keys not fatal, no-orders line dropped, unloadable file crashing instead of blocking.
#   One survived the first pass (lag measured from today); a Sunday-with-a-Thursday-bar test was added and it is caught.
# Real snapshot (prices_daily_asof_2026-10-08.csv, 48 ETFs + SH, run with --today 2026-10-09 --no-update): status OK, 48
#   orders, cash after $0.00, SPY 51.0% of the account, a second run saved _r2 and reported the protocol change after the
#   config gained a start date; scoreboard 2026-09-30 to 2026-10-08 ran (a mechanics check, NOT a finding). A first version
#   scanned 252 days for big moves and warned every day about SLV -28.5% on 2026-01-30 (a genuine move); the default is now
#   the last 10 rows, since the aim is to catch a bad NEW bar.
# NOT YET RUN on your machine against live Yahoo (default_update) or the Alpaca paper account (--from-alpaca).
# ---------------------------------------------------------------------------
