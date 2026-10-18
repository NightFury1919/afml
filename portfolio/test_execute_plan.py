import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from execute_plan import (ExecutionRefused, AlpacaGateway, execute, latest_plan, load_plan, preflight,
                          reconcile)

PLAN = dict(version=2, as_of="2026-10-16", equity=1000.0, holdings={},
            weights={"SPY": 0.6, "QQQ": 0.4},
            orders=[dict(symbol="QQQ", side="buy", notional=400.0), dict(symbol="SPY", side="buy", notional=600.0)])
TODAY = "2026-10-17"


class FakeGateway:
    """Stands in for AlpacaGateway: no network, records every call."""

    def __init__(self, equity=1000.0, status="active", blocked=False, positions=None, open_orders=(), is_open=True,
                 fail=(), final_status="filled"):
        self._acct = SimpleNamespace(status=status, equity=equity, cash=equity, trading_blocked=blocked)
        self._pos, self._open, self._is_open = dict(positions or {}), list(open_orders), is_open
        self.fail, self.final_status = set(fail), final_status
        self.calls, self.submitted = [], []

    def account(self):
        self.calls.append("account")
        return self._acct

    def positions(self):
        self.calls.append("positions")
        return dict(self._pos)

    def open_orders(self):
        self.calls.append("open_orders")
        return list(self._open)

    def clock(self):
        self.calls.append("clock")
        return SimpleNamespace(is_open=self._is_open, next_open="2026-10-19T09:30:00-04:00")

    def submit(self, order):
        self.calls.append(("submit", order["symbol"], order["side"]))
        if order["symbol"] in self.fail:
            raise RuntimeError(f"rejected {order['symbol']}")
        self.submitted.append(order)
        return dict(id=f"id-{len(self.submitted)}", status="accepted")

    def order_status(self, order_id):
        self.calls.append(("status", order_id))
        return self.final_status


def codes(issues, level=None):
    return sorted(i["code"] for i in issues if level is None or i["level"] == level)


def pre(plan=PLAN, gw=None, **kw):
    gw = gw or FakeGateway()
    args = dict(today=TODAY, ledger_exists=False)
    args.update(kw)
    return preflight(plan, gw.account(), gw.positions(), gw.open_orders(), gw.clock(), **args)


def run(tmp_path, plan=PLAN, gw=None, **kw):
    gw = gw or FakeGateway()
    args = dict(orders_dir=tmp_path / "orders", today=TODAY, sleep=lambda s: None)
    args.update(kw)
    return gw, execute(plan, gw, **args)


# ----------------------------------------------------------------------------------- preflight
def test_a_healthy_account_and_a_fresh_plan_have_no_issues():
    assert pre() == []


def test_a_plan_that_is_not_version_2_is_refused():
    assert "plan_version" in codes(pre(plan={**PLAN, "version": 1}), "error")


def test_an_old_plan_is_refused_and_the_age_limit_is_adjustable():
    assert "plan_stale" in codes(pre(today="2026-10-25"), "error")
    assert "plan_stale" not in codes(pre(today="2026-10-25", max_plan_age_days=10))
    assert "plan_stale" not in codes(pre(today="2026-10-20"))                  # 4 days: still fine


def test_an_inactive_or_blocked_account_is_refused():
    assert "account_status" in codes(pre(gw=FakeGateway(status="account_updated")), "error")
    assert "account_blocked" in codes(pre(gw=FakeGateway(blocked=True)), "error")


def test_account_equity_far_from_the_plan_equity_is_refused_but_a_small_drift_is_fine():
    assert "equity_mismatch" in codes(pre(gw=FakeGateway(equity=99999.7)), "error")
    assert "equity_mismatch" not in codes(pre(gw=FakeGateway(equity=1030.0)))
    assert "equity_mismatch" in codes(pre(gw=FakeGateway(equity=1030.0), equity_tol=0.01), "error")


def test_open_orders_block_the_run_so_a_double_submit_is_impossible():
    assert "open_orders" in codes(pre(gw=FakeGateway(open_orders=["SPY"])), "error")


def test_holdings_that_differ_from_what_the_plan_assumed_are_refused_naming_the_symbols():
    issues = pre(gw=FakeGateway(positions={"SPY": 600.0}))
    bad = next(i for i in issues if i["code"] == "holdings_changed")
    assert bad["level"] == "error" and "SPY" in bad["message"]
    assert "holdings_changed" not in codes(pre(gw=FakeGateway(positions={"SPY": 0.5})))   # within $1 dust


def test_a_plan_that_assumed_holdings_matches_when_the_account_holds_exactly_those():
    plan = {**PLAN, "holdings": {"SPY": 300.0}}
    assert "holdings_changed" not in codes(pre(plan=plan, gw=FakeGateway(positions={"SPY": 300.4})))
    assert "holdings_changed" in codes(pre(plan=plan, gw=FakeGateway()), "error")


def test_a_plan_that_was_already_executed_is_refused():
    assert "already_executed" in codes(pre(ledger_exists=True), "error")


def test_a_closed_market_is_an_error_unless_allowed_and_then_a_warning():
    closed = FakeGateway(is_open=False)
    assert "market_closed" in codes(pre(gw=closed), "error")
    assert codes(pre(gw=closed, allow_closed=True), "warn") == ["market_closed"]
    assert codes(pre(gw=closed, allow_closed=True), "error") == []


def test_sells_cannot_be_sent_while_the_market_is_closed_even_if_closed_orders_are_allowed():
    plan = {**PLAN, "holdings": {"SPY": 100.0}, "orders": [dict(symbol="SPY", side="sell", notional=50.0)]}
    gw = FakeGateway(is_open=False, positions={"SPY": 100.0})
    assert "sells_need_open_market" in codes(pre(plan=plan, gw=gw, allow_closed=True), "error")


def test_a_plan_with_no_orders_is_a_warning_not_an_error():
    issues = pre(plan={**PLAN, "orders": []})
    assert codes(issues, "warn") == ["no_orders"] and codes(issues, "error") == []


# ----------------------------------------------------------------------------------- execute
def test_a_clean_run_submits_each_order_once_writes_a_ledger_and_reports_complete(tmp_path):
    gw, led = run(tmp_path)
    assert [(o["symbol"], o["side"], o["notional"]) for o in gw.submitted] == [("QQQ", "buy", 400.0), ("SPY", "buy", 600.0)]
    assert led["status"] == "complete" and [e["symbol"] for e in led["entries"]] == ["QQQ", "SPY"]
    assert all(e["order_id"] and e["error"] is None for e in led["entries"])
    saved = json.loads((tmp_path / "orders" / "orders_2026-10-16.json").read_text(encoding="utf-8"))
    assert saved["status"] == "complete" and saved["as_of"] == "2026-10-16"


def test_a_refused_preflight_raises_and_submits_nothing_and_writes_no_ledger(tmp_path):
    gw = FakeGateway(equity=99999.7)
    with pytest.raises(ExecutionRefused) as e:
        run(tmp_path, gw=gw)
    assert "equity_mismatch" in str(e.value) and gw.submitted == []
    assert not (tmp_path / "orders").exists()


def test_running_the_same_plan_twice_is_refused_the_second_time(tmp_path):
    run(tmp_path)
    gw2 = FakeGateway()
    with pytest.raises(ExecutionRefused, match="already_executed"):
        run(tmp_path, gw=gw2)
    assert gw2.submitted == []


def test_a_rejected_buy_is_recorded_the_others_still_go_and_the_run_is_partial(tmp_path):
    gw, led = run(tmp_path, gw=FakeGateway(fail={"QQQ"}))
    assert [o["symbol"] for o in gw.submitted] == ["SPY"] and led["status"] == "partial"
    qqq = next(e for e in led["entries"] if e["symbol"] == "QQQ")
    assert qqq["order_id"] is None and "rejected QQQ" in qqq["error"]


def test_the_ledger_exists_before_the_first_order_so_a_crash_cannot_allow_a_second_run(tmp_path):
    class Crashy(FakeGateway):
        def submit(self, order):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, gw=Crashy())
    assert (tmp_path / "orders" / "orders_2026-10-16.json").exists()
    with pytest.raises(ExecutionRefused, match="already_executed"):
        run(tmp_path)


SELL_PLAN = {**PLAN, "holdings": {"SPY": 700.0}, "equity": 1000.0,
             "orders": [dict(symbol="SPY", side="sell", notional=100.0), dict(symbol="QQQ", side="buy", notional=400.0)]}


def test_sells_go_first_and_buys_wait_until_the_sells_have_filled(tmp_path):
    gw, led = run(tmp_path, plan=SELL_PLAN, gw=FakeGateway(positions={"SPY": 700.0}))
    kinds = [c for c in gw.calls if isinstance(c, tuple)]
    assert kinds[0] == ("submit", "SPY", "sell") and ("status", "id-1") in kinds
    assert kinds.index(("status", "id-1")) < kinds.index(("submit", "QQQ", "buy"))
    assert led["status"] == "complete"


def test_if_a_sell_does_not_fill_in_time_no_buy_is_sent(tmp_path):
    gw, led = run(tmp_path, plan=SELL_PLAN, gw=FakeGateway(positions={"SPY": 700.0}, final_status="new"), fill_timeout=3)
    assert [o["symbol"] for o in gw.submitted] == ["SPY"] and led["status"] == "aborted"
    assert "buys were not sent" in led["note"]


def test_a_rejected_sell_aborts_before_any_buy(tmp_path):
    gw, led = run(tmp_path, plan=SELL_PLAN, gw=FakeGateway(positions={"SPY": 700.0}, fail={"SPY"}))
    assert gw.submitted == [] and led["status"] == "aborted"


def test_a_sell_that_ends_canceled_or_rejected_aborts_before_any_buy(tmp_path):
    slept = []
    gw, led = run(tmp_path, plan=SELL_PLAN, gw=FakeGateway(positions={"SPY": 700.0}, final_status="canceled"), sleep=slept.append)
    assert [o["symbol"] for o in gw.submitted] == ["SPY"] and led["status"] == "aborted"
    assert "canceled" in led["note"] and slept == []                    # stops at once, does not wait out the timeout


def test_the_fill_wait_polls_with_the_sleep_function_and_stops_at_the_timeout(tmp_path):
    slept = []
    gw = FakeGateway(positions={"SPY": 700.0}, final_status="new")
    run(tmp_path, plan=SELL_PLAN, gw=gw, fill_timeout=10, poll_seconds=2, sleep=slept.append)
    assert slept and set(slept) == {2} and 4 <= len(slept) <= 6


def test_a_plan_with_no_orders_does_nothing_and_says_so(tmp_path):
    gw, led = run(tmp_path, plan={**PLAN, "orders": []})
    assert gw.submitted == [] and led["status"] == "nothing_to_do"


def test_a_closed_market_run_needs_allow_closed_and_then_records_the_warning(tmp_path):
    with pytest.raises(ExecutionRefused, match="market_closed"):
        run(tmp_path, gw=FakeGateway(is_open=False))
    gw, led = run(tmp_path, gw=FakeGateway(is_open=False), allow_closed=True)
    assert len(gw.submitted) == 2 and any(i["code"] == "market_closed" for i in led["warnings"])


# ----------------------------------------------------------------------------------- reconcile
def test_reconcile_compares_each_target_dollar_amount_with_the_position():
    rows = reconcile(PLAN, {"SPY": 599.0, "QQQ": 370.0})
    by = {r["symbol"]: r for r in rows}
    assert by["SPY"]["target"] == 600.0 and by["SPY"]["actual"] == 599.0 and by["SPY"]["ok"]
    assert by["QQQ"]["diff"] == -30.0 and not by["QQQ"]["ok"]


def test_reconcile_flags_a_position_the_plan_does_not_want_and_a_target_with_no_position():
    by = {r["symbol"]: r for r in reconcile(PLAN, {"GLD": 50.0})}
    assert not by["GLD"]["ok"] and by["GLD"]["target"] == 0.0
    assert not by["SPY"]["ok"] and by["SPY"]["actual"] == 0.0


def test_reconcile_tolerance_is_the_larger_of_a_dollar_and_five_percent_of_target():
    assert {r["symbol"]: r["ok"] for r in reconcile(PLAN, {"SPY": 575.0, "QQQ": 400.0})}["SPY"]       # 25 < 5% of 600 = 30
    assert not {r["symbol"]: r["ok"] for r in reconcile(PLAN, {"SPY": 565.0, "QQQ": 400.0})}["SPY"]   # 35 > 30
    tiny = {**PLAN, "weights": {"SPY": 0.005}}
    assert {r["symbol"]: r["ok"] for r in reconcile(tiny, {"SPY": 5.9})}["SPY"]                       # $0.90 off a $5 target


# ----------------------------------------------------------------------------------- plan files
def test_latest_plan_picks_the_newest_date_and_ignores_other_files(tmp_path):
    for name in ["plan_v2_2026-10-08.json", "plan_v2_2026-10-16.json", "plan_2026-10-17.json", "notes.txt"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert latest_plan(tmp_path).name == "plan_v2_2026-10-16.json"
    with pytest.raises(FileNotFoundError):
        latest_plan(tmp_path / "missing")


def test_load_plan_reads_the_json(tmp_path):
    p = tmp_path / "plan_v2_2026-10-16.json"
    p.write_text(json.dumps(PLAN), encoding="utf-8")
    assert load_plan(p)["as_of"] == "2026-10-16"


# ----------------------------------------------------------------------------------- the real gateway, with a fake alpaca
def install_fake_alpaca(monkeypatch):
    import types
    log = dict(client=None, requests=[], closed=[], paper=None)

    class Client:
        def __init__(self, key, secret, paper=False):
            log["paper"] = paper

        def get_account(self):
            return SimpleNamespace(status=SimpleNamespace(value="ACTIVE"), equity="1000", cash="1000", trading_blocked=False)

        def get_all_positions(self):
            return [SimpleNamespace(symbol="SPY", market_value="12.5")]

        def get_orders(self, req):
            log["requests"].append(("get_orders", req))
            return [SimpleNamespace(symbol="QQQ")]

        def get_clock(self):
            return SimpleNamespace(is_open=True, next_open="x")

        def submit_order(self, req):
            log["requests"].append(("submit", req))
            return SimpleNamespace(id="abc", status=SimpleNamespace(value="accepted"))

        def close_position(self, symbol):
            log["closed"].append(symbol)
            return SimpleNamespace(id="def", status=SimpleNamespace(value="pending_new"))

        def get_order_by_id(self, oid):
            return SimpleNamespace(status=SimpleNamespace(value="FILLED"))

    class Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    enums = dict(OrderSide=SimpleNamespace(BUY="BUY", SELL="SELL"), TimeInForce=SimpleNamespace(DAY="DAY"),
                 QueryOrderStatus=SimpleNamespace(OPEN="OPEN"))
    mods = {"alpaca": {}, "alpaca.trading": {}, "alpaca.trading.client": {"TradingClient": Client},
            "alpaca.trading.enums": enums, "alpaca.trading.requests": {"MarketOrderRequest": Req, "GetOrdersRequest": Req}}
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
    return log


def test_the_gateway_uses_the_paper_api_and_converts_alpaca_objects_to_plain_values(monkeypatch):
    log = install_fake_alpaca(monkeypatch)
    gw = AlpacaGateway("k", "s")
    assert log["paper"] is True
    a = gw.account()
    assert a.status == "active" and a.equity == 1000.0 and a.trading_blocked is False
    assert gw.positions() == {"SPY": 12.5} and gw.open_orders() == ["QQQ"] and gw.clock().is_open is True
    assert gw.order_status("abc") == "filled"


def test_the_gateway_sends_notional_day_market_orders_and_closes_a_full_exit_with_close_position(monkeypatch):
    log = install_fake_alpaca(monkeypatch)
    gw = AlpacaGateway("k", "s")
    assert gw.submit(dict(symbol="SPY", side="buy", notional=600.0)) == dict(id="abc", status="accepted")
    gw.submit(dict(symbol="QQQ", side="sell", notional=12.34))
    req_buy, req_sell = [r for k, r in log["requests"] if k == "submit"]
    assert (req_buy.symbol, req_buy.notional, req_buy.side, req_buy.time_in_force) == ("SPY", 600.0, "BUY", "DAY")
    assert (req_sell.notional, req_sell.side) == (12.34, "SELL") and not hasattr(req_buy, "qty")
    assert gw.submit(dict(symbol="GLD", side="sell", close_all=True))["status"] == "pending_new"
    assert log["closed"] == ["GLD"]


# ----------------------------------------------------------------------------------- the CLI
def write_plan_file(tmp_path, plan=PLAN):
    d = tmp_path / "plans"
    d.mkdir(exist_ok=True)
    (d / f"plan_v2_{plan['as_of']}.json").write_text(json.dumps(plan), encoding="utf-8")


def run_cli(monkeypatch, tmp_path, gw, *extra, keys=True):
    import execute_plan
    if keys:
        monkeypatch.setenv("APCA_API_KEY_ID", "k")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    else:
        monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
        monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(execute_plan, "AlpacaGateway", lambda key, secret: gw)
    monkeypatch.setattr(sys, "argv", ["execute_plan.py", "--plans-dir", str(tmp_path / "plans"),
                                      "--orders-dir", str(tmp_path / "orders"), "--today", TODAY, *extra])
    with pytest.raises(SystemExit) as e:
        execute_plan.main()
    return e.value.code


def test_cli_default_is_a_dry_run_that_lists_the_orders_and_sends_nothing(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    gw = FakeGateway()
    assert run_cli(monkeypatch, tmp_path, gw) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "QQQ" in out and "400.00" in out and "--execute" in out
    assert gw.submitted == [] and not (tmp_path / "orders").exists()


def test_cli_dry_run_exits_1_and_says_why_when_a_check_fails(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    assert run_cli(monkeypatch, tmp_path, FakeGateway(equity=99999.7)) == 1
    assert "equity_mismatch" in capsys.readouterr().out


def test_cli_execute_without_the_matching_confirm_date_sends_nothing_and_exits_2(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    gw = FakeGateway()
    assert run_cli(monkeypatch, tmp_path, gw, "--execute") == 2
    assert run_cli(monkeypatch, tmp_path, gw, "--execute", "--confirm", "2026-10-15") == 2
    assert gw.submitted == [] and "--confirm 2026-10-16" in capsys.readouterr().out


def test_cli_execute_with_the_right_confirm_places_the_orders_and_exits_0(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    gw = FakeGateway()
    assert run_cli(monkeypatch, tmp_path, gw, "--execute", "--confirm", "2026-10-16") == 0
    assert len(gw.submitted) == 2 and "PLACED" in capsys.readouterr().out
    assert run_cli(monkeypatch, tmp_path, FakeGateway(), "--execute", "--confirm", "2026-10-16") == 1   # second time: refused


def test_cli_a_partial_run_exits_1(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    assert run_cli(monkeypatch, tmp_path, FakeGateway(fail={"QQQ"}), "--execute", "--confirm", "2026-10-16") == 1
    assert "partial" in capsys.readouterr().out.lower()


def test_cli_reconcile_prints_the_table_and_sends_nothing(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    gw = FakeGateway(positions={"SPY": 600.0, "QQQ": 300.0})
    assert run_cli(monkeypatch, tmp_path, gw, "--reconcile") == 0
    out = capsys.readouterr().out
    assert "RECONCILE" in out and "QQQ" in out and "OFF" in out and gw.submitted == []


def test_cli_without_keys_exits_2_and_touches_nothing(monkeypatch, tmp_path, capsys):
    write_plan_file(tmp_path)
    gw = FakeGateway()
    assert run_cli(monkeypatch, tmp_path, gw, keys=False) == 2
    assert gw.calls == []


def test_cli_with_no_plan_file_exits_2(monkeypatch, tmp_path, capsys):
    assert run_cli(monkeypatch, tmp_path, FakeGateway()) == 2
    assert "no plan_v2" in capsys.readouterr().out


# ----------------------------------------------------------------------------------- source guards
def test_the_module_can_only_reach_the_paper_api_and_never_uses_risky_order_features():
    src = (Path(__file__).resolve().parent / "execute_plan.py").read_text(encoding="utf-8")
    assert "paper=True" in src and "paper=False" not in src and "paper=" not in src.replace("paper=True", "")
    for forbidden in ("cancel_order", "cancel_orders", "replace_order", "TimeInForce.GTC", "extended_hours", "LimitOrderRequest"):
        assert forbidden not in src


def test_no_alpaca_import_happens_at_module_load():
    import importlib
    for m in [m for m in sys.modules if m.startswith("alpaca")]:
        del sys.modules[m]
    importlib.reload(sys.modules["execute_plan"])
    assert not [m for m in sys.modules if m.startswith("alpaca")]

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# same versions as the mlfinlab env; alpaca-py is faked in every test, so nothing here touched Alpaca)
# $ cd portfolio ; pytest test_execute_plan.py -v
#
# collected 41 items
# test_execute_plan.py::test_a_healthy_account_and_a_fresh_plan_have_no_issues PASSED [  2%]
# test_execute_plan.py::test_a_plan_that_is_not_version_2_is_refused PASSED [  4%]
# test_execute_plan.py::test_an_old_plan_is_refused_and_the_age_limit_is_adjustable PASSED [  7%]
# test_execute_plan.py::test_an_inactive_or_blocked_account_is_refused PASSED [  9%]
# test_execute_plan.py::test_account_equity_far_from_the_plan_equity_is_refused_but_a_small_drift_is_fine PASSED [ 12%]
# test_execute_plan.py::test_open_orders_block_the_run_so_a_double_submit_is_impossible PASSED [ 14%]
# test_execute_plan.py::test_holdings_that_differ_from_what_the_plan_assumed_are_refused_naming_the_symbols PASSED [ 17%]
# test_execute_plan.py::test_a_plan_that_assumed_holdings_matches_when_the_account_holds_exactly_those PASSED [ 19%]
# test_execute_plan.py::test_a_plan_that_was_already_executed_is_refused PASSED [ 21%]
# test_execute_plan.py::test_a_closed_market_is_an_error_unless_allowed_and_then_a_warning PASSED [ 24%]
# test_execute_plan.py::test_sells_cannot_be_sent_while_the_market_is_closed_even_if_closed_orders_are_allowed PASSED [ 26%]
# test_execute_plan.py::test_a_plan_with_no_orders_is_a_warning_not_an_error PASSED [ 29%]
# test_execute_plan.py::test_a_clean_run_submits_each_order_once_writes_a_ledger_and_reports_complete PASSED [ 31%]
# test_execute_plan.py::test_a_refused_preflight_raises_and_submits_nothing_and_writes_no_ledger PASSED [ 34%]
# test_execute_plan.py::test_running_the_same_plan_twice_is_refused_the_second_time PASSED [ 36%]
# test_execute_plan.py::test_a_rejected_buy_is_recorded_the_others_still_go_and_the_run_is_partial PASSED [ 39%]
# test_execute_plan.py::test_the_ledger_exists_before_the_first_order_so_a_crash_cannot_allow_a_second_run PASSED [ 41%]
# test_execute_plan.py::test_sells_go_first_and_buys_wait_until_the_sells_have_filled PASSED [ 43%]
# test_execute_plan.py::test_if_a_sell_does_not_fill_in_time_no_buy_is_sent PASSED [ 46%]
# test_execute_plan.py::test_a_rejected_sell_aborts_before_any_buy PASSED  [ 48%]
# test_execute_plan.py::test_a_sell_that_ends_canceled_or_rejected_aborts_before_any_buy PASSED [ 51%]
# test_execute_plan.py::test_the_fill_wait_polls_with_the_sleep_function_and_stops_at_the_timeout PASSED [ 53%]
# test_execute_plan.py::test_a_plan_with_no_orders_does_nothing_and_says_so PASSED [ 56%]
# test_execute_plan.py::test_a_closed_market_run_needs_allow_closed_and_then_records_the_warning PASSED [ 58%]
# test_execute_plan.py::test_reconcile_compares_each_target_dollar_amount_with_the_position PASSED [ 60%]
# test_execute_plan.py::test_reconcile_flags_a_position_the_plan_does_not_want_and_a_target_with_no_position PASSED [ 63%]
# test_execute_plan.py::test_reconcile_tolerance_is_the_larger_of_a_dollar_and_five_percent_of_target PASSED [ 65%]
# test_execute_plan.py::test_latest_plan_picks_the_newest_date_and_ignores_other_files PASSED [ 68%]
# test_execute_plan.py::test_load_plan_reads_the_json PASSED               [ 70%]
# test_execute_plan.py::test_the_gateway_uses_the_paper_api_and_converts_alpaca_objects_to_plain_values PASSED [ 73%]
# test_execute_plan.py::test_the_gateway_sends_notional_day_market_orders_and_closes_a_full_exit_with_close_position PASSED [ 75%]
# test_execute_plan.py::test_cli_default_is_a_dry_run_that_lists_the_orders_and_sends_nothing PASSED [ 78%]
# test_execute_plan.py::test_cli_dry_run_exits_1_and_says_why_when_a_check_fails PASSED [ 80%]
# test_execute_plan.py::test_cli_execute_without_the_matching_confirm_date_sends_nothing_and_exits_2 PASSED [ 82%]
# test_execute_plan.py::test_cli_execute_with_the_right_confirm_places_the_orders_and_exits_0 PASSED [ 85%]
# test_execute_plan.py::test_cli_a_partial_run_exits_1 PASSED              [ 87%]
# test_execute_plan.py::test_cli_reconcile_prints_the_table_and_sends_nothing PASSED [ 90%]
# test_execute_plan.py::test_cli_without_keys_exits_2_and_touches_nothing PASSED [ 92%]
# test_execute_plan.py::test_cli_with_no_plan_file_exits_2 PASSED          [ 95%]
# test_execute_plan.py::test_the_module_can_only_reach_the_paper_api_and_never_uses_risky_order_features PASSED [ 97%]
# test_execute_plan.py::test_no_alpaca_import_happens_at_module_load PASSED [100%]
# ============================== 41 passed in 0.38s ==============================
#
# Mutation checks (sandbox, 34, run with bytecode caching off): each guard removed or weakened (plan version, plan age,
#   account status, blocked account, equity mismatch and its tolerance, open orders, holdings change and tolerance,
#   already-executed ledger, closed market, sells while closed, ledger written before the orders, buys before sells,
#   sell rejection / cancel / timeout not aborting the buys, polling never sleeping, rejected buy reported complete,
#   live instead of paper API, GTC instead of DAY, notional sent as qty, full exit without close_position, confirm not
#   required, dry run sending orders, keys not required, partial run exiting 0, reconcile tolerance and unwanted positions,
#   newest plan picking the oldest) made a test fail. One survived the first pass (a cancelled sell was caught only by the
#   timeout, not at once); the test now checks the note and that it does not wait.
# Bytecode note: a same-length mutation made within one second can leave a stale .pyc that a later run reuses (one false
#   test failure showed it). The mutation scripts now clear __pycache__ and run with PYTHONDONTWRITEBYTECODE=1; the
#   mutation checks for check_fractionable.py and daily_run.py were re-run that way and still catch everything.
# Real plan (plan_v2_2026-10-08.json from the real snapshot): 48 buys, $1,000.00 total, executed against a fake gateway and
#   reconciled with every target within tolerance. That is a format check only.
# NOT YET RUN against the Alpaca paper API. Do the dry run first: python execute_plan.py
# ---------------------------------------------------------------------------
