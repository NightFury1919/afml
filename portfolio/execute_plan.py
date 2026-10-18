"""Place a saved v2 plan on the Alpaca PAPER account, with guardrails. Dry run unless you say otherwise.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env, with the PAPER keys of the account you mean to trade:
    python execute_plan.py                                   # DRY RUN: checks everything, lists the orders, sends nothing
    python execute_plan.py --execute --confirm 2026-10-16    # place them; --confirm must equal the plan's as_of date
    python execute_plan.py --reconcile                       # compare current positions with the plan's targets, send nothing
    python execute_plan.py --execute --confirm DATE --allow-closed   # submit while the market is closed (see below)

What it checks before it sends anything (any ERROR refuses the whole run)
    plan is version 2 and at most 4 days old; account ACTIVE and not trading-blocked; account equity within 5% of the
    plan's equity; no open orders already; positions match the holdings the plan assumed (within $1 each); this plan has
    not been executed before (an orders/orders_<as_of>.json ledger exists); the market is open (or --allow-closed).
    Sells are never sent while the market is closed.

How it places orders
    Market orders by dollar amount (notional), time in force DAY, the only kind Alpaca takes for fractional shares.
    Sells go first and the run waits for them to FILL before any buy; a sell that does not fill in time, or is rejected
    or cancelled, stops the run before the buys. A rejected buy is recorded and the other buys still go ("partial").
    A full exit uses close_position. The ledger file is written BEFORE the first order, so a crash cannot allow a
    second run of the same plan, and it records every order id and error. Nothing is ever cancelled or replaced.

If the market is closed, Alpaca's order docs say a DAY order submitted after the close is queued for the next trading
day. --allow-closed relies on that. Whether fractional orders are queued the same way is not confirmed, so the first
time, expect that Alpaca may reject them and the ledger will say so.

This module only ever talks to the PAPER API: the client is built with the paper flag set and there is no option to change that.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PLAN_RE = re.compile(r"^plan_v2_(\d{4}-\d{2}-\d{2})\.json$")
TERMINAL_BAD = {"canceled", "cancelled", "rejected", "expired", "done_for_day", "suspended"}


class ExecutionRefused(Exception):
    """A pre-flight check failed. Nothing was sent."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(f"{i['code']}: {i['message']}" for i in issues if i["level"] == "error"))


def _issue(level, code, message):
    return dict(level=level, code=code, message=message)


# ------------------------------------------------------------------------------------ plan files
def latest_plan(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    found = [(m.group(1), p) for p in folder.iterdir() if (m := PLAN_RE.match(p.name))]
    if not found:
        raise FileNotFoundError(f"no plan_v2_YYYY-MM-DD.json in {folder}")
    return max(found)[1]


def load_plan(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------ checks
def preflight(plan, account, positions, open_orders, clock, today, ledger_exists=False, max_plan_age_days=4,
              equity_tol=0.05, holdings_tol=1.0, allow_closed=False):
    """List of issues (level 'error' or 'warn'). Any error means: send nothing."""
    issues = []
    if plan.get("version") != 2:
        issues.append(_issue("error", "plan_version", f"plan version is {plan.get('version')!r}, expected 2"))
    age = (pd.Timestamp(today).normalize() - pd.Timestamp(plan["as_of"])).days
    if age > max_plan_age_days:
        issues.append(_issue("error", "plan_stale", f"the plan is {age} days old (as of {plan['as_of']}); make a new one"))
    if account.status != "active":
        issues.append(_issue("error", "account_status", f"account status is {account.status!r}, not 'active'"))
    if account.trading_blocked:
        issues.append(_issue("error", "account_blocked", "the account is trading-blocked"))
    if abs(account.equity / plan["equity"] - 1.0) > equity_tol:
        issues.append(_issue("error", "equity_mismatch", f"account equity ${account.equity:,.2f} is not within "
                             f"{equity_tol:.0%} of the plan's ${plan['equity']:,.2f}"))
    if open_orders:
        issues.append(_issue("error", "open_orders", f"there are open orders already: {', '.join(sorted(open_orders))}"))
    assumed = plan.get("holdings", {})
    off = sorted(s for s in set(assumed) | set(positions) if abs(positions.get(s, 0.0) - assumed.get(s, 0.0)) > holdings_tol)
    if off:
        issues.append(_issue("error", "holdings_changed", "positions differ from what the plan assumed for: " + ", ".join(off)))
    if ledger_exists:
        issues.append(_issue("error", "already_executed", f"an order ledger for {plan['as_of']} already exists; a plan is placed once"))
    orders = plan.get("orders", [])
    if not orders:
        issues.append(_issue("warn", "no_orders", "the plan has no orders"))
    if not clock.is_open:
        if allow_closed:
            issues.append(_issue("warn", "market_closed", f"market is closed (next open {clock.next_open}); "
                                 "DAY orders should queue for then"))
            if any(o["side"] == "sell" for o in orders):
                issues.append(_issue("error", "sells_need_open_market", "sells are never sent while the market is closed"))
        else:
            issues.append(_issue("error", "market_closed", f"market is closed (next open {clock.next_open}); "
                                 "use --allow-closed to queue orders"))
    return issues


def reconcile(plan, positions, tol_abs=1.0, tol_rel=0.05):
    """Rows {symbol, target, actual, diff, ok}: each plan target in dollars against the position held."""
    target = {s: round(w * plan["equity"], 2) for s, w in plan["weights"].items() if w > 0}
    rows = []
    for s in sorted(set(target) | set(positions)):
        t, a = target.get(s, 0.0), float(positions.get(s, 0.0))
        diff = round(a - t, 2)
        rows.append(dict(symbol=s, target=t, actual=a, diff=diff, ok=abs(diff) <= max(tol_abs, tol_rel * t)))
    return rows


# ------------------------------------------------------------------------------------ executing
def _ledger_path(orders_dir, plan):
    return Path(orders_dir) / f"orders_{plan['as_of']}.json"


def _write(path, ledger):
    path.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")


def execute(plan, gateway, orders_dir, today, allow_closed=False, fill_timeout=30, poll_seconds=2, sleep=time.sleep):
    """Run the checks, then place the plan. Returns the ledger dict. Raises ExecutionRefused before sending anything."""
    path = _ledger_path(orders_dir, plan)
    issues = preflight(plan, gateway.account(), gateway.positions(), gateway.open_orders(), gateway.clock(), today,
                       ledger_exists=path.exists(), allow_closed=allow_closed)
    if any(i["level"] == "error" for i in issues):
        raise ExecutionRefused(issues)

    ledger = dict(as_of=plan["as_of"], started=str(pd.Timestamp.now()), plan_equity=plan["equity"], status="running",
                  note="", warnings=[i for i in issues if i["level"] == "warn"], entries=[])
    orders = plan.get("orders", [])
    if not orders:
        ledger["status"] = "nothing_to_do"
        return ledger
    Path(orders_dir).mkdir(parents=True, exist_ok=True)
    _write(path, ledger)                                    # BEFORE the first order: a crash must not allow a re-run

    def place(order):
        entry = dict(symbol=order["symbol"], side=order["side"], notional=order.get("notional"),
                     close_all=bool(order.get("close_all")), order_id=None, status=None, error=None)
        ledger["entries"].append(entry)
        try:
            resp = gateway.submit(order)
            entry["order_id"], entry["status"] = resp["id"], resp["status"]
        except Exception as e:                              # recorded; KeyboardInterrupt and friends still propagate
            entry["error"] = f"{type(e).__name__}: {e}"
        _write(path, ledger)
        return entry

    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]

    sell_entries = [place(o) for o in sells]
    if sells:
        if any(e["error"] for e in sell_entries):
            ledger.update(status="aborted", note="a sell was rejected, so the buys were not sent")
            _write(path, ledger)
            return ledger
        pending, waited = {e["order_id"]: e for e in sell_entries}, 0
        while pending and waited < fill_timeout:
            for oid in list(pending):
                status = gateway.order_status(oid)
                pending[oid]["status"] = status
                if status == "filled":
                    del pending[oid]
                elif status in TERMINAL_BAD:
                    ledger.update(status="aborted", note=f"sell {pending[oid]['symbol']} ended {status}, so the buys were not sent")
                    _write(path, ledger)
                    return ledger
            if pending:
                sleep(poll_seconds)
                waited += poll_seconds
        if pending:
            ledger.update(status="aborted", note="sells did not fill within "
                          f"{fill_timeout}s, so the buys were not sent: " + ", ".join(e["symbol"] for e in pending.values()))
            _write(path, ledger)
            return ledger

    buy_entries = [place(o) for o in buys]
    failed = [e for e in sell_entries + buy_entries if e["error"]]
    ledger["status"] = "partial" if failed else "complete"
    if failed:
        ledger["note"] = "rejected: " + ", ".join(e["symbol"] for e in failed)
    _write(path, ledger)
    return ledger


# ------------------------------------------------------------------------------------ the real gateway
def _text(x):
    return "" if x is None else str(getattr(x, "value", x)).lower()


class AlpacaGateway:
    """Thin wrapper over alpaca-py's TradingClient: plain values in and out, so the logic above is testable."""

    def __init__(self, key, secret):
        from alpaca.trading.client import TradingClient           # imported here so the tests need no alpaca-py

        self.client = TradingClient(key, secret, paper=True)

    def account(self):
        a = self.client.get_account()
        return SimpleNamespace(status=_text(a.status), equity=float(a.equity), cash=float(a.cash),
                               trading_blocked=bool(a.trading_blocked))

    def positions(self):
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def open_orders(self):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return [o.symbol for o in self.client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))]

    def clock(self):
        c = self.client.get_clock()
        return SimpleNamespace(is_open=bool(c.is_open), next_open=str(c.next_open))

    def submit(self, order):
        if order.get("close_all"):
            resp = self.client.close_position(order["symbol"])
        else:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            side = OrderSide.BUY if order["side"] == "buy" else OrderSide.SELL
            resp = self.client.submit_order(MarketOrderRequest(symbol=order["symbol"], notional=float(order["notional"]),
                                                               side=side, time_in_force=TimeInForce.DAY))
        return dict(id=str(resp.id), status=_text(resp.status))

    def order_status(self, order_id):
        return _text(self.client.get_order_by_id(order_id).status)


# ------------------------------------------------------------------------------------ the CLI
def _print_issues(issues):
    for i in issues:
        print(f"  {i['level'].upper():5s} {i['code']}: {i['message']}")


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", default=None, help="a plan_v2_*.json file (default: the newest in --plans-dir)")
    ap.add_argument("--plans-dir", default=str(here / "plans"))
    ap.add_argument("--orders-dir", default=str(here / "orders"))
    ap.add_argument("--execute", action="store_true", help="actually place the orders (needs --confirm)")
    ap.add_argument("--confirm", default=None, help="the plan's as_of date, typed out, to allow --execute")
    ap.add_argument("--allow-closed", action="store_true", help="submit while the market is closed (DAY orders queue)")
    ap.add_argument("--reconcile", action="store_true", help="compare positions with the plan's targets and stop")
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    args = ap.parse_args()

    if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
        print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys), as check_account.py does.")
        sys.exit(2)
    try:
        plan_path = Path(args.plan) if args.plan else latest_plan(args.plans_dir)
    except FileNotFoundError as e:
        print(f"{e}; run daily_run.py first (no plan_v2 file to place)")
        sys.exit(2)
    plan = load_plan(plan_path)
    today = args.today or pd.Timestamp.now(tz="America/New_York").date().isoformat()

    if args.execute and args.confirm != plan["as_of"]:
        print(f"Refusing: to place {plan_path.name} add --confirm {plan['as_of']} (the plan's date, typed out).")
        sys.exit(2)

    gateway = AlpacaGateway(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])

    if args.reconcile:
        rows = reconcile(plan, gateway.positions())
        print(f"RECONCILE {plan_path.name}: targets from the plan against the paper account's positions. Nothing is sent.")
        for r in rows:
            print(f"  {r['symbol']:6s} target ${r['target']:9,.2f}  actual ${r['actual']:9,.2f}  diff {r['diff']:+9,.2f}  "
                  f"{'ok' if r['ok'] else 'OFF'}")
        sys.exit(0)

    if not args.execute:
        issues = preflight(plan, gateway.account(), gateway.positions(), gateway.open_orders(), gateway.clock(), today,
                           ledger_exists=_ledger_path(args.orders_dir, plan).exists(), allow_closed=args.allow_closed)
        print(f"DRY RUN of {plan_path.name} (equity ${plan['equity']:,.2f}). Nothing is sent.")
        _print_issues(issues)
        print("Orders it would place:")
        for o in plan["orders"]:
            amount = "close position" if o.get("close_all") else f"${o['notional']:,.2f}"
            print(f"  {o['side']:4s} {o['symbol']:6s} {amount}")
        errors = [i for i in issues if i["level"] == "error"]
        print("\nREFUSED: fix the errors above." if errors else
              f"\nAll checks pass. To place these: python execute_plan.py --execute --confirm {plan['as_of']}")
        sys.exit(1 if errors else 0)

    try:
        ledger = execute(plan, gateway, args.orders_dir, today, allow_closed=args.allow_closed)
    except ExecutionRefused as e:
        print("REFUSED, nothing was sent:")
        _print_issues(e.issues)
        sys.exit(1)
    print(f"PLACED {plan_path.name}: status {ledger['status']}. {ledger['note']}".rstrip())
    for e in ledger["entries"]:
        print(f"  {e['side']:4s} {e['symbol']:6s} id {e['order_id']}  {e['status']}" + (f"  ERROR {e['error']}" if e["error"] else ""))
    print(f"Ledger: {_ledger_path(args.orders_dir, plan)}")
    print("Check the fills with: python execute_plan.py --reconcile")
    sys.exit(0 if ledger["status"] in ("complete", "nothing_to_do") else 1)


if __name__ == "__main__":
    main()

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
