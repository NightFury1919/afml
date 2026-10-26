"""Place a saved v2 plan on the Alpaca PAPER account, with guardrails. Dry run unless you say otherwise.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env, with the PAPER keys of the account you mean to trade:
    python execute_plan.py                                   # DRY RUN: checks everything, lists the orders, sends nothing
    python execute_plan.py --execute --confirm 2026-10-16    # place them; --confirm must equal the plan's as_of date
    python execute_plan.py --reconcile                       # compare current positions with the plan's targets, send nothing
    python execute_plan.py --execute --confirm DATE --holdings-tol 10   # accept up to $10 of price drift per position (see below)
    python execute_plan.py --execute --confirm DATE --allow-closed   # submit while the market is closed (see below)

What it checks before it sends anything (any ERROR refuses the whole run)
    plan is version 2 and at most 4 days old; account ACTIVE and not trading-blocked; account equity within 5% of the
    plan's equity; no open orders already; positions match the holdings the plan assumed (within $1 each, or within
    --holdings-tol dollars); this plan has not been executed before (an orders/orders_<as_of>.json ledger exists); the market is open (or --allow-closed).
    Sells are never sent while the market is closed.

Equity floor (--equity-floor, default 900)
    If the ACCOUNT's equity is below the floor, nothing is sent: no buys and no sells. Below the floor a person decides
    what happens next, not the script. Exactly at the floor is allowed. A floor below $900 (0 turns it off) is shown as
    a warning and written into the ledger (equity_floor). An equity that is not a number is refused too.

Holdings tolerance (--holdings-tol, default 1.0, never more than 50)
    A plan is made from the account's dollar values while the market is closed. When the market opens those values move
    with prices, so a large position can drift by more than $1 and the run is refused for no real reason. --holdings-tol
    lets one run accept a larger drift per position. A value above $1 is shown as a warning and written into the ledger
    (holdings_tol). Orders are still the plan's own dollar amounts, so a drift only shifts a position by about that much.

Before the buys (cash check) and after them (status check)
    Sells come in at today's prices, so the cash can differ a little from the plan. Cash short of the buys by up to 2% of
    them: the shortfall (rounded up to the cent) comes off the largest buy and is written into the ledger ("trimmed").
    Short by more than 2%: no buy is sent, the run is "aborted" and a person decides. After the buys are sent each one is
    looked at again; one that was rejected, cancelled or expired makes the run "partial". One still working after the wait
    is only noted ("not filled yet when checked"). Sale proceeds can be used the same day (Alpaca treats accounts under
    $2,000 as limited margin accounts, not cash accounts).

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
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PLAN_RE = re.compile(r"^plan_v2_(\d{4}-\d{2}-\d{2})\.json$")
TERMINAL_BAD = {"canceled", "cancelled", "rejected", "expired", "done_for_day", "suspended"}
DEFAULT_HOLDINGS_TOL = 1.0      # dollars of drift per position the holdings check accepts
MAX_HOLDINGS_TOL = 50.0         # a looser value would make the holdings check meaningless
MAX_BUY_SHORTFALL = 0.02        # cash short of the buys by more than this share of them: send no buy, a person decides
DEFAULT_EQUITY_FLOOR = 900.0    # dollars of account equity below which nothing is sent (Ethan's loss limit)


class ExecutionRefused(Exception):
    """A pre-flight check failed. Nothing was sent."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__("; ".join(f"{i['code']}: {i['message']}" for i in issues if i["level"] == "error"))


def _issue(level, code, message):
    return dict(level=level, code=code, message=message)


def check_holdings_tol(tol):
    """The tolerance as a float, or ValueError if it is negative, not a number, or above MAX_HOLDINGS_TOL."""
    tol = float(tol)
    if not 0.0 <= tol <= MAX_HOLDINGS_TOL:                  # also False for nan
        raise ValueError(f"holdings tolerance must be between 0 and {MAX_HOLDINGS_TOL:g} dollars, got {tol}")
    return tol


def check_equity_floor(floor):
    """The floor as a float, or ValueError if it is negative, infinite, not a number, or not numeric (0 = off)."""
    try:
        if isinstance(floor, bool):
            raise TypeError("a yes/no value is not a dollar amount")
        value = float(floor)
    except (TypeError, ValueError):
        raise ValueError(f"equity floor must be a number of dollars, got {floor!r}") from None
    if not 0.0 <= value < float("inf"):                     # also False for nan
        raise ValueError(f"equity floor must be 0 or more dollars, got {value}")
    return value


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
              equity_tol=0.05, holdings_tol=DEFAULT_HOLDINGS_TOL, allow_closed=False,
              equity_floor=DEFAULT_EQUITY_FLOOR):
    """List of issues (level 'error' or 'warn'). Any error means: send nothing."""
    equity_floor = check_equity_floor(equity_floor)
    holdings_tol = check_holdings_tol(holdings_tol)
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
    if not account.equity >= equity_floor:                  # also True for a nan equity
        issues.append(_issue("error", "below_equity_floor", f"account equity ${account.equity:,.2f} is below the floor of "
                             f"${equity_floor:,.2f}; nothing is sent (buys or sells) until a person decides what to do"))
    elif equity_floor < DEFAULT_EQUITY_FLOOR:
        issues.append(_issue("warn", "equity_floor_lowered", f"the equity floor is ${equity_floor:,.2f}, below the "
                             f"default ${DEFAULT_EQUITY_FLOOR:,.2f}" + (" (the check is off)" if equity_floor == 0 else "")))
    if abs(account.equity / plan["equity"] - 1.0) > equity_tol:
        issues.append(_issue("error", "equity_mismatch", f"account equity ${account.equity:,.2f} is not within "
                             f"{equity_tol:.0%} of the plan's ${plan['equity']:,.2f}"))
    if open_orders:
        issues.append(_issue("error", "open_orders", f"there are open orders already: {', '.join(sorted(open_orders))}"))
    assumed = plan.get("holdings", {})
    off = sorted(s for s in set(assumed) | set(positions) if abs(positions.get(s, 0.0) - assumed.get(s, 0.0)) > holdings_tol)
    if off:
        issues.append(_issue("error", "holdings_changed", "positions differ from what the plan assumed for: " + ", ".join(off)))
    if holdings_tol > DEFAULT_HOLDINGS_TOL:
        issues.append(_issue("warn", "holdings_tol_loosened", f"positions may differ from the plan's assumed holdings by up to "
                             f"${holdings_tol:,.2f} each (the default is ${DEFAULT_HOLDINGS_TOL:,.2f})"))
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


def execute(plan, gateway, orders_dir, today, allow_closed=False, fill_timeout=30, poll_seconds=2, sleep=time.sleep,
            holdings_tol=DEFAULT_HOLDINGS_TOL, equity_floor=DEFAULT_EQUITY_FLOOR):
    """Run the checks, then place the plan. Returns the ledger dict. Raises ExecutionRefused before sending anything."""
    holdings_tol = check_holdings_tol(holdings_tol)
    equity_floor = check_equity_floor(equity_floor)
    path = _ledger_path(orders_dir, plan)
    issues = preflight(plan, gateway.account(), gateway.positions(), gateway.open_orders(), gateway.clock(), today,
                       ledger_exists=path.exists(), allow_closed=allow_closed, holdings_tol=holdings_tol,
                       equity_floor=equity_floor)
    if any(i["level"] == "error" for i in issues):
        raise ExecutionRefused(issues)

    ledger = dict(as_of=plan["as_of"], started=str(pd.Timestamp.now()), plan_equity=plan["equity"], status="running",
                  holdings_tol=holdings_tol, equity_floor=equity_floor, note="", warnings=[i for i in issues if i["level"] == "warn"], entries=[])
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

    # Buys are fixed dollar amounts, but the sells came in at today's prices, so the cash can be a little short of them.
    # A small shortfall comes off the largest buy; a big one means something is off, so no buy is sent.
    ledger["trimmed"] = []
    if buys:
        total = sum(float(o["notional"]) for o in buys)
        short = total - float(gateway.account().cash)
        if short > 1e-9:
            if short > MAX_BUY_SHORTFALL * total + 1e-9:
                ledger.update(status="aborted", note=f"the cash after the sells is short of the buys by ${short:,.2f} "
                              f"(limit {MAX_BUY_SHORTFALL:.0%}), so the buys were not sent")
                _write(path, ledger)
                return ledger
            cut = math.ceil(round(short * 100, 6)) / 100
            big = max(buys, key=lambda o: (float(o["notional"]), o["symbol"]))
            sent = round(float(big["notional"]) - cut, 2)
            ledger["trimmed"].append(dict(symbol=big["symbol"], planned=float(big["notional"]), sent=sent))
            buys = [dict(o, notional=sent) if o is big else o for o in buys]

    buy_entries = [place(o) for o in buys]

    # An order can be accepted and rejected a moment later, so look at every buy again before calling the run complete.
    pending, waited = {e["order_id"]: e for e in buy_entries if e["order_id"] and not e["error"]}, 0
    while pending and waited < fill_timeout:
        for oid in list(pending):
            status = gateway.order_status(oid)
            pending[oid]["status"] = status
            if status == "filled" or status in TERMINAL_BAD:
                del pending[oid]
        if pending:
            sleep(poll_seconds)
            waited += poll_seconds
    failed = [e for e in sell_entries + buy_entries if e["error"] or e["status"] in TERMINAL_BAD]
    ledger["status"] = "partial" if failed else "complete"
    notes = []
    if failed:
        notes.append("rejected: " + ", ".join(e["symbol"] for e in failed))
    if pending:
        notes.append("not filled yet when checked: " + ", ".join(e["symbol"] for e in pending.values()))
    ledger["note"] = "; ".join(notes)
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
def describe_plan(plan):
    """One line saying which settings the plan was made under, for the dry run."""
    p = plan.get("protocol")
    n = len(plan.get("weights", {}))
    if not isinstance(p, dict):
        return f"no settings recorded, {n} positions"
    ex = ", ".join(p.get("exclude") or []) or "none"
    cap = "none" if p.get("max_weight") is None else p["max_weight"]
    return (f"spy_share {p.get('spy_share')}, rule {p.get('qf_rule')}, cap {cap}, exposure {p.get('exposure')}, "
            f"exclude {ex}, {n} positions")


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
    ap.add_argument("--holdings-tol", default=DEFAULT_HOLDINGS_TOL, type=float,
                    help=f"dollars of drift allowed per position (default {DEFAULT_HOLDINGS_TOL:g}, at most {MAX_HOLDINGS_TOL:g})")
    ap.add_argument("--equity-floor", default=DEFAULT_EQUITY_FLOOR, type=float,
                    help=f"send nothing if the account's equity is below this many dollars (default {DEFAULT_EQUITY_FLOOR:g}; 0 = off)")
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    args = ap.parse_args()

    try:
        holdings_tol = check_holdings_tol(args.holdings_tol)
    except ValueError as e:
        print(f"Refusing: --holdings-tol: {e}")
        sys.exit(2)
    try:
        equity_floor = check_equity_floor(args.equity_floor)
    except ValueError as e:
        print(f"Refusing: --equity-floor: {e}")
        sys.exit(2)

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
                           ledger_exists=_ledger_path(args.orders_dir, plan).exists(), allow_closed=args.allow_closed,
                           holdings_tol=holdings_tol, equity_floor=equity_floor)
        print(f"DRY RUN of {plan_path.name} (equity ${plan['equity']:,.2f}). Nothing is sent.")
        print(f"Plan settings: {describe_plan(plan)}")
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
        ledger = execute(plan, gateway, args.orders_dir, today, allow_closed=args.allow_closed, holdings_tol=holdings_tol,
                         equity_floor=equity_floor)
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
# TDD RESULTS (pytest, 2026-10-24; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# same versions as the mlfinlab env; alpaca-py is faked in every test, so nothing here touched Alpaca)
# $ cd portfolio ; pytest test_execute_plan.py -v
#
# collected 82 items
# test_execute_plan.py::test_a_healthy_account_and_a_fresh_plan_have_no_issues PASSED [  1%]
# test_execute_plan.py::test_a_plan_that_is_not_version_2_is_refused PASSED [  2%]
# test_execute_plan.py::test_an_old_plan_is_refused_and_the_age_limit_is_adjustable PASSED [  3%]
# test_execute_plan.py::test_an_inactive_or_blocked_account_is_refused PASSED [  4%]
# test_execute_plan.py::test_account_equity_far_from_the_plan_equity_is_refused_but_a_small_drift_is_fine PASSED [  6%]
# test_execute_plan.py::test_open_orders_block_the_run_so_a_double_submit_is_impossible PASSED [  7%]
# test_execute_plan.py::test_holdings_that_differ_from_what_the_plan_assumed_are_refused_naming_the_symbols PASSED [  8%]
# test_execute_plan.py::test_a_plan_that_assumed_holdings_matches_when_the_account_holds_exactly_those PASSED [  9%]
# test_execute_plan.py::test_a_plan_that_was_already_executed_is_refused PASSED [ 10%]
# test_execute_plan.py::test_a_closed_market_is_an_error_unless_allowed_and_then_a_warning PASSED [ 12%]
# test_execute_plan.py::test_sells_cannot_be_sent_while_the_market_is_closed_even_if_closed_orders_are_allowed PASSED [ 13%]
# test_execute_plan.py::test_a_plan_with_no_orders_is_a_warning_not_an_error PASSED [ 14%]
# test_execute_plan.py::test_a_clean_run_submits_each_order_once_writes_a_ledger_and_reports_complete PASSED [ 15%]
# test_execute_plan.py::test_a_refused_preflight_raises_and_submits_nothing_and_writes_no_ledger PASSED [ 17%]
# test_execute_plan.py::test_running_the_same_plan_twice_is_refused_the_second_time PASSED [ 18%]
# test_execute_plan.py::test_a_rejected_buy_is_recorded_the_others_still_go_and_the_run_is_partial PASSED [ 19%]
# test_execute_plan.py::test_the_ledger_exists_before_the_first_order_so_a_crash_cannot_allow_a_second_run PASSED [ 20%]
# test_execute_plan.py::test_sells_go_first_and_buys_wait_until_the_sells_have_filled PASSED [ 21%]
# test_execute_plan.py::test_if_a_sell_does_not_fill_in_time_no_buy_is_sent PASSED [ 23%]
# test_execute_plan.py::test_a_rejected_sell_aborts_before_any_buy PASSED  [ 24%]
# test_execute_plan.py::test_a_sell_that_ends_canceled_or_rejected_aborts_before_any_buy PASSED [ 25%]
# test_execute_plan.py::test_the_fill_wait_polls_with_the_sleep_function_and_stops_at_the_timeout PASSED [ 26%]
# test_execute_plan.py::test_a_plan_with_no_orders_does_nothing_and_says_so PASSED [ 28%]
# test_execute_plan.py::test_a_closed_market_run_needs_allow_closed_and_then_records_the_warning PASSED [ 29%]
# test_execute_plan.py::test_reconcile_compares_each_target_dollar_amount_with_the_position PASSED [ 30%]
# test_execute_plan.py::test_reconcile_flags_a_position_the_plan_does_not_want_and_a_target_with_no_position PASSED [ 31%]
# test_execute_plan.py::test_reconcile_tolerance_is_the_larger_of_a_dollar_and_five_percent_of_target PASSED [ 32%]
# test_execute_plan.py::test_latest_plan_picks_the_newest_date_and_ignores_other_files PASSED [ 34%]
# test_execute_plan.py::test_load_plan_reads_the_json PASSED               [ 35%]
# test_execute_plan.py::test_the_gateway_uses_the_paper_api_and_converts_alpaca_objects_to_plain_values PASSED [ 36%]
# test_execute_plan.py::test_the_gateway_sends_notional_day_market_orders_and_closes_a_full_exit_with_close_position PASSED [ 37%]
# test_execute_plan.py::test_cli_default_is_a_dry_run_that_lists_the_orders_and_sends_nothing PASSED [ 39%]
# test_execute_plan.py::test_cli_dry_run_exits_1_and_says_why_when_a_check_fails PASSED [ 40%]
# test_execute_plan.py::test_cli_execute_without_the_matching_confirm_date_sends_nothing_and_exits_2 PASSED [ 41%]
# test_execute_plan.py::test_cli_execute_with_the_right_confirm_places_the_orders_and_exits_0 PASSED [ 42%]
# test_execute_plan.py::test_cli_a_partial_run_exits_1 PASSED              [ 43%]
# test_execute_plan.py::test_cli_reconcile_prints_the_table_and_sends_nothing PASSED [ 45%]
# test_execute_plan.py::test_cli_without_keys_exits_2_and_touches_nothing PASSED [ 46%]
# test_execute_plan.py::test_cli_with_no_plan_file_exits_2 PASSED          [ 47%]
# test_execute_plan.py::test_a_price_move_beyond_the_default_dollar_is_refused_and_a_looser_tolerance_lets_it_through_with_a_warning PASSED [ 48%]
# test_execute_plan.py::test_a_loosened_tolerance_still_refuses_a_move_larger_than_it_and_names_the_symbol PASSED [ 50%]
# test_execute_plan.py::test_a_tolerance_applies_to_every_symbol_not_just_the_first_one PASSED [ 51%]
# test_execute_plan.py::test_a_drift_exactly_equal_to_the_tolerance_is_accepted_and_one_cent_over_is_refused PASSED [ 52%]
# test_execute_plan.py::test_the_default_tolerance_and_a_tighter_one_add_no_loosened_warning PASSED [ 53%]
# test_execute_plan.py::test_the_tolerance_is_capped_and_a_negative_or_missing_number_is_rejected PASSED [ 54%]
# test_execute_plan.py::test_execute_passes_the_tolerance_through_places_the_orders_and_records_it_in_the_ledger PASSED [ 56%]
# test_execute_plan.py::test_execute_with_the_default_tolerance_still_refuses_the_same_drift_and_sends_nothing PASSED [ 57%]
# test_execute_plan.py::test_a_default_run_records_the_default_tolerance_in_its_ledger PASSED [ 58%]
# test_execute_plan.py::test_cli_dry_run_refuses_a_drift_by_default_and_passes_it_with_holdings_tol PASSED [ 59%]
# test_execute_plan.py::test_cli_execute_with_holdings_tol_places_the_orders_and_the_ledger_remembers_it PASSED [ 60%]
# test_execute_plan.py::test_cli_execute_without_holdings_tol_is_refused_for_the_same_drift_and_sends_nothing PASSED [ 62%]
# test_execute_plan.py::test_cli_rejects_a_negative_or_too_large_tolerance_before_touching_the_account PASSED [ 63%]
# test_execute_plan.py::test_the_module_can_only_reach_the_paper_api_and_never_uses_risky_order_features PASSED [ 64%]
# test_execute_plan.py::test_no_alpaca_import_happens_at_module_load PASSED [ 65%]
# test_execute_plan.py::test_the_default_floor_is_900_and_an_account_at_or_above_it_is_not_stopped PASSED [ 67%]
# test_execute_plan.py::test_equity_below_the_floor_is_refused_even_by_a_cent PASSED [ 68%]
# test_execute_plan.py::test_the_floor_applies_when_the_account_is_far_below_the_plan_too PASSED [ 69%]
# test_execute_plan.py::test_the_floor_is_adjustable_and_lowering_it_is_a_warning PASSED [ 70%]
# test_execute_plan.py::test_a_floor_of_zero_turns_it_off_and_says_so PASSED [ 71%]
# test_execute_plan.py::test_bad_floor_values_are_rejected PASSED          [ 73%]
# test_execute_plan.py::test_an_account_equity_that_is_not_a_number_is_refused PASSED [ 74%]
# test_execute_plan.py::test_execute_below_the_floor_sends_nothing_not_even_sells_and_writes_no_ledger PASSED [ 75%]
# test_execute_plan.py::test_execute_records_the_floor_in_the_ledger_and_a_lowered_floor_in_its_warnings PASSED [ 76%]
# test_execute_plan.py::test_cli_dry_run_below_the_floor_is_refused_and_a_lower_floor_passes PASSED [ 78%]
# test_execute_plan.py::test_cli_execute_below_the_floor_sends_nothing PASSED [ 79%]
# test_execute_plan.py::test_cli_rejects_a_bad_equity_floor_before_touching_the_account PASSED [ 80%]
# test_execute_plan.py::test_buys_that_fit_the_cash_are_sent_exactly_as_planned_and_cash_to_spare_is_not_spent PASSED [ 81%]
# test_execute_plan.py::test_cash_exactly_equal_to_the_buys_needs_no_trim PASSED [ 82%]
# test_execute_plan.py::test_a_small_shortfall_is_taken_off_the_largest_buy_only_to_the_cent_and_recorded PASSED [ 84%]
# test_execute_plan.py::test_a_shortfall_that_is_not_a_whole_cent_is_rounded_up_so_the_buys_never_exceed_the_cash PASSED [ 85%]
# test_execute_plan.py::test_with_equal_largest_buys_the_trim_is_deterministic PASSED [ 86%]
# test_execute_plan.py::test_a_shortfall_over_two_percent_of_the_buys_sends_no_buy_and_stops_for_a_person PASSED [ 87%]
# test_execute_plan.py::test_the_cut_off_is_exactly_two_percent PASSED     [ 89%]
# test_execute_plan.py::test_buys_are_checked_after_they_are_sent_and_a_rejected_one_makes_the_run_partial PASSED [ 90%]
# test_execute_plan.py::test_a_canceled_or_expired_buy_is_also_partial PASSED [ 91%]
# test_execute_plan.py::test_a_buy_still_working_after_the_wait_is_not_a_failure_but_is_noted_and_the_wait_is_bounded PASSED [ 92%]
# test_execute_plan.py::test_buys_only_plans_are_checked_too PASSED        [ 93%]
# test_execute_plan.py::test_a_buy_error_at_submit_and_a_later_rejection_are_both_named PASSED [ 95%]
# test_execute_plan.py::test_a_rejected_buy_ends_the_wait_at_once_and_is_not_also_listed_as_not_filled PASSED [ 96%]
# test_execute_plan.py::test_describe_plan_names_the_settings_the_positions_and_the_exclusions PASSED [ 97%]
# test_execute_plan.py::test_describe_plan_says_none_when_nothing_is_excluded_and_handles_plans_without_settings PASSED [ 98%]
# test_execute_plan.py::test_cli_dry_run_prints_the_plan_settings_line PASSED [100%]
# ============================== 82 passed in 0.55s ==============================
#
# plan settings line (2026-10-24, sandbox): 3 new tests. The dry run prints 'Plan settings: spy_share, rule, cap, exposure, exclude, N positions' so a plan left
#   from an old config is visible when you read the dry run.
#
# equity floor (2026-10-24, sandbox): 12 new tests. Account equity below $900 (default, --equity-floor) refuses the whole run, sells included;
#   exactly $900 passes, $899.99 is refused; a lower floor is a warning and 0 turns it off; a bad value (negative, nan, inf, text, True) is rejected;
#   a nan equity is refused; the floor is written into the ledger. Mutation checks (sandbox, 22): 20 caught; the 2 survivors are a duplicate
#   validation call in execute() (preflight validates the same value) and a nan-only difference in daily_run, neither changes real behavior.
#
# sell-path review (2026-10-24, sandbox): sells-first, wait-for-fill, abort-on-bad-sell and ledger-before-first-order held up. Two gaps fixed with 13 new tests:
#   buys were never re-checked after sending (a later rejection still read 'complete'), and sell proceeds at today's prices can leave the cash a few
#   dollars short of the fixed buy amounts. Now: shortfall up to 2% of the buys comes off the largest buy (rounded up to the cent, ledger 'trimmed'),
#   more than 2% sends no buy ('aborted'); every buy is looked at again and rejected/cancelled/expired makes the run 'partial'. Mutation checks (16): 16 caught.
# Alpaca treats accounts under $2,000 as limited margin, so sale proceeds are usable the same day (alpaca.markets/support/alpaca-cash-accounts).
#
# Mutation checks (sandbox, 34, run with bytecode caching off): each guard removed or weakened (plan version, plan age,
#   account status, blocked account, equity mismatch and its tolerance, open orders, holdings change and tolerance,
#   already-executed ledger, closed market, sells while closed, ledger written before the orders, buys before sells,
#   sell rejection / cancel / timeout not aborting the buys, polling never sleeping, rejected buy reported complete,
#   live instead of paper API, GTC instead of DAY, notional sent as qty, full exit without close_position, confirm not
#   required, dry run sending orders, keys not required, partial run exiting 0, reconcile tolerance and unwanted positions,
#   newest plan picking the oldest) made a test fail. One survived the first pass (a cancelled sell was caught only by the
#   timeout, not at once); the test now checks the note and that it does not wait.
# Holdings tolerance (2026-10-24): --holdings-tol added, tests written first (7 of the 12 new ones failed before the change).
#   Default stays $1; a looser value is warned about (holdings_tol_loosened), recorded in the ledger (holdings_tol) and
#   capped at $50. Mutation checks (sandbox, 17, run with bytecode caching off), all caught: tolerance ignored in
#   execute / CLI dry run / CLI execute, no warning, warning on the default or on a tighter value, no upper cap, negative
#   allowed, cap boundary excluded, cap or default constant changed, ledger not recording it, > turned into >=, only held
#   symbols checked, CLI or preflight not validating, CLI default changed. One was missing a test at first (the exact
#   boundary); a boundary test was added.
# Bytecode note: a same-length mutation made within one second can leave a stale .pyc that a later run reuses (one false
#   test failure showed it). The mutation scripts now clear __pycache__ and run with PYTHONDONTWRITEBYTECODE=1; the
#   mutation checks for check_fractionable.py and daily_run.py were re-run that way and still catch everything.
# Real plan (plan_v2_2026-10-08.json from the real snapshot): 48 buys, $1,000.00 total, executed against a fake gateway and
#   reconciled with every target within tolerance. That is a format check only.
# Run against the Alpaca paper API on 2026-10-20: the first 48 buys were placed and reconciled (all within 8 cents).
# The SELL path has NOT yet run live. Do the dry run first: python execute_plan.py
# ---------------------------------------------------------------------------
