"""Daily run orchestrator: one command from fresh prices to a saved, dated run report. Sends NO orders.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python daily_run.py --init-config                 # once: writes run_config.json (never overwrites)
    python daily_run.py                               # update prices, check them, plan, scoreboard, report
    python daily_run.py --no-update                   # skip the Yahoo pull and use the newest snapshot
    python daily_run.py --from-alpaca                 # read equity and positions from the PAPER account (read-only)

The steps, in order (each is recorded in the report)
    1. update_prices  : append new trading days to the price snapshot (update_prices.py rules).
                        A failed update is recorded, not fatal: step 2 decides whether the old data is still fresh enough.
    2. health_check   : stale data, bad index, non-positive prices, SPY missing, tickers missing on the last day,
                        suspicious one-day jumps. Any ERROR blocks the plan (status "blocked", exit code 1).
    3. plan           : the protocol v2 plan (paper_runner_v2.make_plan). The plan file for a date is never overwritten.
    4. scoreboard     : only if run_config.json has a start date. A future start is a note, not a crash.

The protocol lives in run_config.json, so every day uses the same rules. The report stores a hash of those rules
and shouts if they changed since the previous run: changing the rules after looking at results is how a research
project fools itself. `equity` is not part of the hash because it changes every day. `confirmed` lists the
placeholder settings (spy_share, max_weight) the boss has signed off; the rest are shown as still open.

Reports: runs/run_<as_of>.json and .md (a second run on the same day gets _r2, _r3, ...). Nothing is overwritten.
This module reads the Alpaca account only when asked (--from-alpaca) and never places, changes or cancels an order.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from paper_runner_v2 import PROTOCOL, make_plan, scoreboard, validate, write_plan
from panel_data import load_prices

DEFAULTS = dict(spy_share=0.5, qf_rule="size", max_weight=None, exposure=1.0, rebalance_every=1,
                start=None, equity=1000.0, confirmed=[])
HASH_FIELDS = ("spy_share", "qf_rule", "max_weight", "exposure", "rebalance_every", "start")
PLACEHOLDERS = ("spy_share", "max_weight")      # settings still waiting on the boss's answer
RUN_RE = re.compile(r"^run_(\d{4}-\d{2}-\d{2})(?:_r(\d+))?\.json$")


# ------------------------------------------------------------------------------ data health
def _issue(level, code, message, ticker=None):
    return dict(level=level, code=code, ticker=ticker, message=message)


def health_check(prices, today, max_lag=5, jump=0.25, lookback=10):
    """List of issues (dicts: level 'error'|'warn', code, ticker, message). Empty list = healthy.

    lag = business days between the last price date and the last business day on or before `today`.
    Lag 0 or 1 is normal (a weekend, or today's bar not settled yet); 2+ warns; `max_lag`+ is an error.
    Jumps are checked only over the last `lookback` rows (default 10): the aim is to catch a bad NEW bar after an
    update. Real history has genuine large moves (SLV fell 28.5% on 2026-01-30) that would warn every day for a year.
    """
    if prices is None or len(prices) == 0:
        return [_issue("error", "no_data", "there are no prices to check")]
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        return [_issue("error", "bad_index", "price dates are duplicated or not in increasing order")]

    issues = []
    last = prices.index[-1]
    today = pd.Timestamp(today).normalize()
    last_bd = np.busday_offset(today.date(), 0, roll="backward")
    lag = max(int(np.busday_count(last.date(), last_bd)), 0)
    if lag >= max_lag:
        issues.append(_issue("error", "stale_data", f"last price date {last.date()} is {lag} business days behind"))
    elif lag >= 2:
        issues.append(_issue("warn", "stale_data", f"last price date {last.date()} is {lag} business days behind"))

    for t in prices.columns:
        if (prices[t].dropna() <= 0).any():
            issues.append(_issue("error", "bad_price", f"{t} has a non-positive price", t))

    if "SPY" not in prices.columns or pd.isna(prices["SPY"].iloc[-1]):
        issues.append(_issue("error", "spy_missing", f"SPY has no price on {last.date()}", "SPY"))
    for t in prices.columns:
        if t != "SPY" and pd.isna(prices[t].iloc[-1]):
            issues.append(_issue("warn", "ticker_stale", f"{t} has no price on {last.date()}", t))

    tail = prices.tail(lookback + 1)
    rets = tail.where(tail > 0).pct_change(fill_method=None)
    for t in prices.columns:
        r = rets[t].dropna()
        if len(r) and (r.abs() > jump).any():
            d = r.abs().idxmax()
            issues.append(_issue("warn", "big_move", f"{t} moved {r[d]:+.1%} on {d.date()} (possible split or bad data)", t))
    return issues


# ------------------------------------------------------------------------------------ config
def load_config(path):
    """(config dict, found). A missing file gives the defaults and found=False. Unknown keys are rejected."""
    path = Path(path)
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    if not path.exists():
        return cfg, False
    given = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown key(s) in {path.name}: {', '.join(unknown)}")
    cfg.update(given)
    return cfg, True


def init_config(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; it is never overwritten")
    path.write_text(json.dumps(DEFAULTS, indent=2) + "\n", encoding="utf-8")
    return path


def config_hash(cfg):
    core = {k: cfg.get(k) for k in HASH_FIELDS}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


def protocol_from_config(cfg):
    protocol = dataclasses.replace(PROTOCOL, spy_share=cfg["spy_share"], qf_rule=cfg["qf_rule"],
                                   max_weight=cfg["max_weight"], exposure=cfg["exposure"],
                                   rebalance_every=cfg["rebalance_every"])
    validate(protocol)
    return protocol


# ------------------------------------------------------------------------------------ account
def account_to_inputs(account, positions):
    """(equity, {symbol: market value in dollars}) from alpaca-py Account and Position objects."""
    return float(account.equity), {p.symbol: float(p.market_value) for p in positions}


def fetch_account_inputs():
    """Read-only: one get_account and one get_all_positions on the PAPER account."""
    from alpaca.trading.client import TradingClient          # imported here so the tests need no alpaca-py

    client = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=True)
    return account_to_inputs(client.get_account(), client.get_all_positions())


# ------------------------------------------------------------------------------------ the run
def default_update(folder, universe=None):
    """The update_prices.py procedure, returning a one-line summary. Needs yfinance."""
    import update_prices as up
    from pull_prices import load_tickers

    folder = Path(folder)
    universe = Path(universe) if universe else folder / "universe.txt"
    existing = load_prices(up.latest_snapshot(folder))
    start = (existing.index.max() - pd.Timedelta(days=14)).date()
    fresh = up.drop_incomplete_today(up.pull_since(load_tickers(universe), start))
    merged, rep = up.reconcile_and_append(existing, fresh)
    notes = []
    if rep["rescaled"]:
        notes.append(f"re-adjusted {', '.join(sorted(rep['rescaled']))}")
    if rep["stale"]:
        notes.append(f"no new price for {', '.join(rep['stale'])}")
    try:
        path = up.write_snapshot(merged, folder)
        head = f"added {len(rep['new_dates'])} day(s), saved {path.name}"
    except FileExistsError:
        head = "nothing new: the snapshot for this last date already exists"
    return "; ".join([head] + notes)


def _step(name, status, detail=""):
    return dict(name=name, status=status, detail=detail)


def _previous_hash(runs_dir):
    best = None
    for p in Path(runs_dir).glob("run_*.json"):
        m = RUN_RE.match(p.name)
        if not m:
            continue
        key = (m.group(1), int(m.group(2) or 1))
        try:
            h = json.loads(p.read_text(encoding="utf-8")).get("config_hash")
        except (ValueError, OSError):
            continue
        if h and (best is None or key > best[0]):
            best = (key, h)
    return best[1] if best else None


def _report_path(runs_dir, stem):
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path, n = runs_dir / f"run_{stem}.json", 1
    while path.exists():
        n += 1
        path = runs_dir / f"run_{stem}_r{n}.json"
    return path


def run_daily(folder, plans_dir, runs_dir, config, today, holdings=None, update_fn=None,
              config_found=True, equity=None):
    """Run the four steps and save the report. Returns the report dict (it also has report_path)."""
    from update_prices import latest_snapshot

    protocol = protocol_from_config(config)
    config_equity = float(config["equity"])
    equity = float(config_equity if equity is None else equity)
    holdings = holdings or {}
    today = pd.Timestamp(today).normalize()
    steps, warnings = [], []
    if abs(equity / config_equity - 1.0) > 0.05:
        warnings.append(f"account equity ${equity:,.2f} differs from the config equity ${config_equity:,.2f} by more than 5%; "
                        "the plan is sized on the account equity")
    if not config_found:
        warnings.append("no run_config.json found: using the protocol defaults (run --init-config to write one)")

    if update_fn is None:
        steps.append(_step("update_prices", "skipped", "not requested"))
    else:
        try:
            detail = update_fn(folder)
            steps.append(_step("update_prices", "ok", "" if detail is None else str(detail)))
        except Exception as e:                                      # recorded, never fatal: the health check judges freshness
            steps.append(_step("update_prices", "failed", f"{type(e).__name__}: {e}"))

    prices, snapshot = None, None
    try:
        snapshot = latest_snapshot(folder)
        prices = load_prices(snapshot)
        health = health_check(prices, today)
    except FileNotFoundError:
        health = [_issue("error", "no_data", "no prices_daily_asof_YYYY-MM-DD.csv snapshot found")]
    except ValueError as e:
        health = [_issue("error", "load_failed", f"the prices file could not be loaded: {e}")]
    errors = [i for i in health if i["level"] == "error"]
    steps.append(_step("health_check", "failed" if errors else "ok",
                       f"{len(errors)} error(s), {len(health) - len(errors)} warning(s)"))

    as_of = prices.index[-1].date().isoformat() if prices is not None and len(prices) else None
    plan_summary = None
    if errors:
        steps.append(_step("plan", "skipped", "skipped: the health check found errors"))
    else:
        plan = make_plan(prices, holdings, equity, protocol, today=today)
        warnings += plan["warnings"]
        try:
            path = write_plan(plan, plans_dir)
            saved_now = True
            steps.append(_step("plan", "ok", f"saved {path.name}"))
        except FileExistsError:
            saved_now = False
            path = Path(plans_dir) / f"plan_v2_{plan['as_of']}.json"
            steps.append(_step("plan", "skipped", f"plan already exists for {plan['as_of']}; it was not overwritten"))
        plan_summary = dict(equity=plan["equity"], spy_dollars=plan["spy_dollars"], qf_dollars=plan["qf_dollars"],
                            n_orders=len(plan["orders"]), cash_after=plan["cash_after"], plan_path=str(path),
                            saved_plan_is_this_run=saved_now,
                            top_weights=dict(sorted(plan["weights"].items(), key=lambda kv: -kv[1])[:5]))

    sb_summary = None
    start = config.get("start")
    if not start:
        steps.append(_step("scoreboard", "skipped", "no start date in the config"))
    elif prices is None or errors:
        steps.append(_step("scoreboard", "skipped", "skipped: no usable prices"))
    elif pd.Timestamp(start) > prices.index[-1]:
        steps.append(_step("scoreboard", "skipped", f"no scoreboard yet: start {pd.Timestamp(start).date()} is after "
                           f"the last price date {prices.index[-1].date()}"))
    else:
        sb = scoreboard(prices, pd.Timestamp(start), protocol, equity)
        sb_summary = dict(start=sb.index[0].date().isoformat(), end=sb.index[-1].date().isoformat(), days=len(sb),
                          last_row={k: float(v) for k, v in sb.iloc[-1].items()})
        steps.append(_step("scoreboard", "ok", f"{len(sb)} days, no costs"))

    h, prev = config_hash(config), _previous_hash(runs_dir)
    changed = prev is not None and prev != h
    if changed:
        warnings.append(f"protocol changed since the previous run (hash {prev} -> {h}); do not read results across the change")

    report = dict(status="blocked" if errors else "ok", run_date=today.date().isoformat(), as_of=as_of,
                  snapshot=snapshot.name if snapshot else None, config=config, config_hash=h,
                  previous_config_hash=prev, config_changed=changed,
                  open_placeholders=sorted(set(PLACEHOLDERS) - set(config.get("confirmed", []))),
                  steps=steps, health=health, warnings=warnings, plan=plan_summary, scoreboard=sb_summary,
                  orders_sent=0)
    path = _report_path(runs_dir, as_of or f"nodata_{today.date().isoformat()}")
    report["report_path"] = str(path)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(r):
    lines = [f"# Daily run {r['run_date']}  (prices as of {r['as_of']})", "",
             f"Status: {r['status'].upper()}.  No orders were sent: this run only plans and records.", ""]
    lines += ["## Steps"] + [f"- {s['name']}: {s['status']}" + (f" ({s['detail']})" if s["detail"] else "") for s in r["steps"]]
    if r["health"]:
        lines += ["", "## Data health"] + [f"- {i['level'].upper()} {i['code']}: {i['message']}" for i in r["health"]]
    if r["warnings"]:
        lines += ["", "## Warnings"] + [f"- {w}" for w in r["warnings"]]
    p = r["plan"]
    if p:
        lines += ["", "## Plan", f"- equity ${p['equity']:,.2f}: SPY sleeve ${p['spy_dollars']:,.2f}, QF sleeve ${p['qf_dollars']:,.2f}",
                  f"- {p['n_orders']} order(s) it would place, cash after ${p['cash_after']:,.2f}",
                  "- largest targets: " + ", ".join(f"{k} {v:.1%}" for k, v in p["top_weights"].items())]
        if p.get("saved_plan_is_this_run") is False:
            lines.append("- NOTE: these numbers are a fresh calculation. The saved plan file for this date is the earlier one.")
    s = r["scoreboard"]
    if s:
        lines += ["", f"## Scoreboard {s['start']} to {s['end']} ({s['days']} days, no costs)"] + \
                 [f"- {k}: {v:.4f}" for k, v in s["last_row"].items()]
    lines += ["", "## Protocol", f"- hash {r['config_hash']}" + ("  (CHANGED since the previous run)" if r["config_changed"] else ""),
              "- " + ", ".join(f"{k} {r['config'].get(k)}" for k in HASH_FIELDS),
              "- still waiting on the boss: " + (", ".join(r["open_placeholders"]) or "nothing")]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------ the CLI
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here), help="folder holding the price snapshots")
    ap.add_argument("--plans-dir", default=str(here / "plans"))
    ap.add_argument("--runs-dir", default=str(here / "runs"))
    ap.add_argument("--config", default=str(here / "run_config.json"))
    ap.add_argument("--universe", default=None, help="default: universe.txt in --folder")
    ap.add_argument("--holdings-file", default=None, help="JSON {symbol: dollars held}")
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    ap.add_argument("--no-update", action="store_true", help="skip the Yahoo pull")
    ap.add_argument("--from-alpaca", action="store_true", help="read equity and positions from the paper account")
    ap.add_argument("--init-config", action="store_true", help="write run_config.json once and stop")
    args = ap.parse_args()

    if args.init_config:
        try:
            print(f"Wrote {init_config(args.config)}")
            sys.exit(0)
        except FileExistsError as e:
            print(e)
            sys.exit(2)

    equity, holdings = None, {}
    if args.holdings_file:
        holdings = json.loads(Path(args.holdings_file).read_text(encoding="utf-8"))
    if args.from_alpaca:
        if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
            print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys), as check_account.py does.")
            sys.exit(2)
        equity, holdings = fetch_account_inputs()

    cfg, found = load_config(args.config)
    today = args.today or pd.Timestamp.now(tz="America/New_York").date().isoformat()
    update_fn = None if args.no_update else (lambda folder: default_update(folder, args.universe))
    rep = run_daily(args.folder, args.plans_dir, args.runs_dir, cfg, today, holdings=holdings, update_fn=update_fn,
                    config_found=found, equity=equity)

    print(f"DAILY RUN {rep['run_date']}: {rep['status'].upper()}" + ("" if rep["status"] == "ok" else " (BLOCKED, no plan made)"))
    print("No orders were sent. This run only plans and records.\n")
    print(render_report(rep))
    print(f"Saved {rep['report_path']}")
    sys.exit(0 if rep["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# same versions as the mlfinlab env; Yahoo and Alpaca are faked in the tests, nothing here touched either)
# $ cd portfolio ; pytest test_daily_run.py -v
#
# collected 48 items
# test_daily_run.py::test_clean_data_has_no_issues PASSED                  [  2%]
# test_daily_run.py::test_saturday_with_a_friday_last_date_is_not_stale PASSED [  4%]
# test_daily_run.py::test_monday_with_a_friday_last_date_is_not_stale_because_todays_bar_is_not_settled_yet PASSED [  6%]
# test_daily_run.py::test_lag_is_measured_to_the_last_business_day_so_a_sunday_with_a_thursday_bar_is_one_day_behind_not_two PASSED [  8%]
# test_daily_run.py::test_two_business_days_behind_is_a_warning_and_a_week_behind_is_an_error PASSED [ 10%]
# test_daily_run.py::test_the_error_threshold_is_adjustable PASSED         [ 12%]
# test_daily_run.py::test_duplicate_dates_and_unsorted_dates_are_errors PASSED [ 14%]
# test_daily_run.py::test_a_non_positive_price_is_an_error_naming_the_ticker PASSED [ 16%]
# test_daily_run.py::test_spy_without_a_last_day_price_is_an_error_but_another_ticker_is_only_a_warning PASSED [ 18%]
# test_daily_run.py::test_a_thirty_percent_one_day_move_is_flagged_but_ten_percent_is_not PASSED [ 20%]
# test_daily_run.py::test_the_jump_threshold_is_adjustable PASSED          [ 22%]
# test_daily_run.py::test_only_the_recent_lookback_is_scanned_for_big_moves PASSED [ 25%]
# test_daily_run.py::test_an_empty_frame_is_an_error_not_a_crash PASSED    [ 27%]
# test_daily_run.py::test_config_hash_is_stable_and_changes_with_every_protocol_field_and_the_start_date PASSED [ 29%]
# test_daily_run.py::test_equity_is_not_part_of_the_hash_because_it_changes_every_day PASSED [ 31%]
# test_daily_run.py::test_a_missing_config_file_gives_the_defaults_and_says_so PASSED [ 33%]
# test_daily_run.py::test_init_config_writes_a_loadable_file_and_never_overwrites PASSED [ 35%]
# test_daily_run.py::test_an_unknown_key_in_the_config_is_rejected_so_typos_cannot_silently_change_the_protocol PASSED [ 37%]
# test_daily_run.py::test_protocol_from_config_applies_the_fields_and_validates_them PASSED [ 39%]
# test_daily_run.py::test_account_to_inputs_takes_equity_and_market_values_as_floats PASSED [ 41%]
# test_daily_run.py::test_account_with_no_positions_has_empty_holdings PASSED [ 43%]
# test_daily_run.py::test_a_normal_run_writes_a_plan_and_a_run_report_and_sends_nothing PASSED [ 45%]
# test_daily_run.py::test_the_steps_are_recorded_in_order PASSED           [ 47%]
# test_daily_run.py::test_the_update_step_runs_once_with_the_folder_and_its_result_is_recorded PASSED [ 50%]
# test_daily_run.py::test_a_failed_update_is_recorded_and_the_run_carries_on_with_the_existing_snapshot PASSED [ 52%]
# test_daily_run.py::test_a_failed_update_on_stale_data_blocks_the_plan_because_the_health_check_catches_it PASSED [ 54%]
# test_daily_run.py::test_an_error_level_health_issue_blocks_the_plan_and_says_why PASSED [ 56%]
# test_daily_run.py::test_a_prices_file_that_cannot_be_loaded_blocks_the_run_with_the_reason PASSED [ 58%]
# test_daily_run.py::test_no_snapshot_at_all_blocks_the_run_instead_of_crashing PASSED [ 60%]
# test_daily_run.py::test_running_twice_on_the_same_day_keeps_the_first_plan_and_does_not_overwrite_the_first_report PASSED [ 62%]
# test_daily_run.py::test_a_changed_protocol_between_runs_is_reported_loudly PASSED [ 64%]
# test_daily_run.py::test_the_scoreboard_start_in_the_future_is_a_note_not_a_crash PASSED [ 66%]
# test_daily_run.py::test_a_scoreboard_start_with_prices_reports_the_latest_row PASSED [ 68%]
# test_daily_run.py::test_no_start_date_skips_the_scoreboard PASSED        [ 70%]
# test_daily_run.py::test_holdings_and_equity_come_from_the_caller_and_a_second_account_changes_the_orders PASSED [ 72%]
# test_daily_run.py::test_placeholders_still_waiting_on_the_boss_are_listed_in_every_report PASSED [ 75%]
# test_daily_run.py::test_an_account_equity_far_from_the_config_equity_is_warned_about_because_the_plan_is_sized_on_it PASSED [ 77%]
# test_daily_run.py::test_the_report_says_when_its_plan_numbers_are_a_fresh_calculation_and_not_the_saved_file PASSED [ 79%]
# test_daily_run.py::test_a_missing_config_file_is_flagged_in_the_report PASSED [ 81%]
# test_daily_run.py::test_render_report_is_plain_text_that_says_nothing_was_sent_and_lists_issues PASSED [ 83%]
# test_daily_run.py::test_cli_with_no_update_runs_end_to_end_exits_0_and_prints_a_summary PASSED [ 85%]
# test_daily_run.py::test_cli_exits_1_when_the_run_is_blocked PASSED       [ 87%]
# test_daily_run.py::test_cli_init_config_writes_the_file_once PASSED      [ 89%]
# test_daily_run.py::test_cli_from_alpaca_uses_the_account_equity_and_positions_read_only PASSED [ 91%]
# test_daily_run.py::test_cli_from_alpaca_without_keys_exits_2_and_does_nothing PASSED [ 93%]
# test_daily_run.py::test_the_module_never_places_orders_and_imports_no_trading_client_at_load PASSED [ 95%]
# test_daily_run.py::test_default_update_appends_the_new_day_saves_a_new_snapshot_and_a_second_call_says_nothing_new PASSED [ 97%]
# test_daily_run.py::test_default_update_rejects_a_fresh_pull_that_disagrees_with_the_history PASSED [100%]
# ============================== 48 passed in 2.47s ==============================
#
# Mutation checks (sandbox, 34): stale thresholds, lag measured from today instead of the last business day, bad-index
#   / non-positive-price / SPY-missing / stale-ticker / big-move checks, jump and lookback ignored, config hash missing a
#   field or including equity, unknown config keys accepted, config overwrite, config not validated, failed update fatal,
#   errors not blocking the plan, plan or report overwritten, protocol change not reported, wrong previous run picked,
#   future scoreboard start unguarded, confirmed placeholders still listed, equity override ignored, live instead of paper
#   API, exit codes, missing keys not fatal, no-orders line dropped, unloadable file crashing instead of blocking.
#   plus: account equity far from config equity not warned, fresh-calculation flag/note wrong. One survived the first pass
#   (lag measured from today); a Sunday-with-a-Thursday-bar test was added and it is caught.
# Your Windows run (2026-10-17): --from-alpaca showed the paper account at $99,999.70, not $1,000, so the plan was sized on it and
#   the report still showed numbers from a plan file saved earlier at $1,000. Both fixed: an equity-mismatch warning and a
#   "fresh calculation" note, each with a test.
# Real snapshot (prices_daily_asof_2026-10-08.csv, 48 ETFs + SH, run with --today 2026-10-09 --no-update): status OK, 48
#   orders, cash after $0.00, SPY 51.0% of the account, a second run saved _r2 and reported the protocol change after the
#   config gained a start date; scoreboard 2026-09-30 to 2026-10-08 ran (a mechanics check, NOT a finding). A first version
#   scanned 252 days for big moves and warned every day about SLV -28.5% on 2026-01-30 (a genuine move); the default is now
#   the last 10 rows, since the aim is to catch a bad NEW bar.
# NOT YET RUN on your machine against live Yahoo (default_update) or the Alpaca paper account (--from-alpaca).
# ---------------------------------------------------------------------------
