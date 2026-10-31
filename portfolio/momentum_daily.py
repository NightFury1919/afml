"""Nightly runner for the momentum account: check the data, make today's plan, save it, write a report.

Plain English: like daily_run.py for the other accounts, but the plan comes from momentum_plan.py. It places NO
orders: execute_plan.py does that, by hand, with the floor and the order ledger. Until sleeve_n, sleeve_share and
start are filled in run_config.json the account stays inert ("not_configured") and no plan is made.

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env, with the momentum account's paper keys loaded:
    python momentum_daily.py --init-config --config accounts\\momentum\\run_config.json
    python momentum_daily.py --no-update --from-alpaca --config accounts\\momentum\\run_config.json \\
        --plans-dir accounts\\momentum\\plans --runs-dir accounts\\momentum\\runs
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from daily_run import _report_path, _step, fetch_account_inputs, health_check
from execute_plan import check_equity_floor
from momentum_plan import make_momentum_plan
from paper_runner_v2 import write_plan
from panel_data import load_prices
from update_prices import latest_snapshot

DEFAULTS = dict(sleeve_n=None, sleeve_share=None, every=21, start=None, equity=1000.0, equity_floor=500.0,
                min_notional=1.0)
PLAN_FIELDS = ("sleeve_n", "sleeve_share", "every", "start")


# ---------- config ----------
def load_config(path):
    """(config dict, found). A missing file gives the defaults. Unknown keys are rejected."""
    path = Path(path)
    cfg = dict(DEFAULTS)
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULTS, indent=2) + "\n", encoding="utf-8")
    return path


def config_problems(cfg):
    """Plain-English list of what stops a plan from being made; empty when the settings are usable."""
    out = []
    n = cfg.get("sleeve_n")
    if n is None:
        out.append("sleeve_n is not set")
    elif isinstance(n, bool) or not isinstance(n, int) or n < 1:
        out.append("sleeve_n must be a whole number of at least 1")
    s = cfg.get("sleeve_share")
    if s is None:
        out.append("sleeve_share is not set")
    elif isinstance(s, bool) or not isinstance(s, (int, float)) or not 0.0 <= s <= 1.0:
        out.append("sleeve_share must be between 0 and 1")
    if not cfg.get("start"):
        out.append("start is not set")
    e = cfg.get("every")
    if isinstance(e, bool) or not isinstance(e, int) or e < 1:
        out.append("every must be a whole number of at least 1")
    return out


def plan_mismatch(saved_path, cfg):
    """Names of the settings a saved plan was made with that differ from the config (empty = same)."""
    try:
        proto = json.loads(Path(saved_path).read_text(encoding="utf-8")).get("protocol") or {}
    except (ValueError, OSError):
        return ["unreadable"]
    want = dict(sleeve_n=cfg["sleeve_n"], sleeve_share=cfg["sleeve_share"], every=cfg["every"],
                start=str(pd.Timestamp(cfg["start"]).date()))
    key = dict(sleeve_n="sleeve_n", sleeve_share="sleeve_share", every="every", start="start")
    return [f for f in PLAN_FIELDS if proto.get(key[f]) != want[f]]


# ---------- the run ----------
def run_momentum_daily(folder, plans_dir, runs_dir, config, today, holdings=None, equity=None, config_found=True):
    """Run the steps and save the report. Returns the report dict (it also has report_path). Sends no orders."""
    config_equity = float(config["equity"])
    equity_floor = check_equity_floor(config.get("equity_floor", DEFAULTS["equity_floor"]))
    equity = float(config_equity if equity is None else equity)
    holdings = holdings or {}
    today = pd.Timestamp(today).normalize()
    steps, warnings = [], []
    if abs(equity / config_equity - 1.0) > 0.05:
        warnings.append(f"account equity ${equity:,.2f} differs from the config equity ${config_equity:,.2f} by more "
                        "than 5%; the plan is sized on the account equity")
    below_floor = not equity >= equity_floor
    if below_floor:
        warnings.insert(0, f"account equity ${equity:,.2f} is below the equity floor of ${equity_floor:,.2f}: "
                           "execute_plan.py will refuse to send any order until a person decides what to do")
    if not config_found:
        warnings.append("no run_config.json found: using the defaults (run --init-config to write one)")

    problems = config_problems(config)
    prices, snapshot, health, errors = None, None, [], []
    if not problems:
        try:
            snapshot = latest_snapshot(folder)
            prices = load_prices(snapshot)
            health = health_check(prices, today)
        except FileNotFoundError:
            health = [dict(level="error", code="no_data", ticker=None,
                           message="no prices_daily_asof_YYYY-MM-DD.csv snapshot found")]
        except ValueError as e:
            health = [dict(level="error", code="load_failed", ticker=None,
                           message=f"the prices file could not be loaded: {e}")]
        errors = [i for i in health if i["level"] == "error"]
        steps.append(_step("health_check", "failed" if errors else "ok",
                           f"{len(errors)} error(s), {len(health) - len(errors)} warning(s)"))

    as_of = prices.index[-1].date().isoformat() if prices is not None and len(prices) else None
    plan_summary, mismatch = None, []
    if problems:
        steps.append(_step("plan", "skipped", "not configured: " + "; ".join(problems)))
        status = "not_configured"
    elif errors:
        steps.append(_step("plan", "skipped", "skipped: the health check found errors"))
        status = "blocked"
    else:
        status = "ok"
        plan = make_momentum_plan(prices, holdings, equity, n=config["sleeve_n"], sleeve_share=config["sleeve_share"],
                                  start=config["start"], every=config["every"], min_notional=config["min_notional"],
                                  today=today)
        warnings += plan["warnings"]
        try:
            path = write_plan(plan, plans_dir)
            saved_now = True
            steps.append(_step("plan", "ok", f"saved {path.name}"))
        except FileExistsError:
            saved_now = False
            path = Path(plans_dir) / f"plan_v2_{plan['as_of']}.json"
            mismatch = plan_mismatch(path, config)
            if mismatch:
                warnings.insert(0, f"the saved plan {path.name} does not match the config ({', '.join(mismatch)}). "
                                   "Plans are never overwritten, so execute_plan.py would trade the saved plan. "
                                   "Move it aside and run again to make a new one, or change the config back.")
            steps.append(_step("plan", "skipped", f"plan already exists for {plan['as_of']}; it was not overwritten"))
        plan_summary = dict(equity=plan["equity"], rebalance_day=plan["rebalance_day"], next_signal=plan["next_signal"],
                            n_orders=len(plan["orders"]), cash_after=plan["cash_after"], plan_path=str(path),
                            saved_plan_is_this_run=saved_now,
                            top_weights=dict(sorted(plan["weights"].items(), key=lambda kv: -kv[1])[:5]))

    report = dict(status=status, run_date=today.date().isoformat(), as_of=as_of,
                  snapshot=snapshot.name if snapshot else None, config=config, config_problems=problems,
                  steps=steps, health=health, warnings=warnings, plan=plan_summary, orders_sent=0,
                  below_equity_floor=below_floor, equity_floor=equity_floor, saved_plan_mismatch=mismatch)
    stem = as_of or f"nodata_{today.date().isoformat()}"
    path = _report_path(runs_dir, stem)
    report["report_path"] = str(path)
    text = render_report(report)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(text, encoding="utf-8")
    return report


def render_report(r):
    lines = [f"# Momentum account: daily run {r['run_date']}  (prices as of {r['as_of']})", "",
             f"Status: {r['status'].upper()}.  No orders were sent: this run only plans and records.", ""]
    if r["status"] == "not_configured":
        lines += ["**Not configured yet:** " + "; ".join(r["config_problems"]) + ". No plan is made until these are set.", ""]
    if r.get("saved_plan_mismatch"):
        lines += [f"**SAVED PLAN DOES NOT MATCH THE CONFIG** ({', '.join(r['saved_plan_mismatch'])}).", ""]
    if r.get("below_equity_floor"):
        lines += [f"**BELOW THE EQUITY FLOOR of ${r['equity_floor']:,.2f}.** Trading is halted until a person decides.", ""]
    lines += ["## Steps"] + [f"- {s['name']}: {s['status']}" + (f" ({s['detail']})" if s["detail"] else "")
                             for s in r["steps"]]
    if r["health"]:
        lines += ["", "## Data health"] + [f"- {i['level'].upper()} {i['code']}: {i['message']}" for i in r["health"]]
    if r["warnings"]:
        lines += ["", "## Warnings"] + [f"- {w}" for w in r["warnings"]]
    p = r["plan"]
    if p:
        today_txt = "a rebalance day" if p["rebalance_day"] else "not a rebalance day (hold)"
        lines += ["", "## Plan", f"- equity ${p['equity']:,.2f}; today is {today_txt}",
                  f"- {p['n_orders']} order(s) it would place, cash after ${p['cash_after']:,.2f}",
                  f"- next rebalance signal about {p['next_signal']} (business days, holidays not counted)",
                  "- largest targets: " + ", ".join(f"{k} {v:.1%}" for k, v in p["top_weights"].items())]
        if p.get("saved_plan_is_this_run") is False:
            lines.append("- NOTE: these numbers are a fresh calculation. The saved plan file for this date is the earlier one.")
    c = r["config"]
    lines += ["", "## Settings", "- momentum sleeve: " + ", ".join(f"{k} {c.get(k)}" for k in PLAN_FIELDS)
              + f"; equity floor ${r['equity_floor']:,.2f}"]
    return "\n".join(lines) + "\n"


# ---------- the CLI ----------
def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here), help="folder holding the price snapshots")
    ap.add_argument("--plans-dir", default=str(here / "accounts" / "momentum" / "plans"))
    ap.add_argument("--runs-dir", default=str(here / "accounts" / "momentum" / "runs"))
    ap.add_argument("--config", default=str(here / "accounts" / "momentum" / "run_config.json"))
    ap.add_argument("--today", default=None, help="override today's date (YYYY-MM-DD), for testing")
    ap.add_argument("--no-update", action="store_true", help="accepted for symmetry with daily_run.py; prices are never pulled here")
    ap.add_argument("--from-alpaca", action="store_true", help="read equity and positions from the paper account")
    ap.add_argument("--init-config", action="store_true", help="write run_config.json once and stop")
    args = ap.parse_args(argv)

    if args.init_config:
        try:
            print(f"Wrote {init_config(args.config)}")
            return 0
        except FileExistsError as e:
            print(e)
            return 2
    equity, holdings = None, {}
    if args.from_alpaca:
        if not (os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY")):
            print("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys of the momentum account).")
            return 2
        equity, holdings = fetch_account_inputs()
    cfg, found = load_config(args.config)
    today = args.today or pd.Timestamp.now(tz="America/New_York").date().isoformat()
    rep = run_momentum_daily(args.folder, args.plans_dir, args.runs_dir, cfg, today, holdings=holdings,
                             equity=equity, config_found=found)
    print(f"MOMENTUM DAILY RUN {rep['run_date']}: {rep['status'].upper()}")
    print("No orders were sent. This run only plans and records.\n")
    print(render_report(rep))
    print(f"Saved {rep['report_path']}")
    return 0 if rep["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

# =============================================================================
# TDD RESULTS (2026-10-31)  sandbox: Linux, Python 3.11.18, numpy 1.23.5, pandas 1.5.3
# Command: cd portfolio ; python -m pytest test_momentum_daily.py -v
# Mutation check: 20 deliberate breakages in total (floor, 5% warning, inert-when-unconfigured, no-overwrite, mismatch flag,
# stale-data block, config validation); 4 survived the first round and were closed with new tests, then all were caught.
# Real-file smoke test (sandbox, temp folder, NOT the pre-registered test and no performance numbers): the 2026-10-19 price
# file loads, passes the health check and makes a plan (4 buys, $0.00 left over); the repo's accounts were not touched.
# Real-machine confirmation: DONE 2026-10-31 (Windows block at the bottom), plus a live read-only check of the
# momentum paper account: unconfigured run reported NOT_CONFIGURED, account ACTIVE, equity 1000, no positions.
# =============================================================================
# collecting ... collected 27 items
# test_momentum_daily.py::test_defaults_leave_the_account_inert_and_set_the_500_floor PASSED [ 3%]
# test_momentum_daily.py::test_missing_config_gives_defaults_and_found_false PASSED [ 7%]
# test_momentum_daily.py::test_unknown_config_key_is_rejected PASSED [ 11%]
# test_momentum_daily.py::test_init_config_writes_once_and_never_overwrites PASSED [ 14%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad0] PASSED [ 18%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad1] PASSED [ 22%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad2] PASSED [ 25%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad3] PASSED [ 29%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad4] PASSED [ 33%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad5] PASSED [ 37%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad6] PASSED [ 40%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad7] PASSED [ 44%]
# test_momentum_daily.py::test_unconfigured_account_makes_no_plan_but_still_reports PASSED [ 48%]
# test_momentum_daily.py::test_configured_signal_day_saves_a_plan_and_reports_it PASSED [ 51%]
# test_momentum_daily.py::test_equity_comes_from_the_config_unless_given PASSED [ 55%]
# test_momentum_daily.py::test_second_run_never_overwrites_the_saved_plan PASSED [ 59%]
# test_momentum_daily.py::test_saved_plan_made_with_other_settings_is_flagged PASSED [ 62%]
# test_momentum_daily.py::test_stale_prices_block_the_run_and_make_no_plan PASSED [ 66%]
# test_momentum_daily.py::test_missing_spy_price_blocks PASSED [ 70%]
# test_momentum_daily.py::test_no_price_file_blocks PASSED [ 74%]
# test_momentum_daily.py::test_below_the_500_floor_is_warned_first_and_flagged PASSED [ 77%]
# test_momentum_daily.py::test_bad_floor_value_is_refused PASSED [ 81%]
# test_momentum_daily.py::test_report_text_says_no_orders_and_shows_next_signal PASSED [ 85%]
# test_momentum_daily.py::test_a_share_of_exactly_zero_or_one_is_allowed PASSED [ 88%]
# test_momentum_daily.py::test_one_dollar_under_the_floor_still_counts_as_below PASSED [ 92%]
# test_momentum_daily.py::test_unconfigured_run_does_not_even_read_prices PASSED [ 96%]
# test_momentum_daily.py::test_equity_more_than_five_percent_off_the_config_is_warned PASSED [100%]
# ============================== 27 passed in 0.50s ==============================

# =============================================================================
# REAL-MACHINE RESULT (2026-10-31, Ethan's Windows PC, mlfinlab env), from C:\ws\AFML\portfolio
# Command: python -m pytest test_momentum_plan.py test_momentum_daily.py -v   (the 21 momentum_plan tests also PASSED)
# =============================================================================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# rootdir: C:\ws\AFML\portfolio
# collected 48 items
# test_momentum_daily.py::test_defaults_leave_the_account_inert_and_set_the_500_floor PASSED [ 45%]
# test_momentum_daily.py::test_missing_config_gives_defaults_and_found_false PASSED [ 47%]
# test_momentum_daily.py::test_unknown_config_key_is_rejected PASSED [ 50%]
# test_momentum_daily.py::test_init_config_writes_once_and_never_overwrites PASSED [ 52%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad0] PASSED [ 54%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad1] PASSED [ 56%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad2] PASSED [ 58%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad3] PASSED [ 60%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad4] PASSED [ 62%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad5] PASSED [ 64%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad6] PASSED [ 66%]
# test_momentum_daily.py::test_config_problems_flags_each_bad_setting[bad7] PASSED [ 68%]
# test_momentum_daily.py::test_unconfigured_account_makes_no_plan_but_still_reports PASSED [ 70%]
# test_momentum_daily.py::test_configured_signal_day_saves_a_plan_and_reports_it PASSED [ 72%]
# test_momentum_daily.py::test_equity_comes_from_the_config_unless_given PASSED [ 75%]
# test_momentum_daily.py::test_second_run_never_overwrites_the_saved_plan PASSED [ 77%]
# test_momentum_daily.py::test_saved_plan_made_with_other_settings_is_flagged PASSED [ 79%]
# test_momentum_daily.py::test_stale_prices_block_the_run_and_make_no_plan PASSED [ 81%]
# test_momentum_daily.py::test_missing_spy_price_blocks PASSED [ 83%]
# test_momentum_daily.py::test_no_price_file_blocks PASSED [ 85%]
# test_momentum_daily.py::test_below_the_500_floor_is_warned_first_and_flagged PASSED [ 87%]
# test_momentum_daily.py::test_bad_floor_value_is_refused PASSED [ 89%]
# test_momentum_daily.py::test_report_text_says_no_orders_and_shows_next_signal PASSED [ 91%]
# test_momentum_daily.py::test_a_share_of_exactly_zero_or_one_is_allowed PASSED [ 93%]
# test_momentum_daily.py::test_one_dollar_under_the_floor_still_counts_as_below PASSED [ 95%]
# test_momentum_daily.py::test_unconfigured_run_does_not_even_read_prices PASSED [ 97%]
# test_momentum_daily.py::test_equity_more_than_five_percent_off_the_config_is_warned PASSED [100%]
# === 48 passed in 1.67s ===
