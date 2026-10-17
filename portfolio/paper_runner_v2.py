"""Dry-run paper-trading runner v2: an S&P 500 sleeve against the QF sleeve (no model, no scores).

Run from C:\\ws\\AFML\\portfolio in the mlfinlab env:
    python paper_runner_v2.py                                    # empty account, $1,000, placeholder settings
    python paper_runner_v2.py --equity 1000 --holdings-file holdings.json
    python paper_runner_v2.py --spy-share 0.5 --qf-rule size --start 2026-10-19    # also print the scoreboard

Why v2 exists: the pooled and within-class real-history checks showed no predictive edge (see
preregistration_real_history_check*.md), so v2 drops the model. On 2026-10-13 the boss said to split the
investment and use the S&P 500 as the baseline to compete against the QF strategy. So the account is two sleeves:

    SPY sleeve   spy_share of equity, all in SPY, bought and held (the baseline).
    QF sleeve    the rest, spread over the 48 ETFs by a class-budget rule (non_predictive_weights, or HRP).

What it does, in order (all settings live in PROTOCOL below, see paper_protocol_v2.md)
    1. Loads the newest prices_daily_asof_*.csv and removes the SH hedge.
    2. qf_weights(): keeps the ETFs with a full trailing window of prices, applies the chosen rule, then the
       exposure scalar, then the per-ETF cap (a share of the QF SLEEVE, applied after exposure).
    3. combined_weights(): SPY sleeve + (1 - spy_share) * QF weights. SPY can appear in both sleeves.
    4. rebalance.rebalance_orders() turns the combined weights and your holdings into the orders it WOULD place.
    5. Writes plans/plan_v2_<last price date>.json. It never overwrites a plan.

scoreboard() is separate: a replay of the two sleeves from a start date, as a pure function of the price file
and the protocol. Weights on each day use only prices up to that day and earn the NEXT day's return. It shows
final dollars, cumulative return, the difference and drawdown for each sleeve. No costs, no cash interest.

What it does NOT do: it never sends an order and imports no trading client (a test checks that).
Holdings come from a file you provide, not from Alpaca.

Placeholders, not findings (the boss has not answered yet): spy_share = 0.5, no per-ETF cap, exposure 1.0,
size budgets, daily rebalance. The scoreboard is for watching the dry run go forward from a start date. Do not
use a replay over past dates to choose between rules: that is fitting on one history, which the project has
already done four times.
"""
import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from class_budgets import equal_budgets, inverse_vol_budgets, non_predictive_weights, size_budgets
from etf_classes import CLASS_NAMES, CLASS_OF
from hrp_allocation import hrp_weights, trailing_returns
from panel_data import drop_hedge, load_prices
from portfolio_construction import cap_weights
from rebalance import rebalance_orders

RULES = ("size", "equal", "inverse_vol", "hrp")


@dataclass(frozen=True)
class ProtocolV2:
    spy_share: float = 0.5          # share of equity in the SPY sleeve (placeholder: boss has not said)
    qf_rule: str = "size"           # "size", "equal", "inverse_vol" or "hrp" (size is the neutral default)
    max_weight: float = None        # per-ETF cap as a share of the QF sleeve; None = no cap
    exposure: float = 1.0           # QF sleeve exposure in [0, 1]; the rest of that sleeve stays cash
    window: int = 252               # trailing days for the vol and HRP rules and the "full history" test
    rebalance_every: int = 1        # trading days between rebalances (1 = daily)
    mode: str = "fractional"        # "fractional" (dollars) or "whole" (shares)
    min_notional: float = 1.0       # smallest dollar order worth sending
    max_stale_days: int = 5         # warn if the last price date is older than this


PROTOCOL = ProtocolV2()


def validate(protocol):
    if not 0 <= protocol.spy_share <= 1:
        raise ValueError("spy_share must be in [0, 1]")
    if protocol.qf_rule not in RULES:
        raise ValueError(f"qf_rule must be one of {RULES}, got {protocol.qf_rule!r}")
    if not 0 <= protocol.exposure <= 1:
        raise ValueError("exposure must be in [0, 1]")
    if protocol.max_weight is not None and not protocol.max_weight > 0:
        raise ValueError("max_weight must be positive")
    if int(protocol.window) != protocol.window or protocol.window < 1:
        raise ValueError("window must be a whole number of days >= 1")
    if int(protocol.rebalance_every) != protocol.rebalance_every or protocol.rebalance_every < 1:
        raise ValueError("rebalance_every must be a whole number of days >= 1")


def qf_weights(prices, protocol=PROTOCOL, groups=CLASS_OF):
    """{ETF: weight as a share of the QF sleeve}, long-only, from prices up to the last row only.

    Investable = in `groups`, not the hedge, and a price on each of the last window + 1 rows (so every rule
    sees the same set of ETFs). Order of steps: rule -> exposure -> cap.
    """
    validate(protocol)
    px = drop_hedge(prices)
    need = protocol.window + 1
    tail = px.iloc[-need:]
    tickers = [t for t in px.columns if t in groups and len(tail) == need and tail[t].notna().all()]
    if not tickers:
        raise ValueError(f"no ETF has a full window of {need} prices")

    if protocol.qf_rule == "hrp":
        w = hrp_weights(trailing_returns(px, tickers, protocol.window))
        w = {k: float(v) * protocol.exposure for k, v in w.items() if v * protocol.exposure > 0}
        if protocol.max_weight is not None and w:
            w = cap_weights(w, protocol.max_weight)
        return {k: float(v) for k, v in w.items()}

    if protocol.qf_rule == "size":
        budgets = size_budgets(groups, tickers)
    elif protocol.qf_rule == "equal":
        budgets = equal_budgets(groups, tickers)
    else:
        budgets = inverse_vol_budgets(tail[tickers], groups, window=protocol.window)
    w = non_predictive_weights(groups, budgets, tickers, exposure=protocol.exposure,
                               max_weight=protocol.max_weight)
    return {k: float(v) for k, v in w.items()}


def combined_weights(qf, protocol=PROTOCOL):
    """Account weights: spy_share in SPY plus (1 - spy_share) times the QF sleeve weights."""
    s = protocol.spy_share
    out = {t: (1.0 - s) * w for t, w in qf.items()}
    if s > 0:
        out["SPY"] = out.get("SPY", 0.0) + s
    return {t: float(w) for t, w in out.items() if w > 0}


def _trim_overspend(orders, cash_after, min_notional):
    """Each fractional order is rounded to the cent, and with many positions the rounding errors can add up to
    a few cents MORE than the account holds (48 equal positions on $1,000 overspend by $0.16). A broker could
    reject the last order for that. Trim the largest buy by exactly the overshoot. Nothing else changes."""
    if cash_after >= 0:
        return orders, cash_after
    buys = [o for o in orders if o["side"] == "buy" and "notional" in o]
    if not buys:
        return orders, cash_after
    big = max(buys, key=lambda o: (o["notional"], o["symbol"]))
    over = round(-cash_after, 2)
    new_notional = round(big["notional"] - over, 2)
    if new_notional < min_notional:
        return orders, cash_after
    return [dict(big, notional=new_notional) if o is big else o for o in orders], round(cash_after + over, 2)


def make_plan(prices, holdings, equity, protocol=PROTOCOL, today=None, groups=CLASS_OF):
    """Everything the runner decides, as a plain dict. Sends nothing."""
    validate(protocol)
    if not equity > 0:
        raise ValueError("equity must be positive")
    last = prices.index[-1]
    if protocol.spy_share > 0 and not ("SPY" in prices.columns and pd.notna(prices["SPY"].iloc[-1])):
        raise ValueError(f"SPY has no price on {last.date()}")
    qf = qf_weights(prices, protocol, groups)
    weights = combined_weights(qf, protocol)
    last_prices = prices.iloc[-1].dropna().to_dict()
    orders, cash_after = rebalance_orders(
        weights, holdings, equity, prices=last_prices if protocol.mode == "whole" else None,
        mode=protocol.mode, min_notional=protocol.min_notional)
    orders, cash_after = _trim_overspend(orders, cash_after, protocol.min_notional)

    today = pd.Timestamp.now().normalize() if today is None else pd.Timestamp(today).normalize()
    warnings = []
    age = (today - last).days
    if age > protocol.max_stale_days:
        warnings.append(f"price data is {age} days old (last date {last.date()})")
    return dict(version=2, as_of=last.date().isoformat(), protocol=dataclasses.asdict(protocol),
                spy_dollars=float(equity * protocol.spy_share),
                qf_dollars=float(equity * (1.0 - protocol.spy_share)),
                qf_weights=qf, weights=weights, equity=float(equity),
                holdings={k: float(v) for k, v in holdings.items()},
                orders=orders, cash_after=float(cash_after), warnings=warnings)


def write_plan(plan, folder):
    """Save the plan as plan_v2_<as_of>.json. Raises FileExistsError rather than overwrite one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"plan_v2_{plan['as_of']}.json"
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; plans are never overwritten")
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def _drawdown(values):
    peak = values.cummax().replace(0, np.nan)
    return 1.0 - values / peak


def scoreboard(prices, start, protocol=PROTOCOL, equity=1000.0, groups=CLASS_OF):
    """Replay the two sleeves from the first price date on or after `start`. One row per date.

    Columns: spy_value, qf_value (dollars), spy_return, qf_return (cumulative), qf_minus_spy (difference of
    those two cumulative returns), spy_drawdown, qf_drawdown (fall from the running peak, 0.2 = 20%).
    On each rebalance day the QF weights come from prices up to that day and earn the next day's return.
    Between rebalances the positions drift. A missing price on a held ETF is an error, never a zero return.
    """
    validate(protocol)
    if not equity > 0:
        raise ValueError("equity must be positive")
    px = drop_hedge(prices)
    dates = px.index
    pos0 = int(dates.searchsorted(pd.Timestamp(start)))
    if pos0 >= len(dates):
        raise ValueError(f"start {pd.Timestamp(start).date()} is after the last price date {dates[-1].date()}")
    spy_start, qf_start = equity * protocol.spy_share, equity * (1.0 - protocol.spy_share)
    if spy_start > 0 and "SPY" not in px.columns:
        raise ValueError("SPY has no prices")

    def step(series, pos, name):
        a, b = series.iloc[pos - 1], series.iloc[pos]
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"missing price for {name} on {dates[pos].date()}")
        return b / a

    spy_v, hold, cash = spy_start, {}, qf_start
    rows = []
    for k, pos in enumerate(range(pos0, len(dates))):
        if k > 0:
            if spy_start > 0:
                spy_v *= step(px["SPY"], pos, "SPY")
            hold = {t: v * step(px[t], pos, t) for t, v in hold.items()}
        if qf_start > 0 and k % protocol.rebalance_every == 0:
            total = sum(hold.values()) + cash
            w = qf_weights(px.iloc[:pos + 1], protocol, groups)
            hold = {t: x * total for t, x in w.items()}
            cash = total - sum(hold.values())
        rows.append((dates[pos], spy_v, sum(hold.values()) + cash))

    df = pd.DataFrame(rows, columns=["date", "spy_value", "qf_value"]).set_index("date")
    df["spy_return"] = df["spy_value"] / spy_start - 1.0 if spy_start > 0 else np.nan
    df["qf_return"] = df["qf_value"] / qf_start - 1.0 if qf_start > 0 else np.nan
    df["qf_minus_spy"] = df["qf_return"] - df["spy_return"]
    df["spy_drawdown"] = _drawdown(df["spy_value"])
    df["qf_drawdown"] = _drawdown(df["qf_value"])
    return df


def _class_shares(weights, groups=CLASS_OF):
    out = {c: 0.0 for c in CLASS_NAMES}
    for t, w in weights.items():
        if t in groups:
            out[groups[t]] += w
    return out


def main():
    from update_prices import latest_snapshot

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folder", default=str(here), help="folder holding the price snapshots")
    ap.add_argument("--plans-dir", default=str(here / "plans"))
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--holdings-file", default=None, help="JSON {symbol: dollars held}")
    ap.add_argument("--spy-share", type=float, default=PROTOCOL.spy_share)
    ap.add_argument("--qf-rule", default=PROTOCOL.qf_rule, choices=RULES)
    ap.add_argument("--max-weight", type=float, default=PROTOCOL.max_weight, help="per-ETF cap, share of the QF sleeve")
    ap.add_argument("--exposure", type=float, default=PROTOCOL.exposure)
    ap.add_argument("--rebalance-every", type=int, default=PROTOCOL.rebalance_every)
    ap.add_argument("--start", default=None, help="also print the scoreboard from this date (YYYY-MM-DD)")
    args = ap.parse_args()
    protocol = dataclasses.replace(PROTOCOL, spy_share=args.spy_share, qf_rule=args.qf_rule,
                                   max_weight=args.max_weight, exposure=args.exposure,
                                   rebalance_every=args.rebalance_every)

    src = latest_snapshot(args.folder)
    prices = load_prices(src)
    holdings = json.loads(Path(args.holdings_file).read_text(encoding="utf-8")) if args.holdings_file else {}
    plan = make_plan(prices, holdings, args.equity, protocol)

    print(f"DRY RUN. No orders are sent. Prices: {src.name}  (as of {plan['as_of']})")
    print(f"Settings: spy_share {protocol.spy_share:.0%}, QF rule {protocol.qf_rule}, cap {protocol.max_weight}, "
          f"exposure {protocol.exposure}, rebalance every {protocol.rebalance_every} day(s)")
    print(f"SPY sleeve ${plan['spy_dollars']:,.2f}   QF sleeve ${plan['qf_dollars']:,.2f}\n")
    print("QF sleeve, class shares: " + ", ".join(f"{c} {v:.1%}" for c, v in _class_shares(plan["qf_weights"]).items()))
    print("\nTarget weights (share of the whole account):")
    for k, v in sorted(plan["weights"].items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:6s} {v:6.1%}   ${v * plan['equity']:,.2f}")
    if len(plan["weights"]) > 15:
        print(f"  ... {len(plan['weights']) - 15} more")
    print("\nOrders it would place:")
    for o in plan["orders"] or [{"note": "none"}]:
        print("  ", o)
    print(f"\nCash after: ${plan['cash_after']:,.2f}")
    for w in plan["warnings"]:
        print("WARNING:", w)
    try:
        print(f"Saved {write_plan(plan, args.plans_dir)}")
    except FileExistsError as e:
        print(f"Plan not saved: {e}")

    if args.start:
        start = pd.Timestamp(args.start)
        if start > prices.index[-1]:
            print(f"\nSCOREBOARD: no scoreboard yet. Start {start.date()} is after the last price date "
                  f"{prices.index[-1].date()}; rerun with the same --start once prices through that date exist.")
        else:
            sb = scoreboard(prices, start, protocol, args.equity)
            print(f"\nSCOREBOARD from {sb.index[0].date()} to {sb.index[-1].date()} ({len(sb)} days, no costs)")
            print(sb.tail(5).round(4).to_string())


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-17; sandbox run: Linux, Python 3.10.20, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_paper_runner_v2.py -v
#
# collected 41 items
# test_paper_runner_v2.py::test_size_rule_holds_every_etf_equally PASSED   [  2%]
# test_paper_runner_v2.py::test_equal_rule_gives_each_class_the_same_share_then_splits_it_equally PASSED [  4%]
# test_paper_runner_v2.py::test_size_rule_on_unequal_classes_is_plain_equal_weight PASSED [  7%]
# test_paper_runner_v2.py::test_inverse_vol_gives_the_calmer_class_the_larger_budget PASSED [  9%]
# test_paper_runner_v2.py::test_inverse_vol_uses_only_the_trailing_window PASSED [ 12%]
# test_paper_runner_v2.py::test_hrp_rule_is_the_repo_hrp_weights_over_the_trailing_window PASSED [ 14%]
# test_paper_runner_v2.py::test_an_unknown_rule_is_rejected PASSED         [ 17%]
# test_paper_runner_v2.py::test_an_etf_without_a_full_window_of_prices_is_left_out PASSED [ 19%]
# test_paper_runner_v2.py::test_cap_moves_the_excess_to_the_others_in_proportion PASSED [ 21%]
# test_paper_runner_v2.py::test_exposure_scales_the_sleeve_and_the_cap_applies_to_the_final_weights PASSED [ 24%]
# test_paper_runner_v2.py::test_hrp_rule_honours_exposure_and_cap_too PASSED [ 26%]
# test_paper_runner_v2.py::test_combined_weights_add_the_spy_sleeve_to_the_scaled_qf_sleeve PASSED [ 29%]
# test_paper_runner_v2.py::test_all_spy_sleeve_is_just_spy_and_no_spy_sleeve_is_just_the_qf_book PASSED [ 31%]
# test_paper_runner_v2.py::test_cap_is_a_share_of_the_qf_sleeve_not_of_the_account PASSED [ 34%]
# test_paper_runner_v2.py::test_cash_left_by_exposure_stays_outside_both_sleeves PASSED [ 36%]
# test_paper_runner_v2.py::test_empty_account_plan_buys_both_sleeves_and_spends_all_the_cash PASSED [ 39%]
# test_paper_runner_v2.py::test_cent_rounding_never_spends_more_than_the_account_holds PASSED [ 41%]
# test_paper_runner_v2.py::test_the_cent_trim_leaves_an_exact_account_alone PASSED [ 43%]
# test_paper_runner_v2.py::test_running_again_with_the_resulting_holdings_orders_nothing PASSED [ 46%]
# test_paper_runner_v2.py::test_the_hedge_is_never_bought_and_a_held_hedge_is_closed PASSED [ 48%]
# test_paper_runner_v2.py::test_bad_settings_and_inputs_are_rejected PASSED [ 51%]
# test_paper_runner_v2.py::test_a_spy_sleeve_needs_a_spy_price_on_the_last_date PASSED [ 53%]
# test_paper_runner_v2.py::test_stale_data_produces_a_warning_but_still_a_plan PASSED [ 56%]
# test_paper_runner_v2.py::test_the_plan_records_the_protocol_it_was_made_under PASSED [ 58%]
# test_paper_runner_v2.py::test_making_a_plan_never_loads_any_trading_client PASSED [ 60%]
# test_paper_runner_v2.py::test_write_plan_round_trips_and_never_overwrites PASSED [ 63%]
# test_paper_runner_v2.py::test_first_row_is_the_start_date_with_each_sleeve_at_its_share_of_equity PASSED [ 65%]
# test_paper_runner_v2.py::test_the_sleeves_start_unequal_when_the_split_is_unequal PASSED [ 68%]
# test_paper_runner_v2.py::test_spy_sleeve_compounds_spy_and_a_daily_rebalanced_qf_sleeve_compounds_the_blend PASSED [ 70%]
# test_paper_runner_v2.py::test_without_rebalancing_the_qf_sleeve_drifts_with_the_winner PASSED [ 73%]
# test_paper_runner_v2.py::test_rebalancing_every_third_day_resets_the_weights_on_those_days PASSED [ 75%]
# test_paper_runner_v2.py::test_drawdown_is_the_fall_from_the_running_peak PASSED [ 78%]
# test_paper_runner_v2.py::test_the_scoreboard_never_looks_ahead PASSED    [ 80%]
# test_paper_runner_v2.py::test_scoreboard_matches_an_independent_day_by_day_reference PASSED [ 82%]
# test_paper_runner_v2.py::test_scoreboard_start_must_exist_and_inputs_are_checked PASSED [ 85%]
# test_paper_runner_v2.py::test_a_missing_price_while_holding_is_an_error_not_a_silent_zero PASSED [ 87%]
# test_paper_runner_v2.py::test_an_all_spy_split_leaves_the_qf_columns_undefined_not_divided_by_zero PASSED [ 90%]
# test_paper_runner_v2.py::test_whole_share_mode_spends_no_more_than_equity PASSED [ 92%]
# test_paper_runner_v2.py::test_cli_prints_the_plan_and_saves_it PASSED    [ 95%]
# test_paper_runner_v2.py::test_cli_scoreboard_with_a_start_date_that_has_prices_prints_rows PASSED [ 97%]
# test_paper_runner_v2.py::test_cli_says_so_instead_of_crashing_when_the_scoreboard_start_is_in_the_future PASSED [100%]
# ============================== 41 passed in 1.30s ==============================
#
# Mutation checks (sandbox, 19): SPY slice overwritten instead of added, QF weights not scaled by (1 - spy_share),
#   HRP ignoring exposure, HRP ignoring the cap, class rules ignoring exposure, class rules ignoring the cap, HRP
#   window ignored, scoreboard weights one day ahead, positions never drifting, rebalance_every ignored, missing price
#   as a zero return, plan overwrite allowed, any single price qualifying an ETF, drawdown from the trough, spy_share
#   not validated, overspend trim removed, overspend trim by the wrong amount, CLI future-start guard removed, sleeve dollars swapped each made a
#   test fail; original restored. Two tests were added after review: the one-day look-ahead mutation survived the
#   truncated-prices test (it cannot see a look-ahead of one day), so a day-by-day reference test was added.
# Real snapshot (prices_daily_asof_2026-10-08.csv, 48 ETFs): qf_weights for all 4 rules x {no cap, 5%, 10%} match the
#   earlier independent cap scenarios to 5e-5 (that file is rounded to 4 dp). The first real run found cash_after
#   -$0.16 (48 orders rounded up to the cent); fixed by trimming the largest buy, with a test. HRP with no cap leaves
#   ~$8.92 uninvested at $1,000 because 20 of its ETFs fall below the $1 minimum order (expected, not a bug).
# Windows mlfinlab env (2026-10-17, your run): 38 passed in 7.62s before the 3 CLI tests below were added; the real run on
#   prices_daily_asof_2026-10-16.csv gave cash_after $0.00 with the SPY order trimmed to $510.26. The CLI crashed on a
#   --start after the last price date (2026-10-19 vs 10-16); it now prints a note instead, with 3 CLI tests added.
# The scoreboard ran on the real snapshot as a mechanics check only. Its numbers are NOT a finding.
# ---------------------------------------------------------------------------
