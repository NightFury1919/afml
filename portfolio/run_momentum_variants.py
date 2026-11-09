"""Runner for the trend-filter and group-cap check (pre-registration: preregistration_momentum_variants.md).

Refuses to run unless the pre-registration says `Status: FROZEN`, its price-file SHA-256 and its group-table SHA-256 match
the files given, and no results exist yet in the output folder (the check runs once). Never run this before the
pre-registration is committed and pushed.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_ladder as ml
import momentum_variants as mv
import run_momentum_ladder as rl

SEED = 20261109
N_RESAMPLES = 5000
BLOCK = 4
LEVEL = 0.90
TOUCH_MULTIPLE = 2.0
WORST_LIMIT = -0.50
VARIANTS = ("trend", "cap")


def load_groups(path):
    groups = json.loads(Path(path).read_text())
    if not isinstance(groups, dict) or not all(isinstance(v, str) for v in groups.values()):
        raise ValueError("group table must be a JSON object of ticker -> group name")
    return groups


def check_groups_frozen(prereg_text, groups_sha):
    m = re.search(r"Groups file SHA-256:\s*`?([0-9a-fA-F]{64})`?", prereg_text)
    if not m:
        raise RuntimeError("pre-registration has no 'Groups file SHA-256: <64 hex>' line")
    if m.group(1).lower() != groups_sha.lower():
        raise RuntimeError("the group table is different from the one frozen in the pre-registration")


def spy_value(spy, dates):
    s = spy.reindex(dates).astype(float)
    return s / s.iloc[0]


def cut(series, since):
    s = series.loc[since:]
    return s / s.iloc[0]


def compare(a, b, n_resamples=N_RESAMPLES, seed=SEED):
    diff = ml.period_returns(a).values - ml.period_returns(b).values
    return {"annual_diff": float(diff.mean() * 12.0),
            "ci": ml.block_bootstrap_interval(diff, BLOCK, n_resamples, seed, LEVEL),
            "n_periods": int(len(diff))}


def stats_of(v, spy_v):
    s = ml.window_summary(v, ml.WINDOW, ml.FLOOR)
    s["touch_hard"] = float(ml.window_outcomes(v, ml.WINDOW, rl.HARD_FLOOR)[1].mean())
    s.update(rl.annual_stats(v))
    s["beat_share"], s["beat_avg"] = rl.beat_spy(v, spy_v, ml.WINDOW)
    return s


def evaluate_range(values, spy_v, n_resamples=N_RESAMPLES, seed=SEED):
    """values: dict name -> account value series (must contain 'C50'); every variant is compared with SPY and with C50."""
    out = {name: stats_of(v, spy_v) for name, v in values.items()}
    out["SPY"] = stats_of(spy_v, spy_v)
    for name in values:
        if name == "C50":
            out[f"{name}_minus_SPY"] = compare(values[name], spy_v, n_resamples, seed)
            continue
        out[f"{name}_minus_SPY"] = compare(values[name], spy_v, n_resamples, seed)
        out[f"{name}_minus_C50"] = compare(values[name], values["C50"], n_resamples, seed)
    return out


def verdict(full, name):
    risk_ok = bool(full[name]["touch_floor"] <= TOUCH_MULTIPLE * full["SPY"]["touch_floor"]
                   and full[name]["worst"] >= WORST_LIMIT)
    beats = bool(full[f"{name}_minus_SPY"]["ci"][0] > 0)
    adds = bool(full[f"{name}_minus_C50"]["ci"][0] > 0)
    return {"risk_eligible": risk_ok, "beats_spy": beats, "adds_value_over_C50": adds,
            "recommended_for_live_trial": bool(risk_ok and beats)}


def run_paths(etfs, spy, groups, cost_bps):
    res = {"C50": mv.simulate_variant(etfs, spy, None, cost_bps=cost_bps)}
    res["trend"] = mv.simulate_variant(etfs, spy, "trend", cost_bps=cost_bps)
    res["cap"] = mv.simulate_variant(etfs, spy, "cap", groups, cost_bps=cost_bps)
    return res


def descriptives(res):
    n_trades = len(res["C50"].trade_dates)
    tr, cp = res["trend"], res["cap"]
    parked_picks = sum(len(v) for v in tr.filtered.values())
    return {"rebalances": n_trades,
            "trend": {"rebalances_with_a_parked_pick": len(tr.filtered), "picks_parked": parked_picks,
                      "turnover": float(tr.turnover.sum())},
            "cap": {"rebalances_where_cap_changed_picks": len(cp.skipped),
                    "turnover": float(cp.turnover.sum())},
            "C50": {"turnover": float(res["C50"].turnover.sum())}}


def analyse(etfs, spy, groups, since=rl.SINCE, n_resamples=N_RESAMPLES, seed=SEED, cost_main=ml.COST_BPS):
    results = {"seed": seed, "n_resamples": n_resamples, "since": since, "cost_bps_main": cost_main}
    for label, cost in (("main_cost", cost_main), ("zero_cost", 0.0)):
        res = run_paths(etfs, spy, groups, cost)
        values = {k: r.value for k, r in res.items()}
        sv = spy_value(spy, values["C50"].index)
        block = {"full": evaluate_range(values, sv, n_resamples, seed),
                 "since": evaluate_range({k: cut(v, since) for k, v in values.items()}, cut(sv, since), n_resamples, seed),
                 "descriptive": descriptives(res)}
        results[label] = block
    results["verdict"] = {name: verdict(results["main_cost"]["full"], name) for name in VARIANTS}
    return results


def report_text(r):
    def row(name, s):
        return (f"{name:5s} ann {s['ann_return']*100:6.2f}%  vol {s['ann_vol']*100:5.2f}%  worst fall "
                f"{s['max_drawdown']*100:6.2f}%  worst 35d {s['worst']*100:6.2f}%  touch-10% {s['touch_floor']*100:5.1f}%  "
                f"touch-50% {s['touch_hard']*100:5.1f}%")
    lines = ["Trend filter and group cap on the momentum rule (trial 10 = trend, trial 11 = cap)", ""]
    for label in ("main_cost", "zero_cost"):
        for rng in ("full", "since"):
            blk = r[label][rng]
            lines.append(f"== {label}, range {rng} ==")
            for k in ("trend", "cap", "C50", "SPY"):
                lines.append(row(k, blk[k]))
            for k in ("trend", "cap"):
                for other in ("SPY", "C50"):
                    c = blk[f"{k}_minus_{other}"]
                    lines.append(f"{k}_minus_{other}: {c['annual_diff']*100:+.2f} pts/yr, 90% interval "
                                 f"{c['ci'][0]*100:+.2f} to {c['ci'][1]*100:+.2f} ({c['n_periods']} periods)")
            lines.append("")
        lines.append(f"descriptive ({label}): {r[label]['descriptive']}")
        lines.append("")
    lines.append(f"verdict: {r['verdict']}")
    lines.append("Two variants were tested; a pass by one of them alone is weaker than the same pass in a single test.")
    return "\n".join(lines) + "\n"


def run(price_file, groups_file, prereg_file, out_dir, n_resamples=N_RESAMPLES):
    price_file, groups_file, prereg_file = Path(price_file), Path(groups_file), Path(prereg_file)
    out_dir = Path(out_dir)
    if (out_dir / "results.json").exists():
        raise FileExistsError(f"{out_dir / 'results.json'} already exists: this check runs once")
    text = prereg_file.read_text()
    rl.check_frozen(text, rl.sha256_of(price_file))
    check_groups_frozen(text, rl.sha256_of(groups_file))
    etfs, spy = rl.load_prices(price_file)
    groups = load_groups(groups_file)
    results = analyse(etfs, spy, groups, n_resamples=n_resamples)
    results["price_sha256"] = rl.sha256_of(price_file)
    results["groups_sha256"] = rl.sha256_of(groups_file)
    rl.write_results(out_dir, results, report_text(results))
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--groups", default="portfolio/preregistrations/momentum_groups.json")
    p.add_argument("--prereg", default="portfolio/preregistrations/preregistration_momentum_variants.md")
    p.add_argument("--out", default="portfolio/preregistrations/results_momentum_variants")
    a = p.parse_args(argv)
    print(report_text(run(a.prices, a.groups, a.prereg, a.out)))


if __name__ == "__main__":
    main()

# ---- TDD results (2026-11-08; sandbox Python 3.11, numpy 1.23.5, pandas 1.5.3, pytest 9.2.0; made-up prices only) ----
# Mutation checks: '>=' for '>' on the trend test, no cap skip, no SHY parking, cash dropped, min_periods=1, trend read from the
# trade day instead of the signal day (caught only after adding a test) were each caught by a failing test.
# tests/test_momentum_variants.py::test_trend_ok_table_hand_checked PASSED
# tests/test_momentum_variants.py::test_trend_ok_is_strict_greater_than PASSED
# tests/test_momentum_variants.py::test_failed_pick_is_parked_in_shy_hand_checked PASSED
# tests/test_momentum_variants.py::test_trend_test_uses_the_signal_day_not_the_trade_day PASSED
# tests/test_momentum_variants.py::test_parking_cost_two_legs_worth_of_turnover PASSED
# tests/test_momentum_variants.py::test_failed_pick_stays_cash_when_shy_has_no_price PASSED
# tests/test_momentum_variants.py::test_two_of_three_picks_pass_only_the_failed_slice_is_parked PASSED
# tests/test_momentum_variants.py::test_trend_variant_equals_ladder_when_every_pick_passes PASSED
# tests/test_momentum_variants.py::test_mode_none_equals_ladder_exactly_on_a_random_world PASSED
# tests/test_momentum_variants.py::test_select_capped_hand_checked PASSED
# tests/test_momentum_variants.py::test_select_capped_ties_are_broken_by_name PASSED
# tests/test_momentum_variants.py::test_cap_swaps_in_the_next_group_hand_checked PASSED
# tests/test_momentum_variants.py::test_cap_with_one_group_per_fund_equals_the_ladder PASSED
# tests/test_momentum_variants.py::test_cap_never_holds_two_funds_from_one_group_on_a_random_world PASSED
# tests/test_momentum_variants.py::test_cap_needs_a_complete_group_map PASSED
# tests/test_momentum_variants.py::test_bad_mode_is_refused PASSED
# tests/test_momentum_variants.py::test_no_look_ahead[trend] PASSED
# tests/test_momentum_variants.py::test_no_look_ahead[cap] PASSED
# tests/test_momentum_variants.py::test_trend_variant_filters_something_and_books_stay_consistent PASSED
# tests/test_run_momentum_variants.py::test_refuses_when_not_frozen PASSED
# tests/test_run_momentum_variants.py::test_refuses_when_price_file_differs PASSED
# tests/test_run_momentum_variants.py::test_refuses_when_group_table_differs PASSED
# tests/test_run_momentum_variants.py::test_refuses_when_no_group_hash_line PASSED
# tests/test_run_momentum_variants.py::test_end_to_end_then_refuses_second_run PASSED
# tests/test_run_momentum_variants.py::test_bad_group_table_is_refused PASSED
# tests/test_run_momentum_variants.py::test_verdict_rules_hand_checked PASSED
# tests/test_run_momentum_variants.py::test_compare_hand_checked_mean_times_twelve PASSED
# tests/test_run_momentum_variants.py::test_cut_rebases_to_one PASSED
# ============================== 28 passed in 0.77s ==============================
