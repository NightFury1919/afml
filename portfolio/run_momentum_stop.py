"""Runner for the stop-loss check on the momentum rule (pre-registration: preregistration_momentum_stop.md).

Refuses to run unless the pre-registration says `Status: FROZEN`, its SHA-256 line matches the price file, and no
results exist yet in the output folder (the check runs once). Reuses the ladder runner's helpers and the repo's
Chapter 3 `get_daily_vol`. Never run this before the pre-registration is committed and pushed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_ladder as ml
import momentum_stop as ms
import run_momentum_ladder as rl

SEED = 20261108
N_RESAMPLES = 5000
BLOCK = 4
LEVEL = 0.90
TOUCH_MULTIPLE = 2.0
WORST_LIMIT = -0.50


def spy_value(spy, dates):
    s = spy.reindex(dates).astype(float)
    return s / s.iloc[0]


def cut(series, since):
    """From `since` on, renormalised to start at 1.0 (the account is rebased, not re-run)."""
    s = series.loc[since:]
    return s / s.iloc[0]


def run_paths(etfs, spy, vol, cost_bps):
    stop = ms.simulate_stop(etfs, spy, vol, n=ms.SLEEVE_N, sleeve_share=ms.SLEEVE_SHARE, cost_bps=cost_bps)
    base = ml.simulate(etfs, spy, ms.SLEEVE_N, ms.SLEEVE_SHARE, cost_bps=cost_bps)
    return stop, base


def compare(a, b, n_resamples=N_RESAMPLES, seed=SEED):
    """Mean 21-day-period difference a minus b, times 12, with the paired moving-block interval."""
    diff = ml.period_returns(a).values - ml.period_returns(b).values
    return {"annual_diff": float(diff.mean() * 12.0),
            "ci": ml.block_bootstrap_interval(diff, BLOCK, n_resamples, seed, LEVEL),
            "n_periods": int(len(diff))}


def evaluate_range(stop_v, base_v, spy_v, n_resamples=N_RESAMPLES, seed=SEED):
    out = {}
    for name, v in (("C50s", stop_v), ("C50", base_v), ("SPY", spy_v)):
        s = ml.window_summary(v, ml.WINDOW, ml.FLOOR)
        s["touch_hard"] = float(ml.window_outcomes(v, ml.WINDOW, rl.HARD_FLOOR)[1].mean())
        s.update(rl.annual_stats(v))
        s["beat_share"], s["beat_avg"] = rl.beat_spy(v, spy_v, ml.WINDOW)
        out[name] = s
    out["C50s_minus_SPY"] = compare(stop_v, spy_v, n_resamples, seed)
    out["C50s_minus_C50"] = compare(stop_v, base_v, n_resamples, seed)
    return out


def verdict(full):
    """The four pass rules, on the full range at the main cost."""
    risk_ok = bool(full["C50s"]["touch_floor"] <= TOUCH_MULTIPLE * full["SPY"]["touch_floor"]
                   and full["C50s"]["worst"] >= WORST_LIMIT)
    beats = bool(full["C50s_minus_SPY"]["ci"][0] > 0)
    adds = bool(full["C50s_minus_C50"]["ci"][0] > 0)
    return {"risk_eligible": risk_ok, "beats_spy": beats, "stop_adds_value": adds,
            "recommended_for_live_trial": bool(risk_ok and beats)}


def analyse(etfs, spy, vol, since=rl.SINCE, n_resamples=N_RESAMPLES, seed=SEED, cost_main=ml.COST_BPS):
    results = {"seed": seed, "n_resamples": n_resamples, "since": since, "cost_bps_main": cost_main}
    for label, cost in (("main_cost", cost_main), ("zero_cost", 0.0)):
        stop, base = run_paths(etfs, spy, vol, cost)
        sv = spy_value(spy, stop.value.index)
        block = {"full": evaluate_range(stop.value, base.value, sv, n_resamples, seed),
                 "since": evaluate_range(cut(stop.value, since), cut(base.value, since), cut(sv, since),
                                         n_resamples, seed)}
        n_p, hit = ms.periods_with_stop(stop.stops, stop.trade_dates, stop.value.index[-1])
        block["stops"] = {"n_stops": len(stop.stops), "periods": n_p, "periods_with_stop": hit,
                          "share_periods_with_stop": hit / n_p if n_p else float("nan"),
                          "rebound": ms.stop_rebound_stats(etfs, stop.stops),
                          "superseded": int(stop.superseded),
                          "turnover_stop": float(stop.turnover.sum()), "turnover_base": float(base.turnover.sum()),
                          "cost_paid_stop": float(stop.cost_paid), "stop_leg_cost": float(stop.stop_cost_paid)}
        results[label] = block
    results["verdict"] = verdict(results["main_cost"]["full"])
    return results


def report_text(r):
    def row(name, s):
        return (f"{name:5s} ann {s['ann_return']*100:6.2f}%  vol {s['ann_vol']*100:5.2f}%  worst fall "
                f"{s['max_drawdown']*100:6.2f}%  worst 35d {s['worst']*100:6.2f}%  touch-10% {s['touch_floor']*100:5.1f}%  "
                f"touch-50% {s['touch_hard']*100:5.1f}%")
    lines = ["Stop-loss check on the momentum rule (C50s vs C50 vs SPY)", ""]
    for label in ("main_cost", "zero_cost"):
        for rng in ("full", "since"):
            blk = r[label][rng]
            lines.append(f"== {label}, range {rng} ==")
            for k in ("C50s", "C50", "SPY"):
                lines.append(row(k, blk[k]))
            for k in ("C50s_minus_SPY", "C50s_minus_C50"):
                c = blk[k]
                lines.append(f"{k}: {c['annual_diff']*100:+.2f} pts/yr, 90% interval {c['ci'][0]*100:+.2f} to "
                             f"{c['ci'][1]*100:+.2f} ({c['n_periods']} periods)")
            lines.append("")
        st = r[label]["stops"]
        lines.append(f"stops ({label}): {st['n_stops']} stops, {st['periods_with_stop']}/{st['periods']} periods with one, "
                     f"{st['superseded']} superseded by a rebalance; rebound: {st['rebound']}")
        lines.append(f"turnover stop {st['turnover_stop']:.2f} vs base {st['turnover_base']:.2f}; cost paid {st['cost_paid_stop']:.4f} "
                     f"(of which stop legs {st['stop_leg_cost']:.4f})")
        lines.append("")
    lines.append(f"verdict: {r['verdict']}")
    return "\n".join(lines) + "\n"


def run(price_file, prereg_file, out_dir, repo_root, n_resamples=N_RESAMPLES, get_daily_vol=None):
    price_file, prereg_file = Path(price_file), Path(prereg_file)
    out_dir = Path(out_dir)
    if (out_dir / "results.json").exists():
        raise FileExistsError(f"{out_dir / 'results.json'} already exists: this check runs once")
    rl.check_frozen(prereg_file.read_text(), rl.sha256_of(price_file))
    etfs, spy = rl.load_prices(price_file)
    f = get_daily_vol or ms.load_get_daily_vol(repo_root)
    vol = ms.daily_vol_frame(etfs, f)
    results = analyse(etfs, spy, vol, n_resamples=n_resamples)
    results["price_sha256"] = rl.sha256_of(price_file)
    rl.write_results(out_dir, results, report_text(results))
    return results


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prices", required=True)
    p.add_argument("--prereg", default="portfolio/preregistrations/preregistration_momentum_stop.md")
    p.add_argument("--out", default="portfolio/preregistrations/results_momentum_stop")
    p.add_argument("--repo", default=".")
    a = p.parse_args(argv)
    r = run(a.prices, a.prereg, a.out, a.repo)
    print(report_text(r))


if __name__ == "__main__":
    main()

# ---- TDD results (2026-11-08; Python 3.11, numpy 1.23.5, pandas 1.5.3, pytest 9.2.0; run against a copy of the repo's real ch03 get_daily_vol) ----
# Hand-worked cases on made-up prices; real prices not touched. Ethan's Windows run (mlfinlab env) before the last two test edits: 30 passed, 1 failed (test off by one row, fixed).
# Mutation checks: stop sign, '<' for '<=', one cost leg for two, superseded count were each caught. Two mutants cannot change any output.
# tests/test_momentum_stop.py::test_constants_match_preregistration PASSED
# tests/test_momentum_stop.py::test_stop_distance_hand_checked PASSED
# tests/test_momentum_stop.py::test_stop_triggers_and_sells_at_next_close_actual_price PASSED
# tests/test_momentum_stop.py::test_no_rebuy_before_next_rebalance_even_if_fund_rebounds PASSED
# tests/test_momentum_stop.py::test_costs_on_a_stop_sale_two_legs_into_shy PASSED
# tests/test_momentum_stop.py::test_proceeds_stay_in_cash_when_shy_has_no_price_and_only_one_leg_is_charged PASSED
# tests/test_momentum_stop.py::test_trigger_on_the_day_before_a_rebalance_is_dropped_and_equals_no_stop PASSED
# tests/test_momentum_stop.py::test_barrier_is_reset_at_every_rebalance_for_a_fund_that_is_bought_again PASSED
# tests/test_momentum_stop.py::test_two_funds_stopped_on_the_same_day_are_both_sold_next_close PASSED
# tests/test_momentum_stop.py::test_a_fall_in_spy_never_triggers_a_stop PASSED
# tests/test_momentum_stop.py::test_trigger_on_the_last_row_has_no_sale PASSED
# tests/test_momentum_stop.py::test_no_stop_when_volatility_is_missing_or_zero PASSED
# tests/test_momentum_stop.py::test_stop_needs_a_volatility_table_unless_it_is_off PASSED
# tests/test_momentum_stop.py::test_stop_off_equals_ladder_simulate_exactly PASSED
# tests/test_momentum_stop.py::test_a_stop_that_can_never_trigger_equals_the_baseline PASSED
# tests/test_momentum_stop.py::test_a_real_k_on_a_volatile_world_triggers_some_stops_and_keeps_the_books_consistent PASSED
# tests/test_momentum_stop.py::test_no_look_ahead_changing_future_prices_does_not_change_the_past PASSED
# tests/test_momentum_stop.py::test_daily_vol_frame_drops_missing_prices_and_realigns PASSED
# tests/test_momentum_stop.py::test_load_get_daily_vol_finds_the_function_in_a_fake_repo PASSED
# tests/test_momentum_stop.py::test_load_get_daily_vol_says_clearly_when_it_is_missing PASSED
# tests/test_momentum_stop.py::test_real_chapter3_get_daily_vol_is_causal_and_finite PASSED
# tests/test_momentum_stop.py::test_real_chapter3_get_daily_vol_on_daily_bars_is_about_a_third_above_one_day_volatility PASSED
# tests/test_momentum_stop.py::test_rebound_stats_hand_checked PASSED
# tests/test_momentum_stop.py::test_periods_with_stop_hand_checked PASSED
# tests/test_momentum_stop.py::test_close_exactly_on_the_barrier_counts_as_a_stop PASSED
# tests/test_run_momentum_stop.py::test_refuses_when_not_frozen PASSED
# tests/test_run_momentum_stop.py::test_refuses_when_price_file_differs PASSED
# tests/test_run_momentum_stop.py::test_end_to_end_then_refuses_second_run PASSED
# tests/test_run_momentum_stop.py::test_spy_is_not_in_the_universe_but_shy_is_exactly_as_in_the_ladder PASSED
# tests/test_run_momentum_stop.py::test_verdict_rules_hand_checked PASSED
# tests/test_run_momentum_stop.py::test_compare_hand_checked_mean_times_twelve PASSED
# tests/test_run_momentum_stop.py::test_cut_rebases_to_one PASSED
# ============================== 32 passed in 0.74s ==============================
