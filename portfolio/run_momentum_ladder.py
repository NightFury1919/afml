"""Runner for the momentum risk ladder (pre-registration: preregistration_momentum_ladder.md).

Plain English: this loads the frozen price file, builds the 8 momentum mixes plus a 100% SPY
benchmark, measures every 7-week window, applies the pre-registered rules, and writes the
results ONCE. It refuses to run if the pre-registration is not frozen, if the price file is not
the one named in it, or if results already exist.

Run from the portfolio folder in the mlfinlab env:
    python run_momentum_ladder.py
"""
import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_ladder as ml

# ---- frozen by the pre-registration ----
SEED = 20261028
N_RESAMPLES = 5000
BLOCK = 4
LEVEL = 0.90
SINCE = "2016-01-04"
HARD_FLOOR = -0.50          # the boss's accepted loss: $500 of $1,000
WORST_LIMIT = -0.50         # worst 7-week window must be no worse than this
TOUCH_MULTIPLE = 2.0        # soft-floor (-10%) touch rate must be <= this x SPY's own
EXCLUDE = ("SH", "SPY")
TRADING_DAYS = 252


# ---------- inputs ----------
def newest_price_file(folder="."):
    files = sorted(glob.glob(str(Path(folder) / "prices_daily_asof_*.csv")))
    if not files:
        raise FileNotFoundError("no prices_daily_asof_*.csv found")
    return Path(files[-1])


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prices(path):
    """(ETF prices without SPY and SH, SPY prices), both indexed by date."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    spy = df["SPY"].astype(float)
    etfs = df.drop(columns=[c for c in EXCLUDE if c in df.columns]).astype(float)
    return etfs, spy


def check_frozen(prereg_text, price_sha):
    """Refuse unless the pre-registration says FROZEN and names this exact price file's hash."""
    if not re.search(r"(?m)^Status: FROZEN\s*$", prereg_text):
        raise RuntimeError("pre-registration is not frozen (needs a line 'Status: FROZEN')")
    m = re.search(r"SHA-256:\s*`?([0-9a-fA-F]{64})`?", prereg_text)
    if not m:
        raise RuntimeError("pre-registration has no 'SHA-256: <64 hex>' line for the price file")
    if m.group(1).lower() != price_sha.lower():
        raise RuntimeError("the price file is different from the one frozen in the pre-registration")


# ---------- small statistics ----------
def max_drawdown(value):
    v = np.asarray(value, dtype=float)
    return float((v / np.maximum.accumulate(v) - 1.0).min())


def annual_stats(value):
    v = np.asarray(value, dtype=float)
    steps = len(v) - 1
    daily = v[1:] / v[:-1] - 1.0
    return {"ann_return": float((v[-1] / v[0]) ** (TRADING_DAYS / steps) - 1.0),
            "ann_vol": float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "max_drawdown": max_drawdown(v)}


def beat_spy(value, spy_value, window=ml.WINDOW):
    """(share of windows where the strategy ends strictly above SPY, average amount ahead)."""
    a = np.asarray(value, dtype=float)
    b = np.asarray(spy_value, dtype=float)
    ea = a[window:] / a[:-window] - 1.0
    eb = b[window:] / b[:-window] - 1.0
    return float((ea > eb).mean()), float((ea - eb).mean())


def evaluate(value, spy_value, window=ml.WINDOW, n_resamples=N_RESAMPLES, seed=SEED):
    s = ml.window_summary(value, window, ml.FLOOR)
    s["touch_hard"] = float(ml.window_outcomes(value, window, HARD_FLOOR)[1].mean())
    s.update(annual_stats(value))
    s["beat_share"], s["beat_avg"] = beat_spy(value, spy_value, window)
    diff = ml.period_returns(value).values - ml.period_returns(spy_value).values
    s["diff_ci"] = ml.block_bootstrap_interval(diff, BLOCK, n_resamples, seed, LEVEL)
    return s


# ---------- the rules ----------
def variant_names():
    return [f"{s}{int(r * 100)}" for s in ml.SLEEVE_N for r in ml.RUNGS]


def decide(results, worst_limit=WORST_LIMIT, touch_multiple=TOUCH_MULTIPLE):
    spy = results["SPY"]
    limit = touch_multiple * spy["touch_floor"]
    out = {}
    for name in (n for n in variant_names() if n in results):
        r = results[name]
        out[name] = {"risk_eligible": bool(r["touch_floor"] <= limit and r["worst"] >= worst_limit),
                     "beats_spy": bool(r["diff_ci"][0] > 0),
                     "median_above_spy": bool(r["median"] > spy["median"])}
    rec = {}
    for sleeve in ml.SLEEVE_N:
        rec[sleeve] = None
        for rung in sorted(ml.RUNGS, reverse=True):
            name = f"{sleeve}{int(rung * 100)}"
            if name in out and out[name]["risk_eligible"] and out[name]["median_above_spy"]:
                rec[sleeve] = name
                break
    out["recommended"] = rec
    out["touch_limit"] = limit
    return out


# ---------- the run ----------
def run_all(etfs, spy, since=SINCE, n_resamples=N_RESAMPLES, seed=SEED, window=ml.WINDOW):
    res = {"full": {}, "since": {}}
    for label, bps in (("cost", ml.COST_BPS), ("nocost", 0.0)):
        sims = {"SPY": ml.simulate(etfs, spy, 10, 0.0, cost_bps=bps).value}
        for sleeve, n in ml.SLEEVE_N.items():
            for rung in ml.RUNGS:
                sims[f"{sleeve}{int(rung * 100)}"] = ml.simulate(etfs, spy, n, rung, cost_bps=bps).value
        for rng, keep in (("full", lambda v: v), ("since", lambda v: v[v.index >= pd.Timestamp(since)])):
            vals = {k: keep(v) for k, v in sims.items()}
            res[rng][label] = {k: evaluate(v, vals["SPY"], window, n_resamples, seed)
                               for k, v in vals.items()}
            if label == "cost":
                res[f"{rng}_start"] = str(vals["SPY"].index[0].date())
    res["first_trade"] = res["full_start"]
    res["since_start"] = res["since_start"]
    return res


def format_report(res, decisions, price_name, sha):
    L = [f"Momentum risk ladder: results (price file {price_name}, SHA-256 {sha})",
         f"First trade date {res['first_trade']}; 'since' range starts {res['since_start']}", ""]
    cols = ("worst", "p05", "median", "p95", "best", "share_gt_100", "end_below_floor",
            "touch_floor", "touch_hard", "ann_return", "ann_vol", "max_drawdown", "beat_share")
    for rng in ("full", "since"):
        for label in ("cost", "nocost"):
            L.append(f"== range={rng} {label} ==")
            L.append("name   " + " ".join(f"{c[:11]:>11}" for c in cols) + "   diff_ci(ann, 90%)")
            for name, r in res[rng][label].items():
                L.append(f"{name:<6} " + " ".join(f"{r[c]:>11.3f}" for c in cols)
                         + f"   [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]")
            L.append("")
    L.append(f"Rules applied to range=full cost. Touch-floor limit = {decisions['touch_limit']:.3f}")
    for name in variant_names():
        d = decisions[name]
        L.append(f"{name:<6} risk_eligible={d['risk_eligible']}  beats_spy={d['beats_spy']}  "
                 f"median_above_spy={d['median_above_spy']}")
    L.append(f"Recommended rung per sleeve: {decisions['recommended']}")
    return "\n".join(L)


def write_results(out_dir, results, report):
    out = Path(out_dir)
    if (out / "results.json").exists():
        raise FileExistsError(f"{out / 'results.json'} already exists: this check runs once")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "report.txt").write_text(report)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", help="default: newest prices_daily_asof_*.csv in this folder")
    ap.add_argument("--prereg", default="preregistration_momentum_ladder.md")
    ap.add_argument("--out", default="momentum_results")
    a = ap.parse_args(argv)
    path = Path(a.prices) if a.prices else newest_price_file()
    sha = sha256_of(path)
    check_frozen(Path(a.prereg).read_text(), sha)
    if (Path(a.out) / "results.json").exists():
        raise FileExistsError("results already exist: this check runs once")
    etfs, spy = load_prices(path)
    res = run_all(etfs, spy)
    decisions = decide(res["full"]["cost"])
    res["decisions"] = decisions
    res["price_file"], res["sha256"] = path.name, sha
    report = format_report(res, decisions, path.name, sha)
    write_results(a.out, res, report)
    print(report)


if __name__ == "__main__":
    main()

# =============================================================================
# TDD RESULTS (2026-10-31)  python 3.11.18, numpy 1.23.5, pandas 1.5.3  (sandbox, Linux)
# Command: python -m pytest test_run_momentum_ladder.py -v   (47 tests with test_momentum_ladder.py all pass)
# Smoke run on a made-up world only (never on the real prices): full pipeline OK, second run refused.
# The real draft pre-registration is refused by check_frozen. Real-machine confirmation: DONE 2026-10-31, see the Windows block at the bottom.
# =============================================================================
# platform linux -- Python 3.11.18, pytest-9.2.0, pluggy-1.7.0 -- python
# collecting ... collected 20 items
# test_run_momentum_ladder.py::test_load_prices_drops_spy_and_sh_from_universe PASSED [  5%]
# test_run_momentum_ladder.py::test_newest_price_file_picks_latest_date PASSED [ 10%]
# test_run_momentum_ladder.py::test_sha256_of_known_bytes PASSED           [ 15%]
# test_run_momentum_ladder.py::test_refuses_when_preregistration_is_a_draft PASSED [ 20%]
# test_run_momentum_ladder.py::test_draft_that_merely_mentions_the_frozen_line_is_still_refused PASSED [ 25%]
# test_run_momentum_ladder.py::test_refuses_when_hash_line_missing PASSED  [ 30%]
# test_run_momentum_ladder.py::test_refuses_when_price_file_differs_from_the_frozen_hash PASSED [ 35%]
# test_run_momentum_ladder.py::test_accepts_frozen_with_matching_hash PASSED [ 40%]
# test_run_momentum_ladder.py::test_max_drawdown_hand_checked PASSED       [ 45%]
# test_run_momentum_ladder.py::test_annual_stats_constant_growth PASSED    [ 50%]
# test_run_momentum_ladder.py::test_beat_spy_share_and_average PASSED      [ 55%]
# test_run_momentum_ladder.py::test_evaluate_reports_both_floors_and_bootstrap_for_a_flat_world PASSED [ 60%]
# test_run_momentum_ladder.py::test_evaluate_hard_floor_vs_soft_floor PASSED [ 65%]
# test_run_momentum_ladder.py::test_decide_risk_eligibility_and_beats PASSED [ 70%]
# test_run_momentum_ladder.py::test_decide_recommends_largest_qualifying_rung_per_sleeve PASSED [ 75%]
# test_run_momentum_ladder.py::test_run_all_builds_every_variant_range_and_cost PASSED [ 80%]
# test_run_momentum_ladder.py::test_since_range_starts_at_or_after_the_date PASSED [ 85%]
# test_run_momentum_ladder.py::test_write_results_then_refuses_second_run PASSED [ 90%]
# test_run_momentum_ladder.py::test_main_refuses_a_draft_before_touching_anything PASSED [ 95%]
# test_run_momentum_ladder.py::test_run_constants_match_preregistration PASSED [100%]
# ============================== 20 passed in 0.74s ==============================


# =============================================================================
# REAL-MACHINE RESULT (2026-10-31, Ethan's Windows PC, mlfinlab env), from C:\ws\AFML\portfolio
# Command: python -m pytest test_momentum_ladder.py test_run_momentum_ladder.py -v
# =============================================================================
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# rootdir: C:\ws\AFML\portfolio
# collected 47 items
# test_momentum_ladder.py::test_defaults_match_preregistration PASSED [ 2%]
# test_momentum_ladder.py::test_momentum_score_value_and_warmup PASSED [ 4%]
# test_momentum_ladder.py::test_momentum_score_needs_price_today_and_lookback_ago PASSED [ 6%]
# test_momentum_ladder.py::test_momentum_score_uses_no_future_prices PASSED [ 8%]
# test_momentum_ladder.py::test_select_top_n_orders_by_score PASSED [ 10%]
# test_momentum_ladder.py::test_select_top_n_ties_broken_by_name PASSED [ 12%]
# test_momentum_ladder.py::test_select_top_n_skips_nan_and_handles_short_universe PASSED [ 14%]
# test_momentum_ladder.py::test_first_signal_position_needs_min_eligible PASSED [ 17%]
# test_momentum_ladder.py::test_first_signal_position_none_when_never_enough PASSED [ 19%]
# test_momentum_ladder.py::test_sleeve_only_earns_best_etf_growth_after_trade_date PASSED [ 21%]
# test_momentum_ladder.py::test_zero_sleeve_equals_spy PASSED [ 23%]
# test_momentum_ladder.py::test_half_and_half_resets_at_each_rebalance PASSED [ 25%]
# test_momentum_ladder.py::test_holdings_use_only_data_up_to_signal_date PASSED [ 27%]
# test_momentum_ladder.py::test_equal_weight_across_n_holdings PASSED [ 29%]
# test_momentum_ladder.py::test_first_purchase_cost_charged_on_traded_value PASSED [ 31%]
# test_momentum_ladder.py::test_no_cost_when_rebalance_changes_nothing PASSED [ 34%]
# test_momentum_ladder.py::test_switching_holdings_charges_the_turnover PASSED [ 36%]
# test_momentum_ladder.py::test_window_outcomes_hand_checked PASSED [ 38%]
# test_momentum_ladder.py::test_window_summary_hand_checked PASSED [ 40%]
# test_momentum_ladder.py::test_touch_counts_exactly_at_the_floor PASSED [ 42%]
# test_momentum_ladder.py::test_simulate_defaults_match_preregistration PASSED [ 44%]
# test_momentum_ladder.py::test_window_summary_matches_spy_script_method PASSED [ 46%]
# test_momentum_ladder.py::test_window_summary_too_short_series PASSED [ 48%]
# test_momentum_ladder.py::test_period_returns_every_21_rows PASSED [ 51%]
# test_momentum_ladder.py::test_block_bootstrap_constant_difference_gives_that_constant PASSED [ 53%]
# test_momentum_ladder.py::test_block_bootstrap_is_deterministic_for_a_seed PASSED [ 55%]
# test_momentum_ladder.py::test_block_bootstrap_interval_covers_the_true_mean_for_a_clear_edge PASSED [ 57%]
# test_run_momentum_ladder.py::test_load_prices_drops_spy_and_sh_from_universe PASSED [ 59%]
# test_run_momentum_ladder.py::test_newest_price_file_picks_latest_date PASSED [ 61%]
# test_run_momentum_ladder.py::test_sha256_of_known_bytes PASSED [ 63%]
# test_run_momentum_ladder.py::test_refuses_when_preregistration_is_a_draft PASSED [ 65%]
# test_run_momentum_ladder.py::test_draft_that_merely_mentions_the_frozen_line_is_still_refused PASSED [ 68%]
# test_run_momentum_ladder.py::test_refuses_when_hash_line_missing PASSED [ 70%]
# test_run_momentum_ladder.py::test_refuses_when_price_file_differs_from_the_frozen_hash PASSED [ 72%]
# test_run_momentum_ladder.py::test_accepts_frozen_with_matching_hash PASSED [ 74%]
# test_run_momentum_ladder.py::test_max_drawdown_hand_checked PASSED [ 76%]
# test_run_momentum_ladder.py::test_annual_stats_constant_growth PASSED [ 78%]
# test_run_momentum_ladder.py::test_beat_spy_share_and_average PASSED [ 80%]
# test_run_momentum_ladder.py::test_evaluate_reports_both_floors_and_bootstrap_for_a_flat_world PASSED [ 82%]
# test_run_momentum_ladder.py::test_evaluate_hard_floor_vs_soft_floor PASSED [ 85%]
# test_run_momentum_ladder.py::test_decide_risk_eligibility_and_beats PASSED [ 87%]
# test_run_momentum_ladder.py::test_decide_recommends_largest_qualifying_rung_per_sleeve PASSED [ 89%]
# test_run_momentum_ladder.py::test_run_all_builds_every_variant_range_and_cost PASSED [ 91%]
# test_run_momentum_ladder.py::test_since_range_starts_at_or_after_the_date PASSED [ 93%]
# test_run_momentum_ladder.py::test_write_results_then_refuses_second_run PASSED [ 95%]
# test_run_momentum_ladder.py::test_main_refuses_a_draft_before_touching_anything PASSED [ 97%]
# test_run_momentum_ladder.py::test_run_constants_match_preregistration PASSED [100%]
# === 47 passed in 1.96s ===

# =============================================================================
# LATER CHANGE (2026-10-31, after the Windows run above): check_frozen now compares the hash case-insensitively,
# because PowerShell's Get-FileHash prints UPPERCASE. One test added (test_hash_comparison_ignores_case_...).
# Sandbox: 48 passed (Python 3.11.18). Windows re-run of the 48: DONE 2026-10-31, 48 passed (below).
# =============================================================================
# Windows re-run, run_momentum_ladder tests only shown (the 28 in test_momentum_ladder.py also PASSED; 48 passed in 2.02s):
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\earob\miniconda3\envs\mlfinlab\python.exe
# rootdir: C:\ws\AFML\portfolio
# collected 48 items
# test_run_momentum_ladder.py::test_load_prices_drops_spy_and_sh_from_universe PASSED [ 58%]
# test_run_momentum_ladder.py::test_newest_price_file_picks_latest_date PASSED [ 60%]
# test_run_momentum_ladder.py::test_sha256_of_known_bytes PASSED [ 62%]
# test_run_momentum_ladder.py::test_refuses_when_preregistration_is_a_draft PASSED [ 64%]
# test_run_momentum_ladder.py::test_draft_that_merely_mentions_the_frozen_line_is_still_refused PASSED [ 66%]
# test_run_momentum_ladder.py::test_hash_comparison_ignores_case_because_powershell_prints_uppercase PASSED [ 68%]
# test_run_momentum_ladder.py::test_refuses_when_hash_line_missing PASSED [ 70%]
# test_run_momentum_ladder.py::test_refuses_when_price_file_differs_from_the_frozen_hash PASSED [ 72%]
# test_run_momentum_ladder.py::test_accepts_frozen_with_matching_hash PASSED [ 75%]
# test_run_momentum_ladder.py::test_max_drawdown_hand_checked PASSED [ 77%]
# test_run_momentum_ladder.py::test_annual_stats_constant_growth PASSED [ 79%]
# test_run_momentum_ladder.py::test_beat_spy_share_and_average PASSED [ 81%]
# test_run_momentum_ladder.py::test_evaluate_reports_both_floors_and_bootstrap_for_a_flat_world PASSED [ 83%]
# test_run_momentum_ladder.py::test_evaluate_hard_floor_vs_soft_floor PASSED [ 85%]
# test_run_momentum_ladder.py::test_decide_risk_eligibility_and_beats PASSED [ 87%]
# test_run_momentum_ladder.py::test_decide_recommends_largest_qualifying_rung_per_sleeve PASSED [ 89%]
# test_run_momentum_ladder.py::test_run_all_builds_every_variant_range_and_cost PASSED [ 91%]
# test_run_momentum_ladder.py::test_since_range_starts_at_or_after_the_date PASSED [ 93%]
# test_run_momentum_ladder.py::test_write_results_then_refuses_second_run PASSED [ 95%]
# test_run_momentum_ladder.py::test_main_refuses_a_draft_before_touching_anything PASSED [ 97%]
# test_run_momentum_ladder.py::test_run_constants_match_preregistration PASSED [100%]
# === 48 passed in 2.02s ===
