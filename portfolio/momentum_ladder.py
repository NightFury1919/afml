"""Momentum risk ladder on ETFs: the building blocks.

Pre-registration: preregistration_momentum_ladder.md (DRAFT until committed).
This file holds only the tested building blocks. The runner that uses them on the real price
file is run_momentum_ladder.py; it refuses to run until the pre-registration is frozen.

Plain English first:
  * Momentum score: how much an ETF gained from 273 trading days ago to 21 trading days ago
    ("12 months, skipping the latest month"). The skip keeps very recent noise out.
  * A sleeve holds the top-n ETFs by that score, equal weight. The account is a mix of the
    sleeve and SPY. Every 21 trading days the signal is recomputed and the mix is reset.
  * Signals use data up to the close of day t. Trades happen at the close of day t + 1.
  * Windows: every 35-trading-day stretch, to see the range of 7-week outcomes.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---- frozen by the pre-registration ----
LOOKBACK = 273
SKIP = 21
REBALANCE_EVERY = 21
WINDOW = 35
MIN_ELIGIBLE = 20
LAG = 1
COST_BPS = 5.0
FLOOR = -0.10
SLEEVE_N = {"D": 10, "C": 3}
RUNGS = (0.10, 0.25, 0.50, 1.00)


# ---------- momentum score and selection ----------
def momentum_score(prices, lookback=LOOKBACK, skip=SKIP):
    """P[t - skip] / P[t - lookback] - 1. Needs a price today and `lookback` rows ago, else NaN."""
    score = prices.shift(skip) / prices.shift(lookback) - 1.0
    eligible = prices.notna() & prices.shift(lookback).notna()
    return score.where(eligible)


def select_top_n(score_row, n):
    """Names of the n highest scores. NaN skipped; ties broken by name; fewer than n if fewer exist."""
    s = score_row.dropna()
    order = sorted(s.index, key=lambda k: (-s[k], k))
    return order[:n]


def first_signal_position(prices, lookback=LOOKBACK, min_eligible=MIN_ELIGIBLE):
    """First row where at least `min_eligible` ETFs are eligible, or None if that never happens."""
    eligible = prices.notna() & prices.shift(lookback).notna()
    count = eligible.sum(axis=1).values
    hits = np.flatnonzero(count >= min_eligible)
    return int(hits[0]) if len(hits) else None


# ---------- simulation ----------
@dataclass
class SimResult:
    value: pd.Series                                   # account value, starts at 1.0 on the first trade date
    turnover: pd.Series                                # sum of absolute weight changes, per trade date
    trade_dates: list = field(default_factory=list)
    holdings: dict = field(default_factory=dict)       # trade date -> {name: target weight}


def simulate(prices, spy, n, sleeve_share, cost_bps=COST_BPS, every=REBALANCE_EVERY, lag=LAG,
             lookback=LOOKBACK, skip=SKIP, min_eligible=MIN_ELIGIBLE):
    """Mix of a top-n momentum sleeve (sleeve_share) and SPY (the rest), reset every `every` rows.

    Signal at row t uses data up to t. The trade happens at the close of row t + lag.
    Cost = cost_bps of the traded value (sum of |target weight - drifted weight|) at each trade.
    If the sleeve has no eligible ETF at a trade, its share is held in SPY.
    """
    if not 0.0 <= sleeve_share <= 1.0:
        raise ValueError("sleeve_share must be between 0 and 1")
    spy = spy.reindex(prices.index)
    score = momentum_score(prices, lookback, skip)
    first = first_signal_position(prices, lookback, min_eligible)
    if first is None:
        raise ValueError("never enough eligible ETFs")

    names = list(prices.columns) + ["SPY"]
    col = {k: i for i, k in enumerate(names)}
    all_px = np.column_stack([prices.values.astype(float), spy.values.astype(float)])
    ratio = all_px[1:] / all_px[:-1]
    rel = np.where(np.isfinite(ratio), ratio, 1.0)     # a missing price means no change that day

    n_rows = len(prices)
    signal_rows = list(range(first, n_rows, every))
    trade_rows = [s + lag for s in signal_rows if s + lag < n_rows]
    if not trade_rows:
        raise ValueError("no trade date inside the data")
    signal_of = {s + lag: s for s in signal_rows if s + lag < n_rows}

    hold = np.zeros(len(names))
    cash = 1.0
    values, dates = [], []
    turnover, trade_dates, holdings = {}, [], {}
    for d in range(trade_rows[0], n_rows):
        if d > trade_rows[0]:
            hold = hold * rel[d - 1]
        if d in signal_of:
            total = hold.sum() + cash
            tradable = prices.iloc[d].notna()
            row = score.iloc[signal_of[d]]
            row = row[tradable.reindex(row.index).fillna(False)]
            picks = select_top_n(row, n) if sleeve_share > 0 else []
            target = np.zeros(len(names))
            if picks:
                for k in picks:
                    target[col[k]] += sleeve_share / len(picks)
                target[col["SPY"]] += 1.0 - sleeve_share
            else:
                target[col["SPY"]] = 1.0
            current = hold / total
            traded = float(np.abs(target - current).sum())
            cost = cost_bps / 1e4 * traded * total
            after = total - cost
            hold = target * after
            cash = 0.0
            date = prices.index[d]
            trade_dates.append(date)
            turnover[date] = traded
            holdings[date] = {names[i]: float(target[i]) for i in range(len(names)) if target[i] > 0}
        values.append(hold.sum() + cash)
        dates.append(prices.index[d])
    return SimResult(value=pd.Series(values, index=dates), turnover=pd.Series(turnover),
                     trade_dates=trade_dates, holdings=holdings)


# ---------- window statistics ----------
def window_outcomes(value, window=WINDOW, floor=FLOOR):
    """For every window start s: end return V[s+window]/V[s]-1, and whether the account was at or
    below `floor` (relative to V[s]) at ANY close from s to s+window."""
    v = np.asarray(value, dtype=float)
    if len(v) <= window:
        raise ValueError("series shorter than one window")
    starts = len(v) - window
    end = v[window:] / v[:starts] - 1.0
    low = np.lib.stride_tricks.sliding_window_view(v, window + 1).min(axis=1)
    touch = (low / v[:starts] - 1.0) <= floor
    return end, touch


def window_summary(value, window=WINDOW, floor=FLOOR):
    """Same table as spy_window_stats.py: percentiles by index int(q * (n - 1)) on sorted outcomes."""
    end, touch = window_outcomes(value, window, floor)
    w = np.sort(end)

    def pct(q):
        return float(w[int(q * (len(w) - 1))])

    out = {"n_windows": int(len(w)), "worst": float(w[0]), "p05": pct(0.05), "median": pct(0.5),
           "p95": pct(0.95), "best": float(w[-1])}
    for t, label in ((0.0, "0"), (0.10, "10"), (0.20, "20"), (0.50, "50"), (1.00, "100")):
        out[f"share_gt_{label}"] = float((end > t).mean())
    out["end_below_floor"] = float((end < floor).mean())
    out["touch_floor"] = float(touch.mean())
    return out


# ---------- non-overlapping periods and the paired bootstrap ----------
def period_returns(value, every=REBALANCE_EVERY):
    """Returns over non-overlapping blocks of `every` rows (rows 0, every, 2*every, ...)."""
    v = np.asarray(value, dtype=float)[::every]
    return pd.Series(v[1:] / v[:-1] - 1.0)


def block_bootstrap_interval(diff, block=4, n_resamples=5000, seed=0, level=0.90):
    """Moving-block bootstrap interval for the annualized mean of a paired period difference
    (mean x 12, since a period is 21 trading days). Returns (low, high)."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    if n < block:
        raise ValueError("series shorter than one block")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_resamples, -1)[:, :n]
    means = d[idx].mean(axis=1) * 12.0
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


# =============================================================================
# TDD RESULTS (2026-10-28)  python 3.11, numpy 1.23.5, pandas 1.5.3, pytest 9.2.0
# Command: python -m pytest test_momentum_ladder.py -v
# Mutation check: 18 of 18 deliberate breakages caught by the tests (including the pre-registered
# defaults, the floor boundary, drift vs daily reset, lag, costs, tie-breaking, eligibility, percentiles).
# Real-data cross-check (SPY only, a benchmark, not a candidate): window_summary on SPY since 2016-01-04,
# prices_daily_asof_2026-10-19.csv, reproduces spy_window_stats.py (2679 windows, median +2.8%, p5 -7.1%,
# p95 +9.7%, worst -30.3%, best +28.6%); new: SPY at or below -10% inside a window 6.6% of the time
# (ending below -10%: 2.9%).
# The runner is run_momentum_ladder.py (built 2026-10-31, own tests in test_run_momentum_ladder.py).
# =============================================================================
# ============================= test session starts ==============================
# platform linux -- Python 3.11.18, pytest-9.2.0, pluggy-1.7.0 -- /tmp/claude-0/-home-claude/c61e95b3-92b3-5c27-b674-5e7b010e18ac/scratchpad/momentum/../env311/bin/python
# cachedir: .pytest_cache
# rootdir: /tmp/claude-0/-home-claude/c61e95b3-92b3-5c27-b674-5e7b010e18ac/scratchpad/momentum
# collecting ... collected 27 items
# test_momentum_ladder.py::test_defaults_match_preregistration PASSED      [  3%]
# test_momentum_ladder.py::test_momentum_score_value_and_warmup PASSED     [  7%]
# test_momentum_ladder.py::test_momentum_score_needs_price_today_and_lookback_ago PASSED [ 11%]
# test_momentum_ladder.py::test_momentum_score_uses_no_future_prices PASSED [ 14%]
# test_momentum_ladder.py::test_select_top_n_orders_by_score PASSED        [ 18%]
# test_momentum_ladder.py::test_select_top_n_ties_broken_by_name PASSED    [ 22%]
# test_momentum_ladder.py::test_select_top_n_skips_nan_and_handles_short_universe PASSED [ 25%]
# test_momentum_ladder.py::test_first_signal_position_needs_min_eligible PASSED [ 29%]
# test_momentum_ladder.py::test_first_signal_position_none_when_never_enough PASSED [ 33%]
# test_momentum_ladder.py::test_sleeve_only_earns_best_etf_growth_after_trade_date PASSED [ 37%]
# test_momentum_ladder.py::test_zero_sleeve_equals_spy PASSED              [ 40%]
# test_momentum_ladder.py::test_half_and_half_resets_at_each_rebalance PASSED [ 44%]
# test_momentum_ladder.py::test_holdings_use_only_data_up_to_signal_date PASSED [ 48%]
# test_momentum_ladder.py::test_equal_weight_across_n_holdings PASSED      [ 51%]
# test_momentum_ladder.py::test_first_purchase_cost_charged_on_traded_value PASSED [ 55%]
# test_momentum_ladder.py::test_no_cost_when_rebalance_changes_nothing PASSED [ 59%]
# test_momentum_ladder.py::test_switching_holdings_charges_the_turnover PASSED [ 62%]
# test_momentum_ladder.py::test_window_outcomes_hand_checked PASSED        [ 66%]
# test_momentum_ladder.py::test_window_summary_hand_checked PASSED         [ 70%]
# test_momentum_ladder.py::test_touch_counts_exactly_at_the_floor PASSED   [ 74%]
# test_momentum_ladder.py::test_simulate_defaults_match_preregistration PASSED [ 77%]
# test_momentum_ladder.py::test_window_summary_matches_spy_script_method PASSED [ 81%]
# test_momentum_ladder.py::test_window_summary_too_short_series PASSED     [ 85%]
# test_momentum_ladder.py::test_period_returns_every_21_rows PASSED        [ 88%]
# test_momentum_ladder.py::test_block_bootstrap_constant_difference_gives_that_constant PASSED [ 92%]
# test_momentum_ladder.py::test_block_bootstrap_is_deterministic_for_a_seed PASSED [ 96%]
# test_momentum_ladder.py::test_block_bootstrap_interval_covers_the_true_mean_for_a_clear_edge PASSED [100%]
# ============================== 27 passed in 0.42s ==============================

# =============================================================================
# RESULTS OF THE PRE-REGISTERED RUN (2026-10-31, Ethan's Windows PC, mlfinlab env)
# Command: python run_momentum_ladder.py --prices prices_daily_asof_2026-10-30.csv   (run once; results in momentum_results/)
# Pre-registration: preregistration_momentum_ladder.md. Its frozen text was committed in 9e9c2a6.
# PROCESS NOTE (disclosed): the run executed BEFORE commit 9e9c2a6. The working-copy pre-registration already said
# 'Status: FROZEN' with the unchanged price-file hash, but the earlier commit d1b3581 held the DRAFT text by mistake, and the
# frozen text was committed only after the run. No rule, variant or file was changed after the results were seen
# (the 9e9c2a6 diff is the freeze edit itself). Treat the freeze as a few minutes late, not as a changed test.
# Column 'share_gt_10' in the table is truncated: it is the share of windows ending above +100% (2x).
# =============================================================================
# SUMMARY (plain English; the pre-registered rules were applied, nothing else was tested)
#  * Risk screen (touch rate <= 2x SPY's 8.6% = 17.2%, worst window >= -50%): all 8 variants pass.
#  * Beats SPY (lower end of the 90% interval > 0): NONE. Every interval spans 0 (D50 -1.5% to +2.3% a year; C100 -4.7% to +7.1%).
#  * Median 7-week window above SPY's: D10, D25, D50, C10, C25, C50 (by 0.1 to 0.2 points); not D100 or C100.
#  * Mechanical pick by rule 3: D50 and C50. By the outcome table, a rung is offered only if it ALSO beats SPY. None does,
#    so the registered answer is: more risk did not buy a demonstrable edge; SPY stays the better holding.
#  * No variant ended any 7-week window at 2x (share ending above +100% is 0.000 for all). Best single window: C100 +59.8%.
#  * Since 2016-01-04 every variant is below SPY on annual return (SPY 14.9%; D10 14.5%, D100 10.9%, C10 14.8%, C100 12.1%)
#    and ends ahead of SPY in only 42% to 44% of windows.
#  * Descriptive only (not a tested claim): D100's worst fall was -38.2% against SPY's -55.2% over the full range.
# =============================================================================
# Momentum risk ladder: results (price file prices_daily_asof_2026-10-30.csv, SHA-256 dff1a011fee8a8f662d94d166f58c6ddd685a9d5ac819407e3970f1d2160b30c)
# First trade date 2001-06-28; 'since' range starts 2016-01-04
#
# == range=full cost ==
# name         worst         p05      median         p95        best share_gt_10 end_below_f touch_floor  touch_hard  ann_return     ann_vol max_drawdow  beat_share   diff_ci(ann, 90%)
# SPY         -0.335      -0.088       0.021       0.091       0.286       0.000       0.037       0.086       0.000       0.094       0.189      -0.552       0.000   [+0.000, +0.000]
# D10         -0.320      -0.085       0.022       0.090       0.285       0.000       0.036       0.083       0.000       0.096       0.184      -0.528       0.514   [-0.003, +0.005]
# D25         -0.300      -0.082       0.022       0.089       0.285       0.000       0.033       0.083       0.000       0.098       0.177      -0.492       0.513   [-0.007, +0.012]
# D50         -0.298      -0.082       0.022       0.090       0.283       0.000       0.031       0.083       0.000       0.101       0.171      -0.434       0.512   [-0.015, +0.023]
# D100        -0.293      -0.090       0.020       0.103       0.284       0.000       0.041       0.095       0.000       0.104       0.180      -0.382       0.510   [-0.029, +0.047]
# C10         -0.329      -0.085       0.022       0.090       0.281       0.000       0.035       0.084       0.000       0.096       0.185      -0.542       0.507   [-0.005, +0.007]
# C25         -0.320      -0.085       0.023       0.090       0.275       0.000       0.034       0.088       0.000       0.099       0.182      -0.529       0.506   [-0.012, +0.018]
# C50         -0.305      -0.090       0.022       0.098       0.289       0.000       0.038       0.103       0.000       0.101       0.189      -0.515       0.503   [-0.024, +0.036]
# C100        -0.280      -0.111       0.020       0.134       0.597       0.000       0.065       0.155       0.000       0.097       0.238      -0.551       0.499   [-0.047, +0.071]
#
# == range=full nocost ==
# name         worst         p05      median         p95        best share_gt_10 end_below_f touch_floor  touch_hard  ann_return     ann_vol max_drawdow  beat_share   diff_ci(ann, 90%)
# SPY         -0.335      -0.088       0.021       0.091       0.286       0.000       0.037       0.086       0.000       0.094       0.189      -0.552       0.000   [+0.000, +0.000]
# D10         -0.320      -0.085       0.022       0.090       0.285       0.000       0.036       0.083       0.000       0.096       0.184      -0.528       0.519   [-0.003, +0.005]
# D25         -0.300      -0.082       0.023       0.089       0.285       0.000       0.033       0.082       0.000       0.099       0.177      -0.491       0.518   [-0.007, +0.012]
# D50         -0.297      -0.082       0.023       0.090       0.284       0.000       0.031       0.083       0.000       0.103       0.171      -0.433       0.517   [-0.013, +0.025]
# D100        -0.293      -0.090       0.020       0.103       0.285       0.000       0.040       0.095       0.000       0.107       0.180      -0.380       0.514   [-0.026, +0.049]
# C10         -0.329      -0.085       0.022       0.090       0.282       0.000       0.035       0.084       0.000       0.097       0.185      -0.542       0.513   [-0.004, +0.008]
# C25         -0.320      -0.085       0.023       0.091       0.275       0.000       0.034       0.088       0.000       0.100       0.182      -0.529       0.512   [-0.011, +0.019]
# C50         -0.305      -0.089       0.023       0.098       0.289       0.000       0.038       0.102       0.000       0.103       0.189      -0.514       0.510   [-0.022, +0.038]
# C100        -0.279      -0.111       0.021       0.135       0.598       0.000       0.065       0.155       0.000       0.102       0.238      -0.548       0.506   [-0.043, +0.075]
#
# == range=since cost ==
# name         worst         p05      median         p95        best share_gt_10 end_below_f touch_floor  touch_hard  ann_return     ann_vol max_drawdow  beat_share   diff_ci(ann, 90%)
# SPY         -0.303      -0.071       0.028       0.097       0.286       0.000       0.029       0.065       0.000       0.149       0.177      -0.337       0.000   [+0.000, +0.000]
# D10         -0.302      -0.069       0.028       0.095       0.285       0.000       0.028       0.063       0.000       0.145       0.174      -0.336       0.419   [-0.008, +0.002]
# D25         -0.300      -0.068       0.027       0.093       0.285       0.000       0.023       0.060       0.000       0.140       0.170      -0.335       0.418   [-0.019, +0.006]
# D50         -0.298      -0.064       0.024       0.092       0.283       0.000       0.018       0.059       0.000       0.130       0.168      -0.333       0.417   [-0.038, +0.012]
# D100        -0.293      -0.064       0.019       0.104       0.281       0.000       0.019       0.051       0.000       0.109       0.179      -0.329       0.416   [-0.075, +0.024]
# C10         -0.300      -0.067       0.028       0.094       0.281       0.000       0.025       0.061       0.000       0.148       0.175      -0.334       0.444   [-0.010, +0.010]
# C25         -0.295      -0.065       0.025       0.097       0.275       0.000       0.020       0.058       0.000       0.145       0.177      -0.328       0.443   [-0.024, +0.025]
# C50         -0.287      -0.067       0.024       0.109       0.289       0.000       0.019       0.069       0.000       0.140       0.192      -0.320       0.441   [-0.048, +0.050]
# C100        -0.275      -0.100       0.021       0.150       0.597       0.000       0.049       0.128       0.000       0.121       0.251      -0.306       0.436   [-0.096, +0.101]
#
# == range=since nocost ==
# name         worst         p05      median         p95        best share_gt_10 end_below_f touch_floor  touch_hard  ann_return     ann_vol max_drawdow  beat_share   diff_ci(ann, 90%)
# SPY         -0.303      -0.071       0.028       0.097       0.286       0.000       0.029       0.065       0.000       0.149       0.177      -0.337       0.000   [+0.000, +0.000]
# D10         -0.302      -0.069       0.028       0.095       0.285       0.000       0.028       0.063       0.000       0.146       0.174      -0.336       0.422   [-0.007, +0.003]
# D25         -0.300      -0.068       0.027       0.094       0.285       0.000       0.023       0.060       0.000       0.141       0.170      -0.335       0.422   [-0.018, +0.007]
# D50         -0.297      -0.064       0.024       0.092       0.284       0.000       0.018       0.058       0.000       0.132       0.168      -0.333       0.420   [-0.036, +0.013]
# D100        -0.293      -0.063       0.019       0.104       0.282       0.000       0.018       0.051       0.000       0.112       0.179      -0.329       0.419   [-0.072, +0.027]
# C10         -0.300      -0.067       0.028       0.094       0.282       0.000       0.025       0.061       0.000       0.148       0.175      -0.334       0.448   [-0.009, +0.011]
# C25         -0.295      -0.065       0.025       0.097       0.275       0.000       0.020       0.058       0.000       0.147       0.177      -0.328       0.447   [-0.023, +0.026]
# C50         -0.287      -0.067       0.024       0.110       0.289       0.000       0.019       0.068       0.000       0.142       0.192      -0.320       0.445   [-0.046, +0.053]
# C100        -0.274      -0.099       0.021       0.151       0.598       0.000       0.048       0.127       0.000       0.125       0.251      -0.302       0.440   [-0.092, +0.105]
#
# Rules applied to range=full cost. Touch-floor limit = 0.172
# D10    risk_eligible=True  beats_spy=False  median_above_spy=True
# D25    risk_eligible=True  beats_spy=False  median_above_spy=True
# D50    risk_eligible=True  beats_spy=False  median_above_spy=True
# D100   risk_eligible=True  beats_spy=False  median_above_spy=False
# C10    risk_eligible=True  beats_spy=False  median_above_spy=True
# C25    risk_eligible=True  beats_spy=False  median_above_spy=True
# C50    risk_eligible=True  beats_spy=False  median_above_spy=True
# C100   risk_eligible=True  beats_spy=False  median_above_spy=False
# Recommended rung per sleeve: {'D': 'D50', 'C': 'C50'}
