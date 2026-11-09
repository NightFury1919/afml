"""Momentum with a volatility stop on each sleeve fund: the building blocks.

Pre-registration: preregistrations/preregistration_momentum_stop.md (DRAFT until committed as FROZEN).
This file holds only the tested building blocks. The runner that uses them on the real price file is
run_momentum_stop.py; it refuses to run until the pre-registration is frozen.

Plain English first:
  * The base rule is the ladder's C50: every 21 trading days rank the ETFs by momentum (return from 273 to 21
    trading days back), hold the top 3 equal weight in a sleeve that is 50% of the account, the rest in SPY.
    Signals use data up to the close of day t; trades happen at the close of day t + 1.
  * The stop: when a fund is bought (at a rebalance), set a barrier below its buy price,
        barrier = buy price x (1 - k x daily_volatility x sqrt(21)),      k = 1.5,
    where daily_volatility comes from AFML Chapter 3's get_daily_vol (span 100), measured with data up to and
    including the buy day. The barrier is fixed: it does not move up when the price rises.
  * Each close after the buy, if a fund's close is at or below its barrier, the fund is sold at the NEXT close
    (at that close's actual price, so a gap below the barrier is a real loss) and the money is parked in SHY
    (or held as cash earning 0% on dates when SHY has no price) until the next scheduled rebalance.
  * A stopped fund is not bought back before the next scheduled rebalance. At the rebalance everything is reset:
    the signal is recomputed, new barriers are set for every fund held, and the sleeve/SPY split goes back to target.
  * If the next day is itself a rebalance day, the rebalance takes over and the pending stop sale is dropped.
  * With k=None there is no stop and the result equals the ladder's simulate() exactly (the baseline).
"""
from dataclasses import dataclass, field
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import momentum_ladder as ml

# ---- frozen by the pre-registration ----
K = 1.5
HORIZON = 21          # the stop distance scales daily volatility by sqrt(HORIZON)
SPAN0 = 100           # get_daily_vol's EWMA span (the function's default, AFML snippet 3.1)
PARK = "SHY"
SLEEVE_N = 3
SLEEVE_SHARE = 0.50


# ---------- the stop distance and the volatility input ----------
def stop_distance(sigma, k=K, horizon=HORIZON):
    """Fraction below the buy price at which the barrier sits: k x sigma x sqrt(horizon)."""
    return float(k) * float(sigma) * float(np.sqrt(horizon))


def daily_vol_frame(prices, get_daily_vol, span0=SPAN0):
    """Apply AFML's get_daily_vol to every column. Missing prices are dropped before the call (a fund has no
    returns before it exists) and the result is put back on the full date index."""
    out = {}
    for c in prices.columns:
        s = prices[c].dropna()
        if len(s) < 3:
            out[c] = pd.Series(np.nan, index=prices.index)
            continue
        v = get_daily_vol(s, span0=span0)
        out[c] = v.reindex(prices.index)
    return pd.DataFrame(out, index=prices.index, columns=prices.columns)


def load_get_daily_vol(repo_root, modules=("triple_barrier", "returns", "meta_labeling")):
    """Import the repo's Chapter 3 get_daily_vol (ch03/labeling/<module>.py)."""
    root = str(Path(repo_root))
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)                            # the repo given wins over any other ch03 on the path
    importlib.invalidate_caches()
    for m in modules:
        try:
            mod = importlib.import_module(f"ch03.labeling.{m}")
        except ImportError:
            continue
        if hasattr(mod, "get_daily_vol"):
            return mod.get_daily_vol
    raise ImportError("get_daily_vol not found in ch03/labeling (tried: " + ", ".join(modules) + ")")


# ---------- simulation ----------
@dataclass
class StopResult(ml.SimResult):
    stops: list = field(default_factory=list)          # one dict per executed stop sale
    cost_paid: float = 0.0                              # total trading cost paid (rebalances and stops), in account units
    stop_cost_paid: float = 0.0                         # the part of cost_paid caused by stop sales
    superseded: int = 0                                 # triggers dropped because the next day was a rebalance


def simulate_stop(prices, spy, vol, n=SLEEVE_N, sleeve_share=SLEEVE_SHARE, k=K, cost_bps=ml.COST_BPS,
                  every=ml.REBALANCE_EVERY, lag=ml.LAG, lookback=ml.LOOKBACK, skip=ml.SKIP,
                  min_eligible=ml.MIN_ELIGIBLE, park=PARK, horizon=HORIZON):
    """Same as momentum_ladder.simulate, plus a stop on each sleeve fund (see the module text).

    `vol` is a DataFrame like `prices` with each fund's daily volatility (use daily_vol_frame). With k=None
    the stop is off and `vol` is not used.
    Cost = cost_bps of traded value: at a stop sale the sold fund is one leg and the SHY purchase another
    (only the sale when the proceeds sit in cash).
    """
    if not 0.0 <= sleeve_share <= 1.0:
        raise ValueError("sleeve_share must be between 0 and 1")
    if k is not None and vol is None:
        raise ValueError("a stop needs a volatility table")
    spy = spy.reindex(prices.index)
    score = ml.momentum_score(prices, lookback, skip)
    first = ml.first_signal_position(prices, lookback, min_eligible)
    if first is None:
        raise ValueError("never enough eligible ETFs")

    names = list(prices.columns) + ["SPY"]
    col = {kk: i for i, kk in enumerate(names)}
    all_px = np.column_stack([prices.values.astype(float), spy.values.astype(float)])
    ratio = all_px[1:] / all_px[:-1]
    rel = np.where(np.isfinite(ratio), ratio, 1.0)     # a missing price means no change that day
    park_i = col.get(park)

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
    barriers = {}            # column index -> (entry_row, entry_price, barrier), still being watched
    info = {}                # the same details for every fund bought at the latest rebalance (kept for the sale record)
    pending = []             # column indices triggered at the previous close, to be sold at this close
    stops, cost_paid, stop_cost, superseded = [], 0.0, 0.0, 0
    for d in range(trade_rows[0], n_rows):
        if d > trade_rows[0]:
            hold = hold * rel[d - 1]
        if d in signal_of:
            superseded += len(pending)
            pending = []
            barriers = {}
            info = {}
            total = hold.sum() + cash
            tradable = prices.iloc[d].notna()
            row = score.iloc[signal_of[d]]
            row = row[tradable.reindex(row.index).fillna(False)]
            picks = ml.select_top_n(row, n) if sleeve_share > 0 else []
            target = np.zeros(len(names))
            if picks:
                for kk in picks:
                    target[col[kk]] += sleeve_share / len(picks)
                target[col["SPY"]] += 1.0 - sleeve_share
            else:
                target[col["SPY"]] = 1.0
            current = hold / total
            traded = float(np.abs(target - current).sum())
            cost = cost_bps / 1e4 * traded * total
            after = total - cost
            hold = target * after
            cash = 0.0
            cost_paid += cost
            date = prices.index[d]
            trade_dates.append(date)
            turnover[date] = traded
            holdings[date] = {names[i]: float(target[i]) for i in range(len(names)) if target[i] > 0}
            if k is not None:
                for kk in picks:
                    ci = col[kk]
                    sigma = vol.iloc[d][kk]
                    entry = all_px[d, ci]
                    if np.isfinite(sigma) and sigma > 0 and np.isfinite(entry):
                        barriers[ci] = (d, entry, entry * (1.0 - stop_distance(sigma, k, horizon)))
                        info[ci] = barriers[ci]
        elif pending:
            still = []
            for ci in pending:
                px = all_px[d, ci]
                if not np.isfinite(px):
                    still.append(ci)                    # no price today: try again at the next close
                    continue
                x = hold[ci]
                if x <= 0:
                    continue
                to_park = park_i is not None and np.isfinite(all_px[d, park_i])
                legs = 2.0 if to_park else 1.0
                cost = cost_bps / 1e4 * legs * x
                hold[ci] = 0.0
                if to_park:
                    hold[park_i] += x - cost
                else:
                    cash += x - cost
                cost_paid += cost
                stop_cost += cost
                entry_row, entry_px, barrier = info[ci]
                stops.append({"fund": names[ci], "entry_row": int(entry_row), "entry_price": float(entry_px),
                              "barrier": float(barrier), "trigger_row": int(d - 1), "sale_row": int(d),
                              "sale_price": float(px), "sale_date": str(prices.index[d].date()),
                              "value_sold": float(x), "to": park if to_park else "cash"})
            pending = still
        # look at today's close for new triggers (never on the buy day itself)
        if k is not None and barriers:
            for ci in list(barriers):
                entry_row, entry_px, barrier = barriers[ci]
                px = all_px[d, ci]
                if d > entry_row and np.isfinite(px) and px <= barrier:
                    if ci not in pending:
                        pending.append(ci)
                    del barriers[ci]
        values.append(hold.sum() + cash)
        dates.append(prices.index[d])
    return StopResult(value=pd.Series(values, index=dates), turnover=pd.Series(turnover),
                      trade_dates=trade_dates, holdings=holdings, stops=stops, cost_paid=float(cost_paid),
                      stop_cost_paid=float(stop_cost), superseded=int(superseded))


# ---------- descriptive stop statistics ----------
def stop_rebound_stats(prices, stops, horizon=21):
    """Among executed stops, how often the fund's price `horizon` rows after the sale was HIGHER than the sale
    price, and the average change. Stops too close to the end of the data to measure are left out."""
    changes = []
    for s in stops:
        r = s["sale_row"] + horizon
        if r < len(prices):
            later = float(prices.iloc[r][s["fund"]])
            if np.isfinite(later):
                changes.append(later / s["sale_price"] - 1.0)
    c = np.asarray(changes, dtype=float)
    return {"n_stops": int(len(stops)), "n_measured": int(len(c)),
            "share_higher": float((c > 0).mean()) if len(c) else float("nan"),
            "avg_change": float(c.mean()) if len(c) else float("nan")}


def periods_with_stop(stops, trade_dates, end_date):
    """(number of rebalance periods, number with at least one stop sale). A period runs from one rebalance
    date up to the next (the last one up to `end_date`)."""
    edges = list(pd.to_datetime(trade_dates)) + [pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    sale_dates = [pd.Timestamp(s["sale_date"]) for s in stops]
    hit = 0
    for a, b in zip(edges[:-1], edges[1:]):
        if any(a <= sd < b for sd in sale_dates):
            hit += 1
    return len(edges) - 1, hit

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
