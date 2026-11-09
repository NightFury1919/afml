"""Two variants of the live momentum rule (C50): a trend filter and a group cap.

Pre-registration: preregistrations/preregistration_momentum_variants.md (DRAFT until committed as FROZEN).
This file holds only the tested building blocks. The runner that uses them on the real price file is
run_momentum_variants.py; it refuses to run until the pre-registration is frozen.

Plain English first. The base rule is the ladder's C50: every 21 trading days rank the ETFs by momentum (return from
273 to 21 trading days back), hold the top 3 equal weight in a sleeve that is 50% of the account, the rest in SPY.
Signals use data up to the close of day t; trades happen at the close of day t + 1.

  * Variant "trend" (trial 10): after ranking, a pick is bought only if its adjusted close on the signal day is ABOVE
    its own 200-day average (including that day). A pick that fails is not bought: its slice of the sleeve goes to SHY
    until the next rebalance (held as cash, 0%, on dates when SHY has no price). The check happens at rebalances only.
  * Variant "cap" (trial 11): walk down the ranking and take a fund only if no fund from its group has been taken yet,
    until 3 funds are chosen. The group of each ETF comes from a fixed table (region / asset class).

With every pick passing the trend test, or every fund in its own group, the result equals the ladder's simulate() exactly.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import momentum_ladder as ml

MA_LEN = 200
PARK = "SHY"
SLEEVE_N = 3
SLEEVE_SHARE = 0.50


def trend_ok_table(prices, ma_len=MA_LEN):
    """True where a fund's close is above its own average of the last `ma_len` closes (today included).
    A missing price anywhere in the window makes the average NaN, which counts as NOT ok."""
    avg = prices.rolling(ma_len, min_periods=ma_len).mean()
    return (prices > avg).fillna(False)


def select_capped(score_row, group_map, n):
    """Walk down the ranking (highest score first, ties by name); take a fund only if its group is unused."""
    s = score_row.dropna()
    order = sorted(s.index, key=lambda k: (-s[k], k))
    picks, used = [], set()
    for k in order:
        g = group_map[k]
        if g in used:
            continue
        picks.append(k)
        used.add(g)
        if len(picks) == n:
            break
    return picks


@dataclass
class VariantResult(ml.SimResult):
    filtered: dict = field(default_factory=dict)       # trade date -> picks sent to SHY/cash by the trend filter
    skipped: dict = field(default_factory=dict)        # trade date -> funds the cap passed over (higher ranked than a pick)


def simulate_variant(prices, spy, mode, group_map=None, n=SLEEVE_N, sleeve_share=SLEEVE_SHARE,
                     cost_bps=ml.COST_BPS, every=ml.REBALANCE_EVERY, lag=ml.LAG, lookback=ml.LOOKBACK,
                     skip=ml.SKIP, min_eligible=ml.MIN_ELIGIBLE, ma_len=MA_LEN, park=PARK):
    """Same as ml.simulate, with mode "trend" or "cap" changing which funds the sleeve holds. mode None = ladder."""
    if mode not in (None, "trend", "cap"):
        raise ValueError("mode must be None, 'trend' or 'cap'")
    if not 0.0 <= sleeve_share <= 1.0:
        raise ValueError("sleeve_share must be between 0 and 1")
    if mode == "cap":
        if group_map is None:
            raise ValueError("mode 'cap' needs a group_map")
        missing = [c for c in prices.columns if c not in group_map]
        if missing:
            raise ValueError("group_map is missing: " + ", ".join(missing))
    spy = spy.reindex(prices.index)
    score = ml.momentum_score(prices, lookback, skip)
    first = ml.first_signal_position(prices, lookback, min_eligible)
    if first is None:
        raise ValueError("never enough eligible ETFs")
    ok = trend_ok_table(prices, ma_len) if mode == "trend" else None

    names = list(prices.columns) + ["SPY"]
    col = {k: i for i, k in enumerate(names)}
    all_px = np.column_stack([prices.values.astype(float), spy.values.astype(float)])
    ratio = all_px[1:] / all_px[:-1]
    rel = np.where(np.isfinite(ratio), ratio, 1.0)

    n_rows = len(prices)
    signal_rows = list(range(first, n_rows, every))
    trade_rows = [s + lag for s in signal_rows if s + lag < n_rows]
    if not trade_rows:
        raise ValueError("no trade date inside the data")
    signal_of = {s + lag: s for s in signal_rows if s + lag < n_rows}

    hold = np.zeros(len(names))
    cash = 1.0
    values, dates = [], []
    turnover, trade_dates, holdings, filtered, skipped = {}, [], {}, {}, {}
    for d in range(trade_rows[0], n_rows):
        if d > trade_rows[0]:
            hold = hold * rel[d - 1]
        if d in signal_of:
            s_row = signal_of[d]
            total = hold.sum() + cash
            tradable = prices.iloc[d].notna()
            row = score.iloc[s_row]
            row = row[tradable.reindex(row.index).fillna(False)]
            if sleeve_share > 0:
                picks = select_capped(row, group_map, n) if mode == "cap" else ml.select_top_n(row, n)
            else:
                picks = []
            date = prices.index[d]
            target = np.zeros(len(names))
            if picks:
                dropped = []
                for k in picks:
                    if mode == "trend" and not bool(ok.iloc[s_row][k]):
                        dropped.append(k)
                        if park in col and tradable.get(park, False):
                            target[col[park]] += sleeve_share / len(picks)     # else: stays cash
                    else:
                        target[col[k]] += sleeve_share / len(picks)
                target[col["SPY"]] += 1.0 - sleeve_share
                if dropped:
                    filtered[date] = dropped
                if mode == "cap":
                    plain = ml.select_top_n(row, n)
                    cut = [k for k in plain if k not in picks]
                    if cut:
                        skipped[date] = cut
            else:
                target[col["SPY"]] = 1.0
            current = hold / total
            traded = float(np.abs(target - current).sum())
            cost = cost_bps / 1e4 * traded * total
            after = total - cost
            hold = target * after
            cash = after * max(0.0, 1.0 - float(target.sum()))
            trade_dates.append(date)
            turnover[date] = traded
            holdings[date] = {names[i]: float(target[i]) for i in range(len(names)) if target[i] > 0}
        values.append(hold.sum() + cash)
        dates.append(prices.index[d])
    return VariantResult(value=pd.Series(values, index=dates), turnover=pd.Series(turnover),
                         trade_dates=trade_dates, holdings=holdings, filtered=filtered, skipped=skipped)

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

# ---- RESULT of the frozen run, 2026-11-08 (pre-registration commit 0cc6b36) ----
# Verdict: neither variant passes. Both risk-eligible, neither beats SPY, neither adds value over C50. Trials 10 (trend) and 11 (cap), 11 on this data.
# Full history, 5 bps: trend 10.46%/yr, cap 9.77%, C50 10.06%, SPY 9.42%. Worst fall: trend -41.4%, cap -50.0%, C50 -51.5%, SPY -55.2%.
# trend minus SPY +0.66 pts (90% -2.30 to +3.86); trend minus C50 +0.13 (-1.13 to +1.61). cap minus SPY +0.20 (-2.49 to +3.06); cap minus C50 -0.33 (-1.73 to +1.19).
# Since 2016: trend 13.66%, cap 12.96%, C50 13.97%, SPY 14.87%. Trend parked a pick in 73 of 304 rebalances; cap changed picks in 241 of 304.
# Not retuned. Full report: portfolio/preregistrations/results_momentum_variants/report.txt
