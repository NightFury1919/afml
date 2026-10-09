"""Turn model scores into the target weights that rebalance_orders() consumes.

Long-only, cash is whatever is not invested. The steps, in order:

    1. Select   keep the top `top_share` of the scored ETFs (rounded UP, at least one).
                Ties are broken by symbol so the result never depends on row order.
    2. Weight   selected ETFs are weighted in proportion to `base_weights` (default: equal).
                Pass HRP or inverse-volatility weights here to get the "HRP x signal" design.
    3. Scale    multiply the whole book by `exposure` in [0, 1] (the confidence scalar).
                0 means all cash.
    4. Cap      no ETF may exceed `max_weight` of EQUITY (applied after step 3). The excess is
                shared among the uncapped names in proportion to their weights; if everything is
                capped, what is left stays in cash.

Within-class mode (groups + class_weights): steps 1 and 2 run inside each class, and each class
receives its budget from `class_weights`. A class with fewer than `min_group_assets` scored ETFs
is skipped, and its budget is shared among the other classes in proportion to their budgets (the
same rule the features and labels use). The cap in step 4 can move weight between classes, so it
can override the budgets.

What this is not: it does not compute HRP, the exposure scalar, or class budgets. Those are
inputs. It contains no tuned numbers; top_share=0.2 matches the top-quintile statistic used in
the positive control, and is a default, not a finding.
"""
import math

import pandas as pd

DEFAULT_TOP_SHARE = 0.2
DEFAULT_MIN_GROUP_ASSETS = 3


def _as_series(scores):
    return scores if isinstance(scores, pd.Series) else pd.Series(scores, dtype=float)


def select_top(scores, top_share=DEFAULT_TOP_SHARE):
    """Symbols of the highest scores, best first. NaN scores are ignored."""
    if not 0 < top_share <= 1:
        raise ValueError("top_share must be in (0, 1]")
    s = _as_series(scores).dropna()
    if s.empty:
        return []
    k = max(1, math.ceil(top_share * len(s) - 1e-9))     # the guard stops 0.2 * 15 becoming 4
    return sorted(s.index, key=lambda sym: (-s[sym], sym))[:k]


def cap_weights(weights, max_weight):
    """Cap each weight and share the excess among the uncapped names, in proportion."""
    if not max_weight > 0:
        raise ValueError("max_weight must be positive")
    total = sum(weights.values())
    capped = set()
    out = dict(weights)
    while True:
        free = [s for s in weights if s not in capped]
        remaining = total - max_weight * len(capped)
        base = sum(weights[s] for s in free)
        if not free or base <= 0:
            return out
        scale = remaining / base
        for s in free:
            out[s] = weights[s] * scale
        newly = [s for s in free if out[s] > max_weight + 1e-12]
        if not newly:
            return out
        for s in newly:
            capped.add(s)
            out[s] = max_weight


def _equity_weights(selected, base_weights):
    if base_weights is None:
        raw = {s: 1.0 for s in selected}
    else:
        raw = {s: base_weights.get(s, 0.0) for s in selected}
        if any(not v > 0 for v in raw.values()):
            raise ValueError("every selected ETF needs a positive base weight")
    total = sum(raw.values())
    return {s: v / total for s, v in raw.items()}


def target_weights(scores, base_weights=None, groups=None, class_weights=None,
                   top_share=DEFAULT_TOP_SHARE, max_weight=None, exposure=1.0,
                   min_group_assets=DEFAULT_MIN_GROUP_ASSETS):
    """{symbol: weight of equity}, long-only, summing to at most 1. See the module docstring."""
    if not 0 <= exposure <= 1:
        raise ValueError("exposure must be in [0, 1]")
    s = _as_series(scores).dropna()

    if groups is None:
        if class_weights is not None:
            raise ValueError("class_weights needs groups")
        weights = _equity_weights(select_top(s, top_share), base_weights) if len(s) else {}
    else:
        if class_weights is None:
            raise ValueError("groups needs class_weights")
        if any(v < 0 for v in class_weights.values()):
            raise ValueError("class_weights must be non-negative")
        missing = [x for x in s.index if x not in groups]
        if missing:
            raise ValueError(f"no class for: {missing}")
        active = {}
        for g in sorted({groups[x] for x in s.index}):
            members = s[[x for x in s.index if groups[x] == g]]
            if len(members) >= min_group_assets and class_weights.get(g, 0.0) > 0:
                active[g] = members
        budget_total = sum(class_weights[g] for g in active)
        weights = {}
        for g, members in active.items():
            inside = _equity_weights(select_top(members, top_share), base_weights)
            for sym, w in inside.items():
                weights[sym] = w * class_weights[g] / budget_total

    weights = {k: v * exposure for k, v in weights.items() if v * exposure > 0}
    if max_weight is not None and weights:
        weights = cap_weights(weights, max_weight)
    return weights

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_portfolio_construction.py -v
#
# collected 22 items
#
# test_portfolio_construction.py::test_select_top_takes_the_highest_scores PASSED [  4%]
# test_portfolio_construction.py::test_select_top_rounds_the_count_up_and_keeps_at_least_one PASSED [  9%]
# test_portfolio_construction.py::test_select_top_count_is_not_thrown_off_by_floating_point PASSED [ 13%]
# test_portfolio_construction.py::test_select_top_ignores_missing_scores_and_breaks_ties_by_symbol PASSED [ 18%]
# test_portfolio_construction.py::test_select_top_rejects_a_bad_share PASSED [ 22%]
# test_portfolio_construction.py::test_cap_leaves_weights_alone_when_nothing_exceeds_it PASSED [ 27%]
# test_portfolio_construction.py::test_cap_moves_the_excess_to_the_others_in_proportion PASSED [ 31%]
# test_portfolio_construction.py::test_cap_cascades_when_redistribution_pushes_another_weight_over PASSED [ 36%]
# test_portfolio_construction.py::test_cap_that_cannot_hold_everything_leaves_the_rest_as_cash PASSED [ 40%]
# test_portfolio_construction.py::test_cap_rejects_a_non_positive_cap PASSED [ 45%]
# test_portfolio_construction.py::test_pooled_top_fifth_is_equal_weighted_and_fully_invested PASSED [ 50%]
# test_portfolio_construction.py::test_base_weights_tilt_the_selected_names PASSED [ 54%]
# test_portfolio_construction.py::test_a_selected_name_without_a_positive_base_weight_is_rejected PASSED [ 59%]
# test_portfolio_construction.py::test_exposure_scales_the_whole_book_and_zero_means_cash PASSED [ 63%]
# test_portfolio_construction.py::test_cap_applies_to_the_final_weights_after_exposure PASSED [ 68%]
# test_portfolio_construction.py::test_no_scores_means_no_positions PASSED [ 72%]
# test_portfolio_construction.py::test_output_is_long_only_and_never_over_one PASSED [ 77%]
# test_portfolio_construction.py::test_each_class_gets_its_budget_and_picks_its_own_top PASSED [ 81%]
# test_portfolio_construction.py::test_a_class_is_not_judged_against_the_other_class PASSED [ 86%]
# test_portfolio_construction.py::test_a_class_with_too_few_scored_etfs_is_skipped_and_its_budget_redistributed PASSED [ 90%]
# test_portfolio_construction.py::test_grouped_inputs_are_validated PASSED [ 95%]
# test_portfolio_construction.py::test_target_weights_feed_rebalance_orders_directly PASSED [100%]
#
# 22 passed in 0.31s
# Mutation checks (sandbox): bottom-instead-of-top selection, no floating-point guard on the count,
#   no cap cascade, cap applied before exposure, and no budget redistribution each made tests fail;
#   original restored. A first draft of the floating-point test used a case that is exact in floating
#   point (0.2 * 15) and so did not catch the mutation; it now uses 0.28 * 25.
# Re-run on your Windows mlfinlab env and replace this block with that output if you prefer.
# ---------------------------------------------------------------------------
