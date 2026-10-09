import math

import pandas as pd
import pytest

from portfolio_construction import cap_weights, select_top, target_weights
from rebalance import rebalance_orders


def S(d):
    return pd.Series(d, dtype=float)


def ten(prefix="E"):
    return S({f"{prefix}{i}": float(i) for i in range(10)})        # E9 is the best score


# ------------------------------------------------------------ selection
def test_select_top_takes_the_highest_scores():
    assert select_top(ten(), 0.2) == ["E9", "E8"]


def test_select_top_rounds_the_count_up_and_keeps_at_least_one():
    assert len(select_top(S({f"X{i}": float(i) for i in range(12)}), 0.2)) == 3     # ceil(2.4)
    assert len(select_top(S({"A": 1.0, "B": 2.0}), 0.2)) == 1                       # ceil(0.4) -> 1


def test_select_top_count_is_not_thrown_off_by_floating_point():
    # 0.28 * 25 is 7.000000000000001 in floating point; the right answer is 7, not 8.
    assert 0.28 * 25 > 7
    assert len(select_top(S({f"X{i}": float(i) for i in range(25)}), 0.28)) == 7


def test_select_top_ignores_missing_scores_and_breaks_ties_by_symbol():
    s = S({"B": 1.0, "A": 1.0, "C": 1.0, "D": float("nan"), "E": 0.0})
    assert select_top(s, 0.5) == ["A", "B"]                                          # n=4 scored, k=2
    assert select_top(S({"A": float("nan")}), 0.2) == []


def test_select_top_rejects_a_bad_share():
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            select_top(ten(), bad)


# ------------------------------------------------------------------ caps
def test_cap_leaves_weights_alone_when_nothing_exceeds_it():
    w = {"A": 0.4, "B": 0.6}
    assert cap_weights(w, 0.6) == pytest.approx(w)


def test_cap_moves_the_excess_to_the_others_in_proportion():
    out = cap_weights({"A": 0.7, "B": 0.2, "C": 0.1}, 0.5)
    assert out == pytest.approx({"A": 0.5, "B": 1 / 3, "C": 1 / 6})
    assert sum(out.values()) == pytest.approx(1.0)


def test_cap_cascades_when_redistribution_pushes_another_weight_over():
    out = cap_weights({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}, 0.35)
    assert out == pytest.approx({"A": 0.35, "B": 0.35, "C": 0.15, "D": 0.15})


def test_cap_that_cannot_hold_everything_leaves_the_rest_as_cash():
    out = cap_weights({"A": 0.5, "B": 0.5}, 0.3)
    assert out == pytest.approx({"A": 0.3, "B": 0.3})                                 # sum 0.6, 0.4 in cash


def test_cap_rejects_a_non_positive_cap():
    with pytest.raises(ValueError):
        cap_weights({"A": 1.0}, 0.0)


# -------------------------------------------------------- pooled weights
def test_pooled_top_fifth_is_equal_weighted_and_fully_invested():
    w = target_weights(ten())
    assert w == pytest.approx({"E9": 0.5, "E8": 0.5})


def test_base_weights_tilt_the_selected_names():
    w = target_weights(ten(), base_weights={"E9": 1.0, "E8": 3.0})
    assert w == pytest.approx({"E9": 0.25, "E8": 0.75})


def test_a_selected_name_without_a_positive_base_weight_is_rejected():
    with pytest.raises(ValueError):
        target_weights(ten(), base_weights={"E9": 1.0})                               # E8 missing
    with pytest.raises(ValueError):
        target_weights(ten(), base_weights={"E9": 1.0, "E8": 0.0})


def test_exposure_scales_the_whole_book_and_zero_means_cash():
    assert target_weights(ten(), exposure=0.5) == pytest.approx({"E9": 0.25, "E8": 0.25})
    assert target_weights(ten(), exposure=0.0) == {}
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            target_weights(ten(), exposure=bad)


def test_cap_applies_to_the_final_weights_after_exposure():
    w = target_weights(ten(), exposure=0.5, max_weight=0.2)
    assert w == pytest.approx({"E9": 0.2, "E8": 0.2})                                 # 0.1 of equity left in cash


def test_no_scores_means_no_positions():
    assert target_weights(S({"A": float("nan")})) == {}


def test_output_is_long_only_and_never_over_one():
    w = target_weights(ten(), top_share=1.0, max_weight=0.15)
    assert all(v >= 0 for v in w.values()) and sum(w.values()) <= 1 + 1e-9


# --------------------------------------------------------- within class
def two_classes(n_each=10):
    scores = S({**{f"A{i}": float(i) for i in range(n_each)}, **{f"B{i}": float(i) for i in range(n_each)}})
    groups = {s: s[0] for s in scores.index}
    return scores, groups


def test_each_class_gets_its_budget_and_picks_its_own_top():
    scores, groups = two_classes(10)
    w = target_weights(scores, groups=groups, class_weights={"A": 0.6, "B": 0.4})
    assert w == pytest.approx({"A9": 0.3, "A8": 0.3, "B9": 0.2, "B8": 0.2})


def test_a_class_is_not_judged_against_the_other_class():
    scores, groups = two_classes(5)
    scores[[s for s in scores.index if s[0] == "B"]] -= 100.0                         # class B scored far lower
    w = target_weights(scores, groups=groups, class_weights={"A": 0.5, "B": 0.5})
    assert w == pytest.approx({"A4": 0.5, "B4": 0.5})                                 # B still gets its half


def test_a_class_with_too_few_scored_etfs_is_skipped_and_its_budget_redistributed():
    scores, groups = two_classes(5)
    for s in ["B2", "B3", "B4"]:
        scores[s] = float("nan")                                                      # only 2 scored in B
    w = target_weights(scores, groups=groups, class_weights={"A": 0.6, "B": 0.4})
    assert w == pytest.approx({"A4": 1.0})


def test_grouped_inputs_are_validated():
    scores, groups = two_classes(5)
    with pytest.raises(ValueError):
        target_weights(scores, groups=groups)                                         # no class weights
    with pytest.raises(ValueError):
        target_weights(scores, groups=groups, class_weights={"A": 0.5, "B": -0.5})
    with pytest.raises(ValueError):
        target_weights(scores, groups={"A0": "A"}, class_weights={"A": 1.0})           # symbols without a class
    with pytest.raises(ValueError):
        target_weights(scores, class_weights={"A": 1.0, "B": 0.0})                    # weights without groups


# ------------------------------------------------- feeds the rebalance step
def test_target_weights_feed_rebalance_orders_directly():
    w = target_weights(ten(), exposure=0.8)                                           # E9 0.4, E8 0.4
    orders, cash = rebalance_orders(w, {"E1": 300.0, "E9": 100.0}, equity=1000.0)
    assert orders[0] == {"symbol": "E1", "side": "sell", "close_all": True}
    assert {"symbol": "E8", "side": "buy", "notional": 400.0} in orders
    assert {"symbol": "E9", "side": "buy", "notional": 300.0} in orders
    assert cash == pytest.approx(200.0)

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
