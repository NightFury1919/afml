"""
portfolio_oversight/test_oversight.py

TDD test suite for oversight.py. This module is NOT AFML content (see
oversight.py's own module docstring), so there is no book snippet to
hand-trace against -- instead, expected values are hand-derived directly
from oversight.py's own documented, simple formulas/thresholds, matching
this project's usual hand-traced-exact-value convention applied to this
project's OWN invented logic.

Run (two-pass, per project convention):
    From repo root:              pytest portfolio_oversight/test_oversight.py -v
    From portfolio_oversight/:   pytest test_oversight.py -v
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import oversight  # real module under test


# ---------------------------------------------------------------------
# suggest_capital_allocation
# ---------------------------------------------------------------------

def test_suggest_capital_allocation_hand_traced_long():
    """signal=0.5, capital=10000, fraction=0.10 -> 0.5*10000*0.10 = 500."""
    out = oversight.suggest_capital_allocation(0.5, 10_000.0, max_position_fraction=0.10)
    assert out['has_signal'] is True
    assert out['suggested_usd'] == pytest.approx(500.0)
    assert out['direction'] == 'long'


def test_suggest_capital_allocation_hand_traced_short():
    """signal=-0.3, capital=10000, fraction=0.10 -> -0.3*10000*0.10 = -300."""
    out = oversight.suggest_capital_allocation(-0.3, 10_000.0, max_position_fraction=0.10)
    assert out['suggested_usd'] == pytest.approx(-300.0)
    assert out['direction'] == 'short'


def test_suggest_capital_allocation_flat_signal():
    out = oversight.suggest_capital_allocation(0.0, 10_000.0, max_position_fraction=0.10)
    assert out['suggested_usd'] == pytest.approx(0.0)
    assert out['direction'] == 'flat'


def test_suggest_capital_allocation_none_signal():
    out = oversight.suggest_capital_allocation(None, 10_000.0)
    assert out['has_signal'] is False
    assert out['suggested_usd'] is None
    assert out['direction'] == 'no signal'


def test_suggest_capital_allocation_full_confidence_caps_at_fraction():
    """signal=1.0 (max confidence), capital=10000, fraction=0.10 -> exactly 1000."""
    out = oversight.suggest_capital_allocation(1.0, 10_000.0, max_position_fraction=0.10)
    assert out['suggested_usd'] == pytest.approx(1_000.0)


def test_suggest_capital_allocation_rejects_invalid_fraction():
    with pytest.raises(ValueError, match='max_position_fraction'):
        oversight.suggest_capital_allocation(0.5, 10_000.0, max_position_fraction=1.5)
    with pytest.raises(ValueError, match='max_position_fraction'):
        oversight.suggest_capital_allocation(0.5, 10_000.0, max_position_fraction=0.0)


def test_suggest_capital_allocation_rejects_nonpositive_capital():
    with pytest.raises(ValueError, match='paper_capital_usd'):
        oversight.suggest_capital_allocation(0.5, 0.0)
    with pytest.raises(ValueError, match='paper_capital_usd'):
        oversight.suggest_capital_allocation(0.5, -100.0)


# ---------------------------------------------------------------------
# check_circuit_breaker
# ---------------------------------------------------------------------

def test_check_circuit_breaker_not_triggered_when_both_healthy():
    eval_result = {'prob_overfit': 0.3}
    strategy_risk_result = {'p_fail': 0.2}
    out = oversight.check_circuit_breaker(eval_result, strategy_risk_result)
    assert out['triggered'] is False
    assert out['reasons'] == []
    assert out['pbo'] == 0.3
    assert out['p_fail'] == 0.2


def test_check_circuit_breaker_triggered_by_pbo_only():
    eval_result = {'prob_overfit': 0.8}
    strategy_risk_result = {'p_fail': 0.1}
    out = oversight.check_circuit_breaker(eval_result, strategy_risk_result)
    assert out['triggered'] is True
    assert len(out['reasons']) == 1
    assert 'PBO' in out['reasons'][0]


def test_check_circuit_breaker_triggered_by_p_fail_only():
    eval_result = {'prob_overfit': 0.1}
    strategy_risk_result = {'p_fail': 0.9}
    out = oversight.check_circuit_breaker(eval_result, strategy_risk_result)
    assert out['triggered'] is True
    assert len(out['reasons']) == 1
    assert 'P[fail]' in out['reasons'][0]


def test_check_circuit_breaker_triggered_by_both():
    eval_result = {'prob_overfit': 0.9}
    strategy_risk_result = {'p_fail': 0.9}
    out = oversight.check_circuit_breaker(eval_result, strategy_risk_result)
    assert out['triggered'] is True
    assert len(out['reasons']) == 2


def test_check_circuit_breaker_handles_missing_strategy_risk():
    """strategy_risk_result=None must skip that check, not crash -- e.g.
    if Ch15 couldn't be computed this run (fewer than 2 events)."""
    eval_result = {'prob_overfit': 0.9}
    out = oversight.check_circuit_breaker(eval_result, strategy_risk_result=None)
    assert out['p_fail'] is None
    assert out['triggered'] is True  # PBO alone still fires
    assert len(out['reasons']) == 1


def test_check_circuit_breaker_respects_custom_thresholds():
    eval_result = {'prob_overfit': 0.4}
    out = oversight.check_circuit_breaker(eval_result, pbo_threshold=0.3)
    assert out['triggered'] is True  # 0.4 > 0.3, would NOT trigger at default 0.5


# ---------------------------------------------------------------------
# classify_lifecycle_stage
# ---------------------------------------------------------------------

def test_classify_lifecycle_stage_embargo_small_sample_overrides_high_dsr():
    """T=50 (< default 150) with an otherwise-excellent dsr=0.99 must
    still be EMBARGO -- rule 1 (sample size) is checked BEFORE dsr."""
    eval_result = {'T': 50, 'n_trials': 20, 'dsr': 0.99}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'EMBARGO'
    assert out['small_sample'] is True


def test_classify_lifecycle_stage_embargo_low_dsr():
    eval_result = {'T': 200, 'n_trials': 20, 'dsr': 0.3}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'EMBARGO'
    assert out['small_sample'] is False


def test_classify_lifecycle_stage_paper_trading():
    eval_result = {'T': 200, 'n_trials': 20, 'dsr': 0.7}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'PAPER_TRADING'


def test_classify_lifecycle_stage_graduation_candidate():
    eval_result = {'T': 200, 'n_trials': 20, 'dsr': 0.97}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'GRADUATION_CANDIDATE'


def test_classify_lifecycle_stage_boundary_at_paper_trading_threshold():
    """dsr exactly == paper_trading_dsr (0.5) -> PAPER_TRADING (the '<'
    check for EMBARGO does not include the boundary itself)."""
    eval_result = {'T': 200, 'n_trials': 20, 'dsr': 0.5}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'PAPER_TRADING'


def test_classify_lifecycle_stage_boundary_at_graduation_threshold():
    """dsr exactly == graduation_dsr (0.95) -> GRADUATION_CANDIDATE."""
    eval_result = {'T': 200, 'n_trials': 20, 'dsr': 0.95}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'GRADUATION_CANDIDATE'


def test_classify_lifecycle_stage_small_sample_checks_n_trials_too():
    """T is large enough but n_trials is below min_reliable_trials ->
    still EMBARGO via the small-sample rule."""
    eval_result = {'T': 500, 'n_trials': 3, 'dsr': 0.99}
    out = oversight.classify_lifecycle_stage(eval_result)
    assert out['stage'] == 'EMBARGO'
    assert out['small_sample'] is True


# ---------------------------------------------------------------------
# build_oversight_section (presentation; scope-discipline guard included,
# matching pipeline/orchestration/test_orchestration.py's own guard test)
# ---------------------------------------------------------------------

def _fake_eval_result(T=200, n_trials=20, dsr=0.7, prob_overfit=0.3):
    return {'T': T, 'n_trials': n_trials, 'dsr': dsr, 'prob_overfit': prob_overfit}


def test_build_oversight_section_contains_disclaimer_banner():
    section = oversight.build_oversight_section(
        0.3, _fake_eval_result(), strategy_risk_result={'p_fail': 0.2},
    )
    assert 'NOT FROM AFML' in section
    assert 'EXPERIMENTAL' in section
    assert 'no real capital' in section.lower() or 'no real order execution' in section.lower()


def test_build_oversight_section_never_issues_a_buy_sell_directive():
    # Scope guard, mirrors test_orchestration.py's identical guard for
    # report.py -- this new section must hold the SAME line, not a looser
    # one, even though it's about capital allocation.
    section = oversight.build_oversight_section(
        0.5, _fake_eval_result(dsr=0.97, prob_overfit=0.1),
        strategy_risk_result={'p_fail': 0.05},
    )
    assert 'NOT investment advice' in section
    lowered = section.lower()
    assert 'you should buy' not in lowered
    assert 'you should sell' not in lowered
    assert 'suggested' in lowered  # language stays suggestive, not directive


def test_build_oversight_section_handles_none_signal_and_no_strategy_risk():
    section = oversight.build_oversight_section(None, _fake_eval_result())
    assert 'no current signal' in section.lower()
    assert 'not flagged' in section.lower() or 'FLAGGED' in section
# =============================================================================
# TDD VERIFICATION -- pytest results, real-machine-confirmed 2026-08-15
# (mlfinlab env: Python 3.10.20, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2)
# =============================================================================
# Two-pass run (per project convention):
#
# PASS 1 -- from repo root (pytest portfolio_oversight/test_oversight.py -v):
#   All 23 tests PASSED, 23 passed in 0.20s
#
# PASS 2 -- from portfolio_oversight/ (pytest test_oversight.py -v):
#   Same 23 tests, all PASSED, 23 passed in 0.08s
#
# No bugs found in oversight.py during real-machine confirmation -- sandbox
# and real-machine behavior matched exactly (expected: this file has no
# numpy/pandas/sklearn dependency at all, pure-Python arithmetic and
# string formatting only, so environment mismatch risk was always low).
# =============================================================================
