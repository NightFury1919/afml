"""
pipeline/orchestration/test_risk_context.py

TDD test suite for risk_context.py. Following this project's established
philosophy for thin orchestration wrappers around already-tested chapter
code (see test_features.py, test_orchestration.py): hand-traced exact values for
THIS module's own new wiring (elapsed-time/frequency arithmetic, PT/SL
translation, the stationary/non-stationary branching decision), and
monkeypatched isolation of the already-tested Ch13/Ch15 real functions
(estimate_ou_params, probFailure) so these tests check risk_context.py's
OWN logic, not re-derive Ch13/Ch15's own formula correctness (which have
their own test suites already).

Run (two-pass, per project convention):
    From repo root:              pytest pipeline/orchestration/test_risk_context.py -v
    From pipeline/orchestration: pytest test_risk_context.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import risk_context  # real module under test


# ---------------------------------------------------------------------
# compute_pt_sl_context
# ---------------------------------------------------------------------

def _make_events_with_trgt(trgt_values, ret_values=None):
    n = len(trgt_values)
    idx = pd.date_range('2026-01-01', periods=n, freq='h')
    data = {'trgt': trgt_values}
    if ret_values is not None:
        data['ret'] = ret_values
    return pd.DataFrame(data, index=idx)


def test_compute_pt_sl_context_hand_traced():
    """trgt=[0.01, 0.02, 0.015] -> latest (last row) is 0.015.
    pt_sl=[1, 1] -> implied_pt_pct = 1*0.015 = 0.015, implied_sl_pct = 0.015."""
    events = _make_events_with_trgt([0.01, 0.02, 0.015])
    rebuild_result = {'events': events}
    out = risk_context.compute_pt_sl_context(rebuild_result, pt_sl=[1, 1])

    assert out['pt_sl'] == [1, 1]
    assert out['latest_trgt'] == pytest.approx(0.015)
    assert out['implied_pt_pct'] == pytest.approx(0.015)
    assert out['implied_sl_pct'] == pytest.approx(0.015)


def test_compute_pt_sl_context_asymmetric_pt_sl():
    """pt_sl=[2, 0.5] on latest trgt=0.01 -> pt=0.02, sl=0.005."""
    events = _make_events_with_trgt([0.03, 0.01])
    rebuild_result = {'events': events}
    out = risk_context.compute_pt_sl_context(rebuild_result, pt_sl=[2, 0.5])

    assert out['latest_trgt'] == pytest.approx(0.01)
    assert out['implied_pt_pct'] == pytest.approx(0.02)
    assert out['implied_sl_pct'] == pytest.approx(0.005)


def test_compute_pt_sl_context_raises_on_empty_events():
    rebuild_result = {'events': _make_events_with_trgt([])}
    with pytest.raises(ValueError, match='No events'):
        risk_context.compute_pt_sl_context(rebuild_result, pt_sl=[1, 1])


# ---------------------------------------------------------------------
# compute_strategy_risk (monkeypatched isolation of Ch15's probFailure)
# ---------------------------------------------------------------------

def test_compute_strategy_risk_hand_traced_freq_and_p_bar(monkeypatch):
    """3 events, entry times 2026-01-01/11/21, t1 (exit) 2026-01-02/12/25.
    elapsed = t1.max() - index.min() = 2026-01-25 - 2026-01-01 = 24 days.
    elapsed_years = 24 / 365.25 (hand-computed below via real timedelta
    arithmetic, not hardcoded, to avoid a rounding-transcription error).
    freq_real = 3 / elapsed_years.
    ret = [0.01, -0.02, 0.03] -> 2 of 3 positive -> p_bar = 2/3.
    probFailure itself is monkeypatched (Ch15's own math, already tested
    in ch15/strategy_risk's own suite) to isolate THIS function's
    elapsed-time/frequency/p_bar wiring.
    """
    idx = pd.to_datetime(['2026-01-01', '2026-01-11', '2026-01-21'])
    t1 = pd.to_datetime(['2026-01-02', '2026-01-12', '2026-01-25'])
    events = pd.DataFrame({'t1': t1, 'ret': [0.01, -0.02, 0.03]}, index=idx)
    rebuild_result = {'events': events}

    expected_elapsed_years = (t1.max() - idx.min()).total_seconds() / (365.25 * 24 * 3600)
    expected_freq = 3 / expected_elapsed_years

    captured = {}
    def fake_probFailure(ret, freq, tSR):
        captured['ret'] = list(ret)
        captured['freq'] = freq
        captured['tSR'] = tSR
        return 0.42  # sentinel
    monkeypatch.setattr(risk_context, 'probFailure', fake_probFailure)

    out = risk_context.compute_strategy_risk(rebuild_result, sr_hat=1.234)

    assert out['p_fail'] == 0.42
    assert out['elapsed_years'] == pytest.approx(expected_elapsed_years)
    assert out['freq_real'] == pytest.approx(expected_freq)
    assert out['p_bar'] == pytest.approx(2 / 3)
    assert out['n_events'] == 3
    assert captured['freq'] == pytest.approx(expected_freq)
    assert captured['tSR'] == 1.234
    assert captured['ret'] == [0.01, -0.02, 0.03]


def test_compute_strategy_risk_raises_if_fewer_than_2_events():
    events = _make_events_with_trgt([0.01], ret_values=[0.01])
    events['t1'] = pd.to_datetime(['2026-01-02'])
    rebuild_result = {'events': events}
    with pytest.raises(ValueError, match='at least 2 events'):
        risk_context.compute_strategy_risk(rebuild_result, sr_hat=1.0)


def test_compute_strategy_risk_raises_if_elapsed_years_nonpositive():
    """Both events have the SAME entry (index) and exit (t1) time ->
    elapsed = t1.max() - index.min() = 0 -> must raise, not divide by
    zero silently."""
    same_time = pd.to_datetime(['2026-01-01', '2026-01-01'])
    events = pd.DataFrame({'t1': same_time, 'ret': [0.01, -0.01]}, index=same_time)
    rebuild_result = {'events': events}
    with pytest.raises(ValueError, match='Elapsed window'):
        risk_context.compute_strategy_risk(rebuild_result, sr_hat=1.0)


# ---------------------------------------------------------------------
# compute_otr_finding (monkeypatched isolation of Ch13's estimate_ou_params;
# real build_xy_from_opportunities/simulate_ou_path/best_node run for real
# on small fixture data, matching this project's real-code-where-possible
# principle)
# ---------------------------------------------------------------------

def _make_close_and_events():
    """2 opportunities over a small real close series -- enough for
    build_xy_from_opportunities to run for real without erroring, since
    only estimate_ou_params itself is monkeypatched below."""
    close = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        index=pd.date_range('2026-01-01', periods=6, freq='h'),
    )
    events = pd.DataFrame(
        {'t1': [close.index[2], close.index[5]]},
        index=[close.index[0], close.index[3]],
    )
    return close, events


def test_compute_otr_finding_skips_mesh_when_nonstationary(monkeypatch):
    """phi_hat >= 1 (or <= -1) -- non-stationary -- must skip the mesh
    sweep entirely (best_node stays None) rather than spending compute on
    a sweep the book itself says is uninformative in this regime."""
    close, events = _make_close_and_events()
    rebuild_result = {'close': close, 'events': events}

    monkeypatch.setattr(risk_context, 'estimate_ou_params', lambda X, Y: (1.5, 10.0))

    out = risk_context.compute_otr_finding(rebuild_result)

    assert out['phi_hat'] == 1.5
    assert out['sigma_hat'] == 10.0
    assert out['stationary'] is False
    assert out['best_node'] is None
    assert out['n_opportunities'] == 2
    assert np.isnan(out['half_life'])  # phi_hat=1.5 not in (0,1)


def test_compute_otr_finding_runs_mesh_when_stationary(monkeypatch):
    """phi_hat in (-1,1) -- stationary -- must actually run the mesh
    sweep and return a real best_node 5-tuple. Uses a tiny mesh
    (mesh_points=2, mesh_n_iter=25) purely for test speed -- the sweep
    LOGIC being exercised is the same regardless of mesh size."""
    close, events = _make_close_and_events()
    rebuild_result = {'close': close, 'events': events}

    monkeypatch.setattr(risk_context, 'estimate_ou_params', lambda X, Y: (0.5, 1.0))

    out = risk_context.compute_otr_finding(
        rebuild_result, mesh_n_iter=25, mesh_points=2, random_state=1,
    )

    assert out['phi_hat'] == 0.5
    assert out['stationary'] is True
    assert not np.isnan(out['half_life'])  # phi_hat=0.5 IS in (0,1)
    assert out['best_node'] is not None
    pt, sl, mean, std, sharpe = out['best_node']
    assert isinstance(pt, float) and isinstance(sl, float)


def test_compute_otr_finding_skips_events_with_missing_t1():
    """An event with t1=NaT (still 'in flight', not yet resolved) must be
    skipped, not crash close.loc[entry_t:NaT]."""
    close = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.date_range('2026-01-01', periods=3, freq='h'),
    )
    events = pd.DataFrame({'t1': [close.index[2], pd.NaT]}, index=[close.index[0], close.index[1]])
    rebuild_result = {'close': close, 'events': events}

    out = risk_context.compute_otr_finding(rebuild_result, mesh_n_iter=10, mesh_points=2)
    assert out['n_opportunities'] == 1  # only the non-NaT event survives
# =============================================================================
# TDD VERIFICATION -- pytest results, real-machine-confirmed 2026-08-15
# (mlfinlab env: Python 3.10.20, pandas 1.5.3, numpy 1.23.5, sklearn 1.2.2)
# =============================================================================
# Two-pass run (per project convention):
#
# PASS 1 -- from repo root (pytest pipeline/orchestration/test_risk_context.py -v):
#   test_compute_pt_sl_context_hand_traced PASSED
#   test_compute_pt_sl_context_asymmetric_pt_sl PASSED
#   test_compute_pt_sl_context_raises_on_empty_events PASSED
#   test_compute_strategy_risk_hand_traced_freq_and_p_bar PASSED
#   test_compute_strategy_risk_raises_if_fewer_than_2_events PASSED
#   test_compute_strategy_risk_raises_if_elapsed_years_nonpositive PASSED
#   test_compute_otr_finding_skips_mesh_when_nonstationary PASSED
#   test_compute_otr_finding_runs_mesh_when_stationary PASSED
#   test_compute_otr_finding_skips_events_with_missing_t1 PASSED
#   9 passed (part of a 32-item combined run with test_orchestration.py, 10.84s)
#
# PASS 2 -- from pipeline/orchestration/ (pytest test_risk_context.py -v):
#   Same 9 tests, all PASSED (part of a 32-item combined run, 7.42s)
#
# No bugs found in risk_context.py during this real-machine confirmation --
# sandbox and real-machine behavior matched exactly.
#
# FIRST REAL LIVE RUN WITH THIS MODULE WIRED IN (2026-08-15, via
# run_pipeline_live.py, 103,115 raw trades / 720h BTCUSDT pull): produced
# this pipeline's FIRST-EVER stationary phi_hat (0.9372, half-life=10.7
# bars) across every run to date (static baseline + 3 live runs all prior
# came back non-stationary, ~1.03-1.19). Mesh sweep found a real best node
# (PT=273.77, SL=958.21, Sharpe=0.1154) across 49 real opportunities.
# Flagged as a genuine, worth-tracking data point (same treatment as the
# fracdiff d=0/d=0.1 investigation) -- NOT treated as a resolved finding:
# smallest opportunity count of any live run so far (49), phi_hat itself
# is close to the 1.0 boundary, and PBO/DSR/Ch15 P[fail] on this SAME run
# still say no reliable edge (82.86% / 0.2866 / 0.4831) -- a coherent,
# not contradictory, combination (stationarity is a structural question
# about price behavior, separate from whether direction is predictable).
# =============================================================================
