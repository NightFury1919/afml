"""
pipeline/orchestration/risk_context.py

Phase 4 (2026-08-15): "Portfolio Oversight"-adjacent risk context for the
live report -- Ch13's Optimal Trading Rule (OTR), Ch15's probability of
strategy failure, and rebuild.py's own PT_SL/daily_vol calibration --
computed here so report.py can stay pure presentation logic over already-
computed numbers (see report.py's own module docstring).

This is NOT new AFML content -- every calculation below delegates to
existing, real-machine-confirmed chapter code:
  - ch13/otr/otr.py (build_xy_from_opportunities, estimate_ou_params,
    phi_to_half_life, simulate_ou_path, best_node)
  - ch15/strategy_risk/algorithm.py (probFailure)
  - rebuild.py's own PT_SL constant / per-event 'trgt' (already computed,
    just surfaced)

See the 2026-08-14 handoff, Part 5, for why the report has never
mentioned stop-losses/position-sizing until now: this pipeline stops
exactly where the book's own Strategist/Backtester stations stop, and
Portfolio Oversight -- where stop-losses institutionally live -- is named
by the book (Ch1 Sec 1.3.1) but never given an algorithm. This module
wires in the REAL AFML content that already exists in this project
(Ch13 OTR, Ch15 strategy risk, rebuild.py's PT_SL) without inventing
anything beyond what the book/project already computes. Genuinely NEW
material (capital-based sizing, circuit breakers, the embargo -> paper
trading -> graduation lifecycle) is explicitly OUT of scope here -- see
the 2026-08-14 handoff's Part 5, item 2, for that separate, clearly-
labeled list.

*** LOAD-BEARING (2026-08-15): Ch13 OTR's mesh sweep only runs if the
LIVE phi_hat comes out stationary ***
On every real-data run so far (static March 2026 baseline), phi_hat has
landed non-stationary (~1.03-1.04) -- the book's own degenerate case
(Sec 13.6.1: as phi->1, "there are no recognizable areas where
performance can be maximized"). Re-deriving phi_hat/sigma_hat live is
cheap (one OLS estimate); the mesh sweep itself is real compute
(thousands of simulated paths per node). Running the expensive sweep on
data already known to produce a flat, uninformative mesh would burn time
for no report value. So: ALWAYS re-derive phi_hat/sigma_hat live and
report the finding honestly either way; only spend the sweep's compute if
phi_hat actually comes out stationary this run -- if it does, that would
itself be a genuinely interesting change from the established finding,
worth the compute to investigate properly.

*** LOAD-BEARING (2026-08-15): Ch15's tSR is this run's own winning-trial
sr_hat, not a book-benchmark constant ***
The static chapter_15_strategy_risk.py demo used fixed benchmark tSRs
(0.5, 1.0, 2.0) to illustrate the method in general. For the live report,
using THIS run's own eval_result['sr_hat'] as tSR ties the question
directly to what the report is actually claiming ("this run's own
best-trial Sharpe was sr_hat -- what's the probability the strategy's
TRUE precision can't sustain even that?") rather than an arbitrary
external target.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CH13_OTR = os.path.join(ROOT, 'ch13', 'otr')
if CH13_OTR not in sys.path:
    sys.path.insert(0, CH13_OTR)
CH15_STRATEGY_RISK = os.path.join(ROOT, 'ch15', 'strategy_risk')
if CH15_STRATEGY_RISK not in sys.path:
    sys.path.insert(0, CH15_STRATEGY_RISK)

from otr import (                                     # real module, ch13
    build_xy_from_opportunities, estimate_ou_params,
    phi_to_half_life, simulate_ou_path, best_node,
)
from algorithm import probFailure                       # real module, ch15


def compute_otr_finding(rebuild_result, mesh_n_iter=2_000, mesh_points=8,
                          random_state=7):
    """Live counterpart to chapter_13_otr.py's Part B (+ Part C only if
    warranted). Re-derives {phi_hat, sigma_hat} from THIS run's real
    triple-barrier events (entry-price-centered, matching this project's
    2026-07-22 FINAL decision documented in ch13/otr/otr.py -- NOT
    re-litigated here), and only spends the mesh-sweep compute (Part C)
    if phi_hat actually comes out stationary. Mesh parameters (sigma_hat-
    scaled r_pt/r_sl, rng convention) mirror chapter_13_otr.py's real
    Part C exactly, not reinvented.

    Parameters
    ----------
    rebuild_result : dict, rebuild.py's build_bars_and_labels() output
        (needs 'events' [index=entry time, has a 't1' column] and
        'close')
    mesh_n_iter, mesh_points, random_state : passed to the mesh sweep,
        only used if phi_hat is stationary

    Returns
    -------
    dict with keys:
      'phi_hat', 'sigma_hat' : float
      'stationary'            : bool
      'half_life'             : float (NaN if phi_hat not in (0,1))
      'n_opportunities'       : int, events actually used (>=2-bar paths)
      'best_node'             : (pt, sl, mean, std, sharpe) tuple, or
        None if not stationary (no fittable rule -- mesh sweep skipped)
    """
    close = rebuild_result['close']
    events = rebuild_result['events']

    paths, targets = [], []
    for entry_t, row in events.iterrows():
        exit_t = row['t1']
        if pd.isna(exit_t):
            continue
        path = close.loc[entry_t:exit_t]
        if len(path) < 2:
            continue
        entry_price = path.iloc[0]
        paths.append(path.values)
        targets.append(entry_price)  # entry-price centering, matching
                                       # ch13's FINAL real-data decision

    X, Y = build_xy_from_opportunities(paths, targets)
    phi_hat, sigma_hat = estimate_ou_params(X, Y)
    stationary = -1 < phi_hat < 1
    half_life = phi_to_half_life(phi_hat)

    best = None
    if stationary:
        r_pt = sigma_hat * np.linspace(0.5, 3, mesh_points)
        r_sl = sigma_hat * np.linspace(0.5, 3, mesh_points)
        # half_life is only non-NaN when phi_hat is in (0,1) -- a
        # stationary phi_hat in (-1,0] would leave half_life NaN, so we
        # pass phi_hat straight into simulate_ou_path node-by-node
        # (matching chapter_13_otr.py's real Part C approach) rather than
        # going through batch()/half_life_to_phi, which needs a valid
        # half-life to derive phi back out.
        rng = np.random.default_rng(random_state).normal
        results = []
        for pt in r_pt:
            for sl in r_sl:
                exits = np.array([
                    simulate_ou_path(phi_hat, sigma_hat, 0.0, pt, sl, 100,
                                      seed=0.0, rng=rng)[0]
                    for _ in range(mesh_n_iter)
                ])
                mean, std = exits.mean(), exits.std()
                sharpe = mean / std if std > 0 else float('nan')
                results.append((pt, sl, mean, std, sharpe))
        best = best_node(results)

    return {
        'phi_hat': phi_hat,
        'sigma_hat': sigma_hat,
        'stationary': stationary,
        'half_life': half_life,
        'n_opportunities': len(paths),
        'best_node': best,
    }


def compute_strategy_risk(rebuild_result, sr_hat):
    """Live counterpart to chapter_15_strategy_risk.py's Part C. Uses
    THIS run's real triple-barrier returns and annualized bet frequency,
    with tSR = sr_hat (this run's own winning-trial Sharpe -- see module
    LOAD-BEARING note) instead of the static script's fixed benchmark
    tSRs.

    Parameters
    ----------
    rebuild_result : dict, rebuild.py's build_bars_and_labels() output
        (needs 'events' with a 'ret' column, index = entry time, and a
        't1' column for exit time)
    sr_hat : float, this run's winning-trial Sharpe (eval_result['sr_hat'])

    Returns
    -------
    dict with keys:
      'p_fail'        : float, P[true precision < precision needed to
        sustain tSR=sr_hat]
      'freq_real'     : float, annualized bet frequency used
      'p_bar'         : float, realized (empirical) precision
      'elapsed_years' : float
      'n_events'      : int
    """
    events = rebuild_result['events']
    ret = events['ret'].values
    if len(ret) < 2:
        raise ValueError(
            'Need at least 2 events with a ret value to estimate strategy '
            'risk -- got fewer than that on this run.'
        )

    elapsed = events['t1'].max() - events.index.min()
    elapsed_years = elapsed.total_seconds() / (365.25 * 24 * 3600)
    if elapsed_years <= 0:
        raise ValueError(
            'Elapsed window is zero or negative -- cannot annualize bet '
            'frequency for this run.'
        )
    freq_real = len(events) / elapsed_years

    n_pos = (ret > 0).sum()
    p_bar = n_pos / len(ret)

    p_fail = probFailure(ret, freq_real, sr_hat)

    return {
        'p_fail': float(p_fail),
        'freq_real': float(freq_real),
        'p_bar': float(p_bar),
        'elapsed_years': float(elapsed_years),
        'n_events': len(events),
    }


def compute_pt_sl_context(rebuild_result, pt_sl):
    """Surfaces rebuild.py's ALREADY-computed PT_SL/daily_vol calibration
    as an explicit, reader-facing stop-loss/take-profit statement -- no
    new computation, just translating an existing internal parameter
    into plain language. pt_sl is passed in by the caller (rebuild.PT_SL)
    rather than imported here, so this function has no import-order
    dependency on rebuild.py and stays trivially testable. See the
    2026-08-14 handoff, Part 5.

    Parameters
    ----------
    rebuild_result : dict, rebuild.py's build_bars_and_labels() output
        (needs 'events' with a 'trgt' column)
    pt_sl : list/tuple of 2 floats, e.g. rebuild.PT_SL == [1, 1]

    Returns
    -------
    dict with keys:
      'pt_sl'          : the pt_sl passed in, unchanged
      'latest_trgt'    : float, the most recent event's target return
        magnitude (the daily-vol estimate that set that event's barrier
        widths)
      'implied_pt_pct' : float, pt_sl[0] * latest_trgt
      'implied_sl_pct' : float, pt_sl[1] * latest_trgt
    """
    events = rebuild_result['events']
    if len(events) == 0:
        raise ValueError('No events available to surface a PT/SL level from.')
    latest_trgt = float(events['trgt'].iloc[-1])

    return {
        'pt_sl': pt_sl,
        'latest_trgt': latest_trgt,
        'implied_pt_pct': pt_sl[0] * latest_trgt,
        'implied_sl_pct': pt_sl[1] * latest_trgt,
    }
