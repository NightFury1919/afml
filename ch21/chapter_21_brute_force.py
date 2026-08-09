"""
AFML Chapter 21 -- Brute Force and Quantum Computers
Driver script: combinatorics demo -> synthetic sanity check -> real-data
(gold / crude oil / US T-bonds) dynamic-vs-static trajectory comparison.

Path convention: this script derives its own root via __file__ so it works
for anyone who clones the repo, regardless of OS or username (see CLAUDE.md).
"""
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'ch21', 'portfolio_trajectory'))

import numpy as np  # noqa: E402

import data_prep  # noqa: E402
from brute_force import count_trajectories, dyn_opt_port, eval_sr, eval_t_costs  # noqa: E402
from random_matrix import gen_synthetic_params  # noqa: E402
from static_solution import stat_opt_trajectory  # noqa: E402

RANDOM_STATE = np.random.default_rng(2026)


def part_a_combinatorics_demo():
    """
    Part A: illustrate WHY this needs to be brute force / quantum, using
    the book's own Figure 21.1 example (K=6 units, N=3 assets) plus a
    couple of scaled-up examples to show the blow-up rate.
    """
    print('=' * 70)
    print('PART A: Combinatorial explosion (why quantum computers matter)')
    print('=' * 70)
    for k, n, h in [(6, 3, 1), (6, 3, 2), (3, 3, 2), (4, 3, 2), (5, 3, 2)]:
        sizing = count_trajectories(k, n, h)
        print(
            f'K={k}, N={n}, H={h}: {sizing["num_partitions"]} partitions, '
            f'|Omega|={sizing["omega_size"]}, |trajectories|={sizing["num_trajectories"]:,}'
        )
    print()


def part_b_synthetic_sanity_check():
    """
    Part B: a fully synthetic run (Snippets 21.4-21.7's own numerical
    example, translated to Python 3) as a sanity check BEFORE trusting the
    real-data pipeline -- if this doesn't produce a well-formed comparison,
    nothing downstream can be trusted either.
    """
    print('=' * 70)
    print('PART B: Synthetic sanity check (book\'s own numerical example)')
    print('=' * 70)
    size, horizon = 3, 2
    params = gen_synthetic_params(size, horizon, random_state=RANDOM_STATE)

    w_stat = stat_opt_trajectory(params)
    tcost_stat = eval_t_costs(w_stat, params)
    sr_stat = eval_sr(params, w_stat, tcost_stat)
    print('static SR:', sr_stat)

    w_dyn = dyn_opt_port(params)
    tcost_dyn = eval_t_costs(w_dyn, params)
    sr_dyn = eval_sr(params, w_dyn, tcost_dyn)
    print('dynamic SR:', sr_dyn)
    print()
    return sr_stat, sr_dyn


def part_c_real_data_pipeline():
    """
    Part C: build the real multi-asset return series (gold, crude oil,
    US T-bonds) via roll.py-based continuous-series construction, and the
    resulting per-horizon (mu, V, c) parameters.
    """
    print('=' * 70)
    print('PART C: Real-data pipeline (gold / crude oil / US T-bonds)')
    print('=' * 70)
    commodity_dirs = {
        'gold': os.path.join(root, 'input_data', 'gold'),
        'crude_oil': os.path.join(root, 'input_data', 'crude oil'),
        'us_t_bonds': os.path.join(root, 'input_data', 'US-T bonds'),
    }
    returns_df = data_prep.align_returns(commodity_dirs)
    print(f'Aligned daily returns: {returns_df.shape[0]} trading days, '
          f'{returns_df.shape[1]} assets, {returns_df.index.min().date()} to '
          f'{returns_df.index.max().date()}')
    print(returns_df.describe())

    horizon, lookback, cost_scale = 2, 60, 0.02
    params, meta = data_prep.build_horizon_params(
        returns_df, horizon=horizon, lookback=lookback, cost_scale=cost_scale
    )
    print()
    print(f'Horizon windows (H={horizon}, lookback={lookback} trading days each):')
    for h, (start, end) in enumerate(meta['window_dates']):
        print(f'  Horizon {h + 1}: {start.date()} to {end.date()}')
    print()
    return returns_df, params, meta


def part_d_dynamic_vs_static_real(params, k=4):
    """
    Part D: the actual comparison this chapter is building toward -- does
    jointly optimizing the trajectory (accounting for transaction costs
    between horizons) beat optimizing each horizon myopically, on REAL data?
    """
    print('=' * 70)
    print('PART D: Dynamic (brute-force trajectory) vs. static (myopic) -- real data')
    print('=' * 70)
    n = params[0]['mean'].shape[0]
    h = len(params)
    sizing = count_trajectories(k=k, n=n, h=h)
    print(f'Search space: K={k}, N={n}, H={h} -> {sizing["num_trajectories"]:,} trajectories evaluated')

    w_stat = stat_opt_trajectory(params)
    tcost_stat = eval_t_costs(w_stat, params)
    sr_stat = eval_sr(params, w_stat, tcost_stat)
    print('\nStatic (myopic) trajectory:')
    print(w_stat)
    print('static SR:', sr_stat)

    w_dyn = dyn_opt_port(params, k=k)
    tcost_dyn = eval_t_costs(w_dyn, params)
    sr_dyn = eval_sr(params, w_dyn, tcost_dyn)
    print('\nDynamic (brute-force) trajectory:')
    print(w_dyn)
    print('dynamic SR:', sr_dyn)

    print()
    if sr_dyn > sr_stat:
        print(f'Dynamic solution beats static by {sr_dyn - sr_stat:.4f} SR '
              f'-- jointly optimizing the trajectory paid for its transaction costs better.')
    else:
        print(f'Static solution matched or beat dynamic on this real window '
              f'(diff {sr_dyn - sr_stat:.4f}) -- see README for discussion.')
    print()
    return sr_stat, sr_dyn, w_stat, w_dyn


if __name__ == '__main__':
    part_a_combinatorics_demo()
    part_b_synthetic_sanity_check()
    returns_df, params, meta = part_c_real_data_pipeline()
    part_d_dynamic_vs_static_real(params, k=4)


# =============================================================================
# TDD TEST RESULTS (embedded per project convention -- proactively, after
# real-machine confirmation, not just synthetic sandbox output)
# =============================================================================
# REAL-MACHINE CONFIRMED -- 2026-08-09, mlfinlab conda env
# (Python 3.10.20, pytest 9.0.3, C:\ws\AFML, Windows)
#
# $ python -m pytest ch21\portfolio_trajectory\ -v      (from repo root)
# $ python -m pytest -v                                  (from inside module folder)
# ch21/portfolio_trajectory/test_portfolio_trajectory.py -- 25 passed
# ch21/portfolio_trajectory/test_data_prep.py            -- 18 passed
#   (16 unit tests + 2 real-data integration tests, both passed against the
#    actual gold/crude oil/US T-bonds files -- not skipped)
# TOTAL: 43 passed, 0 failed, two-pass (repo root + module folder) -- 7-8s each
# (Sandbox pre-check, Python 3.12.3: same 43/43 passed -- confirmed match.)
#
# Real-data pipeline (Part C), real-machine run:
#   Aligned daily returns: 4,872 trading days, 3 assets, 1983-03-31 to 2002-10-01
#   (gold + crude oil + US T-bonds continuous series, inner-joined)
#   Note: full-history describe() stats differ from sandbox in the 4th-5th
#   decimal (e.g. crude oil mean 0.000495 real machine vs 0.000475 sandbox)
#   -- traced to pandas/numpy version differences in front-month tie-breaking
#   on same-volume days; does not affect the Part D result below, which is
#   bit-for-bit identical to sandbox.
#
# Real-data comparison (Part D), K=4, H=2, lookback=60, real-machine run:
#   static SR:  0.2986251686981421
#   dynamic SR: 0.30803461924049597
#   -> dynamic trajectory search beat the myopic static solution by ~0.0094 SR
#      on this real window. Identical to sandbox to full float precision.
# =============================================================================
