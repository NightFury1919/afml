"""The formal evaluation rule of the ETF forward shadow model, exactly as registered.

Registered in preregistration_etf_shadow_model.md ("Reporting and decision rules"): at 36 resolved non-overlapping
periods, moving-block bootstrap (block 4), one-sided error 0.05 / 3 per fitted arm, i.e. a 96.7% interval, 20,000
resamples, seed 20261009. For each fitted arm:
    adds_beyond_volatility  lower end of the paired interval for ir(arm) - ir(wc_vol_rule) is above 0
                            AND the arm's vol_tilt is at most 0.7
    positive_alone          lower end of the arm's standalone ir interval is above 0
    edge_found              both

This module is only the rule. It takes the per-period class-neutral active returns (one column per arm, built the
same way as real_history_check_wc.active_series_within_class) and the arms' vol_tilt values.

The resampling is vectorised: all 20,000 block-index draws come from one numpy call. test_shadow_evaluate.py checks
that this gives the same intervals as the loop in real_history_check.py / real_history_check_wc.py for the same seed.
"""
import math

import numpy as np
import pandas as pd

from real_history_check import BOOT_BLOCK, BOOT_SEED, MAX_VOL_TILT

FORMAL_PERIODS = 36
FORMAL_N_BOOT = 20000
FORMAL_ALPHA = 0.05 / 3                      # one-sided error per arm
FORMAL_LEVEL = 1 - 2 * FORMAL_ALPHA          # 0.96667: lower end at the 1.67th percentile
HORIZON = 21
FITTED = ["wc_current", "wc_no_vol", "wc_book"]
REFERENCE = "wc_vol_rule"
ARMS = FITTED + [REFERENCE]

_ROW_CACHE = {}


def resample_rows(n, block=BOOT_BLOCK, n_boot=FORMAL_N_BOOT, seed=BOOT_SEED):
    """(n_boot, n) integer matrix of row positions: moving blocks of `block` periods, cut to length n."""
    key = (n, block, n_boot, seed)
    if key not in _ROW_CACHE:
        rng = np.random.default_rng(seed)
        n_blocks = math.ceil(n / block)
        starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
        rows = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
        _ROW_CACHE[key] = rows
    return _ROW_CACHE[key]


def _irs(x, rows, horizon):
    s = x[rows]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(252.0 / horizon)


def standalone_low(active, level=FORMAL_LEVEL, block=BOOT_BLOCK, n_boot=FORMAL_N_BOOT, seed=BOOT_SEED,
                   horizon=HORIZON):
    """(low, high) interval for the annualized ir of one active-return series."""
    x = np.asarray(active, dtype=float)
    irs = _irs(x, resample_rows(len(x), block, n_boot, seed), horizon)
    return tuple(float(v) for v in np.percentile(irs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def paired_low(a, b, level=FORMAL_LEVEL, block=BOOT_BLOCK, n_boot=FORMAL_N_BOOT, seed=BOOT_SEED, horizon=HORIZON):
    """(low, high) interval for ir(a) - ir(b), resampling the same blocks for both."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError("series must have the same periods")
    rows = resample_rows(len(x), block, n_boot, seed)
    d = _irs(x, rows, horizon) - _irs(y, rows, horizon)
    return tuple(float(v) for v in np.percentile(d, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def ir(active, horizon=HORIZON):
    a = np.asarray(active, dtype=float)
    return float(a.mean() / a.std(ddof=1) * np.sqrt(252.0 / horizon))


def decide_formal(active, vol_tilt, periods=FORMAL_PERIODS, **kw):
    """Apply the registered rules.

    active    DataFrame, one row per resolved period (date order), columns include the three fitted arms and wc_vol_rule
    vol_tilt  {arm: vol_tilt} for the fitted arms
    Uses exactly the first `periods` rows (no looking at later ones) and refuses to run on fewer.
    """
    missing = [c for c in ARMS if c not in active.columns]
    if missing:
        raise ValueError(f"active returns lack columns {missing}")
    if len(active) < periods:
        raise ValueError(f"{len(active)} resolved periods; the formal evaluation needs {periods}")
    act = active[ARMS].iloc[:periods].dropna()
    if len(act) < periods:
        raise ValueError("the first formal periods contain missing values")
    out = {}
    for arm in FITTED:
        p_lo, _ = paired_low(act[arm], act[REFERENCE], **kw)
        s_lo, _ = standalone_low(act[arm], **kw)
        adds = bool(p_lo > 0 and vol_tilt[arm] <= MAX_VOL_TILT)
        alone = bool(s_lo > 0)
        out[arm] = dict(ir=ir(act[arm]), paired_low=p_lo, standalone_low=s_lo, vol_tilt=float(vol_tilt[arm]),
                        adds_beyond_volatility=adds, positive_alone=alone, edge_found=bool(adds and alone))
    return out


# ---------------------------------------------------------------------------
# TDD RESULTS (sandbox py3.10 venv, 2026-10-17): test_shadow_evaluate.py 17 tests and test_noise_test_tier_a.py 12 tests,
# 29 passed together. Mutation checks: 23 mutants of the two modules (alpha, level, period count, resample count, tilt
# limit strict/ignored, or-for-and, wrong reference arm, last-instead-of-first periods, high end for low end, no
# short-sample refusal, ddof, hedge kept, signal leaked to the vol rule, beta ignored, tilt dropped, book not a cousin,
# any-edge as all, standard error, wrong tilt): all caught after two tests were added for the first survivors.
# Not yet run in the mlfinlab env. The registered run (Tier A) has not been run.
# ---------------------------------------------------------------------------
