"""
TDD suite for chow_df.py (AFML Sec 17.4.1 -- Chow-type Dickey-Fuller,
formula-only, no printed book code). Every test pins a known value: either
hand-computed closed-form OLS-through-origin algebra on a tiny fixed
example, or a regression-detection sanity check against a synthetically
injected break at a KNOWN location.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from chow_df import get_dfc, get_sdfc


# =============================================================================
# get_dfc -- hand-computed against closed-form OLS-through-origin
# =============================================================================
class TestGetDfc:
    def test_hand_computed_closed_form(self):
        # Closed-form OLS-through-origin: delta_hat = sum(x*y) / sum(x^2).
        # Build a tiny fixed price series, compute DFC by hand, compare.
        prices = np.array([10.0, 10.5, 11.0, 12.0, 14.0, 17.0, 21.0])
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=7))
        tau_star = 0.5  # break at index round(0.5*7)=4 (0-indexed into prices)

        dfc = get_dfc(logp, tau_star)

        # Manual replication of the exact same construction:
        values = logp.values
        T = len(values)
        k = int(round(tau_star * T))
        dy = np.diff(values)
        y_lag = values[:-1]
        D = np.zeros(T - 1)
        D[k - 1:] = 1.0
        x = y_lag * D
        y = dy
        delta_manual = np.sum(x * y) / np.sum(x ** 2)
        resid = y - delta_manual * x
        # OLS-through-origin, 1 regressor -> df = n - 1
        n = len(y)
        sigma2 = np.sum(resid ** 2) / (n - 1)
        se_manual = (sigma2 / np.sum(x ** 2)) ** 0.5
        dfc_manual = delta_manual / se_manual

        assert dfc == pytest.approx(dfc_manual, abs=1e-10)

    def test_accepts_dataframe_input(self):
        prices = np.array([10.0, 10.5, 11.0, 12.0, 14.0, 17.0, 21.0])
        logp_series = pd.Series(np.log(prices),
                                 index=pd.bdate_range('2020-01-01', periods=7))
        logp_df = logp_series.to_frame()
        assert get_dfc(logp_series, 0.5) == pytest.approx(get_dfc(logp_df, 0.5))

    def test_nan_when_break_too_close_to_edge(self):
        prices = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 10))
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=10))
        assert np.isnan(get_dfc(logp, 0.0))   # k < 1
        assert np.isnan(get_dfc(logp, 1.0))   # k >= T-1

    def test_explosive_switch_gives_high_dfc_at_true_break(self):
        # Random walk for n_rw steps, then genuine exponential growth --
        # DFC evaluated AT the true break fraction should be large and
        # clearly positive (delta significantly > 0).
        rng = np.random.default_rng(5)
        n_rw, n_exp = 30, 20
        rw_part = 100 + np.cumsum(rng.normal(0, 1, n_rw))
        exp_part = rw_part[-1] * np.exp(0.03 * np.arange(1, n_exp + 1))
        prices = np.concatenate([rw_part, exp_part])
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=len(prices)))
        true_tau = n_rw / len(prices)
        dfc_at_break = get_dfc(logp, true_tau)
        assert dfc_at_break > 5.0   # far beyond any conventional critical value

    def test_pure_random_walk_gives_low_dfc(self):
        # No real break anywhere -- DFC at an arbitrary tau* should NOT
        # show the same overwhelming signal a real break does. Checked
        # across multiple seeds/tau* to avoid relying on one lucky draw.
        for seed in range(5):
            rng = np.random.default_rng(seed)
            prices = 100 + np.cumsum(rng.normal(0, 1, 50))
            logp = pd.Series(np.log(prices),
                              index=pd.bdate_range('2020-01-01', periods=50))
            dfc = get_dfc(logp, 0.5)
            assert dfc < 5.0


# =============================================================================
# get_sdfc -- Andrews' sup-DFC, break-location detection
# =============================================================================
class TestGetSdfc:
    def test_locates_known_injected_break(self):
        rng = np.random.default_rng(5)
        n_rw, n_exp = 30, 20
        rw_part = 100 + np.cumsum(rng.normal(0, 1, n_rw))
        exp_part = rw_part[-1] * np.exp(0.03 * np.arange(1, n_exp + 1))
        prices = np.concatenate([rw_part, exp_part])
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=len(prices)))
        true_tau = n_rw / len(prices)

        out = get_sdfc(logp, tau0=0.15)
        # sup should land within a couple of grid points of the true break
        assert abs(out['tau_star'] - true_tau) < 0.05
        assert out['sdfc'] > 5.0

    def test_sdfc_equals_max_of_manual_grid(self):
        rng = np.random.default_rng(2)
        prices = 100 + np.cumsum(rng.normal(0, 1, 40))
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=40))
        out = get_sdfc(logp, tau0=0.15)

        # independently recompute the max over the same integer-index grid
        T = len(logp)
        k_lo, k_hi = int(np.ceil(0.15 * T)), int(np.floor(0.85 * T))
        manual_best = max(get_dfc(logp, k / T) for k in range(k_lo, k_hi + 1)
                           if np.isfinite(get_dfc(logp, k / T)))
        assert out['sdfc'] == pytest.approx(manual_best)

    def test_raises_on_infeasible_tau0(self):
        prices = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 5))
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=5))
        with pytest.raises(ValueError, match="no feasible break points"):
            get_sdfc(logp, tau0=0.45)

    def test_explicit_step_grid(self):
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 60))
        logp = pd.Series(np.log(prices),
                          index=pd.bdate_range('2020-01-01', periods=60))
        out = get_sdfc(logp, tau0=0.15, step=0.05)
        assert 0.15 <= out['tau_star'] <= 0.85
