"""Effective breadth: how many independent bets do N correlated assets really give?

Meucci's effective number of bets (ENB):
    1. take the eigenvalues of the correlation matrix, lambda_1 ... lambda_N
    2. turn them into shares  p_i = lambda_i / sum(lambda)
    3. ENB = exp( -sum p_i * ln p_i )          (the exponential of the entropy)

It runs from 1 (everything moves together) to N (everything independent).

Why it matters: the sample-size math counts INDEPENDENT bets. If 48 ETFs carry the
information of only, say, 8 independent bets, then predictions across them are not
48 separate pieces of evidence.

For a relative-return label (an ETF vs the cross-sectional median) the market-wide
move is removed first, so breadth should be measured on returns relative to the
median (`relative_to_median`), not on raw returns.

Rows with any missing value are dropped before the correlation is computed (no
filling), so the result describes dates when every asset existed.
"""
import numpy as np
import pandas as pd


def enb_from_correlation(corr):
    """Meucci effective number of bets from a correlation matrix."""
    eig = np.linalg.eigvalsh(np.asarray(corr, dtype=float))
    eig = np.clip(eig, 0.0, None)
    p = eig / eig.sum()
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def enb_from_returns(returns, min_obs=250):
    """ENB from a frame of returns (rows = dates, columns = assets)."""
    r = returns.dropna(how="any")
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} complete rows, got {len(r)}")
    if (r.std() == 0).any():
        raise ValueError("an asset has zero variance in this sample")
    return enb_from_correlation(np.corrcoef(r.to_numpy(), rowvar=False))


def relative_to_median(returns):
    """Each date's returns minus that date's cross-sectional median."""
    return returns.sub(returns.median(axis=1), axis=0)


def nonoverlapping_sum(returns, horizon):
    """Sum log returns over consecutive blocks of `horizon` rows; a partial last block is dropped."""
    n_blocks = len(returns) // horizon
    trimmed = returns.iloc[: n_blocks * horizon]
    block = np.arange(len(trimmed)) // horizon
    out = trimmed.groupby(block).sum()
    out.index = trimmed.index[horizon - 1 :: horizon]
    return out


def rolling_enb(returns, window, step, min_obs=250):
    """ENB over rolling windows of `window` rows, advancing `step` rows. Indexed by window end date."""
    ends = list(range(window, len(returns) + 1, step))
    values = [enb_from_returns(returns.iloc[e - window : e], min_obs=min_obs) for e in ends]
    return pd.Series(values, index=[returns.index[e - 1] for e in ends], name="enb")

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-04, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_effective_breadth.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 12 items
#
# test_effective_breadth.py::test_independent_assets_give_full_breadth PASSED          [  8%]
# test_effective_breadth.py::test_perfectly_correlated_assets_give_one_bet PASSED      [ 16%]
# test_effective_breadth.py::test_two_perfectly_correlated_blocks_give_two_bets PASSED [ 25%]
# test_effective_breadth.py::test_two_assets_correlation_point_six_known_value PASSED  [ 33%]
# test_effective_breadth.py::test_independent_noise_returns_are_close_to_full_breadth PASSED [ 41%]
# test_effective_breadth.py::test_one_dominant_factor_collapses_breadth PASSED         [ 50%]
# test_effective_breadth.py::test_rows_with_missing_values_are_dropped_not_filled PASSED [ 58%]
# test_effective_breadth.py::test_too_few_observations_is_an_error PASSED              [ 66%]
# test_effective_breadth.py::test_zero_variance_asset_is_an_error PASSED               [ 75%]
# test_effective_breadth.py::test_relative_to_median_known_values PASSED               [ 83%]
# test_effective_breadth.py::test_nonoverlapping_sum_known_values_and_partial_block_dropped PASSED [ 91%]
# test_effective_breadth.py::test_rolling_enb_windows_and_values PASSED                [100%]
#
# 12 passed in 1.22s
# ---------------------------------------------------------------------------
