import numpy as np
import pandas as pd
import pytest

from effective_breadth import (
    enb_from_correlation,
    enb_from_returns,
    nonoverlapping_sum,
    relative_to_median,
    rolling_enb,
)


def test_independent_assets_give_full_breadth():
    assert enb_from_correlation(np.eye(5)) == pytest.approx(5.0)


def test_perfectly_correlated_assets_give_one_bet():
    assert enb_from_correlation(np.ones((4, 4))) == pytest.approx(1.0)


def test_two_perfectly_correlated_blocks_give_two_bets():
    c = np.zeros((6, 6))
    c[:3, :3] = 1.0
    c[3:, 3:] = 1.0
    assert enb_from_correlation(c) == pytest.approx(2.0)


def test_two_assets_correlation_point_six_known_value():
    # eigenvalues 1.6 and 0.4 -> shares 0.8 and 0.2 -> exp(-(0.8 ln 0.8 + 0.2 ln 0.2)) = 1.6493
    c = np.array([[1.0, 0.6], [0.6, 1.0]])
    assert enb_from_correlation(c) == pytest.approx(1.6493, rel=1e-3)


def test_independent_noise_returns_are_close_to_full_breadth():
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(size=(5000, 10)))
    assert enb_from_returns(r) > 9.8


def test_one_dominant_factor_collapses_breadth():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(2000, 1))
    r = pd.DataFrame(f + 0.05 * rng.normal(size=(2000, 8)))
    assert enb_from_returns(r) < 1.5


def test_rows_with_missing_values_are_dropped_not_filled():
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(size=(600, 6)))
    r.iloc[:100, 3] = np.nan
    assert enb_from_returns(r) == pytest.approx(enb_from_returns(r.iloc[100:]))


def test_too_few_observations_is_an_error():
    r = pd.DataFrame(np.random.default_rng(3).normal(size=(50, 4)))
    with pytest.raises(ValueError):
        enb_from_returns(r, min_obs=100)


def test_zero_variance_asset_is_an_error():
    r = pd.DataFrame(np.random.default_rng(4).normal(size=(300, 4)))
    r[2] = 0.0
    with pytest.raises(ValueError):
        enb_from_returns(r, min_obs=100)


def test_relative_to_median_known_values():
    r = pd.DataFrame({"A": [1.0, 0.0], "B": [2.0, 3.0], "C": [6.0, 9.0]})
    out = relative_to_median(r)
    assert out.iloc[0].tolist() == [-1.0, 0.0, 4.0]   # median 2
    assert out.iloc[1].tolist() == [-3.0, 0.0, 6.0]   # median 3


def test_nonoverlapping_sum_known_values_and_partial_block_dropped():
    r = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]},
                     index=pd.bdate_range("2020-01-01", periods=7))
    out = nonoverlapping_sum(r, 3)
    assert out["A"].tolist() == [6.0, 15.0]           # rows 7 is a partial block, dropped
    assert len(out) == 2


def test_rolling_enb_windows_and_values():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.DataFrame(rng.normal(size=(1000, 8)), index=idx)
    out = rolling_enb(r, window=500, step=100)
    assert len(out) == 6                               # window ends at rows 500, 600, ..., 1000
    assert out.index[0] == idx[499] and out.index[-1] == idx[999]
    assert (out > 7.0).all() and (out <= 8.0 + 1e-9).all()

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
