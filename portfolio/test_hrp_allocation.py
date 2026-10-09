import numpy as np
import pandas as pd
import pytest

from hrp_allocation import hrp_class_shares, hrp_weights, trailing_returns


def returns(n_obs=2000, seed=0, copies=0, n=4, sigma=0.01):
    """n independent assets a0..a{n-1} plus `copies` near-copies of a0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, sigma, (n_obs, n))
    extra = [x[:, [0]] + rng.normal(0, sigma * 0.1, (n_obs, 1)) for _ in range(copies)]
    cols = [f"a{i}" for i in range(n)] + [f"c{i}" for i in range(copies)]
    return pd.DataFrame(np.hstack([x] + extra), columns=cols)


def test_weights_are_positive_sum_to_one_and_keep_labels():
    w = hrp_weights(returns(copies=1))
    assert (w > 0).all() and w.sum() == pytest.approx(1.0) and set(w.index) == {"a0", "a1", "a2", "a3", "c0"}


def test_a_near_copy_is_treated_as_one_bet_not_two():
    r = returns(copies=1, seed=3)
    w = hrp_weights(r)
    plain = 1 / r.var()
    plain = plain / plain.sum()                                  # inverse-variance: the pair gets about 2/5
    assert plain["a0"] + plain["c0"] == pytest.approx(0.4, abs=0.03)
    assert w["a0"] + w["c0"] < 0.3


def test_independent_assets_with_different_variances_get_inverse_variance_weights():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"lo": rng.normal(0, 0.01, 20000), "hi": rng.normal(0, 0.02, 20000)})
    w = hrp_weights(r)
    assert w["lo"] == pytest.approx(0.8, abs=0.02)               # variance ratio 1:4


def test_missing_returns_are_rejected_not_filled():
    r = returns()
    r.iloc[10, 1] = np.nan
    with pytest.raises(ValueError):
        hrp_weights(r)


def test_too_few_assets_or_rows_are_rejected():
    with pytest.raises(ValueError):
        hrp_weights(returns(n=1))
    with pytest.raises(ValueError):
        hrp_weights(returns(n_obs=5))


def test_weights_do_not_depend_on_the_column_order():
    r = returns(copies=1, seed=5)
    a = hrp_weights(r).sort_index()
    b = hrp_weights(r[list(reversed(r.columns))]).sort_index()
    pd.testing.assert_series_equal(a, b)


# ------------------------------------------------------------- inputs / summaries
def test_trailing_returns_are_the_last_window_of_simple_returns_for_the_listed_tickers():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.DataFrame({"A": [100, 110, 121, 121, 133.1, 133.1], "B": [50.0] * 6, "C": [1.0] * 6}, index=idx)
    r = trailing_returns(px, ["A", "B"], window=3)
    assert list(r.columns) == ["A", "B"] and len(r) == 3
    assert r["A"].tolist() == pytest.approx([0.0, 0.1, 0.0])


def test_trailing_returns_reject_an_incomplete_window():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.DataFrame({"A": [1.0, 2, 3, np.nan, 5, 6], "B": [1.0] * 6}, index=idx)
    with pytest.raises(ValueError):
        trailing_returns(px, ["A", "B"], window=3)
    with pytest.raises(ValueError):
        trailing_returns(px, ["A", "B"], window=10)


def test_class_shares_add_the_weights_of_each_class():
    w = pd.Series({"A1": 0.2, "A2": 0.3, "B1": 0.5})
    assert hrp_class_shares(w, {"A1": "A", "A2": "A", "B1": "B"}) == pytest.approx({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError):
        hrp_class_shares(w, {"A1": "A"})


def test_the_book_function_receives_the_covariance_and_the_correlation_matrix(monkeypatch):
    import hrp_allocation as m
    seen = {}
    mod = m._ch16()

    def spy(cov, corr):
        seen["cov"], seen["corr"] = cov, corr
        return pd.Series(1.0 / len(cov), index=cov.index)
    monkeypatch.setattr(mod, "getHRP", spy)
    r = returns(copies=1)
    hrp_weights(r)
    pd.testing.assert_frame_equal(seen["cov"], r.cov())
    pd.testing.assert_frame_equal(seen["corr"], r.corr())
