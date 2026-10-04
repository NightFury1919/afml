import numpy as np
import pandas as pd
import pytest

from panel_time_decay import get_time_decay_by_date

D1, D2, D3, D4 = (pd.Timestamp(d) for d in ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"))


def make_tw(values, dates):
    return pd.Series(values, index=pd.DatetimeIndex(dates))


def test_no_decay_when_c_is_one():
    tw = make_tw([0.5, 0.5, 1.0, 1.0], [D1, D1, D2, D3])
    out = get_time_decay_by_date(tw, clf_last_w=1.0)
    assert np.allclose(out.values, 1.0)


def test_same_date_events_share_one_weight_known_values():
    # Per-date uniqueness totals are 1.0, 1.0, 1.0 -> cumulative 1, 2, 3.
    # With c = 0 the line is y = x / 3, so the dates get 1/3, 2/3 and 1.
    tw = make_tw([0.5, 0.5, 1.0, 1.0], [D1, D1, D2, D3])
    out = get_time_decay_by_date(tw, clf_last_w=0.0)
    assert np.allclose(out.values, [1 / 3, 1 / 3, 2 / 3, 1.0])


def test_newest_date_always_gets_weight_one():
    tw = make_tw([1.0, 1.0, 1.0, 1.0], [D1, D2, D3, D4])
    for c in (0.75, 0.5, 0.0, -0.25, -0.5):
        out = get_time_decay_by_date(tw, clf_last_w=c)
        assert out.loc[D4] == pytest.approx(1.0)


def test_positive_c_known_values():
    # Four equal dates, c = 0.5: slope = 0.5 / 4, const = 0.5 -> 0.625, 0.75, 0.875, 1.0
    tw = make_tw([1.0, 1.0, 1.0, 1.0], [D1, D2, D3, D4])
    out = get_time_decay_by_date(tw, clf_last_w=0.5)
    assert np.allclose(out.values, [0.625, 0.75, 0.875, 1.0])


def test_negative_c_erases_oldest_known_values():
    # Four equal dates, c = -0.5: raw line is -0.5, 0, 0.5, 1.0 -> clipped to 0, 0, 0.5, 1.0
    tw = make_tw([1.0, 1.0, 1.0, 1.0], [D1, D2, D3, D4])
    out = get_time_decay_by_date(tw, clf_last_w=-0.5)
    assert np.allclose(out.values, [0.0, 0.0, 0.5, 1.0])


def test_weights_never_decrease_toward_the_present():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=60)
    tw = pd.Series(rng.uniform(0.1, 1.0, 60), index=dates)
    out = get_time_decay_by_date(tw, clf_last_w=0.2)
    assert (out.diff().dropna() >= -1e-12).all()


def test_result_does_not_depend_on_row_order():
    tw = make_tw([0.5, 0.5, 1.0, 1.0], [D1, D1, D2, D3])
    shuffled = tw.iloc[[3, 1, 0, 2]]
    a = get_time_decay_by_date(tw, clf_last_w=0.3)
    b = get_time_decay_by_date(shuffled, clf_last_w=0.3)
    # Two rows share D1, so compare per date, not per position.
    assert np.allclose(a.groupby(level=0).first().values, b.groupby(level=0).first().values)
    assert np.allclose(b.groupby(level=0).nunique().values, 1)


def test_output_keeps_input_index_and_order():
    tw = make_tw([1.0, 0.5, 0.5], [D3, D1, D1])
    out = get_time_decay_by_date(tw, clf_last_w=0.0)
    assert out.index.equals(tw.index)


def test_matches_book_function_when_dates_are_unique():
    from ch04.sample_weights.time_decay import get_time_decay
    tw = make_tw([0.4, 0.9, 0.7, 1.0], [D1, D2, D3, D4])
    for c in (1.0, 0.5, 0.0, -0.5):
        expected = get_time_decay(tw, clf_last_w=c)
        got = get_time_decay_by_date(tw, clf_last_w=c)
        assert np.allclose(got.values, expected.reindex(tw.index).values)


@pytest.mark.parametrize("bad_c", [-1.0, -1.5])
def test_c_at_or_below_minus_one_is_rejected(bad_c):
    tw = make_tw([1.0, 1.0], [D1, D2])
    with pytest.raises(ValueError):
        get_time_decay_by_date(tw, clf_last_w=bad_c)


def test_zero_total_uniqueness_is_rejected():
    tw = make_tw([0.0, 0.0], [D1, D2])
    with pytest.raises(ValueError):
        get_time_decay_by_date(tw, clf_last_w=0.5)


def test_nan_uniqueness_is_rejected():
    tw = make_tw([1.0, np.nan], [D1, D2])
    with pytest.raises(ValueError):
        get_time_decay_by_date(tw, clf_last_w=0.5)


def test_empty_input_returns_empty():
    tw = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
    out = get_time_decay_by_date(tw, clf_last_w=0.5)
    assert len(out) == 0

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-03, mlfinlab env: Python 3.10.20, pytest 9.0.3)
# $ cd portfolio ; pytest test_panel_time_decay.py -v
#
# platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0
# rootdir: C:\ws\AFML\portfolio
# collected 14 items
#
# test_panel_time_decay.py::test_no_decay_when_c_is_one PASSED                       [  7%]
# test_panel_time_decay.py::test_same_date_events_share_one_weight_known_values PASSED [ 14%]
# test_panel_time_decay.py::test_newest_date_always_gets_weight_one PASSED           [ 21%]
# test_panel_time_decay.py::test_positive_c_known_values PASSED                      [ 28%]
# test_panel_time_decay.py::test_negative_c_erases_oldest_known_values PASSED        [ 35%]
# test_panel_time_decay.py::test_weights_never_decrease_toward_the_present PASSED    [ 42%]
# test_panel_time_decay.py::test_result_does_not_depend_on_row_order PASSED          [ 50%]
# test_panel_time_decay.py::test_output_keeps_input_index_and_order PASSED           [ 57%]
# test_panel_time_decay.py::test_matches_book_function_when_dates_are_unique PASSED  [ 64%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.0] PASSED    [ 71%]
# test_panel_time_decay.py::test_c_at_or_below_minus_one_is_rejected[-1.5] PASSED    [ 78%]
# test_panel_time_decay.py::test_zero_total_uniqueness_is_rejected PASSED            [ 85%]
# test_panel_time_decay.py::test_nan_uniqueness_is_rejected PASSED                   [ 92%]
# test_panel_time_decay.py::test_empty_input_returns_empty PASSED                    [100%]
#
# 14 passed in 1.43s
# ---------------------------------------------------------------------------
