import numpy as np
import pandas as pd
import pytest

from panel_data import load_prices
from update_prices import (drop_incomplete_today, latest_snapshot, reconcile_and_append, snapshot_name,
                           write_snapshot)


def frame(rows, cols=("A", "B"), start="2026-09-28"):
    idx = pd.bdate_range(start, periods=len(rows))
    idx.name = "date"
    df = pd.DataFrame(rows, index=idx, columns=list(cols), dtype=float)
    df.columns.name = "asset"
    return df


EXISTING = frame([[10, 20], [11, 21], [12, 22], [13, 23], [14, 24]])           # 09-28 .. 10-02


def fresh_from(existing, n_overlap, new_rows, scale=None):
    """A fresh pull: the last n_overlap existing rows (optionally re-scaled) plus new rows."""
    ov = existing.iloc[-n_overlap:].copy()
    if scale:
        for c, f in scale.items():
            ov[c] = ov[c] * f
    new = pd.DataFrame(new_rows, columns=ov.columns, dtype=float,
                       index=pd.bdate_range(existing.index[-1] + pd.Timedelta(days=1), periods=len(new_rows)))
    out = pd.concat([ov, new])
    out.index.name = "date"
    return out


# ---------------------------------------------------------------- append
def test_new_days_are_appended_and_old_values_are_kept():
    fresh = fresh_from(EXISTING, 3, [[15, 25], [16, 26]])
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert len(merged) == 7
    pd.testing.assert_frame_equal(merged.iloc[:5], EXISTING, check_freq=False)
    assert merged.iloc[-2:].values.tolist() == [[15, 25], [16, 26]]
    assert rep["rescaled"] == {} and len(rep["new_dates"]) == 2


def test_no_new_days_returns_the_same_history():
    merged, rep = reconcile_and_append(EXISTING, fresh_from(EXISTING, 3, []))
    pd.testing.assert_frame_equal(merged, EXISTING, check_freq=False)
    assert rep["new_dates"] == []


def test_rows_dated_on_or_before_the_last_known_date_never_add_rows():
    fresh = fresh_from(EXISTING, 5, [[15, 25]])
    merged, _ = reconcile_and_append(EXISTING, fresh)
    assert merged.index.is_unique and len(merged) == 6


# ----------------------------------------------- dividend / split re-adjustment
def test_a_readjusted_ticker_is_rescaled_so_returns_stay_continuous():
    # A paid a dividend: the fresh pull restates all of A's history 2% lower. B is untouched.
    fresh = fresh_from(EXISTING, 3, [[15 * 0.98, 25]], scale={"A": 0.98})
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert rep["rescaled"]["A"] == pytest.approx(0.98) and "B" not in rep["rescaled"]
    assert merged["A"].iloc[:5].tolist() == pytest.approx((EXISTING["A"] * 0.98).tolist())
    assert merged["B"].iloc[:5].tolist() == EXISTING["B"].tolist()
    # the return across the join is the real one (15 / 14), not a spurious dividend-sized drop
    assert np.log(merged["A"].iloc[5] / merged["A"].iloc[4]) == pytest.approx(np.log(15 / 14))


def test_a_tiny_difference_inside_the_tolerance_is_not_rescaled():
    fresh = fresh_from(EXISTING, 3, [[15, 25]], scale={"A": 1 + 1e-7})
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert rep["rescaled"] == {}
    assert merged["A"].iloc[:5].tolist() == EXISTING["A"].tolist()


def test_a_ratio_that_changes_across_the_overlap_is_rejected():
    fresh = fresh_from(EXISTING, 3, [[15, 25]])
    fresh.loc[fresh.index[0], "A"] *= 1.05                       # not a constant factor: data differs
    with pytest.raises(ValueError, match="A"):
        reconcile_and_append(EXISTING, fresh)


def test_an_implausibly_large_restatement_is_rejected():
    fresh = fresh_from(EXISTING, 3, [[15 * 2, 25]], scale={"A": 2.0})
    with pytest.raises(ValueError, match="A"):
        reconcile_and_append(EXISTING, fresh)


def test_no_overlap_means_no_way_to_check_so_it_is_rejected():
    fresh = fresh_from(EXISTING, 1, [[15, 25], [16, 26]]).iloc[1:]
    with pytest.raises(ValueError, match="overlap"):
        reconcile_and_append(EXISTING, fresh)


# ----------------------------------------------------------- bad / missing data
def test_a_ticker_missing_from_the_fresh_pull_is_reported_stale_and_not_filled():
    fresh = fresh_from(EXISTING, 3, [[15, 25], [16, 26]])[["A"]]
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert merged["B"].iloc[-2:].isna().all()
    assert rep["stale"] == ["B"]


def test_tickers_not_in_the_history_are_ignored_and_reported():
    fresh = fresh_from(EXISTING, 3, [[15, 25]])
    fresh["Z"] = 1.0
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert list(merged.columns) == ["A", "B"] and rep["ignored"] == ["Z"]


def test_non_positive_new_prices_are_rejected():
    fresh = fresh_from(EXISTING, 3, [[0.0, 25]])
    with pytest.raises(ValueError, match="non-positive"):
        reconcile_and_append(EXISTING, fresh)


def test_new_rows_with_no_prices_at_all_are_dropped():
    fresh = fresh_from(EXISTING, 3, [[15, 25], [np.nan, np.nan]])
    merged, rep = reconcile_and_append(EXISTING, fresh)
    assert len(merged) == 6 and len(rep["new_dates"]) == 1


# ------------------------------------------------------------------ files
def test_snapshot_name_uses_the_last_data_date():
    assert snapshot_name(EXISTING) == "prices_daily_asof_2026-10-02.csv"


def test_write_snapshot_round_trips_and_never_overwrites(tmp_path):
    path = write_snapshot(EXISTING, tmp_path)
    assert path.name == "prices_daily_asof_2026-10-02.csv"
    back = load_prices(path)
    assert back.shape == EXISTING.shape and back.iloc[-1].tolist() == EXISTING.iloc[-1].tolist()
    with pytest.raises(FileExistsError):
        write_snapshot(EXISTING, tmp_path)


def test_latest_snapshot_picks_the_newest_date_and_ignores_other_files(tmp_path):
    for name in ["prices_daily_asof_2026-09-30.csv", "prices_daily_asof_2026-10-02.csv",
                 "prices_daily.csv", "prices_summary.csv", "prices_daily_asof_notadate.csv"]:
        (tmp_path / name).write_text("x")
    assert latest_snapshot(tmp_path).name == "prices_daily_asof_2026-10-02.csv"
    with pytest.raises(FileNotFoundError):
        latest_snapshot(tmp_path / "nope")


# ----------------------------------------------- a half-finished trading day
def test_todays_bar_is_dropped_while_the_us_market_is_open_or_just_closed():
    fresh = frame([[1, 1], [2, 2], [3, 3]], start="2026-10-07")                # 10-07, 10-08, 10-09
    noon = pd.Timestamp("2026-10-09 12:00", tz="America/New_York")
    assert drop_incomplete_today(fresh, noon).index.max() == pd.Timestamp("2026-10-08")
    just_closed = pd.Timestamp("2026-10-09 16:30", tz="America/New_York")
    assert drop_incomplete_today(fresh, just_closed).index.max() == pd.Timestamp("2026-10-08")


def test_todays_bar_is_kept_once_the_data_has_settled():
    fresh = frame([[1, 1], [2, 2], [3, 3]], start="2026-10-07")
    evening = pd.Timestamp("2026-10-09 18:00", tz="America/New_York")
    assert len(drop_incomplete_today(fresh, evening)) == 3


def test_the_cutoff_follows_new_york_time_not_the_computers_time_zone():
    fresh = frame([[1, 1], [2, 2], [3, 3]], start="2026-10-07")
    # 15:30 in Los Angeles on 10-09 is 18:30 in New York: settled.
    la = pd.Timestamp("2026-10-09 15:30", tz="America/Los_Angeles")
    assert len(drop_incomplete_today(fresh, la)) == 3
    la_early = pd.Timestamp("2026-10-09 11:30", tz="America/Los_Angeles")           # 14:30 New York
    assert len(drop_incomplete_today(fresh, la_early)) == 2


def test_earlier_days_are_never_dropped():
    fresh = frame([[1, 1], [2, 2]], start="2026-10-01")
    now = pd.Timestamp("2026-10-09 12:00", tz="America/New_York")
    pd.testing.assert_frame_equal(drop_incomplete_today(fresh, now), fresh, check_freq=False)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_update_prices.py -v
#
# collected 19 items
#
# test_update_prices.py::test_new_days_are_appended_and_old_values_are_kept PASSED [  5%]
# test_update_prices.py::test_no_new_days_returns_the_same_history PASSED  [ 10%]
# test_update_prices.py::test_rows_dated_on_or_before_the_last_known_date_never_add_rows PASSED [ 15%]
# test_update_prices.py::test_a_readjusted_ticker_is_rescaled_so_returns_stay_continuous PASSED [ 21%]
# test_update_prices.py::test_a_tiny_difference_inside_the_tolerance_is_not_rescaled PASSED [ 26%]
# test_update_prices.py::test_a_ratio_that_changes_across_the_overlap_is_rejected PASSED [ 31%]
# test_update_prices.py::test_an_implausibly_large_restatement_is_rejected PASSED [ 36%]
# test_update_prices.py::test_no_overlap_means_no_way_to_check_so_it_is_rejected PASSED [ 42%]
# test_update_prices.py::test_a_ticker_missing_from_the_fresh_pull_is_reported_stale_and_not_filled PASSED [ 47%]
# test_update_prices.py::test_tickers_not_in_the_history_are_ignored_and_reported PASSED [ 52%]
# test_update_prices.py::test_non_positive_new_prices_are_rejected PASSED  [ 57%]
# test_update_prices.py::test_new_rows_with_no_prices_at_all_are_dropped PASSED [ 63%]
# test_update_prices.py::test_snapshot_name_uses_the_last_data_date PASSED [ 68%]
# test_update_prices.py::test_write_snapshot_round_trips_and_never_overwrites PASSED [ 73%]
# test_update_prices.py::test_latest_snapshot_picks_the_newest_date_and_ignores_other_files PASSED [ 78%]
# test_update_prices.py::test_todays_bar_is_dropped_while_the_us_market_is_open_or_just_closed PASSED [ 84%]
# test_update_prices.py::test_todays_bar_is_kept_once_the_data_has_settled PASSED [ 89%]
# test_update_prices.py::test_the_cutoff_follows_new_york_time_not_the_computers_time_zone PASSED [ 94%]
# test_update_prices.py::test_earlier_days_are_never_dropped PASSED        [100%]
#
# 19 passed in 0.36s
# Mutation checks (sandbox): never rescaling, no constant-factor check, no size check, local time
#   instead of New York time, allowing overwrite, appending the last known date again, and allowing
#   no overlap each made tests fail; original restored.
# CLI smoke test (sandbox, fake data in place of Yahoo): appended 3 days, rescaled a dividend-paying
#   ticker by 0.99, saved a new dated file, and a second run reported nothing new.
# NOT yet run against real Yahoo data. Do that on your machine: python update_prices.py
# ---------------------------------------------------------------------------
