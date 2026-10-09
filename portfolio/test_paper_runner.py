import dataclasses
import json
import sys

import numpy as np
import pandas as pd
import pytest

from etf_features import FEATURE_NAMES, compute_features
from paper_runner import PROTOCOL, make_plan, score_live, training_and_live, write_plan

P = dataclasses.replace(PROTOCOL, min_assets=5)


def synth_prices(n_dates=700, n_assets=16, seed=0, with_sh=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_dates)
    cols = [f"E{i:02d}" for i in range(n_assets)] + (["SH"] if with_sh else [])
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (n_dates, len(cols))), axis=0)),
                      index=idx, columns=cols)
    px.index.name, px.columns.name = "date", "asset"
    return px


PX = synth_prices()
TODAY = PX.index[-1] + pd.Timedelta(days=1)


# --------------------------------------------------- training set and live set
def test_training_labels_are_all_resolved_by_the_last_date():
    train, live = training_and_live(PX, P)
    assert (train["t1"] <= PX.index[-1]).all()
    assert train["label"].isin([0, 1]).all()


def test_training_uses_every_label_already_resolved_up_to_the_last_date():
    train, _ = training_and_live(PX, P)
    dates = train.index.get_level_values("date")
    assert dates.max() == PX.index[-1 - P.horizon]          # the newest row whose 21-day label is known
    assert train.loc[dates == dates.max()]["t1"].eq(PX.index[-1]).all()


def test_live_rows_are_only_the_last_date_and_carry_no_labels():
    train, live = training_and_live(PX, P)
    assert set(live.index.get_level_values("date")) == {PX.index[-1]}
    assert list(live.columns) == FEATURE_NAMES
    assert len(live) == 16
    assert not train.index.intersection(live.index).size


def test_live_features_equal_the_feature_pipeline_on_the_last_date():
    _, live = training_and_live(PX, P)
    expected = compute_features(PX, min_assets=5).xs(PX.index[-1], level="date")
    got = live.droplevel("date")
    pd.testing.assert_frame_equal(got.sort_index(), expected.sort_index()[FEATURE_NAMES], check_names=False)


def test_no_lookahead_older_data_gives_a_prefix_of_the_same_training_rows():
    full, _ = training_and_live(PX, P)
    short, _ = training_and_live(PX.iloc[:-15], P)
    shared = short.index
    assert len(shared) > 0 and shared.isin(full.index).all()
    pd.testing.assert_frame_equal(short, full.loc[shared])


# ------------------------------------------------------------------- scoring
def test_scores_are_probabilities_for_exactly_the_live_assets_and_repeatable():
    train, live = training_and_live(PX, P)
    a, b = score_live(train, live, P), score_live(train, live, P)
    assert sorted(a.index) == sorted(live.index.get_level_values("asset"))
    assert ((a > 0) & (a < 1)).all()
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------- plan
def test_empty_account_plan_buys_the_top_fifth_equally_and_spends_all_the_cash():
    plan = make_plan(PX, {}, 1000.0, P, today=TODAY)
    assert plan["as_of"] == PX.index[-1].date().isoformat()
    assert len(plan["weights"]) == 4                                   # ceil(0.2 * 16)
    assert sum(plan["weights"].values()) == pytest.approx(1.0)
    assert all(o["side"] == "buy" for o in plan["orders"])
    assert sum(o["notional"] for o in plan["orders"]) == pytest.approx(1000.0, abs=0.05)
    assert plan["cash_after"] == pytest.approx(0.0, abs=0.05)
    best4 = sorted(plan["scores"], key=lambda k: -plan["scores"][k])[:4]
    assert sorted(plan["weights"]) == sorted(best4)


def test_running_again_with_the_resulting_holdings_orders_nothing():
    first = make_plan(PX, {}, 1000.0, P, today=TODAY)
    held = {s: w * 1000.0 for s, w in first["weights"].items()}
    second = make_plan(PX, held, 1000.0, P, today=TODAY)
    assert second["orders"] == []


def test_exposure_below_one_leaves_the_rest_in_cash():
    half = dataclasses.replace(P, exposure=0.5)
    plan = make_plan(PX, {}, 1000.0, half, today=TODAY)
    assert sum(plan["weights"].values()) == pytest.approx(0.5)
    assert sum(o["notional"] for o in plan["orders"]) == pytest.approx(500.0, abs=0.05)
    assert plan["cash_after"] == pytest.approx(500.0, abs=0.05)


def test_the_hedge_is_never_scored_or_bought_and_a_held_hedge_is_closed():
    px = synth_prices(with_sh=True)
    plan = make_plan(px, {"SH": 200.0}, 1000.0, P, today=TODAY)
    assert "SH" not in plan["scores"] and "SH" not in plan["weights"]
    assert {"symbol": "SH", "side": "sell", "close_all": True} in plan["orders"]


def test_whole_share_mode_spends_no_more_than_equity():
    whole = dataclasses.replace(P, mode="whole")
    plan = make_plan(PX, {}, 5000.0, whole, today=TODAY)
    last = PX.iloc[-1]
    spent = sum(o["qty"] * last[o["symbol"]] for o in plan["orders"])
    assert 0 < spent <= 5000.0 and all(float(o["qty"]).is_integer() for o in plan["orders"])


def test_too_few_scored_etfs_is_an_error():
    few = synth_prices(n_assets=4)
    with pytest.raises(ValueError, match="at least"):
        make_plan(few, {}, 1000.0, dataclasses.replace(PROTOCOL, min_assets=5), today=TODAY)


def test_bad_inputs_are_rejected():
    with pytest.raises(ValueError):
        make_plan(PX, {}, 0.0, P, today=TODAY)
    with pytest.raises(ValueError, match="pooled"):
        make_plan(PX, {}, 1000.0, dataclasses.replace(P, ranking="within_class"), today=TODAY)


def test_stale_data_produces_a_warning_but_still_a_plan():
    late = PX.index[-1] + pd.Timedelta(days=30)
    plan = make_plan(PX, {}, 1000.0, P, today=late)
    assert any("old" in w for w in plan["warnings"]) and plan["orders"]
    assert make_plan(PX, {}, 1000.0, P, today=TODAY)["warnings"] == []


def test_the_plan_records_the_protocol_it_was_made_under():
    plan = make_plan(PX, {}, 1000.0, P, today=TODAY)
    assert plan["protocol"] == dataclasses.asdict(P)


def test_making_a_plan_never_loads_any_trading_client():
    make_plan(PX, {}, 1000.0, P, today=TODAY)
    assert not [m for m in sys.modules if "alpaca" in m.lower()]


# ------------------------------------------------------------------- the file
def test_write_plan_round_trips_and_never_overwrites(tmp_path):
    plan = make_plan(PX, {}, 1000.0, P, today=TODAY)
    path = write_plan(plan, tmp_path)
    assert path.name == f"plan_{plan['as_of']}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["weights"] == pytest.approx(plan["weights"])
    with pytest.raises(FileExistsError):
        write_plan(plan, tmp_path)

# ---------------------------------------------------------------------------
# TDD RESULTS (pytest, 2026-10-09; sandbox run: Linux, Python 3.10, pandas 1.5.3, numpy 1.23.5,
# scikit-learn 1.2.2, the same versions as the mlfinlab env)
# $ cd portfolio ; pytest test_paper_runner.py -v
#
# collected 17 items
#
# test_paper_runner.py::test_training_labels_are_all_resolved_by_the_last_date PASSED [  5%]
# test_paper_runner.py::test_training_uses_every_label_already_resolved_up_to_the_last_date PASSED [ 11%]
# test_paper_runner.py::test_live_rows_are_only_the_last_date_and_carry_no_labels PASSED [ 17%]
# test_paper_runner.py::test_live_features_equal_the_feature_pipeline_on_the_last_date PASSED [ 23%]
# test_paper_runner.py::test_no_lookahead_older_data_gives_a_prefix_of_the_same_training_rows PASSED [ 29%]
# test_paper_runner.py::test_scores_are_probabilities_for_exactly_the_live_assets_and_repeatable PASSED [ 35%]
# test_paper_runner.py::test_empty_account_plan_buys_the_top_fifth_equally_and_spends_all_the_cash PASSED [ 41%]
# test_paper_runner.py::test_running_again_with_the_resulting_holdings_orders_nothing PASSED [ 47%]
# test_paper_runner.py::test_exposure_below_one_leaves_the_rest_in_cash PASSED [ 52%]
# test_paper_runner.py::test_the_hedge_is_never_scored_or_bought_and_a_held_hedge_is_closed PASSED [ 58%]
# test_paper_runner.py::test_whole_share_mode_spends_no_more_than_equity PASSED [ 64%]
# test_paper_runner.py::test_too_few_scored_etfs_is_an_error PASSED        [ 70%]
# test_paper_runner.py::test_bad_inputs_are_rejected PASSED                [ 76%]
# test_paper_runner.py::test_stale_data_produces_a_warning_but_still_a_plan PASSED [ 82%]
# test_paper_runner.py::test_the_plan_records_the_protocol_it_was_made_under PASSED [ 88%]
# test_paper_runner.py::test_making_a_plan_never_loads_any_trading_client PASSED [ 94%]
# test_paper_runner.py::test_write_plan_round_trips_and_never_overwrites PASSED [100%]
#
# 17 passed in 2.80s
# Mutation checks (sandbox): top share ignored, exposure ignored, live row from the wrong date, hedge
#   not removed, holdings ignored, plan overwrite allowed, half the training rows, and labels built from
#   older data each made tests fail; original restored. The last one survived the first version of the
#   tests, so test_training_uses_every_label_already_resolved_up_to_the_last_date was added.
# CLI smoke test (sandbox, synthetic snapshot): printed the plan, saved it, and refused to overwrite.
# NOT yet run on the real snapshot. Do that on your machine: python paper_runner.py
# ---------------------------------------------------------------------------
