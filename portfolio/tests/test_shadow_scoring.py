import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import shadow_scoring as ss
from etf_classes import CLASS_OF
from real_history_check_wc import build_wc_dataset

# Small windows so a 260-day synthetic panel is enough. The frozen defaults are used in the end-to-end tests.
W = {"mom_long": 30, "mom_skip": 5, "mom_med": 15, "ret_short": 3, "vol": 10, "trend": 20}
KW = dict(windows=W, min_assets=6, min_group_assets=3, horizon=5)
GROUPS = {f"A{i}": ("g1" if i < 6 else "g2") for i in range(12)}


def make_panel(n_days=260, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    steps = rng.normal(0.0003, 0.01, size=(n_days, 12))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=list(GROUPS))


PX = make_panel()
DATES = PX.index
FIRST = DATES[150]


# ------------------------------------------------------------------ refit schedule
def test_refit_day_is_the_first_day_and_every_21st_trading_day_after_it():
    flags = [ss.is_refit_day(DATES, DATES[i], FIRST, 21) for i in range(150, 200)]
    expect = [(i - 150) % 21 == 0 for i in range(150, 200)]
    assert flags == expect
    assert ss.is_refit_day(DATES, DATES[171], FIRST, 21) and not ss.is_refit_day(DATES, DATES[172], FIRST, 21)


def test_a_day_before_the_first_scoring_date_is_not_a_refit_day():
    assert not ss.is_refit_day(DATES, DATES[130], FIRST, 21)


def test_days_exactly_one_or_two_periods_before_the_first_date_are_not_refit_days():
    # k = -21 and k = -42 have remainder 0, so a missing 'k >= 0' check would wrongly call them refit days
    assert not ss.is_refit_day(DATES, DATES[129], FIRST, 21)
    assert not ss.is_refit_day(DATES, DATES[108], FIRST, 21)


def test_last_refit_date_is_the_most_recent_refit_on_or_before_the_as_of_date():
    assert ss.last_refit_date(DATES, DATES[150], FIRST, 21) == DATES[150]
    assert ss.last_refit_date(DATES, DATES[170], FIRST, 21) == DATES[150]
    assert ss.last_refit_date(DATES, DATES[171], FIRST, 21) == DATES[171]
    assert ss.last_refit_date(DATES, DATES[200], FIRST, 21) == DATES[192]


def test_last_refit_date_rejects_a_date_before_the_first_scoring_date():
    with pytest.raises(ValueError, match="before the first"):
        ss.last_refit_date(DATES, DATES[149], FIRST, 21)


def test_a_non_trading_date_is_rejected():
    saturday = DATES[150] + pd.Timedelta(days=(5 - DATES[150].weekday()) % 7 or 7)
    with pytest.raises(ValueError, match="not a trading date"):
        ss.is_refit_day(DATES, saturday, FIRST, 21)


# ------------------------------------------------------------------ training data and no look-ahead
def test_training_rows_are_only_labels_that_had_resolved_by_the_as_of_date():
    as_of = DATES[170]
    ds = ss.build_training(PX.loc[:as_of], GROUPS, **KW)
    assert ds["t1"].max() <= as_of
    assert ds.index.get_level_values("date").max() <= DATES[170 - 5]


def test_fit_ignores_everything_after_the_as_of_date():
    as_of = DATES[170]
    later = PX.copy()
    later.iloc[171:] = later.iloc[171:] * np.random.default_rng(9).uniform(0.5, 2.0, size=later.iloc[171:].shape)
    a = ss.fit_models(PX, as_of, GROUPS, **KW)
    b = ss.fit_models(later, as_of, GROUPS, **KW)
    for arm in ss.FITTED:
        assert np.allclose(a["arms"][arm]["coef"], b["arms"][arm]["coef"])
        assert a["arms"][arm]["intercept"] == pytest.approx(b["arms"][arm]["intercept"])
    assert a["meta"]["max_label_end"] <= str(as_of.date())


def test_scores_ignore_everything_after_the_as_of_date():
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    as_of = DATES[180]
    later = PX.copy()
    later.iloc[181:] = later.iloc[181:] * 3.0
    a = ss.score_day(PX, as_of, model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    b = ss.score_day(later, as_of, model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    pd.testing.assert_frame_equal(a, b)


def test_scoring_hands_the_feature_code_prices_only_up_to_the_as_of_date(monkeypatch):
    # Features only look backwards, so cutting the prices changes no output; this checks the contract directly.
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    seen = []
    real = ss.compute_features

    def spy(px, *a, **k):
        seen.append(px.index.max())
        return real(px, *a, **k)

    monkeypatch.setattr(ss, "compute_features", spy)
    ss.score_day(PX, DATES[180], model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    assert seen == [DATES[180]]
    seen.clear()
    ss.fit_models(PX, DATES[170], GROUPS, **KW)
    assert seen == [DATES[170]]


def test_fit_refuses_training_data_whose_labels_end_after_the_as_of_date(monkeypatch):
    real = ss.build_training

    def leaky(px, *a, **k):
        ds = real(px, *a, **k)
        ds = ds.copy()
        ds.iloc[-1, ds.columns.get_loc("t1")] = px.index.max() + pd.Timedelta(days=5)
        return ds

    monkeypatch.setattr(ss, "build_training", leaky)
    with pytest.raises(ValueError, match="after the as-of date"):
        ss.fit_models(PX, DATES[170], GROUPS, **KW)


def test_training_matches_the_history_check_dataset_at_the_frozen_defaults():
    px = default_panel()
    as_of = px.index[-1]
    ours = ss.build_training(px, CLASS_OF)
    theirs = build_wc_dataset(px, CLASS_OF)
    pd.testing.assert_frame_equal(ours, theirs)
    assert len(ours) > 0 and as_of == px.index[-1]


# ------------------------------------------------------------------ the three fitted arms
def test_the_current_arm_matches_a_plain_logistic_regression_on_all_five_features():
    as_of = DATES[170]
    model = ss.fit_models(PX, as_of, GROUPS, **KW)
    ds = ss.build_training(PX.loc[:as_of], GROUPS, **KW)
    lr = LogisticRegression(C=1.0, max_iter=200).fit(ds[ss.FEATURE_NAMES].to_numpy(), ds["label"].to_numpy())
    assert model["arms"]["wc_current"]["features"] == ss.FEATURE_NAMES
    assert np.allclose(model["arms"]["wc_current"]["coef"], lr.coef_[0])
    assert model["arms"]["wc_current"]["intercept"] == pytest.approx(lr.intercept_[0])


def test_the_no_volatility_arm_has_four_features_and_never_sees_vol_60():
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    arm = model["arms"]["wc_no_vol"]
    assert "vol_60" not in arm["features"] and len(arm["features"]) == 4 and len(arm["coef"]) == 4


def test_the_book_arm_uses_the_weights_and_differs_from_the_unweighted_arm():
    as_of = DATES[170]
    model = ss.fit_models(PX, as_of, GROUPS, **KW)
    ds = ss.build_training(PX.loc[:as_of], GROUPS, **KW)
    w = ss.book_weights(ds, PX.loc[:as_of])
    lr = LogisticRegression(C=1.0, max_iter=200).fit(
        ds[ss.FEATURE_NAMES].to_numpy(), ds["label"].to_numpy(), sample_weight=w.to_numpy())
    assert np.allclose(model["arms"]["wc_book"]["coef"], lr.coef_[0])
    assert not np.allclose(model["arms"]["wc_book"]["coef"], model["arms"]["wc_current"]["coef"])


def test_the_book_arm_decay_value_is_the_declared_half_and_is_recorded():
    assert ss.DECAY_C == 0.5
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    assert model["meta"]["decay_c"] == 0.5 and model["meta"]["C"] == 1.0


# ------------------------------------------------------------------ book weights, known values
D = pd.bdate_range("2020-01-01", periods=6)
CLOSE_A = pd.DataFrame({"A": np.arange(6, dtype=float) + 100.0}, index=D)
DS_A = pd.DataFrame(
    {"t1": [D[2], D[3], D[5]]},
    index=pd.MultiIndex.from_arrays([[D[0], D[2], D[4]], ["A", "A", "A"]], names=["date", "asset"]))
UNIQ = np.array([5 / 6, 3 / 4, 1.0])      # worked by hand in test_panel_uniqueness.py


def test_book_weights_known_values_uniqueness_times_half_decay_scaled_to_mean_one():
    cum = np.cumsum(UNIQ)                              # 5/6, 19/12, 31/12
    slope = (1 - 0.5) / cum[-1]
    decay = (1 - slope * cum[-1]) + slope * cum        # oldest ~0.66, newest exactly 1
    raw = UNIQ * decay
    expect = raw * len(raw) / raw.sum()
    got = ss.book_weights(DS_A, CLOSE_A, c=0.5)
    assert np.allclose(got.to_numpy(), expect)
    assert got.mean() == pytest.approx(1.0)


def test_book_weights_without_decay_are_just_scaled_uniqueness():
    expect = UNIQ * 3 / UNIQ.sum()
    assert np.allclose(ss.book_weights(DS_A, CLOSE_A, c=1.0).to_numpy(), expect)


def test_book_weights_keep_the_row_order_of_the_dataset():
    got = ss.book_weights(DS_A, CLOSE_A, c=0.5)
    assert list(got.index) == list(DS_A.index)


# ------------------------------------------------------------------ scoring
def test_the_logistic_formula_matches_sklearn_predict_proba():
    as_of = DATES[170]
    ds = ss.build_training(PX.loc[:as_of], GROUPS, **KW)
    model = ss.fit_models(PX, as_of, GROUPS, **KW)
    scores = ss.score_day(PX, DATES[180], model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    m = model["arms"]["wc_current"]
    lr = LogisticRegression(C=1.0, max_iter=200).fit(ds[ss.FEATURE_NAMES].to_numpy(), ds["label"].to_numpy())
    live = ss.compute_features(PX.loc[:DATES[180]], W, 6, GROUPS, 3).xs(DATES[180], level="date")
    expect = lr.predict_proba(live.loc[scores["asset"], ss.FEATURE_NAMES].to_numpy())[:, 1]
    assert np.allclose(scores["wc_current"].to_numpy(), expect)
    assert m["features"] == ss.FEATURE_NAMES


def test_the_volatility_rule_score_is_the_within_class_vol_60_rank():
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    scores = ss.score_day(PX, DATES[180], model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    live = ss.compute_features(PX.loc[:DATES[180]], W, 6, GROUPS, 3).xs(DATES[180], level="date")
    assert np.allclose(scores["wc_vol_rule"].to_numpy(), live.loc[scores["asset"], "vol_60"].to_numpy())
    assert np.allclose(scores["vol_60_rank"].to_numpy(), scores["wc_vol_rule"].to_numpy())
    # ranked inside each class, so each class spans (0, 1) by itself: six ETFs -> (rank - 0.5) / 6
    g1 = scores[scores["class"] == "g1"]["wc_vol_rule"].sort_values().to_numpy()
    assert np.allclose(g1, (np.arange(1, 7) - 0.5) / 6)


def test_scores_have_the_registered_columns_one_row_per_etf_and_probabilities_between_zero_and_one():
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    scores = ss.score_day(PX, DATES[180], model, GROUPS, windows=W, min_assets=6, min_group_assets=3)
    assert list(scores.columns) == ["asset", "class", "wc_current", "wc_no_vol", "wc_book",
                                    "wc_vol_rule", "vol_60_rank"]
    assert scores["asset"].is_unique and len(scores) == 12
    for arm in ss.FITTED:
        assert ((scores[arm] > 0) & (scores[arm] < 1)).all()
    assert dict(zip(scores["asset"], scores["class"])) == GROUPS


# ------------------------------------------------------------------ files: models and the never-overwritten log
def test_model_file_round_trips_and_is_never_overwritten(tmp_path):
    model = ss.fit_models(PX, DATES[170], GROUPS, **KW)
    path = ss.save_model(tmp_path, DATES[170], model)
    assert path.name == f"model_asof_{DATES[170].date()}.json"
    assert ss.load_model(path)["arms"] == json.loads(json.dumps(model["arms"]))
    with pytest.raises(FileExistsError):
        ss.save_model(tmp_path, DATES[170], model)


def test_score_log_has_a_header_the_scores_and_refuses_to_overwrite(tmp_path):
    scores = pd.DataFrame({"asset": ["A0", "A1"], "class": ["g1", "g1"], "wc_current": [0.4, 0.6],
                           "wc_no_vol": [0.5, 0.5], "wc_book": [0.45, 0.55], "wc_vol_rule": [0.1, 0.9],
                           "vol_60_rank": [0.1, 0.9]})
    header = {"as_of": "2026-10-16", "snapshot_sha256": "abc", "code_version": "v1"}
    path = ss.write_scores(tmp_path, "2026-10-16", scores, header)
    assert path.name == "scores_asof_2026-10-16.csv"
    got_header, got_scores = ss.read_scores(path)
    assert got_header == header
    pd.testing.assert_frame_equal(got_scores, scores)
    with pytest.raises(FileExistsError):
        ss.write_scores(tmp_path, "2026-10-16", scores, header)
    assert ss.read_scores(path)[0] == header      # untouched by the refused second write


# ------------------------------------------------------------------ run_shadow on a synthetic price file
def write_prices(tmp_path, px, name="prices.csv"):
    p = tmp_path / name
    px.rename_axis("date").to_csv(p)
    return p


def run(tmp_path, px, as_of, **kw):
    prices = write_prices(tmp_path, px)
    folder = tmp_path / "shadow"
    args = dict(groups=GROUPS, first_asof=FIRST, code_version="test", **KW)
    args.update(kw)
    return ss.run_shadow(prices, as_of, folder, **args), prices, folder


def test_first_day_fits_a_model_and_the_next_day_reuses_it(tmp_path):
    (p1, _, folder) = run(tmp_path, PX, DATES[150])
    h1, _ = ss.read_scores(p1)
    assert h1["last_refit_date"] == str(DATES[150].date()) and h1["refit_today"] == "True"
    model_file = folder / "models" / f"model_asof_{DATES[150].date()}.json"
    stamp = model_file.stat().st_mtime_ns
    (p2, _, _) = run(tmp_path, PX, DATES[151])
    h2, _ = ss.read_scores(p2)
    assert h2["last_refit_date"] == str(DATES[150].date()) and h2["refit_today"] == "False"
    assert h2["late_fit"] == "False"
    assert model_file.stat().st_mtime_ns == stamp
    assert len(list((folder / "models").glob("*.json"))) == 1


def test_day_21_refits_and_a_new_model_file_appears(tmp_path):
    run(tmp_path, PX, DATES[150])
    (p, _, folder) = run(tmp_path, PX, DATES[171])
    h, _ = ss.read_scores(p)
    assert h["last_refit_date"] == str(DATES[171].date()) and h["refit_today"] == "True"
    assert len(list((folder / "models").glob("*.json"))) == 2


def test_a_missed_refit_day_is_fitted_late_on_the_refit_dates_data_and_flagged(tmp_path):
    (p, _, folder) = run(tmp_path, PX, DATES[175])        # the day-171 run never happened
    h, _ = ss.read_scores(p)
    assert h["last_refit_date"] == str(DATES[171].date())
    assert h["late_fit"] == "True" and h["refit_today"] == "False"
    assert h["fit_prices_through"] == str(DATES[171].date())
    assert (folder / "models" / f"model_asof_{DATES[171].date()}.json").exists()


def test_header_records_the_snapshot_hash_the_code_version_and_training_facts(tmp_path):
    (p, prices, _) = run(tmp_path, PX, DATES[150])
    h, scores = ss.read_scores(p)
    assert h["snapshot_sha256"] == hashlib.sha256(prices.read_bytes()).hexdigest()
    assert h["snapshot_file"] == prices.name
    assert h["code_version"] == "test" and h["as_of"] == str(DATES[150].date())
    assert int(h["model_rows"]) > 0 and h["max_label_end"] <= h["as_of"]
    assert int(h["n_etfs_scored"]) == len(scores) == 12


def test_running_the_same_day_twice_is_refused_and_leaves_the_first_log_alone(tmp_path):
    (p, _, _) = run(tmp_path, PX, DATES[150])
    before = p.read_bytes()
    with pytest.raises(FileExistsError):
        run(tmp_path, PX, DATES[150])
    assert p.read_bytes() == before


def test_a_date_before_the_first_scoring_date_is_refused_unless_testing(tmp_path):
    with pytest.raises(ValueError, match="before the first scoring date"):
        run(tmp_path, PX, DATES[149])
    (p, _, _) = run(tmp_path, PX, DATES[149], allow_before_start=True)
    h, _ = ss.read_scores(p)
    assert h["pre_start"] == "True"


def test_an_as_of_date_missing_from_the_price_file_is_refused(tmp_path):
    short = PX.iloc[:160]
    with pytest.raises(ValueError, match="not in the price file"):
        run(tmp_path, short, DATES[170])


def test_the_hedge_ticker_is_never_scored(tmp_path):
    px = PX.copy()
    px["SH"] = 50.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(px))))
    (p, _, _) = run(tmp_path, px, DATES[150])
    _, scores = ss.read_scores(p)
    assert "SH" not in set(scores["asset"])


def test_two_independent_runs_give_identical_scores(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (pa, _, _) = run(a, PX, DATES[150])
    (pb, _, _) = run(b, PX, DATES[150])
    pd.testing.assert_frame_equal(ss.read_scores(pa)[1], ss.read_scores(pb)[1])


# ------------------------------------------------------------------ end to end at the frozen defaults
def default_panel(end="2026-11-20", n=340, seed=4):
    rng = np.random.default_rng(seed)
    tickers = [t for t in CLASS_OF if t != "SH"]
    idx = pd.bdate_range(end=end, periods=n)
    steps = rng.normal(0.0002, 0.01, size=(n, len(tickers)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=tickers)


def test_command_line_run_at_the_frozen_defaults(tmp_path):
    px = default_panel()
    prices = write_prices(tmp_path, px)
    folder = tmp_path / "shadow"
    code = ss.main(["--prices", str(prices), "--as-of", "2026-10-16", "--folder", str(folder),
                    "--code-version", "test"])
    assert code == 0
    h, scores = ss.read_scores(folder / "scores_asof_2026-10-16.csv")
    assert h["first_scoring_date"] == "2026-10-16" and h["refit_today"] == "True"
    assert h["decay_c"] == "0.5"
    assert len(scores) == int(h["n_etfs_scored"]) and len(scores) >= 40
    assert set(scores["class"]) <= set(CLASS_OF.values())
    assert ss.main(["--prices", str(prices), "--as-of", "2026-10-16", "--folder", str(folder),
                    "--code-version", "test"]) == 1                    # second run refused, exit code 1
